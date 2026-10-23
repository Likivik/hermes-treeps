"""db.py: the storage layer's own guarantees.

The bug this file exists for: eight concurrent ``bind_session`` calls on one
workspace lost four of them. The old implementation read a JSON document,
appended to a list and wrote the whole file back - the last writer won with
the list it had read BEFORE the others wrote. A row per binding with one
transaction per mutation removes the class of bug; these tests pin that.
"""

from __future__ import annotations

import json
import threading

from plugin_loader import STORE

DB = STORE.db


# ── the concurrency guarantee ────────────────────────────────────────────────


def test_eight_concurrent_bind_sessions_all_land():
    """Eight threads binding one workspace: every session must be recorded.

    Prevents (P1-6, concurrency half): the old JSON read-modify-write lost
    four of eight bindings. The count is the whole assertion - a
    read-modify-write loses some of them, so this test fails on the old code.
    """
    DB.write_workspace({"repo": "repo", "slug": "ws", "path": "/tmp/ws"}, path="/tmp/ws")
    barrier = threading.Barrier(8)
    errors: list[Exception] = []

    def _bind(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            DB.add_workspace_session("repo", "ws", f"sid-{i}")
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=_bind, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    landed = sorted(DB.workspace_sessions("repo", "ws"))
    assert len(landed) == 8, f"only {len(landed)} of 8 bindings landed: {landed}"
    assert landed == sorted(f"sid-{i}" for i in range(8)), landed


# ── legacy JSON migration ────────────────────────────────────────────────────


def _legacy_state(tmp_path, monkeypatch):
    """A legacy state dir with one good and one corrupt record."""
    state = tmp_path / "legacy-state"
    state.mkdir()
    monkeypatch.setenv("HERMES_TREEPS_LEGACY_STATE_DIR", str(state))
    (state / "sid-good.json").write_text(json.dumps({
        "session_id": "sid-good", "mode": "repo", "repo": "r", "slug": "s",
        "path": "/legacy/ws", "updated_at": "2020-01-01T00:00:00Z",
    }))
    (state / "sid-broken.json").write_text("{not json at all")
    return state


def test_migration_imports_the_legacy_json_and_is_idempotent(tmp_path, monkeypatch):
    """The legacy state files are imported once, and running it again is safe.

    Prevents (migration): a second run must not duplicate rows, and a corrupt
    file must be skipped with a warning instead of aborting the migration of
    every other record.
    """
    _legacy_state(tmp_path, monkeypatch)

    first = DB.migrate_legacy(workspace_root=tmp_path / "wsroot")
    assert first["imported_bindings"] == 1, first
    assert first["imported_workspaces"] == 0, first
    assert len(first["skipped"]) == 1, first
    assert "sid-broken" in first["skipped"][0], first

    binding = DB.read_binding("sid-good")
    assert binding is not None, "the legacy binding was not imported"
    assert binding["mode"] == "repo", binding
    assert binding["path"] == "/legacy/ws", binding
    # The stamp is recorded, so a later run can tell "migrated" from "never".
    assert DB.get_kv("migrated_at"), "migrated_at was not recorded"

    second = DB.migrate_legacy(workspace_root=tmp_path / "wsroot")
    assert second["imported_bindings"] == 0, second
    assert second["imported_workspaces"] == 0, second
    assert len(DB.list_workspace_rows()) == 0, "the second run created rows"


def test_migration_dry_run_imports_nothing(tmp_path, monkeypatch):
    """`migrate --dry-run` must preview, not import.

    Prevents: the first version of `migrate_legacy` ignored dry_run and wrote
    38 bindings into the live database during a `--dry-run --purge` preview.
    A preview that writes is what makes an operator stop trusting dry runs.
    """
    _legacy_state(tmp_path, monkeypatch)

    preview = DB.migrate_legacy(workspace_root=tmp_path / "wsroot", dry_run=True)
    assert preview["imported_bindings"] == 1, preview  # it counted what it saw
    assert preview["dry_run"] is True, preview
    assert preview["migrated_at"] == "", preview
    assert DB.read_binding("sid-good") is None, "a dry run imported a binding"
    assert DB.get_kv("migrated_at") is None, "a dry run recorded migrated_at"

    real = DB.migrate_legacy(workspace_root=tmp_path / "wsroot")
    assert real["imported_bindings"] == 1, real
    assert DB.read_binding("sid-good") is not None, "the real run imported nothing"


def test_store_wrapper_forwards_dry_run(tmp_path, monkeypatch):
    """The `store` seam the CLI calls must forward `dry_run` too.

    Prevents: `store.migrate_legacy` kept the old one-argument signature after
    `db.migrate_legacy` gained `dry_run`, so the CLI raised TypeError - and
    before that fix, the same wrapper silently imported under --dry-run.
    """
    _legacy_state(tmp_path, monkeypatch)

    preview = STORE.migrate_legacy(workspace_root=tmp_path / "wsroot", dry_run=True)
    assert preview["dry_run"] is True, preview
    assert DB.read_binding("sid-good") is None, "the store wrapper swallowed dry_run"


def test_migration_imports_the_in_tree_workspace_records(tmp_path, monkeypatch):
    """Every ``<workspace>/.hermes/session.json`` under the root is imported."""
    root = tmp_path / "wsroot"
    ws = root / "repo" / "ws1"
    (ws / ".hermes").mkdir(parents=True)
    (ws / ".hermes" / "session.json").write_text(json.dumps({
        "repo": "repo", "slug": "ws1", "vcs": "jj", "sessions": ["sid-a"],
    }))
    out = DB.migrate_legacy(workspace_root=root)
    assert out["imported_workspaces"] == 1, out
    rec = DB.read_workspace(str(ws))
    assert rec is not None, "the workspace record was not imported"
    assert rec["repo"] == "repo" and rec["slug"] == "ws1", rec
    assert rec["sessions"] == ["sid-a"], rec


def test_migration_skips_a_corrupt_workspace_record(tmp_path, monkeypatch):
    """One broken in-tree record must not abort the migration of the rest."""
    root = tmp_path / "wsroot"
    broken = root / "repo" / "broken"
    (broken / ".hermes").mkdir(parents=True)
    (broken / ".hermes" / "session.json").write_text("<<<not json>>>")
    good = root / "repo" / "good"
    (good / ".hermes").mkdir(parents=True)
    (good / ".hermes" / "session.json").write_text(json.dumps({
        "repo": "repo", "slug": "good", "sessions": [],
    }))
    out = DB.migrate_legacy(workspace_root=root)
    assert out["imported_workspaces"] == 1, out
    assert len(out["skipped"]) == 1, out
    assert DB.read_workspace(str(good)) is not None


# ── purging the legacy in-tree records ───────────────────────────────────────


def _known_workspace(tmp_path):
    root = tmp_path / "wsroot"
    known = root / "repo" / "known"
    (known / ".hermes").mkdir(parents=True)
    (known / ".hermes" / "session.json").write_text(json.dumps({
        "repo": "repo", "slug": "known", "sessions": [],
    }))
    stranger = root / "repo" / "stranger"
    (stranger / ".hermes").mkdir(parents=True)
    (stranger / ".hermes" / "session.json").write_text(json.dumps({
        "repo": "repo", "slug": "stranger", "sessions": [],
    }))
    return root, known, stranger


def test_purge_dry_run_lists_without_deleting(tmp_path):
    """A dry run reports what it WOULD delete and deletes nothing."""
    root, known, stranger = _known_workspace(tmp_path)
    DB.write_workspace(
        {"repo": "repo", "slug": "known", "path": str(known)}, path=str(known)
    )
    out = DB.purge_legacy_records(dry_run=True, workspace_root=root)
    assert out["purged"] == [], out
    assert out["would_purge"] == [str(known / ".hermes" / "session.json")], out
    assert (known / ".hermes" / "session.json").exists(), "the dry run deleted a file"
    assert (stranger / ".hermes" / "session.json").exists()


def test_purge_tars_then_deletes_only_known_workspaces(tmp_path):
    """The real purge backs the files up FIRST, and touches only what it knows.

    Prevents (data loss): a tree that merely happens to carry a
    ``.hermes/session.json`` is not ours - deleting it would destroy data the
    plugin never wrote. And the backup must exist before anything is deleted.
    """
    root, known, stranger = _known_workspace(tmp_path)
    DB.write_workspace(
        {"repo": "repo", "slug": "known", "path": str(known)}, path=str(known)
    )
    out = DB.purge_legacy_records(dry_run=False, workspace_root=root)
    assert out["purged"] == [str(known / ".hermes" / "session.json")], out
    assert not (known / ".hermes" / "session.json").exists()
    assert (stranger / ".hermes" / "session.json").exists(), (
        "the purge deleted a record of a workspace the database does not know"
    )
    # The backup exists and holds the deleted file.
    backup = out["backed_up"]
    assert backup, "the purge deleted files without a backup"
    import tarfile

    with tarfile.open(backup) as tar:
        names = tar.getnames()
    assert any("session.json" in n for n in names), names
