"""The one-time move to the dot-dir workspace layout.

The migration MOVES files, so every test here is about what it must NOT do:
never relocate uncommitted work, never move a tree a live session is standing
in, and never report a move it did not verify.

The jj-dependent bodies need a real jj repo (a relative ``.jj/repo`` pointer
only exists there), so they skip when jj is not installed - the same guard
the prepare tests use.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from plugin_loader import STORE

MIGRATE = importlib.import_module("hermes_treeps.migrate_layout")

WorkspaceError = STORE.WorkspaceError


def _legacy_root(tmp_path: Path) -> Path:
    """The OLD workspace root: ``<repo_dir>/workspaces``.

    WHY it is INSIDE the repo dir: in production the old root was
    ``/Git/hermes/workspaces`` and the repo dir was ``/Git/hermes``. The jj
    pointer in a child workspace climbs out of the workspace root and back
    down into the repo dir, and it is RELATIVE - so the pointer survives the
    move ONLY because both roots sit at the same depth under the repo dir
    (``workspaces/REPO/SLUG`` -> ``.workspaces/REPO/SLUG`` is the same number
    of levels). A fixture with a different geometry would prove nothing
    about the real layout.
    """
    return tmp_path / "repos" / "workspaces"


def _new_base(tmp_path: Path) -> Path:
    """The NEW workspace base: ``<repo_dir>/.workspaces``."""
    return tmp_path / "repos" / ".workspaces"


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Point the plugin at a tmp repo dir and tmp legacy root.

    The legacy root is patched into the settings so the migration recognises
    the old paths, and the new base is derived from the repo dir - the real
    derivation, not a stub.
    """
    repo_dir = tmp_path / "repos"
    repo_dir.mkdir()
    legacy = _legacy_root(tmp_path)
    legacy.mkdir()
    monkeypatch.setattr(STORE, "repo_dir", lambda: repo_dir)
    monkeypatch.setattr(
        STORE,
        "settings",
        lambda: {
            "repo_dir": str(repo_dir),
            "workspace_root": str(legacy),
            "repo_roots": [str(repo_dir)],
        },
    )
    # The legacy roots the migration recognises: the shipped default and the
    # configured one.
    monkeypatch.setattr(MIGRATE, "legacy_roots", lambda *a, **kw: [legacy])
    return {"repo_dir": repo_dir, "legacy": legacy, "new_base": _new_base(tmp_path)}


def _seed_workspace(layout, repo="demo", slug="demo-abc123", sessions=()):
    """A workspace directory under the OLD root, with a database record."""
    ws = layout["legacy"] / repo / slug
    ws.mkdir(parents=True)
    STORE.write_record(ws, {
        "repo": repo,
        "slug": slug,
        "path": str(ws),
        "vcs": "jj",
        "sessions": list(sessions),
        "created_at": "2024-01-01T00:00:00",
    })
    return ws


def _git_commit(repo: Path, message: str = "init") -> None:
    """Commit in ``repo``, with an identity that does not depend on the host.

    WHY: a bare CI runner has no ``user.name`` / ``user.email``, and git
    refuses to commit without them ("Please tell me who you are"). A test
    that depends on the developer's global git config passes locally and
    fails in CI - so the identity is set on the repository itself.
    """
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", message],
                   check=True, capture_output=True)


# ── the happy path: a clean, ended workspace moves ───────────────────────────


def test_a_clean_ended_workspace_moves_and_the_db_follows(layout, monkeypatch):
    """A clean workspace under the old root moves, and the records follow.

    WHY the records must follow: a moved directory with a record that still
    names the old path is a workspace the plugin cannot find and cannot
    retire - it is invisible AND unreclaimable, so the disk fills up silently.
    """
    ws = _seed_workspace(layout, sessions=["sid-old"])
    STORE.write_state("sid-old", {
        "mode": "repo", "repo": "demo", "slug": ws.name,
        "path": str(ws), "session_ended_at": "2024-01-02T00:00:00",
    })
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    out = MIGRATE.migrate_workspaces()

    new = layout["new_base"] / "demo" / ws.name
    assert len(out["moved"]) == 1, out
    assert out["moved"][0]["new_path"] == str(new), out
    assert new.is_dir(), "the workspace did not move"
    assert not ws.exists(), "the old directory is still there"

    # the workspace record points at the new path
    record = STORE.db.read_workspace(str(new))
    assert record is not None, "the record was lost by the move"
    assert record["path"] == str(new), record
    assert STORE.db.read_workspace(str(ws)) is None, "a record still names the old path"

    # the session binding points at the new path too
    state = STORE.read_state("sid-old")
    assert state["path"] == str(new), state


def test_a_dry_run_moves_nothing(layout, monkeypatch):
    """The dry run reports what would move and changes nothing.

    WHY: the operator has to be able to see the plan before the plugin
    starts moving directories - and the report and the move must not drift
    apart, so both come from the same code path.
    """
    ws = _seed_workspace(layout)
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    out = MIGRATE.migrate_workspaces(dry_run=True)

    assert len(out["moved"]) == 1, out
    assert out["moved"][0].get("dry_run") is True, out
    assert ws.is_dir(), "the dry run moved the workspace"
    assert not (layout["new_base"] / "demo" / ws.name).exists()


def test_a_workspace_already_in_the_new_layout_is_left_alone(layout, monkeypatch):
    """A workspace that is already where it belongs is not touched."""
    new = layout["new_base"] / "demo" / "demo-abc123"
    new.mkdir(parents=True)
    STORE.write_record(new, {
        "repo": "demo", "slug": "demo-abc123", "path": str(new),
        "vcs": "jj", "sessions": [], "created_at": "2024-01-01T00:00:00",
    })
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    out = MIGRATE.migrate_workspaces()

    assert out["moved"] == [], out
    assert new.is_dir()


# ── the skip rules: the ones that protect the operator's work ────────────────


def test_a_dirty_workspace_is_reported_and_never_moved(layout, monkeypatch):
    """A workspace with uncommitted changes is NOT moved.

    Prevents (the failure this exists for): silently relocating uncommitted
    work. The files are still there afterwards, but in a tree nobody is
    looking at any more, and the session that owned them finds a path that
    does not exist. The change is reported, never performed.
    """
    ws = _seed_workspace(layout)
    monkeypatch.setattr(STORE, "dirty", lambda path: True)

    out = MIGRATE.migrate_workspaces()

    assert out["moved"] == [], out
    assert len(out["skipped_dirty"]) == 1, out
    assert out["skipped_dirty"][0]["path"] == str(ws), out
    assert "uncommitted" in out["skipped_dirty"][0]["reason"], out
    assert ws.is_dir(), "the dirty workspace was moved anyway"
    assert not (layout["new_base"] / "demo" / ws.name).exists()
    # the record still names the old path: the workspace did not move, so
    # its record must not move either
    record = STORE.db.read_workspace(str(ws))
    assert record is not None and record["path"] == str(ws), record


def test_a_workspace_bound_to_a_live_session_is_not_moved(layout, monkeypatch):
    """A tree a live session is bound to is NOT moved.

    Prevents: moving a workspace out from under a running session. The
    session's next command would run in a directory that is no longer there,
    and the plugin's own records would name a path that no longer exists.
    """
    ws = _seed_workspace(layout, sessions=["sid-live"])
    STORE.write_state("sid-live", {
        "mode": "repo", "repo": "demo", "slug": ws.name, "path": str(ws),
    })
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    out = MIGRATE.migrate_workspaces()

    assert out["moved"] == [], out
    assert len(out["in_use"]) == 1, out
    assert out["in_use"][0]["path"] == str(ws), out
    assert ws.is_dir(), "the in-use workspace was moved anyway"
    assert not (layout["new_base"] / "demo" / ws.name).exists()


def test_an_ended_session_does_not_block_the_move(layout, monkeypatch):
    """A session that ENDED does not hold its workspace hostage.

    WHY: the in-use rule exists to protect a running session, not to freeze
    the disk forever. An ended session's tree is exactly what the migration
    is for.
    """
    ws = _seed_workspace(layout, sessions=["sid-ended"])
    STORE.write_state("sid-ended", {
        "mode": "repo", "repo": "demo", "slug": ws.name, "path": str(ws),
        "session_ended_at": "2024-01-02T00:00:00",
    })
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    out = MIGRATE.migrate_workspaces()

    assert len(out["moved"]) == 1, out
    assert (layout["new_base"] / "demo" / ws.name).is_dir()


# ── the verification: a move that cannot be verified is reported ─────────────


def test_a_broken_jj_pointer_is_reported_as_verify_failed(layout, monkeypatch):
    """A workspace whose jj pointer stops resolving is reported, not hidden.

    WHY: the whole point of the move is that the workspace keeps working. If
    the ``.jj/repo`` pointer no longer resolves to the same store after the
    move, the workspace is broken - and a broken workspace that is silently
    reported as "moved" is worse than one that is reported as failed,
    because the operator stops looking.

    The workspace is left where it was moved to: rolling back is a SECOND
    move, and a second move can fail in a new place. The failure is named.
    """
    if not shutil.which("jj"):
        pytest.skip("jj not installed")

    repo_dir = layout["repo_dir"]
    main = repo_dir / "demo"
    main.mkdir()
    subprocess.run(["jj", "git", "init", "--colocate", str(main)],
                   check=True, capture_output=True)
    subprocess.run(["jj", "commit", "-m", "init"], cwd=main,
                   check=True, capture_output=True)

    legacy = layout["legacy"] / "demo"
    legacy.mkdir(parents=True)
    slug = "demo-abc123"
    subprocess.run(["jj", "workspace", "add", str(legacy / slug)], cwd=main,
                   check=True, capture_output=True)
    ws = legacy / slug
    STORE.write_record(ws, {
        "repo": "demo", "slug": slug, "path": str(ws),
        "vcs": "jj", "sessions": [], "created_at": "2024-01-01T00:00:00",
    })
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    # Break the pointer AFTER the migration captured it: the check compares
    # the store BEFORE the move with the store AFTER it.
    real_rename = os.rename

    def _rename_and_break(src, dst, *args, **kwargs):
        real_rename(src, dst, *args, **kwargs)
        # A pointer that names a store which does not exist: the workspace is
        # now broken, and the verification has to catch it.
        (Path(dst) / ".jj" / "repo").write_text("../../../../nope/.jj/repo\n")

    monkeypatch.setattr(MIGRATE.os, "rename", _rename_and_break)

    out = MIGRATE.migrate_workspaces()

    assert len(out["verify_failed"]) == 1, out
    assert out["verify_failed"][0]["new_path"].endswith(slug), out
    assert "instead of" in out["verify_failed"][0]["reason"], out


# ── the git worktree repair step ─────────────────────────────────────────────


def test_a_git_worktree_is_repaired_after_the_move(layout, monkeypatch):
    """A moved git worktree is repaired, so the repo's worktree list is right.

    WHY the repair exists: a git worktree's entry in the main checkout's
    ``.git/worktrees`` names the OLD path, so after the move git marks the
    entry ``prunable`` and the checkout's worktree list points at a
    directory that is no longer there. ``git worktree repair`` rewrites the
    entry to the new path.
    """
    repo_dir = layout["repo_dir"]
    main = repo_dir / "demo"
    main.mkdir()
    subprocess.run(["git", "init", "-q", "."], cwd=main, check=True, capture_output=True)
    (main / "a.txt").write_text("hello\n")
    _git_commit(main)

    legacy = layout["legacy"] / "demo"
    legacy.mkdir(parents=True)
    slug = "demo-abc123"
    ws = legacy / slug
    subprocess.run(["git", "-C", str(main), "worktree", "add", "-q", str(ws)],
                   check=True, capture_output=True)
    STORE.write_record(ws, {
        "repo": "demo", "slug": slug, "path": str(ws),
        "vcs": "git", "sessions": [], "created_at": "2024-01-01T00:00:00",
    })
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    out = MIGRATE.migrate_workspaces()

    new = layout["new_base"] / "demo" / slug
    assert len(out["moved"]) == 1, out
    assert out["repair_failed"] == [], out
    # the worktree list of the main checkout names the NEW path
    listing = subprocess.run(
        ["git", "-C", str(main), "worktree", "list"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert str(new) in listing, listing
    assert str(ws) not in listing, listing


def test_a_failed_repair_is_reported_but_the_move_stays(layout, monkeypatch):
    """A repair that fails is reported; the move is not undone.

    WHY the move is not rolled back: a rollback is a SECOND move, and a
    second move can fail in a new place - leaving the workspace half-way is
    worse than leaving it moved with the problem named. The operator sees
    ``repair_failed`` and can run the repair by hand.
    """
    repo_dir = layout["repo_dir"]
    main = repo_dir / "demo"
    main.mkdir()
    subprocess.run(["git", "init", "-q", "."], cwd=main, check=True, capture_output=True)
    (main / "a.txt").write_text("hello\n")
    _git_commit(main)

    legacy = layout["legacy"] / "demo"
    legacy.mkdir(parents=True)
    slug = "demo-abc123"
    ws = legacy / slug
    subprocess.run(["git", "-C", str(main), "worktree", "add", "-q", str(ws)],
                   check=True, capture_output=True)
    STORE.write_record(ws, {
        "repo": "demo", "slug": slug, "path": str(ws),
        "vcs": "git", "sessions": [], "created_at": "2024-01-01T00:00:00",
    })
    monkeypatch.setattr(STORE, "dirty", lambda path: False)
    monkeypatch.setattr(MIGRATE, "_repair_worktrees", lambda *a, **kw: (False, "git said no"))

    out = MIGRATE.migrate_workspaces()

    assert len(out["moved"]) == 1, out
    assert len(out["repair_failed"]) == 1, out
    assert out["repair_failed"][0]["reason"] == "git said no", out
    assert (layout["new_base"] / "demo" / slug).is_dir(), "the move was undone"


# ── the legacy root: the configured one, not the derived one ─────────────────


def test_the_legacy_root_is_read_from_the_configuration():
    """The migration finds the OLD root even though settings() now derives a new one.

    WHY this matters: ``settings()["workspace_root"]`` returns the DERIVED
    root now. If the migration read it from there, it would look for the old
    workspaces under the NEW directory, find nothing, and report "nothing to
    move" - and the operator would have to move every tree by hand.
    """
    from plugin_loader import SETTINGS

    configured = "/srv/old-workspaces"
    SETTINGS.reset_deprecation_hints()
    cfg = {
        "plugins": {"entries": {"hermes-treeps": {"settings": {
            "repo_dir": "/srv/repos",
            "workspace_root": configured,
        }}}},
    }
    assert SETTINGS.legacy_workspace_root(cfg) == configured
    # and the derived root is a DIFFERENT place - the two must not be confused
    derived = SETTINGS.settings(cfg)["workspace_root"]
    assert derived == "/srv/repos/.workspaces", derived
    assert derived != configured


def test_the_legacy_root_is_empty_when_nothing_was_configured():
    """No configured root means the shipped default applies."""
    from plugin_loader import SETTINGS

    assert SETTINGS.legacy_workspace_root({}) == ""


# ── the report shape ─────────────────────────────────────────────────────────


def test_the_report_has_the_five_documented_keys(layout):
    """The report shape is part of the contract: the CLI prints it, and a
    script may read it. A missing key is a KeyError in somebody's script."""
    out = MIGRATE.migrate_workspaces()
    assert set(out) == {
        "moved", "skipped_dirty", "in_use", "verify_failed", "repair_failed",
    }, out


def test_the_report_text_names_every_bucket(layout, monkeypatch):
    """The human-readable report mentions each bucket, so nothing is silent."""
    text = MIGRATE.report_text({
        "moved": [{"path": "/old/a", "new_path": "/new/a"}],
        "skipped_dirty": [{"path": "/old/b", "reason": "uncommitted changes"}],
        "in_use": [{"path": "/old/c", "reason": "live session"}],
        "verify_failed": [{"path": "/old/d", "new_path": "/new/d", "reason": "broken"}],
        "repair_failed": [{"path": "/new/e", "reason": "git said no"}],
    })
    for needle in ("moved", "skipped_dirty", "in_use", "verify_failed",
                   "repair_failed", "/old/a", "/new/a", "uncommitted changes"):
        assert needle in text, f"{needle!r} missing from the report:\n{text}"


# ── the jj colocated layout: the pointer survives the move ───────────────────


def test_the_jj_pointer_still_resolves_after_the_move(layout, monkeypatch):
    """The relative .jj/repo pointer resolves to the SAME store after the move.

    WHY: a jj child workspace carries a RELATIVE pointer to the main
    workspace's store. The move keeps the depth below the repo dir the same
    (``workspaces/REPO/SLUG`` -> ``.workspaces/REPO/SLUG``), so the pointer
    still resolves. If it did not, every moved workspace would be a broken
    checkout - which is why the migration verifies it and reports
    ``verify_failed`` when it does not hold.
    """
    if not shutil.which("jj"):
        pytest.skip("jj not installed")

    repo_dir = layout["repo_dir"]
    main = repo_dir / "demo"
    main.mkdir()
    subprocess.run(["jj", "git", "init", "--colocate", str(main)],
                   check=True, capture_output=True)
    subprocess.run(["jj", "commit", "-m", "init"], cwd=main,
                   check=True, capture_output=True)

    legacy = layout["legacy"] / "demo"
    legacy.mkdir(parents=True)
    slug = "demo-abc123"
    subprocess.run(["jj", "workspace", "add", str(legacy / slug)], cwd=main,
                   check=True, capture_output=True)
    ws = legacy / slug
    STORE.write_record(ws, {
        "repo": "demo", "slug": slug, "path": str(ws),
        "vcs": "jj", "sessions": [], "created_at": "2024-01-01T00:00:00",
    })
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    store_before = MIGRATE._resolve_jj_store(ws)
    assert store_before is not None, "the fixture did not produce a jj pointer"

    out = MIGRATE.migrate_workspaces()

    new = layout["new_base"] / "demo" / slug
    assert len(out["moved"]) == 1, out
    assert out["verify_failed"] == [], out
    assert new.is_dir(), "the workspace did not move"
    store_after = MIGRATE._resolve_jj_store(new)
    assert store_after == store_before, (
        f"the jj pointer now resolves to {store_after} instead of {store_before}"
    )
