"""Tests for the immediate child reclamation.

The subject: when a delegated child is DONE, its workspace is reclaimed at
once - and the work is never lost. ``jj workspace remove`` snapshots the
working copy (even the dirty one) into a commit in the shared store; the
plugin records that commit id in the database, because jj never auto-gcs
and an unrecorded commit is a commit nobody can find.

These tests use a REAL jj repo in a tmp dir: the whole point of the feature
is the jj command's behaviour, and a mocked ``jj`` would only test the mock.
The jj-installed guard follows the pattern of ``tests/test_prepare.py``.

``jj workspace remove`` needs jj >= 0.46 (it was added there). On an older
jj the removal tests skip - but the GUARD tests (primary workspace, wrong
purpose, git worktree, live session) run everywhere, because they are
decided before any jj command is issued.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from plugin_loader import STORE

CHILD_PREFIX = STORE.CHILD_PURPOSE_PREFIX


def _jj_version() -> tuple[int, ...]:
    """The installed jj version as a tuple, or () when jj is missing."""
    if not shutil.which("jj"):
        return ()
    try:
        out = subprocess.run(["jj", "--version"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ()
    # "jj 0.46.0"
    for token in out.split():
        if token[:1].isdigit():
            parts = []
            for piece in token.split("."):
                try:
                    parts.append(int(piece))
                except ValueError:
                    parts.append(0)
            return tuple(parts)
    return ()


JJ_VERSION = _jj_version()


def _jj_supports_workspace_remove() -> bool:
    """True when the installed jj has ``jj workspace remove``.

    ``jj workspace remove`` was added in jj 0.46 - it is what snapshots the
    dirty working copy into a commit before removing the workspace. On an
    older jj the command does not exist, and the removal tests skip.
    """
    if not JJ_VERSION:
        return False
    try:
        proc = subprocess.run(
            ["jj", "workspace", "--help"], capture_output=True, text=True, timeout=60
        )
    except Exception:
        return False
    return "remove" in (proc.stdout or "")


needs_workspace_remove = pytest.mark.skipif(
    not _jj_supports_workspace_remove(),
    reason="jj workspace remove needs jj >= 0.46",
)


def _jj(repo: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["jj", *args], cwd=str(cwd or repo), check=True, capture_output=True, text=True
    )


@pytest.fixture
def jj_repo(tmp_path):
    """A colocated jj repo with one child workspace, both under tmp_path.

    The layout mirrors the plugin's: the main checkout and the child
    workspace live side by side, and the child's ``.jj/repo`` is a FILE
    pointing at the main workspace's store.
    """
    if not JJ_VERSION:
        pytest.skip("jj not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["jj", "git", "init", "--colocate", str(repo)], check=True, capture_output=True)
    # jj commits are empty by default - there is no --allow-empty flag (that
    # is a git flag), and passing it makes jj refuse the whole command.
    subprocess.run(["jj", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    child = tmp_path / "ws" / "repo-child-abc123"
    child.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["jj", "workspace", "add", str(child), "--name", child.name],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return {"repo": repo, "child": child}


@pytest.fixture
def jj_repo_named(tmp_path):
    """A colocated jj repo, with NO workspace in it yet.

    WHY this exists: the jj store names workspaces, and two workspaces in
    one store cannot share a name. A test that needs a second workspace
    (under the plugin's root, say) has to add it to a store where the name
    is still free.
    """
    if not JJ_VERSION:
        pytest.skip("jj not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["jj", "git", "init", "--colocate", str(repo)], check=True, capture_output=True)
    subprocess.run(["jj", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    return {"repo": repo}


def _record(child: Path, *, purpose: str = CHILD_PREFIX + "parent-1", sessions=("child-1",)):
    """Write the plugin's workspace record for a child workspace."""
    STORE.write_record(
        child,
        {
            "repo": "repo",
            "slug": child.name,
            "path": str(child),
            "vcs": "jj",
            "purpose": purpose,
            "sessions": list(sessions),
            "created_at": "2020-01-01T00:00:00",
        },
    )


def _ended(sessions=("child-1",)):
    """Mark the given sessions ended."""
    for sid in sessions:
        STORE.write_state(
            sid,
            {
                "mode": "repo",
                "repo": "repo",
                "slug": "repo-child-abc123",
                "path": "x",
                "session_ended_at": "2020-01-02T00:00:00",
            },
        )


@pytest.fixture
def plugin_root(tmp_path, monkeypatch):
    """Point the plugin's workspace root at a tmp dir.

    WHY this exists: the BATCH path (``reclaim_children``) looks a workspace
    up by slug through ``find_workspace()``, and that lookup REFUSES any
    path outside the workspace root - a real security guard, not an
    inconvenience. A test that exercises the batch path therefore has to
    put its workspace under the root, exactly like the plugin does in
    production.
    """
    root = tmp_path / "repos" / ".workspaces" / "repo"
    root.mkdir(parents=True)
    monkeypatch.setattr(STORE, "repo_dir", lambda: tmp_path / "repos")
    monkeypatch.setattr(
        STORE,
        "settings",
        lambda: {
            "repo_dir": str(tmp_path / "repos"),
            "workspace_root": str(tmp_path / "repos" / ".workspaces"),
            "repo_roots": [str(tmp_path)],
        },
    )
    return root


# ── (a) the dirty child: the work survives as a snapshot commit ──────────────


@needs_workspace_remove
def test_a_dirty_child_is_reclaimed_and_its_work_becomes_a_commit(jj_repo, monkeypatch):
    """The dirty tree is snapshotted into a commit, and the commit is recorded.

    Prevents: the reclamation deleting a child's uncommitted work - the
    whole point of "reclaim immediately" is that it is SAFE, and the only
    thing that makes it safe is the snapshot.
    """
    repo = jj_repo["repo"]
    child = jj_repo["child"]
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)
    (child / "work.txt").write_text("the child's work\n")
    _record(child)
    _ended()

    out = STORE.reclaim_child_workspace(child)

    assert out["ok"], out
    assert out["was_dirty"] is True, out
    assert out["snapshot_commit_id"], out
    assert not child.exists(), "the tree should be gone after the reclamation"

    # The snapshot commit is REAL: it resolves in the main workspace and it
    # holds the child's file.
    shown = subprocess.run(
        ["jj", "log", "-r", out["snapshot_commit_id"], "--no-graph", "-T", "commit_id"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert shown.returncode == 0, shown.stderr
    assert out["snapshot_commit_id"] in shown.stdout, shown.stdout
    content = subprocess.run(
        ["jj", "file", "show", "-r", out["snapshot_commit_id"], "work.txt"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert content.returncode == 0, content.stderr
    assert content.stdout.strip() == "the child's work", content.stdout


@needs_workspace_remove
def test_the_snapshot_commit_id_is_recorded_in_the_database(jj_repo, monkeypatch):
    """The commit id lands in the database: the DB is the index of the work.

    Prevents: the work surviving in the store with nothing pointing at it.
    jj never auto-gcs, so an unrecorded snapshot is unreachable in practice.
    """
    repo = jj_repo["repo"]
    child = jj_repo["child"]
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)
    (child / "work.txt").write_text("work\n")
    _record(child)
    _ended()

    out = STORE.reclaim_child_workspace(child)

    row = STORE.db.read_child_reclaim("repo", child.name)
    assert row is not None, "the reclamation was not recorded"
    assert row["snapshot_commit_id"] == out["snapshot_commit_id"], row
    assert row["was_dirty"] is True, row
    assert row["reclaimed_at"], row


@needs_workspace_remove
def test_reclaiming_deletes_the_plugin_rows_for_the_workspace(jj_repo, monkeypatch):
    """The workspace row and its session rows are gone after the reclamation.

    Prevents: a stale record pointing at a path that no longer exists -
    the "stale workspace" the plugin's own status output would report.
    """
    repo = jj_repo["repo"]
    child = jj_repo["child"]
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)
    _record(child)
    _ended()

    STORE.reclaim_child_workspace(child)

    assert STORE.db.read_workspace(str(child)) is None
    assert STORE.db.workspace_sessions("repo", child.name) == []


# ── (b) the clean child: nothing to snapshot ─────────────────────────────────


@needs_workspace_remove
def test_a_clean_child_is_reclaimed_without_a_snapshot(jj_repo, monkeypatch):
    """A clean tree has nothing to preserve, so no snapshot is needed."""
    repo = jj_repo["repo"]
    child = jj_repo["child"]
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)
    _record(child)
    _ended()

    out = STORE.reclaim_child_workspace(child)

    assert out["ok"], out
    assert out["was_dirty"] is False, out
    assert not child.exists(), "the tree should be gone"


@needs_workspace_remove
def test_a_clean_child_is_unregistered_from_the_jj_store(jj_repo, monkeypatch):
    """The workspace is GONE from the jj store, not just from the disk.

    Prevents: a stale jj workspace entry. jj would then refuse to add a
    workspace with the same name, and the plugin's own list would show a
    workspace that does not exist.
    """
    repo = jj_repo["repo"]
    child = jj_repo["child"]
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)
    _record(child)
    _ended()

    STORE.reclaim_child_workspace(child)

    listing = subprocess.run(
        ["jj", "workspace", "list"], cwd=str(repo), capture_output=True, text=True
    ).stdout
    assert child.name not in listing, listing


@needs_workspace_remove
def test_the_jj_registration_is_gone_even_when_jj_says_no_such_workspace(
    jj_repo, monkeypatch
):
    """A removal that jj did not do is reported, not counted as a success.

    Prevents: the silent half-failure. ``jj workspace remove`` given a
    PATH prints "Warning: No such workspace" and STILL EXITS 0 - so a
    caller that only checks the return code believes the workspace was
    removed, while jj still has it registered and the next
    ``jj workspace add`` of that name fails. The reclamation must verify
    that the registration is really gone.

    The tree is DIRTY on purpose: the dirty path is the one that runs
    ``jj workspace remove``, which is the command with the exit-0 trap.
    """
    repo = jj_repo["repo"]
    child = jj_repo["child"]
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)
    (child / "work.txt").write_text("work\n")
    _record(child)
    _ended()

    # Make the removal a no-op: pretend jj removed a workspace it never had.
    monkeypatch.setattr(STORE, "_jj_workspace_remove", lambda repo_path, slug: (True, ""))

    out = STORE.reclaim_child_workspace(child)

    assert not out["ok"], out
    assert out["reason"] == "jj_workspace_still_registered", out
    # The stale registration is still there - that is the whole point.
    listing = subprocess.run(
        ["jj", "workspace", "list"], cwd=str(repo), capture_output=True, text=True
    ).stdout
    assert child.name in listing, listing


@needs_workspace_remove
def test_the_reclamation_uses_the_workspace_name_not_its_path(jj_repo, monkeypatch):
    """``jj workspace remove`` takes a NAME; a path makes jj exit 0 and do nothing.

    WHY this is a test and not a comment: jj answers a path with
    "Warning: No such workspace" on stdout and a return code of 0. The
    reclamation therefore has to pass the NAME (the plugin creates its
    workspaces with ``jj workspace add <path> --name <slug>``) - and the
    result has to be verified against ``jj workspace list``.
    """
    repo = jj_repo["repo"]
    child = jj_repo["child"]
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)
    (child / "work.txt").write_text("work\n")
    _record(child)
    _ended()

    out = STORE.reclaim_child_workspace(child)

    assert out["ok"], out
    listing = subprocess.run(
        ["jj", "workspace", "list", "--template", "name ++ \"\\n\""],
        cwd=str(repo),
        capture_output=True,
        text=True,
    ).stdout
    assert child.name not in listing.split(), listing


# ── (c) the primary workspace is refused ─────────────────────────────────────


def test_the_primary_workspace_is_never_reclaimed(tmp_path, monkeypatch):
    """The MAIN jj workspace is refused - jj forgets it with no warning.

    Prevents: upstream jj#5271. ``jj workspace forget`` on the main
    workspace makes jj forget the MAIN checkout without a warning, and the
    repo is then left with no main working copy at all.

    The guard is on the POINTER SHAPE: the main workspace's ``.jj/repo`` is
    a DIRECTORY, a child's is a FILE.
    """
    repo = tmp_path / "repo"
    (repo / ".jj" / "repo").mkdir(parents=True)
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)
    _record(repo, purpose=CHILD_PREFIX + "parent-1")

    out = STORE.reclaim_child_workspace(repo)

    assert not out["ok"], out
    assert out["reason"] == STORE.RECLAIM_REFUSED_PRIMARY, out
    assert repo.exists(), "the primary workspace must survive"


# ── (d) a non-child purpose is refused ───────────────────────────────────────


def test_a_user_workspace_is_never_reclaimed(tmp_path, monkeypatch):
    """A workspace without the ``child:`` purpose is refused.

    Prevents: the reclamation deleting a HUMAN's workspace. The immediate
    path is a machine decision about a throwaway delegation tree.
    """
    ws = tmp_path / "repo" / "my-work"
    ws.mkdir(parents=True)
    monkeypatch.setattr(STORE, "find_repo", lambda name: tmp_path / "repo")
    _record(ws, purpose="")

    out = STORE.reclaim_child_workspace(ws)

    assert not out["ok"], out
    assert out["reason"] == "not_a_child_workspace", out
    assert ws.exists(), "a user workspace must survive"


# ── (e) a git-only worktree is skipped, not reclaimed ────────────────────────


def test_a_git_only_worktree_is_skipped(tmp_path, monkeypatch):
    """A git worktree's dirty tree has no snapshot-safe removal.

    Prevents: losing the uncommitted changes of a git worktree.
    ``git worktree remove`` refuses a dirty worktree, and there is no git
    command that snapshots it into a commit while keeping the work.
    """
    ws = tmp_path / "repo" / "git-child"
    ws.mkdir(parents=True)
    (ws / ".git").write_text("gitdir: /somewhere/.git/worktrees/git-child\n")
    monkeypatch.setattr(STORE, "find_repo", lambda name: tmp_path / "repo")
    _record(ws)

    out = STORE.reclaim_child_workspace(ws)

    assert out["reason"] == STORE.RECLAIM_SKIPPED_GIT_WORKTREE, out
    assert ws.exists(), "a skipped worktree must be left alone"


# ── (f) idempotency ──────────────────────────────────────────────────────────


@needs_workspace_remove
def test_reclaiming_twice_is_a_no_op_success(jj_repo, monkeypatch):
    """The second call succeeds and does not lose the first snapshot id.

    Prevents: a retry (a hook firing twice, an operator running the command
    again) reporting a failure - or worse, overwriting the recorded
    snapshot id with the second call's empty one, which would strand the
    child's work in the store with nothing pointing at it.
    """
    repo = jj_repo["repo"]
    child = jj_repo["child"]
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)
    (child / "work.txt").write_text("work\n")
    _record(child)
    _ended()

    first = STORE.reclaim_child_workspace(child)
    second = STORE.reclaim_child_workspace(child)

    assert first["ok"] and second["ok"], (first, second)
    assert second["already_gone"] is True, second
    assert second["snapshot_commit_id"] == first["snapshot_commit_id"], (first, second)
    row = STORE.db.read_child_reclaim("repo", child.name)
    assert row["snapshot_commit_id"] == first["snapshot_commit_id"], row


@needs_workspace_remove
def test_reclaiming_twice_through_the_batch_path_is_a_no_op_success(
    jj_repo_named, monkeypatch, plugin_root
):
    """The BATCH path is idempotent too, and keeps the first snapshot id.

    WHY this is a separate test: the batch path looks the workspace up by
    slug through find_workspace(), and a reclaimed workspace has NO record
    left - so the lookup fails and the second run reported "unknown
    workspace". The batch path has to fall back to the reclaim row.

    The workspace has to live under the plugin's root: find_workspace()
    refuses any path outside it (a real security guard).
    """
    repo = jj_repo_named["repo"]
    child = plugin_root / "repo-child-abc123"
    subprocess.run(
        ["jj", "workspace", "add", str(child), "--name", child.name],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    (child / "work.txt").write_text("work\n")
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)
    _record(child)
    _ended()

    first = STORE.reclaim_children([child.name])
    second = STORE.reclaim_children([child.name])

    assert first["reclaimed"], first
    assert second["skipped"] == [], second
    assert second["reclaimed"], second
    assert second["reclaimed"][0]["snapshot_commit_id"] == first["reclaimed"][0]["snapshot_commit_id"], (
        first,
        second,
    )


# ── (g) --all-children reclaims exactly the ended children ───────────────────


def test_all_children_reclaims_exactly_the_ended_children(tmp_path, monkeypatch):
    """Only the children whose sessions ALL ended are reclaimed.

    Prevents: a sweep that reclaims a workspace a running session is still
    using - the child is still working in it.
    """
    root = tmp_path / "repos" / ".workspaces" / "repo"
    monkeypatch.setattr(STORE, "repo_dir", lambda: tmp_path / "repos")
    monkeypatch.setattr(STORE, "settings", lambda: {
        "repo_dir": str(tmp_path / "repos"),
        "workspace_root": str(tmp_path / "repos" / ".workspaces"),
        "repo_roots": [str(tmp_path)],
    })
    monkeypatch.setattr(STORE, "find_repo", lambda name: tmp_path / "repo")

    ended_ws = root / "repo-child-ended"
    live_ws = root / "repo-child-live"
    ended_ws.mkdir(parents=True)
    live_ws.mkdir(parents=True)
    _record(ended_ws, sessions=("child-ended",))
    _record(live_ws, sessions=("child-live",))
    _ended(("child-ended",))
    # "child-live" gets a state WITHOUT session_ended_at: still running.

    out = STORE.reclaim_children(None, all_children=True)

    reclaimed = [row["slug"] for row in out["reclaimed"]]
    assert reclaimed == [ended_ws.name], out
    assert live_ws.exists(), "a live child's workspace must survive"
    assert STORE.db.read_workspace(str(live_ws)) is not None


# ── (h) a live session's pinned workspace is refused ─────────────────────────


def test_a_live_session_pinned_workspace_is_refused(tmp_path, monkeypatch):
    """A workspace a live session is bound to is refused.

    Prevents: the tree being removed out from under a running session.
    """
    ws = tmp_path / "repo" / "repo-child-live"
    ws.mkdir(parents=True)
    monkeypatch.setattr(STORE, "find_repo", lambda name: tmp_path / "repo")
    _record(ws, sessions=("child-live",))
    # The session is LIVE: it has a binding row and no session_ended_at.
    STORE.db.write_binding(
        "child-live",
        {"mode": "repo", "repo": "repo", "slug": ws.name, "path": str(ws)},
    )

    out = STORE.reclaim_child_workspace(ws)

    assert not out["ok"], out
    assert out["reason"] == STORE.RECLAIM_REFUSED_IN_USE, out
    assert ws.exists(), "a live session's workspace must survive"


def test_the_manual_reclaim_reports_a_refused_workspace(tmp_path, monkeypatch):
    """The MANUAL reclaim path is where a refusal is reachable.

    Prevents: the in-use guard losing its coverage. The immediate stop
    flow stamps the child's session ended BEFORE it reclaims, so the guard
    never sees a live session there - the stop flow cannot refuse. The
    manual path (``hermes ws reclaim``) stamps nothing, so the guard is
    the thing that decides, and its refusal must be REPORTED as a refusal
    and not as a reclamation.

    WHY this matters for the summary too: a caller that buckets by
    ``ok`` alone would file a refusal under "reclaimed" and claim a tree
    was reclaimed while it is still on disk.
    """
    # The workspace lives under the plugin's root: the manual path looks
    # it up BY SLUG, and that lookup refuses a path outside the root.
    root = tmp_path / "repos" / ".workspaces" / "repo"
    ws = root / "repo-child-live"
    ws.mkdir(parents=True)
    monkeypatch.setattr(STORE, "repo_dir", lambda: tmp_path / "repos")
    monkeypatch.setattr(
        STORE,
        "settings",
        lambda: {
            "repo_dir": str(tmp_path / "repos"),
            "workspace_root": str(tmp_path / "repos" / ".workspaces"),
            "repo_roots": [str(tmp_path)],
        },
    )
    monkeypatch.setattr(STORE, "find_repo", lambda name: tmp_path / "repo")
    _record(ws, sessions=("child-live",))
    # The session is LIVE: a binding row and no session_ended_at.
    STORE.db.write_binding(
        "child-live",
        {"mode": "repo", "repo": "repo", "slug": ws.name, "path": str(ws)},
    )

    out = STORE.reclaim_children([ws.name])

    assert out["reclaimed"] == [], out
    assert len(out["skipped"]) == 1, out
    assert out["skipped"][0]["reason"] == STORE.RECLAIM_REFUSED_IN_USE, out
    assert ws.exists(), "a live session's workspace must survive"


# ── the age-based sweep goes through the SAME primitive ──────────────────────


def test_the_child_sweep_uses_the_shared_reclamation_primitive(tmp_path, monkeypatch):
    """The age-based sweep and the immediate path are ONE code path.

    WHY this is the consistency deliverable: two removal code paths for a
    child workspace is how a tree ends up removed from the disk but still
    registered in the jj store, or removed from the store but still in the
    database. The sweep must call ``reclaim_child_workspace`` - the same
    function the immediate CLI path calls.
    """
    calls: list[str] = []

    def _fake_reclaim(path, **kwargs):
        calls.append(str(path))
        return {"ok": True, "slug": Path(path).name, "detail": "reclaimed", "snapshot_commit_id": "x"}

    monkeypatch.setattr(STORE, "reclaim_child_workspace", _fake_reclaim)
    monkeypatch.setattr(
        STORE,
        "gc_tiers",
        lambda: {
            "tier1": [
                {
                    "slug": "repo-child-abc123",
                    "path": "/tmp/repo-child-abc123",
                    "record": {"purpose": CHILD_PREFIX + "parent-1"},
                }
            ],
            "tier2": [],
            "tier3": [],
        },
    )

    done = STORE.gc_children_apply()

    assert calls == ["/tmp/repo-child-abc123"], calls
    assert done, done


def test_the_sweep_never_reclaims_a_user_workspace(tmp_path, monkeypatch):
    """The sweep touches child workspaces ONLY.

    Prevents: the sweep deciding for a human's workspace. The child tier
    is selected by the ``child:`` purpose, and everything else is left
    alone.
    """
    calls: list[str] = []

    monkeypatch.setattr(
        STORE,
        "reclaim_child_workspace",
        lambda path, **kwargs: calls.append(str(path)) or {"ok": True, "detail": "reclaimed"},
    )
    monkeypatch.setattr(
        STORE,
        "gc_tiers",
        lambda: {
            "tier1": [
                {
                    "slug": "my-work",
                    "path": "/tmp/my-work",
                    "record": {"purpose": ""},
                }
            ],
            "tier2": [],
            "tier3": [],
        },
    )

    STORE.gc_children_apply()

    assert calls == [], calls


# ── the database change is additive ──────────────────────────────────────────


def test_an_existing_database_is_migrated_without_losing_its_rows(tmp_path, monkeypatch):
    """An OLD database gains the new table and keeps every row it had.

    Prevents: the migration breaking a live install. The plugin's database
    lives under the host's plugin-data directory and holds real session
    bindings; opening it must never drop, rewrite or retype anything. The
    new table is CREATED, and the old rows stay.
    """
    import sqlite3

    db_file = tmp_path / "plugin-data" / "data.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(str(db_file))
    legacy.executescript(
        "CREATE TABLE IF NOT EXISTS bindings ("
        " session_id TEXT PRIMARY KEY, repo TEXT, slug TEXT, path TEXT,"
        " vcs TEXT, mode TEXT, title TEXT, inherited_from TEXT, inherited_at TEXT,"
        " session_ended_at TEXT, extra TEXT, updated_at TEXT);"
        "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '',"
        " updated_at TEXT);"
        "INSERT INTO kv(key, value) VALUES('schema_version', '2');"
        "INSERT INTO bindings(session_id, repo, slug, path, vcs, mode)"
        " VALUES('live-session-1', 'repo', 'a-slug', '/some/path', 'jj', 'repo');"
    )
    legacy.commit()
    legacy.close()

    monkeypatch.setenv("HERMES_TREEPS_PLUGIN_DATA_DIR", str(tmp_path / "plugin-data"))
    monkeypatch.setenv("HERMES_TREEPS_DB_PATH", str(db_file))
    STORE.db.reset()

    conn = STORE.db.connect()
    version = conn.execute("SELECT value FROM kv WHERE key='schema_version'").fetchone()
    assert str(version[0]) == str(STORE.db.SCHEMA_VERSION), version
    tables = {
        str(r[0])
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "child_reclaims" in tables, tables
    # The pre-existing row survived the migration.
    row = conn.execute(
        "SELECT path FROM bindings WHERE session_id='live-session-1'"
    ).fetchone()
    assert row is not None and row[0] == "/some/path", row


def test_the_snapshot_index_survives_a_repeated_open(tmp_path, monkeypatch):
    """Opening the database twice does not fail on the migration.

    WHY this is a test: the migration runs on EVERY open of an existing
    database, so it has to be idempotent. A second open that raised would
    break every plugin call after the first one.

    The connection is CACHED per thread (``db.connect()`` keeps it), so
    ``db.reset()`` is what forces the next call to open the database
    again - that is the "open" this test needs to repeat.
    """
    monkeypatch.setenv("HERMES_TREEPS_PLUGIN_DATA_DIR", str(tmp_path / "plugin-data"))
    monkeypatch.setenv("HERMES_TREEPS_DB_PATH", str(tmp_path / "plugin-data" / "data.db"))
    STORE.db.reset()

    for _ in range(3):
        conn = STORE.db.connect()
        assert conn.execute("SELECT 1").fetchone()[0] == 1
        # Force the next call to open the database from scratch.
        STORE.db.reset()


def test_the_table_is_created_even_when_the_version_row_already_says_v3(
    tmp_path, monkeypatch
):
    """The table is created regardless of what the version row says.

    WHY this is a test: another branch (the child-report one) ALSO bumps
    the schema version from 2 to 3, for its own tables. If the migration
    were gated on "the version row is below 3", whichever branch landed
    second would see a row that already says 3, skip its migration, and
    never create its table - the reclamation would then fail with "no such
    table: child_reclaims" on the live install.

    So the migration creates the table UNCONDITIONALLY. This test pins that
    behaviour: a database whose version row already says 3, and which has
    no ``child_reclaims`` table, gets the table.
    """
    import sqlite3

    db_file = tmp_path / "plugin-data" / "data.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(str(db_file))
    legacy.executescript(
        "CREATE TABLE IF NOT EXISTS bindings ("
        " session_id TEXT PRIMARY KEY, repo TEXT, slug TEXT, path TEXT,"
        " vcs TEXT, mode TEXT, title TEXT, inherited_from TEXT, inherited_at TEXT,"
        " session_ended_at TEXT, extra TEXT, updated_at TEXT);"
        "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '',"
        " updated_at TEXT);"
        # A version row that already claims the version ANOTHER branch
        # bumped to - and no child_reclaims table.
        "INSERT INTO kv(key, value) VALUES('schema_version', '3');"
    )
    legacy.commit()
    legacy.close()

    monkeypatch.setenv("HERMES_TREEPS_PLUGIN_DATA_DIR", str(tmp_path / "plugin-data"))
    monkeypatch.setenv("HERMES_TREEPS_DB_PATH", str(db_file))
    STORE.db.reset()

    conn = STORE.db.connect()
    tables = {
        str(r[0])
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "child_reclaims" in tables, tables


# ── the dry run changes nothing ──────────────────────────────────────────────


def test_a_dry_run_reclaims_nothing(tmp_path, monkeypatch):
    """``--dry-run`` reports and touches nothing."""
    ws = tmp_path / "repo" / "repo-child-abc123"
    ws.mkdir(parents=True)
    monkeypatch.setattr(STORE, "find_repo", lambda name: tmp_path / "repo")
    _record(ws)

    out = STORE.reclaim_child_workspace(ws, dry_run=True)

    assert out["ok"], out
    assert out["dry_run"] is True, out
    assert ws.exists(), "a dry run must not remove the tree"
    assert STORE.db.read_workspace(str(ws)) is not None
    assert STORE.db.read_child_reclaim("repo", ws.name) is None


# ── the CLI ──────────────────────────────────────────────────────────────────


def test_the_cli_reports_a_structured_reclaim_report(tmp_path, monkeypatch, capsys):
    """``hermes ws reclaim --slug`` prints the structured report."""
    import importlib

    plugin = importlib.import_module("hermes_treeps")
    ws = tmp_path / "repo" / "repo-child-abc123"
    ws.mkdir(parents=True)
    monkeypatch.setattr(STORE, "find_repo", lambda name: tmp_path / "repo")
    _record(ws)

    class _Args:
        ws_action = "reclaim"
        slug = ws.name
        all_children = False
        dry_run = True
        no_snapshot = False

    assert plugin._cli(_Args()) == 0
    printed = capsys.readouterr().out
    report = json.loads(printed.split("\ndry run:")[0])
    assert "reclaimed" in report and "skipped" in report, report


def test_the_cli_needs_a_slug_or_all_children(capsys):
    """``hermes ws reclaim`` with neither argument says so and refuses."""
    import importlib

    plugin = importlib.import_module("hermes_treeps")

    class _Args:
        ws_action = "reclaim"
        slug = ""
        all_children = False
        dry_run = False
        no_snapshot = False

    assert plugin._cli(_Args()) == 2
    assert "--slug" in capsys.readouterr().out
