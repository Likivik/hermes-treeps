"""store.py: the choke points that keep a workspace inside its root.

Each test names the failure it prevents. These are the bugs that turned a
slug into a path OUTSIDE the workspace root - and retire() then deleted it.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest
from plugin_loader import STORE

WorkspaceError = STORE.WorkspaceError


def _fake_root(tmp_path, monkeypatch):
    """A workspace root under tmp_path, with one real workspace in it."""
    root = tmp_path / "workspaces"
    (root / "repo" / "ws1" / ".hermes").mkdir(parents=True)
    rec = root / "repo" / "ws1" / ".hermes" / "session.json"
    rec.write_text(json.dumps({"repo": "repo", "slug": "ws1", "sessions": ["sid-a"]}))
    monkeypatch.setattr(STORE, "root_dir", lambda: root)
    return root


# ── find_workspace: path traversal ────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_slug",
    [
        "../../outside/victim",
        "../victim",
        "..",
        "/etc",
        "/tmp/absolute",
        "A/B",
        "repo/ws1",
        "ws1;rm -rf /",
        "",
    ],
)
def test_find_workspace_refuses_traversal_slugs(tmp_path, monkeypatch, bad_slug):
    """A slug is a NAME, never a path.

    Prevents (P0-3): find_workspace() built `root.glob(f"*/{slug}")`, so a slug
    like "../../outside/victim" resolved to a path OUTSIDE the root. rename()
    then MOVED it and retire() `rm -rf`'d it - deleting an arbitrary tree.

    The victim directory really exists here, so a slug that MATCHES something
    outside the root is caught: a slug that matched nothing would raise the
    same "no workspace" error and hide the bug.

    The message is asserted too: find_workspace() has TWO defences (the slug
    shape and the resolved-path check) and either one alone would stop the
    traversal. Pinning the message pins the shape check, so removing it is
    caught even though the other defence still holds.
    """
    _fake_root(tmp_path, monkeypatch)
    victim = tmp_path / "outside" / "victim"
    (victim / ".hermes").mkdir(parents=True)
    (victim / ".hermes" / "session.json").write_text(json.dumps({"repo": "repo", "slug": "victim"}))
    assert victim.exists()
    with pytest.raises(WorkspaceError) as exc:
        STORE.find_workspace(bad_slug)
    assert "invalid slug" in str(exc.value), (
        f"{bad_slug!r} was refused for the wrong reason: {exc.value}"
    )


def test_find_workspace_refuses_a_hit_outside_the_root(tmp_path, monkeypatch):
    """A hit that resolves outside the root is refused, even with a valid slug.

    Prevents (P0-3): a symlink under the root pointing elsewhere passes the
    slug-shape check, so the resolved path must be checked against the root.
    """
    root = _fake_root(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    (outside / ".hermes").mkdir(parents=True)
    (outside / ".hermes" / "session.json").write_text(json.dumps({"repo": "repo", "slug": "linked"}))
    (root / "repo" / "linked").symlink_to(outside)
    with pytest.raises(WorkspaceError):
        STORE.find_workspace("linked")


def test_find_workspace_finds_a_real_workspace(tmp_path, monkeypatch):
    """The happy path still works - the fix must not break normal use."""
    root = _fake_root(tmp_path, monkeypatch)
    assert STORE.find_workspace("ws1") == root / "repo" / "ws1"


# ── rename: the old slug is validated too ─────────────────────────────────────


def test_rename_refuses_an_invalid_old_slug(tmp_path, monkeypatch):
    """rename() MOVES a directory, so the OLD slug must be validated as well.

    Prevents (P0-3): rename() validated only new_slug, so a traversal in
    old_slug moved a tree from outside the root.
    """
    _fake_root(tmp_path, monkeypatch)
    with pytest.raises(WorkspaceError) as exc:
        STORE.rename("../../outside/victim", "newname")
    assert "invalid slug" in str(exc.value)
    with pytest.raises(WorkspaceError) as exc:
        STORE.rename("ws1", "../../outside/victim")
    assert "invalid new slug" in str(exc.value)


def test_rename_moves_a_real_workspace(tmp_path, monkeypatch):
    """The happy path still works."""
    root = _fake_root(tmp_path, monkeypatch)
    STORE.rename("ws1", "ws2")
    assert (root / "repo" / "ws2" / ".hermes" / "session.json").exists()
    assert not (root / "repo" / "ws1").exists()


# ── detach ────────────────────────────────────────────────────────────────────


def test_detach_removes_the_session_from_a_record(tmp_path, monkeypatch):
    """detach() drops one session id from a workspace's binding.

    Prevents (P1-6): detach() was dead code - rebinding left the OLD workspace
    still listing the session as live, so retire() refused it forever and it
    stayed on disk, owned by nobody.
    """
    root = _fake_root(tmp_path, monkeypatch)
    ws = root / "repo" / "ws1"
    STORE.detach(ws, "sid-a")
    rec = json.loads((ws / ".hermes" / "session.json").read_text())
    assert rec["sessions"] == []


def test_detach_leaves_other_sessions_alone(tmp_path, monkeypatch):
    """detach() removes one session, not all of them."""
    root = _fake_root(tmp_path, monkeypatch)
    ws = root / "repo" / "ws1"
    rec_path = ws / ".hermes" / "session.json"
    rec = json.loads(rec_path.read_text())
    rec["sessions"] = ["sid-a", "sid-b"]
    rec_path.write_text(json.dumps(rec))
    STORE.detach(ws, "sid-a")
    rec = json.loads(rec_path.read_text())
    assert rec["sessions"] == ["sid-b"]


def test_detach_on_an_unknown_workspace_is_a_noop(tmp_path):
    """detach() on a path with no record does not raise."""
    STORE.detach(tmp_path / "nope", "sid-x")


# ── unpushed_commits: the jj revset ──────────────────────────────────────────


@pytest.mark.skipif(shutil.which("jj") is None, reason="jj is not installed")
def test_unpushed_commits_sees_an_unbookmarked_jj_commit(tmp_path):
    """An agent's own `jj commit` (no bookmark) counts as UNPUSHED.

    Prevents (P0-4): the revset was `remote_bookmarks()..bookmarks()`, which
    only sees BOOKMARKED commits. An agent's own commit read as "pushed" and
    retire() deleted the tree - the work survived only in the op log, on no
    ref at all.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["jj", "git", "init", "--colocate", str(repo)], capture_output=True, check=True)
    (repo / "a.txt").write_text("hello")
    subprocess.run(["jj", "describe", "-m", "work"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["jj", "commit", "-m", "work"], cwd=repo, capture_output=True, check=True)
    # No bookmark was created: this is exactly what an agent does.
    assert STORE.unpushed_commits(repo) is True
    # A clean tree with nothing but the (empty) working-copy commit is pushed.
    empty = tmp_path / "empty"
    empty.mkdir()
    subprocess.run(["jj", "git", "init", "--colocate", str(empty)], capture_output=True, check=True)
    assert STORE.unpushed_commits(empty) is False


def test_unpushed_commits_is_none_for_a_non_repo(tmp_path):
    """No VCS at all: "cannot tell" (None), never "pushed"."""
    assert STORE.unpushed_commits(tmp_path) is None


# ── retire(): VCS cleanup runs in the SOURCE repo, and must not be silent ──


def test_retire_runs_the_vcs_cleanup_in_the_source_repo(tmp_path, monkeypatch):
    """The jj/git cleanup must run in the repo the workspace is REGISTERED in.

    Prevents (P1-5): retire() ran `jj workspace forget` with cwd=<root>/<repo>,
    which is not a jj root, so it always failed - and the returncode was
    ignored, so the directory was deleted while the repo still listed the
    workspace. The call must use `jj -R <repo> workspace forget <slug>` and
    a non-zero returncode must abort the retirement.
    """
    root = _fake_root(tmp_path, monkeypatch)
    ws = root / "repo" / "ws1"
    source = tmp_path / "source-repo"
    source.mkdir()
    rec_path = ws / ".hermes" / "session.json"
    rec = json.loads(rec_path.read_text())
    rec["repo"] = "source-repo"
    rec["vcs"] = "jj"
    rec["sessions"] = []
    rec_path.write_text(json.dumps(rec))

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(STORE.subprocess, "run", fake_run)
    monkeypatch.setattr(STORE, "find_repo", lambda name: source)
    monkeypatch.setattr(STORE, "open_prs", lambda ws: ([], True))
    monkeypatch.setattr(STORE, "dirty", lambda ws: False)
    monkeypatch.setattr(STORE, "unpushed_commits", lambda ws: False)
    monkeypatch.setattr(STORE, "_workspace_branches", lambda ws: [])

    STORE.retire("ws1", force=True)

    assert calls, "retire() did not run any VCS cleanup"
    cmd = calls[0][0]
    assert cmd[:3] == ["jj", "-R", str(source)], f"cleanup ran in the wrong place: {cmd}"
    assert cmd[-3:] == ["workspace", "forget", "ws1"], cmd
    assert not calls[0][1].get("cwd"), (
        "cleanup must not depend on a cwd that is not a repo root"
    )
    assert not ws.exists(), "the workspace directory was not removed"


def test_retire_aborts_when_the_vcs_cleanup_fails(tmp_path, monkeypatch):
    """A failed `jj workspace forget` must NOT be swallowed.

    Prevents (P1-5): the returncode was ignored, so retire() deleted the
    directory while the repo kept a stale workspace entry pointing at a
    directory that no longer exists.
    """
    root = _fake_root(tmp_path, monkeypatch)
    ws = root / "repo" / "ws1"
    source = tmp_path / "source-repo"
    source.mkdir()
    rec_path = ws / ".hermes" / "session.json"
    rec = json.loads(rec_path.read_text())
    rec["repo"] = "source-repo"
    rec["vcs"] = "jj"
    rec["sessions"] = []
    rec_path.write_text(json.dumps(rec))

    monkeypatch.setattr(
        STORE.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "no such workspace"),
    )
    monkeypatch.setattr(STORE, "find_repo", lambda name: source)
    monkeypatch.setattr(STORE, "open_prs", lambda ws: ([], True))
    monkeypatch.setattr(STORE, "dirty", lambda ws: False)
    monkeypatch.setattr(STORE, "unpushed_commits", lambda ws: False)
    monkeypatch.setattr(STORE, "_workspace_branches", lambda ws: [])

    with pytest.raises(WorkspaceError):
        STORE.retire("ws1", force=True)
    assert ws.exists(), "the directory was deleted even though the cleanup failed"


# ── create(): the slug that comes back is the slug that is on disk ──────────


def test_create_returns_the_canonical_slug_and_path(tmp_path, monkeypatch):
    """create() sanitizes the slug; the RECORD must carry the sanitized one.

    Prevents (P1-7): callers did `bind_session(workspace_path(repo, slug))`
    with their OWN (raw) slug, while create() had rewritten it. So
    ws_new("hermes-treeps", "Fix The Widget") created "fix-the-widget" and then
    tried to bind "Fix The Widget" - a path that does not exist, leaking an
    unbound workspace forever.
    """
    root = _fake_root(tmp_path, monkeypatch)
    source = tmp_path / "source-repo"
    source.mkdir()
    monkeypatch.setattr(STORE, "find_repo", lambda name: source)
    monkeypatch.setattr(STORE, "is_jj", lambda repo: False)
    monkeypatch.setattr(STORE, "run", lambda cmd, **kw: None)

    rec = STORE.create("repo", "Fix The Widget", purpose="test")
    assert rec["slug"] == "fix-the-widget", rec
    on_disk = root / "repo" / rec["slug"]
    assert on_disk.is_dir(), f"the record's slug does not match the directory: {on_disk}"
    assert (on_disk / ".hermes" / "session.json").exists()
    assert rec["path"] == str(on_disk), rec


def test_sanitize_slug_is_canonical():
    """The sanitizer is the single source of truth for what a slug becomes."""
    assert STORE.sanitize_slug("Fix The Widget") == "fix-the-widget"
    assert STORE.sanitize_slug("  Mixed CASE  ") == "mixed-case"
    assert STORE.sanitize_slug("a/b") == "a-b"
    assert STORE.sanitize_slug("..") == ""
    assert STORE.sanitize_slug("Fix The Widget") == STORE.sanitize_slug("fix-the-widget")
