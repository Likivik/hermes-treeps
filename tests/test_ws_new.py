"""ws_new: how a session gets bound, and what happens when it re-binds.

The 1:1 rule and the canonical-slug rule are the two things that decide
whether a session ends up pointing at a workspace that actually exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from plugin_loader import PLUGIN, STORE


def _ws_new(args, sid):
    return json.loads(PLUGIN._tool_new(args, session_id=sid))


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A source repo create() can pretend to work against."""
    root = tmp_path / "workspaces"
    (root / "repo").mkdir(parents=True)
    source = tmp_path / "source-repo"
    source.mkdir()
    monkeypatch.setattr(STORE, "root_dir", lambda: root)
    monkeypatch.setattr(STORE, "find_repo", lambda name: source)
    monkeypatch.setattr(STORE, "is_jj", lambda repo: False)
    monkeypatch.setattr(STORE, "run", lambda cmd, **kw: None)
    return root


def test_ws_new_binds_the_CANONICAL_slug(fake_repo, state_home):
    """The bound path must be the one on disk, not the requested spelling.

    Prevents (P1-7): create() sanitises the slug ("Fix The Widget" ->
    "fix-the-widget"), but the caller bound `workspace_path(repo, raw_slug)`.
    The session then pointed at a path that does not exist, and the created
    workspace leaked, unbound, forever.
    """
    out = _ws_new({"repo": "repo", "slug": "Fix The Widget"}, "sid-a")
    assert out["ok"] is True, out
    assert Path(out["path"]).is_dir(), f"bound path does not exist: {out['path']}"
    assert Path(out["path"]).name == "fix-the-widget", out
    state = STORE.read_state("sid-a")
    assert state is not None, "the session was never bound"
    assert state["path"] == out["path"]
    assert state["slug"] == "fix-the-widget"
    # The workspace's own record lists this session.
    rec = STORE.read_record(Path(out["path"]))
    assert rec is not None and "sid-a" in rec["sessions"], rec


def test_ws_new_rebinding_keeps_the_same_workspace(fake_repo, state_home):
    """Re-requesting the SAME workspace (different spelling) is not a re-bind."""
    _ws_new({"repo": "repo", "slug": "fix-the-widget"}, "sid-b")
    out = _ws_new({"repo": "repo", "slug": "Fix The Widget"}, "sid-b")
    assert out["ok"] is True, out
    assert out.get("needs_confirmation") is not True, out


def test_ws_new_refuses_a_different_workspace_without_rebind(fake_repo, state_home):
    """1:1 by default: a second, different workspace needs explicit consent.

    Prevents (P1-6): a session could end up bound to two workspaces at once,
    so retire() refused the first forever while it sat on disk, unowned.
    """
    _ws_new({"repo": "repo", "slug": "one"}, "sid-c")
    out = _ws_new({"repo": "repo", "slug": "two"}, "sid-c")
    assert out["ok"] is False, out
    assert out.get("needs_confirmation") is True, out
    assert "already bound" in out["error"], out


def test_ws_new_rebind_detaches_the_old_workspace(fake_repo, state_home):
    """Rebinding must release the OLD workspace, not just the session record.

    Prevents (P1-6): the old workspace kept listing the session as live, so
    retire() refused it ("bound session(s) still live") and it could never be
    cleaned up. The old tree stays on disk; only the session's claim on it is
    dropped.
    """
    first = _ws_new({"repo": "repo", "slug": "one"}, "sid-d")
    assert first["ok"] is True, first
    old_path = Path(first["path"])
    assert "sid-d" in (STORE.read_record(old_path) or {})["sessions"]

    second = _ws_new({"repo": "repo", "slug": "two", "rebind": True}, "sid-d")
    assert second["ok"] is True, second
    assert second["path"] != first["path"]

    old_rec = STORE.read_record(old_path)
    assert "sid-d" not in (old_rec or {}).get("sessions", []), (
        f"the old workspace still claims the session: {old_rec}"
    )
    new_rec = STORE.read_record(Path(second["path"]))
    assert "sid-d" in (new_rec or {}).get("sessions", []), new_rec
    # The old tree is NOT deleted by a rebind.
    assert old_path.is_dir(), "rebind must not delete the old workspace"


def test_ws_new_rebind_detaches_an_external_binding(fake_repo, state_home, tmp_path):
    """The same rule applies when the OLD binding was an external path."""
    ext = tmp_path / "external-tree"
    ext.mkdir()
    out = _ws_new({"path": str(ext)}, "sid-e")
    assert out["ok"] is True, out
    assert STORE.read_state("sid-e")["mode"] == "external"

    # A different external path needs confirmation.
    other = tmp_path / "other-tree"
    other.mkdir()
    out2 = _ws_new({"path": str(other)}, "sid-e")
    assert out2["ok"] is False and out2.get("needs_confirmation") is True, out2

    out3 = _ws_new({"path": str(other), "rebind": True}, "sid-e")
    assert out3["ok"] is True, out3
    assert STORE.read_state("sid-e")["path"] == str(other)


def test_ws_new_external_path_needs_confirmation_when_repo_bound(
    fake_repo, state_home, tmp_path
):
    """A repo-bound session asked for an external path is a re-bind too.

    Prevents (P1-6): the external branch returned before the confirmation
    check, so switching from a repo workspace to an external tree silently
    dropped the repo binding (and left it claimed).
    """
    _ws_new({"repo": "repo", "slug": "one"}, "sid-f")
    ext = tmp_path / "external-tree"
    ext.mkdir()
    out = _ws_new({"path": str(ext)}, "sid-f")
    assert out["ok"] is False and out.get("needs_confirmation") is True, out
    # The original binding is untouched.
    assert STORE.read_state("sid-f")["mode"] == "repo"


def test_ws_new_default_slug_follows_the_session_id(fake_repo, state_home):
    """The default workspace name comes from the session id, not the title.

    WHY: titles are renamed constantly and there is no rename hook to keep a
    path in sync, so a title-derived path rots the moment the title changes.
    """
    out = _ws_new({"repo": "repo"}, "sid-abcdef123456")
    assert out["ok"] is True, out
    assert Path(out["path"]).name == "repo-123456", out


# ── not on disk, but on a remote: the clone must be an approved step ──────────


@pytest.fixture
def empty_workspace_root(tmp_path, monkeypatch):
    """A workspace root with nothing in it, and a repo that is not on disk."""
    root = tmp_path / "workspaces"
    root.mkdir()
    monkeypatch.setattr(STORE, "root_dir", lambda: root)

    def _missing(name):
        raise STORE.WorkspaceError(f"no repo {name!r} under /Git - existing: ")

    monkeypatch.setattr(STORE, "find_repo", _missing)
    monkeypatch.setattr(STORE, "clone_repo", lambda name, url: pytest.fail(
        "nothing may be cloned without the user's approval"
    ))
    return root


def _fake_remote(monkeypatch, entries):
    """Point the remote catalogue at a fixed list. No network, ever."""
    monkeypatch.setattr(STORE, "remote_repos", lambda limit_per_source=5: list(entries))


def test_ws_new_off_disk_remote_repo_asks_before_cloning(
    empty_workspace_root, state_home, monkeypatch
):
    """A repo that lives on a remote is NOT "does not exist" - it needs a clone.

    Prevents (P2-3): the tools only ever mentioned local /Git roots, so an
    agent whose repo lived on Forgejo cloned it on its own initiative, twice,
    after two failed attempts. The remote catalogue now names the repo and
    asks; nothing is written to disk until the user says yes.
    """
    _fake_remote(monkeypatch, [{
        "name": "achiyon", "owner": "likivik",
        "url": "https://forgejo.example/owner/achiyon.git", "source": "forgejo",
    }])
    out = _ws_new({"repo": "achiyon", "slug": "work"}, "sid-remote-1")
    assert out["ok"] is False, out
    assert out.get("needs_confirmation") is True, out
    assert out["remote"]["url"] == "https://forgejo.example/owner/achiyon.git", out
    assert out["remote"]["source"] == "forgejo", out
    assert "achiyon" in out["error"], out
    assert "create_repo" in out["error"], out
    # Nothing was created: the workspace root is untouched.
    assert list(empty_workspace_root.iterdir()) == [], list(empty_workspace_root.iterdir())


def test_ws_new_with_create_repo_clones_after_approval(
    empty_workspace_root, state_home, monkeypatch
):
    """The SECOND call (create_repo=true) is the one that clones.

    Prevents (P2-3): the approval must be actionable, not just a message.
    The clone is the only thing that writes to the user's disk, and it happens
    exactly once, on the call the user approved.
    """
    _fake_remote(monkeypatch, [{
        "name": "achiyon", "owner": "likivik",
        "url": "https://forgejo.example/owner/achiyon.git", "source": "forgejo",
    }])
    cloned = []
    source = empty_workspace_root.parent / "cloned-achiyon"
    state = {"cloned": False}

    def _fake_clone(name, url):
        cloned.append((name, url))
        source.mkdir(exist_ok=True)
        state["cloned"] = True
        return source

    def _find(name):
        if not state["cloned"]:
            raise STORE.WorkspaceError(f"no repo {name!r} under /Git - existing: ")
        return source

    monkeypatch.setattr(STORE, "clone_repo", _fake_clone)
    monkeypatch.setattr(STORE, "find_repo", _find)
    monkeypatch.setattr(STORE, "is_jj", lambda repo: False)
    monkeypatch.setattr(STORE, "run", lambda cmd, **kw: None)

    out = _ws_new({"repo": "achiyon", "slug": "work", "create_repo": True}, "sid-remote-2")
    assert out["ok"] is True, out
    assert cloned == [("achiyon", "https://forgejo.example/owner/achiyon.git")], cloned


def test_ws_new_unknown_repo_names_the_remotes_it_did_see(
    empty_workspace_root, state_home, monkeypatch
):
    """A repo that is nowhere gets told WHICH repos do exist.

    Prevents (P2-3): the old error only listed local directories, so an agent
    kept guessing repo names against a catalogue it had never been shown.
    """
    _fake_remote(monkeypatch, [
        {"name": "hermes-treeps", "owner": "likivik", "url": "https://gh/o/hermes-treeps", "source": "github"},
        {"name": "spectacle", "owner": "likivik", "url": "https://gh/o/spectacle", "source": "github"},
    ])
    out = _ws_new({"repo": "achiyon", "slug": "work"}, "sid-unknown-1")
    assert out["ok"] is False, out
    assert out.get("needs_confirmation") is not True, out
    assert "hermes-treeps" in out["error"], out
    assert "spectacle" in out["error"], out
    assert list(empty_workspace_root.iterdir()) == [], list(empty_workspace_root.iterdir())


def test_ws_new_survives_a_dead_remote_catalogue(
    empty_workspace_root, state_home, monkeypatch
):
    """A failing catalogue must not turn into a traceback, and must not clone.

    Prevents (P2-3): `remote_repos` shells out to gh and to Forgejo over
    HTTP; a dead network used to be able to break the tool answer itself.
    """

    def _boom(limit_per_source=5):
        raise RuntimeError("network is down")

    monkeypatch.setattr(STORE, "remote_repos", _boom)
    out = _ws_new({"repo": "achiyon", "slug": "work"}, "sid-dead-1")
    assert out["ok"] is False, out
    assert "Traceback" not in out["error"], out
    assert "catalogue" in out["error"].lower(), out
    assert list(empty_workspace_root.iterdir()) == [], list(empty_workspace_root.iterdir())


def test_ws_doctor_says_where_a_repo_could_come_from(state_home, monkeypatch):
    """ws_doctor must answer "where could this come from?", not just "no".

    Prevents (P2-3): with only the local candidate list, a repo that lives on
    GitHub/Forgejo was indistinguishable from one that does not exist.
    """
    _fake_remote(monkeypatch, [{
        "name": "achiyon", "owner": "likivik", "url": "https://gh/o/achiyon", "source": "github",
    }])
    out = json.loads(PLUGIN._tool_doctor({}, session_id="sid-doctor-1"))
    assert "local_candidates" in out, out
    assert "remote_candidates" in out, out
    assert out["repo_roots"], out
    # One canonical name for the local list: "candidate_repos" was an exact
    # duplicate of "local_candidates", so a consumer could read either and
    # never know which one was canonical.
    assert "candidate_repos" not in out, out
    assert [c["name"] for c in out["remote_candidates"]] == ["achiyon"], out
    assert out["remote_candidates"][0]["source"] == "github", out
    assert out["remote_error"] == "", out


def test_ws_doctor_reports_a_dead_catalogue(state_home, monkeypatch):
    """A failed lookup is reported as an error, not as "no remote repos"."""

    def _boom(limit_per_source=5):
        raise RuntimeError("network is down")

    monkeypatch.setattr(STORE, "remote_repos", _boom)
    out = json.loads(PLUGIN._tool_doctor({}, session_id="sid-doctor-2"))
    assert out["remote_candidates"] == [], out
    assert out["remote_error"], out
