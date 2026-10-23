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
