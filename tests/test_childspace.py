"""Tests for childspace: a delegated child gets its OWN workspace.

No network, no real jj repo: ``store`` is monkeypatched, and every write goes
to the tmp database the ``isolated_db`` fixture points at.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from plugin_loader import STORE

# The plugin package is registered under "hermes_treeps" by the loader, so the
# new module is imported as a submodule of it (relative imports work).
CHILDSPACE = importlib.import_module("hermes_treeps.childspace")


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A source repo the child workspace can be created from."""
    root = tmp_path / "workspaces"
    (root / "repo").mkdir(parents=True)
    source = tmp_path / "source-repo"
    (source / ".jj").mkdir(parents=True)
    monkeypatch.setattr(STORE, "root_dir", lambda: root)
    monkeypatch.setattr(STORE, "find_repo", lambda name: source)
    monkeypatch.setattr(STORE, "is_jj", lambda repo: True)
    monkeypatch.setattr(STORE, "run", lambda cmd, **kw: None)
    return root


PARENT_STATE = {
    "mode": "repo",
    "repo": "repo",
    "slug": "parent-tree",
    "path": "/tmp/parent-tree",
    "session_id": "parent-session-1",
}


def test_child_workspace_is_created_once_and_is_idempotent(fake_repo):
    """The first call creates it; the second returns the SAME one."""
    out = CHILDSPACE.ensure_child_workspace("child-session-abcdef", PARENT_STATE)
    assert out["ok"] is True, out
    assert out["repo"] == "repo", out
    assert out["slug"] == "parent-tree-child-abcdef", out
    assert Path(out["path"]).is_dir(), out

    again = CHILDSPACE.ensure_child_workspace("child-session-abcdef", PARENT_STATE)
    assert again["ok"] is True, again
    assert again["path"] == out["path"], (again, out)
    assert again["slug"] == out["slug"], (again, out)
    # Exactly one workspace was created.
    records = STORE.list_workspaces()
    child_records = [r for r in records if str((r.get("record") or {}).get("purpose") or "").startswith("child:")]
    assert len(child_records) == 1, records


def test_external_parent_creates_nothing(fake_repo):
    """An external tree has no repo to branch from: fall back to inheritance."""
    parent = dict(PARENT_STATE, mode="external")
    out = CHILDSPACE.ensure_child_workspace("child-session-abcdef", parent)
    assert out["ok"] is False, out
    assert "external" in out["reason"], out
    assert STORE.list_workspaces() == [], STORE.list_workspaces()


def test_chatting_parent_creates_nothing(fake_repo):
    """A chatting parent has no repo to branch from either."""
    parent = dict(PARENT_STATE, mode="chatting")
    out = CHILDSPACE.ensure_child_workspace("child-session-abcdef", parent)
    assert out["ok"] is False, out
    assert "chatting" in out["reason"], out
    assert STORE.list_workspaces() == [], STORE.list_workspaces()


def test_child_session_already_chatting_is_refused(fake_repo):
    """Guard: a session marked chatting must not get a workspace."""
    STORE.write_state("child-session-abcdef", {"mode": "chatting"})
    out = CHILDSPACE.ensure_child_workspace("child-session-abcdef", PARENT_STATE)
    assert out["ok"] is False, out
    assert "chatting" in out["reason"], out
    assert STORE.list_workspaces() == [], STORE.list_workspaces()


def test_a_raising_store_never_raises_and_writes_no_state(fake_repo, monkeypatch):
    """Every failure path returns ok=False with the message, and no state."""
    calls = []

    def boom(*a, **kw):
        calls.append(a)
        raise RuntimeError("jj exploded")

    monkeypatch.setattr(STORE, "create", boom)
    out = CHILDSPACE.ensure_child_workspace("child-session-abcdef", PARENT_STATE)
    assert out["ok"] is False, out
    assert "RuntimeError" in out["reason"], out
    assert "jj exploded" in out["reason"], out
    assert calls, "store.create was not even called"
    # No partial state was written.
    assert STORE.read_state("child-session-abcdef") is None
    assert STORE.list_workspaces() == [], STORE.list_workspaces()


def test_created_state_carries_inherited_from_and_the_purpose_marker(fake_repo):
    """The child's binding names its parent, and the workspace is marked."""
    out = CHILDSPACE.ensure_child_workspace("child-session-abcdef", PARENT_STATE)
    assert out["ok"] is True, out

    state = STORE.read_state("child-session-abcdef")
    assert state is not None, "the child session was never bound"
    assert state["mode"] == "repo", state
    assert state["repo"] == "repo", state
    assert state["slug"] == "parent-tree-child-abcdef", state
    assert state["inherited_from"] == "parent-session-1", state

    rec = STORE.read_record(Path(out["path"]))
    assert rec is not None, out
    assert rec["purpose"] == "child:parent-session-1", rec
    assert CHILDSPACE.is_child_workspace(rec) is True, rec
    assert "child-session-abcdef" in rec["sessions"], rec


def test_is_child_workspace_true_only_for_the_marker():
    assert CHILDSPACE.is_child_workspace({"purpose": "child:abc123"}) is True
    assert CHILDSPACE.is_child_workspace({"purpose": "child:"}) is True
    assert CHILDSPACE.is_child_workspace({"purpose": ""}) is False
    assert CHILDSPACE.is_child_workspace({"purpose": "other"}) is False
    assert CHILDSPACE.is_child_workspace({}) is False
