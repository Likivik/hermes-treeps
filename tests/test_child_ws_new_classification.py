"""A delegated child may not mint a USER-class workspace.

The live failure this defends against (2026-10-25): the T4 builder child
called ``ws_new(repo='hermes-treeps', purpose='feat: picker UI - flat
tagged list, two-step clone, background bind')`` - it passed its own goal
text as the purpose. The handler stamped that goal text into the record, so
the workspace looked like a USER workspace: immediate reclaim refused it,
``ws gc --children`` skipped it, and ``gc_apply`` never retired it while the
session binding existed. The tree and its record leaked until a human ran
``jj workspace forget`` and ``rm`` by hand.

Two layers stop that from happening again:

1. ``ws_new`` FORCES the child shape when the calling session is a delegated
   child: the caller's purpose AND slug are ignored, and the child shape
   (``child:<parent session id>`` / ``<parent slug>-child-<sid6>``) is
   derived from the same sources the gate uses.
2. The gate SELF-HEALS: a record that a child created before this fix gets
   restamped on the child's next gated call, so it rejoins the reclaim
   system instead of leaking forever.

WHY the child shape is forced rather than merely checked: an agent-run child
cannot be trusted to label its own workspaces. A record without the child
prefix is unreclaimable BY CONSTRUCTION - every reclaim path keys on the
prefix - so the safe default is "child", not "user".
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from plugin_loader import PLUGIN, STORE

CHILD_PREFIX = STORE.CHILD_PURPOSE_PREFIX

PARENT = "parent-session-1"
CHILD = "child-session-abcdef"


def _fake_child_context(monkeypatch, value: bool = True):
    """Install a fake ``agent.delegation_context`` module."""
    mod = types.ModuleType("agent.delegation_context")
    mod.is_delegated_child_context = lambda: value
    monkeypatch.setitem(sys.modules, "agent.delegation_context", mod)


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A source repo create() can pretend to work against."""
    repo_dir = tmp_path / "repos"
    (repo_dir / "repo").mkdir(parents=True)
    source = tmp_path / "source-repo"
    source.mkdir()
    monkeypatch.setattr(STORE, "repo_dir", lambda: repo_dir)
    monkeypatch.setattr(STORE, "find_repo", lambda name: source)
    monkeypatch.setattr(STORE, "is_jj", lambda repo: False)
    monkeypatch.setattr(STORE, "run", lambda cmd, **kw: None)
    return STORE.workspaces_base()


def _ws_new(args, sid):
    return json.loads(PLUGIN._tool_new(args, session_id=sid))


# ── (a) a child calling ws_new gets the forced child shape ──────────────────


def test_a_child_calling_ws_new_cannot_mint_a_user_workspace(fake_repo, state_home, monkeypatch):
    """The child's own purpose and slug are IGNORED and replaced.

    Prevents: the live leak - a child that labels its workspace with goal
    text produces a USER-class record that no reclaim path ever touches.
    """
    _fake_child_context(monkeypatch)
    STORE.db.write_child_run(CHILD, PARENT)
    STORE.write_state(PARENT, {"mode": "repo", "repo": "repo", "slug": "parent-slug", "path": "/tmp/parent"})

    out = _ws_new(
        {
            "repo": "repo",
            "slug": "hermes-treeps-395325",
            "purpose": "feat: picker UI - flat tagged list, two-step clone, background bind",
        },
        CHILD,
    )
    assert out["ok"] is True, out

    rec = STORE.read_record(Path(out["path"]))
    assert rec is not None, out
    # The purpose is the child marker, NOT the goal text the child passed.
    assert rec["purpose"] == f"{CHILD_PREFIX}{PARENT}", rec
    assert not rec["purpose"].startswith("feat:"), rec

    # The slug is child-shaped, not the caller's slug.
    assert rec["slug"] != "hermes-treeps-395325", rec
    assert rec["slug"].startswith("parent-slug-child-"), rec
    assert rec["slug"].endswith(CHILD[-6:]), rec


def test_a_child_ws_new_is_classified_as_a_child_workspace(fake_repo, state_home, monkeypatch):
    """The forced record must classify as a child, so reclaim can see it.

    Prevents: the shape is right in the record but the classification path
    reads something else, and the workspace is still invisible to gc.
    """
    _fake_child_context(monkeypatch)
    STORE.db.write_child_run(CHILD, PARENT)
    STORE.write_state(PARENT, {"mode": "repo", "repo": "repo", "slug": "parent-slug", "path": "/tmp/parent"})

    out = _ws_new({"repo": "repo", "slug": "whatever", "purpose": "goal text"}, CHILD)
    assert out["ok"] is True, out
    rec = STORE.read_record(Path(out["path"]))
    assert PLUGIN.childspace.is_child_workspace(rec), rec
    # And it is visible to the child-reclaim listing.
    child_records = STORE._child_workspace_records()
    assert any(str(r.get("path")) == str(out["path"]) for r in child_records), child_records


# ── (b) a NON-child session keeps its own purpose and slug ─────────────────


def test_a_user_session_calling_ws_new_keeps_its_own_purpose_and_slug(fake_repo, state_home, monkeypatch):
    """The forcing applies to delegated children ONLY.

    Prevents: the fix over-reaching - a normal session's purpose and slug
    are the user's business, and stamping them with the child prefix would
    make the user's workspace reclaimable on age alone.
    """
    _fake_child_context(monkeypatch, value=False)

    out = _ws_new(
        {"repo": "repo", "slug": "my-own-slug", "purpose": "feat: my own goal text"},
        "sid-user-1",
    )
    assert out["ok"] is True, out
    rec = STORE.read_record(Path(out["path"]))
    assert rec["purpose"] == "feat: my own goal text", rec
    assert rec["slug"] == "my-own-slug", rec
    assert not PLUGIN.childspace.is_child_workspace(rec), rec


def test_a_user_session_with_no_purpose_stays_unclassified(fake_repo, state_home, monkeypatch):
    """No purpose from the user means no child marker either."""
    _fake_child_context(monkeypatch, value=False)

    out = _ws_new({"repo": "repo", "slug": "plain"}, "sid-user-2")
    assert out["ok"] is True, out
    rec = STORE.read_record(Path(out["path"]))
    assert not rec["purpose"].startswith(CHILD_PREFIX), rec


# ── (c) + (d) the gate self-heals an already-leaked record ─────────────────


def _leaked_child_record(fake_repo, state_home, monkeypatch):
    """A workspace whose purpose is goal text, bound to a child session.

    This is the shape the live bug produced: a child that called ws_new
    BEFORE this fix, so the record carries the child's goal text instead of
    the child marker. The child session IS in the record's session list -
    that binding is what made gc_apply skip the workspace forever.

    It is built directly (not through ws_new) on purpose: ws_new now forces
    the child shape, so the pre-fix leak can no longer be reproduced through
    the tool - only through history.
    """
    out = _ws_new({"repo": "repo", "slug": "leaked-slug", "purpose": "feat: picker UI goal text"}, "sid-pre-fix")
    assert out["ok"] is True, out
    path = Path(out["path"])
    rec = STORE.read_record(path)
    # The leak shape: the goal text is in the record.
    assert rec["purpose"] == "feat: picker UI goal text", rec
    # Now bind the CHILD session to it, exactly as the pre-fix ws_new did.
    STORE.bind_session(path, CHILD)
    rec = STORE.read_record(path)
    assert CHILD in rec["sessions"], rec
    return path


def test_the_gate_restamps_a_leaked_child_record(fake_repo, state_home, monkeypatch):
    """A gated child call restamps its own record to the child marker.

    Prevents: records that leaked BEFORE this fix stay unreclaimable
    forever - the exact leak we had to clean up by hand today.
    """
    path = _leaked_child_record(fake_repo, state_home, monkeypatch)

    # The child session is bound to the leaked workspace.
    STORE.write_state(
        CHILD,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": "leaked-slug",
            "path": str(path),
            "inherited_from": PARENT,
            "parent_session_id": PARENT,
        },
    )
    STORE.db.write_child_run(CHILD, PARENT)
    _fake_child_context(monkeypatch)

    res = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id=CHILD)
    assert res is None, res

    rec = STORE.read_record(path)
    assert rec["purpose"] == f"{CHILD_PREFIX}{PARENT}", rec
    assert PLUGIN.childspace.is_child_workspace(rec), rec
    # The restamp must not cost the workspace its session list: a record
    # that lost its binding would be reclaimable while still in use.
    assert CHILD in rec["sessions"], rec
    # And it must still be bound to this session.
    bound = STORE.resolve(CHILD)
    assert bound is not None and bound["path"] == str(path), bound


def test_the_self_heal_is_idempotent(fake_repo, state_home, monkeypatch):
    """A second gated child call changes nothing.

    Prevents: a self-heal that rewrites the record on every gated call -
    an UPDATE storm on the hot path, and a record that keeps changing.
    """
    path = _leaked_child_record(fake_repo, state_home, monkeypatch)
    STORE.write_state(
        CHILD,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": "leaked-slug",
            "path": str(path),
            "inherited_from": PARENT,
            "parent_session_id": PARENT,
        },
    )
    STORE.db.write_child_run(CHILD, PARENT)
    _fake_child_context(monkeypatch)

    assert PLUGIN._on_pre_tool_call(tool_name="terminal", session_id=CHILD) is None
    first = STORE.read_record(path)
    assert first["purpose"] == f"{CHILD_PREFIX}{PARENT}", first

    assert PLUGIN._on_pre_tool_call(tool_name="terminal", session_id=CHILD) is None
    second = STORE.read_record(path)
    assert second["purpose"] == f"{CHILD_PREFIX}{PARENT}", second
    assert second == first, "the second gated call rewrote the record"


def test_a_user_record_is_never_restamped_by_the_gate(fake_repo, state_home, monkeypatch):
    """The self-heal must not touch a USER workspace.

    Prevents: the gate stamping the child marker onto a user's workspace
    because the session happens to be a child - that would make the user's
    workspace reclaimable on age alone.

    The child is bound to the user's workspace but does NOT OWN it: it is
    not in the record's session list. That is the shape of an inherited
    workspace, and it must keep the parent's classification.
    """
    out = _ws_new({"repo": "repo", "slug": "user-ws", "purpose": "my own purpose"}, "sid-user-3")
    path = Path(out["path"])
    # The child's binding points at the user's workspace, but the user's
    # session owns the record.
    STORE.write_state(
        CHILD,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": "user-ws",
            "path": str(path),
            "inherited_from": PARENT,
            "parent_session_id": PARENT,
        },
    )
    STORE.db.write_child_run(CHILD, PARENT)
    _fake_child_context(monkeypatch)

    assert PLUGIN._on_pre_tool_call(tool_name="terminal", session_id=CHILD) is None
    rec = STORE.read_record(path)
    assert rec["purpose"] == "my own purpose", rec
