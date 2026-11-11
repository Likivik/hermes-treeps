"""The capacity half of the delegate_task gate: inline at the cap.

WHAT this defends: a delegate_task batch that would push the live child count
past ``capacity_max_children``. The gate must refuse it with a message that
names the counts, lists the refused goals, and tells the agent to do them
inline - not with a generic denial the agent will just retry.
"""

from __future__ import annotations

import importlib

import pytest
from plugin_loader import SETTINGS, STORE

GATE = importlib.import_module("hermes_treeps.gate")
CAPACITY = importlib.import_module("hermes_treeps.capacity")

CAP_OVERRIDES = {
    "capacity_enabled": True,
    "capacity_max_children": 4,
    "capacity_overflow_after": 2,
    "capacity_tracking_scope": "profile",
    "capacity_tracking_profiles": [],
    "capacity_ledger_dir": "",
}

TASKS = [{"goal": f"goal {i}"} for i in range(1, 4)]


@pytest.fixture
def counted(monkeypatch, state_home):
    """A chatting session, a pinned live count, pinned capacity settings.

    The session is bound as chatting so the ONLY thing that can block the
    call is the capacity check under test (the workspace-resolution half of
    the gate is covered by tests/test_gate.py).
    """
    state = {"live": 0}

    def fake_count_live(scope="profile", profiles=None, ledger_dir="", match=None):
        return state["live"]

    def fake_settings():
        base = SETTINGS.settings()
        base.update(CAP_OVERRIDES)
        return base

    STORE.mark_chatting("sess-cap")
    monkeypatch.setattr(CAPACITY, "count_live", fake_count_live)
    monkeypatch.setattr(STORE, "settings", fake_settings)
    return state


def _call(tasks, live=None, settings=None, **kw):
    return GATE._on_pre_tool_call(
        "delegate_task", "sess-cap", args={"tasks": tasks}, **kw
    )


def test_over_capacity_is_blocked_with_the_counts_and_goals(counted):
    counted["live"] = 3
    result = _call(TASKS)  # 3 live + 3 requested > cap 4
    assert result is not None, "the gate allowed an over-cap batch"
    assert result["action"] == "block", result
    msg = result["message"]
    # (a) the counts
    assert "3" in msg and "4" in msg, msg
    # (b) the refused goals
    assert "goal 1" in msg and "goal 3" in msg, msg
    # (c) the inline instruction and the free number, (d) no retry
    assert "INLINE" in msg, msg
    assert "workspace" in msg, msg
    assert "at most 1" in msg, msg
    assert "Do not retry" in msg, msg


def test_at_exactly_the_cap_is_allowed(counted):
    counted["live"] = 1
    assert _call(TASKS) is None  # 1 live + 3 requested == cap 4


def test_under_the_cap_is_allowed(counted):
    counted["live"] = 0
    assert _call(TASKS[:2]) is None


def test_capacity_disabled_never_blocks(counted, monkeypatch):
    overrides = dict(CAP_OVERRIDES, capacity_enabled=False)

    def fake_settings():
        base = SETTINGS.settings()
        base.update(overrides)
        return base

    monkeypatch.setattr(STORE, "settings", fake_settings)
    counted["live"] = 99
    assert _call(TASKS) is None


def test_an_unreadable_request_is_allowed(counted):
    counted["live"] = 4
    # args that cannot be counted (no tasks, no goal): allow, and the normal
    # gate path takes over for a chatting session (which allows).
    assert GATE._on_pre_tool_call("delegate_task", "sess-cap", args={}) is None
    assert GATE._on_pre_tool_call("delegate_task", "sess-cap") is None
    assert (
        GATE._on_pre_tool_call("delegate_task", "sess-cap", args={"tasks": "not-a-list"})
        is None
    )


def test_the_legacy_single_goal_shape_counts_as_one(counted):
    counted["live"] = 4  # cap 4: any new child is over
    result = GATE._on_pre_tool_call("delegate_task", "sess-cap", args={"goal": "the one goal"})
    assert result is not None and result["action"] == "block", result
    assert "the one goal" in result["message"], result


def test_a_broken_capacity_feature_allows_the_call(counted, monkeypatch):
    def boom(**_kw):
        raise RuntimeError("ledger exploded")

    monkeypatch.setattr(CAPACITY, "count_live", boom)
    counted["live"] = 0
    assert _call(TASKS) is None
