"""capacity.py: the live-child ledger the delegation gate counts against.

WHAT this defends: a delegate_task call that asks for more children than the
host can run. The gate needs ONE number - how many children are live RIGHT
NOW - and this module is where it comes from. Every helper here must survive
a foreign or unreadable ledger silently: a capacity feature that crashes the
delegation it was counting is worse than no capacity feature.
"""

from __future__ import annotations

import sqlite3

import pytest
from plugin_loader import CAPACITY, STORE

capacity = CAPACITY
db = STORE.db


# ── the ledger: open, close, count ────────────────────────────────────────────


def test_record_start_opens_a_row_and_record_stop_closes_it():
    capacity.record_start("sub-1", child_session_id="sess-1", model="m", provider="p")
    rows = capacity.open_rows("profile")
    assert [r["subagent_id"] for r in rows] == ["sub-1"], rows
    assert rows[0]["gateway_pid"] > 0, rows
    assert rows[0]["ended_at"] == "", rows
    assert rows[0]["model"] == "m" and rows[0]["provider"] == "p", rows

    capacity.record_stop("sub-1")
    assert capacity.open_rows("profile") == []


def test_a_redispatched_child_does_not_count_twice():
    capacity.record_start("sub-1")
    capacity.record_start("sub-1")
    assert capacity.count_live(scope="profile") == 1


def test_count_live_with_a_match_predicate():
    capacity.record_start("a", provider="custom:neuralwatt")
    capacity.record_start("b", provider="custom:minimax")
    assert capacity.count_live(scope="profile") == 2
    assert capacity.count_live(scope="profile", match=capacity.is_minimax) == 0
    assert (
        capacity.count_live(
            scope="profile", match=lambda row: capacity.is_minimax(row["model"], row["provider"])
        )
        == 1
    )


# ── pid liveness ─────────────────────────────────────────────────────────────


def test_a_dead_pid_row_is_not_live(monkeypatch):
    capacity.record_start("sub-1")

    def dead_pid(*_a, **_k):
        raise ProcessLookupError()

    monkeypatch.setattr(capacity.os, "kill", dead_pid)
    assert capacity.count_live(scope="profile") == 0


def test_an_eperm_pid_is_still_alive(monkeypatch):
    capacity.record_start("sub-1")

    def no_perm(*_a, **_k):
        raise PermissionError()

    monkeypatch.setattr(capacity.os, "kill", no_perm)
    assert capacity.count_live(scope="profile") == 1


# ── scopes ───────────────────────────────────────────────────────────────────


def _foreign_ledger(home, rows):
    """Build a profile-home-shaped tree with a child_capacity ledger in it."""
    d = home / "plugin-data" / "hermes-treeps"
    d.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(d / "data.db")
    conn.executescript(db.V5_TABLES)
    for i, (sub, provider, pid) in enumerate(rows):
        conn.execute(
            "INSERT INTO child_capacity(subagent_id, provider, gateway_pid, started_at)"
            " VALUES(?,?,?,?)",
            (sub, provider, pid, f"2026-01-01T00:00:{i:02d}Z"),
        )
    conn.commit()
    conn.close()
    return d / "data.db"


def test_scope_profiles_reads_the_listed_profile_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "hermes_home", lambda: tmp_path / "hermes")
    _foreign_ledger(tmp_path / "hermes" / "profiles" / "alice", [("x-1", "p1", 0)])
    _foreign_ledger(tmp_path / "hermes" / "profiles" / "bob", [("x-2", "p2", 0)])

    rows = capacity.open_rows("profiles", profiles=["alice", "bob"])
    assert sorted(r["subagent_id"] for r in rows) == ["x-1", "x-2"], rows

    # An absolute path works as a profile home too.
    rows = capacity.open_rows(
        "profiles", profiles=[str(tmp_path / "hermes" / "profiles" / "alice")]
    )
    assert [r["subagent_id"] for r in rows] == ["x-1"], rows


def test_scope_global_reads_every_profile_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setattr(db, "hermes_home", lambda: home)
    _foreign_ledger(home, [("own-1", "p", 0)])
    _foreign_ledger(home / "profiles" / "alice", [("x-1", "p", 0)])
    (home / "profiles" / "empty").mkdir(parents=True, exist_ok=True)
    # A profile home whose ledger does not even exist is skipped silently.
    rows = capacity.open_rows("global")
    assert sorted(r["subagent_id"] for r in rows) == ["own-1", "x-1"], rows


def test_an_unreadable_ledger_is_skipped_silently(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setattr(db, "hermes_home", lambda: home)
    bad = home / "profiles" / "bad" / "plugin-data" / "hermes-treeps" / "data.db"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not a database at all")
    assert capacity.open_rows("global") == []


def test_ledger_dir_points_at_one_explicit_ledger(tmp_path):
    path = _foreign_ledger(tmp_path / "explicit", [("e-1", "p", 0)])
    rows = capacity.open_rows("profile", ledger_dir=str(path.parent))
    assert [r["subagent_id"] for r in rows] == ["e-1"], rows


# ── classification and overflow ──────────────────────────────────────────────


def test_is_minimax():
    assert capacity.is_minimax("MiniMax-M3.1-Flash", "custom:neuralwatt")
    assert capacity.is_minimax("glm-5.3", "custom:minimax-custom")
    assert not capacity.is_minimax("glm-5.3", "custom:neuralwatt")
    assert not capacity.is_minimax("", "")


def test_classify_rows_counts_per_provider():
    rows = [{"provider": "a"}, {"provider": "a"}, {"provider": "b"}, {}]
    assert capacity.classify_rows(rows) == {"a": 2, "b": 1, "": 1}


def test_should_overflow_boundaries():
    # 4 MiniMax slots, none live: the first 4 children fit, the 5th overflows.
    assert not capacity.should_overflow(0, 0, 4)
    assert not capacity.should_overflow(3, 0, 4)
    assert capacity.should_overflow(4, 0, 4)
    # 2 already live: only 2 slots left, so index 2 overflows.
    assert not capacity.should_overflow(1, 2, 4)
    assert capacity.should_overflow(2, 2, 4)


# ── the verdict ───────────────────────────────────────────────────────────────


def test_verdict_allows_under_the_cap_and_names_the_counts():
    allowed, reason = capacity.verdict(requested_children=3, live=2, max_children=8)
    assert allowed, reason
    assert "2" in reason and "3" in reason and "8" in reason, reason


def test_verdict_refuses_over_the_cap_and_names_the_counts():
    allowed, reason = capacity.verdict(requested_children=5, live=6, max_children=8)
    assert not allowed
    assert "5" in reason and "6" in reason and "8" in reason, reason
    assert "2" in reason, reason  # the free slots


def test_verdict_at_exactly_the_cap_is_allowed():
    allowed, _ = capacity.verdict(requested_children=4, live=4, max_children=8)
    assert allowed


# ── the hook wiring (__init__.py) ────────────────────────────────────────────


def test_the_subagent_hooks_open_and_close_the_ledger_row():
    from plugin_loader import PLUGIN

    PLUGIN._on_subagent_start(
        parent_session_id="p-1",
        child_session_id="c-sess-1",
        child_subagent_id="c-sub-1",
        child_role="worker",
        child_goal="g",
    )
    rows = capacity.open_rows("profile")
    assert [r["subagent_id"] for r in rows] == ["c-sub-1"], rows

    PLUGIN._on_subagent_stop(
        parent_session_id="p-1", child_session_id="c-sess-1", child_subagent_id="c-sub-1"
    )
    assert capacity.open_rows("profile") == []


def test_a_stop_without_a_parent_id_still_frees_the_slot():
    """The ledger close runs before the report's parent-id early return."""
    from plugin_loader import PLUGIN

    capacity.record_start("c-sub-2")
    PLUGIN._on_subagent_stop(child_session_id="c-sess-2", child_subagent_id="c-sub-2")
    assert capacity.open_rows("profile") == []


def test_a_broken_capacity_row_never_breaks_the_delegation(monkeypatch):
    from plugin_loader import PLUGIN

    def boom(*_a, **_k):
        raise RuntimeError("ledger exploded")

    monkeypatch.setattr(capacity, "record_start", boom)
    # The child-report hook still ran (its row is there): the delegation is
    # intact even though the capacity half failed.
    PLUGIN._on_subagent_start(
        parent_session_id="p-9", child_session_id="c-sess-9", child_subagent_id="c-sub-9"
    )
    assert STORE.db.read_child_run("c-sess-9") is not None


# ── config_from_settings ─────────────────────────────────────────────────────


def test_config_from_settings_maps_the_capacity_keys():
    cfg = capacity.config_from_settings(
        {
            "capacity_enabled": True,
            "capacity_max_children": 5,
            "capacity_overflow_after": 2,
            "capacity_tracking_scope": "profiles",
            "capacity_tracking_profiles": ["alice"],
            "capacity_ledger_dir": "/tmp/led",
        }
    )
    assert cfg == {
        "enabled": True,
        "max_children": 5,
        "overflow_after": 2,
        "scope": "profiles",
        "profiles": ["alice"],
        "ledger_dir": "/tmp/led",
    }


@pytest.mark.parametrize("missing", ["capacity_tracking_profiles", "capacity_ledger_dir"])
def test_config_from_settings_tolerates_missing_keys(missing):
    s = {
        "capacity_enabled": True,
        "capacity_max_children": 5,
        "capacity_overflow_after": 2,
        "capacity_tracking_scope": "profile",
        "capacity_tracking_profiles": [],
        "capacity_ledger_dir": "",
    }
    del s[missing]
    cfg = capacity.config_from_settings(s)
    assert cfg[missing.removeprefix("capacity_").replace("tracking_", "")] in ([], "")
