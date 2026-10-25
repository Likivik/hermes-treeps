"""The dual-marker child liveness rule.

A graceful child stop stamps ``session_ended_at`` and is reclaimed at once.
A hard-dead child (the gateway restarted mid-child) never gets that stamp, so
the plugin-side heartbeat is the second, independent marker. The hard-death
backstop requires BOTH a stale heartbeat AND an old workspace.
"""

from __future__ import annotations

import time
from pathlib import Path

from plugin_loader import PLUGIN, STORE

CHILD_PREFIX = STORE.CHILD_PURPOSE_PREFIX


def _record(ws: Path, *, created_at: str = "2020-01-01T00:00:00", sessions=("child-1",)):
    STORE.write_record(
        ws,
        {
            "repo": "repo",
            "slug": ws.name,
            "path": str(ws),
            "vcs": "jj",
            "purpose": CHILD_PREFIX + "parent-1",
            "sessions": list(sessions),
            "created_at": created_at,
        },
    )


def _live_binding(ws: Path, sid: str = "child-1"):
    STORE.db.write_binding(
        sid,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": ws.name,
            "path": str(ws),
            "purpose": CHILD_PREFIX + "parent-1",
        },
    )


def test_fresh_heartbeat_blocks_age_reclaim_of_a_never_ended_child(tmp_path, monkeypatch):
    """A fresh heartbeat is proof of life: the age sweep must leave it alone."""
    ws = tmp_path / "repo" / "repo-child-fresh"
    ws.mkdir(parents=True)
    _record(ws)
    _live_binding(ws)
    STORE.record_child_heartbeat("child-1", now=time.time())

    monkeypatch.setattr(STORE, "dirty", lambda path: False)
    tiers = STORE.gc_tiers()

    tier3 = [item for item in tiers["tier3"] if item.get("slug") == ws.name]
    assert tier3, tiers
    assert ws.exists()


def test_stale_heartbeat_and_old_workspace_are_reclaimable_without_end_stamp(
    tmp_path, monkeypatch
):
    """The hard-death backstop needs both thresholds, then the end stamp is waived."""
    ws = tmp_path / "repo" / "repo-child-hard-dead"
    ws.mkdir(parents=True)
    _record(ws)
    _live_binding(ws)
    STORE.db.set_kv(STORE.child_heartbeat_key("child-1"), f"{time.time() - 90000:.6f}")
    monkeypatch.setattr(STORE, "dirty", lambda path: False)
    monkeypatch.setattr(STORE, "find_repo", lambda name: tmp_path / "repo")

    tiers = STORE.gc_tiers()
    assert any(item.get("slug") == ws.name for item in tiers["tier1"]), tiers

    out = STORE.reclaim_child_workspace(ws, dry_run=True)
    assert out["ok"], out
    assert out["reason"] == "", out


def test_stale_heartbeat_but_young_workspace_is_not_reclaimable(tmp_path, monkeypatch):
    """One stale marker alone is not enough: the age threshold must also pass."""
    ws = tmp_path / "repo" / "repo-child-young"
    ws.mkdir(parents=True)
    _record(ws, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _live_binding(ws)
    STORE.db.set_kv(STORE.child_heartbeat_key("child-1"), f"{time.time() - 90000:.6f}")
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    tiers = STORE.gc_tiers()
    assert any(item.get("slug") == ws.name for item in tiers["tier3"]), tiers


def test_ended_stamped_child_reclaims_immediately_regardless_of_heartbeat(
    tmp_path, monkeypatch
):
    """A genuine graceful stop is authoritative over a fresh heartbeat."""
    ws = tmp_path / "repo" / "repo-child-ended"
    ws.mkdir(parents=True)
    _record(ws)
    STORE.db.write_binding(
        "child-1",
        {
            "mode": "repo",
            "repo": "repo",
            "slug": ws.name,
            "path": str(ws),
            "purpose": CHILD_PREFIX + "parent-1",
            "session_ended_at": "2020-01-02T00:00:00",
        },
    )
    STORE.record_child_heartbeat("child-1", now=time.time())
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    tiers = STORE.gc_tiers()
    assert any(item.get("slug") == ws.name for item in tiers["tier1"]), tiers


def test_heartbeat_write_failure_does_not_gate_block(tmp_path, monkeypatch):
    """A broken heartbeat is swallowed: the child still gets its workspace."""
    ws = tmp_path / "repo" / "parent-child-abc123"
    ws.mkdir(parents=True)
    monkeypatch.setattr(
        STORE,
        "read_state",
        lambda sid: {"mode": "repo", "repo": "repo", "slug": "parent", "path": "/tmp/parent"},
    )
    monkeypatch.setattr(
        PLUGIN.childspace,
        "ensure_child_workspace",
        lambda sid, parent_state, **kw: {
            "ok": True,
            "reason": "",
            "repo": "repo",
            "slug": ws.name,
            "path": str(ws),
        },
    )
    # Per-child parent map (child_runs) - not a module global.
    STORE.db.write_child_run("child-1", "parent-1")

    def boom(*args, **kwargs):
        raise RuntimeError("heartbeat database is broken")

    monkeypatch.setattr(STORE, "record_child_heartbeat", boom)

    import sys
    import types

    mod = types.ModuleType("agent.delegation_context")
    mod.is_delegated_child_context = lambda: True
    monkeypatch.setitem(sys.modules, "agent.delegation_context", mod)

    result = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="child-1")
    assert result is None, result


def test_two_rapid_calls_write_the_heartbeat_once(tmp_path, monkeypatch):
    """The rate limit keeps rapid gated calls from churning the kv store."""
    values: dict[str, str] = {}

    def fake_set(key, value):
        values[key] = value

    def fake_get(key, default=None):
        return values.get(key, default)

    monkeypatch.setattr(STORE.db, "set_kv", fake_set)
    monkeypatch.setattr(STORE.db, "get_kv", fake_get)
    now = time.time()
    assert STORE.record_child_heartbeat("child-rate", now=now) is True
    assert STORE.record_child_heartbeat("child-rate", now=now + 1) is False
    assert list(values) == [STORE.child_heartbeat_key("child-rate")], values


def test_gateway_restart_orphan_becomes_reclaimable_after_both_thresholds(
    tmp_path, monkeypatch
):
    """No stop event: the heartbeat goes stale and the age sweep can reclaim."""
    ws = tmp_path / "repo" / "repo-child-restart"
    ws.mkdir(parents=True)
    _record(ws)
    _live_binding(ws)
    STORE.db.set_kv(STORE.child_heartbeat_key("child-1"), f"{time.time() - 90000:.6f}")
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    assert STORE.child_heartbeat_stale("child-1", record={"created_at": "2020-01-01T00:00:00"})
    targets = STORE._ended_child_workspaces()
    assert any(item.get("slug") == ws.name for item in targets), targets
