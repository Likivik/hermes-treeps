"""The one lifecycle derivation, pinned by case.

Every caller that used to derive "is it live / ended / an orphan" on its own
now branches on `lifecycle()`. These cases are the contract: if one changes,
every command that shows a workspace to a user changes with it.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import plugin_loader  # noqa: E402

STORE = plugin_loader.STORE
LIFECYCLE = importlib.import_module("hermes_treeps.lifecycle")


@pytest.fixture()
def ws_home(tmp_path, monkeypatch):
    monkeypatch.setattr(STORE, "workspaces_base", lambda: tmp_path)
    monkeypatch.setattr(STORE, "repo_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(STORE, "state_dir", lambda: tmp_path / "state")
    return tmp_path


def _item(path: Path, record: dict | None, orphan: bool = False) -> dict:
    return {"repo": "repo", "slug": path.name, "path": str(path), "record": record, "orphan": orphan}


def _record(purpose: str = "repo", sessions: list[str] | None = None, created: str = "") -> dict:
    return {"purpose": purpose, "sessions": sessions or [], "created_at": created}


def test_no_record_is_orphan(ws_home):
    tree = ws_home / "no-record"
    tree.mkdir()
    lc = LIFECYCLE.lifecycle(_item(tree, None, orphan=True))
    assert lc.state == LIFECYCLE.ORPHAN
    assert lc.reasons == ("no workspace record",)


def test_record_with_zero_sessions_is_orphan(ws_home):
    tree = ws_home / "no-sessions"
    tree.mkdir()
    lc = LIFECYCLE.lifecycle(_item(tree, _record()))
    assert lc.state == LIFECYCLE.ORPHAN
    assert lc.reasons == ("record with no bound session",)


def test_open_session_is_live(ws_home):
    tree = ws_home / "live"
    tree.mkdir()
    STORE.write_state("s-live", {"mode": "repo", "repo": "repo", "slug": "live", "path": str(tree)})
    lc = LIFECYCLE.lifecycle(_item(tree, _record(sessions=["s-live"])))
    assert lc.state == LIFECYCLE.LIVE
    assert lc.live and not lc.ended and not lc.orphan


def test_ended_session_is_ended(ws_home):
    tree = ws_home / "ended"
    tree.mkdir()
    STORE.write_state(
        "s-ended",
        {"mode": "repo", "slug": "ended", "path": str(tree), "session_ended_at": "2026-01-01T00:00:00"},
    )
    lc = LIFECYCLE.lifecycle(_item(tree, _record(sessions=["s-ended"])))
    assert lc.state == LIFECYCLE.ENDED
    assert lc.reasons == ("all bound sessions ended",)


def test_child_detected_from_purpose(ws_home):
    tree = ws_home / "kid"
    tree.mkdir()
    STORE.write_state(
        "s-kid",
        {"mode": "repo", "slug": "kid", "path": str(tree), "session_ended_at": "2026-01-01T00:00:00"},
    )
    lc = LIFECYCLE.lifecycle(_item(tree, _record(purpose=f"{STORE.CHILD_PURPOSE_PREFIX}parent", sessions=["s-kid"])))
    assert lc.is_child is True


def test_one_live_session_of_two_makes_it_live(ws_home):
    """All-sessions-ended is the rule; a single open session keeps it live."""
    tree = ws_home / "mixed"
    tree.mkdir()
    STORE.write_state("s-a", {"path": str(tree), "session_ended_at": "2026-01-01T00:00:00"})
    STORE.write_state("s-b", {"path": str(tree)})
    lc = LIFECYCLE.lifecycle(_item(tree, _record(sessions=["s-a", "s-b"])))
    assert lc.state == LIFECYCLE.LIVE


def test_hard_dead_child_without_end_stamp_is_ended(ws_home, monkeypatch):
    """The gateway-restart hole: no subagent_stop, but heartbeat went stale."""
    tree = ws_home / "dead-kid"
    tree.mkdir()
    old = time.time() - 3 * 86400
    import os

    os.utime(tree, (old, old))
    STORE.write_state("s-dead", {"path": str(tree)})  # never got session_ended_at
    monkeypatch.setattr(
        LIFECYCLE.db,
        "get_kv",
        lambda key: str(old) if key == LIFECYCLE.child_heartbeat_key("s-dead") else None,
    )
    lc = LIFECYCLE.lifecycle(
        _item(tree, _record(purpose=f"{STORE.CHILD_PURPOSE_PREFIX}parent", sessions=["s-dead"]))
    )
    assert lc.state == LIFECYCLE.ENDED
    assert lc.is_child


def test_young_child_without_heartbeat_stays_live(ws_home, monkeypatch):
    """Age alone must not kill a child; the two-threshold rule needs evidence."""
    tree = ws_home / "young-kid"
    tree.mkdir()
    STORE.write_state("s-young", {"path": str(tree)})
    monkeypatch.setattr(LIFECYCLE.db, "get_kv", lambda key: None)
    lc = LIFECYCLE.lifecycle(
        _item(tree, _record(purpose=f"{STORE.CHILD_PURPOSE_PREFIX}parent", sessions=["s-young"]))
    )
    assert lc.state == LIFECYCLE.LIVE


def test_age_prefers_record_over_mtime(ws_home):
    tree = ws_home / "aged"
    tree.mkdir()
    lc = LIFECYCLE.lifecycle(_item(tree, _record(created="2020-01-01T00:00:00")))
    assert lc.age_days > 1000
