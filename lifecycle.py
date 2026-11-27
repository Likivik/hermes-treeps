"""One derivation of "what is this workspace, really".

WHY this module exists: before it, the same three facts (is a child, has a
live session, is the child hard-dead) were computed independently in
``gc.py``, ``cli.py``, ``retire.py`` and ``childreport.py``. Each copy was
correct on its own and slightly different from its neighbours - the classic
way two commands come to disagree about the same workspace in front of a
user. Everything that needs to know a workspace's state asks here.

The state set is deliberately tiny. ``retireable`` is NOT a state: whether
a workspace may be removed is a policy question (tier rules, evidence) and
belongs to the caller, not to the lifecycle.
"""

from __future__ import annotations

import calendar
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import db
from . import store as _store

CHILD_HEARTBEAT_STALE_HOURS = 24

# The state names a caller may branch on.
LIVE = "live"
ENDED = "ended"
ORPHAN = "orphan"


@dataclass(frozen=True)
class Lifecycle:
    """Everything the classification callers need, computed once."""

    state: str
    is_child: bool
    sessions: tuple[str, ...]
    age_days: float
    # Human-readable evidence for the state, in the order it was established.
    # Every caller that shows a workspace to a user should show these: they
    # are the difference between "tier4" and "why tier4".
    reasons: tuple[str, ...] = ()

    @property
    def live(self) -> bool:
        return self.state == LIVE

    @property
    def ended(self) -> bool:
        return self.state == ENDED

    @property
    def orphan(self) -> bool:
        return self.state == ORPHAN


CHILD_HEARTBEAT_KEY_PREFIX = "child_seen:"
CHILD_HEARTBEAT_MIN_INTERVAL_S = 60


def child_heartbeat_key(session_id: str) -> str:
    """The kv key holding one child's last gated-tool heartbeat."""
    return f"{CHILD_HEARTBEAT_KEY_PREFIX}{session_id}"


def parse_timestamp(raw: str) -> float | None:
    """Parse a heartbeat or workspace timestamp into epoch seconds."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return float(calendar.timegm(time.strptime(text[:19], fmt)))
        except Exception:
            continue
    return None


def workspace_age_days(record: dict[str, Any], path: Path) -> float:
    """Age in days from the record's created_at, else from the tree's mtime."""
    raw = str(record.get("created_at") or "")
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            # created_at is written with time.gmtime() - UTC. time.mktime would
            # read it as local time and make every workspace look hours
            # younger than it is (UTC+3 here: the 7-day TTL fires 3h late).
            created = calendar.timegm(time.strptime(raw[:19], fmt))
            return max(0.0, (time.time() - created) / 86400)
        except Exception:
            continue
    try:
        return max(0.0, (time.time() - path.stat().st_mtime) / 86400)
    except Exception:
        return 0.0


def child_hard_dead(
    record: dict[str, Any],
    path: str | Path,
    *,
    stale_hours: int = CHILD_HEARTBEAT_STALE_HOURS,
) -> bool:
    """True only when BOTH heartbeat staleness and workspace age exceed the threshold.

    A missing heartbeat alone is not enough: a brand-new child that has not
    called a gated tool has no heartbeat yet. The two-threshold rule is the
    extra-sure guard against reclaiming a live child.

    WHY the heartbeat must EXIST here: a child with no heartbeat at all has
    never proved life through the plugin, so only its age can speak for it.
    The hard-death path requires positive evidence that the child WAS alive
    and then stopped reporting.
    """
    age_days = workspace_age_days(record, Path(path))
    if age_days * 24.0 <= float(stale_hours):
        return False
    sids = [str(x) for x in (record.get("sessions") or [])]
    if not sids:
        return False
    for sid in sids:
        try:
            raw = db.get_kv(child_heartbeat_key(sid))
        except Exception:
            return False
        seen = parse_timestamp(raw or "")
        if seen is None:
            # No heartbeat was ever written: absence is not a stale marker.
            return False
        if (time.time() - seen) <= float(stale_hours) * 3600.0:
            return False
    return True


def lifecycle(
    item: dict[str, Any],
    *,
    stale_hours: int = CHILD_HEARTBEAT_STALE_HOURS,
) -> Lifecycle:
    """Derive the state of one workspace item from ``store.list_workspaces()``.

    Order matters and mirrors the historical behaviour exactly:
    1. no record, or a record with zero bound sessions -> orphan
    2. a bound session that has not ended and is not hard-dead -> live
    3. everything else -> ended
    """
    record = item.get("record") or {}
    path = Path(str(item.get("path") or ""))
    sessions = tuple(str(x) for x in (record.get("sessions") or []))
    is_child = str(record.get("purpose") or "").startswith(_store.CHILD_PURPOSE_PREFIX)
    age_days = workspace_age_days(record, path)

    if item.get("orphan") or not sessions:
        reason = "no workspace record" if item.get("orphan") else "record with no bound session"
        return Lifecycle(ORPHAN, is_child, sessions, age_days, (reason,))

    states = [_store.read_state(sid) or {} for sid in sessions]
    ended = all(st.get("session_ended_at") for st in states)
    hard_dead = is_child and child_hard_dead(record, path, stale_hours=stale_hours)

    if not ended and not hard_dead:
        return Lifecycle(LIVE, is_child, sessions, age_days, ("bound session has not ended",))
    if hard_dead and not ended:
        return Lifecycle(
            ENDED,
            is_child,
            sessions,
            age_days,
            (f"child heartbeat older than {stale_hours}h and tree older than that",),
        )
    return Lifecycle(ENDED, is_child, sessions, age_days, ("all bound sessions ended",))
