"""capacity.py: the live-child ledger the delegation gate counts against.

WHAT this is: the plugin's only source of truth for "how many delegated
children are live right now". The host fires ``subagent_start`` /
``subagent_stop`` hooks (wired in ``__init__.py``); each start opens a row in
the ``child_capacity`` table and each stop stamps it closed. The gate reads
the OPEN rows (``ended_at == ''``) whose writer pid is still alive, and
refuses a ``delegate_task`` that would push the live count past the cap.

WHY pid liveness: a gateway that crashed mid-delegation leaves open rows
behind forever. A row written by a pid that no longer exists (``os.kill(pid,
0)`` raises ``ProcessLookupError``) is dead weight, not a live child, and must
not eat a capacity slot for the rest of the install's life. ``EPERM`` means
the pid exists but belongs to someone else: still alive, still counted.

WHY scopes: capacity is only meaningful per model account, and this host
runs several Hermes profiles off the same model quota. The default scope is
``profile`` (this profile's own ledger only); ``profiles`` reads the listed
profile homes; ``global`` reads every profile home under the plugin's hermes
home. A ledger that is absent, unreadable or missing the table is skipped
silently - the count degrades to "the rows I could read", never an exception
out of the gate.

Import discipline: this module must import cleanly with nothing but the
stdlib and ``from . import db`` (the plugin loads on hosts with no Hermes
install at all, and the test suite loads it in a bare interpreter). Settings
and the delegation-config fallback are imported lazily inside functions.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from . import db

logger = logging.getLogger(__name__)

# The row columns every reader of this module agrees on.
CAPACITY_COLUMNS = (
    "subagent_id",
    "child_session_id",
    "delegation_id",
    "model",
    "provider",
    "gateway_pid",
    "started_at",
    "ended_at",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── writes ───────────────────────────────────────────────────────────────────


def record_start(
    subagent_id: str,
    child_session_id: str = "",
    delegation_id: str = "",
    model: str = "",
    provider: str = "",
) -> bool:
    """Open a ledger row for one delegated child.

    WHY an upsert and not a plain INSERT: a re-dispatched child reuses its
    subagent id, and a second open row for the same child would be counted
    twice against the cap. The rewrite is idempotent: same id, one open row.

    Never raises on a database hiccup (returns False): a broken ledger row
    must never break the delegation it is watching.
    """
    if not subagent_id:
        return False
    try:
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO child_capacity"
                " (subagent_id, child_session_id, delegation_id, model, provider,"
                "  gateway_pid, started_at, ended_at)"
                " VALUES(?,?,?,?,?,?,?,'')"
                " ON CONFLICT(subagent_id) DO UPDATE SET"
                "  child_session_id=excluded.child_session_id,"
                "  delegation_id=excluded.delegation_id,"
                "  model=excluded.model,"
                "  provider=excluded.provider,"
                "  gateway_pid=excluded.gateway_pid,"
                "  started_at=excluded.started_at,"
                "  ended_at=''",
                (
                    subagent_id,
                    child_session_id,
                    delegation_id,
                    model,
                    provider,
                    os.getpid(),
                    _utc_now(),
                ),
            )
        return True
    except Exception:
        logger.debug("capacity ledger: could not record start for %s", subagent_id, exc_info=True)
        return False


def record_stop(subagent_id: str) -> bool:
    """Stamp a ledger row closed. A missing row is a no-op, never an error."""
    if not subagent_id:
        return False
    try:
        with db.transaction() as conn:
            conn.execute(
                "UPDATE child_capacity SET ended_at=? WHERE subagent_id=?",
                (_utc_now(), subagent_id),
            )
        return True
    except Exception:
        logger.debug("capacity ledger: could not record stop for %s", subagent_id, exc_info=True)
        return False


# ── reads ─────────────────────────────────────────────────────────────────────


def _pid_is_alive(pid: int) -> bool:
    """True when ``pid`` names a process that exists.

    ``ProcessLookupError`` (ESRCH) is the only proof of death; ``EPERM`` means
    the pid exists but belongs to another user; anything else (a zombie
    ``os.kill`` quirk, a pid of 0) is conservatively alive - a row that cannot
    be proven dead must keep its capacity slot.
    """
    if not pid or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _ledger_paths(scope: str, profiles: list[str] | None, ledger_dir: str) -> list[Path]:
    """The ledger database files a scope resolves to, best-effort."""
    if ledger_dir:
        # An explicit ledger dir wins: one directory holding one data.db.
        return [Path(ledger_dir) / db.DB_FILENAME]
    # The own ledger is resolved through db.db_path(), which honours the
    # same env overrides the writes do - a read path and a write path that
    # disagree would count nothing the plugin itself recorded.
    if scope == "profile":
        try:
            return [Path(db.db_path())]
        except Exception:
            logger.debug("capacity ledger: own db path unreadable", exc_info=True)
            return []
    homes: list[Path] = []
    if scope == "profiles":
        base = db.hermes_home()
        for entry in profiles or []:
            entry = str(entry)
            if entry.startswith("/"):
                homes.append(Path(entry))
            else:
                homes.append(base / "profiles" / entry)
    elif scope == "global":
        base = db.hermes_home()
        homes.append(base)
        profiles_dir = base / "profiles"
        try:
            homes.extend(sorted(p for p in profiles_dir.iterdir() if p.is_dir()))
        except OSError:
            pass
    else:
        logger.debug("capacity ledger: unknown scope %r; falling back to profile", scope)
        homes.append(db.hermes_home())
    return [h / "plugin-data" / db.PLUGIN_NAME / db.DB_FILENAME for h in homes]


def open_rows(
    scope: str = "profile",
    profiles: list[str] | None = None,
    ledger_dir: str = "",
) -> list[dict[str, Any]]:
    """The open ledger rows (``ended_at == ''``) whose writer pid is alive.

    Every ledger that is absent, unreadable or missing the table is skipped
    silently - this runs inside the pre_tool_call gate, where an exception
    would fail the tool call, not just the count.
    """
    rows: list[dict[str, Any]] = []
    for path in _ledger_paths(scope, profiles, ledger_dir):
        rows.extend(_open_rows_at(path))
    return rows


def _open_rows_at(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT " + ", ".join(CAPACITY_COLUMNS) + " FROM child_capacity"
                " WHERE ended_at = ''"
            )
            out = []
            for row in cur.fetchall():
                d = {k: row[k] for k in CAPACITY_COLUMNS}
                if _pid_is_alive(int(d.get("gateway_pid") or 0)):
                    out.append(d)
            return out
        finally:
            conn.close()
    except Exception:
        # Not a database, missing table, locked, unreadable: not our row.
        logger.debug("capacity ledger: skipping unreadable ledger %s", path, exc_info=True)
        return []


def count_live(
    scope: str = "profile",
    profiles: list[str] | None = None,
    ledger_dir: str = "",
    match: Callable[[dict[str, Any]], bool] | None = None,
) -> int:
    """How many children are live, optionally filtered by ``match(row)``."""
    rows = open_rows(scope=scope, profiles=profiles, ledger_dir=ledger_dir)
    if match is None:
        return len(rows)
    try:
        return sum(1 for r in rows if match(r))
    except Exception:
        logger.debug("capacity ledger: match predicate failed; counting nothing", exc_info=True)
        return 0


# ── classification and decisions ──────────────────────────────────────────────


def is_minimax(model: str, provider: str) -> bool:
    """True when the model or the provider string names MiniMax."""
    return "minimax" in f"{model or ''} {provider or ''}".lower()


def classify_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count rows per provider string (an empty provider buckets as ``''``)."""
    counts: dict[str, int] = {}
    for row in rows or []:
        key = str((row or {}).get("provider") or "")
        counts[key] = counts.get(key, 0) + 1
    return counts


def should_overflow(index: int, live_minimax: int, overflow_after: int) -> bool:
    """Should child ``index`` (0-based) in this batch run on the overflow route?

    The first ``overflow_after`` MiniMax-class children fit on the primary
    route; everything past that - including the slots already taken by live
    MiniMax children - overflows to the second provider.
    """
    try:
        return (int(index) + int(live_minimax)) >= int(overflow_after)
    except (TypeError, ValueError):
        return False


def verdict(requested_children: int, live: int, max_children: int) -> tuple[bool, str]:
    """The gate's decision: (allowed, reason naming the counts).

    The reason is shown to the model verbatim when the call is refused, so it
    names every number the decision used: live, requested, cap, and the free
    slots left.
    """
    try:
        requested = int(requested_children)
        live_n = int(live)
        cap = int(max_children)
    except (TypeError, ValueError):
        return True, "capacity verdict: unreadable counts; allowing"
    free = max(0, cap - live_n)
    if live_n + requested <= cap:
        return (
            True,
            f"capacity: {live_n} children live, {requested} requested, cap {cap}"
            f" ({free} free slots)",
        )
    return (
        False,
        f"capacity: {live_n} children live, {requested} requested, cap {cap}"
        f" - only {free} free slot(s)",
    )


# ── settings glue ─────────────────────────────────────────────────────────────


def config_from_settings(s: dict[str, Any]) -> dict[str, Any]:
    """The kwargs the gate and the backend need, from a settings() dict.

    Tolerant of missing keys (a settings dict shaped by an older release):
    every value has the same default settings() itself would fill in.
    """
    return {
        "enabled": bool(s.get("capacity_enabled", True)),
        "max_children": int(s.get("capacity_max_children") or 8),
        "overflow_after": int(s.get("capacity_overflow_after") or 4),
        "scope": str(s.get("capacity_tracking_scope") or "profile"),
        "profiles": list(s.get("capacity_tracking_profiles") or []),
        "ledger_dir": str(s.get("capacity_ledger_dir") or ""),
    }


def delegation_route_defaults() -> tuple[str, str]:
    """The host's configured delegation (model, provider), for ledger rows.

    The subagent hooks do not carry the child's model/provider (verified
    against the host's dispatch payload: parent/child ids, role, goal only),
    so a start row falls back to what the host's ``delegation`` section
    configures for children.
    """
    model = provider = ""
    try:
        from .settings import _load_config

        section = _load_config().get("delegation")
        if isinstance(section, dict):
            model = str(section.get("model") or "")
            provider = str(section.get("provider") or "")
    except Exception:
        logger.debug("capacity ledger: could not read the delegation route", exc_info=True)
    return model, provider
