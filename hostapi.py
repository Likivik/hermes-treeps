"""The only place that imports host internals.

WHY: the plugin reaches into private hermes modules (``agent.runtime_cwd``,
``tools.terminal_tool``, ``tools.approval_context``) to move a session's real
cwd. Those names are not a public contract and have moved before - each move
was discovered live, by a child running in the wrong tree. Keeping every
import here means an upstream rename is a one-file fix with one test file,
not a hunt through the gate.

Every function degrades instead of raising: on a host without the module the
caller keeps working with one DEBUG line, never a crash mid-turn.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_LOGGED: set[str] = set()


def _log_once(what: str) -> None:
    if what in _LOGGED:
        return
    _LOGGED.add(what)
    logger.debug("%s unavailable on this host build", what)


def _import(module: str, name: str) -> Any | None:
    try:
        mod = __import__(module, fromlist=[name])
        return getattr(mod, name)
    except Exception:
        _log_once(f"{module}.{name}")
        return None


def session_key() -> str:
    """The gateway turn key the terminal layer reads its cwd back under."""
    fn = _import("tools.approval_context", "get_current_session_key")
    if fn is None:
        return ""
    try:
        return str(fn() or "")
    except Exception:
        return ""


def set_logical_cwd(path: str) -> bool:
    """Move what context resolvers see as the session's cwd."""
    fn = _import("agent.runtime_cwd", "set_session_cwd")
    if fn is None:
        return False
    try:
        fn(str(path or ""))
        return True
    except Exception:
        return False


def clear_logical_cwd() -> bool:
    fn = _import("agent.runtime_cwd", "clear_session_cwd")
    if fn is None:
        return False
    try:
        fn()
        return True
    except Exception:
        return False


def scoped_cwd() -> str:
    """The pinned cwd as the context resolver reports it, or ""."""
    fn = _import("agent.runtime_cwd", "scoped_session_cwd")
    if fn is None:
        return ""
    try:
        return str(fn() or "")
    except Exception:
        return ""


def record_terminal_cwd(key: str, path: str) -> bool:
    """Record a cwd under one of the terminal's keys (sid, turn key, task id).

    An EMPTY key is refused here: the host collapses it to "default", which
    would re-point every session without its own record.
    """
    if not key:
        return False
    fn = _import("tools.terminal_tool", "record_session_cwd")
    if fn is None:
        return False
    try:
        fn(key, str(path or ""))
        return True
    except Exception:
        return False


def clear_terminal_cwd(key: str) -> bool:
    if not key:
        return False
    fn = _import("tools.terminal_tool", "clear_session_cwd")
    if fn is None:
        return False
    try:
        fn(key)
        return True
    except Exception:
        return False


def get_secret(name: str) -> str:
    """A secret from the host's scoped store, or "" (caller falls back)."""
    fn = _import("agent.secret_scope", "get_secret")
    if fn is None:
        return ""
    try:
        return str(fn(name) or "")
    except Exception:
        return ""


def delegation_provider_class() -> Any | None:
    """The host's DelegationProvider ABC, or None on a host predating the seam."""
    return _import("agent.delegation_provider", "DelegationProvider")


def is_delegated_child() -> bool:
    """True only inside a delegated child's tool dispatch (task-local ContextVar).

    Failing closed is deliberate: an unclassifiable caller must not be able to
    mint a user workspace.
    """
    fn = _import("agent.delegation_context", "is_delegated_child_context")
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def is_pinned_child(path: str) -> bool:
    """True when the host holds a live resume pin that covers this worktree.

    Best effort by design: the ``resumable_subagents`` table arrived in 0.23.2
    and its columns are not a public contract, so the row is matched by
    scanning it for the worktree path rather than by naming columns. A missing
    table, an old version or a lock all answer False - and False means gc may
    apply its own age rules, which is the pre-0.23.2 behaviour.
    """
    import json as _json
    import sqlite3

    target = str(path or "").rstrip("/")
    if not target:
        return False
    from . import db as _db

    try:
        state = _db.hermes_home() / "state.db"
        if not state.is_file():
            return False
        conn = sqlite3.connect(f"file:{state}?mode=ro", uri=True, timeout=5.0)
        try:
            rows = conn.execute("SELECT * FROM resumable_subagents").fetchall()
        finally:
            conn.close()
    except Exception:
        _log_once("state.db/resumable_subagents")
        return False
    for row in rows:
        try:
            blob = _json.dumps(row, default=str)
        except Exception:
            continue
        if target in blob:
            return True
    return False


def register_idle_task(ctx: Any, name: str, *, interval_hours: int, min_idle_hours: int, fn: Any) -> bool:
    """Ask the host to run `fn` when the gateway is idle (hermes >= 0.23.2).

    Below that version the attribute does not exist; the caller must have its
    own trigger (a CLI verb) and this returns False so it can say so once.
    """
    register = getattr(ctx, "register_idle_task", None) if ctx is not None else None
    if register is None:
        _log_once("ctx.register_idle_task")
        return False
    try:
        register(name, interval_hours=interval_hours, min_idle_hours=min_idle_hours, fn=fn)
        return True
    except Exception:
        logger.debug("idle task registration failed", exc_info=True)
        return False
