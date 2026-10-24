"""The landing gate: no push, no merge, while a child's verdict is red.

WHY: two incidents in this project came from a parent claiming green without
re-running the child's checks. A gate that refuses the landing command makes
that impossible instead of relying on the parent's discipline.

The gate is deliberately narrow. It only looks at shell calls that actually
LAND work - ``git push``, ``gh pr merge``, ``jj git push`` - and leaves
everything else (including ``gh pr create``) alone. The decision itself is
delegated to :mod:`verify`, which knows what a red verdict is; this module
only knows what a landing command looks like.

Two rules govern the failure modes:

* the gate's OWN bugs fail OPEN (``None``): a broken gate must not wedge the
  host's shell;
* a RED VERDICT fails CLOSED: the landing command is refused and the message
  names the offending child sessions and what to do about them.

Settings are read through :func:`store.settings` and every key is treated as
optional, because the gate must keep working on configs written before the
keys existed.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: The tools that can land work. Only the shell tool is inspected - a file
#: write cannot push anything.
_LANDING_TOOLS = frozenset({"terminal"})

#: The shell commands that land work. ``gh pr create`` is NOT here: opening a
#: pull request is not landing it.
_LANDING_COMMANDS = (
    re.compile(r"^\s*git\s+push\b"),
    re.compile(r"^\s*gh\s+pr\s+merge\b"),
    re.compile(r"^\s*jj\s+git\s+push\b"),
)

#: The argument names the host's terminal tool may carry the command in.
#: ``command`` is the real one (verified in the host's terminal tool schema);
#: ``cmd`` is accepted as a tolerant fallback.
_COMMAND_KEYS = ("command", "cmd")

#: Shell separators, so a chained command is inspected piece by piece:
#: ``cd x && git push`` must be caught, not waved through.
_SEPARATORS = re.compile(r"(?:&&|\|\||;|\||\n)")

#: Prefixes that still lead into the real command.
_PREFIXES = re.compile(r"^(?:sudo\s+)?(?:env\s+\S+\s+)*")


def landing_tools() -> frozenset[str]:
    """The tool names whose calls are inspected by the gate."""
    return _LANDING_TOOLS


def _command_of(args: Any) -> str:
    """The shell command in a tool call's arguments (empty when absent)."""
    if not isinstance(args, dict):
        return ""
    for key in _COMMAND_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def is_landing_command(command: str) -> bool:
    """True when the shell command lands work."""
    if not command:
        return False
    for segment in _SEPARATORS.split(command):
        stripped = segment.strip()
        if not stripped:
            continue
        stripped = _PREFIXES.sub("", stripped).strip()
        for pattern in _LANDING_COMMANDS:
            if pattern.search(stripped):
                return True
    return False


def _settings() -> dict[str, Any]:
    """The plugin's settings, or an empty mapping when they cannot be read."""
    try:
        from . import store  # type: ignore[attr-defined]

        return dict(store.settings())
    except Exception:  # pragma: no cover - defensive
        logger.debug("landing gate: settings unavailable", exc_info=True)
        return {}


def _verify_module():
    """The :mod:`verify` module, or ``None`` when it is not installed yet."""
    try:
        from . import verify  # type: ignore[attr-defined]

        return verify
    except ImportError:
        return None
    except Exception:  # pragma: no cover - defensive
        logger.debug("landing gate: verify module unusable", exc_info=True)
        return None


def _child_ids(blocker_result: Any) -> list[str]:
    """The offending child session ids in whatever shape the blocker used.

    Accepted shapes: ``None``/falsy (allowed), a list/tuple of ids, a single
    id string, or a mapping carrying the ids under one of the known keys.
    """
    if blocker_result is None or blocker_result is False:
        return []
    if isinstance(blocker_result, str):
        return [blocker_result] if blocker_result else []
    if isinstance(blocker_result, dict):
        for key in ("children", "child_session_ids", "sessions", "ids", "offenders"):
            value = blocker_result.get(key)
            if isinstance(value, (list, tuple, set)):
                return [str(v) for v in value if v]
            if isinstance(value, str) and value:
                return [value]
        return []
    if isinstance(blocker_result, (list, tuple, set)):
        return [str(v) for v in blocker_result if v]
    # Any other truthy value means "blocked, but the ids are unknown".
    return ["(unknown)"]


def check(tool_name: str, args: dict, session_id: str) -> dict | None:
    """Refuse a landing command while a child's verdict is red.

    Returns ``None`` when the call is allowed, or
    ``{"action": "block", "message": ...}`` when it must be refused.
    """
    try:
        if tool_name not in _LANDING_TOOLS:
            return None
        command = _command_of(args)
        if not is_landing_command(command):
            return None

        settings = _settings()
        if not settings.get("landing_gate_enabled", True):
            return None

        verify = _verify_module()
        if verify is None:
            # No verify module: nothing can be red, so nothing is blocked.
            return None
        blocker = getattr(verify, "landing_blocker", None)
        if blocker is None:
            return None
        result = blocker(session_id)
        child_ids = _child_ids(result)
        if not child_ids:
            return None

        verify_command = settings.get("verify_command") or "hermes ws verify"
        ids = ", ".join(child_ids)
        message = (
            f"Landing blocked: {ids} has a red verification verdict.\n"
            f"Re-run the child's verification (`{verify_command}`) or re-run the "
            f"child itself, then land again. The landing command was refused: "
            f"{command.strip()[:200]}"
        )
        return {"action": "block", "message": message}
    except Exception:  # pragma: no cover - the gate's own bugs fail OPEN
        logger.warning("landing gate: internal error, failing open", exc_info=True)
        return None
