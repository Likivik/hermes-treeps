"""Recorded test verdicts for child workspaces - the landing gate's evidence.

WHY this module exists: every audit of this project found the same thing -
the agent reported "all green", and nobody re-ran the tests. A self-report is
not a verdict. This module runs the verification command ITSELF, in the
child's workspace, and stores what actually happened.

The stored verdict is the evidence the parent is asked for before it lands a
child's work: :func:`landing_blocker` answers "may the parent land this?",
and the answer is a human-readable reason naming the offending child.

Design rules, all of them load-bearing:

* :func:`run_verdict` NEVER raises. A verification that crashes is a
  verification that failed; the caller gets ``ok=False`` plus the reason.
* The command runs with ``cwd`` = the workspace, in its own process group,
  so a timeout kills the whole tree instead of leaving orphans behind.
* A verdict is only as good as its timestamp: every stored verdict carries
  ``recorded_at``, and a verdict older than ``max_age_s`` is STALE - the
  child's tree may have changed since the run.
* The bindings table is read through the existing db functions; no SQL is
  added to db.py.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from typing import Any

from . import db

logger = logging.getLogger(__name__)

# How much of the command's combined output is kept.
OUTPUT_TAIL_LIMIT = 4000

# The kv key prefix a session's verdict is stored under.
VERDICT_KEY_PREFIX = "verdict:"


def _verdict_key(session_id: str) -> str:
    return f"{VERDICT_KEY_PREFIX}{session_id}"


# ── running the command ─────────────────────────────────────────────────────


def run_verdict(
    workspace_path: str,
    *,
    command: str,
    timeout_s: int,
) -> dict[str, Any]:
    """Run ``command`` in ``workspace_path`` and report what happened.

    Returns a dict::

        {
            "ok": bool,
            "command": str,
            "exit_code": int | None,   # None on timeout / no start
            "tail": str,               # last 4000 chars of combined output
            "duration_s": float,
            "error": str,              # only when something went wrong
        }

    Never raises. An empty ``command`` is an ERROR naming the command
    ("no verify command configured"), not an exception - the caller asked
    for a verdict and gets one.
    """
    started = time.monotonic()
    cmd = str(command or "").strip()
    result: dict[str, Any] = {
        "ok": False,
        "command": cmd,
        "exit_code": None,
        "tail": "",
        "duration_s": 0.0,
        "error": "",
    }

    def _finish() -> dict[str, Any]:
        result["duration_s"] = round(time.monotonic() - started, 3)
        return result

    if not cmd:
        result["error"] = (
            "no verify command configured (verify_command is empty) - "
            "nothing was run"
        )
        return _finish()

    try:
        timeout = int(timeout_s)
    except (TypeError, ValueError):
        timeout = 900
    timeout = max(1, timeout)

    cwd = str(workspace_path or "")
    if not cwd or not os.path.isdir(cwd):
        result["error"] = f"the workspace {workspace_path!r} is not a directory"
        return _finish()

    try:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        result["error"] = f"could not start the verify command: {type(exc).__name__}: {exc}"
        return _finish()

    try:
        out, _ = proc.communicate(timeout=timeout)
        result["exit_code"] = proc.returncode
        result["tail"] = (out or "")[-OUTPUT_TAIL_LIMIT:]
        result["ok"] = proc.returncode == 0
        if not result["ok"]:
            result["error"] = f"the verify command failed (exit {proc.returncode})"
    except subprocess.TimeoutExpired:
        # Kill the whole process group: a verification that spawns children
        # must not leave them running after the timeout.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            proc.kill()
        try:
            out, _ = proc.communicate(timeout=10)
        except Exception:
            out = ""
        result["exit_code"] = None
        result["tail"] = (
            f"timed out after {timeout}s\n" + (out or "")[-OUTPUT_TAIL_LIMIT:]
        ).strip()
        result["ok"] = False
        result["error"] = f"the verify command timed out after {timeout}s"
    except Exception as exc:  # never raise into the caller
        logger.debug("the verify command failed", exc_info=True)
        result["error"] = f"{type(exc).__name__}: {exc}"
        return _finish()
    return _finish()


# ── storing and reading the verdict ─────────────────────────────────────────


def record_verdict(session_id: str, verdict: dict[str, Any]) -> None:
    """Store ``verdict`` for ``session_id``.

    The stored record is the verdict plus the time it was taken: a verdict
    without a timestamp cannot be aged, and an aged verdict is what the
    landing gate needs.
    """
    payload = dict(verdict or {})
    payload.setdefault("recorded_at", time.time())
    payload.setdefault("recorded_at_iso", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        db.set_kv_json(_verdict_key(str(session_id)), payload)
    except Exception:
        logger.warning("could not record the verdict for %s", session_id, exc_info=True)


def read_verdict(session_id: str) -> dict[str, Any] | None:
    """The verdict recorded for ``session_id``, or None when there is none."""
    try:
        value = db.get_kv_json(_verdict_key(str(session_id)))
    except Exception:
        logger.warning("could not read the verdict for %s", session_id, exc_info=True)
        return None
    if isinstance(value, dict):
        return value
    return None


def _child_session_ids(parent_session_id: str) -> list[str]:
    """Every session whose binding says it was inherited from the parent."""
    parent = str(parent_session_id or "")
    if not parent:
        return []
    try:
        conn = db.connect()
        rows = conn.execute(
            "SELECT session_id FROM bindings WHERE inherited_from=? ORDER BY session_id",
            (parent,),
        ).fetchall()
    except Exception:
        logger.warning("could not list the children of %s", parent, exc_info=True)
        return []
    return [str(r[0]) for r in rows]


def verdicts_for(parent_session_id: str) -> dict[str, dict[str, Any]]:
    """The recorded verdicts of every child of ``parent_session_id``.

    Only children that HAVE a verdict appear: a child with no verdict is a
    missing verdict, and :func:`landing_blocker` reports it as such.
    """
    out: dict[str, dict[str, Any]] = {}
    for child in _child_session_ids(parent_session_id):
        verdict = read_verdict(child)
        if verdict is not None:
            out[child] = verdict
    return out


# ── the landing gate ────────────────────────────────────────────────────────


def _verdict_age_s(verdict: dict[str, Any], now: float) -> float:
    """How old the verdict is, in seconds. A verdict with no timestamp is ancient."""
    raw = verdict.get("recorded_at")
    if raw is None:
        return float("inf")
    try:
        recorded = float(raw)
    except (TypeError, ValueError):
        return float("inf")
    return max(0.0, now - recorded)


def landing_blocker(parent_session_id: str, *, max_age_s: int = 3600) -> str | None:
    """May the parent land its children's work?

    Returns None when every child of ``parent_session_id`` has a fresh,
    passing verdict. Otherwise a human-readable message naming the
    offending child ids and why (red / missing / stale).

    A parent with no children is not blocked: there is nothing unverified to
    land.
    """
    parent = str(parent_session_id or "")
    children = _child_session_ids(parent)
    if not children:
        return None

    now = time.time()
    problems: list[str] = []
    for child in children:
        verdict = read_verdict(child)
        if verdict is None:
            problems.append(f"{child}: no verification verdict was recorded")
            continue
        if not bool(verdict.get("ok")):
            detail = str(verdict.get("error") or "").strip()
            exit_code = verdict.get("exit_code")
            why = detail or (f"exit {exit_code}" if exit_code is not None else "the verdict is not ok")
            problems.append(f"{child}: the verification FAILED ({why})")
            continue
        age = _verdict_age_s(verdict, now)
        if age > max_age_s:
            problems.append(
                f"{child}: the verification verdict is STALE "
                f"({int(age)}s old, limit {max_age_s}s) - re-run it"
            )
    if not problems:
        return None
    return (
        f"cannot land the work of session {parent}: "
        + "; ".join(problems)
    )
