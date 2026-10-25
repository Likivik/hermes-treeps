"""childreport.py: honest child-end reporting.

WHAT this exists for: a delegated child finishes, and the PARENT session has
no idea what happened to the tree the child worked in. The parent's next turn
should see ONE line per child:

    child <slug>: <clean|dirty|state-unknown>, <N> unpushed commit(s) -
    <workspace path> (<status>)

The requirements behind that line, and where each is enforced:

* the child's WORKSPACE, resolved without guessing;
* whether it is CLEAN or DIRTY, and how many UNPUSHED commits it left;
* the child's own status word, straight from the host's ``subagent_stop``;
* delivered ONCE, then cleared - a report that repeats every turn is noise;
* SILENT when there is nothing to report - no empty section, no filler.

WHY the host's ``subagent_start`` / ``subagent_stop`` hooks and not
``on_session_end``: ``on_session_end`` fires when a session is finalized, which
for a delegated child is not the same moment as "the child stopped working",
and it does not carry the parent/child relationship. ``subagent_start`` and
``subagent_stop`` are the host's dedicated subagent lifecycle hooks and they
DO carry it: the start hook names the parent and the child, the stop hook
names the parent, the child and the child's outcome.

WHY the start side writes a row: the stop hook's payload names the child and
the parent, but NOT the child's workspace. The workspace lives in the child's
binding. Rather than reverse-lookup a binding (ambiguous - a session id can
be re-dispatched, and a child can inherit its parent's binding), the start
hook records the mapping explicitly and the stop hook reads it back.

WHY the stop side is FAST: the host runs ``subagent_stop`` on the CALLER
thread - the parent's own thread, inside the delegation finalization. A
``jj status`` or ``git log`` subprocess there blocks the parent's thread pool
and every other delegation behind it. So the expensive checks run on a
short-lived daemon thread and the callback returns immediately; the report row
is written by that thread.

WHY every store call is wrapped: a hook that raises inside the host's
dispatch is logged and swallowed, but a hook that raises while holding the
plugin's write lock can leave a transaction open. Worse, a reporting feature
is never worth failing a delegation over. Every call degrades to "unknown"
instead of propagating.

WHY the stop side also RECLAIMS the child's workspace: a finished child's
tree is dead weight on disk, and there is no reason to keep it for two
weeks. The reclamation happens on the SAME background thread as the report,
right AFTER the report row is written - the report describes the tree at
stop time, the reclamation then snapshots that dirty tree into a commit and
removes it. The setting ``child_immediate_reclaim`` turns the reclamation
off without turning the reporting off.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from . import contract, store

logger = logging.getLogger(__name__)

#: The report's states. ``unknown`` is the honest answer when the check
#: failed - "state unknown" beats a wrong "clean".
STATE_CLEAN = "clean"
STATE_DIRTY = "dirty"
STATE_UNKNOWN = "state-unknown"


# ── the start side ───────────────────────────────────────────────────────────


def record_child_start(
    parent_session_id: str,
    child_session_id: str,
    *,
    child_subagent_id: str = "",
    parent_turn_id: str = "",
    parent_subagent_id: str = "",
    child_role: str = "",
    child_goal: str = "",
) -> bool:
    """Record "child X belongs to parent Y". True when the row was written.

    Called from the host's ``subagent_start`` hook. Never raises: the host
    treats a raising observer as a plugin bug, and a missing row is
    survivable (the stop side falls back to the bindings table).
    """
    try:
        if not parent_session_id or not child_session_id:
            return False
        store.db.write_child_run(
            child_session_id,
            parent_session_id,
            child_subagent_id=str(child_subagent_id or ""),
            parent_turn_id=str(parent_turn_id or ""),
            parent_subagent_id=str(parent_subagent_id or ""),
            child_role=str(child_role or ""),
            child_goal=str(child_goal or ""),
        )
        return True
    except Exception:
        # A broken start side must never break the delegation it is
        # announcing: the child is about to run, and the host's contract is
        # that observers cannot veto work.
        logger.warning("child report: could not record child start", exc_info=True)
        return False


# ── the stop side ────────────────────────────────────────────────────────────


def _resolve_child_workspace(child_session_id: str) -> dict[str, str]:
    """The child's workspace, as ``{"path", "slug", "repo"}``.

    The source is the child's own binding - the gate bound the child to its
    own workspace, and that binding is the authoritative record of where it
    worked. It also works when the start hook never fired (the plugin was
    enabled mid-child), which is the fallback the requirement asks for.

    Never raises: a missing workspace yields empty strings, and the report
    says "no workspace" rather than blowing up the delegation.
    """
    out = {"path": "", "slug": "", "repo": ""}
    try:
        state = store.read_state(child_session_id) or {}
        out["path"] = str(state.get("path") or "")
        out["slug"] = str(state.get("slug") or "")
        out["repo"] = str(state.get("repo") or "")
    except Exception:
        logger.warning("child report: could not read the child's binding", exc_info=True)
    return out


def _unpushed_count(workspace: Path) -> int | None:
    """How many commits are not on the remote. None when it cannot be told.

    ``store.unpushed_commits`` answers "is there anything unpushed?" (a
    bool-or-None). The report wants a COUNT, so the check runs twice: the
    bool first (cheap, and it short-circuits the common clean case), and the
    count only when there is something to count.

    WHY None and not 0: "cannot tell" (offline, no upstream) must not read
    as "zero unpushed" - the parent's whole point is to learn the truth.
    """
    try:
        has_unpushed = store.unpushed_commits(workspace)
    except Exception:
        logger.warning("child report: unpushed check failed", exc_info=True)
        return None
    if has_unpushed is None:
        return None
    if not has_unpushed:
        return 0
    try:
        count = store.count_unpushed_commits(workspace)
    except Exception:
        logger.warning("child report: unpushed COUNT failed", exc_info=True)
        return None
    return count


def _dirty_state(workspace: Path) -> str:
    """``clean`` / ``dirty`` / ``state-unknown``.

    WHY ``store.dirty_state`` and not ``store.dirty``: the boolean answers
    a yes/no for the RETIRE guard, and a workspace that is not a usable
    checkout (retired mid-run, deleted, never a repo) makes it answer
    "no" - because the status command fails with empty output there, and
    empty output reads as "no changes".

    That is right for retiring (nothing to lose) and wrong here: a report
    that says "clean" about a workspace nobody could inspect is a lie, and
    this feature exists to stop lies. So the reporting path asks for the
    three-way answer and renders ``state-unknown`` when it gets None.
    """
    try:
        state = store.dirty_state(workspace)
    except Exception:
        logger.warning("child report: dirty check failed", exc_info=True)
        return STATE_UNKNOWN
    if state == STATE_DIRTY:
        return STATE_DIRTY
    if state == STATE_CLEAN:
        return STATE_CLEAN
    return STATE_UNKNOWN


def _mark_child_session_ended(db_file: str, child_session_id: str) -> bool:
    """Stamp the child's binding as ENDED. True when a stamp was written.

    WHY this exists HERE, and why it has to exist at all: a session with
    NO ``session_ended_at`` is LIVE as far as ``jjlayout.live_session_paths``
    is concerned, and guard (c) of ``store.reclaim_child_workspace`` refuses
    to remove the workspace of a live session. The child's binding row was
    created by ``childspace`` with no end stamp, and nothing in the stop
    flow ever wrote one - so EVERY freshly-stopped child looked live, the
    reclamation was refused as ``live_session_pinned``, and the
    never-raising report thread swallowed the refusal. The tree stayed on
    disk forever and the parent never learned that the reclamation had
    failed. That is the live bug this stamps out.

    WHY the stamp belongs to ``subagent_stop``: the child IS finished by
    contract at this point - the stop hook is the host's own signal that
    the delegation is over. A session that has been stopped is not using
    its tree any more, so saying so in the row is not a guess, it is the
    truth the row was missing.

    WHY the stamp comes BEFORE the reclamation: the guard reads the row.
    A stamp written after the reclaim would be too late - the reclaim has
    already been refused.

    WHY ``db_file`` is a PARAMETER: the stamp must land in the SAME
    database the report went into and the reclaim reads from. A thread
    that resolved the path on its own could write the stamp into a
    different database - and then the guard would still see the child as
    live, and the stamp would leak into the live install. See
    :func:`store.db.pinned_path`.

    Guard hygiene - the stamp is only written when it is both MEANINGFUL
    and NEEDED:

    * the session must have a binding row (no row, nothing to stamp - the
      plugin may have been enabled mid-child);
    * the row must not ALREADY carry an end stamp. The first end wins: a
      later stop (a re-dispatch of the same session id) must not rewrite
      the stamp, because the first one is the truth about when the session
      actually ended.

    Never raises: a failure to stamp costs the reclamation, not the report.
    """
    try:
        sid = str(child_session_id or "")
        if not sid:
            return False
        with store.db.pinned_path(db_file):
            state = store.read_state(sid)
            if not state:
                # No binding row: the plugin was enabled mid-child, or the
                # row was never written. There is nothing to stamp.
                return False
            if state.get("session_ended_at"):
                # The first end wins - never rewrite the stamp.
                return False
            state["session_ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            store.write_state(sid, state)
        return True
    except Exception:
        # The report is already written and delivered. A broken stamp must
        # not turn the child's end into a failure the parent sees - it only
        # means the reclamation may be refused as in-use, and the tree
        # stays for the age-based sweep.
        logger.warning(
            "child report: could not mark the child session ended", exc_info=True
        )
        return False


def _reclaim_child_workspace(
    db_file: str,
    workspace: dict[str, str],
) -> dict[str, Any]:
    """Reclaim the child's workspace right now, on this thread.

    WHY this is HERE and not in the host's hook: the reclamation is a
    subprocess (``jj workspace remove``), and the host runs ``subagent_stop``
    on the CALLER's thread - the parent's own thread. Doing it there would
    block the parent's thread pool and every delegation queued behind it.
    So it happens on the SAME background thread that writes the report.

    WHY the report is written FIRST: the report says whether the tree was
    dirty or clean AT STOP TIME. The reclamation then snapshots that dirty
    tree into a commit and removes it. If the order were reversed, the
    report would describe a tree that no longer exists - and it would say
    "clean" about a tree whose work is sitting in a snapshot commit. The
    order is the whole point: the report is a description of the child's
    end state, the reclamation is what happens to the tree afterwards.

    WHY the kill-switch: an operator who does not trust the snapshot path
    must be able to turn the reclamation off WITHOUT losing the reporting.
    The reporting is what tells a human what happened; the reclamation is
    what acts on it. With the switch off, the tree stays and the age-based
    sweep (or ``hermes ws reclaim``) takes it later.

    WHY the database is PINNED for the duration: the reclamation reads the
    workspace record and writes the reclaim row, and it must do that in
    the SAME database the report went into. Without the pin this thread
    would resolve ``db_path()`` on its own and could write into a
    different database than the one its caller named - the exact leak the
    pinned-path report write exists to prevent.

    Never raises: a failed reclamation must not cost the parent its report.
    """
    path = str(workspace.get("path") or "")
    if not path:
        # No workspace means nothing to reclaim. The report already says
        # "no workspace", and there is nothing to act on.
        return {"ok": False, "reason": "no_workspace", "detail": "the child has no workspace"}
    try:
        enabled = bool(store.settings().get("child_immediate_reclaim", True))
    except Exception:
        # A broken settings read must not silently DISABLE the
        # reclamation - the default is ON, and a configuration error must
        # not change what happens to a child's tree.
        logger.warning(
            "child report: could not read the reclamation setting; reclaiming anyway",
            exc_info=True,
        )
        enabled = True
    if not enabled:
        # The operator turned it off: report only, the tree stays.
        return {
            "ok": False,
            "reason": "disabled",
            "detail": "settings.child_immediate_reclaim is false - report only",
        }
    try:
        with store.db.pinned_path(db_file):
            return store.reclaim_child_workspace(path)
    except Exception:
        # The report is already written and delivered; a reclamation that
        # blew up (jj missing, the tree locked, the store unhappy) must
        # not turn the child's end into a failure the parent sees. The
        # tree stays, and the age-based sweep picks it up later.
        logger.warning(
            "child report: could not reclaim %s; the tree stays for the sweep", path,
            exc_info=True,
        )
        return {"ok": False, "reason": "reclaim_failed", "detail": str(path)}


def _write_report(
    db_file: str,
    parent_session_id: str,
    child_session_id: str,
    workspace: dict[str, str],
    child_status: str,
    child_role: str,
) -> None:
    """Compute the child's end state and store one report row.

    Runs on the report thread, never on the parent's thread.

    WHY ``db_file`` is a PARAMETER and not resolved here: the report thread
    can outlive the code that started it. If it resolved the database path
    itself, it would resolve whatever the environment says at that moment -
    and a test's temporary-database override is undone when the test ends,
    so the thread would land in the LIVE database. That is not theoretical:
    it is exactly how a row once ended up in the live database. The caller
    captures the path while its environment still says what it means.

    WHY the reclamation happens HERE, after the row is written: see
    :func:`_reclaim_child_workspace`. Report first, reclaim second - the
    report describes the tree at stop time, the reclamation then snapshots
    and removes it.
    """
    path = str(workspace.get("path") or "")
    dirty_state = STATE_UNKNOWN
    unpushed: int | None = None
    if path:
        ws = Path(path)
        dirty_state = _dirty_state(ws)
        unpushed = _unpushed_count(ws)
    try:
        store.db.write_child_report_at(
            db_file,
            child_session_id,
            parent_session_id,
            child_role=child_role,
            child_status=child_status,
            workspace_path=path,
            workspace_slug=str(workspace.get("slug") or ""),
            workspace_repo=str(workspace.get("repo") or ""),
            dirty=dirty_state,
            unpushed=unpushed,
        )
    except Exception:
        logger.warning("child report: could not write the report row", exc_info=True)

    # The report is written; NOW the tree can go.
    #
    # WHY the child's session is marked ENDED first: the reclamation's
    # in-use guard reads the bindings table, and a session with no
    # ``session_ended_at`` counts as LIVE. The child's binding never
    # carried one, so without this stamp the guard refused EVERY freshly
    # stopped child as ``live_session_pinned`` and the reclamation silently
    # never fired - the live bug. The child is finished by contract at
    # ``subagent_stop``, so the row says so before the guard runs.
    _mark_child_session_ended(db_file, child_session_id)
    _reclaim_child_workspace(db_file, workspace)


def record_child_stop(
    parent_session_id: str,
    child_session_id: str,
    *,
    child_status: str = "",
    child_role: str = "",
) -> bool:
    """Schedule the child's end report. True when the work was handed off.

    Returns IMMEDIATELY. The dirty and unpushed checks are subprocess calls
    (``jj status``, ``git log``) and the host runs this callback on the
    PARENT's thread, inside the delegation finalization: blocking there
    stalls the parent's thread pool and every delegation queued behind it.

    So the checks go to a short-lived daemon thread and the report row is
    written by that thread. A daemon thread (not a pool worker) so a wedged
    check can never keep the process alive at shutdown.

    WHY the database path is captured HERE: the report thread outlives this
    function, and a thread that resolves the database path on its own
    resolves whatever the environment says when it gets there. See
    :func:`_write_report`.
    """
    try:
        if not parent_session_id or not child_session_id:
            return False
        # Resolve the workspace on THIS thread: it is a database read, not a
        # subprocess, and it must happen before the callback returns so the
        # fallback path (a plugin installed mid-child) still works.
        workspace = _resolve_child_workspace(child_session_id)
        # Capture the database file NOW, while the environment still says
        # which database this session means.
        db_file = str(store.db.db_path())
        thread = threading.Thread(
            target=_write_report,
            args=(
                db_file,
                parent_session_id,
                child_session_id,
                workspace,
                str(child_status or ""),
                str(child_role or ""),
            ),
            name=f"treeps-child-report-{child_session_id[-6:]}",
            daemon=True,
        )
        thread.start()
        return True
    except Exception:
        # The report is a nicety; the delegation's result is not. A failure
        # here must never surface as a failed child.
        logger.warning("child report: could not schedule the child report", exc_info=True)
        return False


# ── the delivery side ────────────────────────────────────────────────────────


def take_pending_reports(session_id: str) -> list[dict[str, Any]]:
    """The parent's pending reports. Reading does NOT delete them.

    WHY the delete is separate: the caller renders first and only then
    clears. A render that raised must leave the report PENDING, so the
    parent's next turn gets it instead of losing it - the plugin being the
    thing that broke must not cost the parent its report.
    """
    if not session_id:
        return []
    try:
        return store.db.read_child_reports(session_id)
    except Exception:
        logger.warning("child report: could not read pending reports", exc_info=True)
        return []


def clear_delivered_reports(reports: list[dict[str, Any]]) -> int:
    """Delete the reports that were actually rendered. Returns the row count.

    WHY the version-scoped delete and not "delete by child id" (P1-2): the
    report row is keyed by the child session id and is REWRITTEN IN PLACE
    when the child is re-dispatched. The parent reads the pending rows,
    renders them, and only then clears - so a report that the child's
    report thread wrote BETWEEN the read and the clear shares the child id
    and was destroyed by an id-only delete, WITHOUT EVER BEING DELIVERED.
    The parent silently lost the child's end state.

    The delete therefore names the exact version it read (report_seq). A
    rewrite bumps the sequence, the delete no longer matches, and the newer
    report survives to the parent's next turn.
    """
    if not reports:
        return 0
    try:
        return store.db.delete_delivered_reports(reports)
    except Exception:
        logger.warning("child report: could not clear delivered reports", exc_info=True)
        return 0


def deliver_reports(session_id: str) -> str:
    """The parent's pending reports, rendered, and then cleared.

    The whole delivery in one call, so the caller cannot render without
    clearing or clear without rendering: a report that is rendered but not
    cleared repeats every turn, and one that is cleared but not rendered is
    lost.

    Returns the EMPTY STRING when there is nothing pending - silent is the
    requirement.
    """
    reports = take_pending_reports(session_id)
    if not reports:
        return ""
    text = contract.render_reports(reports)
    if not text:
        # Nothing renderable: leave the rows pending rather than dropping
        # them on the floor.
        return ""
    clear_delivered_reports(reports)
    return text


def render_reports(reports: list[dict[str, Any]]) -> str:
    """Render the reports as prompt text. Empty string when there is nothing.

    The WORDS live in :mod:`contract` (this module owns the data, that module
    owns the prompt text); this is the thin seam between them.
    """
    return contract.render_reports(reports)
