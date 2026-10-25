"""The workspace gate: what a session may touch before it resolves a workspace.

Pure code motion out of ``__init__.py``. The gate functions and the cwd
pinning they share live here; ``__init__.py`` keeps the hook wiring.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from . import childspace, store
from . import telegram_picker as picker

logger = logging.getLogger(__name__)

def _session_id(kw: dict[str, Any]) -> str:
    for key in ("session_id", "session", "durable_session_key"):
        if kw.get(key):
            return str(kw[key])
    import os

    return os.environ.get("HERMES_SESSION_ID", "") or "unknown"

# ── session cwd pinning ───────────────────────────────────────────────────────
# The gate blocks tools while the session still sits in the gateway's own
# directory. The host can instead PIN a logical cwd per session
# (agent.runtime_cwd._SESSION_CWD, read by resolve_agent_cwd()), which is what
# Claude Code's --worktree and its jj equivalent do: the tool then physically
# runs in the workspace. The gate stays as the second line of defence.

_PIN_LOGGED = {"at": False}

def _turn_key_for_log() -> str:
    """The gateway turn key the terminal layer would read, for the pin log."""
    try:
        from tools.terminal_tool import get_current_session_key

        return str(get_current_session_key() or "")
    except Exception:
        return "<unavailable>"

def _pin_session_cwd(
    path: str | os.PathLike[str] | None, sid: str = "", task_id: str = ""
) -> None:
    """Pin this session's logical cwd to a workspace path. Never raises.

    Two layers on purpose: `agent.runtime_cwd.set_session_cwd` moves what
    context resolvers see, while `tools.terminal_tool.record_session_cwd` moves
    where the terminal tool's shell actually starts (terminal_tool resolves cwd
    from its own per-task record and TERMINAL_CWD, never from the ContextVar).
    The terminal record is keyed by session id and is only written when we know
    it: an empty key collapses to "default", which would re-point EVERY
    session without its own record.
    """
    try:
        from agent.runtime_cwd import set_session_cwd
    except Exception:
        if not _PIN_LOGGED["at"]:
            _PIN_LOGGED["at"] = True
            logger.debug("agent.runtime_cwd unavailable: the gate stays the only barrier")
        return
    try:
        set_session_cwd(str(path or ""))
        # One INFO line per pin: the terminal reads its cwd back under one of
        # three keys (agent session id, gateway turn key, child task id) and
        # the live probe showed the child's shell STILL landing in the parent
        # tree - this line is how we find out which key the write missed.
        logger.info(
            "cwd pin: sid=%r task_id=%r turn_key=%r path=%s",
            sid, task_id, _turn_key_for_log(), path,
        )
        if sid:
            from tools.terminal_tool import get_current_session_key, record_session_cwd

            record_session_cwd(sid, str(path or ""))
            # The terminal resolves its cwd from get_current_session_key() -
            # the gateway's turn key ("agent:...:<chat>") - NOT from the agent
            # session id. Record under both: the sid for the file-tools layer,
            # the turn key for the shell. An empty key collapses to "default",
            # which would re-point EVERY session, so it is only written when
            # the key is known.
            key = get_current_session_key()
            if key:
                record_session_cwd(key, str(path or ""))
            # A delegated child's tools read their cwd under the CHILD's
            # task_id ("subagent-N-..."), which the host seeds from the
            # PARENT's record at spawn. Overwrite it with the child's own
            # workspace - verified live: without this, the child's shell runs
            # in the parent's tree even though its binding points elsewhere.
            if task_id:
                record_session_cwd(task_id, str(path or ""))
    except Exception:
        logger.debug("could not pin the session cwd", exc_info=True)

def _unpin_session_cwd(sid: str = "") -> None:
    try:
        from agent.runtime_cwd import clear_session_cwd
    except Exception:
        return
    try:
        clear_session_cwd()
        if sid:
            from tools.terminal_tool import (
                clear_session_cwd as _clear_terminal_cwd,
            )
            from tools.terminal_tool import (
                get_current_session_key,
            )

            _clear_terminal_cwd(sid)
            key = get_current_session_key()
            if key:
                _clear_terminal_cwd(key)
    except Exception:
        logger.debug("could not clear the session cwd", exc_info=True)

def _pinned_workspace() -> str:
    """The pinned cwd when it really points inside the workspace root, else "".

    Deliberately narrow: a pinned path only counts as a workspace when it lives
    under the configured workspace_root. Anything else (the gateway dir, /, a
    hand-pinned tmp path) must fall through to the gate.
    """
    try:
        from agent.runtime_cwd import scoped_session_cwd

        raw = scoped_session_cwd()
        if not raw:
            return ""
        candidate = Path(raw).resolve()
        root = Path(store.settings()["workspace_root"]).resolve()
    except Exception:
        return ""
    if candidate == root or root not in candidate.parents:
        return ""
    return str(candidate) if candidate.is_dir() else ""

def _caller_is_delegated_child() -> bool:
    """Is the CALLING session a delegated child?

    The same resolution the gate uses: ``is_delegated_child_context()`` is a
    task-local ContextVar set by the host's ``_run_single_child`` and
    inherited by the child's tool dispatch. It is False for every fresh
    session.

    WHY there is no module global here: a global can only ever name ONE
    delegation, so two children in flight would read each other's answer.
    Wave A removed ``_LAST_DELEGATION`` for exactly that reason and it must
    not come back in another shape.
    """
    try:
        from agent.delegation_context import is_delegated_child_context

        return bool(is_delegated_child_context())
    except Exception:
        # No delegation context module (a plain session, or a test): the
        # caller is NOT a child. Failing open here would be wrong - it would
        # let an unclassifiable caller mint a user workspace.
        return False

def _forced_child_shape(sid: str) -> tuple[str, str]:
    """The (purpose, slug) a delegated child's workspace MUST carry.

    WHY this exists: on 2026-10-25 a builder child called
    ``ws_new(repo='hermes-treeps', purpose='feat: picker UI - flat tagged
    list, two-step clone, background bind')`` - it passed its own goal text
    as the purpose, and the handler stamped that goal text into the record.
    The workspace then classified as a USER workspace: immediate reclaim
    refused it, ``ws gc --children`` skipped it, and ``gc_apply`` never
    retired it while the session binding existed. The tree and its record
    leaked until a human ran ``jj workspace forget`` and ``rm`` by hand.

    WHY the shape is FORCED rather than merely validated: an agent-run child
    cannot be trusted to label its own workspaces, and an unlabelled one is
    unreclaimable BY CONSTRUCTION - every reclaim path keys on the child
    prefix. So the caller's purpose AND slug are both ignored and the
    child shape is derived from the same sources the gate uses.

    Returns ``("", "")`` when the caller is not a delegated child, or when
    the child's parent cannot be resolved (the gate blocks such a child
    anyway; minting a user workspace here would be the leak again).
    """
    if not _caller_is_delegated_child():
        return "", ""
    parent_session_id = _resolve_child_parent(sid)
    if not parent_session_id:
        return "", ""
    purpose = f"{store.CHILD_PURPOSE_PREFIX}{parent_session_id}"
    # The slug shape matches childspace.ensure_child_workspace exactly:
    # "<parent slug>-child-<last 6 of the child session id>", sanitised.
    parent_slug = ""
    try:
        parent_state = store.read_state(parent_session_id) or {}
        parent_slug = str(parent_state.get("slug") or "")
    except Exception:
        logger.debug("could not read the parent's binding for %s", parent_session_id, exc_info=True)
    raw_slug = f"{parent_slug}-child-{str(sid)[-6:]}" if parent_slug else f"child-{str(sid)[-6:]}"
    slug = store.sanitize_slug(raw_slug)
    if not slug:
        return "", ""
    return purpose, slug

def _readonly_allowed(tool_name: str, kw: dict[str, Any]) -> bool:
    """Read-only diagnosis is never gated: the gate must not block its own repair.

    The invariant is about *editing* and *running commands* in a tree nobody
    chose. Blocking reads is what turned a stale binding into a dead session: the
    agent could not read the plugin's own source to say why resolution failed,
    and neither could it read the state file that held the answer. Logged so the
    relaxation stays visible.
    """
    if tool_name not in {"read_file", "search_files"}:
        return False
    logger.info("gate: allowing read-only %s while unresolved (repair path)", tool_name)
    return True

def _candidate_listing() -> str:
    """One line of candidate repos for the gate's block messages."""
    try:
        repos = store.candidate_repos(limit=6)
        roots = ", ".join(store.settings()["repo_roots"])
        return ", ".join(repos) if repos else f"(none found under {roots})"
    except Exception as scan_exc:
        return f"(could not scan the configured repo roots: {type(scan_exc).__name__})"

def _stale_block(detail: dict, listing: str) -> dict:
    """The stale-binding block: name the dead path and the two recovery routes."""
    return {
        "action": "block",
        "message": (
            "This session's workspace binding is STALE, not missing: "
            f"{detail['reason']}\n"
            "Tell the user the exact path that disappeared, then call ONE of:\n"
            "  ws_new(path=<new path>, rebind=true) - bind where the tree lives now\n"
            "  ws_chatting()                  - talk-only, if the work is no longer in a checkout\n"
            "  ws_doctor(forget=true)         - drop the dead record ONLY, then pick one of the two\n"
            "                                   above: forgetting alone leaves the session unbound\n"
            "                                   and still gated\n"
            f"Candidate repos: {listing}"
        ),
    }

def _resolve_child_parent(child_sid: str) -> str:
    """The parent session id of THIS child, from per-child channels only.

    WHY this exists and WHY there is no global any more: the plugin used to
    keep the sid of "the most recent delegate_task dispatch" in a module
    global and read it back when the child's first gated call arrived. Two
    concurrent delegations overwrote that global, so child A resolved its
    parent as B - and A then built its own workspace inside B's repo, with
    B's parent stamped as ``inherited_from``. A module global can only ever
    name ONE parent, so it is structurally incapable of answering "who is
    MY parent" for more than one live child.

    The channels used here are all keyed by the CHILD's own session id, so
    two children can never see each other's parent:

    1. ``child_runs`` - written by the ``subagent_start`` hook, whose payload
       names BOTH the parent and the child. The host fires it synchronously
       when the child agent is constructed and BEFORE the child runs its
       first tool call, so the row is there when the child asks.
    2. the child's own binding row - ``parent_session_id``, written when the
       child's workspace is created. It survives a restart and a re-dispatch.

    WHY NOT the recency tracker: ``picker.last_session()`` is process-global
    too. It is only consulted as a LAST resort, and only when the tracker
    names a session that is the child's own recorded parent - never as a
    guess about who dispatched this child. A child whose parent cannot be
    resolved is BLOCKED, not pointed at a stranger's workspace.
    """
    if not child_sid:
        return ""
    # 1. the child_runs row (subagent_start) - the authoritative per-child map
    try:
        run = store.db.read_child_run(child_sid)
    except Exception:
        logger.debug("could not read the child run row for %s", child_sid, exc_info=True)
        run = None
    if run:
        parent = str(run.get("parent_session_id") or "")
        if parent:
            return parent
    # 2. the child's own binding row
    try:
        state = store.read_state(child_sid) or {}
    except Exception:
        logger.debug("could not read the child's binding for %s", child_sid, exc_info=True)
        state = {}
    for key in ("parent_session_id", "inherited_from"):
        parent = str(state.get(key) or "")
        if parent and parent != child_sid:
            return parent
    # 3. the recency tracker - ONLY when it names the child's own recorded
    #    parent. Never a guess: a stranger's workspace is worse than a block.
    tracked = picker.last_session(max_age_s=600)
    if tracked and tracked == str(state.get("parent_session_id") or ""):
        return tracked
    return ""

def _self_heal_child_record(sid: str, parent_session_id: str) -> bool:
    """Restamp THIS child's own workspace record to the child marker.

    LAYER 2 of the "a child may not mint a user-class workspace" fix.

    WHY this exists: before layer 1, a delegated child could call ws_new with
    its own goal text as the purpose, and the record was stamped with that
    goal text. The workspace then classified as a USER workspace: immediate
    reclaim refused it, ``ws gc --children`` skipped it, and ``gc_apply``
    never retired it while the session binding existed. The tree and its
    record leaked until a human intervened - which is exactly what we had to
    do on 2026-10-25 (``jj workspace forget`` + ``rm`` by hand).

    This is the SELF-HEAL for records that leaked BEFORE the fix: on the
    child's next gated call the record is restamped, so it rejoins the
    reclaim system instead of leaking forever.

    It is ONE UPDATE and it is IDEMPOTENT: a record that already carries the
    child prefix is left alone, so the hot path does not rewrite the record
    on every gated call.

    A user's workspace is never touched: the restamp only happens when the
    session resolves as a delegated child AND the child's own session id is
    in the workspace record's session list - i.e. this child OWNS the
    workspace. A workspace the child merely INHERITED from its parent (the
    chatting/external fallback) belongs to the parent and keeps the parent's
    classification.
    """
    if not sid or not parent_session_id:
        return False
    wanted = f"{store.CHILD_PURPOSE_PREFIX}{parent_session_id}"
    try:
        state = store.read_state(sid) or {}
        path = str(state.get("path") or "")
        if not path or state.get("mode") != "repo":
            return False
        record = store.read_record(Path(path))
        if record is None:
            return False
        if str(record.get("purpose") or "") == wanted:
            return False  # already correct: idempotent no-op
        if str(record.get("purpose") or "").startswith(store.CHILD_PURPOSE_PREFIX):
            # It is SOME child's workspace. Do not re-stamp it: the prefix
            # names a different parent, and rewriting it would move the
            # workspace into another parent's reclaim bucket.
            return False
        # The child must OWN the workspace. A workspace the child merely
        # inherited from its parent is the parent's, and restamping it would
        # put the parent's workspace into the child reclaim bucket.
        sessions = [str(s) for s in (record.get("sessions") or [])]
        if sid not in sessions:
            return False
        record["purpose"] = wanted
        store.write_record(Path(path), record)
        logger.info(
            "gate: restamped leaked workspace %s to the child marker (%s)", path, wanted
        )
        return True
    except Exception:
        # The self-heal is best-effort: a broken bookkeeping path must never
        # gate-block the child.
        logger.debug("could not self-heal the child record for %s", sid, exc_info=True)
        return False

def _on_pre_tool_call(tool_name: str = "", session_id: str = "", **kw: Any) -> dict[str, Any] | None:
    """Block repo-touching tools until the session resolves a workspace."""
    # WHY the recency tracker is NOT read here any more: it is process-global,
    # so it can only ever name ONE session, and using it as a parent source
    # is what let two concurrent delegations cross-inherit. A child's parent
    # is resolved per child, in _resolve_child_parent().
    picker.note_session_id(session_id)
    try:
        if tool_name not in store.GATED_TOOLS:
            return None
        sid = session_id or _session_id(kw)
        bound = store.resolve(sid)
        if bound:
            # A resumed child keeps its workspace and keeps proving life: the
            # heartbeat is written on EVERY gated call, not only on the first
            # one, so a long-running child is never mistaken for an orphan.
            try:
                state = store.read_state(sid) or {}
                if str(state.get("purpose") or "").startswith(store.CHILD_PURPOSE_PREFIX):
                    store.record_child_heartbeat(sid)
                else:
                    # LAYER 2 (self-heal), bound branch: a record this child
                    # created BEFORE the fix carries goal text instead of the
                    # child marker, so it classifies as a USER workspace -
                    # and a bound USER workspace is exactly what gc_apply
                    # never retires. Restamp it here, on the child's next
                    # gated call, so the already-leaked records rejoin the
                    # reclaim system instead of leaking forever.
                    if _caller_is_delegated_child():
                        _self_heal_child_record(sid, _resolve_child_parent(sid))
            except Exception:
                # Broken heartbeat bookkeeping must not gate-block the child.
                pass
            # Resumed after a gateway restart: the runtime pin is gone and
            # on_session_start does not re-fire for a resumed session, so the
            # shell silently falls back to the gateway directory. Re-arm the
            # pin lazily on the first gated call; the pin-present check makes
            # this a no-op for every later call.
            if not _pinned_workspace():
                if bound.get("mode") == "chatting":
                    _pin_session_cwd(store.hermes_home(), sid=sid)
                else:
                    _pin_session_cwd(str(bound.get("path") or ""), sid=sid)
            return None
        # AIRTIGHT inheritance: only a real delegate_task child may inherit.
        # `is_delegated_child_context()` is a task-local ContextVar set by
        # `_run_single_child` and inherited by the child's tool dispatch - it is
        # False for every fresh session, so a new session can NEVER silently
        # grab another session's workspace. (The recency tracker alone was not
        # airtight: two concurrent delegations could cross-inherit.)
        try:
            from agent.delegation_context import is_delegated_child_context

            is_child = is_delegated_child_context()
        except Exception:
            is_child = False
        # NOTE: there is deliberately NO "recent delegation dispatch" fallback
        # here any more. The old fallback set is_child=True for ANY unresolved
        # session while the delegation pointer was younger than 600 s, so an
        # unrelated pre-existing session inherited another session's workspace
        # (verified live). The upstream bug it worked around (#96483,
        # `_run_single_child` not entering delegated_child_context) is FIXED in
        # the installed Hermes: children run inside delegated_child_context(),
        # so is_delegated_child_context() is authoritative.
        if is_child:
            # WHICH parent: resolved from per-child channels only (the
            # child's own child_runs row, its own binding row). The old
            # module global could name only ONE parent, so two concurrent
            # delegations made the second child inherit the first child's
            # parent - and build its workspace in the WRONG repo.
            parent = _resolve_child_parent(sid) or ""
            if not parent:
                # No per-child record of who dispatched this child. Guessing
                # (the recency tracker) is what let two children cross-
                # inherit, so the child is refused instead: a blocked child
                # is recoverable, a child in a stranger's tree is not.
                return {
                    "action": "block",
                    "message": (
                        "Delegated child, but this session's parent could not be "
                        "identified. The plugin records the child->parent mapping "
                        "at dispatch (subagent_start); that record is missing, so "
                        "the child is not allowed to guess a parent's workspace.\n"
                        "Report this to the user: re-dispatch the task, or run it "
                        "directly in the parent session."
                    ),
                }
            parent_state = store.read_state(parent) or {} if parent else {}
            # LAYER 2 (self-heal): a record this child created BEFORE the
            # fix carries goal text instead of the child marker, so it
            # classifies as a USER workspace and no reclaim path ever
            # touches it. Restamp it here - one idempotent UPDATE - so it
            # rejoins the reclaim system on the child's next gated call.
            _self_heal_child_record(sid, parent)
            if parent_state.get("mode") not in {"repo", "chatting", "external"}:
                return {
                    "action": "block",
                    "message": (
                        "Delegated child, but the parent session has no workspace binding. "
                        "Report this to the user: the parent must run ws_new/ws_chatting first."
                    ),
                }
            # Isolation first: give the child its OWN workspace in the parent's
            # repo. Two children in one tree is the collision that let an agent
            # in this repo edit another agent's uncommitted work. Inheritance is
            # only the fallback, for a parent with no repo to branch from.
            made = childspace.ensure_child_workspace(sid, parent_state)
            if made.get("ok"):
                _pin_session_cwd(
                    str(made.get("path") or ""), sid=sid, task_id=str(kw.get("task_id") or "")
                )
                try:
                    state = store.read_state(sid) or {}
                    purpose = str(state.get("purpose") or "")
                    if not purpose:
                        record = store.read_record(Path(str(made.get("path") or ""))) or {}
                        purpose = str(record.get("purpose") or "")
                    if purpose.startswith(store.CHILD_PURPOSE_PREFIX):
                        store.record_child_heartbeat(sid)
                except Exception:
                    # Broken heartbeat bookkeeping must not gate-block the child.
                    pass
                logger.info(
                    "gate: delegated child %s got its own workspace %s", sid, made.get("slug")
                )
                return None
            logger.debug("gate: no child workspace for %s (%s)", sid, made.get("reason"))
            if parent_state.get("mode") == "repo":
                # A repo parent with a failed childspace (git-only repo, jj
                # error) must NOT fall through to inheritance: that puts the
                # child into the PARENT'S tree - the exact two-agents-one-tree
                # collision this branch exists to prevent. Inheritance is only
                # for parents with no repo to branch from (chatting/external).
                return {
                    "action": "block",
                    "message": (
                        "Delegated child could not get its own workspace: "
                        f"{made.get('reason')}. Repo-touching tools are blocked.\n"
                        "Run the child's task directly in this session instead, or "
                        "convert the repo to jj (jj git init --colocate) and re-dispatch."
                    ),
                }
            # Non-destructive inheritance: the child's own record wins. If the
            # child already has a record (it resolved its own workspace, or it
            # was written by an earlier hook), it is NOT overwritten - the old
            # code copied the parent's dict wholesale and clobbered it.
            if store.read_state(sid) is None:
                inherited = {
                    k: v for k, v in parent_state.items()
                    if k not in {"session_id", "inherited_from", "inherited_at", "parent_session_id"}
                }
                inherited["inherited_from"] = parent
                # The explicit parent link, so the NEXT gated call of this
                # child (or of a re-dispatched one) can resolve its parent
                # from its own row instead of from any shared/global state.
                inherited["parent_session_id"] = parent
                inherited["inherited_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                store.write_state(sid, inherited)
            logger.info("gate: delegated child %s inherits binding of %s", sid, parent)
            return None
        # The state is resolved ONCE here and reused by every branch below.
        detail = store.resolve_detail(sid)
        if detail["state"] == "stale":
            # A stale binding must warn EVEN WHEN the cwd is pinned: the pin
            # means the shell runs in a workspace, but the workspace it NAMES
            # is dead - the user is never told if the pin floor wins.
            listing = _candidate_listing()
            return _stale_block(detail, listing)
        if detail["state"] == "unbound" and _readonly_allowed(tool_name, kw):
            return None
        # Second line of defence: a pinned cwd inside the workspace root means
        # the tool physically runs in a workspace, whatever the database read
        # says - so a corrupted or unreachable binding row cannot fail open.
        # Unbound (not stale) only: a stale session was already warned above.
        if _pinned_workspace():
            return None
        try:
            repos = store.candidate_repos(limit=6)
            roots = ", ".join(store.settings()["repo_roots"])
            listing = ", ".join(repos) if repos else f"(none found under {roots})"
        except Exception as scan_exc:
            listing = f"(could not scan the configured repo roots: {type(scan_exc).__name__})"
        return {
            "action": "block",
            "message": (
                "No workspace resolved for this session. Ask the user, then call ONE of:\n"
                f"  ws_new(repo, slug)  - repo work; creates {store.workspaces_base()}/<repo>/<slug>\n"
                "  ws_chatting()       - talk-only session, no repo, full tools\n"
                "  repo on GitHub/Forgejo but not on disk: ws_new(repo, slug, create_repo=true) - clone it, after the user approves\n"
                f"Candidate repos: {listing}\n"
                "Ask in one short line offering these repos or 'just chatting'."
            ),
        }
    except Exception as exc:
        logger.exception("workspace gate failed")
        # A pinned cwd inside the workspace root is independent evidence the
        # session sits in a workspace - the SAME evidence the normal path
        # accepts, so the error path accepts it too. Without this, a session
        # whose cwd is a real workspace is blocked on a db hiccup but allowed
        # the moment the db recovers - the two branches disagreeing about what
        # a pinned cwd proves.
        if _pinned_workspace():
            return None
        # Fail CLOSED for the tools this gate exists to stop. Failing open meant
        # a locked or corrupt database silently allowed `terminal` - the exact
        # hole the audits found. Repair stays possible: read-only tools are
        # allowed, and ws_new / ws_chatting / ws_doctor are not gated tools.
        if tool_name in store.GATED_TOOLS and not _readonly_allowed(tool_name, kw):
            return {
                "action": "block",
                "message": (
                    "Workspace gate unavailable: the plugin could not read its own state.\n"
                    f"  error: {type(exc).__name__}: {exc}\n"
                    "Repo-touching tools are blocked until it recovers.\n"
                    "  ws_doctor   - show the binding and the error\n"
                    "  ws_chatting - declare a talk-only session, which needs no state read\n"
                    "Report the error above to the user; do not retry the same tool."
                ),
            }
        return None
