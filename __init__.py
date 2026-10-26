"""hermes-treeps: one jj workspace per session, enforced.

The gate is the point. Until a session resolves either a repo (``ws_new``) or
"just chatting" (``ws_chatting``), the tools that could touch a filesystem are
refused with a message telling the model exactly what to call. That makes
"which repo are we in?" a decision the user has to make, not something the agent
infers - which is what produced six unattributable pylon directories.

"Just chatting" is deliberately unrestricted: cwd is HERMES_HOME and every tool
stays available. The only thing it changes is the standing instruction to offer a
workspace when the conversation turns into repository work.

The Telegram picker (inline keyboard, taps, /new wiring) lives in
`telegram_picker.py`. The tools live in `tools.py`, the gate in `gate.py`
and the CLI in `cli.py`; this module owns the hooks and the prompt.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from . import childreport, prepare, store
from . import contract as contract
from . import telegram_picker as picker
from .cli import _cli, _register_cli
from .gate import _on_pre_tool_call as _gate_on_pre_tool_call
from .gate import _pin_session_cwd, _session_id, _unpin_session_cwd
from .tools import _opening_prompt, _repo_roots_label, _ws_tools

logger = logging.getLogger(__name__)

# {repos} is filled at register() from the live candidate list: a hardcoded
# example here went stale the first time a repo moved.

_PROMPT_TEMPLATE = """\
## Workspace discipline (hermes-treeps)

Repo work needs a workspace. On an unresolved session, ask in ONE line: the
session title + which repo - {repos}, an external path, a new repo - or just
chatting. Then `ws_new`, or `ws_chatting` for a talk-only session. Children
are bound for you: never bind a child yourself.

A chatting session that drifts into repo work (a path under a repo root, a
clone, a git/jj command, a PR): say "this looks like repo work - want a
workspace?" and offer `ws_new`.
"""

def _on_pre_tool_call(tool_name: str = "", session_id: str = "", **kw: Any) -> dict[str, Any] | None:
    """The pre_tool_call hook. The gate itself lives in gate.py."""
    return _gate_on_pre_tool_call(tool_name=tool_name, session_id=session_id, **kw)

def _on_session_start(session_id: str = "", **kw: Any) -> None:
    """Seed the first turn so the choice prompt is the first thing the user sees.

    On Telegram the picker keyboard is that first thing (see telegram_picker).
    Elsewhere, inject_message() queues user-role input, which needs a
    session_key or an origin - without one there is nowhere to inject, and the
    system-prompt section already asks the question on the next turn.
    """
    try:
        sid = session_id or _session_id(kw)
        picker.note_session_id(sid)
        detail = store.resolve_detail(sid)
        if detail["state"] == "bound":
            # Resumed session: re-pin (a chatting session has no path, and
            # pinning "" would look like a workspace).
            if detail.get("path"):
                _pin_session_cwd(detail["path"], sid=sid)
            else:
                _unpin_session_cwd(sid=sid)
            return  # already resolved once: don't re-ask on every resume
        if detail["state"] == "stale":
            # A dead path must not stay pinned: the session would keep running
            # tools in a directory it no longer owns. HERMES_HOME, not an
            # empty pin (see ws_chatting for why).
            _pin_session_cwd(store.hermes_home(), sid=sid)
            # Do NOT go quiet on a dead path: fall through to the picker/inject
            # path so the user gets a fresh choice instead of a session that
            # silently refuses every tool.
            logger.warning("session %s has a stale binding: %s", sid, detail["reason"])

        # Telegram: an inline keyboard is the whole point - one tap, no typing.
        # `platform` arrives as whatever the agent carries (str, enum, or empty),
        # so normalise before matching: a strict == "telegram" silently skipped
        # this whole branch once already.
        if picker.handle_session_start(sid, **kw):
            return

        session_key = kw.get("session_key") or kw.get("durable_session_key")
        origin = kw.get("origin") or kw.get("source")
        if origin is not None and hasattr(origin, "to_dict"):
            origin = origin.to_dict()
        kwargs: dict[str, Any] = {}
        if session_key:
            kwargs["session_key"] = str(session_key)
        elif isinstance(origin, dict) and origin:
            kwargs["origin"] = origin
        else:
            # No session_key and no origin: nothing to inject into. The system
            # prompt section already asks the question, which is why this path
            # needs no setting any more.
            return
        _CTX.inject_message(_opening_prompt(), role="user", **kwargs)
    except Exception:
        logger.debug("workspace opening prompt skipped", exc_info=True)

def _on_session_finalize(session_id: str = "", **kw: Any) -> None:
    """Stamp the session state at the /new boundary: the gateway fires this with
    the session about to end, so `ws list` can tell a normal ending from a
    vanished session (a true orphan has no state file at all)."""
    try:
        sid = session_id or _session_id(kw)
        state = store.read_state(sid)
        if state is None:
            return  # never resolved a workspace; nothing to stamp
        if isinstance(state, dict):
            state.setdefault("mode", "unknown")
            state["session_ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            store.write_state(sid, state)
    except Exception:
        logger.debug("could not stamp session end", exc_info=True)

# ── child-end reporting (subagent_start / subagent_stop) ────────────────────
#
# WHY these two hooks and not ``on_session_end``: the host's subagent hooks
# are the ONLY ones that know a child ENDED and which parent it belonged to.
# ``on_session_end`` fires when a session is finalized, which is a different
# moment, and it carries no parent/child relationship.
#
# WHY the parent learns about it on its NEXT TURN: the host freezes plugin
# system-prompt sections for the life of a session (they are rendered once
# and cached), so a section registered at start() cannot change mid-session.
# The per-turn channel is ``pre_llm_call``, whose context rides the user
# message of that turn. So the reports go there, and the registered section
# carries them too for the sessions whose prompt is built after the child
# stopped.

def _on_subagent_start(**kw: Any) -> None:
    """Record the child->parent mapping before the child starts working.

    WHY this row exists: ``subagent_stop``'s payload names the child and the
    parent but not the child's WORKSPACE, and the workspace lives in the
    child's binding. Writing the mapping here means the stop side never has
    to reverse-lookup a binding - which is ambiguous when a session id is
    re-dispatched or a child inherited its parent's binding.

    Never raises: the host treats a raising observer as a plugin bug, and a
    missing row is survivable (the stop side falls back to the bindings
    table).
    """
    try:
        parent_sid = str(kw.get("parent_session_id") or "")
        child_sid = str(kw.get("child_session_id") or "")
        childreport.record_child_start(
            parent_sid,
            child_sid,
            child_subagent_id=str(kw.get("child_subagent_id") or ""),
            parent_turn_id=str(kw.get("parent_turn_id") or ""),
            parent_subagent_id=str(kw.get("parent_subagent_id") or ""),
            child_role=str(kw.get("child_role") or ""),
            child_goal=str(kw.get("child_goal") or ""),
        )
    except Exception:
        # A broken reporter must never break the delegation it is watching:
        # the child is about to run, and observers cannot veto work.
        logger.warning("child report: start hook failed", exc_info=True)

def _on_subagent_stop(**kw: Any) -> None:
    """Hand the child's end state to a background thread, then return.

    WHY a background thread: the host runs ``subagent_stop`` on the CALLER
    thread - the parent's own thread, inside the delegation finalization.
    The dirty and unpushed checks are subprocess calls (``jj status``,
    ``git log``), and blocking there stalls the parent's thread pool and
    every delegation queued behind it. The callback therefore returns
    immediately and the report row is written by the daemon thread.

    WHY the fallback to the bindings table: the plugin may have been
    installed (or enabled) while a child was already running, so
    ``subagent_start`` never fired for it and there is no mapping row. The
    child's own binding still names its workspace, and the stop payload still
    names the parent - so the report is still possible.

    WHY the child's workspace is RECLAIMED from here: the child is done by
    contract, so its tree is dead weight, and the reclamation preserves the
    work (jj snapshots the dirty working copy into a commit). It happens on
    the report thread, after the report row is written, so the report still
    describes the tree as it was at stop time. The setting
    ``child_immediate_reclaim`` turns it off; the reporting is unaffected.
    """
    try:
        parent_sid = str(kw.get("parent_session_id") or "")
        child_sid = str(kw.get("child_session_id") or "")
        if not parent_sid:
            # The parent id is the one thing we cannot reconstruct from the
            # bindings table without guessing: a session can be bound to a
            # workspace but that binding does not say who spawned it.
            # Without it there is no session to deliver a report to.
            logger.debug("child report: stop hook without a parent session id")
            return
        childreport.record_child_stop(
            parent_sid,
            child_sid,
            child_status=str(kw.get("child_status") or ""),
            child_role=str(kw.get("child_role") or ""),
        )
    except Exception:
        # The report is a nicety; the delegation's result is not. A failure
        # here must never surface as a failed child.
        logger.warning("child report: stop hook failed", exc_info=True)

def _on_pre_llm_call(**kw: Any) -> Any:
    """Deliver the parent's pending child-end reports, once, on this turn.

    WHY ``pre_llm_call`` and not the system prompt: the host freezes plugin
    system-prompt sections for the life of a session, so a section rendered
    when the session started cannot carry a report about a child that has not
    finished yet. The per-turn channel is this hook: its context rides the
    user message of the turn it fires on.

    SILENT when there is nothing pending: no context, no empty section, no
    filler line. A parent with no finished children pays nothing.
    """
    try:
        # WHY ``_session_id(kw)`` and not ``kw.get("session_id")``: the host
        # names the session differently depending on the hook - the literal
        # ``session_id`` key is not the only one that carries it. The
        # plugin's own helper already knows every spelling
        # (``session_id``, ``session``, ``durable_session_key``, and the
        # ``HERMES_SESSION_ID`` environment). Reading only the one literal
        # key meant a payload keyed any other way silently delivered NO
        # report: the parent never heard what happened to its child.
        sid = _session_id(kw)
        if not sid:
            return None
        text = childreport.deliver_reports(sid)
        if not text:
            return None
        return {"context": text}
    except Exception:
        # Injection is best-effort: a failed report must not break the turn.
        logger.warning("child report: could not inject the pending reports", exc_info=True)
        return None

def _reports_prompt_section(info: Any = None) -> str:
    """The child-end reports section for the system prompt.

    Rendered ONCE per session, like every other plugin section. A session
    whose prompt is built after a child finished gets the reports here; the
    per-turn delivery (``pre_llm_call``) covers the rest.

    Returns the empty string when there is nothing to say, and the host
    skips an empty section entirely.
    """
    try:
        sid = ""
        # WHY the Mapping check and not a dict check: the host passes a
        # read-only ``types.MappingProxyType``, which is NOT a dict. An
        # ``isinstance(info, dict)`` test is False for it, and the section
        # would silently render nothing on every real session - a feature
        # that looks shipped and is not.
        #
        # WHY ``_session_id(info)`` and not ``info.get("session_id")``: the
        # same reason as the per-turn channel - the session id is not
        # always under the literal ``session_id`` key. The helper knows
        # every spelling the host uses.
        if isinstance(info, Mapping):
            sid = _session_id(info)
        if not sid:
            return ""
        return childreport.deliver_reports(sid)
    except Exception:
        logger.warning("child report: could not render the prompt section", exc_info=True)
        return ""

def register(ctx) -> None:
    global _CTX
    _CTX = ctx
    for name, schema, handler in _ws_tools():
        ctx.register_tool(name=name, toolset="workspaces", schema=schema, handler=handler, emoji="🗂️")

    ctx.register_cli_command(
        name="ws",
        help="Workspaces: one per session (new/list/status/retire/reclaim/chatting)",
        setup_fn=_register_cli,
        handler_fn=_cli,
        description=(
            "Session-owned workspaces under <repo_dir>/.workspaces/<repo>/<slug>. "
            "Creates the jj workspace and its record in the plugin database together."
        ),
    )

    repos = store.candidate_repos(limit=6)
    listing = ", ".join(repos) if repos else f"(none found under {_repo_roots_label()})"
    ctx.register_system_prompt_section(
        "workspace-discipline",
        # ONE section, routing only. It ships in every session's prompt - parent
        # and child - so every line here is a tax on every turn. The gate blocks
        # repo tools until a workspace exists; the ask is the only thing it
        # cannot do for the session, so the ask is all this carries.
        _PROMPT_TEMPLATE.format(repos=listing),
        position="after_memory",
    )

    # The child-end reports section. It is a CALLABLE, so it renders with
    # the session's own id and stays silent when the session has no pending
    # reports - the host skips an empty section entirely.
    ctx.register_system_prompt_section(
        "child-end-reports",
        _reports_prompt_section,
        position="after_memory",
    )

    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    # The host's dedicated subagent lifecycle hooks. NOT on_session_end: that
    # one fires when a session is finalized and carries no parent/child
    # relationship, so it cannot tell the parent which workspace its child
    # used.
    ctx.register_hook("subagent_start", _on_subagent_start)
    ctx.register_hook("subagent_stop", _on_subagent_stop)
    # The per-turn delivery channel for the parent's next turn. The host
    # freezes plugin system-prompt sections per session, so a section cannot
    # carry a report about a child that finishes later.
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    picker.register(ctx)
    prepare.register(ctx)
