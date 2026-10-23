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
`telegram_picker.py`; this module owns tools, the gate, the CLI and the prompt.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from . import store
from . import telegram_picker as picker

logger = logging.getLogger(__name__)

# sid of the most recent delegate_task DISPATCH (the parent), and when. Only
# dispatches move it - ordinary tool calls from other sessions cannot clobber.
_LAST_DELEGATION = {"sid": "", "at": 0.0}

# {repos} is filled at register() from the live candidate list: a hardcoded
# example here went stale the first time a repo moved.
_PROMPT_TEMPLATE = """\
## Workspace discipline (hermes-treeps)

This session starts unresolved. Repository work - anything that reads, writes or
runs commands in a checkout - requires a workspace first:

- `ws_new(repo, slug)` creates `/Git/hermes/workspaces/<repo>/<slug>` and binds
  this session to it. Use for any real work: code, docs, config, git. Omit slug
  and it defaults to `<repo>-<last 6 of the session id>` - the path is derived
  from the id, never from the session title, because titles get renamed.
- `ws_chatting()` declares a talk-only session: cwd stays HERMES_HOME and all
  tools stay available, but nothing repo-shaped should be edited.

On the first turn, ask BOTH questions in a single line (never two messages, never
a menu): the session TITLE (required - it becomes the workspace name), and which
repo - or just chatting. Example: "Session title? (it names the workspace) - and
which repo: {repos}, an external path, a new repo, or just chatting?"

If a chatting session drifts into repository work (a path under /Git, a clone, a
git/jj command, a PR), say one line: "this looks like repo work - want a
workspace for it?" and offer `ws_new <repo> <slug>`. Do not silently start
editing.

Never create a workspace without the user's choice, and never work in another
session's workspace.
"""


def _ws_tools() -> tuple[tuple[str, dict[str, Any], Any], ...]:
    new_schema = {
        "name": "ws_new",
        "description": (
            "Create this session's own workspace for a repo and bind the session to it. "
            "Path: /Git/hermes/workspaces/<repo>/<slug>. Call this before repo work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "repo name under /Git, e.g. hermes-pylon"},
                "slug": {"type": "string", "description": "session name, lowercase - use the session TITLE slugified"},
                "title": {"type": "string", "description": "the session title the user gave (required when known)"},
                "purpose": {"type": "string", "description": "one line: what this session is for"},
                "rebind": {"type": "boolean", "description": "true to move this session from its current workspace to the new one"},
                "path": {"type": "string", "description": "absolute path to an EXISTING repo outside /Git - binds it directly, no checkout"},
                "create_repo": {"type": "boolean", "description": "create /Git/<repo> first (brand-new project)"},
            },
            "required": ["slug"],
        },
    }
    chat_schema = {
        "name": "ws_chatting",
        "description": "Declare this a talk-only session: no workspace, cwd stays HERMES_HOME, all tools available.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string", "description": "the session title the user gave"}},
        },
    }
    rename_schema = {
        "name": "ws_rename",
        "description": "Rename this session's workspace (moves the directory, fixes records). Use when the user gives a title after a picker tap.",
        "parameters": {
            "type": "object",
            "properties": {
                "new_slug": {"type": "string", "description": "new name, lowercase"},
                "slug": {"type": "string", "description": "workspace to rename; defaults to this session's"},
            },
            "required": ["new_slug"],
        },
    }
    status_schema = {
        "name": "ws_status",
        "description": "Show this session's workspace binding and every workspace under the root.",
        "parameters": {"type": "object", "properties": {}},
    }
    doctor_schema = {
        "name": "ws_doctor",
        "description": (
            "Diagnose why workspace resolution failed: binding state (unbound / stale / bound), "
            "the recorded path and whether it still exists, candidate repos, and whether jj/gh "
            "are resolvable. Set forget=true to drop a dead record."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "forget": {"type": "boolean", "description": "drop this session's record, making it unbound"},
            },
        },
    }
    return (
        ("ws_new", new_schema, _tool_new),
        ("ws_chatting", chat_schema, _tool_chatting),
        ("ws_rename", rename_schema, _tool_rename),
        ("ws_status", status_schema, _tool_status),
        ("ws_doctor", doctor_schema, _tool_doctor),
    )


def _session_id(kw: dict[str, Any]) -> str:
    for key in ("session_id", "session", "durable_session_key"):
        if kw.get(key):
            return str(kw[key])
    import os

    return os.environ.get("HERMES_SESSION_ID", "") or "unknown"


def _tool_new(args: dict[str, Any], **kw: Any) -> str:
    try:
        sid = _session_id(kw)
        title = str(args.get("title") or "").strip()
        slug = str(args.get("slug") or "").strip()

        existing = store.read_state(sid)
        if (
            existing
            and existing.get("mode") == "repo"
            and not args.get("rebind")
            and str(existing.get("slug")) != slug
        ):
            # 1:1 by default; re-binding is explicit, never a silent append.
            return json.dumps({
                "ok": False,
                "needs_confirmation": True,
                "error": (
                    f"this session is already bound to {existing.get('repo')}/{existing.get('slug')}. "
                    f"Rebind to {args.get('repo') or args.get('path')}? The old workspace stays, just "
                    "detached. Ask the user, then call ws_new again with rebind=true."
                ),
            })

        if args.get("path"):
            state = store.bind_external(str(args.get("path")), sid, title=title)
            return json.dumps({"ok": True, "mode": "external", "path": state["path"]})

        repo = str(args.get("repo") or "").strip()
        if not repo:
            return json.dumps({"ok": False, "error": "ws_new needs repo (or path for an external repo)"})
        if not slug:
            # Stable naming: the path follows the session id, never the title.
            # Titles get renamed constantly and there is no rename hook to keep a
            # path in sync; an id-derived slug needs no syncing at all.
            slug = f"{repo}-{sid[-6:]}" if sid and sid != "unknown" else repo
        if args.get("create_repo"):
            store.create_repo(repo)
        record = store.create(repo, slug, purpose=str(args.get("purpose") or ""))
        store.bind_session(store.workspace_path(repo, slug), sid)
        state = {"mode": "repo", "repo": repo, "slug": slug, "path": record["path"]}
        if title:
            state["title"] = title
        store.write_state(sid, state)
        return json.dumps({"ok": True, "path": record["path"], "vcs": record["vcs"]})
    except store.WorkspaceError as e:
        return json.dumps({"ok": False, "error": str(e)})
    except Exception as e:  # never raise into the tool dispatcher
        logger.exception("ws_new failed")
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _tool_rename(args: dict[str, Any], **kw: Any) -> str:
    try:
        sid = _session_id(kw)
        slug = str(args.get("slug") or "").strip()
        if not slug:
            state = store.read_state(sid) or {}
            slug = str(state.get("slug") or "")
        if not slug:
            return json.dumps({"ok": False, "error": "no workspace bound to this session"})
        return json.dumps({"ok": True, "result": store.rename(slug, str(args.get("new_slug") or "").strip())})
    except store.WorkspaceError as e:
        return json.dumps({"ok": False, "error": str(e)})
    except Exception as e:
        logger.exception("ws_rename failed")
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _tool_chatting(args: dict[str, Any], **kw: Any) -> str:
    try:
        store.mark_chatting(_session_id(kw), title=str(args.get("title") or ""))
        return json.dumps({"ok": True, "mode": "chatting"})
    except Exception as e:
        logger.exception("ws_chatting failed")
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _tool_status(args: dict[str, Any], **kw: Any) -> str:
    session_id = _session_id(kw)
    rows = []
    for item in store.list_workspaces():
        record = item["record"] or {}
        rows.append(
            {
                "repo": item["repo"],
                "slug": item["slug"],
                "path": item["path"],
                "orphan": item["orphan"],
                "last_synced": record.get("last_synced"),
            }
        )
    return json.dumps(
        {"session_id": session_id, "resolved": store.resolve(session_id), "workspaces": rows},
        indent=2,
    )


def _tool_doctor(args: dict[str, Any], **kw: Any) -> str:
    """Answer "why won't this session resolve?" from one place, without shell.

    Deliberately usable while unresolved: it is the repair tool for the gate, so
    it must never be behind the gate.
    """
    sid = _session_id(kw)
    try:
        detail = store.resolve_detail(sid)
        out: dict[str, Any] = {
            "session_id": sid,
            "state": detail["state"],
            "reason": detail["reason"],
            "path": detail["path"],
            "path_exists": bool(detail["path"]) and Path(str(detail["path"])).exists(),
            "record": detail["record"],
            "candidate_repos": store.candidate_repos(limit=8),
            "tools": {"jj": shutil.which("jj"), "gh": shutil.which("gh")},
            "state_dir": str(store.state_dir()),
        }
        if args.get("forget"):
            out["forgot"] = store.forget(sid)
        return json.dumps(out, indent=2, default=str)
    except Exception as e:
        logger.exception("ws_doctor failed")
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


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


_TITLE_SYNC = {"at": 0.0, "interval": 60.0}


def _on_pre_gateway_dispatch(**kw: Any) -> None:
    """Mirror a renamed session's title into its records, at most once a minute.

    Lazy by necessity: this build has no session-rename hook, and polling the
    state DB on every message would be wasteful. Metadata only - the workspace
    path is id-derived and never follows the title.
    """
    now = time.time()
    if now - _TITLE_SYNC["at"] < _TITLE_SYNC["interval"]:
        return None
    sid = _session_id(kw) or picker.last_session(max_age_s=600)
    if not sid or sid == "unknown":
        return None
    _TITLE_SYNC["at"] = now
    try:
        result = store.sync_title(sid)
        if result.get("changed"):
            logger.info("title synced for %s -> %r", sid, result.get("title"))
    except Exception:
        logger.debug("title sync failed for %s", sid, exc_info=True)
    return None


def _on_pre_tool_call(tool_name: str = "", session_id: str = "", **kw: Any) -> dict[str, Any] | None:
    """Block repo-touching tools until the session resolves a workspace."""
    # Read the recency tracker BEFORE noting this call's session id: the tracker
    # must still hold the PARENT here - noting first would overwrite it with the
    # child's own (unresolved) id and inheritance could never fire.
    parent = picker.last_session(max_age_s=600)
    picker.note_session_id(session_id)
    # Only delegate_task dispatches may set the parent pointer: a plain tool
    # call from ANY other session used to clobber it before the child's first
    # gated call (the test suite caught exactly that).
    dispatch_sid = session_id or _session_id(kw)
    if tool_name == "delegate_task" and dispatch_sid:
        _LAST_DELEGATION["sid"] = dispatch_sid
        _LAST_DELEGATION["at"] = time.time()
    if _LAST_DELEGATION["sid"] and time.time() - _LAST_DELEGATION["at"] > 600:
        _LAST_DELEGATION["sid"] = ""
    try:
        if tool_name not in store.GATED_TOOLS:
            return None
        sid = session_id or _session_id(kw)
        if store.resolve(sid):
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
        if not is_child and _LAST_DELEGATION["sid"] and time.time() - _LAST_DELEGATION["at"] < 600:
            # FALLBACK for batch children: upstream #96483 - `_run_single_child`
            # never enters delegated_child_context(), so batch-spawned children
            # look like ordinary sessions. Treat "unresolved + recent
            # delegation dispatch" as a batch child. REMOVE when upstream fixes
            # #96483; until then a fresh session created within the 10-minute
            # pointer window inherits too (the known residue).
            is_child = True
        if is_child:
            # Parent = the dispatching session (delegation pointer first -
            # ordinary tool calls cannot move it - then the general recency
            # tracker). The durable copy written here covers the child's life.
            parent = _LAST_DELEGATION["sid"] or parent
            parent_state = store.read_state(parent) or {} if parent else {}
            if parent_state.get("mode") not in {"repo", "chatting", "external"}:
                return {
                    "action": "block",
                    "message": (
                        "Delegated child, but the parent session has no workspace binding. "
                        "Report this to the user: the parent must run ws_new/ws_chatting first."
                    ),
                }
            parent_state["inherited_from"] = parent
            parent_state["inherited_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            store.write_state(sid, parent_state)
            logger.info("gate: delegated child %s inherits binding of %s", sid, parent)
            return None
        try:
            repos = store.candidate_repos(limit=6)
            listing = ", ".join(repos) if repos else "(none found under /Git)"
        except Exception:
            listing = "(could not scan /Git)"
        detail = store.resolve_detail(sid)
        if detail["state"] == "unbound" and _readonly_allowed(tool_name, kw):
            return None
        if detail["state"] == "stale":
            # A dead path is NOT "no choice yet". Say which tree vanished: the
            # user can usually point straight at where it moved.
            return {
                "action": "block",
                "message": (
                    "This session's workspace binding is STALE, not missing: "
                    f"{detail['reason']}\n"
                    "Tell the user the exact path that disappeared, then call ONE of:\n"
                    "  ws_doctor()                    - full diagnosis (state, record, candidates, tool paths)\n"
                    "  ws_doctor(forget=true)         - drop the dead record; the session starts unbound\n"
                    "  ws_new(path=<new path>, rebind=true) - bind where the tree lives now\n"
                    "  ws_chatting()                  - talk-only, if the work is no longer in a checkout\n"
                    f"Candidate repos: {listing}"
                ),
            }
        return {
            "action": "block",
            "message": (
                "No workspace resolved for this session. Ask the user, then call ONE of:\n"
                "  ws_new(repo, slug)  - repo work; creates /Git/hermes/workspaces/<repo>/<slug>\n"
                "  ws_chatting()       - talk-only session, no repo, full tools\n"
                f"Candidate repos: {listing}\n"
                "Ask in one short line offering these repos or 'just chatting'."
            ),
        }
    except Exception:
        logger.exception("workspace gate failed")
        return None  # fail open: a broken plugin must not brick the agent


def _inject_first_message_enabled() -> bool:
    """Opt-in: let the plugin open gateway sessions with the question itself.

    Default off. Injection runs a turn, so on Telegram it lands right after the
    /new greeting - the question twice. The prompt section already puts the
    question in that greeting, which is why this is a switch, not the default.
    """
    import os

    env = os.environ.get("HERMES_WS_INJECT_FIRST")
    if env is not None:
        return env.strip().lower() in {"1", "true", "yes", "on"}
    try:
        from hermes_cli.config import load_config_readonly  # type: ignore

        cfg = load_config_readonly() or {}
        entry = ((cfg.get("plugins") or {}).get("entries") or {}).get("hermes-treeps") or {}
        return bool(entry.get("inject_first_message"))
    except Exception:
        return False


def _opening_prompt() -> str:
    repos = store.candidate_repos()
    listing = ", ".join(repos) if repos else "(none found under /Git)"
    return (
        "New session, no workspace yet. Ask me exactly one line: which repo to work in "
        f"({listing}) - or 'just chatting' for a talk-only session with no checkout. "
        "Wait for my answer before doing anything else; do not start work, do not guess a repo."
    )


def _on_session_start(session_id: str = "", **kw: Any) -> None:
    """Seed the first turn so the choice prompt is the first thing the user sees.

    On Telegram the picker keyboard is that first thing (see telegram_picker).
    Elsewhere, inject_message() queues user-role input so the session opens with
    the question itself - gateway sessions additionally need a session_key/origin
    plus `plugins.entries.hermes-treeps.inject_first_message: true`.
    """
    try:
        sid = session_id or _session_id(kw)
        picker.note_session_id(sid)
        detail = store.resolve_detail(sid)
        if detail["state"] == "bound":
            return  # already resolved once: don't re-ask on every resume
        if detail["state"] == "stale":
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
            if not _inject_first_message_enabled():
                return
            key = store.session_key_for(sid)
            if not key:
                return
            kwargs["session_key"] = key
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


def _register_cli(subparser) -> None:
    sub = subparser.add_subparsers(dest="ws_action")

    p_new = sub.add_parser("new", help="create a workspace + session record")
    p_new.add_argument("repo")
    p_new.add_argument("slug")
    p_new.add_argument("--purpose", default="")

    sub.add_parser("list", help="list workspaces and their records")

    p_status = sub.add_parser("status", help="detail for one workspace")
    p_status.add_argument("slug")

    p_retire = sub.add_parser("retire", help="archive + forget (refuses open PRs / dirty / unpushed; --force for unknowns)")
    p_retire.add_argument("slug")
    p_retire.add_argument("--force", action="store_true", help="retire even when PR/push checks could not run")
    p_retire.add_argument("--ultraforce", action="store_true", help="retire despite open PRs or a dirty tree (archive is still kept)")

    p_gc = sub.add_parser("gc", help="classify reclaimable workspaces (dry-run unless --apply)")
    p_gc.add_argument("--apply", action="store_true", help="retire the tier1 set (ended + clean + pushed + no open PRs)")

    p_chat = sub.add_parser("chatting", help="mark a session as talk-only")
    p_chat.add_argument("--session", default="")

    subparser.set_defaults(ws_action="list")


def _cli(args) -> int:
    action = getattr(args, "ws_action", "list") or "list"
    try:
        if action == "new":
            record = store.create(args.repo, args.slug, purpose=args.purpose)
            print(record["path"])
            return 0
        if action == "list":
            items = store.list_workspaces()
            if not items:
                print(f"no workspaces under {store.root_dir()}")
                return 0
            for item in items:
                record = item["record"] or {}
                flags = []
                if item["orphan"]:
                    flags.append("ORPHAN(no session.json)")
                else:
                    # A workspace whose every session was finalised is done, not
                    # orphaned - distinguish that from a vanished session.
                    ends = []
                    for sid in record.get("sessions") or []:
                        st = store.read_state(str(sid)) or {}
                        if st.get("session_ended_at"):
                            ends.append(str(st["session_ended_at"]))
                    if ends and len(ends) == len(record.get("sessions") or []):
                        flags.append(f"session ended {max(ends)}")
                    elif record.get("last_synced"):
                        flags.append(f"synced {record['last_synced']}")
                print(f"{item['repo']}/{item['slug']}  {item['path']}  {' '.join(flags)}".rstrip())
            return 0
        if action == "status":
            ws = store.find_workspace(args.slug)
            record = store.read_record(ws)
            prs, gh_ok = store.open_prs(ws)
            print(json.dumps({"path": str(ws), "record": record, "open_prs": prs, "pr_check_ok": gh_ok}, indent=2, default=str))
            return 0
        if action == "retire":
            print(store.retire(
                args.slug,
                force=getattr(args, "force", False),
                ultraforce=getattr(args, "ultraforce", False),
            ))
            return 0
        if action == "gc":
            tiers = store.gc_tiers()
            titles = {
                "tier1": "safe to retire (session ended, clean, pushed, no open PRs)",
                "tier2": "needs a look first (unverified, orphans, open PRs, or dirty)",
                "tier3": "in use (session still live)",
            }
            total = sum(len(v) for v in tiers.values())
            if total == 0:
                print("no workspaces found - nothing to reclaim")
                return 0
            for name in ("tier1", "tier2", "tier3"):
                rows = tiers[name]
                print(f"\n{titles[name]}: {len(rows)}")
                for row in rows:
                    print(f"  {row['repo']}/{row['slug']}")
                    print(f"    {row['path']}")
                    print(f"    why: {row['why']}")
            if getattr(args, "apply", False):
                done = store.gc_apply()
                print(f"\nretired: {len(done)}" if done else "\nretired: nothing eligible")
                for path in done:
                    print(f"  {path}")
            else:
                print(
                    "\nNothing was deleted. Reclaim the 'safe to retire' set with: "
                    "hermes ws gc --apply"
                )
            return 0
        if action == "chatting":
            store.mark_chatting(args.session or "cli")
            print("session marked as chatting")
            return 0
    except store.WorkspaceError as e:
        print(f"refused: {e}")
        return 2
    return 0


def register(ctx) -> None:
    global _CTX
    _CTX = ctx
    for name, schema, handler in _ws_tools():
        ctx.register_tool(name=name, toolset="workspaces", schema=schema, handler=handler, emoji="🗂️")

    ctx.register_cli_command(
        name="ws",
        help="Workspaces: one per session (new/list/status/retire/chatting)",
        setup_fn=_register_cli,
        handler_fn=_cli,
        description=(
            "Session-owned workspaces under /Git/hermes/workspaces/<repo>/<slug>. "
            "Creates the jj workspace and its .hermes/session.json together."
        ),
    )

    repos = store.candidate_repos(limit=6)
    listing = ", ".join(repos) if repos else "(none found under /Git)"
    ctx.register_system_prompt_section(
        "workspace-discipline", _PROMPT_TEMPLATE.format(repos=listing), position="after_memory"
    )

    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    try:
        ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    except Exception:
        logger.debug("pre_gateway_dispatch hook not available in this host", exc_info=True)
    picker.register(ctx)
