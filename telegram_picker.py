"""Telegram side of the workspace picker: inline keyboard, taps, /new wiring.

All of this lives behind `register_telegram_handler`, so the module must import
cleanly without python-telegram-bot present (imports stay inside functions).
State is one module-level dict: this plugin runs once per gateway process.

Everything that was once a single slot is keyed by chat (``<chat_id>:<thread>``):
two topics can hold a /new in flight at the same time without stealing each
other's picker.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from . import store

logger = logging.getLogger(__name__)

TG: dict[str, Any] = {
    "application": None,
    "adapter": None,
    "loop": None,
    # chat_key -> {"coords": (chat_id, thread), "at": ts}; the most recent entry
    # is the fallback chat when a session row has no answer.
    "chats": {},
    "last_chat_key": None,
    # chat_key -> claim dict while a /new is in flight in that chat
    "pending_picker": {},
    # (chat_id, thread) -> ts of the last picker sent there
    "last_picker": {},
    "warned_no_allowlist": False,
}

# Last session id seen from any hook, with a timestamp: the gate uses recency
# to let subagents inherit their parent's binding (a subagent's first gated
# tool call comes seconds after its parent dispatched it).
_LAST_SESSION = {"sid": "", "at": 0.0}


def note_session_id(session_id: str) -> None:
    if session_id:
        _LAST_SESSION["sid"] = session_id
        _LAST_SESSION["at"] = time.time()


def last_session(max_age_s: float = 0.0) -> str:
    if _LAST_SESSION["sid"] and (not max_age_s or time.time() - _LAST_SESSION["at"] <= max_age_s):
        return _LAST_SESSION["sid"]
    return ""


def _chat_key(chat_id: Any, thread_id: Any) -> str:
    return f"{chat_id}:{thread_id}"


def _most_recent_chat() -> tuple[Any, Any] | None:
    chats = TG.get("chats") or {}
    if not chats:
        return None
    key = max(chats, key=lambda k: chats[k].get("at", 0))
    return chats[key].get("coords")


def _allowlist() -> list[int]:
    """User ids allowed to tap the picker. Empty/unset means open (legacy)."""
    try:
        from hermes_cli.config import load_config_readonly  # type: ignore

        cfg = load_config_readonly() or {}
        entry = ((cfg.get("plugins") or {}).get("entries") or {}).get("hermes-treeps") or {}
        ids = entry.get("allowed_user_ids") or []
        return [int(i) for i in ids]
    except Exception:
        return []


def _tapper_allowed(user_id: Any) -> bool:
    ids = _allowlist()
    if not ids:
        if not TG.get("warned_no_allowlist"):
            TG["warned_no_allowlist"] = True
            logger.warning(
                "hermes-treeps.allowed_user_ids is unset: ANY group member can tap "
                "the workspace picker and create workspaces under this session"
            )
        return True
    try:
        return int(user_id) in ids
    except Exception:
        return False


# ── adapter wiring ────────────────────────────────────────────────────────────


def wire(application: Any, adapter: Any) -> None:
    """Register native handlers: capture loop/state, route our button taps."""
    try:
        from telegram import Update
        from telegram.ext import CallbackQueryHandler, TypeHandler
    except Exception:  # PTB lives in the hermes env, not in plugin import scope
        return
    TG["application"] = application
    TG["adapter"] = adapter

    async def _capture_loop(update: Any, context: Any) -> None:
        # Runs first (group -1) and must never block: PTB processes updates in
        # order, and a slow handler here delayed the user's approval taps.
        TG["loop"] = asyncio.get_running_loop()
        chat = getattr(update, "effective_chat", None)
        if chat is not None:
            msg = getattr(update, "effective_message", None)
            coords = (
                getattr(chat, "id", None),
                getattr(msg, "message_thread_id", None) if msg is not None else None,
            )
            key = _chat_key(*coords)
            TG["chats"][key] = {"coords": coords, "at": time.time()}
            TG["last_chat_key"] = key
        await _maybe_claim_on_new(update)

    async def _maybe_claim_on_new(update: Any) -> None:
        """/new claims the slot for ITS chat; the on_session_reset hook sends."""
        msg = getattr(update, "effective_message", None)
        text = (getattr(msg, "text", "") or "").strip()
        if not text.startswith("/new"):
            return
        chat_id, thread = (TG.get("chats", {}).get(TG.get("last_chat_key")) or {}).get(
            "coords", (None, None)
        )
        prev = (store.session_for_chat(chat_id, thread) or {}).get("session_id")
        TG["pending_picker"][_chat_key(chat_id, thread)] = {
            "coords": (chat_id, thread), "prev": prev, "at": time.time(),
        }

    application.add_handler(TypeHandler(Update, _capture_loop), group=-1)
    application.add_handler(CallbackQueryHandler(_on_choice_callback, pattern=r"^ws:"))


# ── hooks ─────────────────────────────────────────────────────────────────────


def on_session_reset(session_id: str = "", new_session_id: str = "", **kw: Any) -> None:
    """The gateway fires this at /new (`gateway/slash_commands_session.py:214`),
    with the NEW session id, before the "Session reset!" notice is sent."""
    sid = new_session_id or session_id or ""
    if sid:
        note_session_id(sid)
    # Which chat? The new session's own row in sessions.json - /new may have run
    # in a different chat than the last update we saw.
    row = store.session_entry(sid) or {}
    coords = store.chat_coords(
        str(row.get("session_key") or ""), str(row.get("chat_type") or "dm")
    )
    if coords == (None, None):
        return
    key = _chat_key(*coords)
    if key not in TG.get("pending_picker", {}):
        return  # no /new claim waiting in that chat
    try:
        async def _send_after_notice() -> None:
            # The notice is sent on return from this hook, so a short wait keeps
            # the keyboard below it. The hook is the event; this is only ordering.
            await asyncio.sleep(1.5)
            await fire_picker_once(sid, key)

        loop = TG.get("loop")
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(_send_after_notice(), loop)
        # else: the claim stays armed; the session-start path still covers it
    except Exception:
        logger.warning("could not schedule the reset picker", exc_info=True)


def handle_session_start(session_id: str, **kw: Any) -> bool:
    """The Telegram branch of on_session_start. True = handled (caller returns)."""
    platform = str(kw.get("platform") or "").strip().lower()
    if "telegram" not in platform:
        return False
    # Fallback claim for brand-new sessions that never saw a /new update.
    row = store.session_entry(session_id) or {}
    coords = store.chat_coords(
        str(row.get("session_key") or ""), str(row.get("chat_type") or "dm")
    )
    if coords != (None, None):
        TG["pending_picker"].setdefault(
            _chat_key(*coords), {"coords": coords, "prev": None, "at": time.time()}
        )
    try:
        coro = fire_picker_once(session_id)
        loop = TG.get("loop")
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop)
        elif TG.get("application") is not None:
            TG["application"].create_task(coro)
        else:  # in-process CLI host: we ARE in a loop
            logger.debug("no gateway loop for picker; skipping")
            coro.close()
    except Exception:
        logger.warning("could not schedule the workspace button prompt", exc_info=True)
    return True


# ── sending ───────────────────────────────────────────────────────────────────


async def fire_picker_once(session_id: str, key: str | None = None) -> None:
    """Single consumer per chat: pop that chat's claim, then send. The sender
    dedupes per chat, so a second trigger cannot stack a keyboard."""
    pending = TG.get("pending_picker") or {}
    if key is None:
        if not pending:
            return
        key = next(iter(pending))
    pending.pop(key, None)
    await send_choice_message(session_id, hint_key=key)


async def send_choice_message(session_id: str, hint_key: str | None = None) -> None:
    """Send the repo picker keyboard to the session's chat."""
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        # The session's OWN key first: the last-update chat is only a fallback -
        # in a multi-topic group it can be a different thread entirely (that
        # mistake once put a keyboard in topic 29039 while the user sat in 14650).
        entry = store.session_entry(session_id) or {}
        chat_id, thread_id = store.chat_coords(
            str(entry.get("session_key") or ""), str(entry.get("chat_type") or "dm")
        )
        source = "session"
        if chat_id is None:
            claim = (TG.get("pending_picker") or {}).get(hint_key or "", {})
            chat_id, thread_id = claim.get("coords") or _most_recent_chat() or (None, None)
            source = "last-update"

        adapter = TG.get("adapter")
        bot = getattr(adapter, "_bot", None) or getattr(adapter, "bot", None) if adapter else None
        if chat_id is None or bot is None:
            logger.warning(
                "workspace picker skipped: chat_id=%s bot=%s session=%s",
                chat_id, bool(bot), session_id,
            )
            return

        # Dedupe per chat+thread (a global flag once blocked chat B because chat
        # A got a picker a minute earlier).
        key = (chat_id, thread_id)
        last = TG["last_picker"].get(key)
        if last and (time.time() - float(last)) < 60:
            logger.info("workspace picker skipped: chat %s got one recently", key)
            return

        # Remote-first: the user's repos from GitHub + Forgejo (most recently
        # pushed), NOT what happens to be on disk - that is what he asked for.
        repos = store.remote_repos(limit_per_source=5)
        if not repos:
            logger.info("workspace picker skipped: no repos from any source")
            return
        token = store.new_token(repos)
        rows = []
        for i in range(0, len(repos), 2):
            pair = repos[i : i + 2]
            rows.append([
                InlineKeyboardButton(
                    f"{r['name']} ({r['source'][:2]})",
                    callback_data=f"ws:{token}:{i + j}",
                )
                for j, r in enumerate(pair)
            ])
        rows.append([InlineKeyboardButton("just chatting", callback_data=f"ws:{token}:chat")])
        rows.append([InlineKeyboardButton("new repo…", callback_data=f"ws:{token}:newrepo")])
        msg = await bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text="Workspace for this session? Pick one, or say just chatting.",
            reply_markup=InlineKeyboardMarkup(rows),
            disable_web_page_preview=True,
        )
        logger.info(
            "workspace picker sent: session=%s chat=%s thread=%s repos=%d via=%s msg_id=%s",
            session_id, chat_id, thread_id, len(repos), source, getattr(msg, "message_id", "?"),
        )
        TG["last_picker"][key] = time.time()
    except Exception:
        logger.warning("workspace picker failed for session %s", session_id, exc_info=True)


# ── taps ──────────────────────────────────────────────────────────────────────


async def _on_choice_callback(update: Any, context: Any) -> None:
    """A repo/chatting button was tapped: record the choice, retire the keyboard."""
    q = getattr(update, "callback_query", None)
    if q is None:
        return
    try:
        _, token, choice = (q.data or "").split(":", 2)
    except ValueError:
        await q.answer()
        return

    user = getattr(q, "from_user", None)
    if user is not None and not _tapper_allowed(getattr(user, "id", None)):
        await q.answer("Not allowed to pick this session's workspace.", show_alert=True)
        return

    record = store.take_token(token)
    if not record:
        await q.answer("This prompt expired - send /wsbtn for a fresh one.", show_alert=True)
        return

    # Resolve the session NOW, at tap time: the token predates the reset, so the
    # session id it would have carried is already dead.
    session_id = last_session()
    if not session_id:
        chat = getattr(update, "effective_chat", None)
        msg = getattr(update, "effective_message", None)
        row = store.session_for_chat(
            getattr(chat, "id", None),
            getattr(msg, "message_thread_id", None) if msg is not None else None,
        )
        session_id = str((row or {}).get("session_id") or "")
    if not session_id:
        await q.answer("No active session for this chat - cannot bind a workspace.", show_alert=True)
        return
    note_session_id(session_id)

    already = store.read_state(session_id)
    if choice not in {"other", "newrepo"} and already and already.get("mode") == "repo":
        await q.answer(
            f"This session is already bound to {already.get('repo')}/{already.get('slug')} - "
            "tell the agent to rebind if you meant to switch.",
            show_alert=True,
        )
        return

    try:
        if choice == "chat":
            store.mark_chatting(session_id)
            label = "just chatting"
        elif choice == "newrepo":
            await q.answer()
            try:
                await q.edit_message_text(
                    "Tell the agent the name for the new repo and it will create it "
                    "under /Git and give this session a workspace in it."
                )
            except Exception:
                pass
            return
        else:
            repos = record.get("repos") or []
            idx = int(choice)
            if idx < 0 or idx >= len(repos):
                raise store.WorkspaceError(f"unknown choice {choice!r}")
            entry = repos[idx]
            repo = entry["name"] if isinstance(entry, dict) else entry
            slug = f"{repo}-{session_id[-6:]}" if session_id else repo
            import asyncio

            def _bind_local() -> dict:
                if isinstance(entry, dict) and entry.get("url"):
                    store.ensure_local(repo, entry["url"])
                # WHY the record is returned and used: create() SANITISES the
                # slug, so the slug we passed in is not necessarily the slug on
                # disk. Binding our own copy pointed at a non-existent path:
                # the session stayed unbound and the workspace leaked.
                return store.create(repo, slug, purpose="")

            await q.answer(f"setting up {repo} (cloning if needed)…")
            # A clone can take a minute; run it in a thread or every Telegram
            # update in the process stalls behind it.
            record = await asyncio.to_thread(_bind_local)
            bound_slug = str(record.get("slug") or slug)
            bound_path = str(record.get("path") or store.workspace_path(repo, bound_slug))
            store.bind_session(Path(bound_path), session_id)
            store.write_state(
                session_id,
                {"mode": "repo", "repo": repo, "slug": bound_slug, "path": bound_path},
            )
            label = f"{repo}/{bound_slug}"
    except store.WorkspaceError as e:
        await q.answer(str(e)[:180], show_alert=True)
        return
    except Exception:
        logger.exception("workspace button choice failed")
        await q.answer("Something went wrong - use /ws status.", show_alert=True)
        return

    await q.answer(f"-> {label}")
    try:
        await q.edit_message_text(f"Workspace: {label}")
    except Exception:
        logger.debug("could not edit the choice message", exc_info=True)


# ── manual trigger ────────────────────────────────────────────────────────────


def cmd_buttons(raw_args: str = "") -> str:
    """/wsbtn - send the picker on demand; also the test path for the sender."""
    try:
        key = TG.get("last_chat_key")
        if key is None:
            return "no chat captured yet - send a normal message first, then /wsbtn"
        loop = TG.get("loop")
        if loop is None or not loop.is_running():
            return "no gateway loop captured yet - send any message first"
        asyncio.run_coroutine_threadsafe(send_choice_message(last_session(), hint_key=key), loop)
        return "workspace buttons sent"
    except Exception as e:
        logger.warning("/wsbtn failed", exc_info=True)
        return f"/wsbtn failed: {type(e).__name__}: {e}"


# ── registration ──────────────────────────────────────────────────────────────


def register(ctx: Any) -> None:
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_command("wsbtn", cmd_buttons, description="Send the workspace picker keyboard")
    try:
        ctx.register_telegram_handler(wire)
    except Exception:
        logger.debug("telegram handler not available in this host", exc_info=True)
