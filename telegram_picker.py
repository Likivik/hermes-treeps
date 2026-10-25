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

from . import discovery, store

logger = logging.getLogger(__name__)

TG: dict[str, Any] = {
    "application": None,
    "adapter": None,
    "loop": None,
    "chats": {},
    "last_chat_key": None,
    "pending_picker": {},
    "last_picker": {},
    "warned_no_allowlist": False,
}

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


def _clone_root_label() -> str:
    from . import settings as _settings

    try:
        return str(_settings.settings()["clone_root"])
    except Exception:
        return "the configured clone root"


def _allowlist() -> list[int]:
    from . import settings as _settings

    raw = _settings.settings().get("allowed_user_ids") or []
    out: list[int] = []
    for value in raw:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            logger.warning("allowed_user_ids: skipping a non-numeric entry %r", value)
    return out


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


def _forge_name(entry: dict[str, Any]) -> str:
    tag = str(entry.get("tag") or entry.get("source") or "")
    return tag.split(":", 1)[-1] if ":" in tag else tag


def _entry_name(entry: dict[str, Any]) -> str:
    return str(entry.get("name") or "")


def _entry_clone_url(entry: dict[str, Any]) -> str:
    return str(entry.get("clone_url") or entry.get("url") or "")


def _is_local(entry: dict[str, Any]) -> bool:
    tag = str(entry.get("tag") or entry.get("source") or "")
    return tag == discovery.TAG_LOCAL


def _find_entry(entries: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for entry in entries:
        if _entry_name(entry) == name:
            return entry
    return None


def wire(application: Any, adapter: Any) -> None:
    """Register native handlers: capture loop/state, route our button taps."""
    try:
        from telegram import Update
        from telegram.ext import CallbackQueryHandler, TypeHandler
    except Exception:
        return
    TG["application"] = application
    TG["adapter"] = adapter

    async def _capture_loop(update: Any, context: Any) -> None:
        TG["loop"] = asyncio.get_running_loop()
        chat = getattr(update, "effective_chat", None)
        if chat is not None:
            msg = getattr(update, "effective_message", None)
            coords = (
                getattr(chat, "id", None),
                store.normalize_thread(
                    getattr(msg, "message_thread_id", None) if msg is not None else None
                ),
            )
            key = _chat_key(*coords)
            TG["chats"][key] = {"coords": coords, "at": time.time()}
            TG["last_chat_key"] = key
        await _maybe_claim_on_new(update)

    async def _maybe_claim_on_new(update: Any) -> None:
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
    application.add_handler(CallbackQueryHandler(_on_choice_callback, pattern=r"^(sel|clone-confirm):"))


def on_session_reset(session_id: str = "", new_session_id: str = "", **kw: Any) -> None:
    """The gateway fires this at /new, with the NEW session id, before the notice."""
    sid = new_session_id or session_id or ""
    if sid:
        note_session_id(sid)
    row = store.session_entry(sid) or {}
    coords = store.chat_coords(
        str(row.get("session_key") or ""), str(row.get("chat_type") or "dm")
    )
    if coords == (None, None):
        return
    key = _chat_key(*coords)
    if key not in TG.get("pending_picker", {}):
        return

    async def _send_after_notice() -> None:
        await asyncio.sleep(1.5)
        await fire_picker_once(sid, key)

    loop = TG.get("loop")
    if loop is not None and loop.is_running():
        asyncio.run_coroutine_threadsafe(_send_after_notice(), loop)


def handle_session_start(session_id: str, **kw: Any) -> bool:
    """The Telegram branch of on_session_start. True = handled."""
    platform = str(kw.get("platform") or "").strip().lower()
    if "telegram" not in platform:
        return False
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
        else:
            coro.close()
    except Exception:
        logger.warning("could not schedule the workspace button prompt", exc_info=True)
    return True


async def fire_picker_once(session_id: str, key: str | None = None) -> None:
    """Single consumer per chat: pop that chat's claim, then send."""
    pending = TG.get("pending_picker") or {}
    if key is None:
        if not pending:
            return
        key = next(iter(pending))
    pending.pop(key, None)
    await send_choice_message(session_id, hint_key=key)


async def send_choice_message(session_id: str, hint_key: str | None = None) -> None:
    """Send the flat, tagged repo picker keyboard to the session's chat."""
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        from . import settings as _settings

        if not _settings.settings()["picker_enabled"]:
            logger.info("workspace picker skipped: picker_enabled is false")
            return

        entry = store.session_entry(session_id) or {}
        chat_id, thread_id = store.chat_coords(
            str(entry.get("session_key") or ""), str(entry.get("chat_type") or "dm")
        )
        source = "session"
        if chat_id is None:
            claim = (TG.get("pending_picker") or {}).get(hint_key or "", {})
            chat_id, thread_id = claim.get("coords") or _most_recent_chat() or (None, None)
            thread_id = store.normalize_thread(thread_id)
            source = "last-update"

        adapter = TG.get("adapter")
        bot = getattr(adapter, "_bot", None) or getattr(adapter, "bot", None) if adapter else None
        if chat_id is None or bot is None:
            logger.warning(
                "workspace picker skipped: chat_id=%s bot=%s session=%s",
                chat_id, bool(bot), session_id,
            )
            return

        key = (chat_id, thread_id)
        last = TG["last_picker"].get(key)
        if last and (time.time() - float(last)) < 60:
            logger.info("workspace picker skipped: chat %s got one recently", key)
            return

        repos, errors = discovery.discover()
        if not repos:
            logger.info("workspace picker skipped: no repos from any source")
            return
        token = store.new_token(repos, session_id=session_id)
        rows = []
        for repo in repos:
            rows.append([InlineKeyboardButton(
                f"{_entry_name(repo)} ({repo.get('tag') or 'local'})",
                callback_data=f"sel:{token}:{_entry_name(repo)}",
            )])
        rows.append([InlineKeyboardButton("just chatting", callback_data=f"sel:{token}:__chat__")])
        rows.append([InlineKeyboardButton("new repo…", callback_data=f"sel:{token}:__newrepo__")])
        text = "Workspace for this session? Pick one, or say just chatting."
        if errors:
            text += "\nSome forges were unavailable."
        msg = await bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=text,
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


async def _ask_to_clone(q: Any, entry: dict[str, Any], session_id: str) -> None:
    """First forge tap: replace the keyboard with an explicit clone confirmation."""
    repo = _entry_name(entry)
    forge = _forge_name(entry) or "the remote"
    owner = str(entry.get("owner") or "")
    label = f"{owner}/{repo}" if owner else repo
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    except Exception:
        logger.warning("telegram is not importable; cannot ask for a clone confirmation")
        return
    token = store.new_token([entry], session_id=session_id, kind="clone")
    try:
        await q.answer(f"{repo} is not on disk. Confirm to clone it from {forge}.", show_alert=True)
    except Exception:
        pass
    try:
        await q.edit_message_text(
            f"Clone {label} from {forge} into {_clone_root_label()}?",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton(
                    f"clone {repo} from {forge}?",
                    callback_data=f"clone-confirm:{token}:{repo}",
                )],
                [InlineKeyboardButton("cancel", callback_data=f"sel:{token}:__cancel__")],
            ]),
        )
    except Exception:
        logger.debug("could not edit the choice message for the clone confirmation", exc_info=True)


def _resolve_session(update: Any, record: dict[str, Any]) -> str:
    chat = getattr(update, "effective_chat", None)
    msg = getattr(update, "effective_message", None)
    token_sid = str((record or {}).get("sid") or "")
    if token_sid and store.session_entry(token_sid):
        return token_sid
    row = store.session_for_chat(
        getattr(chat, "id", None),
        getattr(msg, "message_thread_id", None) if msg is not None else None,
    )
    session_id = str((row or {}).get("session_id") or "")
    return session_id or last_session()


def _bind_local(entry: dict[str, Any], session_id: str) -> dict[str, Any]:
    repo = _entry_name(entry)
    slug = f"{repo}-{session_id[-6:]}" if session_id else repo
    workspace = store.create(repo, slug, purpose="")
    bound_slug = str(workspace.get("slug") or slug)
    bound_path = str(workspace.get("path") or store.workspace_path(repo, bound_slug))
    store.bind_session(Path(bound_path), session_id)
    store.write_state(
        session_id,
        {"mode": "repo", "repo": repo, "slug": bound_slug, "path": bound_path},
    )
    return {"repo": repo, "slug": bound_slug, "path": bound_path}


def _clone_and_bind(entry: dict[str, Any], session_id: str) -> dict[str, Any]:
    repo = _entry_name(entry)
    url = _entry_clone_url(entry)
    # The URL is checked on this thread before the clone starts, so a hostile
    # transport-helper scheme can never reach the clone subprocess.
    store.validate_clone_url(repo, url)
    store.ensure_local(repo, url)
    return _bind_local(entry, session_id)


async def _run_clone_in_background(
    q: Any, entry: dict[str, Any], session_id: str,
) -> None:
    """Clone off the Telegram loop, then bind and report the terminal state.

    The status message gets exactly two edits: started and finished/failed.
    This prevents a retrying clone from flooding the chat with progress edits.
    """
    repo = _entry_name(entry)
    try:
        await q.edit_message_text(f"Cloning {repo}…")
    except Exception:
        logger.debug("could not edit the clone status message", exc_info=True)

    try:
        result = await asyncio.to_thread(_clone_and_bind, entry, session_id)
    except Exception as exc:
        logger.warning("background clone of %s failed", repo, exc_info=True)
        text = f"Clone failed for {repo}: {str(exc)[:240]}"
        try:
            await q.edit_message_text(text)
        except Exception:
            logger.debug("could not edit the failed clone status", exc_info=True)
        return

    text = f"Workspace: {result['repo']}/{result['slug']}"
    try:
        await q.edit_message_text(text)
    except Exception:
        logger.debug("could not edit the completed clone status", exc_info=True)


async def _on_choice_callback(update: Any, context: Any) -> None:
    """Handle the flat picker and the two-step clone confirmation."""
    q = getattr(update, "callback_query", None)
    if q is None:
        return
    data = str(getattr(q, "data", "") or "")
    parts = data.split(":", 2)
    if len(parts) != 3 or parts[0] not in {"sel", "clone-confirm", "ws"}:
        await q.answer()
        return
    action, token, choice = parts
    if action == "ws":
        # Compatibility with the pre-flat-menu callback shape: the old
        # keyboard selected by index and used ``clone`` for the confirm tap.
        action = "clone-confirm" if choice == "clone" else "sel"
        if action == "sel":
            try:
                choice = str(int(choice))
            except ValueError:
                await q.answer("This prompt expired - send /wsbtn for a fresh one.", show_alert=True)
                return
    elif action == "clone-confirm" and choice == "clone":
        action = "clone-confirm"
        choice = ""

    user = getattr(q, "from_user", None)
    if user is not None and not _tapper_allowed(getattr(user, "id", None)):
        await q.answer("Not allowed to pick this session's workspace.", show_alert=True)
        return

    record = store.take_token(token)
    if not record:
        await q.answer("This prompt expired - send /wsbtn for a fresh one.", show_alert=True)
        return

    session_id = _resolve_session(update, record)
    if not session_id:
        await q.answer("No active session for this chat - cannot bind a workspace.", show_alert=True)
        return
    note_session_id(session_id)

    already = store.read_state(session_id)
    if already and already.get("mode") == "repo":
        await q.answer(
            f"This session is already bound to {already.get('repo')}/{already.get('slug')} - "
            "tell the agent to rebind if you meant to switch.",
            show_alert=True,
        )
        return

    entries = [e for e in (record.get("repos") or []) if isinstance(e, dict)]
    if action == "clone-confirm":
        if str(record.get("kind") or "pick") != "clone":
            # A pick token cannot be replayed as clone approval: the second
            # tap must be backed by the token minted for the confirm button.
            await q.answer(
                "This button does not approve a clone - send /wsbtn for a fresh one.",
                show_alert=True,
            )
            return
        entry = _find_entry(entries, choice)
        if not entry and len(entries) == 1:
            entry = entries[0]
        if not entry:
            await q.answer("Nothing to clone - send /wsbtn for a fresh one.", show_alert=True)
            return
        repo = _entry_name(entry)
        url = _entry_clone_url(entry)
        try:
            # Validate at the confirm tap, before the background thread starts.
            store.validate_clone_url(repo, url)
        except store.WorkspaceError as exc:
            await q.answer(str(exc)[:240], show_alert=True)
            return
        await q.answer(f"cloning {repo}…")
        await _run_clone_in_background(q, entry, session_id)
        return

    if choice == "__chat__":
        store.mark_chatting(session_id)
        label = "just chatting"
    elif choice == "__newrepo__":
        await q.answer()
        try:
            await q.edit_message_text(
                "Tell the agent the name for the new repo and it will create it "
                f"under {_clone_root_label()} and give this session a workspace in it."
            )
        except Exception:
            pass
        return
    elif choice == "__cancel__":
        await q.answer("Cancelled - nothing was cloned.")
        try:
            await q.edit_message_text("Cancelled - nothing was cloned.")
        except Exception:
            pass
        return
    else:
        entry = _find_entry(entries, choice)
        if not entry and choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(entries):
                entry = entries[idx]
        if not entry:
            await q.answer("This prompt expired - send /wsbtn for a fresh one.", show_alert=True)
            return
        if not _is_local(entry):
            # A forge row needs a second, explicit tap before it can clone.
            await _ask_to_clone(q, entry, session_id)
            return
        try:
            result = await asyncio.to_thread(_bind_local, entry, session_id)
        except store.WorkspaceError as exc:
            await q.answer(str(exc)[:180], show_alert=True)
            return
        except Exception:
            logger.exception("workspace button choice failed")
            await q.answer("Something went wrong - use /ws status.", show_alert=True)
            return
        label = f"{result['repo']}/{result['slug']}"

    await q.answer(f"-> {label}")
    try:
        await q.edit_message_text(f"Workspace: {label}")
    except Exception:
        logger.debug("could not edit the choice message", exc_info=True)


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


def register(ctx: Any) -> None:
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_command("wsbtn", cmd_buttons, description="Send the workspace picker keyboard")
    try:
        ctx.register_telegram_handler(wire)
    except Exception:
        logger.debug("telegram handler not available in this host", exc_info=True)
