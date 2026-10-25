"""Host-state helpers: sessions.json lookups used by the picker and the contract."""
from __future__ import annotations

import json
import logging
from typing import Any

from . import store  # late-bound: tests monkeypatch store.<helper>, so bind at call time

logger = logging.getLogger(__name__)

# ── sessions.json helpers (host state, read-only) ─────────────────────────────


def session_entry(session_id: str) -> dict[str, Any] | None:
    """The sessions.json row for a session id (platform, chat_type, key)."""
    p = store.hermes_home() / "sessions" / "sessions.json"
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    for key, entry in data.items():
        if isinstance(entry, dict) and entry.get("session_id") == session_id:
            return {**entry, "session_key": entry.get("session_key") or key}
    return None


def chat_coords(session_key: str, chat_type: str) -> tuple[int | None, int | None]:
    """(chat_id, thread_id) from a durable key like agent:main:telegram:group:-100…:29039."""
    parts = session_key.split(":")
    try:
        if chat_type == "group" and len(parts) >= 2:
            return int(parts[-2]), int(parts[-1])
        return int(parts[-1]), None
    except Exception:
        return None, None


def session_for_chat(chat_id: Any, thread_id: Any = None) -> dict[str, Any] | None:
    """The sessions.json row bound to a chat (and topic), if any.

    Used as a state signal: a /new reset replaces this row's session_id, so
    watching it is a real event, not a guessed delay.
    """
    p = store.hermes_home() / "sessions" / "sessions.json"
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    suffix = f":{chat_id}:{thread_id}" if thread_id else f":{chat_id}"
    for key, entry in data.items():
        if isinstance(entry, dict) and key.endswith(suffix):
            return {**entry, "session_key": key}
    return None


