"""The forum General topic is topic id 1, which the Bot API refuses.

Observed live: the picker died with BadRequest "Message thread not found" for a
session keyed agent:main:telegram:group:-100…:1 — the General topic — because
message_thread_id=1 was passed. The picker must fall back to no thread there.
"""
from __future__ import annotations

import asyncio

from plugin_loader import PLUGIN, SETTINGS, STORE

TP = PLUGIN.telegram_picker
SESSIONS = STORE  # store re-exports the session helpers


def test_chat_coords_drops_the_general_topic_id():
    assert SESSIONS.chat_coords("agent:main:telegram:group:-1004303910638:1", "group") == (
        -1004303910638,
        None,
    )


def test_chat_coords_keeps_a_real_topic_id():
    assert SESSIONS.chat_coords("agent:main:telegram:group:-1004303910638:14650", "group") == (
        -1004303910638,
        14650,
    )


def test_normalize_thread_treats_only_general_and_empty_as_no_thread():
    assert SESSIONS.normalize_thread(1) is None
    assert SESSIONS.normalize_thread("1") is None
    assert SESSIONS.normalize_thread(0) is None
    assert SESSIONS.normalize_thread(None) is None
    assert SESSIONS.normalize_thread("") is None
    assert SESSIONS.normalize_thread(14650) == 14650


class _FakeButton:
    def __init__(self, text, callback_data):
        self.text = text
        self.callback_data = callback_data


class _FakeMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard


class _FakeTelegram:
    InlineKeyboardButton = _FakeButton
    InlineKeyboardMarkup = _FakeMarkup


class _FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return type("Msg", (), {"message_id": 1})()


def _send_for(monkeypatch, session_key, chat_type="group"):
    monkeypatch.setitem(__import__("sys").modules, "telegram", _FakeTelegram)
    monkeypatch.setattr(
        STORE, "session_entry", lambda sid: {"session_key": session_key, "chat_type": chat_type}
    )
    monkeypatch.setattr(TP.discovery, "discover", lambda: ([{"name": "alpha", "tag": "local"}], []))
    monkeypatch.setattr(SETTINGS, "settings", lambda: {"picker_enabled": True, "clone_root": "/tmp"})

    bot = _FakeBot()
    adapter = type("Adapter", (), {"_bot": bot})()
    monkeypatch.setitem(TP.TG, "adapter", adapter)
    monkeypatch.setitem(TP.TG, "last_picker", {})
    monkeypatch.setitem(TP.TG, "pending_picker", {})

    asyncio.run(TP.send_choice_message("sid-general"))
    return bot.sent


def test_picker_sends_without_a_thread_in_the_general_topic(monkeypatch, state_home):
    sent = _send_for(monkeypatch, "agent:main:telegram:group:-1004303910638:1")

    assert len(sent) == 1, sent
    assert sent[0]["chat_id"] == -1004303910638
    assert sent[0]["message_thread_id"] is None


def test_picker_keeps_a_real_topic_thread(monkeypatch, state_home):
    sent = _send_for(monkeypatch, "agent:main:telegram:group:-1004303910638:14650")

    assert sent[0]["message_thread_id"] == 14650
