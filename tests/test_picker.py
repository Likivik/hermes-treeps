from __future__ import annotations

import asyncio
import threading

from plugin_loader import PLUGIN, STORE

TP = PLUGIN.telegram_picker


class FakeButton:
    def __init__(self, text, callback_data):
        self.text = text
        self.callback_data = callback_data

    def __str__(self):
        return self.text


class FakeMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard

    def __str__(self):
        return " ".join(str(button) for row in self.inline_keyboard for button in row)


class FakeTelegram:
    InlineKeyboardButton = FakeButton
    InlineKeyboardMarkup = FakeMarkup


class FakeQuery:
    def __init__(self, data: str):
        self.data = data
        self.from_user = None
        self.answers: list[str | None] = []
        self.edits: list[tuple[str, object]] = []

    async def answer(self, text=None, show_alert=False):
        self.answers.append(text)

    async def edit_message_text(self, text, *args, **kwargs):
        self.edits.append((text, kwargs.get("reply_markup")))
        return None


class FakeUpdate:
    def __init__(self, query: FakeQuery):
        self.callback_query = query
        self.effective_chat = type("Chat", (), {"id": -100123})()
        self.effective_message = type("Msg", (), {"message_thread_id": 42})()


def _patch_session(monkeypatch, sid="sid-picker"):
    monkeypatch.setattr(STORE, "session_entry", lambda session_id: {"session_id": session_id})
    monkeypatch.setattr(STORE, "session_for_chat", lambda chat_id, thread: {"session_id": sid})
    monkeypatch.setattr(STORE, "read_state", lambda session_id: None)
    monkeypatch.setattr(STORE, "write_state", lambda session_id, payload: payload)


def test_the_token_carries_its_session_and_the_tap_prefers_it(state_home):
    """A picker tap must bind the session the keyboard was sent FOR."""
    token = STORE.new_token([{"name": "hermes-treeps", "url": "x", "source": "local"}], session_id="sid-fresh")
    record = STORE.take_token(token)
    assert record["sid"] == "sid-fresh", record


def test_a_dead_token_session_falls_back_to_the_chat_row(monkeypatch, state_home):
    """A /new reset kills the token's session; the chat row then decides."""
    monkeypatch.setattr(STORE, "session_entry", lambda sid: None)
    monkeypatch.setattr(
        STORE, "session_for_chat", lambda chat_id, thread: {"session_id": "sid-after-reset"}
    )
    token = STORE.new_token([{"name": "repo", "url": "x", "source": "local"}], session_id="sid-dead")

    class Q:
        answered = []

        def __init__(self, token):
            self.data = f"ws:{token}:chat"
            self.from_user = None

        async def answer(self, text=None, show_alert=False):
            self.answered.append(text)

        async def edit_message_text(self, *a, **kw):
            return None

    class Chat:
        id = -100123

    class Msg:
        message_thread_id = 42

    q = Q(token)
    q.data = f"sel:{token}:__chat__"

    class Upd:
        effective_chat = Chat()
        effective_message = Msg()
        callback_query = q

    calls = {}
    monkeypatch.setattr(STORE, "mark_chatting", lambda sid, **kw: calls.setdefault("sid", sid))

    asyncio.run(TP._on_choice_callback(Upd(), None))
    assert calls.get("sid") == "sid-after-reset", (calls, Q.answered)


def test_local_tap_binds_immediately_without_a_confirmation(monkeypatch):
    """A local row binds on its first tap: a confirm would be a needless step."""
    _patch_session(monkeypatch)
    entry = {"name": "alpha", "tag": "local", "clone_url": "", "local_path": "/tmp/alpha", "owner": ""}
    token = STORE.new_token([entry], session_id="sid-picker")
    query = FakeQuery(f"sel:{token}:alpha")
    bound = {}

    monkeypatch.setattr(STORE, "create", lambda repo, slug, purpose="": {"repo": repo, "slug": slug, "path": "/tmp/ws"})
    monkeypatch.setattr(STORE, "bind_session", lambda path, sid: bound.update(path=str(path), sid=sid))
    monkeypatch.setattr(STORE, "write_state", lambda sid, payload: bound.update(state=payload))

    asyncio.run(TP._on_choice_callback(FakeUpdate(query), None))

    assert bound["sid"] == "sid-picker"
    assert bound["state"]["repo"] == "alpha"
    assert query.answers == ["-> alpha/alpha-picker"]
    assert all("confirm" not in str(edit[1]).lower() for edit in query.edits)


def test_forge_tap_shows_confirmation_and_does_not_clone(monkeypatch):
    """The first tap on a forge row is only an approval request."""
    _patch_session(monkeypatch)
    monkeypatch.setitem(__import__("sys").modules, "telegram", FakeTelegram)
    entry = {
        "name": "beta",
        "tag": "forge:github",
        "clone_url": "https://github.com/me/beta",
        "local_path": "",
        "owner": "me",
    }
    token = STORE.new_token([entry], session_id="sid-picker")
    query = FakeQuery(f"sel:{token}:beta")
    clone_calls = []
    monkeypatch.setattr(STORE, "ensure_local", lambda *args: clone_calls.append(args))

    asyncio.run(TP._on_choice_callback(FakeUpdate(query), None))

    assert clone_calls == []
    assert len(query.edits) == 1
    text, markup = query.edits[0]
    assert "Clone me/beta from github" in text
    assert "clone beta from github?" in str(markup)


def test_confirm_with_a_bad_scheme_refuses_and_names_the_scheme(monkeypatch):
    """The confirm tap validates before any clone starts."""
    _patch_session(monkeypatch)
    entry = {
        "name": "beta",
        "tag": "forge:github",
        "clone_url": "ext::sh -c evil",
        "local_path": "",
        "owner": "me",
    }
    token = STORE.new_token([entry], session_id="sid-picker", kind="clone")
    query = FakeQuery(f"clone-confirm:{token}:beta")
    clone_calls = []
    monkeypatch.setattr(STORE, "ensure_local", lambda *args: clone_calls.append(args))

    asyncio.run(TP._on_choice_callback(FakeUpdate(query), None))

    assert clone_calls == []
    assert any("ext" in str(answer) for answer in query.answers), query.answers


def test_confirm_clones_in_a_background_thread_and_binds_on_completion(monkeypatch):
    """A slow clone cannot block Telegram, and success binds automatically."""
    _patch_session(monkeypatch)
    entry = {
        "name": "beta",
        "tag": "forge:github",
        "clone_url": "https://github.com/me/beta",
        "local_path": "",
        "owner": "me",
    }
    token = STORE.new_token([entry], session_id="sid-picker", kind="clone")
    query = FakeQuery(f"clone-confirm:{token}:beta")
    release = threading.Event()
    entered = threading.Event()
    bound = {}
    clone_threads = []

    def slow_clone(repo, url):
        clone_threads.append(threading.current_thread())
        entered.set()
        release.wait(2)
        return "/tmp/beta"

    monkeypatch.setattr(STORE, "ensure_local", slow_clone)
    monkeypatch.setattr(STORE, "create", lambda repo, slug, purpose="": {"repo": repo, "slug": slug, "path": "/tmp/ws"})
    monkeypatch.setattr(STORE, "bind_session", lambda path, sid: bound.update(path=str(path), sid=sid))
    monkeypatch.setattr(STORE, "write_state", lambda sid, payload: bound.update(state=payload))

    async def run():
        task = asyncio.create_task(TP._on_choice_callback(FakeUpdate(query), None))
        await asyncio.to_thread(entered.wait, 1)
        assert bound == {}, "clone completion must not be awaited inline"
        release.set()
        await task

    asyncio.run(run())
    assert clone_threads and clone_threads[0] is not threading.main_thread()
    assert bound["sid"] == "sid-picker"
    assert bound["state"]["repo"] == "beta"


def test_status_message_is_edited_at_most_twice(monkeypatch):
    """Clone progress has one started edit and one terminal edit - no loops."""
    _patch_session(monkeypatch)
    entry = {
        "name": "beta",
        "tag": "forge:github",
        "clone_url": "https://github.com/me/beta",
        "local_path": "",
        "owner": "me",
    }
    token = STORE.new_token([entry], session_id="sid-picker", kind="clone")
    query = FakeQuery(f"clone-confirm:{token}:beta")
    monkeypatch.setattr(STORE, "ensure_local", lambda repo, url: "/tmp/beta")
    monkeypatch.setattr(STORE, "create", lambda repo, slug, purpose="": {"repo": repo, "slug": slug, "path": "/tmp/ws"})
    monkeypatch.setattr(STORE, "bind_session", lambda path, sid: None)
    monkeypatch.setattr(STORE, "write_state", lambda sid, payload: None)

    asyncio.run(TP._on_choice_callback(FakeUpdate(query), None))

    assert len(query.edits) <= 2, query.edits
