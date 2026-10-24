from __future__ import annotations

from plugin_loader import PLUGIN, STORE

TP = PLUGIN.telegram_picker




def test_the_token_carries_its_session_and_the_tap_prefers_it(state_home):
    """A picker tap must bind the session the keyboard was sent FOR.

    Prevents the live bug Kirill hit on a /new session in a shared chat: the
    tap resolved via the process-global recency tracker, which still named the
    PREVIOUS session - the new session got "already bound to <other
    session's workspace>" instead of its own choice.
    """
    token = STORE.new_token([{"name": "hermes-treeps", "url": "x", "source": "local"}], session_id="sid-fresh")
    record = STORE.take_token(token)
    assert record["sid"] == "sid-fresh", record


def test_a_dead_token_session_falls_back_to_the_chat_row(monkeypatch, state_home):
    """A /new reset kills the token's session; the chat row then decides.

    Prevents: binding a workspace to a dead session id - the original reason
    the resolution happens at tap time.
    """
    # no session_entry for the token sid -> fallback path; chat row is faked
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

    class Upd:
        effective_chat = Chat()
        effective_message = Msg()
        callback_query = q

    import asyncio
    # stub the bind side: mark_chatting is what choice "chat" calls
    calls = {}
    monkeypatch.setattr(STORE, "mark_chatting", lambda sid, **kw: calls.setdefault("sid", sid))

    asyncio.run(TP._on_choice_callback(Upd(), None))
    assert calls.get("sid") == "sid-after-reset", (calls, Q.answered)
