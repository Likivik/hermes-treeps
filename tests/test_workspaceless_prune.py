"""Workspaceless session rows must not accumulate forever.

WHY this exists: a cron session (or any chat-only session) hits the gate, gets
a binding row so the session can record its mode, and then never fires
``on_session_finalize``. Nothing was wrong with those rows individually - they
claim no path, so they pin no tree - but one per cron run piles up, and a
bindings table full of sessions that will never end is how "live" stops
meaning anything. The sweep prunes them; these tests pin the boundary: old
rows go, fresh rows and rows that DO claim a workspace stay.
"""

from __future__ import annotations

import plugin_loader

DB = plugin_loader.PLUGIN.db
STORE = plugin_loader.STORE


def _age_binding(session_id: str, days: int) -> None:
    old = DB.time.strftime("%Y-%m-%dT%H:%M:%S", DB.time.gmtime(DB.time.time() - days * 86400))
    with DB.transaction() as conn:
        conn.execute("UPDATE bindings SET updated_at=? WHERE session_id=?", (old, session_id))


def test_an_old_workspaceless_binding_is_pruned():
    DB.write_binding("cron_old", {"mode": "chatting", "slug": "", "path": ""})
    _age_binding("cron_old", 3)

    pruned = DB.prune_workspaceless_bindings()

    assert pruned == 1
    assert DB.read_binding("cron_old") is None


def test_a_fresh_workspaceless_binding_survives():
    """A cron session in flight must keep its row: it is the session's state."""
    DB.write_binding("cron_fresh", {"mode": "chatting", "slug": "", "path": ""})

    pruned = DB.prune_workspaceless_bindings()

    assert pruned == 0
    assert DB.read_binding("cron_fresh") is not None


def test_a_binding_that_claims_a_workspace_is_never_pruned():
    """The prune may only delete rows that can pin nothing."""
    DB.write_binding("bound_old", {"mode": "repo", "slug": "demo", "path": "/tmp/demo/ws"})
    _age_binding("bound_old", 30)

    pruned = DB.prune_workspaceless_bindings()

    assert pruned == 0
    assert DB.read_binding("bound_old") is not None
