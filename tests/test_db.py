"""db.py: the storage layer's own guarantees.

The bug this file exists for: eight concurrent ``bind_session`` calls on one
workspace lost four of them. The old implementation read a JSON document,
appended to a list and wrote the whole file back - the last writer won with
the list it had read BEFORE the others wrote. A row per binding with one
transaction per mutation removes the class of bug; these tests pin that.
"""

from __future__ import annotations

import threading

from plugin_loader import STORE

DB = STORE.db


# ── the concurrency guarantee ────────────────────────────────────────────────


def test_eight_concurrent_bind_sessions_all_land():
    """Eight threads binding one workspace: every session must be recorded.

    Prevents (P1-6, concurrency half): the old JSON read-modify-write lost
    four of eight bindings. The count is the whole assertion - a
    read-modify-write loses some of them, so this test fails on the old code.
    """
    DB.write_workspace({"repo": "repo", "slug": "ws", "path": "/tmp/ws"}, path="/tmp/ws")
    barrier = threading.Barrier(8)
    errors: list[Exception] = []

    def _bind(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            DB.add_workspace_session("repo", "ws", f"sid-{i}")
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=_bind, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    landed = sorted(DB.workspace_sessions("repo", "ws"))
    assert len(landed) == 8, f"only {len(landed)} of 8 bindings landed: {landed}"
    assert landed == sorted(f"sid-{i}" for i in range(8)), landed


def test_forget_leaves_no_ghost_in_the_workspace():
    """forget() must also drop the workspace's claim on the session.

    Prevents: a forgotten session stayed listed in the record's `sessions`, so
    `retire()` refused the workspace forever ("bound session(s) still live")
    while nothing was bound to it and there was no way to clear the ghost.
    """
    DB.write_workspace({"repo": "repo", "slug": "ghost", "path": "/tmp/ghost"}, path="/tmp/ghost")
    DB.add_workspace_session("repo", "ghost", "sid-ghost")
    assert DB.read_workspace("/tmp/ghost")["sessions"] == ["sid-ghost"]

    assert DB.delete_binding("sid-ghost") is False  # no binding row yet
    DB.write_binding("sid-ghost", {"mode": "repo", "repo": "repo", "slug": "ghost", "path": "/tmp/ghost"})
    assert DB.delete_binding("sid-ghost") is True

    left = DB.read_workspace("/tmp/ghost")["sessions"]
    assert left == [], f"the ghost survived forget(): {left}"


def test_an_existing_database_gets_the_schema_version_row(state_home):
    """A database created before schema_version existed must still get the row.

    Prevents: the write lived inside the create-schema branch, so every
    upgraded install had no version row and a future migration could not tell
    an upgraded v2 from a v1.
    """
    DB.reset()
    DB.connect()  # first open creates the schema
    # simulate an upgraded install: drop the row, then force a fresh connection
    conn = DB.connect()
    conn.execute("DELETE FROM kv WHERE key='schema_version'")
    conn.commit()
    DB.reset()  # bump the generation so the next connect() opens a new connection
    assert DB.get_kv("schema_version") == str(DB.SCHEMA_VERSION), "upgraded db left unversioned"
