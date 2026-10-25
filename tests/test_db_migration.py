"""The v3 migration must not break an EXISTING database on open.

WHAT this defends: the live install's database already exists with its rows.
Opening it with the new code must ADD the new tables and touch nothing else.
A migration that dropped or rewrote a table would take every live session's
binding with it - and the session-binding table is the one thing in this
plugin that must never be lost.

The test builds a v2-shaped database by hand (the old schema, the old
version row, a real binding row), opens it with the current code, and
checks: the new tables exist, the old rows are still there, and the
version row moved forward.
"""

from __future__ import annotations

import sqlite3

import pytest
from plugin_loader import STORE

# The schema as it was BEFORE the child-report tables existed.
V2_SCHEMA = """
CREATE TABLE workspaces (
    repo        TEXT NOT NULL,
    slug        TEXT NOT NULL,
    path        TEXT NOT NULL DEFAULT '',
    vcs         TEXT NOT NULL DEFAULT 'jj',
    purpose     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT '',
    last_synced TEXT,
    extra       TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (repo, slug)
);
CREATE TABLE bindings (
    session_id       TEXT PRIMARY KEY,
    mode             TEXT,
    repo             TEXT,
    slug             TEXT,
    path             TEXT,
    title            TEXT,
    inherited_from   TEXT,
    inherited_at     TEXT,
    session_ended_at TEXT,
    updated_at       TEXT,
    extra            TEXT NOT NULL DEFAULT '{}',
    payload          TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE workspace_sessions (
    repo       TEXT NOT NULL,
    slug       TEXT NOT NULL,
    session_id TEXT NOT NULL,
    PRIMARY KEY (repo, slug, session_id)
);
CREATE TABLE kv (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL DEFAULT '',
    updated_at TEXT
);
"""


@pytest.fixture
def v2_database(tmp_path, monkeypatch):
    """A database shaped like the v2 live install, with a live binding in it."""
    path = tmp_path / "plugin-data" / "data.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(V2_SCHEMA)
    conn.execute(
        "INSERT INTO bindings(session_id, mode, repo, slug, path, updated_at, payload)"
        " VALUES(?,?,?,?,?,?,?)",
        (
            "a-live-session",
            "repo",
            "repo",
            "live-tree",
            "/tmp/live-tree",
            "2026-01-01T00:00:00Z",
            '{"mode": "repo", "repo": "repo", "slug": "live-tree"}',
        ),
    )
    conn.execute("INSERT INTO kv(key, value) VALUES('schema_version', '2')")
    conn.commit()
    conn.close()

    monkeypatch.setenv("HERMES_TREEPS_PLUGIN_DATA_DIR", str(path.parent))
    monkeypatch.setenv("HERMES_TREEPS_DB_PATH", str(path))
    STORE.db.reset()
    return path


def test_an_existing_database_opens_and_keeps_its_rows(v2_database):
    """Opening a v2 database adds the new tables and loses nothing.

    Prevents: an upgrade that takes every live session's binding with it.
    """
    binding = STORE.db.read_binding("a-live-session")
    assert binding is not None, "the live binding vanished on open"
    assert binding["mode"] == "repo", binding
    assert binding["slug"] == "live-tree", binding

    conn = sqlite3.connect(v2_database)
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "child_runs" in tables, sorted(tables)
        assert "child_reports" in tables, sorted(tables)
        version = conn.execute(
            "SELECT value FROM kv WHERE key='schema_version'"
        ).fetchone()
        assert str(version[0]) == str(STORE.db.SCHEMA_VERSION), version
    finally:
        conn.close()


def test_the_new_tables_work_on_a_migrated_database(v2_database):
    """The child-report rows can be written on a migrated database."""
    STORE.db.write_child_run("child-1", "parent-1", child_role="worker")
    STORE.db.write_child_report(
        "child-1", "parent-1", child_status="completed", dirty="clean", unpushed=0
    )
    run = STORE.db.read_child_run("child-1")
    assert run is not None and run["parent_session_id"] == "parent-1", run
    reports = STORE.db.read_child_reports("parent-1")
    assert len(reports) == 1, reports


def test_a_fresh_database_has_the_new_tables(tmp_path, monkeypatch):
    """A fresh install gets the new tables from the schema script."""
    path = tmp_path / "plugin-data" / "data.db"
    monkeypatch.setenv("HERMES_TREEPS_PLUGIN_DATA_DIR", str(path.parent))
    monkeypatch.setenv("HERMES_TREEPS_DB_PATH", str(path))
    STORE.db.reset()

    # Open a connection so the schema is actually created: the plugin creates
    # the database lazily, on the first use.
    STORE.db.connect()

    conn = sqlite3.connect(path)
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"child_runs", "child_reports"} <= tables, sorted(tables)
    finally:
        conn.close()
