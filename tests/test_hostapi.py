"""The host adapter degrades; it never raises mid-turn."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import plugin_loader  # noqa: E402

HOST = plugin_loader.PLUGIN.hostapi


@pytest.fixture(autouse=True)
def no_host(monkeypatch):
    monkeypatch.setattr(HOST, "_import", lambda module, name: None)
    monkeypatch.setattr(HOST, "_LOGGED", set())


def test_every_call_degrades_without_the_host():
    assert HOST.session_key() == ""
    assert HOST.set_logical_cwd("/x") is False
    assert HOST.clear_logical_cwd() is False
    assert HOST.scoped_cwd() == ""
    assert HOST.record_terminal_cwd("k", "/x") is False
    assert HOST.clear_terminal_cwd("k") is False
    assert HOST.is_delegated_child() is False


def test_empty_key_is_refused_before_reaching_the_host(monkeypatch):
    called = []
    monkeypatch.setattr(HOST, "_import", lambda m, n: (lambda *a: called.append(a)))
    assert HOST.record_terminal_cwd("", "/x") is False
    assert HOST.clear_terminal_cwd("") is False
    assert called == [], "an empty key would collapse to 'default' and re-point every session"


def test_host_exception_is_swallowed(monkeypatch):
    def boom(*_a):
        raise RuntimeError("host exploded")

    monkeypatch.setattr(HOST, "_import", lambda m, n: boom)
    assert HOST.set_logical_cwd("/x") is False
    assert HOST.is_delegated_child() is False


def test_idle_task_below_0232_is_a_clean_no():
    class OldCtx:
        pass

    assert HOST.register_idle_task(OldCtx(), "reconcile", interval_hours=6, min_idle_hours=1, fn=lambda: None) is False


def test_idle_task_registers_when_available():
    seen = {}

    class NewCtx:
        def register_idle_task(self, name, **kw):
            seen[name] = kw

    assert HOST.register_idle_task(NewCtx(), "reconcile", interval_hours=6, min_idle_hours=1, fn="f") is True
    assert seen["reconcile"]["interval_hours"] == 6


def test_pin_requires_pinned_flag_and_future_expiry(tmp_path, monkeypatch):
    import importlib
    import sqlite3
    import time

    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute(
        "CREATE TABLE resumable_subagents (subagent_id TEXT, state TEXT, pinned INTEGER,"
        " record_json TEXT, expires_at REAL)"
    )
    now = time.time()
    conn.execute("INSERT INTO resumable_subagents VALUES ('a','resumable',1,?,?)", ('{"worktree": "/w/kid1"}', now + 3600))
    conn.execute("INSERT INTO resumable_subagents VALUES ('b','resumable',0,?,?)", ('{"worktree": "/w/kid2"}', now + 3600))
    conn.execute("INSERT INTO resumable_subagents VALUES ('c','resumable',1,?,?)", ('{"worktree": "/w/kid3"}', now - 60))
    conn.commit()
    conn.close()

    monkeypatch.setattr(HOST, "_LOGGED", set())
    db_mod = importlib.import_module("hermes_treeps.db")
    monkeypatch.setattr(db_mod, "hermes_home", lambda: tmp_path)
    assert HOST.is_pinned_child("/w/kid1") is True
    assert HOST.is_pinned_child("/w/kid2") is False, "not pinned"
    assert HOST.is_pinned_child("/w/kid3") is False, "expired"
