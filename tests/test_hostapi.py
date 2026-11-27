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
