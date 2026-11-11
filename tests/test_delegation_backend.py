"""delegation_backend.py: overflow past the MiniMax slots onto a second route.

WHAT this defends: the delegation seam provider must (a) keep the primary
route for the first ``capacity_overflow_after`` MiniMax-class children and
overflow the rest, (b) log one INFO line per decision, and (c) stay entirely
inert when the host predates the seam (today's 0.21.7) - the plugin must load
and register cleanly there.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types

import pytest
from plugin_loader import STORE

BACKEND = importlib.import_module("hermes_treeps.delegation_backend")
CAPACITY = importlib.import_module("hermes_treeps.capacity")

CONFIG = {"provider": "custom:neuralwatt", "model": "glm-5.3"}

BACKEND_SETTINGS = {
    "capacity_enabled": True,
    "capacity_max_children": 8,
    "capacity_overflow_after": 4,
    "capacity_overflow_provider": "custom:neuralwatt",
    "capacity_overflow_model": "deepseek-v4.1-flash-flex",
    "capacity_tracking_scope": "profile",
    "capacity_tracking_profiles": [],
    "capacity_ledger_dir": "",
}


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(STORE, "settings", lambda: dict(BACKEND_SETTINGS))
    return BACKEND.HermesTreepsDelegationProvider()


def test_a_six_task_batch_splits_four_default_two_overflow(backend, monkeypatch):
    monkeypatch.setattr(
        CAPACITY, "count_live", lambda **kw: 0
    )  # live=0, overflow_after=4
    routes = [backend.route_task({"goal": f"g{i}", "index": i}, CONFIG) for i in range(6)]
    for r in routes[:4]:
        assert r == CONFIG, r  # the default route: inherit the delegation config
    for r in routes[4:]:
        assert r == {
            "provider": "custom:neuralwatt",
            "model": "deepseek-v4.1-flash-flex",
            "label": "overflow",
        }, r


def test_live_minimax_children_shrink_the_default_slots(backend, monkeypatch):
    monkeypatch.setattr(CAPACITY, "count_live", lambda **kw: 2)
    routes = [backend.route_task({"goal": "g", "index": i}, CONFIG) for i in range(3)]
    assert routes[0] == CONFIG and routes[1] == CONFIG, routes
    assert routes[2].get("label") == "overflow", routes


def test_one_info_line_per_decision(backend, monkeypatch, caplog):
    monkeypatch.setattr(CAPACITY, "count_live", lambda **kw: 0)
    with caplog.at_level(logging.INFO, logger="hermes_treeps.delegation_backend"):
        for i in range(3):
            backend.route_task({"goal": "g", "index": i}, CONFIG)
    lines = [r.message for r in caplog.records if "child" in r.message]
    assert len(lines) == 3, caplog.records
    # Each line names the child index, the route, and the overflow decision.
    assert "child 0" in lines[0] and "child 2" in lines[2], lines
    assert "overflow=no" in lines[0], lines


def test_register_provider_returns_false_when_the_seam_is_absent(caplog):
    """On Hermes 0.21.7 (no agent.delegation_provider) the half stays inert."""
    class _Ctx:
        def register_delegation_provider(self, provider):
            raise AssertionError("must not register without the seam")

    with caplog.at_level(logging.INFO, logger="hermes_treeps.delegation_backend"):
        assert BACKEND.register_provider(_Ctx()) is False
    assert any("absent" in r.message for r in caplog.records), caplog.records


def test_register_provider_attaches_the_seam_when_present(monkeypatch):
    """When the seam exists, the provider is registered through ctx."""
    registered = {}

    base = types.ModuleType("agent")
    sub = types.ModuleType("agent.delegation_provider")

    class DelegationProvider:
        registered_providers = []

        def settings(self, config):
            return {}

    sub.DelegationProvider = DelegationProvider
    base.delegation_provider = sub
    monkeypatch.setitem(sys.modules, "agent", base)
    monkeypatch.setitem(sys.modules, "agent.delegation_provider", sub)

    class _Ctx:
        def register_delegation_provider(self, provider):
            registered["provider"] = provider

    assert BACKEND.register_provider(_Ctx()) is True
    provider = registered["provider"]
    assert provider.name == "hermes-treeps", provider.name
    assert callable(provider.route_task)


def test_the_module_imports_without_the_seam():
    """The fresh-interpreter rule: no host import at module import time.

    Mirrors tests/test_module_layout.py's loader: a fresh interpreter imports
    the plugin by file location the way the host does, then pulls in the
    backend module. On a host with no ``agent`` package at all (0.21.7, and
    this test machine) both imports must succeed.
    """
    import subprocess
    from pathlib import Path

    root = Path(BACKEND.__file__).resolve().parent
    prelude = f"""
import importlib.util, sys
from pathlib import Path
ns = type(sys)("hermes_plugins"); ns.__path__ = []; ns.__package__ = "hermes_plugins"
sys.modules["hermes_plugins"] = ns
spec = importlib.util.spec_from_file_location(
    "hermes_plugins.hermes_treeps", {str(root / "__init__.py")!r},
    submodule_search_locations=[{str(root)!r}])
module = importlib.util.module_from_spec(spec)
module.__package__ = "hermes_plugins.hermes_treeps"
module.__path__ = [{str(root)!r}]
sys.modules["hermes_plugins.hermes_treeps"] = module
spec.loader.exec_module(module)
importlib.import_module("hermes_plugins.hermes_treeps.delegation_backend")
assert "agent" not in sys.modules, "the backend imported a host module eagerly"
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", prelude],
        capture_output=True,
        text=True,
        cwd=str(root),
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "ok" in result.stdout


def test_a_task_without_an_index_defaults_to_zero(backend, monkeypatch):
    monkeypatch.setattr(CAPACITY, "count_live", lambda **kw: 0)
    assert backend.route_task({"goal": "g"}, CONFIG) == CONFIG
