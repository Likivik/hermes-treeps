"""Shared fixtures for the plugin's test suite.

The plugin directory is a PACKAGE (``__init__.py`` next to ``store.py``), so
pytest's default rootdir-based collection tries to import the plugin's
``__init__.py`` as a test module and fails on the relative import
``from . import store``. ``pytest.ini`` pins ``testpaths = tests`` and
``--import-mode=importlib``; this file provides the fixtures.
"""

from __future__ import annotations

import pytest
from plugin_loader import PLUGIN, STORE


@pytest.fixture(scope="session")
def plugin():
    return PLUGIN


@pytest.fixture(scope="session")
def store():
    return STORE


@pytest.fixture
def state_home(tmp_path, monkeypatch):
    """Point the plugin's per-session state at a tmp dir.

    WHY: the live ``state/`` directory holds real session records. A test that
    wrote there would corrupt other sessions' bindings (and the gate tests
    would read whatever the last real session left behind). Every state write
    in the suite goes through this fixture instead.
    """
    d = tmp_path / "state"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(STORE, "state_dir", lambda: d)
    return d
