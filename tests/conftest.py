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


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point the plugin's database (and every legacy path) at a tmp dir.
    WHY: the plugin's state lives in ONE sqlite file under the host's
    plugin-data directory. Without this fixture the suite reads and WRITES
    the live database - a test binding a session would overwrite a real
    session's binding, and rows from one test leak into the next. The
    environment overrides are the plugin's own supported test seams
    (``HERMES_TREEPS_DB_PATH`` and friends), so nothing has to be patched
    inside the plugin to keep the live install untouched.
    """
    monkeypatch.setenv("HERMES_TREEPS_PLUGIN_DATA_DIR", str(tmp_path / "plugin-data"))
    monkeypatch.setenv("HERMES_TREEPS_DB_PATH", str(tmp_path / "plugin-data" / "data.db"))
    monkeypatch.setenv("HERMES_TREEPS_LEGACY_STATE_DIR", str(tmp_path / "legacy-state"))
    monkeypatch.setenv("HERMES_TREEPS_LEGACY_WORKSPACE_ROOT", str(tmp_path / "legacy-workspaces"))
    # The archive of retired workspaces also lives under the Hermes home.
    monkeypatch.setattr(STORE, "hermes_home", lambda: tmp_path / "hermes-home")
    # Environment overrides win over config.yaml, so the suite must not run
    # with any of them set: a FORGEJO_URL in the developer's shell would
    # otherwise change what settings() returns and fail the defaults test.
    for var in STORE._settings.ENV_OVERRIDES:
        monkeypatch.delenv(var, raising=False)
    STORE.db.reset()
    yield
    STORE.db.reset()


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
