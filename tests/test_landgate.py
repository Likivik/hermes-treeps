"""The landing gate: a red child verdict must stop the parent's push.

Each test names the failure it prevents. The two incidents this module exists
for were both "the parent landed while a child's verdict was red" - so the
tests are about that, not about the regexes.
"""

from __future__ import annotations

import sys
import types

import pytest
from plugin_loader import PLUGIN, STORE


@pytest.fixture
def gate():
    """The gate module, imported as part of the plugin package."""
    import hermes_treeps.landgate as landgate

    return landgate


@pytest.fixture
def fake_verify(monkeypatch):
    """A fake ``verify`` module whose ``landing_blocker`` the test controls."""

    def install(result=None, raises=None):
        mod = types.ModuleType("hermes_treeps.verify")

        def landing_blocker(session_id):
            if raises is not None:
                raise raises
            return result

        mod.landing_blocker = landing_blocker
        monkeypatch.setitem(sys.modules, "hermes_treeps.verify", mod)
        monkeypatch.setattr(PLUGIN, "verify", mod, raising=False)
        return mod

    install.result = None
    install.raises = None
    return install


def _settings(monkeypatch, **overrides):
    """Patch the plugin settings the gate reads."""
    base = {
        "landing_gate_enabled": True,
        "verify_command": "hermes ws verify",
    }
    base.update(overrides)
    monkeypatch.setattr(STORE, "settings", lambda: dict(base))
    return base


def test_blocks_git_push_when_a_child_is_red(gate, fake_verify, monkeypatch):
    """A red child stops ``git push``.

    Prevents: the parent pushing a tree whose child's verification is red -
    exactly the incident this gate was written for.
    """
    _settings(monkeypatch)
    fake_verify(["child-red-1"])
    res = gate.check("terminal", {"command": "git push origin x"}, "sid-parent")
    assert res is not None, "git push must be blocked while a child is red"
    assert res["action"] == "block"


def test_blocks_gh_pr_merge(gate, fake_verify, monkeypatch):
    """Merging a PR is landing too."""
    _settings(monkeypatch)
    fake_verify(["child-red-1"])
    res = gate.check("terminal", {"command": "gh pr merge 12 --squash"}, "sid-parent")
    assert res is not None and res["action"] == "block", res


def test_blocks_jj_git_push(gate, fake_verify, monkeypatch):
    """The jj VCS pushes through the same gate."""
    _settings(monkeypatch)
    fake_verify(["child-red-1"])
    res = gate.check("terminal", {"command": "jj git push"}, "sid-parent")
    assert res is not None and res["action"] == "block", res


def test_allows_ordinary_shell_commands(gate, fake_verify, monkeypatch):
    """Status, tests and logs are not landing."""
    _settings(monkeypatch)
    fake_verify(["child-red-1"])
    for command in ("git status", "pytest -q", "jj log", "git commit -m x"):
        assert gate.check("terminal", {"command": command}, "sid-parent") is None, command


def test_blocks_a_chained_push(gate, fake_verify, monkeypatch):
    """``cd x && git push`` is still a push."""
    _settings(monkeypatch)
    fake_verify(["child-red-1"])
    res = gate.check("terminal", {"command": "cd /tmp && git push origin x"}, "sid-parent")
    assert res is not None and res["action"] == "block", res


def test_allows_gh_pr_create(gate, fake_verify, monkeypatch):
    """Opening a pull request is not landing it."""
    _settings(monkeypatch)
    fake_verify(["child-red-1"])
    assert gate.check("terminal", {"command": "gh pr create --fill"}, "sid-parent") is None


def test_allows_everything_when_the_gate_is_disabled(gate, fake_verify, monkeypatch):
    """``landing_gate_enabled: false`` turns the gate off."""
    _settings(monkeypatch, landing_gate_enabled=False)
    fake_verify(["child-red-1"])
    assert gate.check("terminal", {"command": "git push origin x"}, "sid-parent") is None


def test_allows_when_the_blocker_finds_nothing(gate, fake_verify, monkeypatch):
    """No red verdict -> no block."""
    _settings(monkeypatch)
    fake_verify(None)
    assert gate.check("terminal", {"command": "git push origin x"}, "sid-parent") is None


def test_allows_when_the_verify_module_is_missing(gate, monkeypatch):
    """Without the verify module there is no verdict, so nothing is blocked."""
    _settings(monkeypatch)
    monkeypatch.delitem(sys.modules, "hermes_treeps.verify", raising=False)
    monkeypatch.delattr(PLUGIN, "verify", raising=False)
    assert gate.check("terminal", {"command": "git push origin x"}, "sid-parent") is None


def test_block_message_names_the_child(gate, fake_verify, monkeypatch):
    """The message must say WHICH child is red and what to do about it."""
    _settings(monkeypatch)
    fake_verify(["child-red-42"])
    res = gate.check("terminal", {"command": "git push origin x"}, "sid-parent")
    assert res is not None
    assert "child-red-42" in res["message"], res["message"]
    assert "verify" in res["message"], res["message"]


def test_a_raising_blocker_does_not_crash_the_call(gate, fake_verify, monkeypatch):
    """The gate's own bugs fail OPEN: a broken blocker must not wedge the host."""
    _settings(monkeypatch)
    fake_verify(raises=RuntimeError("boom"))
    assert gate.check("terminal", {"command": "git push origin x"}, "sid-parent") is None


def test_non_terminal_tools_are_ignored(gate, fake_verify, monkeypatch):
    """Only the shell tool is inspected."""
    _settings(monkeypatch)
    fake_verify(["child-red-1"])
    assert gate.check("read_file", {"command": "git push"}, "sid-parent") is None


def test_landing_tools_names_the_shell_tool(gate):
    """The gate's tool set is explicit."""
    assert "terminal" in gate.landing_tools()


def test_missing_settings_key_does_not_crash(gate, fake_verify, monkeypatch):
    """An old config without the new keys must not break the gate."""
    monkeypatch.setattr(STORE, "settings", lambda: {})
    fake_verify(["child-red-1"])
    res = gate.check("terminal", {"command": "git push origin x"}, "sid-parent")
    # The key is absent: the gate stays enabled and blocks a red child.
    assert res is not None and res["action"] == "block", res
