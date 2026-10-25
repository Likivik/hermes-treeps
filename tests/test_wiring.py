"""The four modules wired into the plugin, and what each wiring prevents.

These tests exist because a module that is imported but never called is a
feature that looks shipped and is not: the four modules landed green while
nothing in __init__.py called them.
"""

from __future__ import annotations

import sys
import types

from plugin_loader import PLUGIN, STORE


def _fake_child_context(monkeypatch, value: bool = True):
    mod = types.ModuleType("agent.delegation_context")
    mod.is_delegated_child_context = lambda: value
    monkeypatch.setitem(sys.modules, "agent.delegation_context", mod)


def test_a_child_gets_its_own_workspace_before_inheriting(tmp_path, monkeypatch, state_home):
    """The gate creates a child workspace and pins it, instead of inheriting.

    Prevents: two children editing the parent's single tree - the collision
    that let one agent overwrite another's uncommitted work.
    """
    ws = tmp_path / "repo" / "parent-child-abc123"
    ws.mkdir(parents=True)
    _fake_child_context(monkeypatch)
    STORE.write_state("parent-1", {"mode": "repo", "repo": "repo", "slug": "parent", "path": "/tmp/parent"})
    # The child->parent map is per child (child_runs), never a module global:
    # a global can only name ONE parent, so two concurrent delegations made
    # the second child inherit the first child's parent.
    STORE.db.write_child_run("child-1", "parent-1")
    calls = {}

    def fake_ensure(sid, parent_state, **kw):
        calls["sid"] = sid
        calls["parent_state"] = parent_state
        return {"ok": True, "reason": "", "repo": "repo", "slug": ws.name, "path": str(ws)}

    monkeypatch.setattr(PLUGIN.childspace, "ensure_child_workspace", fake_ensure)
    res = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="child-1")
    assert res is None, res
    assert calls["sid"] == "child-1", calls
    assert calls["parent_state"]["repo"] == "repo", calls


def test_the_prompt_section_carries_the_delegation_contract(monkeypatch):
    """The contract ships inside the system prompt, not in a skill file.

    Prevents: the rules being remembered only by an operator who is in a hurry.
    """
    registered = {}

    class Ctx:
        def register_tool(self, **kw):
            pass

        def register_cli_command(self, **kw):
            pass

        def register_command(self, *a, **kw):
            pass

        def register_hook(self, *a, **kw):
            pass

        def register_system_prompt_section(self, name, text, **kw):
            registered[name] = text

    monkeypatch.setattr(STORE, "candidate_repos", lambda limit=6: ["repo"])
    PLUGIN.register(Ctx())
    text = registered.get("workspace-discipline", "")
    assert PLUGIN.contract.SECTION_TITLE in text, text[:200]
    assert "ws_new" in text


