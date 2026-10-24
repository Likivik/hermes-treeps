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
    monkeypatch.setattr(PLUGIN, "_LAST_DELEGATION", {"sid": "parent-1", "at": __import__("time").time()})
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


def test_the_landing_gate_runs_before_the_binding_check(monkeypatch, state_home):
    """A red child verdict blocks `git push` even for a perfectly bound parent.

    Prevents: the parent merging unverified work because the workspace binding
    was fine - the binding says where you are, not whether the work passes.
    """
    monkeypatch.setattr(
        PLUGIN.landgate,
        "check",
        lambda tool_name, args, session_id: {"action": "block", "message": "child-9 verification FAILED"},
    )
    res = PLUGIN._on_pre_tool_call(
        tool_name="terminal", session_id="parent-2", args={"command": "git push origin x"}
    )
    assert res is not None and res["action"] == "block", res
    assert "child-9" in res["message"], res


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
