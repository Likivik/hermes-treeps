"""The gate: what an unbound / stale / bound session may and may not do.

Each test names the failure it prevents. The gate is the whole point of the
plugin, so its bypasses are the highest-consequence bugs in it.
"""

from __future__ import annotations

import json

from plugin_loader import PLUGIN, STORE


def _gate(tool: str, sid: str):
    return PLUGIN._on_pre_tool_call(tool_name=tool, session_id=sid)


def test_unbound_blocks_terminal(state_home):
    """Unbound session: the tools that touch a tree are refused.

    Prevents: an agent doing repo work with no workspace, in whatever cwd it
    inherited - the six-unattributable-directories failure.
    """
    res = _gate("terminal", "sid-unbound-1")
    assert res is not None, "terminal must be blocked for an unbound session"
    assert res["action"] == "block"
    assert "ws_new" in res["message"]


def test_unbound_blocks_browser_exec(state_home):
    """browser_exec runs arbitrary host Python - it must be gated too.

    Prevents (P0-1): the gate listed terminal/read_file/write_file/... but not
    browser_exec, which on this host runs arbitrary Python. An unbound session
    therefore had a FULL BYPASS of the gate: it could execute code on the host
    with the session's cwd, defeating the point of the gate entirely.
    """
    res = _gate("browser_exec", "sid-unbound-2")
    assert res is not None, "browser_exec must be blocked for an unbound session"
    assert res["action"] == "block"


def test_unbound_allows_read_file(state_home):
    """Read-only diagnosis is never gated: the gate must not block its repair.

    Prevents: a dead session that cannot even read the plugin's own source to
    explain why it is stuck.
    """
    assert _gate("read_file", "sid-unbound-3") is None


def test_stale_reports_the_missing_path(state_home, tmp_path):
    """A stale binding says WHICH tree vanished, not "no workspace".

    Prevents: a session whose workspace was moved being reported as merely
    unbound, so the user is asked a question whose answer they already gave.
    """
    gone = tmp_path / "moved-away"
    STORE.write_state("sid-stale", {"mode": "repo", "repo": "r", "slug": "s", "path": str(gone)})
    detail = STORE.resolve_detail("sid-stale")
    assert detail["state"] == "stale"
    res = _gate("terminal", "sid-stale")
    assert res is not None and res["action"] == "block"
    assert "path is gone" in res["message"]
    assert str(gone) in res["message"]


def test_bound_session_is_not_gated(state_home, tmp_path):
    """A bound session runs everything."""
    ws = tmp_path / "ws"
    ws.mkdir()
    STORE.write_state("sid-bound", {"mode": "repo", "repo": "r", "slug": "s", "path": str(ws)})
    assert STORE.resolve_detail("sid-bound")["state"] == "bound"
    assert _gate("terminal", "sid-bound") is None


def test_chatting_session_is_not_gated(state_home):
    """'Just chatting' is deliberately unrestricted."""
    STORE.mark_chatting("sid-chat")
    assert _gate("terminal", "sid-chat") is None


def test_forgetting_the_record_leaves_the_session_gated(state_home, tmp_path):
    """ws_doctor(forget=true) drops the record - it must NOT unlock the session.

    Prevents (P0, from #5): forgetting the record made the session look
    "no workspace" and the gate's answer for that was to let it through. A
    forgotten binding has to land back on the UNBOUND path, which is blocked.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    STORE.write_state("sid-forget", {"mode": "repo", "repo": "r", "slug": "s", "path": str(ws)})
    out = json.loads(PLUGIN._tool_doctor({"forget": True}, session_id="sid-forget"))
    assert out["forgot"] is True
    assert STORE.resolve_detail("sid-forget")["state"] == "unbound"
    res = _gate("terminal", "sid-forget")
    assert res is not None and res["action"] == "block", "forgetting must not unblock the session"


def test_unbound_message_offers_the_third_route(state_home):
    """The unbound gate must name all three routes, including the clone.

    Prevents (P2-3): the message listed only ws_new and ws_chatting, both of
    which assume the repo is already on disk. An agent whose repo lived on
    GitHub/Forgejo saw no sanctioned way forward and cloned it on its own
    initiative - so the gate now offers create_repo=true explicitly.
    """
    res = _gate("terminal", "sid-unbound-routes")
    assert res is not None and res["action"] == "block", res
    msg = res["message"]
    assert "ws_new(repo, slug)" in msg, msg
    assert "ws_chatting()" in msg, msg
    assert "create_repo" in msg, msg
    assert "GitHub/Forgejo" in msg, msg


def test_gate_tools_are_a_mapping_with_reasons():
    """Every gated tool carries WHY it is gated.

    Prevents (P0-1): a bare set of names gave no clue that browser_exec (host
    code execution) belonged there, so the next host tool was missed too.
    """
    assert isinstance(STORE.GATED_TOOLS, dict)
    for tool in ("terminal", "read_file", "write_file", "patch", "search_files",
                 "execute_code", "browser_exec", "computer_use", "cronjob_manage",
                 "process_manage"):
        assert tool in STORE.GATED_TOOLS, f"{tool} must be gated"
        assert STORE.GATED_TOOLS[tool].strip(), f"{tool} needs a reason"


def test_unbound_session_does_not_inherit_another_sessions_workspace(state_home, tmp_path):
    """An unrelated session must NOT inherit another session's binding.

    Prevents (P0-2): the 'batch child' fallback treated ANY unresolved
    session as a delegated child while a delegation pointer was younger than
    600 s, so a pre-existing unrelated session silently inherited someone
    else's workspace. The fallback is gone; only a real delegated child
    (is_delegated_child_context()) inherits.
    """
    other_ws = tmp_path / "other-ws"
    other_ws.mkdir()
    STORE.write_state("sid-owner", {"mode": "repo", "repo": "r", "slug": "s", "path": str(other_ws)})
    # A fresh, unrelated session with NO record of its own.
    assert STORE.read_state("sid-stranger") is None
    # Simulate a recent delegation pointer - the old fallback keyed off this.
    PLUGIN._LAST_DELEGATION["sid"] = "sid-owner"
    PLUGIN._LAST_DELEGATION["at"] = 1e12
    try:
        res = _gate("terminal", "sid-stranger")
        assert res is not None and res["action"] == "block", (
            "an unrelated session must not be treated as a delegated child"
        )
        assert STORE.read_state("sid-stranger") is None, (
            "an unrelated session must not inherit another session's binding"
        )
    finally:
        PLUGIN._LAST_DELEGATION["sid"] = ""
        PLUGIN._LAST_DELEGATION["at"] = 0.0


# ── cwd pinning ──────────────────────────────────────────────────────────────


class _FakeRuntimeCwd:
    """Stand-in for agent.runtime_cwd: records pins, answers scoped_session_cwd."""

    def __init__(self, path: str = ""):
        self.pinned: list[str] = []
        self.cleared = 0
        self._path = path

    def set_session_cwd(self, cwd):
        self.pinned.append(cwd)
        self._path = cwd
        return object()

    def clear_session_cwd(self):
        self.cleared += 1
        self._path = ""

    def scoped_session_cwd(self):
        return self._path


def _install_fake_runtime_cwd(monkeypatch, path: str = ""):
    import sys as _sys
    import types

    fake = _FakeRuntimeCwd(path)
    mod = types.ModuleType("agent.runtime_cwd")
    mod.set_session_cwd = fake.set_session_cwd
    mod.clear_session_cwd = fake.clear_session_cwd
    mod.scoped_session_cwd = fake.scoped_session_cwd
    monkeypatch.setitem(_sys.modules, "agent.runtime_cwd", mod)
    return fake


def test_binding_pins_the_session_cwd(tmp_path, monkeypatch, state_home):
    """ws_new moves the session into the workspace it just bound.

    Prevents: a bound session whose tools still run in the gateway's own
    directory - the gate said "bound" while every writes landed somewhere else.
    """
    fake = _install_fake_runtime_cwd(monkeypatch)
    ws = tmp_path / "ws"
    ws.mkdir()
    PLUGIN._pin_session_cwd(ws)
    assert fake.pinned == [str(ws)], fake.pinned


def test_chatting_clears_the_pin(monkeypatch, state_home):
    """Declaring chatting drops the pin, so the tree is not written by accident."""
    fake = _install_fake_runtime_cwd(monkeypatch, path="/somewhere")
    PLUGIN._tool_chatting({}, session_id="sid-chat-pin")
    assert fake.cleared == 1, "ws_chatting left the previous workspace pinned"


def test_gate_allows_when_the_pinned_cwd_is_a_workspace(tmp_path, monkeypatch, state_home):
    """A pinned cwd inside the workspace root satisfies the gate on its own.

    Prevents the worst failure mode: when the binding read fails, the gate used
    to fail open with no floor under it. Now the physical cwd is the floor.
    """
    ws_root = tmp_path / "workspaces"
    ws = ws_root / "repo" / "slug"
    ws.mkdir(parents=True)
    monkeypatch.setattr(STORE, "settings", lambda: {"workspace_root": str(ws_root)})
    _install_fake_runtime_cwd(monkeypatch, path=str(ws))
    res = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="sid-pinned-1")
    assert res is None, f"a pinned workspace must not be blocked: {res}"


def test_gate_blocks_when_the_pinned_cwd_is_outside(tmp_path, monkeypatch, state_home):
    """A pin outside the workspace root is not a workspace and must not pass."""
    ws_root = tmp_path / "workspaces"
    ws_root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setattr(STORE, "settings", lambda: {"workspace_root": str(ws_root)})
    _install_fake_runtime_cwd(monkeypatch, path=str(outside))
    res = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="sid-pinned-2")
    assert res is not None and res["action"] == "block", res
