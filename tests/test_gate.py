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
