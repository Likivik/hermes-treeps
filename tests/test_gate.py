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
    # (Today there is no such pointer: the parent is resolved per child from
    # the child's own row, so a stranger cannot be mistaken for a child.)
    try:
        res = _gate("terminal", "sid-stranger")
        assert res is not None and res["action"] == "block", (
            "an unrelated session must not be treated as a delegated child"
        )
        assert STORE.read_state("sid-stranger") is None, (
            "an unrelated session must not inherit another session's binding"
        )
    finally:
        pass


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


def test_chatting_pins_hermes_home(monkeypatch, state_home):
    """Chatting re-points the cwd at HERMES_HOME instead of clearing it.

    Prevents: clearing the override falls back to os.getcwd() - the gateway's
    own directory - when TERMINAL_CWD is unset, so a "talk only" session could
    still write into whatever directory the gateway was launched from.
    """
    fake = _install_fake_runtime_cwd(monkeypatch, path="/somewhere")
    PLUGIN._tool_chatting({}, session_id="sid-chat-pin")
    assert fake.pinned, "ws_chatting left the previous workspace pinned"
    assert str(STORE.hermes_home()) in fake.pinned[-1], fake.pinned


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


def test_the_gate_fails_closed_when_its_own_state_read_breaks(monkeypatch, state_home):
    """A database error blocks gated tools instead of silently allowing them.

    Prevents the hole the audits found: `except Exception: return None` meant a
    locked or corrupt db let `terminal` run for a session that was never bound.
    Repair stays possible - read-only tools and the ws_* tools are not gated.
    """
    def boom(*a, **kw):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(PLUGIN.store, "resolve", boom)
    res = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="sid-broken-db")
    assert res is not None and res["action"] == "block", res
    assert "unavailable" in res["message"], res

    assert PLUGIN._on_pre_tool_call(tool_name="read_file", session_id="sid-broken-db") is None, (
        "read-only tools must stay usable so the session can be repaired"
    )


def test_the_child_branch_runs_before_the_pinned_fallback(monkeypatch, state_home):
    """A delegated child must reach childspace even though it inherits the
    parent's pinned cwd.

    Prevents: the pinned-cwd fallback used to return before the child branch,
    and a child inherits the parent's ContextVar pin - so every child passed
    the gate inside the PARENT's tree and the child-workspace isolation was
    dead code.
    """
    import sys
    import types
    _install_fake_runtime_cwd(monkeypatch, path="/Git/hermes/workspaces/repo/parent")
    # A bound parent's pin is in the child's inherited context; force the
    # fallback to fire so the test proves the ORDER, not the fallback's mood.
    monkeypatch.setattr(PLUGIN, "_pinned_workspace", lambda: "/Git/hermes/workspaces/repo/parent")

    child_calls = []
    fake_childspace = types.ModuleType("childspace")
    fake_childspace.ensure_child_workspace = lambda sid, parent_state, **kw: (
        child_calls.append((sid, dict(parent_state)))
        or {"ok": True, "path": "/Git/hermes/workspaces/repo/parent-child-abc123", "slug": "child"}
    )
    monkeypatch.setitem(sys.modules, "childspace", fake_childspace)
    monkeypatch.setattr(PLUGIN, "childspace", fake_childspace)

    fake_delegation = types.ModuleType("agent.delegation_context")
    fake_delegation.is_delegated_child_context = lambda: True
    monkeypatch.setitem(sys.modules, "agent.delegation_context", fake_delegation)

    STORE.db.write_child_run("child-abc123", "parent-1")
    monkeypatch.setattr(PLUGIN.store, "read_state", lambda sid: {"mode": "repo", "repo": "repo", "slug": "parent", "path": "/Git/hermes/workspaces/repo/parent"})
    res = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="child-abc123")
    assert res is None, res
    assert child_calls, "the child never reached childspace - it stayed in the parent's tree"


def test_fail_closed_still_allows_a_session_pinned_into_a_workspace(monkeypatch, state_home):
    """The error path must accept the same evidence the normal path accepts.

    Prevents: a session sitting in a real workspace being blocked by a db
    hiccup, then allowed the moment the db recovers - the two branches
    disagreeing about what a pinned cwd proves.
    """
    _install_fake_runtime_cwd(monkeypatch, path="/Git/hermes/workspaces/repo/ws")
    monkeypatch.setattr(PLUGIN, "_pinned_workspace", lambda: "/Git/hermes/workspaces/repo/ws")
    monkeypatch.setattr(PLUGIN.store, "resolve", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("database is locked")))
    assert PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="sid-pinned-ok") is None


def test_a_chatting_parent_may_delegate_but_an_unbound_one_may_not(monkeypatch, state_home):
    """delegate_task is gated: no workspace binding, no delegation.

    Prevents: an unbound session spawning tool-running children that inherit
    nothing and land nowhere - the exact residue the 10-minute pointer window
    used to create.
    """
    res = PLUGIN._on_pre_tool_call(tool_name="delegate_task", session_id="sid-no-binding")
    assert res is not None and res["action"] == "block", res

    PLUGIN._tool_chatting({}, session_id="sid-chat-delegate")
    assert PLUGIN._on_pre_tool_call(tool_name="delegate_task", session_id="sid-chat-delegate") is None, (
        "a chatting session is bound (to HERMES_HOME) and must be able to delegate"
    )


def test_the_fail_closed_message_names_the_actual_error(monkeypatch, state_home):
    """The block message must carry the exception, not point at a log line.

    Prevents: 'Report the error above' with nothing above - the error lived
    only in the gateway log.
    """
    monkeypatch.setattr(PLUGIN.store, "resolve", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("database is locked")))
    res = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="sid-db-err")
    assert "database is locked" in res["message"], res


def test_a_failed_childspace_for_a_repo_parent_blocks_instead_of_inheriting(monkeypatch, state_home):
    """A git-only parent must not hand the child its own tree via inheritance.

    Prevents the P2 the second audit found: childspace refuses non-jj repos,
    the old fallback then wrote the PARENT's binding into the child - the
    two-agents-one-tree collision the child branch exists to prevent.
    """
    import types
    fake_childspace = types.ModuleType("childspace")
    fake_childspace.ensure_child_workspace = lambda sid, parent_state, **kw: {
        "ok": False, "reason": "repo 'repo' is not a jj repo - no child workspace",
    }
    monkeypatch.setattr(PLUGIN, "childspace", fake_childspace)
    fake_delegation = types.ModuleType("agent.delegation_context")
    fake_delegation.is_delegated_child_context = lambda: True
    monkeypatch.setitem(__import__("sys").modules, "agent.delegation_context", fake_delegation)
    STORE.db.write_child_run("child-git-parent", "parent-1")
    monkeypatch.setattr(PLUGIN.store, "read_state", lambda sid: {
        "mode": "repo", "repo": "repo", "slug": "parent", "path": "/Git/hermes/workspaces/repo/parent"})
    res = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="child-git-parent")
    assert res is not None and res["action"] == "block", res
    assert "own workspace" in res["message"], res


def test_a_stale_session_is_warned_even_when_pinned(monkeypatch, state_home):
    """The stale warning must survive the pinned-cwd floor.

    Prevents: a session whose binding names a dead tree but whose cwd is still
    pinned inside the workspace root was silently allowed - the user was never
    told the binding was dead.
    """
    monkeypatch.setattr(PLUGIN, "_pinned_workspace", lambda: "/Git/hermes/workspaces/repo/ws")
    monkeypatch.setattr(PLUGIN.store, "resolve", lambda *a, **kw: None)
    monkeypatch.setattr(PLUGIN.store, "resolve_detail", lambda *a, **kw: {
        "state": "stale", "reason": "path /old/tree no longer exists"})
    monkeypatch.setattr(PLUGIN.store, "candidate_repos", lambda limit=6: [])
    res = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="sid-stale-pinned")
    assert res is not None and res["action"] == "block", res
    assert "STALE" in res["message"], res


def test_a_bound_session_rearms_the_pin_after_a_restart(monkeypatch, state_home):
    """A resumed session must re-pin on its first gated call.

    Prevents: after a gateway restart the runtime pin is gone and
    on_session_start does not re-fire - the shell silently fell back to the
    gateway directory until the session rebound.
    """
    fake = _install_fake_runtime_cwd(monkeypatch, path="")
    monkeypatch.setattr(PLUGIN, "_pinned_workspace", lambda: "")
    monkeypatch.setattr(PLUGIN.store, "resolve", lambda *a, **kw: {
        "mode": "repo", "path": "/Git/hermes/workspaces/repo/ws"})
    assert PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="sid-resumed") is None
    assert any("workspaces/repo/ws" in p for p in fake.pinned), fake.pinned
