"""Tests for child-end reporting: the parent learns what its child left.

The requirement these tests defend, in one sentence: after a delegated child
finishes, the PARENT session's next turn sees ONE line - which workspace the
child used, clean-or-dirty, and how many unpushed commits it left.

Every test here is about a failure that would otherwise be SILENT:

* a child that ends without a word (no start row, no stop row, no report);
* a report that repeats every turn instead of being delivered once;
* a report that says "clean" for a workspace the check could not read;
* a stop hook that blocks the parent's thread on a git subprocess;
* a broken store call that takes the whole delegation down with it.

No network, no real jj repo: the workspace checks are monkeypatched and every
write goes to the tmp database the ``isolated_db`` fixture points at.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time

import pytest
from plugin_loader import PLUGIN, STORE

PARENT = "parent-session-1"
CHILD = "child-session-abcdef"


@pytest.fixture
def child_workspace(tmp_path):
    """A workspace path the checks can pretend to inspect."""
    ws = tmp_path / "child-tree"
    ws.mkdir(parents=True)
    return ws


@pytest.fixture
def bound_child(child_workspace):
    """A child session bound to its own workspace, like the gate binds it."""
    STORE.write_state(
        CHILD,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": "parent-child-abcdef",
            "path": str(child_workspace),
            "inherited_from": PARENT,
        },
    )
    return child_workspace


def _wait_for_report(parent=PARENT, timeout=5.0):
    """Block until the report thread wrote its row. Returns the rows."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = STORE.db.read_child_reports(parent)
        if rows:
            return rows
        time.sleep(0.01)
    return STORE.db.read_child_reports(parent)


# ── (a) the start side writes its row ───────────────────────────────────────


def test_subagent_start_writes_the_child_mapping(bound_child):
    """The start hook records child -> parent, so the stop side never guesses.

    Prevents: a report landing in the wrong session because the stop side
    had to reverse-lookup which parent the child belonged to.
    """
    PLUGIN._on_subagent_start(
        parent_session_id=PARENT,
        child_session_id=CHILD,
        child_subagent_id="sub-1",
        parent_turn_id="turn-1",
        parent_subagent_id="",
        child_role="worker",
        child_goal="do the thing",
    )
    run = STORE.db.read_child_run(CHILD)
    assert run is not None, "the start hook wrote no row"
    assert run["parent_session_id"] == PARENT, run
    assert run["child_subagent_id"] == "sub-1", run
    assert run["child_role"] == "worker", run
    assert run["child_goal"] == "do the thing", run


def test_the_start_hook_never_raises(monkeypatch):
    """A broken store call must not break the delegation it is announcing.

    Prevents: the plugin turning every delegation into a failed one because
    a reporting row could not be written.
    """
    def boom(*a, **kw):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(STORE.db, "write_child_run", boom)
    PLUGIN._on_subagent_start(parent_session_id=PARENT, child_session_id=CHILD)
    # No exception escaped: reaching this line IS the assertion.


# ── (b) the stop side writes the report ─────────────────────────────────────


def test_subagent_stop_reports_dirty_and_unpushed(bound_child, monkeypatch):
    """A dirty child with unpushed commits says so, with the count."""
    monkeypatch.setattr(STORE, "dirty_state", lambda ws: "dirty")
    monkeypatch.setattr(STORE, "unpushed_commits", lambda ws: True)
    monkeypatch.setattr(STORE, "count_unpushed_commits", lambda ws: 3)

    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="completed",
    )
    rows = _wait_for_report()
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["parent_session_id"] == PARENT, row
    assert row["child_session_id"] == CHILD, row
    assert row["dirty"] == "dirty", row
    assert row["unpushed"] == 3, row
    assert row["child_status"] == "completed", row
    assert row["workspace_slug"] == "parent-child-abcdef", row
    assert row["workspace_path"] == str(bound_child), row


def test_subagent_stop_reports_a_clean_child(bound_child, monkeypatch):
    """A clean child with nothing unpushed says exactly that."""
    monkeypatch.setattr(STORE, "dirty_state", lambda ws: "clean")
    monkeypatch.setattr(STORE, "unpushed_commits", lambda ws: False)

    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="completed",
    )
    rows = _wait_for_report()
    assert len(rows) == 1, rows
    assert rows[0]["dirty"] == "clean", rows[0]
    assert rows[0]["unpushed"] == 0, rows[0]


# ── (c) the parent sees the line, once ──────────────────────────────────────


def test_the_parents_next_turn_carries_the_line_and_clears_the_rows(bound_child, monkeypatch):
    """The line reaches the parent's next turn, and only that turn.

    Prevents: a report that repeats every turn (noise) or that never
    arrives (a parent that never learns what its child left behind).
    """
    monkeypatch.setattr(STORE, "dirty_state", lambda ws: "dirty")
    monkeypatch.setattr(STORE, "unpushed_commits", lambda ws: True)
    monkeypatch.setattr(STORE, "count_unpushed_commits", lambda ws: 2)

    PLUGIN._on_subagent_start(parent_session_id=PARENT, child_session_id=CHILD)
    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="completed",
    )
    assert _wait_for_report(), "the report row was never written"

    out = PLUGIN._on_pre_llm_call(session_id=PARENT)
    assert out is not None, "the parent got no context at all"
    text = out["context"]
    assert "parent-child-abcdef" in text, text
    assert "dirty" in text, text
    assert "2 unpushed commits" in text, text
    assert str(bound_child) in text, text
    assert "completed" in text, text

    # ONE-TIME DELIVERY: the rows are gone, so the next turn is silent.
    assert STORE.db.read_child_reports(PARENT) == [], STORE.db.read_child_reports(PARENT)
    assert PLUGIN._on_pre_llm_call(session_id=PARENT) is None


def test_the_prompt_section_carries_the_report_too(bound_child, monkeypatch):
    """The registered system-prompt section renders the same report.

    WHY both channels: the host freezes plugin prompt sections per session,
    so a section cannot carry a report about a child that finishes later.
    The section covers the sessions whose prompt is built after the child
    stopped; the per-turn hook covers the rest.
    """
    monkeypatch.setattr(STORE, "dirty_state", lambda ws: "clean")
    monkeypatch.setattr(STORE, "unpushed_commits", lambda ws: False)
    STORE.db.write_child_report(
        CHILD, PARENT, child_status="completed",
        workspace_path=str(bound_child), workspace_slug="parent-child-abcdef",
        dirty="clean", unpushed=0,
    )
    text = PLUGIN._reports_prompt_section({"session_id": PARENT})
    assert text, "the section rendered nothing"
    assert PLUGIN.contract.REPORTS_SECTION_TITLE in text, text
    assert "parent-child-abcdef" in text, text
    # And it clears: the section is a delivery, not a poll.
    assert PLUGIN._reports_prompt_section({"session_id": PARENT}) == ""


def test_the_prompt_section_works_with_the_hosts_read_only_mapping(bound_child):
    """The section must render with the host's ``MappingProxyType``.

    WHY this test exists: the host passes the session info to a callable
    section as ``types.MappingProxyType``, which is NOT a dict. A section
    that checked ``isinstance(info, dict)`` would render nothing on every
    real session - a feature that looks shipped and is not, and no
    monkeypatched test would have caught it.
    """
    import types

    STORE.db.write_child_report(
        CHILD, PARENT, child_status="completed",
        workspace_path=str(bound_child), workspace_slug="parent-child-abcdef",
        dirty="clean", unpushed=0,
    )
    frozen = types.MappingProxyType({"session_id": PARENT, "model": "m", "platform": "cli"})
    text = PLUGIN._reports_prompt_section(frozen)
    assert text, "the section rendered nothing for the host's real mapping type"
    assert "parent-child-abcdef" in text, text


# ── (d) stop without start falls back to the bindings ───────────────────────


def test_stop_without_start_falls_back_to_the_bindings_table(bound_child, monkeypatch):
    """A child that started before the plugin was installed still reports.

    Prevents: a silent gap whenever the plugin is enabled mid-child - the
    child's own binding still names its workspace, so the report is possible.
    """
    assert STORE.db.read_child_run(CHILD) is None, "the test needs a child with no start row"
    monkeypatch.setattr(STORE, "dirty_state", lambda ws: "dirty")
    monkeypatch.setattr(STORE, "unpushed_commits", lambda ws: False)

    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="failed",
    )
    rows = _wait_for_report()
    assert len(rows) == 1, rows
    assert rows[0]["workspace_slug"] == "parent-child-abcdef", rows[0]
    assert rows[0]["dirty"] == "dirty", rows[0]
    assert rows[0]["child_status"] == "failed", rows[0]


# ── (e) a broken check yields state-unknown, and the hook survives ──────────


def test_a_broken_dirty_check_yields_state_unknown_and_the_hook_survives(bound_child, monkeypatch):
    """A failed check must read as UNKNOWN, not as "clean".

    Prevents: a report that lies. "state-unknown" is the honest answer; the
    whole feature exists to stop the parent from being told a workspace is
    clean when nobody managed to look.
    """
    def boom(ws):
        raise RuntimeError("jj is not a repo")

    monkeypatch.setattr(STORE, "dirty_state", boom)
    monkeypatch.setattr(STORE, "unpushed_commits", lambda ws: None)

    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="completed",
    )
    rows = _wait_for_report()
    assert len(rows) == 1, rows
    assert rows[0]["dirty"] == "state-unknown", rows[0]
    assert rows[0]["unpushed"] is None, rows[0]

    out = PLUGIN._on_pre_llm_call(session_id=PARENT)
    assert out is not None
    assert "state-unknown" in out["context"], out["context"]


def test_a_workspace_that_is_not_a_checkout_reports_state_unknown(bound_child):
    """The REAL check, unmocked: a directory that is not a checkout.

    WHY this test exists and why it is not mocked: ``store.dirty`` (the
    retire guard's boolean) answers "no changes" for a directory that is
    not a checkout at all - the status command fails with empty output and
    empty output reads as "clean". The reporting path uses the three-way
    ``dirty_state`` for exactly this reason, and this test proves the two
    really do differ.
    """
    not_a_repo = bound_child / "not-a-repo"
    not_a_repo.mkdir()

    # The retire guard's boolean says "clean"...
    assert STORE.dirty(not_a_repo) is False, "the boolean guard changed"

    # ...and the reporting path says "cannot tell".
    assert STORE.dirty_state(not_a_repo) is None

    STORE.write_state(
        CHILD,
        {"mode": "repo", "repo": "repo", "slug": "parent-child-abcdef",
         "path": str(not_a_repo)},
    )
    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="completed",
    )
    rows = _wait_for_report()
    assert len(rows) == 1, rows
    assert rows[0]["dirty"] == "state-unknown", rows[0]


def test_a_missing_workspace_reports_state_unknown(tmp_path):
    """A workspace that is GONE reports state-unknown, not clean."""
    gone = tmp_path / "gone"
    STORE.write_state(
        CHILD,
        {"mode": "repo", "repo": "repo", "slug": "parent-child-abcdef",
         "path": str(gone)},
    )
    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="completed",
    )
    rows = _wait_for_report()
    assert len(rows) == 1, rows
    assert rows[0]["dirty"] == "state-unknown", rows[0]
    assert rows[0]["unpushed"] is None, rows[0]


def test_a_broken_store_call_never_escapes_the_stop_hook(bound_child, monkeypatch):
    """A broken store call must never crash the delegation.

    Prevents: the reporting feature taking down the delegation it watches.
    """
    def boom(*a, **kw):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(STORE.db, "write_child_report", boom)
    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="completed",
    )
    # Reaching this line IS the assertion: no exception escaped the hook.


# ── (f) silent when there is nothing to report ───────────────────────────────


def test_a_failed_render_leaves_the_report_pending(bound_child, monkeypatch):
    """A broken render must not cost the parent its report.

    Prevents: the plugin being the thing that broke swallowing the report
    anyway. The rows are cleared only AFTER a successful render.
    """
    monkeypatch.setattr(STORE, "dirty_state", lambda ws: "clean")
    monkeypatch.setattr(STORE, "unpushed_commits", lambda ws: False)
    STORE.db.write_child_report(
        CHILD, PARENT, child_status="completed",
        workspace_path=str(bound_child), workspace_slug="parent-child-abcdef",
        dirty="clean", unpushed=0,
    )

    def boom(reports):
        raise RuntimeError("renderer exploded")

    # The renderer is reached through the ``contract`` MODULE, not through a
    # bound name - so the patch has to be on the module attribute.
    monkeypatch.setattr(PLUGIN.contract, "render_reports", boom)
    assert PLUGIN._on_pre_llm_call(session_id=PARENT) is None
    # The report is STILL pending: the parent's next turn gets it.
    rows = STORE.db.read_child_reports(PARENT)
    assert len(rows) == 1, f"a failed render swallowed the report: {rows}"


def test_a_parent_with_no_finished_children_gets_nothing():
    """No pending reports means no context and no section - not an empty one."""
    assert PLUGIN._on_pre_llm_call(session_id=PARENT) is None
    assert PLUGIN._reports_prompt_section({"session_id": PARENT}) == ""
    assert PLUGIN.contract.render_reports([]) == ""


def test_the_line_format_is_pinned():
    """The line says the four things, in the documented order.

    WHY pinned: the parent reads this line to decide what to do next. A
    reworded line is not a cosmetic change - "2 unpushed commits" must not
    become "2 commits" (pushed?) or "dirty, unpushed: 2" read as two facts
    about two things.
    """
    line = PLUGIN.contract._report_line(
        {
            "workspace_slug": "parent-tree-child-abcdef",
            "workspace_path": "<repo_dir>/.workspaces/repo/parent-tree-child-abcdef",
            "dirty": "dirty",
            "unpushed": 2,
            "child_status": "completed",
            "child_role": "worker",
        }
    )
    assert line == (
        "child parent-tree-child-abcdef: dirty, 2 unpushed commits"
        " - <repo_dir>/.workspaces/repo/parent-tree-child-abcdef (worker completed)"
    ), line


def test_the_line_says_unknown_when_the_checks_failed():
    """A failed check reads as unknown, in both of its places."""
    line = PLUGIN.contract._report_line(
        {
            "workspace_slug": "s",
            "workspace_path": "/p",
            "dirty": "state-unknown",
            "unpushed": None,
            "child_status": "failed",
            "child_role": "",
        }
    )
    assert "state-unknown" in line, line
    assert "unpushed unknown" in line, line
    assert "failed" in line, line


def test_a_line_with_no_workspace_still_renders():
    """A child with no workspace still gets a line, not a crash."""
    line = PLUGIN.contract._report_line(
        {"dirty": "state-unknown", "unpushed": None, "child_status": "failed"}
    )
    assert "no workspace" in line, line
    assert "state-unknown" in line, line


def test_a_session_that_is_not_a_parent_gets_nothing(bound_child):
    """Another session's reports never leak into this one."""
    STORE.db.write_child_report(
        CHILD, PARENT, child_status="completed", workspace_slug="s",
        workspace_path="/p", dirty="clean", unpushed=0,
    )
    assert PLUGIN._on_pre_llm_call(session_id="some-other-session") is None
    assert PLUGIN._reports_prompt_section({"session_id": "some-other-session"}) == ""


def test_a_stop_without_a_parent_writes_nothing(bound_child, monkeypatch):
    """A report with no session to deliver it to is dropped, not misfiled.

    Prevents: a report landing in a session that never spawned the child.
    """
    monkeypatch.setattr(STORE, "dirty_state", lambda ws: "clean")
    monkeypatch.setattr(STORE, "unpushed_commits", lambda ws: False)
    PLUGIN._on_subagent_stop(parent_session_id="", child_session_id=CHILD)
    time.sleep(0.2)
    assert STORE.db.read_child_reports("") == []
    assert STORE.db.read_child_reports(PARENT) == []


def test_the_same_child_stopped_twice_reports_the_last_state(bound_child, monkeypatch):
    """A re-dispatched child reports ONCE, with its last state.

    WHY: the host promises one ``subagent_stop`` per child, but a re-run
    child under the same session id would otherwise show the parent two
    lines about the same workspace - one of them stale.
    """
    monkeypatch.setattr(STORE, "dirty_state", lambda ws: "dirty")
    monkeypatch.setattr(STORE, "unpushed_commits", lambda ws: True)
    monkeypatch.setattr(STORE, "count_unpushed_commits", lambda ws: 5)

    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="completed",
    )
    assert _wait_for_report(), "the first report never landed"

    monkeypatch.setattr(STORE, "dirty_state", lambda ws: "clean")
    monkeypatch.setattr(STORE, "unpushed_commits", lambda ws: False)
    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="failed",
    )

    deadline = time.time() + 5.0
    rows = []
    while time.time() < deadline:
        rows = STORE.db.read_child_reports(PARENT)
        if rows and rows[0]["child_status"] == "failed":
            break
        time.sleep(0.01)
    assert len(rows) == 1, f"the parent would see {len(rows)} lines for one child: {rows}"
    assert rows[0]["child_status"] == "failed", rows[0]
    assert rows[0]["dirty"] == "clean", rows[0]
    assert rows[0]["unpushed"] == 0, rows[0]


# ── (g) the stop callback must not block the parent's thread ────────────────


def test_the_report_thread_cannot_write_to_the_live_database(bound_child, monkeypatch, tmp_path):
    """The report thread must use the database its CALLER named.

    WHY this test exists: it is a regression test for a real leak. The
    report thread outlives the code that started it, and the database path
    is resolved from the environment. A test's temporary-database override
    is UNDONE when the test ends - so a report thread that resolved the
    path itself, after that point, resolved the LIVE database and wrote a
    row into it. One row did.

    The fix: the caller captures the path while its environment still says
    what it means, and the thread is handed that path.
    """
    tmp_db = tmp_path / "the-right-db" / "data.db"
    monkeypatch.setenv("HERMES_TREEPS_DB_PATH", str(tmp_db))
    monkeypatch.setenv("HERMES_TREEPS_PLUGIN_DATA_DIR", str(tmp_db.parent))
    STORE.db.reset()
    # The binding lives in the database we just pointed at, so re-bind it
    # there: the report resolves the workspace through the binding.
    STORE.write_state(
        CHILD,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": "parent-child-abcdef",
            "path": str(bound_child),
        },
    )

    # The check is made SLOW on purpose: the override must be gone by the
    # time the report is WRITTEN, which is the exact window in which the
    # leak happened. A fast check would write before the override is
    # undone and the bug would hide.
    started = threading.Event()
    release = threading.Event()

    def slow_clean(ws):
        started.set()
        release.wait(10)
        return "clean"

    monkeypatch.setattr(STORE, "dirty_state", slow_clean)

    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="completed",
    )
    assert started.wait(5), "the report thread never started"

    # The override goes away, exactly as monkeypatch does at the end of a
    # test. A thread that resolves the path NOW gets the live database.
    monkeypatch.undo()
    os.environ.pop("HERMES_TREEPS_DB_PATH", None)
    os.environ.pop("HERMES_TREEPS_PLUGIN_DATA_DIR", None)
    assert str(STORE.db.db_path()) != str(tmp_db), "the override should be gone"

    # Now let the check finish: the write happens with no override in place.
    release.set()

    # The report still lands in the RIGHT database.
    deadline = time.time() + 10.0
    rows = 0
    while time.time() < deadline:
        if tmp_db.exists():
            conn = sqlite3.connect(tmp_db)
            try:
                rows = conn.execute("SELECT count(*) FROM child_reports").fetchone()[0]
            except sqlite3.OperationalError:
                rows = 0
            finally:
                conn.close()
        if rows:
            break
        time.sleep(0.02)
    assert rows == 1, f"the report did not land in the tmp database: {rows}"


def test_the_stop_callback_returns_before_the_checks_finish(bound_child, monkeypatch):
    """The callback returns immediately; the checks run off the parent thread.

    WHY: the host runs ``subagent_stop`` on the CALLER thread - the parent's
    own thread, inside the delegation finalization. A ``jj status`` there
    blocks the parent's thread pool and every delegation queued behind it.

    The test asserts the callback returned BEFORE the (deliberately slow)
    check finished, and then waits for the check to land the row.
    """
    started = threading.Event()
    release = threading.Event()

    def slow_dirty(ws):
        started.set()
        release.wait(10)
        return "dirty"

    monkeypatch.setattr(STORE, "dirty_state", slow_dirty)

    t0 = time.monotonic()
    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="completed",
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, f"the stop callback blocked for {elapsed:.2f}s"
    assert started.wait(5), "the report thread never started"
    # The row is NOT there yet - the check is still running.
    assert STORE.db.read_child_reports(PARENT) == [], "the row appeared before the check finished"

    release.set()
    rows = _wait_for_report()
    assert len(rows) == 1, rows
    assert rows[0]["dirty"] == "dirty", rows[0]


# ── the wiring ──────────────────────────────────────────────────────────────


def test_the_plugin_registers_the_subagent_hooks(monkeypatch):
    """The hooks are registered, and the manifest declares them.

    Prevents: a feature that looks shipped and is not - the module landed
    green while nothing in __init__.py called it.
    """
    registered: list[str] = []
    sections: dict[str, object] = {}

    class Ctx:
        def register_tool(self, **kw):
            pass

        def register_cli_command(self, **kw):
            pass

        def register_command(self, *a, **kw):
            pass

        def register_hook(self, name, *a, **kw):
            registered.append(name)

        def register_system_prompt_section(self, name, text, **kw):
            sections[name] = text

    monkeypatch.setattr(STORE, "candidate_repos", lambda limit=6: ["repo"])
    PLUGIN.register(Ctx())

    assert "subagent_start" in registered, registered
    assert "subagent_stop" in registered, registered
    assert "pre_llm_call" in registered, registered
    assert "child-end-reports" in sections, sorted(sections)


def test_the_manifest_declares_exactly_the_registered_hooks(monkeypatch):
    """``provides_hooks`` must match the hooks the plugin registers.

    WHY: the manifest is what the host reads to decide which events the
    plugin wants. A hook the code registers but the manifest does not
    declare is a hook the host may refuse to call.

    WHY the hand-rolled parser and not PyYAML: the suite runs against the
    STDLIB ALONE (see tests/plugin_loader.py - the plugin must import
    without a Hermes install), and CI installs only pytest. Adding a YAML
    dependency to read six lines of a manifest is not worth it.
    """
    from pathlib import Path

    manifest_path = Path(PLUGIN.__file__).parent / "plugin.yaml"
    declared = _manifest_hooks(manifest_path.read_text())

    registered: list[str] = []

    class Ctx:
        def register_tool(self, **kw):
            pass

        def register_cli_command(self, **kw):
            pass

        def register_command(self, *a, **kw):
            pass

        def register_hook(self, name, *a, **kw):
            registered.append(name)

        def register_system_prompt_section(self, name, text, **kw):
            pass

    monkeypatch.setattr(STORE, "candidate_repos", lambda limit=6: ["repo"])
    PLUGIN.register(Ctx())

    assert set(registered) == declared, (
        f"manifest declares {sorted(declared)}, code registers {sorted(set(registered))}"
    )


def _manifest_hooks(text: str) -> set[str]:
    """The hook names under ``provides_hooks:`` in the manifest.

    A deliberately small reader: the manifest's ``provides_hooks`` block is
    a flat list of ``- name`` entries, and the next top-level key ends it.
    """
    hooks: set[str] = set()
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            inside = stripped.rstrip(":") == "provides_hooks"
            continue
        if not inside:
            continue
        if stripped.startswith("- "):
            hooks.add(stripped[2:].strip().strip("\"'"))
    return hooks
