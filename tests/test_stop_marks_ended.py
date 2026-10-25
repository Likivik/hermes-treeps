"""The stop hook marks the child's session ENDED, so the tree can be reclaimed.

THE LIVE BUG THIS FILE EXISTS FOR: the immediate reclamation silently never
fired in a live run, while the whole test suite was green. The reason was a
row that nobody wrote:

``jjlayout.live_session_paths()`` treats a session with NO
``session_ended_at`` as LIVE, and guard (c) of
``store.reclaim_child_workspace`` refuses to remove the workspace of a live
session. The child's binding row was created by ``childspace`` with no
``session_ended_at`` and NOTHING in the stop flow ever set one - so every
freshly-stopped child was "live" by that rule, the reclamation was refused
with ``in_use``, and the never-raising daemon thread swallowed the refusal.
The tree stayed, and the parent never learned that the reclamation had
failed.

The old tests did not catch it because their fixture PRE-MARKED the child's
session as ended - which is what made the reclamation succeed in the suite
and fail in the live world. The fixture below is the live shape: a child
binding with NO ``session_ended_at``.

What is asserted here:

* the end-to-end stop flow reclaims the tree (the dir is GONE and a
  ``child_reclaims`` row exists) - on current main this FAILS, because the
  guard refuses the tree as in-use;
* marking ended does not RESURRECT an already-set ``session_ended_at``
  (the first end wins - the timestamp must not be rewritten);
* the PARENT's own session is untouched - only the child's binding gets an
  end stamp.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import pytest
from plugin_loader import PLUGIN, STORE

#: The plugin's own layout module. It is a SUBMODULE of the plugin package
#: (the package directory is not a valid identifier, so it cannot be
#: imported by name) - the loader registers it under this name.
JJLAYOUT = sys.modules["hermes_treeps.jjlayout"]

CHILD_PREFIX = STORE.CHILD_PURPOSE_PREFIX

PARENT = "parent-session-marks-ended"
CHILD = "child-session-marks-ended"

#: A fixed stamp, so the "does not resurrect" test can compare exactly.
EARLIER_END = "2020-01-02T00:00:00"


# ── the jj fixture ──────────────────────────────────────────────────────────


def _jj_version() -> tuple[int, ...]:
    """The installed jj version as a tuple, or () when jj is missing."""
    if not shutil.which("jj"):
        return ()
    try:
        out = subprocess.run(["jj", "--version"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ()
    for token in out.split():
        if token[:1].isdigit():
            parts = []
            for piece in token.split("."):
                try:
                    parts.append(int(piece))
                except ValueError:
                    parts.append(0)
            return tuple(parts)
    return ()


JJ_VERSION = _jj_version()


def _jj_supports_workspace_remove() -> bool:
    """True when the installed jj has ``jj workspace remove``.

    ``jj workspace remove`` was added in jj 0.46 - it is what snapshots the
    dirty working copy into a commit before removing the workspace. On an
    older jj the command does not exist, and the removal tests skip.
    """
    if not JJ_VERSION:
        return False
    try:
        proc = subprocess.run(
            ["jj", "workspace", "--help"], capture_output=True, text=True, timeout=60
        )
    except Exception:
        return False
    return "remove" in (proc.stdout or "")


needs_workspace_remove = pytest.mark.skipif(
    not _jj_supports_workspace_remove(),
    reason="jj workspace remove needs jj >= 0.46",
)


@pytest.fixture
def live_child_workspace(tmp_path, monkeypatch):
    """A real jj child workspace bound the way the LIVE flow binds it.

    The one difference from the older fixture matters: the child's binding
    has NO ``session_ended_at``. That is what ``childspace.create_child``
    writes, and it is what the live run had - so the reclamation guard saw
    a LIVE session and refused the tree.
    """
    if not JJ_VERSION:
        pytest.skip("jj not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["jj", "git", "init", "--colocate", str(repo)], check=True, capture_output=True
    )
    subprocess.run(["jj", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    child = tmp_path / "ws" / "repo-child-marks-ended"
    child.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["jj", "workspace", "add", str(child), "--name", child.name],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)

    STORE.write_record(
        child,
        {
            "repo": "repo",
            "slug": child.name,
            "path": str(child),
            "vcs": "jj",
            "purpose": CHILD_PREFIX + PARENT,
            "sessions": [CHILD],
            "created_at": "2020-01-01T00:00:00",
        },
    )
    # The LIVE shape: no ``session_ended_at``. The child's session is over
    # by contract at ``subagent_stop``, but nothing has said so in the row.
    STORE.write_state(
        CHILD,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": child.name,
            "path": str(child),
            "inherited_from": PARENT,
            "purpose": CHILD_PREFIX + PARENT,
        },
    )
    # The parent is bound to a workspace of its own - it must NOT be
    # touched by the child's stop.
    parent_ws = tmp_path / "ws" / "repo-parent-tree"
    parent_ws.mkdir(parents=True, exist_ok=True)
    STORE.write_state(
        PARENT,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": parent_ws.name,
            "path": str(parent_ws),
        },
    )
    return {"repo": repo, "child": child, "parent_ws": parent_ws}


def _wait_for_reclaim(slug, timeout=15.0):
    """Block until the reclaim row exists. Returns the row or None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = STORE.db.read_child_reclaim_by_slug(slug)
        if row is not None:
            return row
        time.sleep(0.01)
    return STORE.db.read_child_reclaim_by_slug(slug)


def _stop(**kw):
    """Fire the host's subagent_stop hook."""
    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="completed", **kw
    )


# ── (a) the end-to-end stop flow reclaims the tree ─────────────────────────


def test_the_stop_frees_the_childs_tree_from_the_live_session_guard(
    live_child_workspace,
):
    """After the stop, the child's tree is NO LONGER a live session's tree.

    WHY this is the sharpest form of the live bug, and why it does not
    need a recent jj: the bug was never about ``jj workspace remove``
    working. It was that guard (c) of ``reclaim_child_workspace`` REFUSED
    the tree, because ``jjlayout.live_session_paths()`` said the child was
    still LIVE. The refusal is what kept the tree on disk, and the
    never-raising report thread swallowed it.

    So the thing that has to change is the guard's VERDICT, and the
    verdict is observable without removing anything: before the stop the
    child's tree is in ``live_session_paths()``, after the stop it is not.
    That is the difference between "the reclamation fires" and "the
    reclamation is refused", and it is the difference the live run felt.

    WHY this test and not only the removal tests: the removal tests need
    jj >= 0.46 for ``jj workspace remove``, so on an older jj they SKIP -
    and that is exactly the environment in which the live bug shipped
    unnoticed. The guard itself does not need a recent jj, so this test
    runs everywhere and keeps the bug from returning behind a green
    suite.
    """
    child = live_child_workspace["child"]
    normalised = str(child.resolve()) if child.exists() else str(child)

    # BEFORE the stop: the child's binding has no ``session_ended_at``, so
    # the guard sees its tree as live. This is the live shape.
    live_before = JJLAYOUT.live_session_paths(STORE.db)
    assert any(
        os.path.normpath(p) == os.path.normpath(normalised) for p in live_before
    ), f"the fixture does not reproduce the live shape: {live_before}"

    _stop()

    deadline = time.time() + 15.0
    live_after: set[str] = set()
    while time.time() < deadline:
        live_after = JJLAYOUT.live_session_paths(STORE.db)
        if not any(
            os.path.normpath(p) == os.path.normpath(normalised) for p in live_after
        ):
            break
        time.sleep(0.01)

    assert not any(
        os.path.normpath(p) == os.path.normpath(normalised) for p in live_after
    ), (
        "the child's tree is STILL a live session's tree after the stop: "
        "the in-use guard will refuse the reclamation, so the tree stays "
        "on disk and the parent never learns that it failed"
    )


@needs_workspace_remove
def test_a_stopped_child_is_reclaimed_even_though_its_row_says_live(
    live_child_workspace,
):
    """The stop flow reclaims a child whose binding row never said "ended".

    Prevents: the live bug - the reclamation silently never fired because
    the child's row still looked LIVE, so the tree stayed on disk forever
    and the parent never learned that the reclamation had been refused.
    """
    child = live_child_workspace["child"]
    (child / "work.txt").write_text("the child's work\n")

    _stop()

    reclaim = _wait_for_reclaim(child.name)
    assert reclaim is not None, (
        "the reclamation never ran: the child's binding still looked LIVE, "
        "so the in-use guard refused the tree"
    )
    assert not child.exists(), "the tree should be gone after the reclamation"
    assert reclaim["snapshot_commit_id"], reclaim


@needs_workspace_remove
def test_the_stop_marks_the_child_session_ended(live_child_workspace):
    """The child's binding carries an end stamp after the stop.

    WHY this is the fix and not a detail: ``live_session_paths`` reads a
    session with no ``session_ended_at`` as LIVE, and guard (c) of
    ``reclaim_child_workspace`` refuses the tree of a live session. The
    child IS finished by contract at ``subagent_stop``, so the stop flow
    must say so in the row - otherwise the reclamation is refused for every
    freshly-stopped child and nothing tells the parent.
    """
    _stop()

    deadline = time.time() + 15.0
    state: dict = {}
    while time.time() < deadline:
        state = STORE.read_state(CHILD) or {}
        if state.get("session_ended_at"):
            break
        time.sleep(0.01)
    assert state.get("session_ended_at"), (
        "the child's binding was not marked ended at stop"
    )


# ── (b) the stamp does not resurrect an ended session ──────────────────────


def test_the_stop_does_not_resurrect_an_already_ended_session(live_child_workspace):
    """An already-set ``session_ended_at`` is LEFT ALONE.

    WHY: the first end wins. A later stop (a re-dispatch of the same
    session id, a second stop hook for the same child) must not rewrite
    the end stamp - the first one is the truth about when the session
    actually ended, and a rewritten stamp is a lie about it.
    """
    STORE.write_state(
        CHILD,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": live_child_workspace["child"].name,
            "path": str(live_child_workspace["child"]),
            "session_ended_at": EARLIER_END,
        },
    )

    _stop()

    deadline = time.time() + 15.0
    state: dict = {}
    while time.time() < deadline:
        state = STORE.read_state(CHILD) or {}
        if state.get("session_ended_at"):
            break
        time.sleep(0.01)
    assert state.get("session_ended_at") == EARLIER_END, (
        f"the earlier end stamp was overwritten: {state.get('session_ended_at')!r}"
    )


# ── (c) the parent is untouched ─────────────────────────────────────────────


def test_the_stop_does_not_touch_the_parent_session(live_child_workspace):
    """Only the CHILD's binding gets an end stamp.

    WHY: the parent's session is still running - it is the one that will
    read the report on its next turn. Marking it ended would make the
    plugin treat the parent's own tree as free to reclaim, and would make
    the parent's session look dead to every other guard in the plugin.
    """
    before = STORE.read_state(PARENT) or {}
    assert not before.get("session_ended_at"), before

    _stop()

    # Wait for the child's stamp to appear, so the thread has certainly
    # run past the parent's row by the time we look.
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if (STORE.read_state(CHILD) or {}).get("session_ended_at"):
            break
        time.sleep(0.01)

    after = STORE.read_state(PARENT) or {}
    assert not after.get("session_ended_at"), (
        f"the parent's session was marked ended: {after}"
    )
    assert after.get("path") == before.get("path"), after


# ── (d) the marking never raises ────────────────────────────────────────────


def test_a_broken_state_read_never_breaks_the_stop(live_child_workspace, monkeypatch):
    """A failure while marking the end stamp costs the stamp, not the stop.

    WHY: the stop hook runs inside the host's delegation finalization. A
    raising observer is a plugin bug, and the report must still land.
    """
    real_read = STORE.read_state

    def boom(session_id):
        if session_id == CHILD:
            raise RuntimeError("the bindings table is unhappy")
        return real_read(session_id)

    monkeypatch.setattr(STORE, "read_state", boom)

    # The hook must not raise.
    _stop()

    deadline = time.time() + 15.0
    rows: list = []
    while time.time() < deadline:
        rows = STORE.db.read_child_reports(PARENT)
        if rows:
            break
        time.sleep(0.01)
    assert len(rows) == 1, "the report did not land"


def test_a_stop_for_an_unknown_child_does_not_raise():
    """A child that has no binding row at all is not an error.

    WHY: the plugin may be enabled mid-child, so the stop hook can fire
    for a child that never got a binding. The stamp is skipped and the
    rest of the flow continues.
    """
    _stop()
    # Nothing to assert but "it did not raise": the hook is the contract.


# ── (e) the stamp lands in the PINNED database ──────────────────────────────


def test_the_stamp_lands_in_the_pinned_database(live_child_workspace, monkeypatch, tmp_path):
    """The end stamp goes into the database the report went into.

    WHY this is a test and not an accident of the code: the report thread
    outlives the code that started it, so a stamp that resolved the
    database path on its own could land in the LIVE database - the exact
    leak the pinned-path report write exists to prevent. And a stamp in
    the wrong database would not unlock the guard either: the reclaim
    reads the pinned database, where the child would still look live.
    """
    import sqlite3

    tmp_db = tmp_path / "the-pinned-db" / "data.db"
    monkeypatch.setenv("HERMES_TREEPS_DB_PATH", str(tmp_db))
    monkeypatch.setenv("HERMES_TREEPS_PLUGIN_DATA_DIR", str(tmp_db.parent))
    STORE.db.reset()
    STORE.write_state(
        CHILD,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": live_child_workspace["child"].name,
            "path": str(live_child_workspace["child"]),
        },
    )

    _stop()

    deadline = time.time() + 15.0
    row = None
    while time.time() < deadline:
        if tmp_db.exists():
            conn = sqlite3.connect(tmp_db)
            try:
                row = conn.execute(
                    "SELECT session_ended_at FROM bindings WHERE session_id=?",
                    (CHILD,),
                ).fetchone()
            except sqlite3.OperationalError:
                row = None
            finally:
                conn.close()
        if row and row[0]:
            break
        time.sleep(0.02)
    assert row and row[0], (
        "the end stamp did not land in the pinned database - it went "
        "somewhere else, and the guard would still see the child as live"
    )
