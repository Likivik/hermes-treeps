"""Tests for the wired stop->reclaim path.

The subject: when a delegated child STOPS, the plugin writes its end report
AND reclaims its workspace - on the SAME background thread, in that order.
A finished child's tree is gone right away, and its work is not lost: jj
snapshots the dirty working copy into a commit and the plugin records the
commit id.

The four failures these tests defend against:

* the tree stays on disk forever (the reclamation never ran);
* the report says "clean" about a tree whose work is in a snapshot commit
  (the order was wrong);
* an operator who does not trust the snapshot path loses the REPORTING
  too (the kill-switch took the reporting down with it);
* a refusal (a live session still pinned to the tree) breaks the report
  flow and the parent never hears what happened.

The reclamation tests use a REAL jj repo in a tmp dir: the whole point is
jj's behaviour, and a mocked jj would only test the mock.
``jj workspace remove`` needs jj >= 0.46, so the removal tests follow the
skip-guard pattern of ``tests/test_reclaim.py``.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest
from plugin_loader import PLUGIN, STORE

CHILD_PREFIX = STORE.CHILD_PURPOSE_PREFIX

PARENT = "parent-session-1"
CHILD = "child-session-abcdef"


# ── the jj fixtures ──────────────────────────────────────────────────────────


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
def child_workspace(tmp_path, monkeypatch):
    """A real jj child workspace, recorded and bound like the plugin does.

    The layout mirrors the plugin's: the main checkout and the child
    workspace live side by side, and the child's ``.jj/repo`` is a FILE
    pointing at the main workspace's store.
    """
    if not JJ_VERSION:
        pytest.skip("jj not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["jj", "git", "init", "--colocate", str(repo)], check=True, capture_output=True
    )
    subprocess.run(["jj", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    child = tmp_path / "ws" / "repo-child-abc123"
    child.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["jj", "workspace", "add", str(child), "--name", child.name],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)

    # The plugin's own record for the workspace: the ``child:`` purpose is
    # what makes it reclaimable at all.
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
    # The child's binding: it is how the report finds the workspace. The
    # session ENDED, so the workspace is not "live" and the reclamation is
    # allowed to touch it.
    STORE.write_state(
        CHILD,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": child.name,
            "path": str(child),
            "session_ended_at": "2020-01-02T00:00:00",
        },
    )
    return {"repo": repo, "child": child}


def _wait_for_report(parent=PARENT, timeout=10.0):
    """Block until the report thread wrote its row. Returns the rows."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = STORE.db.read_child_reports(parent)
        if rows:
            return rows
        time.sleep(0.01)
    return STORE.db.read_child_reports(parent)


def _wait_for_reclaim(slug, timeout=10.0):
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


# ── (a) the dirty child: report row, tree gone, snapshot recorded ───────────


@needs_workspace_remove
def test_a_stopped_dirty_child_is_reclaimed_and_its_work_survives(child_workspace):
    """The tree is gone right away, and its work became a commit.

    Prevents: the whole point of the wiring - a finished child's tree is
    dead weight, and the work in it must survive as a snapshot commit.
    """
    repo = child_workspace["repo"]
    child = child_workspace["child"]
    (child / "work.txt").write_text("the child's work\n")

    _stop()

    rows = _wait_for_report()
    assert len(rows) == 1, rows
    reclaim = _wait_for_reclaim(child.name)
    assert reclaim is not None, "the reclamation was not recorded"
    assert reclaim["snapshot_commit_id"], reclaim
    assert reclaim["was_dirty"] is True, reclaim
    assert not child.exists(), "the tree should be gone after the reclamation"

    # The snapshot commit is REAL: it resolves in the main workspace and it
    # holds the child's file.
    shown = subprocess.run(
        ["jj", "log", "-r", reclaim["snapshot_commit_id"], "--no-graph", "-T", "commit_id"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert shown.returncode == 0, shown.stderr
    content = subprocess.run(
        ["jj", "file", "show", "-r", reclaim["snapshot_commit_id"], "work.txt"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert content.returncode == 0, content.stderr
    assert content.stdout.strip() == "the child's work", content.stdout


@needs_workspace_remove
def test_the_report_row_and_the_reclaim_row_are_linked(child_workspace):
    """The report finds its reclamation, and with it the snapshot commit id.

    WHY this is a linkage and not a copied column: the report row already
    carries the workspace slug and repo, which are the primary key of the
    reclamation table. So the join is a query - no schema change, and no
    second copy of the commit id that could drift away from the reclamation
    that produced it.
    """
    child = child_workspace["child"]
    (child / "work.txt").write_text("work\n")

    _stop()

    rows = _wait_for_report()
    assert len(rows) == 1, rows
    report = rows[0]
    assert report["workspace_slug"] == child.name, report

    # The reclamation runs AFTER the report row, so wait for it: the
    # linkage only exists once the reclaim row is there.
    reclaim = _wait_for_reclaim(child.name)
    assert reclaim is not None, "the reclamation was not recorded"

    linked = STORE.db.read_child_reclaim_for_report(report)
    assert linked is not None, "the report is not linked to its reclamation"
    assert linked["snapshot_commit_id"], linked
    assert linked["slug"] == child.name, linked
    assert linked["snapshot_commit_id"] == reclaim["snapshot_commit_id"], linked


# ── (b) the kill-switch: report only, tree intact ───────────────────────────


@needs_workspace_remove
def test_the_kill_switch_keeps_the_report_and_leaves_the_tree(
    child_workspace, monkeypatch
):
    """With the switch off, nothing is reclaimed - but the report still lands.

    WHY this is the shape of the switch: an operator who does not trust the
    snapshot path must be able to turn the reclamation off WITHOUT losing
    the reporting. The reporting is what tells a human what happened; the
    reclamation is what acts on it.
    """
    child = child_workspace["child"]
    (child / "work.txt").write_text("work\n")
    monkeypatch.setattr(
        STORE, "settings", lambda: {"child_immediate_reclaim": False}
    )

    _stop()

    rows = _wait_for_report()
    assert len(rows) == 1, rows
    assert rows[0]["dirty"] in ("dirty", "state-unknown"), rows[0]
    assert child.exists(), "the tree must survive when the switch is off"
    assert STORE.db.read_child_reclaim_by_slug(child.name) is None, (
        "the reclamation ran even though the switch is off"
    )


# ── (c) a refusal does not break the report flow ────────────────────────────


@needs_workspace_remove
def test_a_refused_reclamation_still_leaves_the_report(child_workspace, monkeypatch):
    """A refusal (the tree is in use) costs the reclamation, not the report.

    Prevents: the parent's next turn learning NOTHING because the
    reclamation was refused - the refusal is a fact about the tree, and the
    report is how the parent learns it.
    """
    child = child_workspace["child"]
    (child / "work.txt").write_text("work\n")
    # The child's session is LIVE again: it is pinned to the tree, so the
    # reclamation refuses it.
    STORE.write_state(
        CHILD,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": child.name,
            "path": str(child),
        },
    )

    _stop()

    rows = _wait_for_report()
    assert len(rows) == 1, "the refusal swallowed the report"
    assert rows[0]["child_status"] == "completed", rows[0]
    assert child.exists(), "a live tree must survive the refusal"
    assert STORE.db.read_child_reclaim_by_slug(child.name) is None


def test_a_broken_reclamation_never_breaks_the_report(child_workspace, monkeypatch):
    """A reclamation that raises must not cost the parent its report.

    Prevents: the reporting feature taking down the delegation it watches
    because a jj subprocess failed.
    """
    child = child_workspace["child"]

    def boom(path, **kwargs):
        raise RuntimeError("jj exploded")

    monkeypatch.setattr(STORE, "reclaim_child_workspace", boom)

    _stop()

    rows = _wait_for_report()
    assert len(rows) == 1, "the broken reclamation swallowed the report"
    assert child.exists(), "the tree must stay when the reclamation blew up"


# ── (d) the report describes the PRE-reclaim state ──────────────────────────


@needs_workspace_remove
def test_the_report_still_says_dirty_after_the_reclamation(child_workspace):
    """The report describes the tree at stop time, not after the snapshot.

    WHY this matters: the reclamation snapshots the dirty tree into a
    commit and removes it. A report that re-checked the tree afterwards
    would say "clean" - about a tree that no longer exists, and whose work
    is sitting in a commit. The order (report, then reclaim) is what keeps
    the report honest.
    """
    child = child_workspace["child"]
    (child / "work.txt").write_text("work\n")

    _stop()

    rows = _wait_for_report()
    assert len(rows) == 1, rows
    assert rows[0]["dirty"] == "dirty", (
        f"the report lost the pre-reclaim dirty state: {rows[0]}"
    )
    # And the reclamation really did happen after the report.
    reclaim = _wait_for_reclaim(child.name)
    assert reclaim is not None, "the reclamation never ran"
    assert not child.exists()


# ── the wiring is inside the report thread ──────────────────────────────────


def test_the_reclamation_runs_on_the_report_thread(child_workspace, monkeypatch):
    """The reclamation happens on the background thread, not the parent's.

    WHY: the host runs ``subagent_stop`` on the CALLER's thread - the
    parent's own thread, inside the delegation finalization. A ``jj
    workspace remove`` there blocks the parent's thread pool and every
    delegation queued behind it.
    """
    threads: list[str] = []
    real_reclaim = STORE.reclaim_child_workspace

    def spy(path, **kwargs):
        import threading

        threads.append(threading.current_thread().name)
        return real_reclaim(path, **kwargs)

    monkeypatch.setattr(STORE, "reclaim_child_workspace", spy)

    PLUGIN._on_subagent_stop(
        parent_session_id=PARENT, child_session_id=CHILD, child_status="completed"
    )
    assert _wait_for_report(), "the report never landed"

    deadline = time.time() + 10.0
    while time.time() < deadline and not threads:
        time.sleep(0.01)
    assert threads, "the reclamation never ran"
    assert "treeps-child-report" in threads[0], threads


def test_the_reclamation_happens_after_the_report_row(child_workspace, monkeypatch):
    """The report row is written BEFORE the reclamation starts.

    WHY the order is pinned: the report says what the tree was at stop
    time; the reclamation then snapshots and removes it. The other order
    would produce a report about a tree that is already gone.
    """
    order: list[str] = []
    real_reclaim = STORE.reclaim_child_workspace
    real_write = STORE.db.write_child_report_at

    def spy_reclaim(path, **kwargs):
        order.append("reclaim")
        return real_reclaim(path, **kwargs)

    def spy_write(db_file, *args, **kwargs):
        order.append("report")
        return real_write(db_file, *args, **kwargs)

    monkeypatch.setattr(STORE, "reclaim_child_workspace", spy_reclaim)
    monkeypatch.setattr(STORE.db, "write_child_report_at", spy_write)

    _stop()

    deadline = time.time() + 10.0
    while time.time() < deadline and "reclaim" not in order:
        time.sleep(0.01)
    assert order[:2] == ["report", "reclaim"], order


def test_the_reclamation_uses_the_reports_database(child_workspace, monkeypatch, tmp_path):
    """The reclaim row lands in the SAME database as the report row.

    WHY this is a test: the report thread outlives the code that started
    it, so a thread that resolves the database path on its own can end up
    in the LIVE database. That happened once. The reclamation must obey
    the same rule as the report.
    """
    import sqlite3

    tmp_db = tmp_path / "the-right-db" / "data.db"
    monkeypatch.setenv("HERMES_TREEPS_DB_PATH", str(tmp_db))
    monkeypatch.setenv("HERMES_TREEPS_PLUGIN_DATA_DIR", str(tmp_db.parent))
    STORE.db.reset()
    # The record and the binding live in the database we just pointed at:
    # the reclamation reads the workspace record from the SAME database it
    # writes its reclaim row into.
    STORE.write_record(
        child_workspace["child"],
        {
            "repo": "repo",
            "slug": child_workspace["child"].name,
            "path": str(child_workspace["child"]),
            "vcs": "jj",
            "purpose": CHILD_PREFIX + PARENT,
            "sessions": [CHILD],
            "created_at": "2020-01-01T00:00:00",
        },
    )
    STORE.write_state(
        CHILD,
        {
            "mode": "repo",
            "repo": "repo",
            "slug": child_workspace["child"].name,
            "path": str(child_workspace["child"]),
            "session_ended_at": "2020-01-02T00:00:00",
        },
    )

    _stop()

    deadline = time.time() + 10.0
    rows = 0
    reclaims = 0
    while time.time() < deadline:
        if tmp_db.exists():
            conn = sqlite3.connect(tmp_db)
            try:
                rows = conn.execute("SELECT count(*) FROM child_reports").fetchone()[0]
                reclaims = conn.execute("SELECT count(*) FROM child_reclaims").fetchone()[0]
            except sqlite3.OperationalError:
                rows = 0
                reclaims = 0
            finally:
                conn.close()
        if rows and reclaims:
            break
        time.sleep(0.02)
    assert rows == 1, f"the report did not land in the tmp database: {rows}"
    assert reclaims == 1, f"the reclaim did not land in the tmp database: {reclaims}"


# ── the kill-switch itself ──────────────────────────────────────────────────


def test_the_kill_switch_defaults_to_on():
    """The shipped default is ON: a finished child's tree goes right away."""
    from plugin_loader import SETTINGS

    assert SETTINGS.DEFAULTS["child_immediate_reclaim"] is True
    assert SETTINGS.settings()["child_immediate_reclaim"] is True


def test_the_kill_switch_is_a_declared_setting():
    """The switch is in SETTINGS_KEYS and in the manifest's config_schema.

    WHY: the Desktop settings form is generated from the manifest, so a
    key that is not there cannot be turned off from the UI - and the test
    suite compares the two sets, so a drift fails the build.
    """
    from plugin_loader import SETTINGS

    assert "child_immediate_reclaim" in SETTINGS.SETTINGS_KEYS
    manifest = (Path(PLUGIN.__file__).parent / "plugin.yaml").read_text()
    assert "child_immediate_reclaim:" in manifest, manifest


def test_the_kill_switch_must_be_a_boolean():
    """A non-boolean value is refused, not coerced.

    WHY: ``"false"`` (the string) is TRUTHY in Python, so a coercion would
    turn the kill-switch the wrong way round - an operator who typed
    ``child_immediate_reclaim: "false"`` would get the reclamation ON.
    """
    from plugin_loader import SETTINGS

    try:
        SETTINGS.settings(
            {
                "plugins": {
                    "entries": {
                        "hermes-treeps": {
                            "settings": {"child_immediate_reclaim": "false"}
                        }
                    }
                }
            }
        )
    except Exception as exc:
        assert "child_immediate_reclaim" in str(exc), exc
    else:
        raise AssertionError("a string kill-switch was accepted")


# ── the pinned database path ────────────────────────────────────────────────


def test_the_pinned_path_is_thread_local_and_released():
    """The pin applies to this thread only, and it is released on exit.

    WHY: a process-wide pin would let two reporting threads fight over
    one database, and a pin that leaked would keep writing into a database
    nobody chose.
    """
    import threading

    seen: list[str] = []

    def other() -> None:
        seen.append(str(STORE.db.db_path()))

    with STORE.db.pinned_path("/tmp/pinned.db"):
        assert str(STORE.db.db_path()) == "/tmp/pinned.db"
        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert seen and seen[0] != "/tmp/pinned.db", seen
    assert str(STORE.db.db_path()) != "/tmp/pinned.db"
