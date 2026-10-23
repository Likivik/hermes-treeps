"""The storage layer under concurrency: the defects that made the gate fail open.

Each test names the failure it prevents. The shared-connection bug is the
one that mattered most: python's sqlite3 caches ONE prepared statement per
(connection, sql) pair, so two threads on the same connection clobber each
other's statement state. The audit reproduced
"sqlite3.InterfaceError: bad parameter or other API misuse" out of
``db.read_binding``, reads that returned 0/2/14 rows for a five-row SELECT,
and a read that returned ANOTHER workspace's row.

Because ``bind_session`` keys its INSERT on the record it just read, a
corrupted read became a corrupted WRITE: a session landed under a workspace
it was never given. The gate catches that exception and fails OPEN, so an
unbound session kept running gated tools - 616 of 2667 tool calls in the
audit's control run, 0 with per-thread connections.

The other defects are quieter but just as real: ``reset()`` segfaulted the
process by closing a connection another thread was using, a failed write
left a dirty transaction that the NEXT unrelated commit published, the
purge deleted files it had failed to back up, the env override did not
actually isolate the database, and the connection was published before its
schema existed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from plugin_loader import STORE

DB = STORE.db

ROOT = Path(__file__).resolve().parent.parent


# ── 1. concurrent reads, writes and binds ───────────────────────────────────


def test_eight_threads_never_cross_contaminate_rows(tmp_path):
    """Eight threads on one database: no exceptions, no wrong rows, no
    misfiled sessions.

    Prevents (P0): the shared connection. python's sqlite3 caches one
    prepared statement per (connection, sql) pair, so two threads
    executing on the SAME connection clobber each other's statement state.
    The audit saw "bad parameter or other API misuse", reads that returned
    another workspace's row, and a session bound to the WRONG workspace -
    which the gate then let through, because it fails open.
    """
    n_threads = 8
    iterations = 60
    sessions = [f"sid-c{i}" for i in range(n_threads)]
    paths = [str(tmp_path / f"ws{i}") for i in range(n_threads)]
    repos = [f"repo{i}" for i in range(n_threads)]
    slugs = [f"slug{i}" for i in range(n_threads)]
    for i in range(n_threads):
        DB.write_workspace(
            {"repo": repos[i], "slug": slugs[i], "path": paths[i], "purpose": "p"},
            path=paths[i],
        )

    errors: list[str] = []
    wrong_rows: list[str] = []
    start = threading.Barrier(n_threads)

    def worker(idx: int) -> None:
        sid = sessions[idx]
        path = paths[idx]
        try:
            start.wait(timeout=30)
            for _ in range(iterations):
                DB.write_binding(
                    sid,
                    {
                        "mode": "repo",
                        "repo": repos[idx],
                        "slug": slugs[idx],
                        "path": path,
                        "marker": idx,
                    },
                )
                # A read must return THIS session's row - never another
                # thread's. A corrupted read returned someone else's.
                got = DB.read_binding(sid)
                if got is None or got.get("marker") != idx:
                    wrong_rows.append(
                        f"read_binding({sid}) returned {got!r}, expected marker {idx}"
                    )
                # The same for the workspace row: it must be the record of
                # THIS path, not another workspace's.
                rec = DB.read_workspace(path)
                if rec is None or rec.get("repo") != repos[idx] or rec.get("path") != path:
                    wrong_rows.append(
                        f"read_workspace({path}) returned {rec!r}, expected repo {repos[idx]}"
                    )
                # bind_session keys its INSERT on the record it just read:
                # a corrupted read here files the session under the wrong
                # workspace, which is the security-relevant half of the bug.
                STORE.bind_session(Path(path), sid)
        except Exception as exc:  # noqa: BLE001 - the point is to collect it
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            try:
                start.wait(timeout=30)
            except threading.BrokenBarrierError:
                pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    for t in threads:
        assert not t.is_alive(), "a worker thread hung"

    assert not errors, f"concurrent access raised: {errors[:5]}"
    assert not wrong_rows, f"concurrent access returned the wrong row: {wrong_rows[:5]}"

    # Every session must be claimed by EXACTLY the workspace it was bound
    # to. The audit saw one session claimed by two workspaces at once.
    for i in range(n_threads):
        sid = sessions[i]
        claimants = [
            (repos[j], slugs[j])
            for j in range(n_threads)
            if sid in DB.workspace_sessions(repos[j], slugs[j])
        ]
        assert claimants == [(repos[i], slugs[i])], (
            f"{sid} is claimed by {claimants}, expected only {(repos[i], slugs[i])}"
        )


# ── 2. reset() while another thread is writing ──────────────────────────────


def test_reset_while_another_thread_writes_does_not_crash(tmp_path):
    """``reset()`` must not pull the database out from under a live thread.

    Prevents (P0): ``reset()`` called ``conn.close()`` on the ONE shared
    connection while another thread was mid-statement on it, and CPython
    segfaulted the whole process (exit 139, 5/5 runs). Now every thread
    owns its connection, and ``reset()`` only invalidates the connections
    - it never closes one another thread is inside.
    """
    stop = threading.Event()
    errors: list[str] = []

    def writer() -> None:
        n = 0
        while not stop.is_set():
            try:
                DB.write_binding(f"sid-w{n % 5}", {"mode": "repo", "path": "/tmp/x"})
                n += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                return

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        for _ in range(200):
            DB.reset()
            time.sleep(0.001)
    finally:
        stop.set()
        t.join(timeout=30)

    assert not t.is_alive(), "the writer thread hung"
    assert not errors, f"the writer thread died: {errors}"


# ── 3. a failed write leaves no partial state ───────────────────────────────


def test_failed_write_leaves_nothing_behind(tmp_path):
    """A write that raises half-way through must leave NOTHING behind.

    Prevents (P0): ``write_workspace`` did a DELETE plus an upsert with no
    rollback on the failure path. The connection kept the dirty
    transaction, and the NEXT unrelated commit published the half-write -
    the repro showed ``purpose`` CHANGED and the session rows gone. Now
    every mutation runs in a transaction that commits or rolls back, and
    no function leaves the connection in a transaction.
    """
    path = str(tmp_path / "ws")
    DB.write_workspace(
        {"repo": "repo", "slug": "slug", "path": path, "purpose": "ORIGINAL",
         "sessions": ["s-keep"]},
        path=path,
    )
    # A trigger that makes the SECOND half of the write fail.
    trigger = (
        "CREATE TRIGGER boom BEFORE INSERT ON workspace_sessions "
        "WHEN NEW.session_id = 's-boom' BEGIN SELECT RAISE(ABORT, 'boom'); END;"
    )
    DB.connect().execute(trigger)
    DB.connect().commit()

    with pytest.raises(sqlite3.Error):
        DB.write_workspace(
            {"repo": "repo", "slug": "slug", "path": path, "purpose": "CHANGED",
             "sessions": ["s-keep", "s-boom"]},
            path=path,
        )

    # An unrelated write on the SAME connection used to publish the
    # half-write: the dirty transaction was committed by the next
    # ``set_kv()``.
    DB.set_kv("unrelated", "1")

    # And the connection must not be sitting in a transaction.
    assert not DB.connect().in_transaction, "the failed write left a dirty transaction"

    # An independent connection sees the committed truth.
    observer = sqlite3.connect(str(DB.db_path()))
    try:
        row = observer.execute(
            "SELECT purpose FROM workspaces WHERE repo='repo' AND slug='slug'"
        ).fetchone()
        assert row is not None and row[0] == "ORIGINAL", (
            f"the failed write published a half-write: purpose={row!r}"
        )
        sids = [
            r[0]
            for r in observer.execute(
                "SELECT session_id FROM workspace_sessions WHERE repo='repo' AND slug='slug'"
            ).fetchall()
        ]
        assert sids == ["s-keep"], f"the failed write changed the session list: {sids}"
    finally:
        observer.close()


# ── 4. the purge must not delete what it could not back up ─────────────────


def test_purge_keeps_a_file_it_could_not_back_up(tmp_path):
    """A record the backup could not read must SURVIVE the purge.

    Prevents (P0): ``_write_backup`` swallowed the per-file tar error, and
    the caller deleted the files unconditionally - so an unreadable record
    was purged with an EMPTY tarball. The record was simply gone. Now the
    purge deletes only what is really in the tarball and reports the rest.
    """
    root = tmp_path / "wsroot"
    good = root / "repo" / "good"
    bad = root / "repo" / "bad"
    for ws in (good, bad):
        (ws / ".hermes").mkdir(parents=True)
        rec = ws / ".hermes" / "session.json"
        rec.write_text(json.dumps({"repo": "repo", "slug": ws.name, "sessions": []}))
        DB.write_workspace(
            {"repo": "repo", "slug": ws.name, "path": str(ws), "purpose": ""},
            path=str(ws),
        )

    unreadable = bad / ".hermes" / "session.json"
    unreadable.chmod(0o000)
    try:
        out = DB.purge_legacy_records(dry_run=False, workspace_root=root)
    finally:
        if unreadable.exists():
            unreadable.chmod(0o644)

    assert str(good / ".hermes" / "session.json") in out["purged"], out
    assert not (good / ".hermes" / "session.json").exists(), out
    assert str(unreadable) in out["not_backed_up"], (
        f"the purge did not report the file it could not back up: {out}"
    )
    assert unreadable.exists(), "the purge deleted a file that was never backed up"


# ── 5. the env override must actually isolate the database ─────────────────


def _fake_host_module(live_db: Path):
    """A stand-in for the host's ``plugins.plugin_storage``.

    The real host's ``plugin_db()`` ALWAYS opens the LIVE database, no
    matter which env vars the plugin was given. The tests have to model
    that faithfully, or they would prove nothing: without a Hermes install
    the plugin falls back to a plain sqlite file and the override bug is
    invisible.
    """
    import types

    mod = types.ModuleType("plugins.plugin_storage")

    def plugin_data_dir(name: str) -> Path:
        d = live_db.parent / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def plugin_db(name: str, filename: str = "data.db") -> sqlite3.Connection:
        d = plugin_data_dir(name)
        conn = sqlite3.connect(d / filename, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    mod.plugin_data_dir = plugin_data_dir
    mod.plugin_db = plugin_db
    pkg = types.ModuleType("plugins")
    pkg.__path__ = []  # type: ignore[attr-defined]
    pkg.plugin_storage = mod  # type: ignore[attr-defined]
    return pkg, mod


def test_plugin_data_dir_override_isolates_the_database(tmp_path, monkeypatch):
    """``HERMES_TREEPS_PLUGIN_DATA_DIR`` alone must isolate the database.

    Prevents (P0): ``db_path()`` honoured the data-dir override, but
    ``connect()`` still called the host's ``plugin_db()`` unless
    ``HERMES_TREEPS_DB_PATH`` was set - and ``plugin_db()`` resolves the
    LIVE data dir. So ``db_path()`` reported a tmp path while
    ``PRAGMA database_list`` showed
    ``/var/lib/hermes/.hermes/plugin-data/hermes-treeps/data.db``. The
    override did not isolate anything.

    The test models the host API, because without it the plugin's plain
    sqlite fallback hides the bug.
    """
    live = tmp_path / "live" / "hermes-treeps" / "data.db"
    pkg, storage = _fake_host_module(live)
    monkeypatch.setitem(sys.modules, "plugins", pkg)
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", storage)

    override = tmp_path / "override"
    monkeypatch.setenv("HERMES_TREEPS_PLUGIN_DATA_DIR", str(override))
    monkeypatch.delenv("HERMES_TREEPS_DB_PATH", raising=False)
    DB.reset()

    conn = DB.connect()
    row = conn.execute("PRAGMA database_list").fetchone()
    assert row is not None and row[2], "PRAGMA database_list returned nothing"
    opened = Path(str(row[2]))
    assert opened == override / "data.db", (
        f"the override did not isolate the database: it opened {opened}"
    )

    # And the live database must not have been touched at all.
    assert not live.exists(), f"the live database was created: {live}"


# ── 6. two processes opening a fresh database at the same time ──────────────


_CHILD = """
import importlib.util, sys, time
root = {root!r}
spec = importlib.util.spec_from_file_location(
    "hermes_treeps", root + "/__init__.py", submodule_search_locations=[root]
)
mod = importlib.util.module_from_spec(spec)
sys.modules["hermes_treeps"] = mod
spec.loader.exec_module(mod)
db = mod.store.db
# Open at the same instant as the other children.
time.sleep(max(0.0, float(sys.argv[2]) - time.time()))
conn = db.connect()
row = conn.execute("SELECT count(*) FROM bindings").fetchone()
db.write_binding("sid-child", {{"mode": "repo"}})
print("ok", row[0])
"""


def _child_env(db_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HERMES_TREEPS_DB_PATH"] = str(db_path)
    env.pop("HERMES_TREEPS_PLUGIN_DATA_DIR", None)
    env["PYTHONPATH"] = str(ROOT)
    return env


def test_two_processes_can_open_a_fresh_database_together(tmp_path):
    """Two processes opening a FRESH database must not collide.

    Prevents (P0): ``PRAGMA journal_mode=WAL`` ran outside any lock, and
    a journal-mode change does NOT go through SQLite's busy handler - so
    ``timeout=30`` did not save it and two processes opening a fresh
    database together died with "database is locked". The pragma is now
    retried, and the schema is created before the connection is published.
    """
    db_path = tmp_path / "fresh" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time() + 1.0
    script = tmp_path / "child.py"
    script.write_text(_CHILD.format(root=str(ROOT)))

    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(db_path), str(start)],
            env=_child_env(db_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=120)
        outs.append((p.returncode, out, err))

    for code, out, err in outs:
        assert code == 0, (
            f"a child process failed with exit {code}: {err}\nstdout: {out}"
        )
        assert "database is locked" not in err, err
        assert out.startswith("ok"), f"unexpected child output: {out!r}"
