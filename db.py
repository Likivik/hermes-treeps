"""SQLite storage for hermes-treeps, in the host's sanctioned plugin-data dir.

WHY a database and not files: the old implementation stored a session's record
in one JSON file and the workspace's record in another, and every mutation was
a read-modify-write of a JSON document. Eight concurrent ``bind_session``
calls on one workspace lost four of them (the last writer won with a list it
had read before the others wrote). A row per binding with one transaction per
mutation removes the class of bug.

WHERE the file lives: ``<hermes home>/plugin-data/hermes-treeps/data.db``,
opened through the host's ``plugins.plugin_storage.plugin_db``. The plugin
install directory is git-pulled and deleted on update, so state must never
live there.

The module is import-safe without a Hermes install (CI runs the suite with
the stdlib alone): the host API is imported lazily and a plain sqlite file is
used as the fallback. ``HERMES_TREEPS_DB_PATH`` overrides the path, which is
how the tests stay out of the live database.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

PLUGIN_NAME = "hermes-treeps"
DB_FILENAME = "data.db"

SCHEMA_VERSION = 4

# One lock for every multi-statement write. It is NEVER held across a
# subprocess call: those can take seconds, and a lock held that long would
# serialise every other session in the process.
#
# WHY the lock is still needed with per-thread connections: WAL allows many
# readers and ONE writer at a time. The lock is what keeps two writers in
# this process from colliding on that single writer slot (and from seeing
# "database is locked" instead of waiting).
_WRITE_LOCK = threading.RLock()

# ── one connection per THREAD ───────────────────────────────────────────────
#
# WHY: python's sqlite3 caches ONE prepared statement per (connection, sql)
# pair. Two threads executing on the SAME connection therefore clobber each
# other's statement state. The audit reproduced this on the live code:
# "sqlite3.InterfaceError: bad parameter or other API misuse" out of
# db.read_binding, reads that returned 0/2/14 rows for a five-row SELECT, and
# a read that returned ANOTHER workspace's row.
#
# WHY that was a security bug and not a flake: bind_session keys its INSERT
# on the record it just read, so a corrupted read became a corrupted WRITE -
# a session landed under a workspace it was never given, and one session was
# seen claimed by two workspaces at once. The gate catches that exception and
# fails OPEN, so an unbound session kept running gated tools (616 of 2667
# tool calls in the audit's control run; 0 with per-thread connections).
#
# The per-thread connection is held in thread-local storage. The module-level
# registry below records each connection's IDENTITY and the GENERATION it was
# opened in, so reset() can invalidate every connection in the process
# without ever closing one another thread is using.
_THREAD_LOCAL = threading.local()

# Identity + generation of every live connection, keyed by the owning
# THREAD OBJECT (not its ident: idents are reused, and a dead thread's
# entry would be inherited by an unrelated new thread). The registry holds
# IDs, not connections: a strong reference to the connection would keep a
# dead thread's file descriptor open forever.
_REGISTRY: dict[object, dict[str, Any]] = {}
_REGISTRY_LOCK = threading.RLock()

# Bumped by reset(). A connection opened in an older generation is stale:
# its owner closes it (and opens a fresh one) the next time it asks for a
# connection. Nobody closes another thread's connection - that is exactly
# what segfaulted the process.
_GENERATION = 0

# Serialises "open a connection + make sure the schema exists" so two
# threads opening the database at the same time cannot race the DDL.
_INIT_LOCK = threading.RLock()

# How long to wait for a lock that another process holds. SQLite's busy
# handler covers ordinary locking; the JOURNAL MODE change does NOT go
# through it, so that one gets an explicit retry loop (see _set_wal).
_BUSY_TIMEOUT_S = 30.0
_WAL_RETRIES = 60
_WAL_RETRY_DELAY_S = 0.1

SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    repo        TEXT NOT NULL,
    slug        TEXT NOT NULL,
    path        TEXT NOT NULL DEFAULT '',
    vcs         TEXT NOT NULL DEFAULT 'jj',
    purpose     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT '',
    last_synced TEXT,
    extra       TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (repo, slug)
);
CREATE INDEX IF NOT EXISTS workspaces_path_idx ON workspaces(path);
CREATE INDEX IF NOT EXISTS workspaces_slug_idx ON workspaces(slug);

CREATE TABLE IF NOT EXISTS bindings (
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
CREATE INDEX IF NOT EXISTS bindings_path_idx ON bindings(path);

CREATE TABLE IF NOT EXISTS workspace_sessions (
    repo       TEXT NOT NULL,
    slug       TEXT NOT NULL,
    session_id TEXT NOT NULL,
    PRIMARY KEY (repo, slug, session_id)
);

CREATE TABLE IF NOT EXISTS kv (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL DEFAULT '',
    updated_at TEXT
);

-- ── child lifecycle (v3) ────────────────────────────────────────────────────
--
-- WHY these tables exist: the plugin must tell the PARENT session, on its next
-- turn, which workspace a finished child used, whether that workspace is
-- clean, and how many unpushed commits it left. The host's subagent_start /
-- subagent_stop hooks give us the child's identity and the parent's, but
-- neither tells us which workspace the child was bound to - that lookup is
-- what these two tables make cheap and explicit.
--
-- child_runs: one row per child, written at subagent_start. It is the map
--   from a child session to its parent, so subagent_stop never has to guess
--   which parent a report belongs to (a reverse lookup over bindings is
--   ambiguous when a session id is reused).
--
-- child_reports: one row per child, written at subagent_stop and read once by
--   the parent's next prompt. The row is DELETED after it is delivered
--   (one-time delivery), so a stale report can never be shown twice.

CREATE TABLE IF NOT EXISTS child_runs (
    child_session_id TEXT PRIMARY KEY,
    child_subagent_id TEXT NOT NULL DEFAULT '',
    parent_session_id TEXT NOT NULL DEFAULT '',
    parent_turn_id   TEXT NOT NULL DEFAULT '',
    parent_subagent_id TEXT NOT NULL DEFAULT '',
    child_role       TEXT NOT NULL DEFAULT '',
    child_goal       TEXT NOT NULL DEFAULT '',
    started_at       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS child_runs_parent_idx ON child_runs(parent_session_id);

CREATE TABLE IF NOT EXISTS child_reports (
    child_session_id TEXT PRIMARY KEY,
    parent_session_id TEXT NOT NULL DEFAULT '',
    child_role       TEXT NOT NULL DEFAULT '',
    child_status     TEXT NOT NULL DEFAULT '',
    workspace_path   TEXT NOT NULL DEFAULT '',
    workspace_slug   TEXT NOT NULL DEFAULT '',
    workspace_repo   TEXT NOT NULL DEFAULT '',
    dirty            TEXT NOT NULL DEFAULT 'unknown',
    unpushed         INTEGER,
    created_at       TEXT NOT NULL DEFAULT '',
    -- WHY report_seq: the row is rewritten in place when the child is
    -- re-dispatched, so "delete by child id" can destroy a report that was
    -- never delivered. The sequence lets the delivery name the exact
    -- version it read - see V5_COLUMNS.
    report_seq       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS child_reports_parent_idx ON child_reports(parent_session_id);

-- ── child reclamation (v4) ───────────────────────────────────────────────────
--
-- WHY this table exists: when a delegated child's workspace is reclaimed
-- IMMEDIATELY (the child finished, so its tree is free), the work is not
-- thrown away - `jj workspace remove` snapshots the working copy into a
-- commit in the SHARED store, and that commit id is recorded HERE. This
-- table IS the index: jj never auto-gcs, so the commit stays reachable
-- through this row until a manual `jj gc --expire` evicts it.
--
-- WHY a table and not a column on `workspaces`: the workspace row is DELETED
-- on reclamation (the tree is gone), so the record of what its work turned
-- into has to live somewhere else. The primary key is (repo, slug): one
-- reclamation per child workspace, and a second reclamation of the same slug
-- is an idempotent no-op that does not overwrite the first snapshot.
CREATE TABLE IF NOT EXISTS child_reclaims (
    repo              TEXT NOT NULL,
    slug              TEXT NOT NULL,
    session_id        TEXT NOT NULL DEFAULT '',
    workspace_path    TEXT NOT NULL DEFAULT '',
    snapshot_commit_id TEXT NOT NULL DEFAULT '',
    was_dirty         INTEGER NOT NULL DEFAULT 0,
    reclaimed_at      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (repo, slug)
);
CREATE INDEX IF NOT EXISTS child_reclaims_snapshot_idx
    ON child_reclaims(snapshot_commit_id);
""""""
CREATE TABLE IF NOT EXISTS workspaces (
    repo        TEXT NOT NULL,
    slug        TEXT NOT NULL,
    path        TEXT NOT NULL DEFAULT '',
    vcs         TEXT NOT NULL DEFAULT 'jj',
    purpose     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT '',
    last_synced TEXT,
    extra       TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (repo, slug)
);
CREATE INDEX IF NOT EXISTS workspaces_path_idx ON workspaces(path);
CREATE INDEX IF NOT EXISTS workspaces_slug_idx ON workspaces(slug);

CREATE TABLE IF NOT EXISTS bindings (
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
CREATE INDEX IF NOT EXISTS bindings_path_idx ON bindings(path);

CREATE TABLE IF NOT EXISTS workspace_sessions (
    repo       TEXT NOT NULL,
    slug       TEXT NOT NULL,
    session_id TEXT NOT NULL,
    PRIMARY KEY (repo, slug, session_id)
);

CREATE TABLE IF NOT EXISTS kv (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL DEFAULT '',
    updated_at TEXT
);

-- ── child lifecycle (v3) ────────────────────────────────────────────────────
--
-- WHY these tables exist: the plugin must tell the PARENT session, on its next
-- turn, which workspace a finished child used, whether that workspace is
-- clean, and how many unpushed commits it left. The host's subagent_start /
-- subagent_stop hooks give us the child's identity and the parent's, but
-- neither tells us which workspace the child was bound to - that lookup is
-- what these two tables make cheap and explicit.
--
-- child_runs: one row per child, written at subagent_start. It is the map
--   from a child session to its parent, so subagent_stop never has to guess
--   which parent a report belongs to (a reverse lookup over bindings is
--   ambiguous when a session id is reused).
--
-- child_reports: one row per child, written at subagent_stop and read once by
--   the parent's next prompt. The row is DELETED after it is delivered
--   (one-time delivery), so a stale report can never be shown twice.

CREATE TABLE IF NOT EXISTS child_runs (
    child_session_id TEXT PRIMARY KEY,
    child_subagent_id TEXT NOT NULL DEFAULT '',
    parent_session_id TEXT NOT NULL DEFAULT '',
    parent_turn_id   TEXT NOT NULL DEFAULT '',
    parent_subagent_id TEXT NOT NULL DEFAULT '',
    child_role       TEXT NOT NULL DEFAULT '',
    child_goal       TEXT NOT NULL DEFAULT '',
    started_at       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS child_runs_parent_idx ON child_runs(parent_session_id);

CREATE TABLE IF NOT EXISTS child_reports (
    child_session_id TEXT PRIMARY KEY,
    parent_session_id TEXT NOT NULL DEFAULT '',
    child_role       TEXT NOT NULL DEFAULT '',
    child_status     TEXT NOT NULL DEFAULT '',
    workspace_path   TEXT NOT NULL DEFAULT '',
    workspace_slug   TEXT NOT NULL DEFAULT '',
    workspace_repo   TEXT NOT NULL DEFAULT '',
    dirty            TEXT NOT NULL DEFAULT 'unknown',
    unpushed         INTEGER,
    created_at       TEXT NOT NULL DEFAULT '',
    -- WHY report_seq: the row is rewritten in place when the child is
    -- re-dispatched, so "delete by child id" can destroy a report that was
    -- never delivered. The sequence lets the delivery name the exact
    -- version it read - see V5_COLUMNS.
    report_seq       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS child_reports_parent_idx ON child_reports(parent_session_id);

CREATE TABLE IF NOT EXISTS child_reclaims (
    repo              TEXT NOT NULL,
    slug              TEXT NOT NULL,
    session_id        TEXT NOT NULL DEFAULT '',
    workspace_path    TEXT NOT NULL DEFAULT '',
    snapshot_commit_id TEXT NOT NULL DEFAULT '',
    was_dirty         INTEGER NOT NULL DEFAULT 0,
    reclaimed_at      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (repo, slug)
);
CREATE INDEX IF NOT EXISTS child_reclaims_snapshot_idx
    ON child_reclaims(snapshot_commit_id);
"""

# Tables added after v2. The schema script above creates them for a FRESH
# database; an EXISTING live database needs them created too, and it must
# happen WITHOUT rewriting the rows it already has.
#
# WHY an explicit list and not "run the whole SCHEMA again": the script is
# already idempotent (every statement is CREATE ... IF NOT EXISTS), but
# running it wholesale on every open would take the DDL lock on every
# connection. Listing the additions keeps the open path cheap and makes the
# migration auditable: this tuple IS the migration.
#
# WHY the statements are IF NOT EXISTS and the migration runs them
# UNCONDITIONALLY: two branches that both bumped the schema version would
# otherwise race for it - whichever landed second saw a version row that
# already said 3, skipped its own migration, and never created its table.
# The child-report branch and the child-reclamation branch both did exactly
# that. Running the CREATEs unconditionally makes the tables independent of
# the version row; only the SCHEMA_VERSION constant itself is a merge
# decision.
V3_TABLES = """
CREATE TABLE IF NOT EXISTS child_runs (
    child_session_id TEXT PRIMARY KEY,
    child_subagent_id TEXT NOT NULL DEFAULT '',
    parent_session_id TEXT NOT NULL DEFAULT '',
    parent_turn_id   TEXT NOT NULL DEFAULT '',
    parent_subagent_id TEXT NOT NULL DEFAULT '',
    child_role       TEXT NOT NULL DEFAULT '',
    child_goal       TEXT NOT NULL DEFAULT '',
    started_at       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS child_runs_parent_idx ON child_runs(parent_session_id);

CREATE TABLE IF NOT EXISTS child_reports (
    child_session_id TEXT PRIMARY KEY,
    parent_session_id TEXT NOT NULL DEFAULT '',
    child_role       TEXT NOT NULL DEFAULT '',
    child_status     TEXT NOT NULL DEFAULT '',
    workspace_path   TEXT NOT NULL DEFAULT '',
    workspace_slug   TEXT NOT NULL DEFAULT '',
    workspace_repo   TEXT NOT NULL DEFAULT '',
    dirty            TEXT NOT NULL DEFAULT 'unknown',
    unpushed         INTEGER,
    created_at       TEXT NOT NULL DEFAULT '',
    -- WHY report_seq: the row is rewritten in place when the child is
    -- re-dispatched, so "delete by child id" can destroy a report that was
    -- never delivered. The sequence lets the delivery name the exact
    -- version it read - see V5_COLUMNS.
    report_seq       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS child_reports_parent_idx ON child_reports(parent_session_id);

CREATE TABLE IF NOT EXISTS child_reclaims (
    repo              TEXT NOT NULL,
    slug              TEXT NOT NULL,
    session_id        TEXT NOT NULL DEFAULT '',
    workspace_path    TEXT NOT NULL DEFAULT '',
    snapshot_commit_id TEXT NOT NULL DEFAULT '',
    was_dirty         INTEGER NOT NULL DEFAULT 0,
    reclaimed_at      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (repo, slug)
);
CREATE INDEX IF NOT EXISTS child_reclaims_snapshot_idx
    ON child_reclaims(snapshot_commit_id);
"""

# Columns added to a table that already exists in the live install.
#
# WHY an explicit list and not "run the whole SCHEMA again": a CREATE TABLE
# IF NOT EXISTS does NOT add a column to a table that is already there, so
# the CREATE-based migration cannot deliver a new column. These are the
# ALTER TABLE ADD COLUMN statements, and they are additive: SQLite's
# ADD COLUMN cannot drop or retype anything, so running it on the live
# database can never lose a row.
#
# WHY a new column at all (report_seq): the child's report row is keyed by
# child_session_id and REWRITTEN in place when the child is re-dispatched.
# The delivery side read the pending rows, then deleted "the rows for these
# child ids" - so a report that arrived between the read and the delete was
# destroyed WITHOUT EVER BEING DELIVERED. The sequence number lets the
# delete name the exact version it read: a rewrite bumps the sequence, the
# stale delete no longer matches, and the newer report survives to the
# parent's next turn.
V5_COLUMNS = (
    ("child_reports", "report_seq", "INTEGER NOT NULL DEFAULT 0"),
)

# The keys of a session record that get their own column. Everything else a
# caller stored (e.g. ``not_a_repo``) rides along in ``extra``.
BINDING_COLUMNS = (
    "mode", "repo", "slug", "path", "title",
    "inherited_from", "inherited_at", "session_ended_at",
)

# The keys of a workspace record that get their own column.
WORKSPACE_COLUMNS = ("repo", "slug", "path", "vcs", "purpose", "created_at", "last_synced")


# ── where the database lives ─────────────────────────────────────────────────


def hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()


def plugin_data_dir() -> Path:
    """``<hermes home>/plugin-data/hermes-treeps`` (created on demand).

    ``HERMES_TREEPS_PLUGIN_DATA_DIR`` overrides it (tests).
    """
    override = os.environ.get("HERMES_TREEPS_PLUGIN_DATA_DIR")
    if override:
        d = Path(override)
        d.mkdir(parents=True, exist_ok=True)
        return d
    try:
        from plugins.plugin_storage import plugin_data_dir as _host_dir  # type: ignore

        return Path(_host_dir(PLUGIN_NAME))
    except Exception:
        d = hermes_home() / "plugin-data" / PLUGIN_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d


def db_path() -> Path:
    """The database file. ``HERMES_TREEPS_DB_PATH`` overrides it (tests).

    WHY a thread-local PIN comes first: a background thread (the child
    report thread) can outlive the code that scheduled it, and the
    environment it resolves when it gets there may no longer be the one its
    caller meant. :func:`pinned_path` lets the caller hand the thread the
    database it wants, for as long as the thread needs it - without a
    process-wide environment variable that another thread would race on.
    """
    pinned = getattr(_THREAD_LOCAL, "pinned_path", None)
    if pinned:
        return Path(pinned)
    override = os.environ.get("HERMES_TREEPS_DB_PATH")
    if override:
        return Path(override)
    return plugin_data_dir() / DB_FILENAME


@contextlib.contextmanager
def pinned_path(path: str | Path) -> Iterator[Path]:
    """Pin the database file for the CURRENT THREAD only.

    WHY this exists: the child report thread writes its report row into a
    database file its caller named, and then reclaims the child's workspace
    - which also touches the database (reading the workspace record,
    writing the reclaim row). Without a pin, that second half would resolve
    ``db_path()`` on its own and could land in a DIFFERENT database than the
    report - the exact leak the pinned-path write was invented to stop.

    WHY thread-local and not an environment variable: an env var is
    process-wide, so two threads reporting at the same time would fight
    over it and one would write into the other's database.

    WHY it is a context manager and not a plain setter: the pin must be
    released even when the body raises, or the thread would keep writing
    into a database nobody chose for it afterwards.
    """
    previous = getattr(_THREAD_LOCAL, "pinned_path", None)
    _THREAD_LOCAL.pinned_path = str(path)
    try:
        yield Path(path)
    finally:
        _THREAD_LOCAL.pinned_path = previous


def _env_override_active() -> bool:
    """True when the caller pinned the database location with an env var.

    WHY: ``db_path()`` honours BOTH ``HERMES_TREEPS_DB_PATH`` and
    ``HERMES_TREEPS_PLUGIN_DATA_DIR``, but the old ``connect()`` only
    checked the first one before calling the host's ``plugin_db()``. The
    host's ``plugin_db()`` ALWAYS resolves the LIVE data dir, so with only
    the data-dir override set, ``db_path()`` reported a tmp path while
    ``PRAGMA database_list`` showed
    ``/var/lib/hermes/.hermes/plugin-data/hermes-treeps/data.db``: the
    override did not isolate anything. Any override means "open this file
    directly".
    """
    return bool(
        os.environ.get("HERMES_TREEPS_DB_PATH")
        or os.environ.get("HERMES_TREEPS_PLUGIN_DATA_DIR")
    )


def _set_wal(conn: sqlite3.Connection) -> None:
    """Put the database in WAL mode, retrying while another process holds it.

    WHY the retry: changing the journal mode needs a brief exclusive lock
    and does NOT go through SQLite's busy handler, so two processes opening
    a FRESH database at the same time crashed with "database is locked" -
    a crash that ``timeout=30`` does not save. Retrying with a short sleep
    turns that race into a wait.
    """
    last: Exception | None = None
    for attempt in range(_WAL_RETRIES):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            last = exc
            time.sleep(_WAL_RETRY_DELAY_S)
    raise last if last is not None else sqlite3.OperationalError("could not set WAL mode")


def _host_connection(path: str) -> sqlite3.Connection | None:
    """Open the host's sanctioned database, or None to open it ourselves.

    WHY the host API is only used when NOTHING is overridden: the host's
    ``plugin_db()`` always resolves the LIVE plugin-data dir, so an
    override that still went through it would read and write the install
    the caller meant to stay out of. With any override set we open the
    requested file directly.

    WHY the retry: ``plugin_db()`` sets the journal mode itself, and that
    change does not go through SQLite's busy handler - two processes
    opening a FRESH database at the same time crash with "database is
    locked" there. Retrying turns the race into a wait.
    """
    last: Exception | None = None
    for _ in range(_WAL_RETRIES):
        try:
            from plugins.plugin_storage import plugin_db  # type: ignore

            conn = plugin_db(PLUGIN_NAME)
        except ImportError:
            return None
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            last = exc
            time.sleep(_WAL_RETRY_DELAY_S)
            continue
        # The host may resolve a different file than we asked for (a profile
        # switch, say). If it does, use OUR path instead: a connection to
        # the wrong file is worse than no connection at all.
        try:
            row = conn.execute("PRAGMA database_list").fetchone()
            opened = str(row[2]) if row is not None and len(row) > 2 else ""
        except Exception:
            opened = ""
        if opened and Path(opened) != Path(path):
            _close_quietly(conn)
            return None
        return conn
    if last is not None:
        raise last
    return None


def _migrate_to_current(conn: sqlite3.Connection) -> None:
    """Bring an EXISTING database up to ``SCHEMA_VERSION``, additively.

    WHY this exists: the schema script only runs for a FRESH database (the
    ``bindings`` table has to be absent). The live install's database already
    has it, so the tables added after v2 would never be created there and
    every child report or immediate reclamation would fail with "no such
    table".

    WHY it is safe to run on every open: every step only CREATEs a missing
    table or index. Nothing is dropped, retyped or rewritten, so opening an
    old database can never lose a live session's binding.

    WHY the CREATEs run UNCONDITIONALLY and not behind a version check: the
    statements are ``IF NOT EXISTS``, so running them is free and idempotent
    - and it makes the migration independent of the version row. Two
    branches that both bumped the schema version would otherwise race for
    it: whichever landed second saw a version row that already said 3,
    skipped its own migration, and never created its table. The child-report
    branch and the child-reclamation branch both did exactly that. Running
    the CREATEs unconditionally removes the coupling: every table is there
    after any open, whatever the version row says.

    WHY the failure is swallowed: a database that cannot be migrated (a
    read-only file, a locked disk) must not make every plugin call fail. The
    report code degrades to "no report" and the reclamation code to "could
    not record the snapshot" instead.
    """
    try:
        conn.executescript(V3_TABLES)
        conn.commit()
        _add_missing_columns(conn)
        conn.commit()
    except Exception:
        logger.warning("plugin db migration failed; continuing on the old schema", exc_info=True)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Add the columns a newer release needs to an OLDER table.

    WHY this exists: the CREATE-based migration above cannot add a column to
    a table that already exists, and the live install's database predates
    every column added after v3. SQLite's ``ALTER TABLE ... ADD COLUMN`` is
    additive by construction - it cannot drop or retype a row - so this is
    safe to run on the live database.

    WHY the failure is swallowed per column: one missing column must not
    stop the others from being added, and a database that cannot be altered
    must still open.
    """
    for table, column, decl in V5_COLUMNS:
        try:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        except Exception:
            logger.debug("could not inspect %s for the %s column", table, column, exc_info=True)
            continue
        if not cols:
            # The table itself does not exist yet (a fresh database): the
            # CREATE-based migration creates it with the column inline.
            continue
        if column in cols:
            continue
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        except Exception:
            logger.warning("could not add the %s.%s column", table, column, exc_info=True)


def _open_connection(path: str) -> sqlite3.Connection:
    """Open a fresh connection to ``path`` with the schema in place.

    The schema is created BEFORE the connection is handed out, and the
    creation happens under ``_INIT_LOCK``: the old code published the
    connection and only then created the schema, so a concurrent reader
    could hit "no such table: bindings" on a database that existed for
    microseconds.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn: sqlite3.Connection | None = None
    if not _env_override_active():
        try:
            conn = _host_connection(path)
        except Exception:
            logger.debug("host plugin_db unavailable; using a local sqlite file", exc_info=True)
            conn = None
    if conn is None:
        conn = sqlite3.connect(path, check_same_thread=False, timeout=_BUSY_TIMEOUT_S)
    conn.row_factory = sqlite3.Row
    try:
        with _INIT_LOCK:
            # A long busy timeout: the host's connection defaults to five
            # seconds, which turns a busy database into "database is
            # locked" instead of a wait.
            conn.execute(f"PRAGMA busy_timeout={int(_BUSY_TIMEOUT_S * 1000)}")
            _set_wal(conn)
            conn.execute("PRAGMA foreign_keys=ON")
            # WHY the existence check instead of a "have I done this already"
            # flag: the database file can be replaced under us (a test
            # pointing at a fresh path, a profile switch), and then the
            # schema has to be created again.
            have = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='bindings'"
            ).fetchone()
            if have is None:
                conn.executescript(SCHEMA)
                conn.commit()
            else:
                # An EXISTING database (the live install, or one a test made
                # before this release): bring it forward ADDITIVELY. The
                # migration only ever CREATEs a missing table - it never
                # drops, rewrites or retypes a row - so opening an old
                # database can never lose a live session's binding.
                _migrate_to_current(conn)
            # Record the version on the SAME connection, for fresh AND existing
            # databases alike: a future migration reads this row to decide what
            # to do, and an upgraded install must not look like a v1 database.
            row = conn.execute("SELECT value FROM kv WHERE key='schema_version'").fetchone()
            if row is None or str(row[0]) != str(SCHEMA_VERSION):
                _write_kv(conn, "schema_version", str(SCHEMA_VERSION))
                conn.commit()
    except Exception:
        # Never hand out a half-initialised connection.
        try:
            conn.close()
        except Exception:
            logger.debug("could not close a failed connection", exc_info=True)
        raise
    return conn


def connect_to(path: str | Path) -> sqlite3.Connection:
    """Open a connection to a SPECIFIC database file, bypassing ``db_path()``.

    WHY this exists: ``connect()`` resolves the path when it is called, so a
    background thread that outlives the code that scheduled it can resolve a
    DIFFERENT database than the one its caller meant. That is not
    hypothetical: the plugin's own tests point the database at a tmp dir with
    an environment override, and the override is undone when the test ends -
    so a report thread still running after that point resolved the LIVE
    database and wrote a row into it.

    So: the caller captures the path ONCE, while its environment is still
    what it meant, and the background thread is handed that path.

    The connection is NOT cached in thread-local storage and NOT registered
    for ``reset()``: it belongs to one caller, is closed by that caller, and
    must not be closed by anybody else (closing another thread's connection
    is exactly what segfaulted this process once).
    """
    return _open_connection(str(path))


@contextlib.contextmanager
def transaction_at(path: str | Path, commit: bool = True) -> Iterator[sqlite3.Connection]:
    """One write transaction against a SPECIFIC database file.

    The same contract as :func:`transaction`, but on a connection this
    caller owns. Use it from a background thread that must not re-resolve
    the database path.
    """
    conn = connect_to(path)
    try:
        with _WRITE_LOCK:
            try:
                yield conn
            except BaseException:
                try:
                    conn.rollback()
                except Exception:
                    logger.debug("rollback failed", exc_info=True)
                raise
            else:
                if commit:
                    conn.commit()
                else:
                    conn.rollback()
    finally:
        _close_quietly(conn)


def connect() -> sqlite3.Connection:
    """This thread's connection to the plugin database.

    WHY one connection per thread: python's sqlite3 caches one prepared
    statement per (connection, sql) pair, so two threads sharing a
    connection clobber each other's statement state. That produced
    "bad parameter or other API misuse", reads that returned the wrong
    number of rows, and - because ``bind_session`` keys its INSERT on the
    record it just read - a session bound to the WRONG workspace. The gate
    swallows that exception and fails open, so it was a security hole.

    The connection is cached in thread-local storage and is therefore used
    by exactly one thread. WAL already allows many readers beside one
    writer, and ``_WRITE_LOCK`` keeps the writers serialised.
    """
    path = str(db_path())
    owner = threading.current_thread()
    with _REGISTRY_LOCK:
        generation = _GENERATION
        entry = _REGISTRY.get(owner)
        _prune_dead_threads()

    cached = getattr(_THREAD_LOCAL, "conn", None)
    if cached is not None:
        fresh = (
            entry is not None
            and entry["generation"] == generation
            and entry["path"] == path
            and entry["id"] == id(cached)
        )
        if fresh:
            return cached
        # Stale: reset() ran (or the path changed) since this connection was
        # opened. Close it HERE - this thread owns it, so nobody else can be
        # inside a statement on it - and open a fresh one.
        _close_quietly(cached)
        _THREAD_LOCAL.conn = None

    conn = _open_connection(path)
    with _REGISTRY_LOCK:
        _REGISTRY[owner] = {
            "id": id(conn),
            "generation": generation,
            "path": path,
            "thread": owner,
        }
    _THREAD_LOCAL.conn = conn
    return conn


def _prune_dead_threads() -> None:
    """Forget the registry entries of threads that have exited.

    WHY: the registry lives for the lifetime of the process, and a dead
    thread's entry would otherwise be inherited by an unrelated thread
    that reuses the same identifier.
    """
    for key in [
        k
        for k, v in _REGISTRY.items()
        if not v.get("thread") or not v["thread"].is_alive()
    ]:
        _REGISTRY.pop(key, None)


def _close_quietly(conn: sqlite3.Connection | None) -> None:
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        logger.debug("could not close the plugin db", exc_info=True)


def reset() -> None:
    """Invalidate every connection in the process.

    WHY this cannot close other threads' connections: the old
    implementation called ``conn.close()`` on the ONE shared connection
    while another thread was mid-statement on it, and CPython segfaulted
    the process (5/5 runs). Now every thread owns its connection, so
    ``reset()`` bumps the generation counter; each thread notices the
    change the next time it asks for a connection and reopens its own.
    Only the CALLING thread's connection is closed here - and it is closed
    by the thread that owns it, which is the only safe way to close it.
    """
    global _GENERATION
    with _REGISTRY_LOCK:
        _GENERATION += 1
    owner = threading.current_thread()
    cached = getattr(_THREAD_LOCAL, "conn", None)
    _THREAD_LOCAL.conn = None
    _close_quietly(cached)
    with _REGISTRY_LOCK:
        _REGISTRY.pop(owner, None)


@contextlib.contextmanager
def transaction(commit: bool = True) -> Iterator[sqlite3.Connection]:
    """One write transaction: commits on success, rolls back on any error.

    WHY a context manager: every mutating function used to do its writes
    and then ``conn.commit()`` with no rollback on the failure path. A
    function that raised half-way through left a DIRTY transaction on the
    connection, and the NEXT unrelated commit published the half-write (a
    changed ``purpose`` and the session rows gone). A transaction that
    always ends in commit-or-rollback removes that class of bug, and
    guarantees no function leaves the connection in a transaction.

    ``commit=False`` rolls back on success too - that is what a dry run
    needs: it must SEE what it would do without publishing any of it.
    """
    conn = connect()
    with _WRITE_LOCK:
        try:
            yield conn
        except BaseException:
            # Roll back on ANY failure, including a KeyboardInterrupt: a
            # half-written transaction must never survive this block.
            try:
                conn.rollback()
            except Exception:
                logger.debug("rollback failed", exc_info=True)
            raise
        else:
            if commit:
                conn.commit()
            else:
                conn.rollback()


# ── kv helpers (tokens, caches, markers) ─────────────────────────────────────


def get_kv(key: str, default: str | None = None) -> str | None:
    conn = connect()
    row = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    return str(row[0])


def set_kv(key: str, value: str) -> None:
    with transaction() as conn:
        _write_kv(conn, key, value)


def get_kv_in(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    """The kv value, read on an EXISTING connection.

    WHY this exists: a read-then-write pair must be atomic, and the
    read has to happen on the SAME connection (and inside the same
    transaction) as the write. Reading through :func:`get_kv` opens its
    own connection, so the value it returned could already be stale by
    the time the write landed.
    """
    row = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    return str(row[0])


def _write_kv(conn: sqlite3.Connection, key: str, value: str) -> None:
    """The kv upsert, on a given connection.

    WHY it is separate: schema creation has to record the schema version
    on the SAME connection it created the schema on, and it must not call
    ``set_kv()`` while holding ``_INIT_LOCK`` - that would take the locks
    in the opposite order to ``transaction()`` (``_WRITE_LOCK`` then
    ``_INIT_LOCK``) and deadlock two threads against each other.
    """
    conn.execute(
        "INSERT INTO kv(key, value, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    )


def get_kv_json(key: str, default: Any = None) -> Any:
    raw = get_kv(key)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except Exception:
        logger.warning("kv %s is not valid JSON; ignoring", key)
        return default


def set_kv_json(key: str, value: Any) -> None:
    set_kv(key, json.dumps(value, sort_keys=True))


# ── session records (the old state/<sid>.json) ───────────────────────────────


def read_binding(session_id: str) -> dict[str, Any] | None:
    """The session record, or None when the session has none."""
    conn = connect()
    row = conn.execute("SELECT * FROM bindings WHERE session_id=?", (session_id,)).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"] or "{}")
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    extra = {}
    try:
        extra = json.loads(row["extra"] or "{}")
    except Exception:
        extra = {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            payload.setdefault(k, v)
    payload["session_id"] = row["session_id"]
    payload["updated_at"] = row["updated_at"]
    return payload


def write_binding(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Replace a session record. One transaction, one write."""
    payload = dict(payload)
    payload.setdefault("session_id", session_id)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    extra = {k: v for k, v in payload.items() if k not in BINDING_COLUMNS and k not in ("session_id", "updated_at")}
    with transaction() as conn:
        conn.execute(
            "INSERT INTO bindings(session_id, mode, repo, slug, path, title, inherited_from,"
            " inherited_at, session_ended_at, updated_at, extra, payload)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(session_id) DO UPDATE SET mode=excluded.mode, repo=excluded.repo,"
            " slug=excluded.slug, path=excluded.path, title=excluded.title,"
            " inherited_from=excluded.inherited_from, inherited_at=excluded.inherited_at,"
            " session_ended_at=excluded.session_ended_at, updated_at=excluded.updated_at,"
            " extra=excluded.extra, payload=excluded.payload",
            (
                session_id,
                payload.get("mode"),
                payload.get("repo"),
                payload.get("slug"),
                payload.get("path"),
                payload.get("title"),
                payload.get("inherited_from"),
                payload.get("inherited_at"),
                payload.get("session_ended_at"),
                payload["updated_at"],
                json.dumps(extra, sort_keys=True),
                json.dumps(payload, sort_keys=True),
            ),
        )
    return payload


def delete_binding(session_id: str) -> bool:
    """Drop a session record and its claim on any workspace. True when one existed.

    WHY the join row goes too: `forget()` used to delete only the binding, so
    the workspace kept listing the session in `sessions` and `retire()` refused
    it forever with "bound session(s) still live" - a ghost with no binding,
    no record and no way to clear it from the workspace.
    """
    with transaction() as conn:
        cur = conn.execute("DELETE FROM bindings WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM workspace_sessions WHERE session_id=?", (session_id,))
        return cur.rowcount > 0


def workspace_key(record: dict[str, Any]) -> tuple[str, str]:
    repo = str(record.get("repo") or "")
    slug = str(record.get("slug") or "")
    return repo, slug


# ── child lifecycle rows (v3) ───────────────────────────────────────────────
#
# WHY a dedicated row per child rather than a kv blob: the stop side has to
# look the child up by its session id, and a kv blob would mean reading and
# rewriting one JSON document per child - the exact read-modify-write race
# this module's own docstring says cost four of eight concurrent binds.


def write_child_run(
    child_session_id: str,
    parent_session_id: str,
    *,
    child_subagent_id: str = "",
    parent_turn_id: str = "",
    parent_subagent_id: str = "",
    child_role: str = "",
    child_goal: str = "",
) -> None:
    """Record a child run: which child, which parent.

    Idempotent on ``child_session_id``: a re-dispatched child overwrites its
    own row instead of accumulating duplicates.
    """
    with transaction() as conn:
        conn.execute(
            "INSERT INTO child_runs(child_session_id, child_subagent_id, parent_session_id,"
            " parent_turn_id, parent_subagent_id, child_role, child_goal, started_at)"
            " VALUES(?,?,?,?,?,?,?,?)"
            " ON CONFLICT(child_session_id) DO UPDATE SET child_subagent_id=excluded.child_subagent_id,"
            " parent_session_id=excluded.parent_session_id, parent_turn_id=excluded.parent_turn_id,"
            " parent_subagent_id=excluded.parent_subagent_id, child_role=excluded.child_role,"
            " child_goal=excluded.child_goal, started_at=excluded.started_at",
            (
                child_session_id,
                child_subagent_id,
                parent_session_id,
                parent_turn_id,
                parent_subagent_id,
                child_role,
                child_goal,
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )


def read_child_run(child_session_id: str) -> dict[str, Any] | None:
    """The child's run row, or None when the start side never ran."""
    if not child_session_id:
        return None
    conn = connect()
    row = conn.execute(
        "SELECT * FROM child_runs WHERE child_session_id=?", (child_session_id,)
    ).fetchone()
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


def write_child_report(
    child_session_id: str,
    parent_session_id: str,
    *,
    child_role: str = "",
    child_status: str = "",
    workspace_path: str = "",
    workspace_slug: str = "",
    workspace_repo: str = "",
    dirty: str = "unknown",
    unpushed: int | None = None,
) -> None:
    """Record the child's end state, keyed by the child.

    LAST STATE WINS: the row is keyed by ``child_session_id``, so a child that
    stops twice (or is re-dispatched under the same session id) overwrites
    its own earlier report instead of showing the parent two lines about the
    same child.

    WHY ``report_seq`` is bumped on every write: the row is overwritten IN
    PLACE, so a delivery that read the old version cannot tell it apart from
    the new one by id alone. Bumping the sequence on each write lets the
    delivery name the exact version it read - so a report rewritten between
    the read and the delete is NOT deleted, and the parent still gets it.
    """
    with transaction() as conn:
        conn.execute(
            "INSERT INTO child_reports(child_session_id, parent_session_id, child_role,"
            " child_status, workspace_path, workspace_slug, workspace_repo, dirty, unpushed,"
            " created_at, report_seq) VALUES(?,?,?,?,?,?,?,?,?,?,1)"
            " ON CONFLICT(child_session_id) DO UPDATE SET parent_session_id=excluded.parent_session_id,"
            " child_role=excluded.child_role, child_status=excluded.child_status,"
            " workspace_path=excluded.workspace_path, workspace_slug=excluded.workspace_slug,"
            " workspace_repo=excluded.workspace_repo, dirty=excluded.dirty,"
            " unpushed=excluded.unpushed, created_at=excluded.created_at,"
            " report_seq=child_reports.report_seq+1",
            (
                child_session_id,
                parent_session_id,
                child_role,
                child_status,
                workspace_path,
                workspace_slug,
                workspace_repo,
                dirty,
                unpushed,
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )


def write_child_report_at(
    path: str | Path,
    child_session_id: str,
    parent_session_id: str,
    *,
    child_role: str = "",
    child_status: str = "",
    workspace_path: str = "",
    workspace_slug: str = "",
    workspace_repo: str = "",
    dirty: str = "unknown",
    unpushed: int | None = None,
) -> None:
    """Record the child's end state in a SPECIFIC database file.

    The pinned-path twin of :func:`write_child_report`, for the report
    thread. See :func:`connect_to` for why the path must be captured by the
    caller instead of resolved here: a background thread that resolves the
    path itself can end up in the LIVE database.
    """
    with transaction_at(path) as conn:
        conn.execute(
            "INSERT INTO child_reports(child_session_id, parent_session_id, child_role,"
            " child_status, workspace_path, workspace_slug, workspace_repo, dirty, unpushed,"
            " created_at, report_seq) VALUES(?,?,?,?,?,?,?,?,?,?,1)"
            " ON CONFLICT(child_session_id) DO UPDATE SET parent_session_id=excluded.parent_session_id,"
            " child_role=excluded.child_role, child_status=excluded.child_status,"
            " workspace_path=excluded.workspace_path, workspace_slug=excluded.workspace_slug,"
            " workspace_repo=excluded.workspace_repo, dirty=excluded.dirty,"
            " unpushed=excluded.unpushed, created_at=excluded.created_at,"
            " report_seq=child_reports.report_seq+1",
            (
                child_session_id,
                parent_session_id,
                child_role,
                child_status,
                workspace_path,
                workspace_slug,
                workspace_repo,
                dirty,
                unpushed,
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )


def read_child_reports(parent_session_id: str) -> list[dict[str, Any]]:
    """Every pending report for a parent session, oldest first.

    Reading does NOT delete: the caller deletes what it delivered, so a
    failed render leaves the report pending rather than losing it.
    """
    if not parent_session_id:
        return []
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM child_reports WHERE parent_session_id=? ORDER BY created_at ASC, rowid ASC",
        (parent_session_id,),
    ).fetchall()
    return [{k: row[k] for k in row.keys()} for row in rows]


def delete_child_reports(child_session_ids: list[str]) -> int:
    """Delete the reports that were delivered. Returns the row count.

    WHY this is ID-ONLY and why that is a trap: the caller passes the child
    ids it read. The row is rewritten IN PLACE when the child is
    re-dispatched, so a report that arrived between the read and this
    delete shares the id and is destroyed WITHOUT EVER BEING DELIVERED. Use
    :func:`delete_delivered_reports` - it names the exact version.
    """
    ids = [str(x) for x in child_session_ids if x]
    if not ids:
        return 0
    with transaction() as conn:
        marks = ",".join("?" for _ in ids)
        cur = conn.execute(
            f"DELETE FROM child_reports WHERE child_session_id IN ({marks})", tuple(ids)
        )
        return int(cur.rowcount or 0)


def delete_delivered_reports(delivered: list[dict[str, Any]]) -> int:
    """Delete EXACTLY the report versions that were delivered.

    WHY this exists (P1-2): the delivery reads the pending rows, renders
    them, and then deletes them. Between the read and the delete the child
    can be re-dispatched, and its report thread REWRITES the same row (the
    primary key is the child session id). Deleting by child id then
    destroyed the NEW report - one the parent never saw. The parent silently
    lost the child's end state, which is the whole point of the report.

    The fix names the version: every write bumps ``report_seq``, the read
    captures it, and this delete matches on ``(child_session_id, report_seq)``.
    A rewrite between the read and the delete changes the sequence, so the
    delete matches nothing and the newer report survives to the parent's next
    turn.

    Rows with NO sequence (an old database written before the column
    existed) are deleted by id alone - there is nothing else to match on,
    and leaving them pending forever would be worse.
    """
    clauses: list[str] = []
    params: list[Any] = []
    for row in delivered:
        child_id = str(row.get("child_session_id") or "")
        if not child_id:
            continue
        seq = row.get("report_seq")
        if seq is None:
            clauses.append("(child_session_id=? AND report_seq=0)")
            params.append(child_id)
        else:
            clauses.append("(child_session_id=? AND report_seq=?)")
            params.extend([child_id, int(seq)])
    if not clauses:
        return 0
    with transaction() as conn:
        cur = conn.execute(
            "DELETE FROM child_reports WHERE " + " OR ".join(clauses), tuple(params)
        )
        return int(cur.rowcount or 0)


def read_workspace(path: str | Path) -> dict[str, Any] | None:
    """The workspace record for a path, or None."""
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM workspaces WHERE path=? ORDER BY created_at DESC", (str(path),)
    ).fetchall()
    if not rows:
        return None
    row = rows[0]
    record = _workspace_row_to_dict(conn, row)
    return record


def _workspace_row_to_dict(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    record = {
        "repo": row["repo"],
        "slug": row["slug"],
        "path": row["path"],
        "vcs": row["vcs"],
        "purpose": row["purpose"],
        "created_at": row["created_at"],
        "last_synced": row["last_synced"],
    }
    try:
        extra = json.loads(row["extra"] or "{}")
    except Exception:
        extra = {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            record.setdefault(k, v)
    record["sessions"] = [
        str(r[0])
        for r in conn.execute(
            "SELECT session_id FROM workspace_sessions WHERE repo=? AND slug=? ORDER BY session_id",
            (row["repo"], row["slug"]),
        )
    ]
    return record


def write_workspace(record: dict[str, Any], path: str | Path | None = None) -> None:
    """Replace a workspace record (and its session list). One transaction."""
    repo, slug = workspace_key(record)
    if not repo or not slug:
        raise ValueError(f"a workspace record needs repo and slug, got {record!r}")
    where = str(path if path is not None else record.get("path") or "")
    extra = {
        k: v for k, v in record.items()
        if k not in WORKSPACE_COLUMNS and k != "sessions"
    }
    sessions = [str(s) for s in (record.get("sessions") or [])]
    with transaction() as conn:
        conn.execute(
            "INSERT INTO workspaces(repo, slug, path, vcs, purpose, created_at, last_synced, extra)"
            " VALUES(?,?,?,?,?,?,?,?)"
            " ON CONFLICT(repo, slug) DO UPDATE SET path=excluded.path, vcs=excluded.vcs,"
            " purpose=excluded.purpose, created_at=excluded.created_at,"
            " last_synced=excluded.last_synced, extra=excluded.extra",
            (
                repo,
                slug,
                where,
                str(record.get("vcs") or "jj"),
                str(record.get("purpose") or ""),
                str(record.get("created_at") or ""),
                record.get("last_synced"),
                json.dumps(extra, sort_keys=True),
            ),
        )
        conn.execute("DELETE FROM workspace_sessions WHERE repo=? AND slug=?", (repo, slug))
        for sid in sessions:
            conn.execute(
                "INSERT OR IGNORE INTO workspace_sessions(repo, slug, session_id) VALUES(?,?,?)",
                (repo, slug, sid),
            )


def rename_workspace(repo: str, old_slug: str, new_slug: str, new_path: str) -> None:
    """Move a workspace row to a new slug AND path, in one transaction.

    The primary key is ``(repo, slug)``, so a rename is a key change: the old
    row has to go and the session rows have to move with it. Leaving the old
    row behind would leave a stale workspace pointing at a path that no longer
    exists, and leaving the session rows behind would keep the OLD workspace
    claiming the session - which is exactly what retire() refuses forever.
    """
    with transaction() as conn:
        row = conn.execute(
            "SELECT path FROM workspaces WHERE repo=? AND slug=?", (repo, old_slug)
        ).fetchone()
        if row is None:
            raise ValueError(f"no workspace {repo}/{old_slug} to rename")
        conn.execute(
            "UPDATE workspaces SET slug=?, path=? WHERE repo=? AND slug=?",
            (new_slug, new_path, repo, old_slug),
        )
        conn.execute(
            "UPDATE workspace_sessions SET slug=? WHERE repo=? AND slug=?",
            (new_slug, repo, old_slug),
        )


def add_workspace_session(repo: str, slug: str, session_id: str) -> None:
    """Bind one session to a workspace. Idempotent, one transaction.

    This is the operation the old JSON read-modify-write lost: eight
    concurrent calls each appended to a list they had read, and the last
    writer wrote back a list missing the other seven.
    """
    with transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_sessions(repo, slug, session_id) VALUES(?,?,?)",
            (repo, slug, str(session_id)),
        )


def remove_workspace_session(repo: str, slug: str, session_id: str) -> None:
    with transaction() as conn:
        conn.execute(
            "DELETE FROM workspace_sessions WHERE repo=? AND slug=? AND session_id=?",
            (repo, slug, str(session_id)),
        )


def workspace_sessions(repo: str, slug: str) -> list[str]:
    conn = connect()
    return [
        str(r[0])
        for r in conn.execute(
            "SELECT session_id FROM workspace_sessions WHERE repo=? AND slug=? ORDER BY session_id",
            (repo, slug),
        )
    ]


def workspace_paths_for_slug(slug: str) -> list[str]:
    """Every recorded path for a slug (there can be more than one: ambiguous)."""
    conn = connect()
    return [
        str(r[0])
        for r in conn.execute("SELECT path FROM workspaces WHERE slug=? ORDER BY created_at", (slug,))
    ]


def all_workspaces() -> list[dict[str, Any]]:
    """Every workspace record, oldest first.

    WHY the migration needs the whole table and not a per-slug lookup: the
    move is driven by the recorded PATH (is it under the old root?), and a
    workspace whose record no longer matches its slug would be invisible to
    a slug-keyed query.
    """
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM workspaces ORDER BY created_at, repo, slug"
    ).fetchall()
    return [_workspace_row_to_dict(conn, row) for row in rows]


def all_bindings() -> list[dict[str, Any]]:
    """Every session binding row, as plain dicts.

    WHY the raw rows and not ``read_binding``: the migration needs to know
    which sessions are still LIVE, and that is the ``session_ended_at``
    column - a per-session query would be one round trip per row.
    """
    conn = connect()
    rows = conn.execute(
        "SELECT session_id, mode, repo, slug, path, session_ended_at FROM bindings"
    ).fetchall()
    return [
        {
            "session_id": str(r["session_id"]),
            "mode": str(r["mode"] or ""),
            "repo": str(r["repo"] or ""),
            "slug": str(r["slug"] or ""),
            "path": str(r["path"] or ""),
            "session_ended_at": str(r["session_ended_at"] or ""),
        }
        for r in rows
    ]


def rebind_workspace_path(repo: str, slug: str, new_path: str) -> None:
    """Point a workspace (and every session bound to it) at a new path.

    WHY the session rows move with the workspace row: a binding that still
    names the old path is a session whose tree "is gone" - resolve_detail
    calls that state stale, and the session is broken until somebody repairs
    it by hand.

    WHY this goes through ``write_binding`` and not a bare UPDATE: the
    binding's path lives in TWO places - the ``path`` column AND the
    ``payload`` JSON - and ``read_binding`` returns the payload. A bare
    UPDATE of the column would leave the payload naming the old path, so the
    session would still resolve to a tree that is gone.
    """
    with transaction() as conn:
        conn.execute(
            "UPDATE workspaces SET path=? WHERE repo=? AND slug=?",
            (new_path, repo, slug),
        )
    rows = connect().execute(
        "SELECT session_id FROM bindings WHERE repo=? AND slug=?",
        (repo, slug),
    ).fetchall()
    for row in rows:
        sid = str(row["session_id"])
        payload = read_binding(sid)
        if not payload:
            continue
        payload["path"] = new_path
        write_binding(sid, payload)


def delete_workspace(repo: str, slug: str) -> None:
    with transaction() as conn:
        conn.execute("DELETE FROM workspace_sessions WHERE repo=? AND slug=?", (repo, slug))
        conn.execute("DELETE FROM workspaces WHERE repo=? AND slug=?", (repo, slug))


# ── child reclamation index ───────────────────────────────────────────────────


def record_child_reclaim(
    repo: str,
    slug: str,
    *,
    session_id: str = "",
    workspace_path: str = "",
    snapshot_commit_id: str = "",
    was_dirty: bool = False,
    reclaimed_at: str = "",
) -> None:
    """Record that a child workspace was reclaimed, and what its work became.

    WHY this table is the index: jj NEVER auto-gcs. A snapshot commit created
    by ``jj workspace remove`` stays in the shared store until a manual
    ``jj gc --expire`` evicts it, so the commit id is enough to find the work
    later - but only if something records it. This row IS that index.

    WHY the row is written and NOT upserted over an existing one: a second
    reclamation of the same slug is an idempotent no-op, and overwriting the
    first snapshot id would lose the pointer to the FIRST snapshot - which
    is the one holding the child's work.
    """
    with transaction() as conn:
        conn.execute(
            "INSERT INTO child_reclaims"
            " (repo, slug, session_id, workspace_path, snapshot_commit_id, was_dirty, reclaimed_at)"
            " VALUES(?,?,?,?,?,?,?)"
            " ON CONFLICT(repo, slug) DO NOTHING",
            (
                repo,
                slug,
                str(session_id or ""),
                str(workspace_path or ""),
                str(snapshot_commit_id or ""),
                1 if was_dirty else 0,
                str(reclaimed_at or ""),
            ),
        )


def read_child_reclaim(repo: str, slug: str) -> dict[str, Any] | None:
    """The reclamation row for a child workspace, or None."""
    conn = connect()
    row = conn.execute(
        "SELECT * FROM child_reclaims WHERE repo=? AND slug=?", (repo, slug)
    ).fetchone()
    if row is None:
        return None
    return {
        "repo": row["repo"],
        "slug": row["slug"],
        "session_id": row["session_id"],
        "workspace_path": row["workspace_path"],
        "snapshot_commit_id": row["snapshot_commit_id"],
        "was_dirty": bool(row["was_dirty"]),
        "reclaimed_at": row["reclaimed_at"],
    }


def read_child_reclaim_by_slug(slug: str) -> dict[str, Any] | None:
    """The reclamation row for a slug in ANY repo, or None.

    WHY the repo is not required: the idempotent second call happens after
    the workspace row was deleted, and that row was the only thing that
    said which repo the workspace belonged to. Slugs name a workspace tree
    and are unique within a root, so the slug alone identifies the row.

    WHY the newest row wins: two repos can hold the same slug (a repo
    named "app" in two roots). The reclamation that happened LAST is the
    one a caller asking "what happened to this slug?" wants.
    """
    conn = connect()
    row = conn.execute(
        "SELECT * FROM child_reclaims WHERE slug=? ORDER BY reclaimed_at DESC LIMIT 1",
        (slug,),
    ).fetchone()
    if row is None:
        return None
    return {
        "repo": row["repo"],
        "slug": row["slug"],
        "session_id": row["session_id"],
        "workspace_path": row["workspace_path"],
        "snapshot_commit_id": row["snapshot_commit_id"],
        "was_dirty": bool(row["was_dirty"]),
        "reclaimed_at": row["reclaimed_at"],
    }


def read_child_reclaim_for_report(report: dict[str, Any]) -> dict[str, Any] | None:
    """The reclamation row that belongs to a CHILD REPORT row, or None.

    WHY this exists: a report says "this child left a dirty tree here", and
    the reclamation says "that tree became THIS commit". Linking the two is
    what lets a caller answer "where did the child's work go?" with one
    lookup instead of guessing a slug.

    WHY no schema change: the report row already carries ``workspace_slug``
    and ``workspace_repo``, which are exactly the primary key of
    ``child_reclaims``. So the join needs no new column and no migration -
    the linkage is a QUERY, not a second copy of the commit id that could
    drift away from the reclamation that produced it.

    WHY the repo is tried first and the slug alone second: the workspace
    row (which held the repo) is deleted by the reclamation, and a report
    row written before the plugin knew the repo name can carry an empty
    one. The slug alone is unique within a root, so it is a safe fallback.
    """
    slug = str(report.get("workspace_slug") or "")
    if not slug:
        return None
    repo = str(report.get("workspace_repo") or "")
    conn = connect()
    row = None
    if repo:
        row = conn.execute(
            "SELECT * FROM child_reclaims WHERE repo=? AND slug=?",
            (repo, slug),
        ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT * FROM child_reclaims WHERE slug=? ORDER BY reclaimed_at DESC LIMIT 1",
            (slug,),
        ).fetchone()
    if row is None:
        return None
    return {
        "repo": row["repo"],
        "slug": row["slug"],
        "session_id": row["session_id"],
        "workspace_path": row["workspace_path"],
        "snapshot_commit_id": row["snapshot_commit_id"],
        "was_dirty": bool(row["was_dirty"]),
        "reclaimed_at": row["reclaimed_at"],
    }


def legacy_state_dir() -> Path:
    """Where the old per-session JSON files live (the plugin install dir).

    ``HERMES_TREEPS_LEGACY_STATE_DIR`` overrides it, which is how the tests
    stay out of the live install directory.
    """
    override = os.environ.get("HERMES_TREEPS_LEGACY_STATE_DIR")
    if override:
        return Path(override)
    return hermes_home() / "plugins" / PLUGIN_NAME / "state"
