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
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

PLUGIN_NAME = "hermes-treeps"
DB_FILENAME = "data.db"

SCHEMA_VERSION = 2

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
"""

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
    """The database file. ``HERMES_TREEPS_DB_PATH`` overrides it (tests)."""
    override = os.environ.get("HERMES_TREEPS_DB_PATH")
    if override:
        return Path(override)
    return plugin_data_dir() / DB_FILENAME


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


def delete_workspace(repo: str, slug: str) -> None:
    with transaction() as conn:
        conn.execute("DELETE FROM workspace_sessions WHERE repo=? AND slug=?", (repo, slug))
        conn.execute("DELETE FROM workspaces WHERE repo=? AND slug=?", (repo, slug))


def legacy_state_dir() -> Path:
    """Where the old per-session JSON files live (the plugin install dir).

    ``HERMES_TREEPS_LEGACY_STATE_DIR`` overrides it, which is how the tests
    stay out of the live install directory.
    """
    override = os.environ.get("HERMES_TREEPS_LEGACY_STATE_DIR")
    if override:
        return Path(override)
    return hermes_home() / "plugins" / PLUGIN_NAME / "state"


def legacy_workspace_records(workspace_root: Path) -> list[Path]:
    """Every ``<workspace>/.hermes/session.json`` under the workspace root.

    Two layouts are supported, because the root predates this plugin: the flat
    one (``<root>/<repo>-<slug>``) and the nested one (``<root>/<repo>/<slug>``).
    """
    out: list[Path] = []
    if not workspace_root.is_dir():
        return out
    for top in sorted(p for p in workspace_root.iterdir() if p.is_dir()):
        rec = top / ".hermes" / "session.json"
        if rec.is_file():
            out.append(rec)
            continue
        for ws in sorted(p for p in top.iterdir() if p.is_dir()):
            rec = ws / ".hermes" / "session.json"
            if rec.is_file():
                out.append(rec)
    return out


def _derive_repo_slug(record: dict[str, Any], path: Path, root: Path | None = None) -> tuple[str, str]:
    """repo/slug for a legacy record, from its fields or from where it sits.

    ``path`` is the record FILE, so it has to be reduced to the workspace
    directory first: taking ``path.parent.name`` gave every workspace whose
    record carried no repo/slug the slug "``.hermes``" and its own name as the
    repo. That shipped once (the hermes-treeps row came out as
    ``repo="hermes-treeps-ee445923", slug=".hermes"``), which is why the
    in-tree record - not the record's fields - is the source of truth here.
    """
    ws = path.parent.parent if path.parent.name == ".hermes" else path
    repo = str(record.get("repo") or "")
    slug = str(record.get("slug") or "")
    if not slug:
        slug = ws.name
    if not repo:
        # Nested layout: <root>/<repo>/<slug>. Flat layout: <root>/<repo>-<slug>.
        # The root decides, not the names: a nested slug usually repeats the
        # repo name ("hermes-treeps/hermes-treeps-ee445923"), so any
        # name-prefix heuristic misfires on exactly the common case.
        if root is not None and ws.parent.resolve() == Path(root).resolve():
            repo = ws.name.split("-")[0]
        else:
            repo = ws.parent.name
    return repo, slug


def _import_binding(conn: sqlite3.Connection, session_id: str, payload: dict[str, Any]) -> bool:
    extra = {k: v for k, v in payload.items() if k not in BINDING_COLUMNS and k not in ("session_id", "updated_at")}
    row = conn.execute("SELECT session_id FROM bindings WHERE session_id=?", (session_id,)).fetchone()
    if row is not None:
        return False  # never overwrite a live binding with a legacy file
    conn.execute(
        "INSERT INTO bindings(session_id, mode, repo, slug, path, title, inherited_from,"
        " inherited_at, session_ended_at, updated_at, extra, payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
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
            payload.get("updated_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            json.dumps(extra, sort_keys=True),
            json.dumps(payload, sort_keys=True),
        ),
    )
    return True


def _import_workspace_record(
    conn: sqlite3.Connection, path: Path, record: dict[str, Any], root: Path | None = None
) -> bool:
    repo, slug = _derive_repo_slug(record, path, root)
    if not repo or not slug:
        return False
    row = conn.execute(
        "SELECT repo, slug FROM workspaces WHERE repo=? AND slug=?", (repo, slug)
    ).fetchone()
    if row is not None:
        return False
    extra = {
        k: v for k, v in record.items()
        if k not in WORKSPACE_COLUMNS and k != "sessions"
    }
    conn.execute(
        "INSERT INTO workspaces(repo, slug, path, vcs, purpose, created_at, last_synced, extra)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (
            repo,
            slug,
            # The workspace directory, not the .hermes directory inside it:
            # path.parent is <ws>/.hermes, path.parent.parent is <ws>.
            str(record.get("path") or path.parent.parent),
            str(record.get("vcs") or "jj"),
            str(record.get("purpose") or ""),
            str(record.get("created_at") or ""),
            record.get("last_synced"),
            json.dumps(extra, sort_keys=True),
        ),
    )
    for sid in record.get("sessions") or []:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_sessions(repo, slug, session_id) VALUES(?,?,?)",
            (repo, slug, str(sid)),
        )
    return True


def migrate_legacy(workspace_root: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Import the legacy JSON records into the database. Idempotent.

    Imports the per-session state files and every workspace's
    ``.hermes/session.json``. A corrupt file is skipped with a warning - one
    broken record must not abort the migration of the rest. Existing rows are
    never overwritten, so running this twice cannot lose a binding.

    ``dry_run`` counts what would be imported and then rolls the whole thing
    back: a preview that writes is not a preview, and the first version of
    this function proved that the hard way by importing 38 rows under
    ``--dry-run``.
    """
    root = Path(workspace_root) if workspace_root is not None else None
    imported_bindings = 0
    imported_workspaces = 0
    skipped: list[str] = []

    # A dry run rolls the whole import back: it must SEE what it would do
    # without publishing any of it.
    with transaction(commit=not dry_run) as conn:
        state_dir = legacy_state_dir()
        files = sorted(state_dir.glob("*.json")) if state_dir.is_dir() else []
        for f in files:
            if f.name in ("tokens.json", "remote_repos.json"):
                continue  # caches, not records
            try:
                payload = json.loads(f.read_text())
            except Exception as exc:
                logger.warning("legacy state %s is unreadable (%s); skipping", f, exc)
                skipped.append(str(f))
                continue
            if not isinstance(payload, dict):
                logger.warning("legacy state %s is not an object; skipping", f)
                skipped.append(str(f))
                continue
            sid = str(payload.get("session_id") or f.stem)
            try:
                if _import_binding(conn, sid, payload):
                    imported_bindings += 1
            except Exception as exc:
                logger.warning("could not import legacy state %s (%s); skipping", f, exc)
                skipped.append(str(f))

        if root is not None:
            for rec_path in legacy_workspace_records(root):
                try:
                    record = json.loads(rec_path.read_text())
                except Exception as exc:
                    logger.warning("legacy workspace record %s is unreadable (%s); skipping", rec_path, exc)
                    skipped.append(str(rec_path))
                    continue
                if not isinstance(record, dict):
                    logger.warning("legacy workspace record %s is not an object; skipping", rec_path)
                    skipped.append(str(rec_path))
                    continue
                try:
                    if _import_workspace_record(conn, rec_path, record, root):
                        imported_workspaces += 1
                except Exception as exc:
                    logger.warning("could not import legacy record %s (%s); skipping", rec_path, exc)
                    skipped.append(str(rec_path))

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not dry_run:
        set_kv("migrated_at", stamp)
    return {
        "imported_bindings": imported_bindings,
        "imported_workspaces": imported_workspaces,
        "skipped": skipped,
        "migrated_at": "" if dry_run else stamp,
        "dry_run": dry_run,
    }


def purge_legacy_records(dry_run: bool = False, workspace_root: Path | None = None) -> dict[str, Any]:
    """Delete the in-tree ``.hermes/session.json`` files - after a backup.

    Only files whose workspace the database knows about are touched: an
    unrelated tree that happens to carry a ``.hermes/session.json`` is left
    alone, and a file whose record did NOT make it into the database is kept
    (deleting it would lose data). The tarball backup is written BEFORE
    anything is deleted, so a mistake here is recoverable.

    A file that could NOT be backed up (unreadable, vanished) is NOT
    deleted: it is reported in ``not_backed_up`` and left on disk. The old
    code swallowed the per-file backup error and then deleted the file
    anyway, so an unreadable record ended up purged with an EMPTY tarball -
    the record was simply gone.
    """
    conn = connect()
    root = Path(workspace_root) if workspace_root is not None else None
    candidates: list[Path] = []
    if root is not None:
        known = {str(r[0]) for r in conn.execute("SELECT path FROM workspaces WHERE path <> ''")}
        for rec_path in legacy_workspace_records(root):
            ws_path = str(rec_path.parent.parent)
            if ws_path not in known:
                logger.warning("not purging %s: the database has no record of %s", rec_path, ws_path)
                continue
            candidates.append(rec_path)
    purged: list[str] = []
    not_backed_up: list[str] = []
    backed_up = ""
    backed: set[str] = set()
    if candidates and not dry_run:
        backed_up, backed = _write_backup(candidates)
    for rec_path in candidates:
        if dry_run:
            # A dry run lists and deletes NOTHING.
            continue
        if str(rec_path) not in backed:
            # NOT in the tarball: keeping it is the only safe option.
            logger.warning("not purging %s: it is not in the backup", rec_path)
            not_backed_up.append(str(rec_path))
            continue
        try:
            rec_path.unlink()
            purged.append(str(rec_path))
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("could not delete %s (%s)", rec_path, exc)
    return {
        "purged": purged,
        "would_purge": [str(p) for p in candidates] if dry_run else [],
        "backed_up": backed_up,
        "not_backed_up": not_backed_up,
    }


def _write_backup(files: Iterable[Path]) -> tuple[str, set[str]]:
    """A tarball of the legacy records, written before they are deleted.

    Returns the tarball path and the set of file paths that ACTUALLY made
    it into it. The set is read back from the finished tarball rather than
    assumed: ``tar.add`` can fail half-way through a file, and a file that
    is not in the tarball must not be deleted.

    A failure of the backup as a whole (an unwritable directory, a broken
    tarball) still raises: that is not a per-file problem, and continuing
    would delete records with no backup at all.
    """
    import tarfile

    d = plugin_data_dir()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = d / f"legacy-records-{stamp}.tar.gz"
    with tarfile.open(target, "w:gz") as tar:
        for f in files:
            try:
                tar.add(str(f), arcname=str(f).lstrip("/"))
            except Exception as exc:
                logger.warning("could not add %s to the backup (%s)", f, exc)
    # Read the tarball back: what is in it is what is backed up.
    backed: set[str] = set()
    try:
        with tarfile.open(target, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                backed.add("/" + member.name.lstrip("/"))
    except Exception:
        logger.warning("could not read back the backup %s", target, exc_info=True)
    return str(target), backed
