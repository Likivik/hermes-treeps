"""Session <-> workspace bookkeeping for the hermes-treeps plugin.

One invariant holds the whole thing together: a session is bound to exactly one
workspace, or to no workspace at all ("just chatting").

The records live in a SQLite database under the host's plugin-data directory
(``<hermes home>/plugin-data/hermes-treeps/data.db``), opened through
``plugins.plugin_storage.plugin_db``. They used to live in two kinds of JSON
file - one per session in the plugin install directory, one per workspace
inside the checkout - and every mutation was a read-modify-write. Eight
concurrent ``bind_session`` calls on one workspace lost four of them, because
the last writer wrote back a list it had read before the others wrote. A row
per binding with one transaction per mutation removes that class of bug.

The install directory is git-pulled and deleted on update, so nothing is
written there: the old JSON files are imported once (see ``db.migrate_legacy``)
and then purged explicitly.

The paths, the VCS and the discovery sources come from the configuration (see
:mod:`settings`), and the jj/git differences sit behind a small adapter (see
:mod:`vcs`).
"""

from __future__ import annotations

import calendar
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from . import db, discovery
from . import settings as _settings
from . import vcs as _vcs
from .errors import WorkspaceError

logger = logging.getLogger(__name__)


def settings() -> dict[str, Any]:
    """The plugin's effective configuration (see :mod:`settings`)."""
    return _settings.settings()


def repo_roots() -> list[str]:
    """The roots that hold repos, from the configuration."""
    return list(settings()["repo_roots"])


def extra_bin_dirs() -> list[str]:
    """Extra PATH entries for `gh`, `jj` and friends."""
    return list(settings()["extra_bin_dirs"])


def clone_root() -> Path:
    """Where a new repo is created/cloned: ``clone_root``, else a repo root."""
    configured = Path(settings()["clone_root"])
    if configured.is_dir():
        return configured
    for r in repo_roots():
        if Path(r).is_dir():
            return Path(r)
    return configured


def ensure_tool_path() -> list[str]:
    """Prepend missing bin dirs to PATH; returns the dirs that were added."""
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    missing = [d for d in extra_bin_dirs() if d not in parts and Path(d).is_dir()]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing + parts)
    return missing


ensure_tool_path()

# Tools that can touch a filesystem, run commands, or execute code on the host.
# Blocked until the session resolves a workspace (see the gate in __init__).
#
# WHY each one is here, because the failure this prevents is a gate BYPASS:
# an ungated tool reaches the host with the session's (possibly someone else's)
# cwd, which is exactly what the gate exists to stop. The list is a MAPPING, not
# a set, so the reason travels with the name and the next addition is obvious:
# if the tool can run code, read or write a tree, or control the machine, it
# belongs here.
GATED_TOOLS: dict[str, str] = {
    "terminal": "runs shell commands in the host cwd",
    "read_file": "reads files anywhere on the host",
    "write_file": "writes files anywhere on the host",
    "patch": "edits files anywhere on the host",
    "search_files": "walks the host filesystem",
    "execute_code": "executes Python on the host",
    # These four were missing and each one is a full gate bypass on this host:
    "browser_exec": "runs arbitrary host Python (default browser backend here)",
    "computer_use": "drives the host desktop: clicks, keys, arbitrary actions",
    "cronjob_manage": "installs no_agent script jobs that run commands on a schedule",
    "process_manage": "signals and kills host processes",
    # These two write host files / spawn tool-running children without touching
    # the repo through a gated tool first:
    "skill_manage": "writes SKILL.md files under HERMES_HOME/skills",
    "delegate_task": "spawns tool-running child sessions",
}


def hermes_home() -> Path:
    """The Hermes home directory, resolved by the one implementation in db.

    WHY one: three copies of this resolver lived here (one importing
    ``utils.paths``, which does not exist in the host, so it always fell
    through the env fallback while the others used the host's profile-aware
    resolver) and they could disagree under a profile switch.
    """
    return db.hermes_home()


def state_dir() -> Path:
    """The LEGACY per-session state directory.

    Nothing is written here any more: the plugin install directory is
    git-pulled and deleted on update, so it is not a place for state. The
    records live in the plugin-data database; this path is the migration
    SOURCE only.
    """
    return db.legacy_state_dir()


def db_path() -> Path:
    """The plugin's database file."""
    return db.db_path()


def root_dir() -> Path:
    """Workspace root: the configured one (``settings.workspace_root``)."""
    return Path(settings()["workspace_root"])


# ── per-session state ─────────────────────────────────────────────────────────


def read_state(session_id: str) -> dict[str, Any] | None:
    """The session record, or None when the session has none."""
    return db.read_binding(session_id)


def write_state(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Replace a session record. One row, one transaction."""
    return db.write_binding(session_id, payload)


def resolve_detail(session_id: str) -> dict[str, Any]:
    """Resolution *with* the reason, so a stale binding is never read as unbound.

    Three states, and the middle one is the one that used to hide: a session
    whose record points at a path that has since moved, been renamed, or been
    trashed. ``resolve()`` returned the record anyway, the gate then refused with
    "no workspace resolved", and the user saw a session that simply would not
    work. Never collapse stale into unbound again.
    """
    st = read_state(session_id)
    if not st:
        return {"state": "unbound", "record": None, "path": None, "reason": "no record for this session"}
    mode = st.get("mode")
    if mode not in {"repo", "chatting", "external"}:
        return {
            "state": "unbound",
            "record": st,
            "path": None,
            "reason": f"record has unknown mode {mode!r}",
        }
    if mode == "chatting":
        return {"state": "bound", "record": st, "path": None, "reason": "chatting"}
    path = str(st.get("path") or "")
    if not path:
        return {"state": "stale", "record": st, "path": "", "reason": "record names no path"}
    p = Path(path).expanduser()
    if not p.exists():
        return {
            "state": "stale",
            "record": st,
            "path": path,
            "reason": f"workspace path is gone: {path}",
        }
    if not p.is_dir():
        return {
            "state": "stale",
            "record": st,
            "path": path,
            "reason": f"workspace path is not a directory: {path}",
        }
    return {"state": "bound", "record": st, "path": path, "reason": ""}


def resolve(session_id: str) -> dict[str, Any] | None:
    """The resolved mode for a session, or None when it has no *usable* binding."""
    detail = resolve_detail(session_id)
    return detail["record"] if detail["state"] == "bound" else None


def forget(session_id: str) -> bool:
    """Drop this session's record so it starts unbound again. True if one existed."""
    return db.delete_binding(session_id)


def mark_chatting(session_id: str, title: str = "") -> dict[str, Any]:
    state = {"mode": "chatting"}
    if title:
        state["title"] = title.strip()
    return write_state(session_id, state)


# ── title mirroring ──────────────────────────────────────────────────────────
#
# A session title is a UI label the user renames whenever he likes, and this
# build exposes no rename hook (VALID_HOOKS has no on_session_rename), so the
# records are reconciled lazily from the state DB instead. Metadata only: the
# workspace PATH is derived from the session id and never follows a rename
# (moving a jj workspace leaves jj's internal workspace name stale).


def find_repo(name: str) -> Path:
    """Locate a repo by name under the configured repo roots."""
    if not _SLUG_RE.match(name.lower()):
        raise WorkspaceError(f"invalid repo name {name!r}")
    roots = repo_roots()
    for base in roots:
        cand = Path(base) / name
        if (cand / ".jj").exists() or (cand / ".git").exists() or (cand / ".git").is_file():
            return cand
    raise WorkspaceError(
        f"no repo {name!r} under {' or '.join(roots)} - existing: "
        + ", ".join(sorted(p.name for b in roots if Path(b).is_dir() for p in Path(b).iterdir()))[:300]
    )


def is_jj(repo: Path) -> bool:
    return _vcs.detect(repo) == _vcs.JJS


def run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=180
    )
    if proc.returncode != 0:
        raise WorkspaceError(f"{' '.join(cmd)} failed: {(proc.stderr or proc.stdout).strip()[:400]}")
    return proc.stdout.strip()


def workspace_path(repo_name: str, slug: str) -> Path:
    return root_dir() / repo_name / slug


def sanitize_slug(slug: str) -> str:
    """The canonical form of a slug: lowercase, safe characters only.

    Callers must use THIS, not their raw input: create() sanitises the slug
    internally, so a caller that binds `workspace_path(repo, raw_slug)` binds
    a path that does not exist.
    """
    import re as _re

    out = _re.sub(r"[^a-z0-9._-]+", "-", str(slug or "").strip().lower()).strip("-._")
    return out


def create(repo_name: str, slug: str, purpose: str = "") -> dict[str, Any]:
    """Create the workspace and its record together: neither exists without the other."""
    slug = sanitize_slug(slug)
    if not _SLUG_RE.match(slug):
        raise WorkspaceError(f"invalid slug {slug!r}: lowercase letters, digits, '.', '_', '-'; <=64 chars")

    repo = find_repo(repo_name)
    dest = workspace_path(repo_name, slug)
    if dest.exists():
        raise WorkspaceError(f"{dest} already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)

    kind = _vcs.create_workspace(repo, dest, slug, vcs=settings()["vcs"], runner=run)
    # The workspace directory and its record are created TOGETHER: neither
    # exists without the other. jj/git create the directory themselves, but
    # a runner that does not (or a checkout that was removed underneath us)
    # must not leave a record pointing at a path that is not there.
    dest.mkdir(parents=True, exist_ok=True)

    record = {
        "repo": repo_name,
        "slug": slug,
        "path": str(dest),
        "purpose": purpose,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vcs": kind,
        "sessions": [],
        "prs": [],
        "last_synced": None,
    }
    write_record(dest, record)
    return record


def read_record(ws: Path) -> dict[str, Any] | None:
    """The workspace record for a path, or None when there is none."""
    return db.read_workspace(str(ws))


def write_record(ws: Path, record: dict[str, Any]) -> None:
    """Replace the workspace record for a path. One transaction."""
    db.write_workspace(record, path=str(ws))


def bind_session(ws: Path, session_id: str) -> dict[str, Any]:
    """Bind a session to a workspace.

    The old implementation read the workspace record, appended the session id
    to a list and wrote the whole file back: eight concurrent calls on one
    workspace lost four of them. This is one INSERT in one transaction, so
    every call lands.
    """
    record = read_record(ws)
    if record is None:
        raise WorkspaceError(f"{ws} has no workspace record - not a workspace this plugin made")
    db.add_workspace_session(str(record.get("repo") or ""), str(record.get("slug") or ""), session_id)
    return read_record(ws) or record


def detach(ws: Path, session_id: str) -> None:
    """Remove a session from a workspace's binding (the 1:1 rule: rebinding
    detaches the old tree instead of letting two claim the session)."""
    record = read_record(ws)
    if record is None:
        return
    db.remove_workspace_session(
        str(record.get("repo") or ""), str(record.get("slug") or ""), session_id
    )


def create_repo(name: str) -> Path:
    """Create a brand-new repo under the clone root, ready for ws_new."""
    if not _SLUG_RE.match(name.lower()):
        raise WorkspaceError(f"invalid repo name {name!r}")
    base = clone_root()
    dest = base / name
    if dest.exists():
        raise WorkspaceError(f"{dest} already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _vcs.init_colocated(dest)
    return dest


def bind_external(path: str, session_id: str, title: str = "") -> dict[str, Any]:
    """Bind a session to an existing directory OUTSIDE the workspace root.

    No checkout is created: the session works in that tree directly. Any absolute
    directory is accepted - refusing non-repos here pushed people into inventing
    workarounds - but a non-repo binding is flagged in the record so the prompt
    can say so. A missing path is reported as *missing*, never as "not a
    directory": that message cost a whole session once (a tree had been moved to
    /Storage/trash-workspaces-* and the tool refused the new path with the wrong
    word).
    """
    if not path or not str(path).strip():
        raise WorkspaceError("path must be a non-empty absolute path")
    raw = Path(str(path)).expanduser()
    if not raw.is_absolute():
        raise WorkspaceError(f"path must be absolute, got {path!r}")
    p = raw.resolve()
    if not p.exists():
        raise WorkspaceError(f"{p} does not exist (the path is gone, not merely invalid)")
    if not p.is_dir():
        raise WorkspaceError(f"{p} exists but is not a directory")
    is_repo = is_jj(p) or (p / ".git").exists() or (p / ".git").is_file()
    state: dict[str, Any] = {"mode": "external", "repo": p.name, "slug": p.name, "path": str(p)}
    if not is_repo:
        state["not_a_repo"] = True
    if title:
        state["title"] = title.strip()
    write_state(session_id, state)
    return state


def rename(old_slug: str, new_slug: str) -> str:
    """Move a workspace directory and fix every record that mentions it.

    Safe for jj: workspaces are keyed by working-copy commit, not path (verified
    - `jj st` works after a plain mv). The internal jj name stays the old slug;
    our records are authoritative.
    """
    # The OLD slug is validated too: rename() is the one call that MOVES a
    # directory, so a traversal in old_slug would move (not just read) a tree
    # outside the root. find_workspace() already refuses such a slug; this is
    # the explicit, local refusal so the failure names the bad argument.
    # It is checked BEFORE the lookup, so the error names the bad argument
    # instead of "no workspace <traversal>".
    if not _SLUG_RE.match(str(old_slug or "")):
        raise WorkspaceError(f"invalid slug {old_slug!r} (old slug): lowercase letters, digits, '.', '_', '-'; <=64 chars")
    if not _SLUG_RE.match(str(new_slug or "")):
        raise WorkspaceError(f"invalid new slug {new_slug!r}")
    ws = find_workspace(old_slug)
    dest = ws.parent / new_slug
    if dest.exists():
        raise WorkspaceError(f"{dest} already exists")
    old_path = str(ws)
    record = read_record(ws) or {}
    repo = str(record.get("repo") or ws.parent.name)
    ws.rename(dest)
    # The workspace row is keyed by (repo, slug), so a rename is a key change:
    # the row moves with the directory, and the session rows move with it (a
    # stale row would keep the old workspace claiming the session, and
    # retire() would refuse it forever).
    db.rename_workspace(repo, str(old_slug), str(new_slug), str(dest))
    record["slug"] = new_slug
    record["path"] = str(dest)
    for sid in record.get("sessions") or []:
        st = read_state(str(sid))
        if st and st.get("path") == old_path:
            st["path"] = str(dest)
            st["slug"] = new_slug
            write_state(str(sid), st)
    return f"renamed {old_slug} -> {dest}"


# Slugs are the only user-supplied path component: this pattern is what keeps
# a traversal from ever reaching the filesystem (see rename(), find_workspace()).
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def find_workspace(slug: str) -> Path:
    """Locate a workspace by slug, refusing anything that could escape the root.

    The workspaces live in the database, so the lookup is a query - but the
    slug is still a NAME, never a path, and the result is still checked
    against the root. Two defences, because either alone is bypassable:
      1. the slug must match _SLUG_RE - no separators, no dots-only names;
      2. every hit is resolved and must be inside the resolved root, which also
         catches symlinks pointing out of the tree.
    """
    slug = str(slug or "")
    if not _SLUG_RE.match(slug):
        raise WorkspaceError(f"invalid slug {slug!r}: lowercase letters, digits, '.', '_', '-'; <=64 chars")
    root = root_dir()
    try:
        root_resolved = root.resolve()
    except OSError:
        root_resolved = root
    hits: list[Path] = []
    for cand in db.workspace_paths_for_slug(slug):
        p = Path(cand)
        if not p.exists():
            continue
        try:
            resolved = p.resolve()
        except OSError:
            continue
        # A hit outside the root is never a workspace of ours: refuse it rather
        # than hand back a path that rename()/retire() would then destroy.
        if not resolved.is_relative_to(root_resolved):
            logger.warning("find_workspace: refusing out-of-root hit %s", p)
            continue
        if p not in hits:
            hits.append(p)
    if not hits:
        raise WorkspaceError(f"no workspace {slug!r} under {root}")
    if len(hits) > 1:
        raise WorkspaceError(f"{slug!r} is ambiguous: " + ", ".join(str(h) for h in hits))
    return hits[0]


# ── listing / status ──────────────────────────────────────────────────────────


def _is_workspace_dir(p: Path) -> bool:
    """A workspace is a checkout: it has its own VCS metadata or a record."""
    return (p / ".jj").exists() or (p / ".git").exists() or db.read_workspace(str(p)) is not None


def list_workspaces() -> list[dict[str, Any]]:
    """Every workspace under the root, plus orphans in both directions.

    Two layouts are supported because the root predates this plugin: the flat
    one (``<root>/<repo>-<slug>``, e.g. ``spectacle-serenity-oom``) and the
    nested one (``<root>/<repo>/<slug>``). A checkout is recognised by its own
    VCS metadata or its database record, so a repo directory is not mistaken
    for a workspace.
    """
    root = root_dir()
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for top in sorted(p for p in root.iterdir() if p.is_dir()):
        if _is_workspace_dir(top):
            # flat: legacy workspace, repo name is the leading segment
            record = read_record(top)
            repo = (record or {}).get("repo") or top.name.split("-")[0]
            out.append(
                {"repo": repo, "slug": top.name, "path": str(top), "record": record, "orphan": record is None}
            )
            continue
        for ws in sorted(p for p in top.iterdir() if p.is_dir()):
            record = read_record(ws)
            out.append(
                {
                    "repo": top.name,
                    "slug": ws.name,
                    "path": str(ws),
                    "record": record,
                    "orphan": record is None,
                }
            )
    return out


# ── VCS inspection (through the adapter) ─────────────────────────────────────


def _workspace_branches(workspace: Path) -> set[str]:
    """Branch/bookmark names owned by THIS workspace - the unit that open PRs
    must be matched against. A repo-wide PR list would block retiring a fresh
    workspace over PRs it never made (real case: spectacle #102/#93/#70/#67)."""
    return _vcs.branches(workspace)


def open_prs(workspace: Path) -> tuple[list[dict[str, Any]], bool]:
    """(open PRs, gh_worked).

    The second element exists because the two cases are opposites for retire:
    "gh says zero" is a clean bill, "gh couldn't run" is UNKNOWN and must not
    silently pass the guard (Hermes worktree_gc fails safe the same way:
    offline degrades to keep).
    """
    return _vcs.open_prs(workspace)


def unpushed_commits(workspace: Path) -> bool | None:
    """True = commits not on the remote. None = cannot tell (offline, no
    upstream) - retire must treat UNKNOWN as unsafe, like worktree_gc does.

    Caveat inherited from the same idea in worktree_gc: content that was pushed
    and then rebased re-appears unpushed, because the comparison is by commit
    id, not content.
    """
    return _vcs.unpushed(workspace)


def dirty(workspace: Path) -> bool:
    """True when the workspace has changes that retiring would lose.

    The plugin's own bookkeeping is ignored: it is not work, and counting it
    would make every workspace permanently unretireable.
    """
    return _vcs.dirty(workspace)


# ── retire ────────────────────────────────────────────────────────────────────


def _source_repo(record: dict[str, Any], ws: Path) -> Path | None:
    """The repo a workspace is registered in (NOT ``ws.parent``)."""
    name = str(record.get("repo") or "")
    if name:
        try:
            return find_repo(name)
        except WorkspaceError:
            return None
    try:
        return find_repo(ws.parent.name)
    except WorkspaceError:
        return None


def retire(slug: str, force: bool = False, ultraforce: bool = False) -> str:
    """ultraforce: bypass even the hard blockers (open PRs, dirty tree).

    It still archives the record first - nothing about the *bookkeeping* is
    skipped, only the safety refusals. For "I know exactly what I am deleting".
    """
    ws = find_workspace(slug)
    record = read_record(ws) or {}
    if not ultraforce:
        return _retire_checked(ws, slug, record, force)

    prs, _gh_ok = open_prs(ws)
    live = [p for p in prs if p.get("state") == "OPEN"]
    warnings = []
    if live:
        warnings.append("open PRs " + ", ".join(f"#{p['number']}" for p in live))
    if dirty(ws):
        warnings.append("uncommitted changes")
    if warnings:
        logger.warning("ULTRAFORCE retiring %s despite: %s", slug, "; ".join(warnings))
    return _retire_checked(ws, slug, record, force=True, skip_hard=True)


def _retire_checked(
    ws: Path, slug: str, record: dict[str, Any], force: bool, skip_hard: bool = False
) -> str:
    prs, gh_ok = open_prs(ws)
    mine = _workspace_branches(ws)
    live = [p for p in prs if p.get("state") == "OPEN" and p.get("headRefName") in mine]

    # Hard blockers: known loss. Bypassable ONLY by ultraforce, which has just
    # logged what it is deleting.
    if live and not skip_hard:
        raise WorkspaceError(
            "refusing to retire: open PRs " + ", ".join(f"#{p['number']}" for p in live)
        )
    if dirty(ws) and not skip_hard:
        raise WorkspaceError("refusing to retire: uncommitted changes in the workspace")
    if not skip_hard:
        live_sids = [
            sid for sid in (record.get("sessions") or [])
            if not (read_state(str(sid)) or {}).get("session_ended_at")
        ]
        if live_sids:
            raise WorkspaceError(
                "refusing to retire: bound session(s) still live: "
                + ", ".join(str(x) for x in live_sids)
            )

    # UNKNOWNs: fail safe like worktree_gc (offline -> keep). Retirable, but
    # only with an explicit --force acknowledging we could not verify.
    unknown = []
    if not gh_ok:
        unknown.append("could not check open PRs (gh missing, unauthenticated or slow)")
    pushed = unpushed_commits(ws)
    if pushed is True:
        raise WorkspaceError(
            "refusing to retire: commits not pushed to the remote "
            "(push or repeat with --force to retire anyway)"
        )
    if pushed is None:
        unknown.append("could not verify pushed state (offline or no upstream)")
    if unknown and not force:
        raise WorkspaceError(
            "refusing to retire: " + "; ".join(unknown)
            + ". Repeat with --force to retire anyway."
        )
    archive = hermes_home() / "archive" / "workspaces"
    archive.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    target = archive / f"{record.get('repo', ws.parent.name)}-{slug}-{stamp}.json"
    target.write_text(json.dumps({"record": record, "retired_at": stamp}, indent=2, sort_keys=True))
    vcs_kind = record.get("vcs", "jj")
    # WHY the repo is resolved from the record and not from `parent`:
    # `parent` is <root>/<repo>, which is NOT a jj/git root, so `jj workspace
    # forget` run there always failed - and its returncode was ignored, so the
    # workspace was left registered in the repo while the directory was deleted.
    # Resolve the SOURCE repo (where the workspace is registered) and run the
    # cleanup there, and CHECK the returncode: a failed forget leaves a stale
    # workspace entry behind, which must not be silent.
    repo = _source_repo(record, ws)
    if repo is not None:
        try:
            _vcs.forget_workspace(repo, slug, ws, vcs=vcs_kind)
        except WorkspaceError as exc:
            raise WorkspaceError(f"refusing to retire {slug}: {exc}") from exc
    # No `rm -rf`: the path was validated by find_workspace() (slug shape +
    # resolved-inside-root), and shutil.rmtree on that exact path deletes the
    # tree without a shell in between. `rm -rf` on an unvalidated path is how
    # a traversal bug becomes a data-loss bug.
    shutil.rmtree(ws, ignore_errors=True)
    db.delete_workspace(str(record.get("repo") or ""), str(record.get("slug") or slug))
    return f"retired {slug} -> {target}"


# ── sessions.json helpers (host state, read-only) ─────────────────────────────


def session_entry(session_id: str) -> dict[str, Any] | None:
    """The sessions.json row for a session id (platform, chat_type, key)."""
    p = hermes_home() / "sessions" / "sessions.json"
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    for key, entry in data.items():
        if isinstance(entry, dict) and entry.get("session_id") == session_id:
            return {**entry, "session_key": entry.get("session_key") or key}
    return None


def chat_coords(session_key: str, chat_type: str) -> tuple[int | None, int | None]:
    """(chat_id, thread_id) from a durable key like agent:main:telegram:group:-100…:29039."""
    parts = session_key.split(":")
    try:
        if chat_type == "group" and len(parts) >= 2:
            return int(parts[-2]), int(parts[-1])
        return int(parts[-1]), None
    except Exception:
        return None, None


def session_for_chat(chat_id: Any, thread_id: Any = None) -> dict[str, Any] | None:
    """The sessions.json row bound to a chat (and topic), if any.

    Used as a state signal: a /new reset replaces this row's session_id, so
    watching it is a real event, not a guessed delay.
    """
    p = hermes_home() / "sessions" / "sessions.json"
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    suffix = f":{chat_id}:{thread_id}" if thread_id else f":{chat_id}"
    for key, entry in data.items():
        if isinstance(entry, dict) and key.endswith(suffix):
            return {**entry, "session_key": key}
    return None


# ── gc ────────────────────────────────────────────────────────────────────────


# A child workspace is throwaway by design: its commit is the handoff artifact,
# so it is rarely pushed, and `gc` would otherwise keep every delegation's
# checkout forever ("unpushed commits", tier2). After this many days it is
# reclaimable once its session ended and its tree is clean.
CHILD_TTL_DAYS = 7

# Written into a workspace record's `purpose` by childspace when a delegated
# child gets its own tree. One definition, imported by childspace, so gc and
# the creator cannot disagree about what a child workspace is.
CHILD_PURPOSE_PREFIX = "child:"


def _workspace_age_days(record: dict[str, Any], path: Path) -> float:
    """Age in days from the record's created_at, else from the tree's mtime."""
    raw = str(record.get("created_at") or "")
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            # created_at is written with time.gmtime() - UTC. time.mktime would
            # read it as local time and make every workspace look hours
            # younger than it is (UTC+3 here: the 7-day TTL fires 3h late).
            created = calendar.timegm(time.strptime(raw[:19], fmt))
            return max(0.0, (time.time() - created) / 86400)
        except Exception:
            continue
    try:
        return max(0.0, (time.time() - path.stat().st_mtime) / 86400)
    except Exception:
        return 0.0


def gc_tiers() -> dict[str, list[dict[str, Any]]]:
    """Classify workspaces for reclamation. Dry-run by design: this only labels.

    tier1: session(s) ended AND clean AND pushed AND no open PRs  -> safe to retire
    tier2: ended but something could not be verified (or orphans) -> check first
    tier3: a bound session is still live                          -> in use
    """
    tiers: dict[str, list[dict[str, Any]]] = {"tier1": [], "tier2": [], "tier3": []}
    for item in list_workspaces():
        record = item["record"] or {}
        path = Path(item["path"])
        sids = [str(x) for x in (record.get("sessions") or [])]
        if item["orphan"] or not sids:
            tiers["tier2"].append({**item, "why": "orphan: no bound session record"})
            continue
        states = [read_state(sid) or {} for sid in sids]
        if not all(st.get("session_ended_at") for st in states):
            tiers["tier3"].append({**item, "why": "session still live"})
            continue
        # Child workspaces: unpublished by design, so the "pushed" test that
        # protects a session's own work does not apply. Age does.
        if str(record.get("purpose") or "").startswith(CHILD_PURPOSE_PREFIX):
            if dirty(path):
                tiers["tier2"].append({**item, "why": "child workspace with uncommitted changes"})
                continue
            age_days = _workspace_age_days(record, path)
            if unpushed_commits(path):
                # The agreed rule: a workspace whose commit is neither applied
                # nor pushed is never deleted - a child's handoff commit is
                # exactly that until the parent lands it. tier2 keeps it
                # visible instead of silently skipping at retire time.
                tiers["tier2"].append(
                    {**item, "why": "child workspace, handoff commit not applied/pushed"}
                )
                continue
            if age_days >= CHILD_TTL_DAYS:
                tiers["tier1"].append(
                    {**item, "why": f"child workspace, clean, {age_days:.0f}d old"}
                )
            else:
                tiers["tier2"].append(
                    {**item, "why": f"child workspace, clean, {age_days:.0f}d old (ttl {CHILD_TTL_DAYS}d)"}
                )
            continue
        prs, gh_ok = open_prs(path)
        mine = _workspace_branches(path)
        live = [p for p in prs if p.get("state") == "OPEN" and p.get("headRefName") in mine]
        pushed = unpushed_commits(path)
        if live:
            tiers["tier2"].append(
                {**item, "why": "open PRs " + ", ".join(f"#{p['number']}" for p in live)}
            )
        elif dirty(path):
            tiers["tier2"].append({**item, "why": "uncommitted changes"})
        elif pushed is not False or not gh_ok:
            tiers["tier2"].append(
                {**item, "why": "could not verify pushed state" if pushed is None else "unpushed commits"}
            )
        else:
            tiers["tier1"].append({**item, "why": "ended, clean, pushed, no open PRs"})
    return tiers


def gc_apply() -> list[str]:
    """Retire exactly the tier1 set, through the same guarded retire()."""
    done: list[str] = []
    for item in gc_tiers()["tier1"]:
        try:
            done.append(retire(item["slug"]))
        except WorkspaceError as e:
            logger.warning("gc skipped %s: %s", item["slug"], e)
    return done


# ── one-shot picker tokens ────────────────────────────────────────────────────

_TOKEN_LOCK = threading.RLock()


def new_token(repos: list[dict[str, Any]]) -> str:
    """Mint a one-shot token that carries the repo list for a picker keyboard."""
    with _TOKEN_LOCK:
        tokens = db.get_kv_json("tokens", {}) or {}
        token = secrets.token_hex(4)  # 8 chars: fits Telegram's 64-byte callback budget
        tokens[token] = {"repos": repos, "at": time.time()}
        # drop tokens older than an hour so the table cannot grow forever
        cutoff = time.time() - 3600
        tokens = {k: v for k, v in tokens.items() if not isinstance(v, dict) or v.get("at", 0) > cutoff}
        db.set_kv_json("tokens", tokens)
    return token


def take_token(token: str) -> dict[str, Any] | None:
    with _TOKEN_LOCK:
        tokens = db.get_kv_json("tokens", {}) or {}
        entry = tokens.pop(token, None)
        if entry is not None:
            db.set_kv_json("tokens", tokens)
    return entry if isinstance(entry, dict) else None


# ── remote repo listing (GitHub + Forgejo + local) ───────────────────────────

# Mutated in place rather than rebound: a module-level name reassigned from a
# function reads to static analysis as never used.
_REMOTE_CACHE: dict[str, Any] = {}

# Why the last catalogue lookup failed, so a caller can say "the catalogue was
# unavailable" instead of "there are no remote repos" - the two mean very
# different things to the user.
_REMOTE_ERROR: str = ""


def remote_repos(limit_per_source: int = 5, cache_s: int = 600) -> list[dict[str, Any]]:
    """The user's repos, most recently pushed first.

    Remote-first picker source. Falls back to the stale cache, then to local
    candidate_repos(), so a dead network never produces an empty keyboard.

    NEVER raises: this shells out to `gh` and to the Forgejo HTTP API, and a
    catalogue lookup is a diagnostic, not a gate. A failure must degrade to
    "no remote repos known" (recorded in _REMOTE_ERROR), never to a traceback
    in the middle of a ws_new / ws_doctor answer.
    """
    global _REMOTE_ERROR
    try:
        repos, err = _remote_repos(limit_per_source=limit_per_source, cache_s=cache_s)
    except Exception as exc:  # a catalogue lookup must never break the caller
        logger.debug("remote catalogue lookup failed", exc_info=True)
        _REMOTE_ERROR = f"remote catalogue unavailable: {type(exc).__name__}: {exc}"
        return []
    _REMOTE_ERROR = err
    return repos


def remote_catalogue(limit_per_source: int = 5) -> tuple[list[dict[str, Any]], str]:
    """(entries, error) from the remote catalogue. Never raises.

    The single read-only entry point for "does this repo exist on GitHub or
    Forgejo?". The error string is empty when the lookup worked - including
    when it worked and found nothing, which is a different answer from "the
    lookup itself failed".
    """
    global _REMOTE_ERROR
    _REMOTE_ERROR = ""  # a fresh lookup must not inherit the last one's error
    try:
        entries = remote_repos(limit_per_source=limit_per_source)
    except Exception as exc:  # defensive: a caller must never see a traceback
        logger.debug("remote catalogue lookup failed", exc_info=True)
        return [], f"remote catalogue unavailable: {type(exc).__name__}: {exc}"
    if not isinstance(entries, list):
        return [], "remote catalogue unavailable: unexpected result"
    clean = [e for e in entries if isinstance(e, dict)]
    return clean, _REMOTE_ERROR


def _remote_repos(limit_per_source: int = 5, cache_s: int = 600) -> tuple[list[dict[str, Any]], str]:
    """The catalogue lookup itself; returns (repos, error)."""
    now = time.time()
    if _REMOTE_CACHE and now - _REMOTE_CACHE.get("at", 0) < cache_s:
        return _REMOTE_CACHE["repos"], ""

    entries, errors = discovery.discover(limit_per_source=limit_per_source)
    if entries:
        _REMOTE_CACHE.update({"at": now, "repos": entries})
        db.set_kv_json("remote_repos", {"at": now, "repos": entries})
        return entries, ""

    # stale cache, then local disk
    stale = db.get_kv_json("remote_repos", {}) or {}
    if stale.get("repos"):
        return stale["repos"], ""
    err = ("remote catalogue unavailable: " + "; ".join(errors)) if errors else ""
    return (
        [r for r in discovery.local_repos(limit=6)],
        err,
    )


def candidate_repos(limit: int = 8) -> list[str]:
    """Repo names under the configured repo roots."""
    return discovery.local_names(limit=limit)


# ── cloning ───────────────────────────────────────────────────────────────────


def clone_target(name: str) -> Path:
    """Where a clone of `name` would land, without cloning it.

    Read-only: the gate and ws_new both want to TELL the user the
    destination before anything is written to their disk.
    """
    return clone_root() / name


def clone_repo(name: str, url: str) -> Path:
    """Clone `name` from `url` into the clone root. The explicit, approved step.

    Separate from ensure_local() on purpose: ensure_local() clones as a
    side effect of "give me a checkout", while this one is only ever reached
    after the user approved a clone (ws_new with create_repo=true).
    """
    if not url:
        raise WorkspaceError(f"repo {name!r} has no clone URL")
    dest = clone_target(name)
    if dest.exists():
        raise WorkspaceError(f"{dest} already exists but is not a repo")
    base = dest.parent
    base.mkdir(parents=True, exist_ok=True)
    run(["jj", "git", "clone", "--colocate", url, str(dest)], cwd=base)
    return dest


def ensure_local(name: str, url: str) -> Path:
    """Return a local checkout for a remote repo, cloning into the clone root
    when it is not on disk yet."""
    try:
        return find_repo(name)
    except WorkspaceError:
        pass
    if not url:
        raise WorkspaceError(f"repo {name!r} is not local and has no clone URL")
    base = clone_root()
    if not base.is_dir():
        raise WorkspaceError(f"no writable repo root under {repo_roots()}")
    dest = base / name
    if dest.exists():
        raise WorkspaceError(f"{dest} already exists but is not a repo")
    run(["jj", "git", "clone", "--colocate", url, str(dest)], cwd=base)
    return dest


# ── migration ─────────────────────────────────────────────────────────────────


def migrate_legacy(workspace_root: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Import the legacy JSON records into the database. Idempotent.

    ``dry_run`` is forwarded, not swallowed: this wrapper dropped it once and
    the CLI then imported 38 rows under --dry-run while the underlying
    function was already fixed. The wrapper is the seam the CLI actually
    calls, so it carries every argument.
    """
    return db.migrate_legacy(
        workspace_root if workspace_root is not None else root_dir(), dry_run=dry_run
    )


def purge_legacy_records(dry_run: bool = False, workspace_root: Path | None = None) -> dict[str, Any]:
    """Delete the in-tree ``.hermes/session.json`` files, after a backup."""
    return db.purge_legacy_records(
        dry_run=dry_run, workspace_root=workspace_root if workspace_root is not None else root_dir()
    )
