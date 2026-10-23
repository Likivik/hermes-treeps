"""Session <-> jj workspace bookkeeping for the hermes-treeps plugin.

One invariant holds the whole thing together: a session is bound to exactly one
workspace, or to no workspace at all ("just chatting"). Records live in the
workspace itself (``.hermes/session.json``) so a workspace directory can always
answer "which session am I, what did I ship" without a central database that can
drift out of sync with the filesystem.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "/Git/hermes/workspaces"
REPO_ROOTS = ("/Git", "/Git/hermes", "/Git/likivik", "/Git/repos")

# The gateway service runs with a narrowed PATH — /run/current-system/sw/bin is
# not in it — so `gh` and `jj` are invisible to every subprocess unless the dirs
# are added here. Done once at import so all call sites stay unchanged. Without
# this, `gh repo list` raises FileNotFoundError inside remote_repos() and the
# Telegram workspace picker never sends its keyboard.
_EXTRA_BIN_DIRS = ("/run/current-system/sw/bin", "/usr/local/bin", "/usr/bin")


def ensure_tool_path() -> list[str]:
    """Prepend missing bin dirs to PATH; returns the dirs that were added."""
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    missing = [d for d in _EXTRA_BIN_DIRS if d not in parts and Path(d).is_dir()]
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
}


def is_gated(tool_name: str) -> bool:
    """True when the tool must be blocked until the session resolves a workspace."""
    return tool_name in GATED_TOOLS

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class WorkspaceError(RuntimeError):
    """A refusal with a message meant for the user, not a traceback."""


def hermes_home() -> Path:
    try:
        from utils.paths import get_hermes_home  # type: ignore

        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def state_dir() -> Path:
    d = hermes_home() / "plugins" / "hermes-treeps" / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def root_dir() -> Path:
    """Workspace root: config override, else the fleet convention."""
    override = os.environ.get("HERMES_WS_ROOT")
    if override:
        return Path(override)
    try:
        from hermes_cli.config import load_config_readonly  # type: ignore

        cfg = load_config_readonly() or {}
        entry = ((cfg.get("plugins") or {}).get("entries") or {}).get("hermes-treeps") or {}
        if entry.get("root"):
            return Path(str(entry["root"]))
    except Exception:
        pass
    return Path(DEFAULT_ROOT)


# ── per-session state ─────────────────────────────────────────────────────────


def _state_path(session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", session_id or "unknown")
    return state_dir() / f"{safe}.json"


def read_state(session_id: str) -> dict[str, Any] | None:
    p = _state_path(session_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def write_state(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload.setdefault("session_id", session_id)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p = _state_path(session_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(p)
    return payload


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
    p = _state_path(session_id)
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False


def mark_chatting(session_id: str, title: str = "") -> dict[str, Any]:
    state = {"mode": "chatting"}
    if title:
        state["title"] = title.strip()
    return write_state(session_id, state)


# ── title mirroring ───────────────────────────────────────────────────────────
#
# A session title is a UI label the user renames whenever they like, and this
# build exposes no rename hook (VALID_HOOKS has no on_session_rename), so the
# records are reconciled lazily from the state DB instead. Metadata only: the
# workspace PATH is derived from the session id and never follows a rename
# (moving a jj workspace leaves jj's internal workspace name stale).


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def session_title(session_id: str) -> str:
    """The session's current title from ``state.db``, or "" when unavailable."""
    if not session_id:
        return ""
    db = _hermes_home() / "state.db"
    if not db.exists():
        return ""
    try:
        import sqlite3

        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
        try:
            row = con.execute("SELECT title FROM sessions WHERE id=?", (session_id,)).fetchone()
        finally:
            con.close()
        return str(row[0] or "") if row else ""
    except Exception:
        logger.debug("session title lookup failed for %s", session_id, exc_info=True)
        return ""


def sync_title(session_id: str) -> dict[str, Any]:
    """Mirror a renamed session's title into the session state and workspace record."""
    title = session_title(session_id)
    if not title:
        return {"changed": False, "title": ""}
    st = read_state(session_id)
    if not st:
        return {"changed": False, "title": title}
    changed = st.get("title") != title
    if changed:
        st["title"] = title
        write_state(session_id, st)
    path = str(st.get("path") or "")
    if path:
        ws = Path(path)
        if ws.is_dir():
            rec = read_record(ws) or {}
            titles = dict(rec.get("titles") or {})
            if titles.get(session_id) != title:
                titles[session_id] = title
                rec["titles"] = titles
                write_record(ws, rec)
                changed = True
    return {"changed": changed, "title": title}


# ── repo + workspace discovery ────────────────────────────────────────────────


def find_repo(name: str) -> Path:
    """Locate a repo by name under /Git or /Git/repos."""
    if not _SLUG_RE.match(name.lower()):
        raise WorkspaceError(f"invalid repo name {name!r}")
    for base in REPO_ROOTS:
        cand = Path(base) / name
        if (cand / ".jj").exists() or (cand / ".git").exists() or (cand / ".git").is_file():
            return cand
    raise WorkspaceError(
        f"no repo {name!r} under {' or '.join(REPO_ROOTS)} - existing: "
        + ", ".join(sorted(p.name for b in REPO_ROOTS if Path(b).is_dir() for p in Path(b).iterdir()))[:300]
    )


def is_jj(repo: Path) -> bool:
    return (repo / ".jj").exists()


def run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=180)
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

    if is_jj(repo):
        run(["jj", "workspace", "add", str(dest), "--name", slug], cwd=repo)
    else:
        run(["git", "worktree", "add", str(dest), "-b", slug], cwd=repo)

    record = {
        "repo": repo_name,
        "slug": slug,
        "path": str(dest),
        "purpose": purpose,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vcs": "jj" if is_jj(repo) else "git",
        "sessions": [],
        "prs": [],
        "last_synced": None,
    }
    write_record(dest, record)
    return record


def record_path(ws: Path) -> Path:
    return ws / ".hermes" / "session.json"


def read_record(ws: Path) -> dict[str, Any] | None:
    p = record_path(ws)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def write_record(ws: Path, record: dict[str, Any]) -> None:
    p = record_path(ws)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, sort_keys=True))


def bind_session(ws: Path, session_id: str) -> dict[str, Any]:
    record = read_record(ws)
    if record is None:
        raise WorkspaceError(f"{ws} has no .hermes/session.json - not a workspace this plugin made")
    sessions = record.setdefault("sessions", [])
    if session_id not in sessions:
        sessions.append(session_id)
    write_record(ws, record)
    return record


def detach(ws: Path, session_id: str) -> None:
    """Remove a session from a workspace's binding (the 1:1 rule: rebinding
    detaches the old tree instead of letting two claim the session)."""
    record = read_record(ws)
    if record is None:
        return
    sessions = [x for x in (record.get("sessions") or []) if str(x) != str(session_id)]
    record["sessions"] = sessions
    write_record(ws, record)


def create_repo(name: str) -> Path:
    """Create a brand-new repo under the first repo root, ready for ws_new."""
    if not _SLUG_RE.match(name.lower()):
        raise WorkspaceError(f"invalid repo name {name!r}")
    base = Path("/Git/hermes") if Path("/Git/hermes").is_dir() else Path(REPO_ROOTS[0])
    dest = base / name
    if dest.exists():
        raise WorkspaceError(f"{dest} already exists")
    run(["jj", "git", "init", "--colocate", str(dest)], cwd=base)
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
    ws = find_workspace(old_slug)
    # The OLD slug is validated too: rename() is the one call that MOVES a
    # directory, so a traversal in old_slug would move (not just read) a tree
    # outside the root. find_workspace() already refuses such a slug; this is
    # the explicit, local refusal so the failure names the bad argument.
    if not _SLUG_RE.match(str(old_slug or "")):
        raise WorkspaceError(f"invalid old slug {old_slug!r}")
    if not _SLUG_RE.match(new_slug):
        raise WorkspaceError(f"invalid new slug {new_slug!r}")
    dest = ws.parent / new_slug
    if dest.exists():
        raise WorkspaceError(f"{dest} already exists")
    old_path = str(ws)
    ws.rename(dest)
    record = read_record(dest) or {}
    record["slug"] = new_slug
    record["path"] = str(dest)
    write_record(dest, record)
    for sid in record.get("sessions") or []:
        st = read_state(str(sid))
        if st and st.get("path") == old_path:
            st["path"] = str(dest)
            st["slug"] = new_slug
            write_state(str(sid), st)
    return f"renamed {old_slug} -> {dest}"


def find_workspace(slug: str) -> Path:
    """Locate a workspace by slug, refusing anything that could escape the root.

    The glob `*/{slug}` is a path-traversal primitive: a slug like
    `../../outside/victim` globs to a path OUTSIDE the root, and rename()/retire()
    then move or `rm -rf` that path. This is the single choke point every
    caller goes through (rename, retire, gc), so the check lives here.

    Two defences, because either alone is bypassable:
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
    hits = []
    for p in root.glob(f"*/{slug}"):
        if not (p / ".hermes" / "session.json").exists():
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
        hits.append(p)
    if not hits:
        raise WorkspaceError(f"no workspace {slug!r} under {root}")
    if len(hits) > 1:
        raise WorkspaceError(f"{slug!r} is ambiguous: " + ", ".join(str(h) for h in hits))
    return hits[0]


# ── listing / status ──────────────────────────────────────────────────────────


def _is_workspace_dir(p: Path) -> bool:
    """A workspace is a checkout: it has its own VCS metadata or a session record."""
    return (p / ".hermes" / "session.json").exists() or (p / ".jj").exists() or (p / ".git").exists()


def list_workspaces() -> list[dict[str, Any]]:
    """Every workspace under the root, plus orphans in both directions.

    Two layouts are supported because the root predates this plugin: the flat one
    (``<root>/<repo>-<slug>``, e.g. ``spectacle-serenity-oom``) and the nested one
    (``<root>/<repo>/<slug>``). A checkout is recognised by its own VCS metadata
    or session record, so a repo directory is not mistaken for a workspace.
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



def _remote_url(workspace: Path) -> str:
    """The origin URL without relying on git (jj workspaces have no .git dir,
    which makes plain `gh pr list` fail with 'not a git repository')."""
    vcs = "jj" if (workspace / ".jj").exists() else "git"
    try:
        if vcs == "jj":
            out = subprocess.run(["jj", "git", "remote", "list"], cwd=str(workspace), capture_output=True, text=True, timeout=30)
            for line in (out.stdout or "").splitlines():
                if line.startswith("origin "):
                    return line.split()[1]
        else:
            out = subprocess.run(["git", "remote", "get-url", "origin"], cwd=str(workspace), capture_output=True, text=True, timeout=10)
            return (out.stdout or "").strip()
    except Exception:
        return ""
    return ""


def _workspace_branches(workspace: Path) -> set[str]:
    """Branch/bookmark names owned by THIS workspace - the unit that open PRs
    must be matched against. A repo-wide PR list would block retiring a fresh
    workspace over PRs it never made (real case: spectacle #102/#93/#70/#67)."""
    names: set[str] = set()
    try:
        if (workspace / ".jj").exists():
            out = subprocess.run(["jj", "bookmark", "list"], cwd=str(workspace), capture_output=True, text=True, timeout=30)
            for line in (out.stdout or "").splitlines():
                name = line.strip().split(":")[0].strip()
                if name and not name.startswith("git"):
                    names.add(name.split()[-1] if "@" in name else name)
        else:
            out = subprocess.run(["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"], cwd=str(workspace), capture_output=True, text=True, timeout=10)
            names |= {ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()}
    except Exception:
        return set()
    return names


def open_prs(workspace: Path) -> tuple[list[dict[str, Any]], bool]:
    """(open PRs, gh_worked).

    The second element exists because the two cases are opposites for retire:
    "gh says zero" is a clean bill, "gh couldn't run" is UNKNOWN and must not
    silently pass the guard (Hermes worktree_gc fails safe the same way:
    offline degrades to keep).
    """
    try:
        repo_ref = _remote_url(workspace)
        cmd = ["gh", "pr", "list", "--json", "number,title,url,state,headRefName", "--limit", "50"]
        if repo_ref:
            cmd += ["-R", repo_ref]  # gh cannot self-locate a jj workspace
        raw = subprocess.run(cmd, cwd=str(workspace), capture_output=True, text=True, timeout=60)
        if raw.returncode != 0:
            return [], False
        return json.loads(raw.stdout or "[]"), True
    except Exception:
        return [], False


def unpushed_commits(workspace: Path) -> bool | None:
    """True = commits not on the remote. None = cannot tell (offline, no
    upstream) - retire must treat UNKNOWN as unsafe, like worktree_gc does.

    Caveat inherited from the same idea in worktree_gc: content that was pushed
    and then rebased re-appears unpushed, because the comparison is by commit
    id, not content.
    """
    try:
        if (workspace / ".jj").exists():
            # WHY `remote_bookmarks()..@` and not `..bookmarks()`: the old
            # revset only saw BOOKMARKED commits, so an agent's own `jj commit`
            # (no bookmark) looked "pushed" and retire() deleted a tree whose
            # work existed only in the op log. `@` is the working-copy commit,
            # so the range now covers every commit not on a remote bookmark,
            # bookmarked or not.
            #
            # Empty commits are filtered out by the template: the working-copy
            # commit is always "not on a remote bookmark", so counting it would
            # make every workspace look unpushed forever. A commit with real
            # content (the agent's own work) is never empty.
            #
            # `--ignore-working-copy` is deliberately NOT used: it would hide
            # the working copy commit, which is exactly where an agent's
            # uncommitted work sits. dirty() covers the working tree separately.
            out = subprocess.run(
                ["jj", "log", "--no-graph",
                 "-r", "remote_bookmarks()..@",
                 "-T", 'if(empty, "", commit_id ++ "\\n")'],
                cwd=str(workspace), capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0:
                return bool(out.stdout.strip())
            # Fallback for a jj too old for the `empty` template keyword:
            # compare trees instead of commits.
            diff = subprocess.run(
                ["jj", "diff", "--summary", "-r", "remote_bookmarks()..@"],
                cwd=str(workspace), capture_output=True, text=True, timeout=30,
            )
            if diff.returncode != 0:
                return None
            return bool(diff.stdout.strip())
        up = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "@{u}"],
            cwd=str(workspace), capture_output=True, text=True, timeout=10,
        )
        if up.returncode != 0:
            return None  # no upstream configured
        out = subprocess.run(
            ["git", "log", "--oneline", "@{u}.."],
            cwd=str(workspace), capture_output=True, text=True, timeout=30,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def dirty(workspace: Path) -> bool:
    """True when the workspace has changes that retiring would lose.

    The plugin's own ``.hermes/`` record is ignored: it is bookkeeping, not work,
    and counting it would make every workspace permanently unretireable.
    """
    try:
        if (workspace / ".jj").exists():
            # --summary + the same .hermes/ filter the git branch has: the
            # plugin's own session.json bookkeeping must not make every jj
            # workspace permanently unretireable.
            out = subprocess.run(["jj", "diff", "--summary"], cwd=str(workspace), capture_output=True, text=True, timeout=60)
            lines = [ln for ln in (out.stdout or "").splitlines() if ".hermes/" not in ln]
            return bool(lines)
        out = subprocess.run(["git", "status", "--porcelain"], cwd=str(workspace), capture_output=True, text=True, timeout=60)
        lines = [ln for ln in (out.stdout or "").splitlines() if ".hermes/" not in ln]
        return bool(lines)
    except Exception:
        return True  # assume dirty: refuse rather than lose work



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
    vcs = record.get("vcs", "jj")
    # WHY the repo is resolved from the record and not from `parent`:
    # `parent` is <root>/<repo>, which is NOT a jj/git root, so `jj workspace
    # forget` run there always failed - and its returncode was ignored, so the
    # workspace was left registered in the repo while the directory was deleted.
    # Resolve the SOURCE repo (where the workspace is registered) and run the
    # cleanup there, and CHECK the returncode: a failed forget leaves a stale
    # workspace entry behind, which must not be silent.
    repo = _source_repo(record, ws)
    if vcs == "jj":
        if repo is not None:
            proc = subprocess.run(
                ["jj", "-R", str(repo), "workspace", "forget", slug],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                raise WorkspaceError(
                    f"refusing to retire {slug}: jj workspace forget failed in {repo}: "
                    + ((proc.stderr or proc.stdout or "").strip()[:300])
                )
    else:
        if repo is not None:
            proc = subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", str(ws)],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                raise WorkspaceError(
                    f"refusing to retire {slug}: git worktree remove failed in {repo}: "
                    + ((proc.stderr or proc.stdout or "").strip()[:300])
                )
    # No `rm -rf`: the path was validated by find_workspace() (slug shape +
    # resolved-inside-root), and shutil.rmtree on that exact path deletes the
    # tree without a shell in between. `rm -rf` on an unvalidated path is how
    # a traversal bug becomes data loss.
    if ws.exists():
        shutil.rmtree(str(ws), ignore_errors=False)
    return f"retired {slug} -> {target}"


def _source_repo(record: dict[str, Any], ws: Path) -> Path | None:
    """The repo a workspace belongs to, resolved from its record.

    The record carries the repo NAME; find_repo() maps it back to a path on
    disk. Returns None when the repo is gone (nothing to clean up then).
    """
    name = str(record.get("repo") or "")
    if not name:
        name = ws.parent.name
    try:
        return find_repo(name)
    except WorkspaceError:
        logger.warning("source repo %r not found for workspace %s", name, ws)
        return None

def candidate_repos(limit: int = 8) -> list[str]:
    """Repo names available under the repo roots, for the opening prompt."""
    names: list[str] = []
    for base in REPO_ROOTS:
        b = Path(base)
        if not b.is_dir():
            continue
        for p in sorted(b.iterdir()):
            if p.is_dir() and ((p / ".jj").exists() or (p / ".git").exists() or (p / ".git").is_file()):
                if p.name not in names and not p.name.startswith("."):
                    names.append(p.name)
    return names[:limit]

def session_key_for(session_id: str) -> str | None:
    """Map an ephemeral session id to its durable gateway session key.

    Gateway injection needs the durable key (``agent:main:telegram:dm:<id>``),
    and on_session_start only hands us the ephemeral id, so this is the bridge.
    """
    p = hermes_home() / "sessions" / "sessions.json"
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    for key, entry in data.items():
        if isinstance(entry, dict) and entry.get("session_id") == session_id:
            return str(entry.get("session_key") or key)
    return None

# ── one-shot tokens for inline-button choices ────────────────────────────────


def _tokens_path() -> Path:
    return state_dir() / "tokens.json"


_TOKEN_LOCK = __import__("threading").Lock()


def _load_tokens() -> dict[str, Any]:
    try:
        return json.loads(_tokens_path().read_text())
    except Exception:
        return {}


def _save_tokens(tokens: dict[str, Any]) -> None:
    """Atomic: temp file + rename, so a crash never leaves a torn tokens.json."""
    p = _tokens_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(tokens, indent=2, sort_keys=True))
    os.replace(tmp, p)


def new_token(repos: list[str]) -> str:
    """Mint a one-shot token bound to a repo list - NOT to a session.

    The session is resolved when the button is TAPPED (the tap carries the live
    chat, whose session row we can look up); binding at send time produced
    tokens with session_id="" that created workspaces no session could resolve.
    """
    import secrets

    with _TOKEN_LOCK:
        tokens = _load_tokens()
        token = secrets.token_hex(4)  # 8 chars: fits Telegram's 64-byte callback budget
        tokens[token] = {"repos": repos, "at": time.time()}
        # drop tokens older than an hour so the file cannot grow forever
        cutoff = time.time() - 3600
        tokens = {k: v for k, v in tokens.items() if not isinstance(v, dict) or v.get("at", 0) > cutoff}
        _save_tokens(tokens)
    return token


def take_token(token: str) -> dict[str, Any] | None:
    with _TOKEN_LOCK:
        tokens = _load_tokens()
        entry = tokens.pop(token, None)
        if entry is not None:
            _save_tokens(tokens)
    return entry if isinstance(entry, dict) else None


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


# ── remote repo listing (GitHub + Forgejo) ───────────────────────────────────

_REMOTE_CACHE: dict[str, Any] | None = None


def remote_repos(limit_per_source: int = 5, cache_s: int = 600) -> list[dict[str, Any]]:
    """The user's repos from GitHub and Forgejo, most recently pushed first.

    Remote-first picker source. Falls back to the stale cache, then to local
    candidate_repos(), so a dead network never produces an empty keyboard.
    """
    import urllib.request

    global _REMOTE_CACHE
    now = time.time()
    if _REMOTE_CACHE and now - _REMOTE_CACHE.get("at", 0) < cache_s:
        return _REMOTE_CACHE["repos"]

    repos: list[dict[str, Any]] = []
    gh = shutil.which("gh")
    try:
        if not gh:
            raise FileNotFoundError("gh is not on PATH")
        raw = subprocess.run(
            [gh, "repo", "list", "--limit", str(limit_per_source),
             "--json", "nameWithOwner,url,pushedAt"],
            capture_output=True, text=True, timeout=30,
        )
        if raw.returncode != 0:
            raise RuntimeError((raw.stderr or "").strip()[:200] or f"gh exited {raw.returncode}")
        for r in json.loads(raw.stdout or "[]"):
            repos.append({
                "name": r["nameWithOwner"].split("/")[-1],
                "owner": r["nameWithOwner"].split("/")[0],
                "url": r["url"],
                "source": "github",
                "pushed_at": r.get("pushedAt", ""),
            })
    except Exception:
        logger.debug("gh repo list failed", exc_info=True)

    try:
        import os

        url = os.environ["FORGEJO_URL"].rstrip("/") + "/api/v1/repos/search"
        req = urllib.request.Request(
            url + f"?limit={limit_per_source}&sort=updated",
            headers={"Authorization": "token " + os.environ["FORGEJO_TOKEN"]},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        for r in data.get("data", []):
            repos.append({
                "name": r["name"],
                "owner": (r.get("owner") or {}).get("login", ""),
                "url": r.get("clone_url", ""),
                "source": "forgejo",
                "pushed_at": r.get("updated_at", ""),
            })
    except Exception:
        logger.debug("forgejo repo search failed", exc_info=True)

    # dedupe by name (github wins on ties), newest first
    seen: dict[str, dict[str, Any]] = {}
    for r in sorted(repos, key=lambda x: x.get("pushed_at", ""), reverse=True):
        seen.setdefault(r["name"], r)
    repos = sorted(seen.values(), key=lambda x: x.get("pushed_at", ""), reverse=True)[: limit_per_source * 2]

    if repos:
        _REMOTE_CACHE = {"at": now, "repos": repos}
        try:
            p = state_dir() / "remote_repos.json"
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps({"at": now, "repos": repos}, indent=2))
            os.replace(tmp, p)
        except Exception:
            pass
        return repos

    # stale cache, then local disk
    try:
        stale = json.loads((state_dir() / "remote_repos.json").read_text())
        if stale.get("repos"):
            return stale["repos"]
    except Exception:
        pass
    return [{"name": n, "url": "", "source": "local", "pushed_at": ""} for n in candidate_repos(limit=6)]


def ensure_local(name: str, url: str) -> Path:
    """Return a local checkout for a remote repo, cloning into the first
    writable repo root (/Git/hermes) when it is not on disk yet."""
    try:
        return find_repo(name)
    except WorkspaceError:
        pass
    if not url:
        raise WorkspaceError(f"repo {name!r} is not local and has no clone URL")
    # Prefer the agent's own root: /Git itself is likivik's, and clones landing
    # there would violate the likivik/hermes ownership split.
    for b in ("/Git/hermes", "/Git"):
        base = Path(b)
        if base.is_dir() and os.access(base, os.W_OK):
            break
    else:
        raise WorkspaceError(f"no writable repo root under {REPO_ROOTS}")
    dest = base / name
    if dest.exists():
        raise WorkspaceError(f"{dest} already exists but is not a repo")
    run(["jj", "git", "clone", "--colocate", url, str(dest)], cwd=base)
    return dest
