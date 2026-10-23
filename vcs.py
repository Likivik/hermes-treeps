"""A small VCS adapter: jj and git behind one interface.

The plugin used to hard-wire jj in some places and git in others, and the
choice was implicit (whatever the repo happened to contain). The adapter
makes it explicit and configurable:

* ``auto`` (default) - jj when the repo has a ``.jj`` directory, git otherwise.
* ``jj`` / ``git``    - forced. A forced choice that contradicts the repo on
  disk is REFUSED with a clear message rather than half-applied.

Every jj behaviour here is the one the plugin had before: ``jj workspace add``
for a new workspace, ``jj -R <repo> workspace forget`` for cleanup,
``remote_bookmarks()..@`` for "not pushed", and a colocated ``jj git init``.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from .errors import WorkspaceError

logger = logging.getLogger(__name__)

JJS = "jj"
GIT = "git"
VCS_CHOICES = ("auto", JJS, GIT)


def detect(repo: Path) -> str:
    """Which VCS a repo uses: jj when it has a ``.jj`` directory, else git."""
    return JJS if (repo / ".jj").exists() else GIT


def resolve(repo: Path, configured: str = "auto") -> str:
    """The VCS to use for ``repo`` under the configured policy.

    ``auto`` follows the repo. An explicit ``jj``/``git`` that contradicts the
    repo on disk is refused: silently doing something else is how a workspace
    ends up registered in one VCS and cleaned up in another.
    """
    configured = (configured or "auto").lower()
    if configured not in VCS_CHOICES:
        raise WorkspaceError(
            f"invalid vcs {configured!r}: expected one of {', '.join(VCS_CHOICES)}"
        )
    actual = detect(repo)
    if configured == "auto":
        return actual
    if configured == JJS and actual != JJS:
        raise WorkspaceError(
            f"vcs=jj was requested but {repo} is a git repo (no .jj directory)"
        )
    if configured == GIT and actual == JJS:
        raise WorkspaceError(
            f"vcs=git was requested but {repo} is a jj repo (.jj present) - "
            f"a git worktree cannot be added to it"
        )
    return configured


# ── running commands ─────────────────────────────────────────────────────────


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 180) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise WorkspaceError(f"{' '.join(cmd)} failed: {(proc.stderr or proc.stdout).strip()[:400]}")
    return proc.stdout.strip()


def _run_raw(cmd: list[str], cwd: Path | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout
        )
    except Exception:
        return subprocess.CompletedProcess(cmd, 1, "", "command failed")


# ── workspace lifecycle ──────────────────────────────────────────────────────


def create_workspace(
    repo: Path,
    dest: Path,
    slug: str,
    vcs: str = "auto",
    runner: Any = None,
) -> str:
    """Create a workspace of ``dest`` for ``repo``. Returns the VCS used.

    ``runner`` is the command runner to use (default: this module's). The
    caller passes its own so the plugin's own ``store.run`` stays the single
    subprocess seam - that is what the tests patch.
    """
    kind = resolve(repo, vcs)
    do = runner or run
    if kind == JJS:
        do(["jj", "workspace", "add", str(dest), "--name", slug], cwd=repo)
    else:
        do(["git", "worktree", "add", str(dest), "-b", slug], cwd=repo)
    return kind


def forget_workspace(repo: Path, slug: str, path: Path, vcs: str = "auto") -> None:
    """Unregister the workspace from its repo.

    Raises on failure: a stale registration in the repo is worse than a
    refusal, because the directory is gone but the repo still points at it.
    """
    kind = vcs if vcs in (JJS, GIT) else detect(repo)
    if kind == JJS:
        proc = _run_raw(["jj", "-R", str(repo), "workspace", "forget", slug])
        if proc.returncode != 0:
            raise WorkspaceError(
                f"jj workspace forget failed in {repo}: " + ((proc.stderr or proc.stdout or "").strip()[:300])
            )
    else:
        proc = _run_raw(["git", "-C", str(repo), "worktree", "remove", str(path)])
        if proc.returncode != 0:
            raise WorkspaceError(
                f"git worktree remove failed in {repo}: " + ((proc.stderr or proc.stdout or "").strip()[:300])
            )


def init_colocated(path: Path) -> Path:
    """Create a new colocated jj repo at ``path`` (the plugin's clone style)."""
    run(["jj", "git", "init", "--colocate", str(path)], cwd=path.parent)
    return path


# ── inspection ───────────────────────────────────────────────────────────────


def dirty(ws: Path) -> bool:
    """True when the workspace has changes that retiring would lose.

    The plugin's own bookkeeping is ignored: it is not work, and counting it
    would make every workspace permanently unretireable.
    """
    try:
        if detect(ws) == JJS:
            out = _run_raw(["jj", "diff", "--summary"], cwd=ws, timeout=60)
            lines = [ln for ln in (out.stdout or "").splitlines() if ".hermes/" not in ln]
            return bool(lines)
        out = _run_raw(["git", "status", "--porcelain"], cwd=ws, timeout=60)
        lines = [ln for ln in (out.stdout or "").splitlines() if ".hermes/" not in ln]
        return bool(lines)
    except Exception:
        return True  # assume dirty: refuse rather than lose work


def unpushed(ws: Path) -> bool | None:
    """True = commits not on the remote. None = cannot tell.

    The jj revset is ``remote_bookmarks()..@`` - not ``..bookmarks()``: the
    old revset only saw BOOKMARKED commits, so an agent's own ``jj commit``
    (no bookmark) looked "pushed" and retire() deleted a tree whose work
    existed only in the op log. ``@`` is the working-copy commit.
    """
    try:
        if detect(ws) == JJS:
            out = _run_raw(
                ["jj", "log", "--no-graph", "-r", "remote_bookmarks()..@",
                 "-T", 'if(empty, "", commit_id ++ "\\n")'],
                cwd=ws, timeout=30,
            )
            if out.returncode == 0:
                return bool(out.stdout.strip())
            # Fallback for a jj too old for the `empty` template keyword.
            diff = _run_raw(["jj", "diff", "--summary", "-r", "remote_bookmarks()..@"], cwd=ws, timeout=30)
            if diff.returncode != 0:
                return None
            return bool(diff.stdout.strip())
        up = _run_raw(["git", "rev-parse", "--abbrev-ref", "@{u}"], cwd=ws, timeout=10)
        if up.returncode != 0:
            return None  # no upstream configured
        out = _run_raw(["git", "log", "--oneline", "@{u}.."], cwd=ws, timeout=30)
        return bool(out.stdout.strip())
    except Exception:
        return None


def branches(ws: Path) -> set[str]:
    """Branch/bookmark names owned by THIS workspace.

    A repo-wide list would block retiring a fresh workspace over PRs it never
    made.
    """
    names: set[str] = set()
    try:
        if detect(ws) == JJS:
            out = _run_raw(["jj", "bookmark", "list"], cwd=ws, timeout=30)
            for line in (out.stdout or "").splitlines():
                name = line.strip().split(":")[0].strip()
                if name and not name.startswith("git"):
                    names.add(name.split()[-1] if "@" in name else name)
        else:
            out = _run_raw(
                ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"], cwd=ws, timeout=10
            )
            names |= {ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()}
    except Exception:
        return set()
    return names


def remote_url(ws: Path) -> str:
    """The origin URL without relying on git (jj workspaces have no .git dir,
    which makes plain ``gh pr list`` fail with 'not a git repository')."""
    try:
        if detect(ws) == JJS:
            out = _run_raw(["jj", "git", "remote", "list"], cwd=ws, timeout=30)
            for line in (out.stdout or "").splitlines():
                if line.startswith("origin "):
                    return line.split()[1]
        else:
            out = _run_raw(["git", "remote", "get-url", "origin"], cwd=ws, timeout=10)
            return (out.stdout or "").strip()
    except Exception:
        return ""
    return ""


def open_prs(ws: Path) -> tuple[list[dict[str, Any]], bool]:
    """(open PRs, gh_worked).

    ``gh_ok`` is False when the lookup itself failed: "gh says zero" is a
    clean bill, "gh could not run" is UNKNOWN and must not silently pass the
    retire guard.
    """
    import json

    try:
        repo_ref = remote_url(ws)
        cmd = ["gh", "pr", "list", "--json", "number,title,url,state,headRefName", "--limit", "50"]
        if repo_ref:
            cmd += ["-R", repo_ref]  # gh cannot self-locate a jj workspace
        raw = _run_raw(cmd, cwd=ws, timeout=60)
        if raw.returncode != 0:
            return [], False
        return json.loads(raw.stdout or "[]"), True
    except Exception:
        return [], False
