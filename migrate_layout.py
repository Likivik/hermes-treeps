"""The one-time move from the old workspace layout to the dot-dir layout.

The old layout put every session workspace under a configured
``workspace_root`` (``/Git/hermes/workspaces/<repo>/<slug>``). The new one
puts them under ``<repo_dir>/.workspaces/<repo>/<slug>`` - next to the repos
they belong to, in a dot-dir, so a plain listing of the repo dir shows repos
and nothing else.

This module MOVES the existing workspaces. It is deliberately conservative:

* a workspace with uncommitted changes is REPORTED and left alone. Relocating
  somebody's uncommitted work without asking is the one failure mode that
  loses data, and it loses it silently: the files are still there, but in a
  tree nobody is looking at any more.
* a workspace that a live session is bound to (or has pinned as its cwd) is
  REPORTED and left alone. Moving it out from under a running session is how
  a session loses its tree mid-task.
* a move that cannot be VERIFIED afterwards is reported as ``verify_failed``
  and NOT rolled back. A rollback is a second move, and a second move can
  fail in a new place; leaving the workspace where it is, with the problem
  named, is the recoverable state.

The move is a plain ``os.rename`` - same filesystem, no copy, no half-state.
The jj pointer inside a moved workspace (``.jj/repo``) is RELATIVE and its
depth below the base is unchanged (``workspaces/REPO/SLUG`` ->
``.workspaces/REPO/SLUG`` is the same number of levels), so the pointer
still resolves to the same store. That is exactly what the post-move
verification checks, and what a broken pointer must be caught by.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from . import db, store

# The old layout's default root. Kept as a constant because the migration
# must recognise the OLD path even after the setting changed: the setting
# moved on, the directories did not.
LEGACY_WORKSPACE_ROOT = "/Git/hermes/workspaces"


def legacy_roots(settings: dict[str, Any] | None = None) -> list[Path]:
    """Every root the OLD layout could have used.

    Two of them: the shipped default, and whatever the operator configured
    (the deprecated ``workspace_root`` key). Both are candidates - a
    workspace under EITHER has to be moved.

    WHY the configured root comes from ``legacy_workspace_root`` and not from
    ``settings()["workspace_root"]``: the settings dict now returns the
    DERIVED root, so reading it here would look for the old workspaces under
    the NEW directory and find nothing.
    """
    roots: list[Path] = [Path(LEGACY_WORKSPACE_ROOT)]
    if settings is None:
        configured = store._settings.legacy_workspace_root()
    else:
        configured = str(settings.get("workspace_root") or "")
    if configured:
        cand = Path(configured)
        if cand not in roots:
            roots.append(cand)
    return roots


def _under(path: Path, root: Path) -> bool:
    """True when ``path`` is ``root`` or lives under it."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_jj_store(workspace: Path) -> Path | None:
    """The jj store a workspace points at, or None when it has no pointer.

    A child jj workspace carries ``.jj/repo`` as a FILE holding a RELATIVE
    path to the main workspace's repo dir. A colocated main workspace has
    ``.jj/repo`` as a DIRECTORY. Both are resolved here, and the answer is
    what the pre-move and post-move checks compare: the same store means the
    pointer survived the move.
    """
    repo_link = workspace / ".jj" / "repo"
    if not repo_link.exists():
        return None
    try:
        if repo_link.is_dir():
            return repo_link.resolve()
        target = Path(repo_link.read_text().strip())
        return (workspace / ".jj" / target).resolve()
    except Exception:
        return None


def _is_git_worktree(workspace: Path) -> bool:
    """True when the workspace is a git WORKTREE of some checkout.

    WHY the distinction: a plain jj workspace has NO ``.git`` at all - it
    reaches the shared repo through its ``.jj/repo`` pointer, not through a
    git worktree entry. ``git worktree repair`` only makes sense for the
    former, and running it on a jj-only workspace fails with "unable to
    locate repository", which would be reported as a repair failure for
    something that never needed repairing.
    """
    git_marker = workspace / ".git"
    return git_marker.is_file()


def _main_checkout(workspace: Path) -> Path | None:
    """The MAIN checkout of a workspace, or None when it cannot be found.

    Two shapes, because a workspace can be either kind of checkout:

    * a jj workspace: ``.jj/repo`` points at ``<main>/.jj/repo``, so the main
      checkout is two levels up from the resolved repo dir.
    * a git worktree: its ``.git`` is a FILE saying
      ``gitdir: <main>/.git/worktrees/<name>``, so the main checkout is two
      levels up from the resolved gitdir.

    ``git worktree repair`` has to run in the MAIN checkout, with the moved
    workspace as its argument - git repairs the main repo's worktree list,
    and a repair run from the workspace itself would have no list to repair.
    """
    store_dir = _resolve_jj_store(workspace)
    if store_dir is not None:
        # <main>/.jj/repo -> <main>
        main = store_dir.parent.parent
        if main.is_dir():
            return main
    git_marker = workspace / ".git"
    if git_marker.is_file():
        try:
            raw = git_marker.read_text().strip()
        except Exception:
            return None
        if not raw.startswith("gitdir:"):
            return None
        gitdir = Path(raw.split(":", 1)[1].strip())
        if not gitdir.is_absolute():
            gitdir = (workspace / gitdir).resolve()
        else:
            gitdir = gitdir.resolve()
        # <main>/.git/worktrees/<name> -> <main>
        main = gitdir.parent.parent
        return main if main.is_dir() else None
    return None


def _live_session_paths() -> set[str]:
    """The paths that a LIVE session is bound to or has pinned as its cwd.

    Two sources, because a session can be using a tree in two ways: the
    binding row (the plugin's own record) and the pinned cwd (where the
    terminal actually runs). A session whose row says the workspace but whose
    cwd was pinned elsewhere is still using the tree.

    A session is LIVE when it has no ``session_ended_at`` - the same rule
    the gc tiers use to decide "in use".
    """
    out: set[str] = set()
    for row in db.all_bindings():
        sid = str(row.get("session_id") or "")
        if not sid:
            continue
        if row.get("session_ended_at"):
            continue  # the session is over: its tree is free to move
        path = str(row.get("path") or "")
        if path:
            out.add(os.path.normpath(path))
    pinned = _pinned_cwd()
    if pinned:
        out.add(os.path.normpath(pinned))
    return out


def _pinned_cwd() -> str:
    """The cwd pinned for THIS session, or "" when there is none.

    Narrow on purpose: the plugin's own ``_pinned_workspace`` only counts a
    path that lives under the workspace root, and the root has just moved.
    The migration wants the raw pinned path - it is asking "is somebody
    standing in this directory right now", and the directory can be under
    the OLD root.
    """
    try:
        from agent.runtime_cwd import scoped_session_cwd

        return str(scoped_session_cwd() or "")
    except Exception:
        return ""


def _rewrite_records(old_path: str, new_path: str, repo: str, slug: str) -> None:
    """Point the database at the new path.

    TWO rows mention the path: the workspace row itself and the session
    binding rows. The binding rows are what a live session reads to find its
    tree - a moved workspace with an unchanged binding is a session that
    resolves to a path that is gone, which is the "stale" state
    ``resolve_detail`` exists to report.
    """
    record = db.read_workspace(old_path)
    if record is not None:
        record["path"] = new_path
        # write_workspace() upserts on (repo, slug) and carries the session
        # list with it, so the workspace row and its session claims move
        # together in one transaction.
        db.write_workspace(record, new_path)
    # The binding rows are keyed by session id, not by path: they need their
    # own UPDATE.
    db.rebind_workspace_path(repo, slug, new_path)


def _repair_worktrees(workspace: Path, main: Path) -> tuple[bool, str]:
    """Run ``git worktree repair`` for a moved workspace.

    WHY: a colocated jj workspace is also a git worktree of the main
    checkout, and the main checkout's ``.git/worktrees`` entries still name
    the OLD path. Git then thinks the worktree is gone and the next command
    in it fails with "not a git repository" - the workspace looks broken
    right after a successful move.

    A failure here is NOT fatal: the move itself succeeded and the records
    are correct. It is reported so the operator can run the repair by hand.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(main), "worktree", "repair", str(workspace)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "").strip()[:400]
    return True, ""


def migrate_workspaces(dry_run: bool = False) -> dict[str, Any]:
    """Move every workspace under the old layout into the new one.

    Returns a structured report::

        {
            "moved": [ {"path": old, "new_path": new, "repo": r, "slug": s}, ... ],
            "skipped_dirty": [ {"path": old, "reason": str}, ... ],
            "in_use": [ {"path": old, "reason": str}, ... ],
            "verify_failed": [ {"path": old, "new_path": new, "reason": str}, ... ],
            "repair_failed": [ {"path": new, "reason": str}, ... ],
        }

    Never raises: a failure on one workspace is reported and the others are
    still processed. A migration that dies half-way leaves a layout nobody
    can reason about.
    """
    report: dict[str, Any] = {
        "moved": [],
        "skipped_dirty": [],
        "in_use": [],
        "verify_failed": [],
        "repair_failed": [],
    }

    roots = legacy_roots()
    new_base = store.workspaces_base()
    in_use = _live_session_paths()

    for record in db.all_workspaces():
        old_path = str(record.get("path") or "")
        if not old_path:
            continue
        old = Path(old_path)
        if not any(_under(old, root) for root in roots):
            continue  # already in the new layout (or somewhere else entirely)
        if _under(old, new_base):
            continue  # already where it belongs

        repo = str(record.get("repo") or "")
        slug = str(record.get("slug") or old.name)
        if not repo:
            report["skipped_dirty"].append(
                {"path": str(old), "reason": "the record names no repo - not moved"}
            )
            continue

        new = store.ws_path(repo, slug)

        # (a) dirty trees are never moved
        try:
            is_dirty = store.dirty(old)
        except Exception as exc:
            is_dirty = True
            report["skipped_dirty"].append(
                {"path": str(old), "reason": f"could not check the tree: {exc}"}
            )
            continue
        if is_dirty:
            report["skipped_dirty"].append(
                {"path": str(old), "reason": "uncommitted changes - not moved"}
            )
            continue

        # (b) a workspace a live session is using is never moved
        if os.path.normpath(str(old)) in in_use:
            report["in_use"].append(
                {"path": str(old), "reason": "bound to or pinned by a live session"}
            )
            continue

        if dry_run:
            report["moved"].append(
                {"path": str(old), "new_path": str(new), "repo": repo, "slug": slug,
                 "dry_run": True}
            )
            continue

        if new.exists():
            report["skipped_dirty"].append(
                {"path": str(old), "reason": f"{new} already exists - not moved"}
            )
            continue

        # the store the pointer resolves to, BEFORE the move: the post-move
        # check compares against it
        store_before = _resolve_jj_store(old)

        new.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(str(old), str(new))
        except OSError as exc:
            report["skipped_dirty"].append(
                {"path": str(old), "reason": f"the move failed: {type(exc).__name__}: {exc}"}
            )
            continue

        _rewrite_records(str(old), str(new), repo, slug)
        report["moved"].append(
            {"path": str(old), "new_path": str(new), "repo": repo, "slug": slug}
        )

        # (d) repair the git worktree list of the main checkout. Only a real
        # git worktree has an entry to repair - a jj-only workspace has no
        # .git at all and never needed it.
        main = _main_checkout(new)
        if main is not None and _is_git_worktree(new):
            ok, why = _repair_worktrees(new, main)
            if not ok:
                report["repair_failed"].append({"path": str(new), "reason": why})

        # (e) verify: the jj pointer must still resolve to the same store
        store_after = _resolve_jj_store(new)
        if store_before is not None and store_after != store_before:
            report["verify_failed"].append(
                {
                    "path": str(old),
                    "new_path": str(new),
                    "reason": (
                        f"the .jj/repo pointer now resolves to {store_after} "
                        f"instead of {store_before}"
                    ),
                }
            )

    return report


def report_text(report: dict[str, Any]) -> str:
    """The report, as text a human reads in a terminal."""
    lines: list[str] = []
    lines.append(f"moved:          {len(report.get('moved') or [])}")
    for row in report.get("moved") or []:
        lines.append(f"  {row.get('path')} -> {row.get('new_path')}")
    lines.append(f"skipped_dirty:  {len(report.get('skipped_dirty') or [])}")
    for row in report.get("skipped_dirty") or []:
        lines.append(f"  {row.get('path')}: {row.get('reason')}")
    lines.append(f"in_use:         {len(report.get('in_use') or [])}")
    for row in report.get("in_use") or []:
        lines.append(f"  {row.get('path')}: {row.get('reason')}")
    lines.append(f"verify_failed:  {len(report.get('verify_failed') or [])}")
    for row in report.get("verify_failed") or []:
        lines.append(f"  {row.get('new_path')}: {row.get('reason')}")
    lines.append(f"repair_failed:  {len(report.get('repair_failed') or [])}")
    for row in report.get("repair_failed") or []:
        lines.append(f"  {row.get('path')}: {row.get('reason')}")
    return "\n".join(lines)


__all__ = [
    "LEGACY_WORKSPACE_ROOT",
    "legacy_roots",
    "migrate_workspaces",
    "report_text",
]
