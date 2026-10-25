"""Shared jj/git layout facts - the geometry of a workspace on disk.

WHY this module exists: three different flows need to answer the SAME
questions about a directory ("is this a jj workspace?", "is this the MAIN
checkout?", "is a live session standing in it?"), and each answer has to be
identical in all of them. The first implementation lived in
``migrate_layout`` (the one-time layout move). The immediate child
reclamation needs the same answers, and importing ``migrate_layout`` from
``store`` would be a CIRCULAR import: ``migrate_layout`` already imports
``store``.

So the geometry lives here, and ``migrate_layout`` re-exports the old
private names so its own tests (and any caller) keep working unchanged.

The facts, once:

* a jj CHILD workspace has ``.jj/repo`` as a FILE holding a RELATIVE path
  to the main workspace's repo dir.
* a jj MAIN (colocated) workspace has ``.jj/repo`` as a DIRECTORY.
* a jj child of a colocated main ALSO has a ``.git`` FILE - it is a real
  git worktree - while a child of a non-colocated main has no ``.git`` at
  all.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_jj_store(workspace: Path) -> Path | None:
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


def is_git_worktree(workspace: Path) -> bool:
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


def is_jj_workspace(workspace: Path) -> bool:
    """True when the directory carries jj workspace metadata."""
    return (workspace / ".jj").exists()


def is_primary_jj_workspace(workspace: Path) -> bool:
    """True when the workspace is the MAIN jj workspace of its store.

    WHY this must be checked before ANY removal: the main workspace's
    ``.jj/repo`` is a DIRECTORY (a child workspace's is a FILE pointing at
    it). ``jj workspace forget``/``remove`` on the main workspace makes jj
    FORGET THE MAIN WORKSPACE - upstream issue #5271 - and jj does it
    without a warning. The repo then has no main checkout at all, and every
    later command in it fails.

    The check is on the POINTER SHAPE, not on a name: a workspace that
    happens to be called "default" is not necessarily the main one, and the
    main one is not necessarily called "default".
    """
    repo_link = workspace / ".jj" / "repo"
    try:
        return repo_link.is_dir()
    except OSError:
        return False


def main_checkout(workspace: Path) -> Path | None:
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
    store_dir = resolve_jj_store(workspace)
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


def pinned_cwd() -> str:
    """The cwd pinned for THIS session, or "" when there is none.

    Narrow on purpose: the plugin's own ``_pinned_workspace`` only counts a
    path that lives under the workspace root. The migration wants the raw
    pinned path - it is asking "is somebody standing in this directory right
    now", and the directory can be under the OLD root.
    """
    try:
        from agent.runtime_cwd import scoped_session_cwd

        return str(scoped_session_cwd() or "")
    except Exception:
        return ""


def live_session_paths(db_module=None) -> set[str]:
    """The paths that a LIVE session is bound to or has pinned as its cwd.

    Two sources, because a session can be using a tree in two ways: the
    binding row (the plugin's own record) and the pinned cwd (where the
    terminal actually runs). A session whose row says the workspace but whose
    cwd was pinned elsewhere is still using the tree.

    A session is LIVE when it has no ``session_ended_at`` - the same rule
    the gc tiers use to decide "in use".
    """
    if db_module is None:
        from . import db as db_module  # local import: avoids a cycle at import time
    out: set[str] = set()
    for row in db_module.all_bindings():
        sid = str(row.get("session_id") or "")
        if not sid:
            continue
        if row.get("session_ended_at"):
            continue  # the session is over: its tree is free to move
        path = str(row.get("path") or "")
        if path:
            out.add(os.path.normpath(path))
    pinned = pinned_cwd()
    if pinned:
        out.add(os.path.normpath(pinned))
    return out


__all__ = [
    "is_git_worktree",
    "is_jj_workspace",
    "is_primary_jj_workspace",
    "live_session_paths",
    "main_checkout",
    "pinned_cwd",
    "resolve_jj_store",
]
