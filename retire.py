"""Workspace retirement: work-preserving tree removal and child reclamation."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from . import (
    db,
    jjlayout,
    store,  # late-bound: tests monkeypatch store.<helper>, so bind at call time
)
from . import vcs as _vcs
from .errors import WorkspaceError

logger = logging.getLogger(__name__)

# ── retire ────────────────────────────────────────────────────────────────────


def _source_repo(record: dict[str, Any], ws: Path) -> Path | None:
    """The repo a workspace is registered in (NOT ``ws.parent``)."""
    name = str(record.get("repo") or "")
    if name:
        try:
            return store.find_repo(name)
        except WorkspaceError:
            return None
    try:
        return store.find_repo(ws.parent.name)
    except WorkspaceError:
        return None


def retire(
    slug: str, force: bool = False, ultraforce: bool = False, allow_unpushed: bool = False
) -> str:
    """ultraforce: bypass even the hard blockers (open PRs, dirty tree).

    It still archives the record first - nothing about the *bookkeeping* is
    skipped, only the safety refusals. For "I know exactly what I am deleting".

    ``allow_unpushed``: skip the pushed-state refusal. Only for CHILD
    workspaces (gc_apply passes it): their commits survive retirement in the
    shared store, so "not pushed" is not "lost".
    """
    ws = store.find_workspace(slug)
    record = store.read_record(ws) or {}
    if not ultraforce:
        return _retire_checked(ws, slug, record, force, allow_unpushed=allow_unpushed)

    prs, _gh_ok = store.open_prs(ws)
    live = [p for p in prs if p.get("state") == "OPEN"]
    warnings = []
    if live:
        warnings.append("open PRs " + ", ".join(f"#{p['number']}" for p in live))
    if store.dirty(ws):
        warnings.append("uncommitted changes")
    if warnings:
        logger.warning("ULTRAFORCE retiring %s despite: %s", slug, "; ".join(warnings))
    return _retire_checked(ws, slug, record, force=True, skip_hard=True)


def _retire_checked(
    ws: Path, slug: str, record: dict[str, Any], force: bool, skip_hard: bool = False,
    allow_unpushed: bool = False,
) -> str:
    prs, gh_ok = store.open_prs(ws)
    mine = store._workspace_branches(ws)
    live = [p for p in prs if p.get("state") == "OPEN" and p.get("headRefName") in mine]

    # Hard blockers: known loss. Bypassable ONLY by ultraforce, which has just
    # logged what it is deleting.
    if live and not skip_hard:
        raise WorkspaceError(
            "refusing to retire: open PRs " + ", ".join(f"#{p['number']}" for p in live)
        )
    if store.dirty(ws) and not skip_hard:
        raise WorkspaceError("refusing to retire: uncommitted changes in the workspace")
    if not skip_hard:
        live_sids = [
            sid for sid in (record.get("sessions") or [])
            if not (store.read_state(str(sid)) or {}).get("session_ended_at")
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
    pushed = store.unpushed_commits(ws)
    if pushed is True and not allow_unpushed:
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
    archive = store.hermes_home() / "archive" / "workspaces"
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
    # No `rm -rf`: the path was validated by store.find_workspace() (slug shape +
    # resolved-inside-root), and shutil.rmtree on that exact path deletes the
    # tree without a shell in between. `rm -rf` on an unvalidated path is how
    # a traversal bug becomes a data-loss bug.
    shutil.rmtree(ws, ignore_errors=True)
    db.delete_workspace(str(record.get("repo") or ""), str(record.get("slug") or slug))
    return f"retired {slug} -> {target}"



# ── child reclamation ─────────────────────────────────────────────────────────
#
# WHY this section exists: a delegated child's workspace is throwaway, and
# once the child is DONE the tree is dead weight on disk. But "reclaim the
# tree" must not mean "throw the work away".
#
# `jj workspace remove` (jj >= 0.46) does both in one step: it snapshots
# the working copy - EVEN THE DIRTY ONE - into a commit in the SHARED
# store, then removes the workspace and its directory. The commit survives
# in the repo; the tree does not. So the work is preserved as a commit, and
# the plugin's job is to RECORD WHICH COMMIT, because jj never auto-gcs and
# an unrecorded commit is a commit nobody can find.
#
# `jj workspace forget` (the older command) does NOT snapshot: it only
# unregisters the workspace and leaves the directory on disk. For a CLEAN
# child there is nothing to preserve, so forget is enough - but the
# directory still has to be deleted, and the commit id recorded, or the
# next `jj workspace add` of the same name collides with a forgotten
# workspace.
#
# Every child-reclamation flow goes through ONE function:
# :func:`reclaim_child_workspace`. The age-based sweep and the immediate
# CLI path MUST NOT each grow their own removal code - that is how a
# workspace ends up removed from disk but still registered in the repo (a
# stale jj workspace entry), or removed from the repo but still in the
# database (a workspace row pointing at a path that is gone).

# The reason strings a reclamation can be REFUSED with. They are stable
# because the CLI prints them and the tests assert on them.
RECLAIM_REFUSED_NOT_CHILD = "not_a_child_workspace"
RECLAIM_REFUSED_PRIMARY = "primary_workspace"
RECLAIM_REFUSED_IN_USE = "live_session_pinned"
RECLAIM_SKIPPED_GIT_WORKTREE = "skipped_git_worktree"
RECLAIM_REFUSED_MISSING = "workspace_missing"


def _jj_snapshot_commit_id(workspace: Path) -> str:
    """The commit id of the workspace's working-copy commit, or "".

    WHY this is read BEFORE the removal: ``jj workspace remove`` snapshots
    the working copy into the working-copy commit and then removes the
    workspace. The commit that holds the child's work IS that commit, and
    its id does not change - jj rewrites the working-copy commit in place
    when it snapshots, so the id read here is the id that holds the work
    afterwards.

    WHY "before" is safe: the workspace still exists at this point, so
    ``jj log -r @`` in it resolves to the working-copy commit and not to
    the parent.
    """
    try:
        proc = subprocess.run(
            ["jj", "log", "--no-graph", "-r", "@", "-T", "commit_id"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        logger.warning("could not read the workspace commit id: %s", exc)
        return ""
    if proc.returncode != 0:
        logger.warning(
            "jj log in %s failed: %s", workspace, (proc.stderr or proc.stdout or "").strip()[:300]
        )
        return ""
    return (proc.stdout or "").strip().splitlines()[0].strip() if (proc.stdout or "").strip() else ""


def _jj_workspace_remove(repo: Path, slug: str) -> tuple[bool, str]:
    """``jj workspace remove`` the workspace, by NAME.

    Returns ``(ok, reason)``.

    WHY the NAME and not the path: ``jj workspace remove`` takes workspace
    NAMES. Given a path it prints "Warning: No such workspace: <path>" -
    and EXITS 0. A caller that only checks the return code then believes
    the removal worked, while the workspace is still registered in the
    store: the directory is gone, but jj still has a workspace entry for a
    path that does not exist, and the next ``jj workspace add`` of that
    name fails. The plugin creates its workspaces with
    ``jj workspace add <path> --name <slug>``, so the NAME IS the slug.

    WHY ``-R <repo>``: the command has to run against the store that
    HOLDS the workspace. Running it inside the workspace works too (jj
    finds the store through the pointer), but running it against the main
    workspace is the shape that does not delete the caller's cwd out from
    under it - "Removing the current workspace works on Unix, but Windows
    won't remove a directory that a process is running in" (jj docs). The
    plugin runs on Linux, but the main workspace is the stable choice
    anyway.
    """
    try:
        proc = subprocess.run(
            ["jj", "-R", str(repo), "workspace", "remove", slug],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "").strip()[:400]
    # WHY the output is checked even though jj exited 0: "No such
    # workspace" is a WARNING on jj's side and it still exits 0. Without
    # this check a failed removal looks like a success, and the workspace
    # stays registered in the store forever.
    noise = (proc.stderr or "") + (proc.stdout or "")
    if "No such workspace" in noise:
        return False, noise.strip()[:400]
    return True, ""


def _jj_workspace_registered(repo: Path, slug: str) -> bool:
    """True when the workspace is still registered in the jj store.

    WHY this exists: jj's "No such workspace" is a WARNING that still exits
    0, so a return code alone cannot tell a removal that happened from one
    that did not. The registration is the thing that has to be gone: a
    stale entry means the next ``jj workspace add`` of the same name fails
    with "workspace already exists".
    """
    try:
        proc = subprocess.run(
            ["jj", "-R", str(repo), "workspace", "list", "--template", "name ++ \"\\n\""],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    for line in (proc.stdout or "").splitlines():
        if line.strip() == slug:
            return True
    return False


def _jj_workspace_forget(repo: Path, slug: str) -> tuple[bool, str]:
    """``jj workspace forget`` the workspace, by NAME.

    WHY forget and not remove for a CLEAN workspace: forget does NOT
    snapshot, and a clean workspace has nothing to snapshot. It is the
    cheaper, older, and better-understood command - and on a jj older than
    0.46 it is the ONLY one that exists.

    WHY the NAME: forget takes the workspace name, and the plugin's slug
    IS that name (the workspace was created as ``<repo>/<slug>``).
    """
    try:
        proc = subprocess.run(
            ["jj", "-R", str(repo), "workspace", "forget", slug],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "").strip()[:400]
    return True, ""


def reclaim_child_workspace(
    path: str | Path,
    *,
    snapshot: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Reclaim ONE child workspace, preserving its work.

    This is THE child-reclamation primitive: the immediate path (a finished
    child, reclaimed by the ``subagent_stop`` hook) and the age-based sweep
    both call it, so the guards, the removal and the bookkeeping can never
    drift apart.

    WHY the immediate path calls THIS and not something of its own: a
    second implementation would be a second set of guards, and the guards
    are the whole point - they are what keeps a user workspace, the main
    checkout and a live session's tree from being deleted.

    Returns a dict::

        {
            "ok": bool,             # the workspace is gone (or was already gone)
            "slug": str,
            "path": str,
            "repo": str,
            "snapshot_commit_id": str,   # "" when there was nothing to snapshot
            "was_dirty": bool,
            "reason": str,              # "" on a plain success; else the refusal
                                        # reason, or a non-fatal outcome label
                                        # (e.g. "skipped_git_worktree", or
                                        # "no_workspace_record" on an
                                        # already-gone workspace) - a SUCCESS
                                        # can carry a reason, so callers must
                                        # not read "reason is set" as
                                        # "the reclamation failed"
            "detail": str,              # human-readable explanation
            "already_gone": bool,       # idempotent no-op
            "dry_run": bool,
        }

    The guards, in order, each one commented with the failure it prevents:

    (a) NOT a child workspace (no ``child:`` purpose) -> refused. A user
        workspace is a human's tree; the reclamation is a machine decision
        about a throwaway child, and applying it to a user workspace would
        delete work nobody asked to delete.

    (b) the MAIN jj workspace -> refused. Its ``.jj/repo`` is a DIRECTORY
        (a child's is a FILE). jj forgets the main workspace with no
        warning (upstream issue #5271) and the repo is then left with no
        main checkout at all.

    (c) a LIVE session's workspace -> refused. A session bound to the tree,
        or with the tree pinned as its cwd, is still using it. Removing the
        tree out from under a running session is how a session loses its
        files mid-task.

    (d) a git-only worktree -> SKIPPED. A git worktree's dirty tree has no
        snapshot-safe equivalent: ``git worktree remove`` refuses a dirty
        worktree, and there is no "snapshot it into a commit" command that
        keeps the work. No child workspace on this host is git-only, so the
        skip is honest rather than a silent loss.

    (e) jj workspace -> ``jj workspace remove`` (which snapshots the dirty
        working copy into a commit) or ``jj workspace forget`` for a clean
        tree. The snapshot commit id is recorded in the database.

    (f) after removal the plugin's own rows for that workspace are deleted:
        the ``workspaces`` row and its ``workspace_sessions`` rows - the
        same rows the gc path deletes.

    (g) idempotent: reclaiming an already-reclaimed workspace is a
        SUCCESS with ``already_gone=True``, and the first snapshot's
        commit id is NOT overwritten.

    WHY there is no TTL here: the TTL (``store.CHILD_TTL_DAYS``) exists to
    protect UNFINISHED children - a child that is still working, or was
    never heard from, keeps its tree until the age says it is abandoned.
    A child past ``subagent_stop`` is FINISHED BY CONTRACT, so the
    immediate path is safe to reclaim at once. The age-based sweep keeps
    running for everything the immediate path did not reclaim.
    """
    from .gc import _child_hard_dead  # lazy: the two modules call each other
    ws = Path(path)
    out: dict[str, Any] = {
        "ok": False,
        "slug": ws.name,
        "path": str(ws),
        "repo": "",
        "snapshot_commit_id": "",
        "was_dirty": False,
        "reason": "",
        "detail": "",
        "already_gone": False,
        "dry_run": bool(dry_run),
    }

    # ── (g) idempotency: an already-reclaimed workspace is a success ──────
    # WHY this comes FIRST: the second call must not fall through to the
    # "no such workspace" branch and report a failure for work that is
    # already done. The recorded snapshot id is returned so the caller can
    # still point at the child's work.
    record = db.read_workspace(str(ws))
    if record is None:
        # The workspace row is gone. If a reclamation was recorded for this
        # slug, this is the idempotent second call. The repo is not known
        # any more (the row is gone), so the reclaim row is looked up by
        # slug alone - slugs are unique per workspace tree.
        prior = db.read_child_reclaim_by_slug(out["slug"])
        if prior is not None:
            out.update(
                {
                    "ok": True,
                    "already_gone": True,
                    "repo": str(prior.get("repo") or ""),
                    "snapshot_commit_id": str(prior.get("snapshot_commit_id") or ""),
                    "was_dirty": bool(prior.get("was_dirty")),
                    "detail": "already reclaimed",
                }
            )
            return out
        out.update(
            {
                "ok": True,
                "already_gone": True,
                "reason": RECLAIM_REFUSED_MISSING,
                "detail": "no workspace record and no reclamation record - nothing to do",
            }
        )
        return out

    out["repo"] = str(record.get("repo") or "")
    out["slug"] = str(record.get("slug") or ws.name)
    purpose = str(record.get("purpose") or "")

    # ── (a) only a CHILD workspace may be reclaimed ──────────────────────
    # WHY: this function is the machine-side reclamation of a throwaway
    # delegation tree. A user workspace holds a human's work, and the
    # guards below (no open PRs, pushed, no live session) are about a
    # CHILD's lifecycle - they are not a substitute for asking a human.
    if not purpose.startswith(store.CHILD_PURPOSE_PREFIX):
        out.update(
            {
                "reason": "not_a_child_workspace",
                "detail": (
                    f"purpose {purpose!r} is not a child purpose "
                    f"(expected the {store.CHILD_PURPOSE_PREFIX!r} prefix) - a user "
                    "workspace is never reclaimed by this path"
                ),
            }
        )
        return out

    # ── the workspace must be a live session's tree? NO - check FIRST ────
    # The workspace still exists (the row is there), so it is a candidate.
    # The in-use check comes after the primary-workspace check because a
    # MAIN workspace is refused outright, whatever session it serves.
    if not ws.exists():
        out.update(
            {
                "ok": True,
                "already_gone": True,
                "reason": RECLAIM_REFUSED_MISSING,
                "detail": "the directory is gone but the record is not - cleaning the record up",
            }
        )
        _forget_child_records(record, out)
        return out

    # ── (b) refuse a PRIMARY (main) jj workspace ──────────────────────────
    # WHY: `jj workspace forget` / `jj workspace remove` on the MAIN
    # workspace makes jj forget the main checkout - upstream issue #5271 -
    # and jj does it WITHOUT A WARNING. The repo is then left with no main
    # working copy, and every later command in it fails.
    if jjlayout.is_jj_workspace(ws) and jjlayout.is_primary_jj_workspace(ws):
        out.update(
            {
                "reason": RECLAIM_REFUSED_PRIMARY,
                "detail": (
                    f"{ws} is the MAIN jj workspace of its store (.jj/repo is a "
                    "directory) - removing it would make jj forget the main "
                    "checkout with no warning (upstream jj#5271)"
                ),
            }
        )
        return out

    # ── (c) refuse a workspace a LIVE session is using ───────────────────
    # WHY: a session bound to the tree, or with the tree pinned as its
    # cwd, is still working in it. Removing it mid-task is how a session
    # loses its files.
    #
    # The hard-death exception is deliberately narrow: a child that can
    # never be stamped ended (the gateway restarted mid-child) would be
    # stranded forever if the end stamp were mandatory. It is waived ONLY
    # when the heartbeat is stale AND the workspace is older than both
    # thresholds - belt and braces, so one missing marker cannot delete a
    # live tree.
    hard_dead = _child_hard_dead(record, ws)
    if os.path.normpath(str(ws)) in jjlayout.live_session_paths(db) and not hard_dead:
        out.update(
            {
                "reason": RECLAIM_REFUSED_IN_USE,
                "detail": "the workspace is bound to (or pinned by) a live session",
            }
        )
        return out

    # ── (d) a git-only worktree is SKIPPED, not reclaimed ─────────────────
    # WHY: `git worktree remove` refuses a dirty worktree, and there is no
    # git command that snapshots a dirty worktree into a commit while
    # keeping the work. Reclaiming it would either fail or (with --force)
    # lose the uncommitted changes. No child workspace on this host is
    # git-only, so the honest answer is "skipped", not a silent loss.
    if not jjlayout.is_jj_workspace(ws) and jjlayout.is_git_worktree(ws):
        out.update(
            {
                "ok": True,
                "reason": RECLAIM_SKIPPED_GIT_WORKTREE,
                "detail": (
                    "the workspace is a git-only worktree: its dirty tree has no "
                    "snapshot-safe removal, so it is left alone"
                ),
            }
        )
        return out

    repo = _source_repo(record, ws)
    if repo is None:
        out.update(
            {
                "reason": "no_source_repo",
                "detail": "the source repo of this workspace could not be found",
            }
        )
        return out

    was_dirty = bool(store.dirty(ws))
    out["was_dirty"] = was_dirty

    if dry_run:
        out.update(
            {
                "ok": True,
                "detail": (
                    "would remove the jj workspace and record the snapshot commit"
                    + (" (the tree is dirty, so the work would be snapshotted)" if was_dirty else "")
                ),
            }
        )
        return out

    # ── (e) remove the workspace, snapshotting the work ──────────────────
    # WHY remove for a DIRTY workspace: `jj workspace remove` snapshots the
    # working copy into a commit in the shared store, so the uncommitted
    # work survives the removal as a commit. `jj workspace forget` does
    # NOT snapshot - it would delete the tree and lose the work.
    snapshot_commit_id = ""
    if jjlayout.is_jj_workspace(ws):
        if was_dirty and snapshot:
            # Read the commit id BEFORE the removal: the working-copy commit
            # is the one that holds the work, and the removal rewrites it
            # in place (the id does not change).
            snapshot_commit_id = store._jj_snapshot_commit_id(ws)
            ok, why = store._jj_workspace_remove(repo, out["slug"])
            if not ok:
                out.update(
                    {
                        "reason": "jj_workspace_remove_failed",
                        "detail": f"jj workspace remove failed: {why}",
                    }
                )
                return out
        else:
            # A CLEAN workspace has nothing to snapshot. `jj workspace
            # forget` is the right command: it unregisters the workspace
            # from the store, and the tree is deleted below.
            ok, why = store._jj_workspace_forget(repo, out["slug"])
            if not ok:
                out.update(
                    {
                        "reason": "jj_workspace_forget_failed",
                        "detail": f"jj workspace forget failed: {why}",
                    }
                )
                return out
            if ws.exists():
                shutil.rmtree(ws, ignore_errors=True)
    else:
        # A plain directory that is not a jj workspace and not a git
        # worktree: nothing to unregister, just the tree.
        shutil.rmtree(ws, ignore_errors=True)

    # The workspace must be GONE FROM THE STORE, not just from the disk.
    # WHY this check: jj's "No such workspace" is a warning that exits 0,
    # so a removal that did not happen looks like a success. A stale
    # registration means the next `jj workspace add` of that name fails,
    # and the plugin's own status output would name a workspace that does
    # not exist. Removing the tree while leaving the registration behind
    # is exactly the "stale workspace" the whole bookkeeping exists to
    # avoid.
    if jjlayout.is_jj_workspace(ws) and store._jj_workspace_registered(repo, out["slug"]):
        out.update(
            {
                "reason": "jj_workspace_still_registered",
                "detail": (
                    f"{out['slug']} is still registered in the jj store of {repo} - "
                    "the tree was removed but jj still points at it"
                ),
            }
        )
        return out

    # The tree is gone (jj removed it, or we did). Make sure - a failed
    # removal that left the directory behind would make the NEXT claim
    # ("the workspace is reclaimed") a lie.
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    if ws.exists():
        out.update(
            {
                "reason": "removal_failed",
                "detail": f"{ws} still exists after the removal",
            }
        )
        return out

    # ── record the snapshot BEFORE deleting the workspace rows ───────────
    # WHY here: the row is the index that makes the child's work findable.
    # If the removal succeeded but the recording failed, the work is still
    # in the store - but nobody could find it. Recording first means a
    # failure to delete the rows leaves a reclaim row pointing at a
    # workspace that is already gone, which is recoverable.
    out["snapshot_commit_id"] = snapshot_commit_id
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        db.record_child_reclaim(
            out["repo"],
            out["slug"],
            session_id=str((record.get("sessions") or [""])[0] or ""),
            workspace_path=str(ws),
            snapshot_commit_id=snapshot_commit_id,
            was_dirty=was_dirty,
            reclaimed_at=stamp,
        )
    except Exception as exc:
        logger.warning("could not record the reclaim of %s: %s", out["slug"], exc)
        out["detail"] = f"reclaimed, but the snapshot could not be recorded: {exc}"

    # ── (f) delete the plugin's rows for this workspace ──────────────────
    _forget_child_records(record, out)

    out["ok"] = True
    if not out["detail"]:
        out["detail"] = (
            f"reclaimed {out['slug']}"
            + (f" - snapshot {snapshot_commit_id}" if snapshot_commit_id else "")
        )
    return out


def _forget_child_records(record: dict[str, Any], out: dict[str, Any]) -> None:
    """Delete the plugin's rows for a reclaimed workspace.

    The SAME rows the gc path deletes: the ``workspaces`` row and its
    ``workspace_sessions`` rows. Leaving them behind is what makes a
    workspace "stale" - a record that points at a path that is gone.
    """
    repo = str(record.get("repo") or out.get("repo") or "")
    slug = str(record.get("slug") or out.get("slug") or "")
    if not repo or not slug:
        return
    try:
        db.delete_workspace(repo, slug)
    except Exception as exc:
        logger.warning("could not delete the records of %s: %s", slug, exc)


def reclaim_children(
    slugs: list[str] | None = None,
    *,
    all_children: bool = False,
    snapshot: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Reclaim child workspaces, and report what happened.

    ``slugs``: the child workspaces to reclaim, by slug.

    ``all_children``: reclaim every child workspace whose sessions have ALL
    ENDED. A child with a live session is not reclaimable - the immediate
    path is for a FINISHED child, and "all sessions ended" is how the
    plugin knows the child is finished without waiting for a hook.

    Returns::

        {
            "reclaimed": [ {slug, snapshot_commit_id, was_dirty, ...}, ... ],
            "skipped":   [ {slug, reason, detail}, ... ],
        }
    """
    from .gc import _ended_child_workspaces  # lazy: the two modules call each other
    reclaimed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    if all_children:
        targets = _ended_child_workspaces()
    else:
        targets = []
        for slug in slugs or []:
            try:
                ws = store.find_workspace(slug)
            except WorkspaceError as exc:
                # WHY the fallback: the workspace may already be reclaimed.
                # Its record is gone, so store.find_workspace() cannot find it -
                # but the reclamation row is still there, and re-running the
                # command has to be a no-op SUCCESS, not a "no such
                # workspace" failure. The path recorded in the reclaim row
                # is the one the first reclamation used.
                prior = db.read_child_reclaim_by_slug(str(slug))
                if prior is not None:
                    targets.append(
                        {
                            "slug": str(slug),
                            "path": str(prior.get("workspace_path") or ""),
                            "already_reclaimed": True,
                        }
                    )
                    continue
                skipped.append({"slug": str(slug), "reason": "unknown_workspace", "detail": str(exc)})
                continue
            targets.append({"slug": slug, "path": str(ws)})

    for item in targets:
        slug = str(item.get("slug") or "")
        path = str(item.get("path") or "")
        if not path:
            skipped.append({"slug": slug, "reason": "no_path", "detail": "the workspace record names no path"})
            continue
        result = store.reclaim_child_workspace(path, snapshot=snapshot, dry_run=dry_run)
        # WHY the bucket is decided by ok AND reason, and not by ok alone:
        # a SKIPPED git worktree is reported with ``ok=True`` (the call
        # did what it was asked to do - it left the tree alone) and
        # ``reason="skipped_git_worktree"``. Bucketing on ``ok`` alone put
        # it under "reclaimed": the summary then claimed a tree was
        # reclaimed when it is still on disk, with its work untouched. A
        # skip is a skip, whatever the call's own ok flag says.
        skipped_reason = str(result.get("reason") or "")
        if result.get("ok") and not skipped_reason:
            reclaimed.append(
                {
                    "slug": result.get("slug") or slug,
                    "snapshot_commit_id": result.get("snapshot_commit_id") or "",
                    "was_dirty": bool(result.get("was_dirty")),
                    "path": result.get("path") or path,
                    "repo": result.get("repo") or "",
                    "already_gone": bool(result.get("already_gone")),
                    "dry_run": bool(result.get("dry_run")),
                }
            )
        else:
            skipped.append(
                {
                    "slug": result.get("slug") or slug,
                    "reason": skipped_reason or "refused",
                    "detail": result.get("detail") or "",
                    "path": result.get("path") or path,
                }
            )
    return {"reclaimed": reclaimed, "skipped": skipped}


