"""The ``hermes ws`` command line: argparse setup and the action handlers.

Pure code motion out of ``__init__.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import migrate_layout, store


def _register_cli(subparser) -> None:
    sub = subparser.add_subparsers(dest="ws_action")

    p_new = sub.add_parser("new", help="create a workspace + session record")
    p_new.add_argument("repo")
    p_new.add_argument("slug")
    p_new.add_argument("--purpose", default="")

    sub.add_parser("list", help="list workspaces and their records")

    p_status = sub.add_parser("status", help="detail for one workspace")
    p_status.add_argument("slug")

    p_retire = sub.add_parser("retire", help="archive + forget (refuses open PRs / dirty / unpushed; --force for unknowns)")
    p_retire.add_argument("slug")
    p_retire.add_argument("--force", action="store_true", help="retire even when PR/push checks could not run")
    p_retire.add_argument("--ultraforce", action="store_true", help="retire despite open PRs or a dirty tree (archive is still kept)")

    p_gc = sub.add_parser("gc", help="classify reclaimable workspaces (dry-run unless --apply)")
    p_gc.add_argument("--apply", action="store_true", help="retire the tier1 set (ended + clean + pushed + no open PRs)")
    p_gc.add_argument("--children", action="store_true", help="only child workspaces (delegated children); with --apply, safe for an unattended cron - never decides for user workspaces")

    p_forget = sub.add_parser(
        "forget",
        help="targeted removal of a stale workspace record; refuses live sessions and unforced trees",
    )
    p_forget.add_argument("--slug", required=True, help="the workspace record to forget (by slug)")
    p_forget.add_argument(
        "--force",
        action="store_true",
        help="snapshots the work first when the tree is still on disk",
    )

    # The IMMEDIATE reclamation path: a child that is done has its tree
    # reclaimed at once, and the work is preserved as a jj snapshot commit.
    p_reclaim = sub.add_parser(
        "reclaim",
        help="reclaim a finished child's workspace now (the work is preserved as a jj snapshot commit)",
    )
    p_reclaim.add_argument(
        "--slug",
        default="",
        help="the child workspace to reclaim (by slug)",
    )
    p_reclaim.add_argument(
        "--all-children",
        action="store_true",
        help="reclaim every child workspace whose sessions have all ended",
    )
    p_reclaim.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be reclaimed, reclaim nothing",
    )
    p_reclaim.add_argument(
        "--no-snapshot",
        action="store_true",
        help="do not snapshot a dirty tree into a commit (a dirty child's work would be LOST - for a tree you already know is worthless)",
    )

    p_chat = sub.add_parser("chatting", help="mark a session as talk-only")
    p_chat.add_argument("--session", default="")

    p_mig = sub.add_parser(
        "migrate",
        help="move workspaces from the old <workspace_root>/<repo>/<slug> layout "
             "to the new <repo_dir>/.workspaces/<repo>/<slug> layout (one-time)",
    )
    p_mig.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would move, move nothing",
    )

    subparser.set_defaults(ws_action="list")

def _cli_ws_forget(args) -> int:
    """``hermes ws forget --slug <slug>``: targeted removal of ONE record.

    WHY this exists next to ``retire``: retire is the LADDER (open PRs, dirty
    tree, unpushed commits, live sessions - a user's own workspace), while
    forget is the OPERATOR's targeted cleanup of a STALE record - one named
    slug, one transaction, and every guard that could delete somebody's work
    still in front of it.

    The order is deliberate:
      1. the record must exist (the slug is resolved through the DATABASE,
         not through the filesystem - a stale record's tree is gone);
      2. a LIVE session may not be bound to it;
      3. a tree still on disk needs --force, because --force is what
         authorizes the snapshot-first retirement of that tree;
      4. a user-class record needs --force too: the gc caveat for user
         workspaces ("never decided for you") applies here as well;
      5. the record and its bindings go in ONE transaction.
    """
    slug = str(getattr(args, "slug", "") or "")
    force = bool(getattr(args, "force", False))

    record = None
    for row in store.db.all_workspaces():
        if str(row.get("slug") or "") == slug:
            record = dict(row)
            break
    if record is None:
        available = sorted(
            {
                str(row.get("slug") or "")
                for row in store.db.all_workspaces()
                if str(row.get("slug") or "")
            }
        )
        print(f"refused: no workspace record for slug {slug!r}")
        print("available slugs: " + (", ".join(available) if available else "(none)"))
        return 1

    path = Path(str(record.get("path") or ""))
    purpose = str(record.get("purpose") or "")
    is_child = purpose.startswith(store.CHILD_PURPOSE_PREFIX)

    # (2) liveness: the same check the reclamation primitive uses - a LIVE
    # session bound to (or pinned at) the tree. A record whose tree is
    # already gone has nothing a live session can still be using.
    if path.is_dir() and os.path.normpath(str(path)) in store.jjlayout.live_session_paths(store.db):
        print(f"refused: {slug} is bound to (or pinned by) a live session")
        return 1

    tree_exists = path.is_dir()

    # (3) a tree on disk is never removed without --force: --force is the
    # explicit "snapshot the work first" authorization.
    if tree_exists and not force:
        print(
            f"refused: the tree for {slug} is still on disk ({path}); "
            "repeat with --force to snapshot the work first and remove it"
        )
        return 1

    # (4) a user-class record is a human's workspace: the gc caveat applies
    # here too - the operator has to say so explicitly.
    if not is_child and not force:
        print(
            f"refused: {slug} is a user workspace ({purpose or 'no purpose'}); "
            "repeat with --force to forget its record"
        )
        return 1

    snapshot_commit_id = ""
    if tree_exists:
        if is_child:
            # The child-classified dirty/unpushed protection lives inside the
            # reclamation primitive - reuse it, do NOT re-implement the checks
            # and do NOT write a second removal path.
            result = store.reclaim_child_workspace(path)
            if not result.get("ok"):
                detail = str(result.get("detail") or result.get("reason") or "retirement refused")
                print(f"refused: {slug} was not retired: {detail}")
                return 1
            snapshot_commit_id = str(result.get("snapshot_commit_id") or "")
        else:
            # A user workspace is retired by the SAME ladder `ws retire`
            # uses - --force acknowledges the unverifiable checks, and
            # --ultraforce is what lets a dirty tree go (the work is archived
            # first, so the record is still recoverable).
            try:
                store.retire(slug, force=True, ultraforce=True)
            except store.WorkspaceError as exc:
                print(f"refused: {slug} was not retired: {exc}")
                return 1

    # (5) ONE transaction: the record and its bindings go together, or
    # neither goes.
    repo = str(record.get("repo") or "")
    with store.db.transaction() as conn:
        cur_ws = conn.execute(
            "DELETE FROM workspaces WHERE repo=? AND slug=?", (repo, slug)
        )
        cur_bind = conn.execute("DELETE FROM bindings WHERE slug=?", (slug,))
        rows_removed = int(cur_ws.rowcount or 0) + int(cur_bind.rowcount or 0)

    print(f"forgot {slug} ({path}) - removed {rows_removed} row(s)")
    if snapshot_commit_id:
        print(f"snapshot commit: {snapshot_commit_id}")
    return 0

def _cli(args) -> int:
    action = getattr(args, "ws_action", "list") or "list"
    try:
        if action == "new":
            record = store.create(args.repo, args.slug, purpose=args.purpose)
            print(record["path"])
            return 0
        if action == "list":
            items = store.list_workspaces()
            if not items:
                print(f"no workspaces under {store.root_dir()}")
                return 0
            for item in items:
                record = item["record"] or {}
                flags = []
                if item["orphan"]:
                    flags.append("ORPHAN(no session.json)")
                else:
                    # A workspace whose every session was finalised is done, not
                    # orphaned - distinguish that from a vanished session.
                    ends = []
                    for sid in record.get("sessions") or []:
                        st = store.read_state(str(sid)) or {}
                        if st.get("session_ended_at"):
                            ends.append(str(st["session_ended_at"]))
                    if ends and len(ends) == len(record.get("sessions") or []):
                        flags.append(f"session ended {max(ends)}")
                    elif record.get("last_synced"):
                        flags.append(f"synced {record['last_synced']}")
                print(f"{item['repo']}/{item['slug']}  {item['path']}  {' '.join(flags)}".rstrip())
            return 0
        if action == "status":
            ws = store.find_workspace(args.slug)
            record = store.read_record(ws)
            prs, gh_ok = store.open_prs(ws)
            print(json.dumps({"path": str(ws), "record": record, "open_prs": prs, "pr_check_ok": gh_ok}, indent=2, default=str))
            return 0
        if action == "retire":
            print(store.retire(
                args.slug,
                force=getattr(args, "force", False),
                ultraforce=getattr(args, "ultraforce", False),
            ))
            return 0
        if action == "forget":
            return _cli_ws_forget(args)
        if action == "gc":
            tiers = store.gc_tiers()
            if getattr(args, "children", False):
                for name in ("tier1", "tier2", "tier3"):
                    tiers[name] = [
                        row for row in tiers[name]
                        if str((row.get("record") or {}).get("purpose") or "").startswith(store.CHILD_PURPOSE_PREFIX)
                    ]
            titles = {
                "tier1": "safe to retire (session ended, clean, pushed, no open PRs)",
                "tier2": "needs a look first (unverified, orphans, open PRs, or dirty)",
                "tier3": "in use (session still live)",
            }
            total = sum(len(v) for v in tiers.values())
            if total == 0:
                print("no workspaces found - nothing to reclaim")
                return 0
            for name in ("tier1", "tier2", "tier3"):
                rows = tiers[name]
                print(f"\n{titles[name]}: {len(rows)}")
                for row in rows:
                    print(f"  {row['repo']}/{row['slug']}")
                    print(f"    {row['path']}")
                    print(f"    why: {row['why']}")
            if getattr(args, "apply", False):
                done = store.gc_children_apply() if getattr(args, "children", False) else store.gc_apply()
                print(f"\nretired: {len(done)}" if done else "\nretired: nothing eligible")
                for path in done:
                    print(f"  {path}")
            else:
                print(
                    "\nNothing was deleted. Reclaim the 'safe to retire' set with: "
                    "hermes ws gc --apply"
                )
            return 0
        if action == "reclaim":
            slug = str(getattr(args, "slug", "") or "")
            all_children = bool(getattr(args, "all_children", False))
            if not slug and not all_children:
                print("nothing to reclaim: pass --slug <slug> or --all-children")
                return 2
            if slug and all_children:
                print("pass either --slug or --all-children, not both")
                return 2
            out = store.reclaim_children(
                [slug] if slug else None,
                all_children=all_children,
                snapshot=not bool(getattr(args, "no_snapshot", False)),
                dry_run=bool(getattr(args, "dry_run", False)),
            )
            # The structured report: what was reclaimed (and where its work
            # ended up) and what was skipped (and why).
            print(json.dumps(out, indent=2, default=str))
            if getattr(args, "dry_run", False):
                print("\ndry run: nothing was reclaimed")
            return 0
        if action == "chatting":
            store.mark_chatting(args.session or "cli")
            print("session marked as chatting")
            return 0
        if action == "migrate":
            # The one-time LAYOUT migration: move the workspaces that live
            # under the old root into the dot-dir layout. It never runs
            # implicitly - the operator asks for it once.
            out = migrate_layout.migrate_workspaces(dry_run=bool(getattr(args, "dry_run", False)))
            print(migrate_layout.report_text(out))
            if getattr(args, "dry_run", False):
                print("\ndry run: nothing was moved")
            else:
                print(
                    f"\nmoved {len(out.get('moved') or [])}, "
                    f"skipped {len(out.get('skipped_dirty') or [])} dirty, "
                    f"in use {len(out.get('in_use') or [])}, "
                    f"verify failed {len(out.get('verify_failed') or [])}, "
                    f"repair failed {len(out.get('repair_failed') or [])}"
                )
            return 0
    except store.WorkspaceError as e:
        print(f"refused: {e}")
        return 2
    return 0
