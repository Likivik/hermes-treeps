"""GC classification and the age-based sweeps. Dry-run unless the caller applies."""
from __future__ import annotations

import calendar
import logging
import time
from pathlib import Path
from typing import Any

from . import (
    db,
    store,  # late-bound: tests monkeypatch store.<helper>, so bind at call time
)
from .errors import WorkspaceError

logger = logging.getLogger(__name__)

# ── gc ────────────────────────────────────────────────────────────────────────


# A child workspace is throwaway by design. Retiring it does NOT lose committed
# work: `jj workspace forget` / `git worktree remove` delete the working copy
# and metadata, but the commits live in the SHARED repo store and the child's
# branch bookmark keeps pointing at them. Only uncommitted changes are lost,
# and the dirty check blocks those. So after this many days a clean, ended
# child is reclaimable regardless of push state - a never-applied handoff
# commit stays recoverable as a dangling commit until the repo's own gc.

# A hard-dead child (the gateway restarted mid-child) never receives the
# ``subagent_stop`` event, so it never gets the graceful ``session_ended_at``
# stamp. The heartbeat is the second, independent marker for that case.
CHILD_HEARTBEAT_STALE_HOURS = 24
CHILD_HEARTBEAT_KEY_PREFIX = "child_seen:"
CHILD_HEARTBEAT_MIN_INTERVAL_S = 60


def child_heartbeat_key(session_id: str) -> str:
    """The kv key holding one child's last gated-tool heartbeat."""
    return f"{CHILD_HEARTBEAT_KEY_PREFIX}{session_id}"


def record_child_heartbeat(session_id: str, *, now: float | None = None) -> bool:
    """Record one child heartbeat, rate-limited to one write per minute.

    WHY the read-then-write: a child can make hundreds of gated calls per
    minute, and one sqlite upsert per call is needless WAL churn. The
    heartbeat only needs minute-level resolution for a 24-hour staleness
    threshold.

    WHY the read happens UNDER the write lock: the read-then-write pair is
    only atomic if nothing else writes the same key in between. Two
    threads heartbeating for the same child (a re-dispatch, or a parent
    and a child sharing an id) could both read the old stamp, both decide
    the interval has passed, and both write - the rate limit would then be
    bypassed.

    WHY the clock-step guard: the stamp is a WALL-CLOCK time, and wall
    clocks step backwards (NTP correction, a manual change, a resume from
    a suspended VM). A backwards step made ``stamp - previous`` NEGATIVE,
    and a negative difference is below the one-minute interval, so EVERY
    subsequent heartbeat was rate-limited away: the child's heartbeat
    froze at the pre-step value and the child looked stale to the sweep
    while it was very much alive. The comparison therefore uses
    ``abs()``: a backwards step is treated as "the interval passed" and
    the new (lower) stamp is written, so the heartbeat recovers at once
    instead of freezing for 24 hours.

    WHY this must never raise: the heartbeat is liveness evidence, not a
    gate. A broken database write must not block the child's tool call or
    turn a bookkeeping failure into a workspace-access failure.
    """
    if not session_id:
        return False
    stamp = time.time() if now is None else float(now)
    key = child_heartbeat_key(session_id)
    try:
        with db.transaction() as conn:
            # WHY the read is on THIS connection: the read-then-write pair
            # is only atomic if the read sees the same database state the
            # write will land on. ``db.get_kv()`` opens its own
            # connection, so its answer could already be stale.
            previous = db.get_kv_in(conn, key)
            if previous:
                try:
                    # WHY abs(): a backwards clock step must not freeze the
                    # heartbeat for the next 24 hours (see the docstring).
                    if abs(stamp - float(previous)) < CHILD_HEARTBEAT_MIN_INTERVAL_S:
                        return False
                except (TypeError, ValueError):
                    pass
            db._write_kv(conn, key, f"{stamp:.6f}")
        return True
    except Exception as exc:
        # A failed heartbeat must not gate-block the child: liveness evidence
        # is best-effort, and turning its failure into a blocked tool call
        # would strand a healthy child.
        logger.warning("could not record child heartbeat for %s: %s", session_id, exc)
        return False


def _parse_timestamp(raw: str) -> float | None:
    """Parse a heartbeat or workspace timestamp into epoch seconds."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return float(calendar.timegm(time.strptime(text[:19], fmt)))
        except Exception:
            continue
    return None


def child_heartbeat_stale(
    session_id: str,
    stale_hours: int = CHILD_HEARTBEAT_STALE_HOURS,
    *,
    record: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> bool:
    """True when this child has no fresh heartbeat.

    A heartbeat older than ``stale_hours`` is stale. When no heartbeat was
    ever written, the WORKSPACE record's ``created_at`` is the age anchor:
    a brand-new child that has not called a gated tool yet must not look
    dead. ``updated_at`` and the tree mtime are fallbacks for an older
    record that has no creation stamp.

    NOTE: the hard-death backstop is stricter - it requires a heartbeat that
    EXISTS and is stale (see :func:`_child_hard_dead`). This helper answers
    the general liveness question; the backstop needs positive evidence.
    """
    age_s = max(0.0, float(stale_hours) * 3600.0)
    try:
        raw = db.get_kv(child_heartbeat_key(session_id))
    except Exception:
        raw = None
    seen = _parse_timestamp(raw or "")
    if seen is not None:
        return (time.time() - seen) > age_s

    anchor: Any = None
    if record:
        anchor = record.get("created_at") or record.get("updated_at")
    if not anchor and path:
        try:
            anchor = Path(path).stat().st_mtime
        except Exception:
            anchor = None
    if anchor is None:
        state = store.read_state(session_id) or {}
        anchor = state.get("updated_at")
    created = _parse_timestamp(str(anchor or ""))
    if created is None:
        return False
    return (time.time() - created) > age_s


def _child_hard_dead(
    record: dict[str, Any],
    path: str | Path,
    *,
    stale_hours: int = CHILD_HEARTBEAT_STALE_HOURS,
) -> bool:
    """True only when BOTH heartbeat staleness and workspace age exceed 24h.

    A missing heartbeat alone is not enough: a brand-new child that has not
    called a gated tool has no heartbeat yet. The two-threshold rule is the
    extra-sure guard against reclaiming a live child.

    WHY the heartbeat must EXIST here: a child with no heartbeat at all has
    never proved life through the plugin, so only its age can speak for it.
    The hard-death path requires positive evidence that the child WAS alive
    and then stopped reporting.
    """
    age_days = _workspace_age_days(record, Path(path))
    if age_days * 24.0 <= float(stale_hours):
        return False
    sids = [str(x) for x in (record.get("sessions") or [])]
    if not sids:
        return False
    for sid in sids:
        try:
            raw = db.get_kv(child_heartbeat_key(sid))
        except Exception:
            return False
        seen = _parse_timestamp(raw or "")
        if seen is None:
            # No heartbeat was ever written: absence is not a stale marker.
            return False
        if (time.time() - seen) <= float(stale_hours) * 3600.0:
            return False
    return True


def _child_workspace_records() -> list[dict[str, Any]]:
    """Every child workspace record, read straight from the database.

    WHY this exists: the gc sweep walks the workspace ROOT on disk, so a
    record whose tree lives outside that root would be invisible to it.
    The database is the plugin's own index of every workspace it created,
    so the dual-marker rule must consult it too.
    """
    out: list[dict[str, Any]] = []
    try:
        rows = db.all_workspaces()
    except Exception:
        rows = []
    for record in rows:
        if not str(record.get("purpose") or "").startswith(store.CHILD_PURPOSE_PREFIX):
            continue
        path = str(record.get("path") or "")
        if not path:
            continue
        out.append({**record, "path": path, "slug": str(record.get("slug") or Path(path).name)})
    return out


# Written into a workspace record's `purpose` by childspace when a delegated
# child gets its own tree. One definition, imported by childspace, so gc and
# the creator cannot disagree about what a child workspace is.



def _ended_child_workspaces() -> list[dict[str, Any]]:
    """Every child workspace whose sessions have ALL ended.

    WHY "all ended" and not "any ended": a workspace can be shared by more
    than one session (a child and its follow-up), and reclaiming it while
    one of them still runs is the failure the live-session guard exists to
    prevent. The check here is the cheap version of that guard; the
    authoritative one runs inside :func:`reclaim_child_workspace`.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in store.list_workspaces():
        record = item.get("record") or {}
        if not str(record.get("purpose") or "").startswith(store.CHILD_PURPOSE_PREFIX):
            continue
        sids = [str(x) for x in (record.get("sessions") or [])]
        if not sids:
            continue
        states = [store.read_state(sid) or {} for sid in sids]
        ended = all(st.get("session_ended_at") for st in states)
        if not ended and not _child_hard_dead(record, item.get("path") or ""):
            continue
        out.append({"slug": str(item.get("slug") or ""), "path": str(item.get("path") or "")})
        seen.add(str(item.get("path") or ""))
    # The disk walk above only sees trees under the workspace root. The
    # database also knows about child workspaces elsewhere, and a hard-dead
    # orphan must be reclaimable wherever its tree lives.
    for record in _child_workspace_records():
        path = str(record.get("path") or "")
        if path in seen:
            continue
        sids = [str(x) for x in (record.get("sessions") or [])]
        if not sids:
            continue
        states = [store.read_state(sid) or {} for sid in sids]
        ended = all(st.get("session_ended_at") for st in states)
        if not ended and not _child_hard_dead(record, path):
            continue
        out.append({"slug": str(record.get("slug") or ""), "path": path})
    return out


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
    listed: set[str] = set()
    for item in store.list_workspaces():
        record = item["record"] or {}
        path = Path(item["path"])
        listed.add(str(path))
        sids = [str(x) for x in (record.get("sessions") or [])]
        if item["orphan"] or not sids:
            tiers["tier2"].append({**item, "why": "orphan: no bound session record"})
            continue
        states = [store.read_state(sid) or {} for sid in sids]
        ended = all(st.get("session_ended_at") for st in states)
        is_child = str(record.get("purpose") or "").startswith(store.CHILD_PURPOSE_PREFIX)
        hard_dead = is_child and _child_hard_dead(record, path)
        if not ended and not hard_dead:
            tiers["tier3"].append({**item, "why": "session still live"})
            continue
        # Child workspaces: unpublished by design, so the "pushed" test that
        # protects a session's own work does not apply. Age does.
        if is_child:
            if store.dirty(path):
                tiers["tier2"].append({**item, "why": "child workspace with uncommitted changes"})
                continue
            age_days = _workspace_age_days(record, path)
            # NOTE: no pushed check here - see store.CHILD_TTL_DAYS above. A child's
            # handoff commit survives retirement in the shared store.
            if age_days >= store.CHILD_TTL_DAYS:
                tiers["tier1"].append(
                    {**item, "why": f"child workspace, clean, {age_days:.0f}d old"}
                )
            else:
                tiers["tier2"].append(
                    {**item, "why": f"child workspace, clean, {age_days:.0f}d old (ttl {store.CHILD_TTL_DAYS}d)"}
                )
            continue
        prs, gh_ok = store.open_prs(path)
        mine = store._workspace_branches(path)
        live = [p for p in prs if p.get("state") == "OPEN" and p.get("headRefName") in mine]
        pushed = store.unpushed_commits(path)
        if live:
            tiers["tier2"].append(
                {**item, "why": "open PRs " + ", ".join(f"#{p['number']}" for p in live)}
            )
        elif store.dirty(path):
            tiers["tier2"].append({**item, "why": "uncommitted changes"})
        elif pushed is not False or not gh_ok:
            tiers["tier2"].append(
                {**item, "why": "could not verify pushed state" if pushed is None else "unpushed commits"}
            )
        else:
            tiers["tier1"].append({**item, "why": "ended, clean, pushed, no open PRs"})
    # The disk walk only sees trees under the workspace root. The database
    # also knows about child workspaces elsewhere; a hard-dead orphan must
    # be reclaimable wherever its tree lives.
    for record in _child_workspace_records():
        path = str(record.get("path") or "")
        if path in listed:
            continue
        listed.add(path)
        item = {
            "repo": str(record.get("repo") or ""),
            "slug": str(record.get("slug") or Path(path).name),
            "path": path,
            "record": record,
            "orphan": False,
        }
        sids = [str(x) for x in (record.get("sessions") or [])]
        if not sids:
            tiers["tier2"].append({**item, "why": "orphan: no bound session record"})
            continue
        states = [store.read_state(sid) or {} for sid in sids]
        ended = all(st.get("session_ended_at") for st in states)
        hard_dead = _child_hard_dead(record, path)
        if not ended and not hard_dead:
            tiers["tier3"].append({**item, "why": "session still live"})
            continue
        if store.dirty(Path(path)):
            tiers["tier2"].append({**item, "why": "child workspace with uncommitted changes"})
            continue
        age_days = _workspace_age_days(record, Path(path))
        if age_days >= store.CHILD_TTL_DAYS:
            tiers["tier1"].append(
                {**item, "why": f"child workspace, clean, {age_days:.0f}d old"}
            )
        else:
            tiers["tier2"].append(
                {
                    **item,
                    "why": (
                        f"child workspace, clean, {age_days:.0f}d old "
                        f"(ttl {store.CHILD_TTL_DAYS}d)"
                    ),
                }
            )
    return tiers


def gc_apply() -> list[str]:
    """Retire exactly the tier1 set, through the same guarded store.retire().

    WHY the child tier goes through :func:`reclaim_child_workspace` instead
    of ``store.retire()``: the child's work must SURVIVE the reclamation as a
    snapshot commit, and only the reclamation primitive does that. ``store.retire()``
    REFUSES a dirty tree - which is exactly the tree that holds the work
    worth preserving. Both paths end in the same place (the tree is gone,
    the rows are gone), and there is now ONE code path for a child.
    """
    done: list[str] = []
    for item in store.gc_tiers()["tier1"]:
        record = item.get("record") or {}
        is_child = str(record.get("purpose") or "").startswith(store.CHILD_PURPOSE_PREFIX)
        if is_child:
            try:
                result = store.reclaim_child_workspace(str(item["path"]))
            except Exception as e:
                logger.warning("gc skipped %s: %s", item["slug"], e)
                continue
            if result.get("ok"):
                done.append(str(result.get("detail") or f"reclaimed {item['slug']}"))
            else:
                logger.warning(
                    "gc skipped %s: %s", item["slug"], result.get("detail") or result.get("reason")
                )
            continue
        try:
            done.append(store.retire(item["slug"], allow_unpushed=False))
        except WorkspaceError as e:
            # A REFUSAL: the guard said no. Log it and move on.
            logger.warning("gc skipped %s: %s", item["slug"], e)
        except Exception as e:
            # A BROKEN retirement: a sqlite error in the row deletion, an
            # OSError writing the archive, anything that is not a refusal.
            #
            # WHY this is caught and not allowed to escape: this loop is
            # the sweep. An exception here ended the sweep on the FIRST
            # bad workspace, and every workspace after it was never
            # retired - so the sweep silently did less and less over
            # time, and the stale trees it exists to remove stayed. One
            # broken workspace must not stop the other retirements.
            logger.warning("gc could not retire %s: %s", item["slug"], e)
    # Workspaceless session rows (cron, chat-only) never fire the session-end
    # hook, so they accumulate one per run and make "live" mean nothing.
    # They claim no path, so pruning them cannot pin or unpin any tree.
    try:
        pruned = db.prune_workspaceless_bindings()
        if pruned:
            logger.debug("gc pruned %d workspaceless session row(s)", pruned)
    except Exception as exc:  # a broken prune must not fail the sweep
        logger.warning("gc could not prune workspaceless session rows: %s", exc)
    return done


def gc_children_apply() -> list[str]:
    """Reclaim ONLY expired CHILD workspaces (tier1 with the child purpose).

    Used by the plugin's daily housekeeping sweep. User workspaces are never
    touched here: their tier1 rules require pushed commits and no open PRs,
    and an automatic sweep must not decide those.

    WHY this calls the SAME primitive as the immediate path
    (:func:`reclaim_child_workspace`): one code path for a child, so the
    guards, the removal and the snapshot recording cannot drift apart. The
    only difference between the two is the TTL - see the note in
    :func:`reclaim_child_workspace`.
    """
    done: list[str] = []
    for item in store.gc_tiers()["tier1"]:
        record = item.get("record") or {}
        if not str(record.get("purpose") or "").startswith(store.CHILD_PURPOSE_PREFIX):
            continue
        try:
            result = store.reclaim_child_workspace(str(item["path"]))
        except Exception as e:
            logger.warning("child gc skipped %s: %s", item["slug"], e)
            continue
        if result.get("ok"):
            done.append(str(result.get("detail") or f"reclaimed {item['slug']}"))
        else:
            logger.warning(
                "child gc skipped %s: %s", item["slug"], result.get("detail") or result.get("reason")
            )
    return done


