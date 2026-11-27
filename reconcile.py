"""Reconcile what treeps believes against what is on disk and in the host.

WHY: nothing ever compared the three ledgers. The result accumulated quietly -
record-less trees (67 of them, once), capacity rows whose gateway pid was long
gone, and children the host still considered resumable. Every one of those was
found by hand, weeks late.

This module REPORTS. It never retires, deletes or forgets anything: a
reconciler that quietly fixes drift is a reconciler that quietly deletes a
workspace somebody was standing in. Removal stays in ``gc`` behind a human.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from . import capacity, db, hostapi, store

logger = logging.getLogger(__name__)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def drift() -> dict[str, list[dict[str, Any]]]:
    """Every disagreement between the DB, the disk and the host, in one dict.

    Keys are the finding names; each value is a list of small dicts naming the
    workspace or row. Empty lists are the healthy answer.
    """
    out: dict[str, list[dict[str, Any]]] = {
        "missing_tree": [],
        "unrecorded_tree": [],
        "dead_capacity_row": [],
        "pinned_child": [],
    }

    for item in store.list_workspaces():
        record = item.get("record") or {}
        path = Path(str(item.get("path") or ""))
        name = {"repo": item.get("repo"), "slug": item.get("slug"), "path": str(path)}
        if record and not path.exists():
            out["missing_tree"].append({**name, "reason": "record without a tree"})
        if not record and path.exists():
            out["unrecorded_tree"].append({**name, "reason": "tree without a record"})
        # A child the host still holds a live resume pin for must never be
        # reclaimed on age alone - the host intends to continue it.
        purpose = str(record.get("purpose") or "")
        if purpose.startswith(store.CHILD_PURPOSE_PREFIX) and hostapi.is_pinned_child(str(path)):
            out["pinned_child"].append({**name, "reason": "host resume pin is active"})

    for row in _dead_capacity_rows():
        out["dead_capacity_row"].append(row)

    return {k: v for k, v in out.items() if v}


def _dead_capacity_rows() -> list[dict[str, Any]]:
    """Open capacity rows whose writing gateway is gone (the stale-counter class)."""
    try:
        alive, total = capacity.count_live_detail()
    except Exception:
        logger.debug("reconcile: capacity ledger unreadable", exc_info=True)
        return []
    if total <= alive:
        return []
    return [
        {
            "reason": f"{total - alive} open row(s) with no live pid",
            "alive": alive,
            "open_rows": total,
        }
    ]


def report(*, write: bool = True) -> dict[str, Any]:
    """The drift report, optionally written next to the plugin's data."""
    found = drift()
    payload = {
        "at": _now(),
        "findings": found,
        "total": sum(len(v) for v in found.values()),
    }
    if payload["total"]:
        logger.info(
            "reconcile: %s finding(s) - %s",
            payload["total"],
            ", ".join(f"{k}={len(v)}" for k, v in found.items()),
        )
    if write:
        try:
            path = db.hermes_home() / "plugin-data" / db.PLUGIN_NAME / "reconcile.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:
            logger.debug("reconcile: could not write the report", exc_info=True)
    return payload


def register(ctx: Any) -> bool:
    """Ask the host to run the reconciler when idle (hermes >= 0.23.2)."""
    return hostapi.register_idle_task(
        ctx,
        "reconcile",
        interval_hours=6,
        min_idle_hours=1,
        fn=lambda: report(write=True),
    )
