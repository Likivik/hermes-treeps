"""capacity.live_from_logs: the route-aware, cross-session live count.

WHY this exists next to the SQLite ledger: the ledger cannot name a child's
route. ``subagent_start`` carries no model/provider, so a start row falls back
to the configured delegation route - and a child this plugin already sent to the
overflow provider then counts as primary. Counting it as primary is not merely
conservative, it poisons the next batch: with every slot "taken", the whole
fleet moves to the overflow provider while the earlier children are still
running.

The host's transcripts do carry the answer, so these tests pin that reading:
a manifest names the batch's provider/model, a log with no ``final`` marker
whose mtime is recent is a running child, and a log with no marker that has not
been written for hours is an orphan rather than a live child.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from plugin_loader import PLUGIN

capacity = PLUGIN.capacity


def _delegation(root: Path, deleg_id: str, provider: str, model: str, tasks: int) -> Path:
    """Write one manifest + one log per task, as the host lays them out."""
    folder = root / deleg_id
    folder.mkdir(parents=True, exist_ok=True)
    described = []
    for index in range(tasks):
        log = folder / f"task-{index}.log"
        log.write_text(f"2026-11-11 10:00:00 ready    | task {index} starting\n", encoding="utf-8")
        described.append({"index": index, "goal": f"goal {index}", "log": str(log), "status": "running"})
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "delegation_id": deleg_id,
                "started": "2026-11-11 10:00:00",
                "task_count": tasks,
                "model": model,
                "provider": provider,
                "tasks": described,
            }
        ),
        encoding="utf-8",
    )
    return folder


def _finish(folder: Path, index: int, status: str = "completed") -> None:
    log = folder / f"task-{index}.log"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"2026-11-11 10:05:00 final    | end status={status} exit_reason={status}\n")


def _age(folder: Path, index: int, seconds: float) -> None:
    log = folder / f"task-{index}.log"
    stamp = time.time() - seconds
    os.utime(log, (stamp, stamp))


def test_a_finished_log_is_not_live(tmp_path):
    folder = _delegation(tmp_path, "deleg_a", "minimax", "MiniMax-M3.1-Flash", tasks=2)
    _finish(folder, 0)
    _finish(folder, 1, status="failed")

    assert capacity.live_from_logs(root=tmp_path) == 0


def test_a_log_without_the_final_marker_is_live(tmp_path):
    folder = _delegation(tmp_path, "deleg_b", "minimax", "MiniMax-M3.1-Flash", tasks=2)
    _finish(folder, 0)

    assert capacity.live_from_logs(root=tmp_path) == 1
    assert capacity.live_from_logs(root=tmp_path, minimax_only=True) == 1


def test_the_route_comes_from_the_manifest_not_the_plugin_setting(tmp_path):
    """The overflow child is NOT counted as primary - the self-poisoning bug.

    Prevents: every child a previous batch sent to deepseek counting as
    MiniMax, which pushes the next batch entirely onto the overflow provider.
    """
    minimax = _delegation(tmp_path, "deleg_m", "minimax", "MiniMax-M3.1-Flash", tasks=1)
    deepseek = _delegation(tmp_path, "deleg_d", "custom:neuralwatt", "deepseek-v4.1-flash-flex", tasks=2)
    _finish(minimax, 0)

    assert capacity.live_from_logs(root=tmp_path) == 2
    assert capacity.live_from_logs(root=tmp_path, minimax_only=True) == 0

    # A live primary child is counted, a live overflow child is not.
    _delegation(tmp_path, "deleg_m2", "minimax", "MiniMax-M3.1-Flash", tasks=1)
    assert capacity.live_from_logs(root=tmp_path, minimax_only=True) == 1
    assert deepseek.is_dir()


def test_an_orphan_log_is_not_live_forever(tmp_path):
    """No final marker, last written hours ago: a hard kill, not a running child.

    Prevents: one SIGKILLed child pinning the primary route for every later
    batch. The window is long because children legitimately run for hours.
    """
    folder = _delegation(tmp_path, "deleg_orphan", "minimax", "MiniMax-M3.1-Flash", tasks=1)
    _age(folder, 0, seconds=capacity.DEFAULT_STALE_SECONDS + 60)

    assert capacity.live_from_logs(root=tmp_path, minimax_only=True) == 0


def test_a_missing_or_broken_tree_counts_zero(tmp_path):
    assert capacity.live_from_logs(root=tmp_path / "nope") == 0
    broken = tmp_path / "deleg_broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    assert capacity.live_from_logs(root=tmp_path) == 0


def test_overflow_boundary_uses_the_live_primary_count(tmp_path):
    """With one primary child live, a 6-task batch overflows from index 3."""
    folder = _delegation(tmp_path, "deleg_live", "minimax", "MiniMax-M3.1-Flash", tasks=1)
    assert folder.is_dir()
    live = capacity.live_from_logs(root=tmp_path, minimax_only=True)
    assert live == 1
    flags = [capacity.should_overflow(i, live, 4) for i in range(6)]
    assert flags == [False, False, False, True, True, True], flags
