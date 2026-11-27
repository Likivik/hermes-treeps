"""A refusal must name the ledger and the evidence, not just a number."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import plugin_loader  # noqa: E402

CAP = plugin_loader.PLUGIN.capacity


def test_allowed_message_counts_slots():
    ok, reason = CAP.verdict(2, 3, 8)
    assert ok
    assert "3 children live" in reason and "2 requested" in reason and "5 free" in reason


def test_refusal_names_the_numbers():
    ok, reason = CAP.verdict(4, 8, 8)
    assert not ok
    assert "8 children live" in reason and "0 free slot" in reason


def test_refusal_names_a_stale_ledger():
    """The 2026-11-26 shape: open rows with no live pid behind them."""
    ok, reason = CAP.verdict(1, 8, 8, open_rows=8)
    assert not ok
    assert "8 open row(s)" in reason


def test_refusal_shows_a_mismatch_between_rows_and_live_pids():
    ok, reason = CAP.verdict(1, 0, 8, open_rows=3)
    assert ok, "0 live is allowed - the rows are the evidence, not the count"
    assert "3 open row(s)" in reason and "0 matched a live pid" in reason


def test_detail_returns_both_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(CAP, "_ledger_paths", lambda **_kw: [])
    assert CAP.count_live_detail() == (0, 0)
