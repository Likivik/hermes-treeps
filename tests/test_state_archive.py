"""Retiring the pre-DB JSON files: rename, never delete, refuse the unknown."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import plugin_loader  # noqa: E402

DB = plugin_loader.STORE.db


@pytest.fixture()
def legacy(tmp_path, monkeypatch):
    d = tmp_path / "state"
    d.mkdir()
    monkeypatch.setattr(DB, "legacy_state_dir", lambda: d)
    return d


def _write(dir_: Path, sid: str, payload: dict) -> None:
    (dir_ / f"{sid}.json").write_text(json.dumps(payload))


def test_no_dir_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(DB, "legacy_state_dir", lambda: tmp_path / "absent")
    report = DB.archive_legacy_state(apply=True)
    assert report["files"] == 0
    assert report["applied"] is False


def test_dry_run_never_renames(legacy):
    _write(legacy, "s1", {"mode": "repo"})
    DB.write_binding("s1", {"mode": "repo"})
    report = DB.archive_legacy_state(apply=False)
    assert report["files"] == 1
    assert report["applied"] is False
    assert legacy.is_dir()


def test_unmigrated_file_refuses_the_whole_archive(legacy):
    _write(legacy, "known", {"mode": "repo"})
    _write(legacy, "stranger", {"mode": "repo"})
    DB.write_binding("known", {"mode": "repo"})
    report = DB.archive_legacy_state(apply=True)
    assert report["unmigrated"] == ["stranger.json"]
    assert report["applied"] is False
    assert legacy.is_dir(), "a refused archive must not move anything"


def test_all_migrated_archives_by_rename(legacy):
    _write(legacy, "s1", {"mode": "repo"})
    DB.write_binding("s1", {"mode": "repo"})
    report = DB.archive_legacy_state(apply=True)
    assert report["applied"] is True
    assert not legacy.exists()
    target = Path(report["archived_to"])
    assert target.is_dir() and (target / "s1.json").exists()
