"""The reconciler reports drift and deletes nothing."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import plugin_loader  # noqa: E402

PLUGIN = plugin_loader.PLUGIN

REC = importlib.import_module("hermes_treeps.reconcile")
STORE = plugin_loader.STORE


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(REC, "capacity", REC.capacity)
    monkeypatch.setattr(REC.hostapi, "is_pinned_child", lambda path: False)
    monkeypatch.setattr(REC.capacity, "count_live_detail", lambda **_kw: (0, 0))
    # Anything that could remove a workspace fails the test loudly.
    for mod, name in ((REC.store, "retire"), (REC.store, "forget"), (REC.store, "reclaim_child_workspace")):
        monkeypatch.setattr(mod, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("reconcile deleted something")))


def _item(tmp_path, slug, record):
    return {"repo": "repo", "slug": slug, "path": str(tmp_path / slug), "record": record}


def test_healthy_state_has_no_findings(tmp_path, monkeypatch):
    (tmp_path / "ok").mkdir()
    monkeypatch.setattr(STORE, "list_workspaces", lambda: [_item(tmp_path, "ok", {"sessions": ["s"]})])
    assert REC.drift() == {}


def test_record_without_a_tree_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(STORE, "list_workspaces", lambda: [_item(tmp_path, "gone", {"sessions": ["s"]})])
    found = REC.drift()
    assert [f["slug"] for f in found["missing_tree"]] == ["gone"]


def test_tree_without_a_record_is_reported(tmp_path, monkeypatch):
    (tmp_path / "stray").mkdir()
    monkeypatch.setattr(STORE, "list_workspaces", lambda: [_item(tmp_path, "stray", None)])
    found = REC.drift()
    assert [f["slug"] for f in found["unrecorded_tree"]] == ["stray"]


def test_dead_capacity_rows_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(STORE, "list_workspaces", lambda: [])
    monkeypatch.setattr(REC.capacity, "count_live_detail", lambda **_kw: (0, 3))
    found = REC.drift()
    assert found["dead_capacity_row"][0]["open_rows"] == 3


def test_a_pinned_child_is_flagged(tmp_path, monkeypatch):
    (tmp_path / "kid").mkdir()
    monkeypatch.setattr(
        STORE, "list_workspaces",
        lambda: [_item(tmp_path, "kid", {"sessions": ["s"], "purpose": f"{STORE.CHILD_PURPOSE_PREFIX}p"})],
    )
    monkeypatch.setattr(REC.hostapi, "is_pinned_child", lambda path: True)
    assert [f["slug"] for f in REC.drift()["pinned_child"]] == ["kid"]


def test_report_writes_json_and_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(STORE, "list_workspaces", lambda: [_item(tmp_path, "gone", {"sessions": ["s"]})])
    monkeypatch.setattr(REC.db, "hermes_home", lambda: tmp_path)
    payload = REC.report(write=True)
    assert payload["total"] == 1
    assert (tmp_path / "plugin-data" / REC.db.PLUGIN_NAME / "reconcile.json").is_file()
