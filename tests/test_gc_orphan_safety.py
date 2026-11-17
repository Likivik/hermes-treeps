"""gc must not treat orphans as a blanket-reclaimable class.

Measured on the real fleet (2026-11-17): of 67 orphans, 21 had dirty trees
with real uncommitted work and 10 had been touched within days — yet the
orphan branch put ALL of them in tier2 without looking at the tree. Worse, a
live session was working inside an unregistered tree (coppertally/console)
that gc had no way to see. The orphan path now runs the same evidence checks
tier1 uses, and anything unproven lands in tier4 ("needs attention"), which
``gc --apply`` never touches.
"""

from __future__ import annotations

import os
import time

from plugin_loader import STORE

FRESH_DAYS = 7.0


def _mk_root(tmp_path, monkeypatch):
    root = tmp_path / "repos" / ".workspaces"
    root.mkdir(parents=True)
    monkeypatch.setattr(STORE, "repo_dir", lambda: tmp_path / "repos")
    monkeypatch.setattr(
        STORE,
        "settings",
        lambda: {
            "repo_dir": str(tmp_path / "repos"),
            "workspace_root": str(root),
            "repo_roots": [str(tmp_path)],
        },
    )
    monkeypatch.setattr(STORE, "open_prs", lambda path: ([], True))
    monkeypatch.setattr(STORE, "_child_workspace_records", lambda: [])
    return root


def _orphan(root, slug):
    ws = root / "repo" / slug
    ws.mkdir(parents=True)
    return ws


def _tiers():
    tiers = STORE.gc_tiers()
    slugs = {t: [i["slug"] for i in tiers[t]] for t in tiers}
    whys = {i["slug"]: i["why"] for t in tiers.values() for i in t}
    return slugs, whys


def test_dirty_orphan_lands_in_needs_attention(tmp_path, monkeypatch):
    root = _mk_root(tmp_path, monkeypatch)
    ws = _orphan(root, "dirty-orphan")
    monkeypatch.setattr(STORE, "dirty", lambda path: True)
    monkeypatch.setattr(STORE, "unpushed_commits", lambda path: False)

    tiers, whys = _tiers()
    assert ws.name not in tiers["tier2"]
    assert ws.name in tiers["tier4"]
    assert "dirty" in whys[ws.name]


def test_unpushed_orphan_lands_in_needs_attention(tmp_path, monkeypatch):
    root = _mk_root(tmp_path, monkeypatch)
    ws = _orphan(root, "unpushed-orphan")
    monkeypatch.setattr(STORE, "dirty", lambda path: False)
    monkeypatch.setattr(STORE, "unpushed_commits", lambda path: True)

    tiers, whys = _tiers()
    assert ws.name in tiers["tier4"]
    assert "unpushed" in whys[ws.name]


def test_unverifiable_orphan_lands_in_needs_attention(tmp_path, monkeypatch):
    root = _mk_root(tmp_path, monkeypatch)
    ws = _orphan(root, "opaque-orphan")
    monkeypatch.setattr(STORE, "dirty", lambda path: False)
    monkeypatch.setattr(STORE, "unpushed_commits", lambda path: None)

    tiers, whys = _tiers()
    assert ws.name in tiers["tier4"]
    assert "could not verify" in whys[ws.name]


def test_recent_orphan_lands_in_needs_attention(tmp_path, monkeypatch):
    root = _mk_root(tmp_path, monkeypatch)
    ws = _orphan(root, "fresh-orphan")
    monkeypatch.setattr(STORE, "dirty", lambda path: False)
    monkeypatch.setattr(STORE, "unpushed_commits", lambda path: False)

    tiers, whys = _tiers()
    assert ws.name in tiers["tier4"]
    assert "touched" in whys[ws.name]


def test_old_clean_orphan_stays_in_tier2(tmp_path, monkeypatch):
    root = _mk_root(tmp_path, monkeypatch)
    ws = _orphan(root, "old-clean-orphan")
    old = time.time() - (FRESH_DAYS + 5) * 86400
    os.utime(ws, (old, old))
    monkeypatch.setattr(STORE, "dirty", lambda path: False)
    monkeypatch.setattr(STORE, "unpushed_commits", lambda path: False)

    tiers, whys = _tiers()
    assert ws.name in tiers["tier2"]
    assert "clean" in whys[ws.name]
    assert ws.name not in tiers["tier4"]


def test_gc_apply_never_touches_tier4(tmp_path, monkeypatch, capsys):
    root = _mk_root(tmp_path, monkeypatch)
    ws = _orphan(root, "dirty-orphan")
    (ws / "draft.md").write_text("real uncommitted work\n")
    monkeypatch.setattr(STORE, "dirty", lambda path: True)
    monkeypatch.setattr(STORE, "unpushed_commits", lambda path: False)
    retired = []

    monkeypatch.setattr(STORE, "retire", lambda slug, **kw: retired.append(slug) or "ok")

    from plugin_loader import CLI

    rc = CLI._cli(__import__("argparse").Namespace(ws_action="gc", apply=True, children=False))
    out = capsys.readouterr().out
    assert rc == 0
    assert retired == [], "gc --apply must never retire a needs-attention workspace"
    assert ws.exists(), "the dirty tree must survive"
    assert "needs attention" in out
