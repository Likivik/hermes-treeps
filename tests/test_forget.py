"""Tests for ``hermes ws forget``: targeted removal of a stale workspace record."""

from __future__ import annotations

import argparse
from pathlib import Path

from plugin_loader import CLI, STORE

CHILD_PREFIX = STORE.CHILD_PURPOSE_PREFIX


def _cli(action="forget", **kwargs):
    args = argparse.Namespace(ws_action=action, **kwargs)
    return CLI._cli(args)


def _record(path: Path, *, slug: str, repo: str = "demo", purpose: str = "", sessions=()):
    STORE.write_record(
        path,
        {
            "repo": repo,
            "slug": slug,
            "path": str(path),
            "vcs": "jj",
            "purpose": purpose,
            "sessions": list(sessions),
            "created_at": "2020-01-01T00:00:00",
        },
    )


def _workspace(tmp_path, *, slug="demo-stale", purpose="", create_tree=False):
    ws = tmp_path / slug
    if create_tree:
        ws.mkdir(parents=True)
    return ws


def test_forget_removes_a_stale_record_and_its_bindings(tmp_path, capsys):
    ws = _workspace(tmp_path)
    _record(ws, slug=ws.name, purpose=CHILD_PREFIX + "parent-1", sessions=("sid-a",))
    STORE.db.write_binding(
        "sid-a",
        {
            "mode": "repo",
            "repo": "demo",
            "slug": ws.name,
            "path": str(ws),
            "session_ended_at": "2020-01-02T00:00:00",
        },
    )

    rc = _cli(slug=ws.name, force=False)

    out = capsys.readouterr().out
    assert rc == 0, out
    assert ws.name in out and str(ws) in out, out
    assert STORE.db.read_workspace(str(ws)) is None
    assert STORE.db.read_binding("sid-a") is None


def test_forget_refuses_a_live_session(tmp_path, capsys):
    ws = _workspace(tmp_path, create_tree=True)
    _record(ws, slug=ws.name, sessions=("sid-live",))
    STORE.db.write_binding(
        "sid-live",
        {"mode": "repo", "repo": "demo", "slug": ws.name, "path": str(ws)},
    )

    rc = _cli(slug=ws.name, force=False)

    out = capsys.readouterr().out
    assert rc == 1, out
    assert "live" in out.lower(), out
    assert STORE.db.read_workspace(str(ws)) is not None
    assert STORE.db.read_binding("sid-live") is not None


def test_forget_refuses_a_tree_without_force(tmp_path, capsys):
    ws = _workspace(tmp_path, create_tree=True)
    _record(ws, slug=ws.name)

    rc = _cli(slug=ws.name, force=False)

    out = capsys.readouterr().out
    assert rc == 1, out
    assert "--force" in out, out
    assert STORE.db.read_workspace(str(ws)) is not None


def test_forget_with_force_retires_the_tree_and_removes_the_record(tmp_path, capsys, monkeypatch):
    ws = _workspace(tmp_path, create_tree=True)
    _record(ws, slug=ws.name, purpose=CHILD_PREFIX + "parent-1", sessions=("sid-a",))
    STORE.db.write_binding(
        "sid-a",
        {
            "mode": "repo",
            "repo": "demo",
            "slug": ws.name,
            "path": str(ws),
            "session_ended_at": "2020-01-02T00:00:00",
        },
    )
    calls: list[str] = []

    def _fake_retire(path, **kwargs):
        calls.append(str(path))
        return {"ok": True, "snapshot_commit_id": "abc123", "was_dirty": True}

    monkeypatch.setattr(STORE, "reclaim_child_workspace", _fake_retire)

    rc = _cli(slug=ws.name, force=True)

    out = capsys.readouterr().out
    assert rc == 0, out
    assert calls == [str(ws)], calls
    assert "abc123" in out, out
    assert STORE.db.read_workspace(str(ws)) is None
    assert STORE.db.read_binding("sid-a") is None


def test_forget_unknown_slug_lists_the_available_slugs(tmp_path, capsys):
    ws = _workspace(tmp_path)
    _record(ws, slug=ws.name)

    rc = _cli(slug="nope", force=False)

    out = capsys.readouterr().out
    assert rc == 1, out
    assert ws.name in out, out


def test_forget_refuses_a_user_class_record_without_force(tmp_path, capsys):
    ws = _workspace(tmp_path)
    _record(ws, slug=ws.name, purpose="user work")

    rc = _cli(slug=ws.name, force=False)

    out = capsys.readouterr().out
    assert rc == 1, out
    assert STORE.db.read_workspace(str(ws)) is not None
