"""``ws list`` must describe what the orphan flag actually means.

The old label said ``ORPHAN(no session.json)`` while the code checked the
DATABASE record (``store.list_workspaces`` → ``db.read_workspace``). Two
real states were invisible: a record that exists but has no bound sessions
(the post-migration shape), and the label pointed at a file nothing reads.
"""

from __future__ import annotations

import argparse

from plugin_loader import CLI, STORE


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
    return root


def _list(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(STORE, "root_dir", lambda: tmp_path / "repos" / ".workspaces")
    rc = CLI._cli(argparse.Namespace(ws_action="list"))
    out = capsys.readouterr().out
    assert rc == 0
    return out


def test_dir_without_record_is_flagged_as_no_db_record(tmp_path, monkeypatch, capsys):
    root = _mk_root(tmp_path, monkeypatch)
    ws = root / "repo" / "ghost"
    ws.mkdir(parents=True)

    out = _list(tmp_path, monkeypatch, capsys)
    assert "ORPHAN(no db record)" in out
    assert "session.json" not in out


def test_record_without_sessions_is_unbound_not_orphan(tmp_path, monkeypatch, capsys):
    root = _mk_root(tmp_path, monkeypatch)
    ws = root / "repo" / "pre-upgrade"
    ws.mkdir(parents=True)
    STORE.write_record(
        ws,
        {
            "repo": "repo",
            "slug": "pre-upgrade",
            "path": str(ws),
            "vcs": "jj",
            "purpose": "old-format record",
            # no "sessions" key: the post-migration shape gc also catches
        },
    )

    out = _list(tmp_path, monkeypatch, capsys)
    line = next(ln for ln in out.splitlines() if "pre-upgrade" in ln)
    assert "UNBOUND(no bound sessions)" in line
    assert "ORPHAN" not in line
