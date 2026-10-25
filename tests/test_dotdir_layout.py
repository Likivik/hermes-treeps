"""The dot-dir layout: where a workspace path comes from, and the CLI.

Two things are pinned here:

* the ONE derivation of a workspace path (every caller goes through it), and
* the CLI that runs the one-time layout migration.
"""

from __future__ import annotations

import importlib

from plugin_loader import STORE

# ── the derivation: one function, one shape ──────────────────────────────────


def test_the_workspace_path_is_derived_from_the_repo_dir(tmp_path, monkeypatch):
    """A workspace lives at ``<repo_dir>/.workspaces/<repo>/<slug>``.

    WHY the dot-dir: the workspaces sit next to the repos they belong to,
    in a directory a plain listing skips - so a workspace can never be
    mistaken for a repo, and the repo scanner never walks into one.
    """
    repo_dir = tmp_path / "repos"
    repo_dir.mkdir()
    monkeypatch.setattr(STORE, "repo_dir", lambda: repo_dir)

    assert STORE.workspaces_base() == repo_dir / ".workspaces"
    assert STORE.ws_path("demo", "demo-abc123") == repo_dir / ".workspaces" / "demo" / "demo-abc123"


def test_the_root_dir_and_the_workspace_base_are_the_same_place(tmp_path, monkeypatch):
    """root_dir() and workspaces_base() name the same directory.

    WHY they must agree: root_dir() is the historical name, read by the
    gate, the gc tiers and the CLI. If the two could differ, one of them
    would look in the wrong place - and the wrong place is "no workspaces
    found", which reads like "there is nothing to reclaim".
    """
    repo_dir = tmp_path / "repos"
    repo_dir.mkdir()
    monkeypatch.setattr(STORE, "repo_dir", lambda: repo_dir)

    assert STORE.root_dir() == STORE.workspaces_base()


def test_the_slug_shape_did_not_change(tmp_path, monkeypatch):
    """The slug format is unchanged by the layout change.

    WHY it matters: the slug is what the record, the directory and the
    session binding agree on. Changing the layout must not change the
    naming, or every existing record would need renaming too.
    """
    repo_dir = tmp_path / "repos"
    repo_dir.mkdir()
    monkeypatch.setattr(STORE, "repo_dir", lambda: repo_dir)

    parent = STORE.ws_path("demo", "demo-abc123")
    child = STORE.ws_path("demo", "demo-abc123-child-004210")
    assert parent.name == "demo-abc123"
    assert child.name == "demo-abc123-child-004210"
    assert parent.parent.name == "demo"
    assert child.parent.name == "demo"


def test_the_slug_is_sanitised_the_same_way(tmp_path, monkeypatch):
    """sanitize_slug() is unchanged - the layout does not touch naming."""
    assert STORE.sanitize_slug("Fix The Widget") == "fix-the-widget"


def test_the_prepare_step_accepts_a_workspace_in_the_new_layout(tmp_path, monkeypatch):
    """The prepare command runs inside a workspace under the DERIVED root.

    WHY: the prepare step refuses to run its command outside the workspace
    root. After the layout change that root is the derived one, so a
    workspace in the new layout has to be accepted - otherwise every
    workspace created after the change would silently lose its prepare
    step.
    """

    prepare = importlib.import_module("hermes_treeps.prepare")
    repo_dir = tmp_path / "repos"
    repo_dir.mkdir()
    monkeypatch.setattr(STORE, "repo_dir", lambda: repo_dir)

    repo = repo_dir / "demo"
    repo.mkdir()
    ws = STORE.ws_path("demo", "demo-abc123")
    ws.mkdir(parents=True)

    out = prepare.run_prepare(
        repo,
        ws,
        {
            "prepare_enabled": True,
            "prepare_command": "pwd",
            "prepare_files": [],
            "prepare_timeout_seconds": 30,
            "workspace_root": str(STORE.root_dir()),
        },
    )
    assert out["command"]["ran"] is True, out
    assert "not inside the workspace root" not in out["command"]["output_tail"], out

