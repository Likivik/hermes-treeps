"""The workspace prepare step.

WHY these tests exist: the prepare step runs OPERATOR-CONFIGURED CODE. The
guard rails are the whole point of the feature - copy only what exists,
never overwrite, never run when disabled, never run outside the workspace,
and always report a failure instead of raising (a broken prepare must not
destroy a workspace or take the session down).

No network, no jj, no Hermes install: tmp dirs and the stdlib only.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest
from plugin_loader import PLUGIN

PREPARE = importlib.import_module("hermes_treeps.prepare")


def _settings(**over):
    base = {
        "prepare_enabled": True,
        "prepare_command": "",
        "prepare_files": [".env"],
        "prepare_timeout_seconds": 300,
    }
    base.update(over)
    return base


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    (r / ".env").write_text("TOKEN=secret\n")
    return r


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """A git workspace that ignores .env.

    git matters: prepare only copies a file git already ignores there, because
    a copied .env in a checkout is one `git add -A` away from being committed.
    """
    ws = tmp_path / "workspace"
    ws.mkdir()
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    (ws / ".gitignore").write_text(".env\n")
    return ws


def test_prepare_copies_an_existing_env_file(repo, workspace):
    out = PREPARE.run_prepare(repo, workspace, _settings())
    assert out["ok"] is True, out
    assert out["copied"] == [".env"], out
    assert (workspace / ".env").read_text() == "TOKEN=secret\n"


def test_prepare_skips_a_missing_file_without_failing(repo, workspace):
    out = PREPARE.run_prepare(repo, workspace, _settings(prepare_files=[".env", ".env.local"]))
    assert out["ok"] is True, out
    assert out["copied"] == [".env"], out
    assert any("not present" in note for note in out["skipped"]), out
    assert not (workspace / ".env.local").exists()


def test_prepare_never_overwrites_an_existing_workspace_file(repo, workspace):
    (workspace / ".env").write_text("MINE=keep\n")
    out = PREPARE.run_prepare(repo, workspace, _settings())
    assert out["ok"] is True, out
    assert out["copied"] == [], out
    assert (workspace / ".env").read_text() == "MINE=keep\n"
    assert any("not overwritten" in note for note in out["skipped"]), out


def test_prepare_disabled_runs_nothing(repo, workspace):
    out = PREPARE.run_prepare(
        repo,
        workspace,
        _settings(prepare_enabled=False, prepare_command="touch SIDE_EFFECT"),
    )
    assert out["ok"] is True, out
    assert out["copied"] == [], out
    assert out["command"]["ran"] is False, out
    assert not (workspace / "SIDE_EFFECT").exists()
    assert any("disabled" in note for note in out["skipped"]), out


def test_a_failing_command_is_reported_and_leaves_the_workspace_intact(repo, workspace):
    out = PREPARE.run_prepare(
        repo,
        workspace,
        _settings(prepare_command="echo boom-please && exit 1"),
    )
    assert out["ok"] is False, out
    assert out["command"]["ran"] is True, out
    assert out["command"]["exit_code"] == 1, out
    assert "boom-please" in out["command"]["output_tail"], out
    # The workspace is still there - the operator fixes it by hand.
    assert workspace.is_dir()
    assert (workspace / ".env").exists()


def test_a_command_that_times_out_is_reported_and_killed(repo, workspace):
    out = PREPARE.run_prepare(
        repo,
        workspace,
        _settings(prepare_command="sleep 30", prepare_timeout_seconds=1),
    )
    assert out["ok"] is False, out
    assert out["command"]["ran"] is True, out
    assert "timed out" in out["command"]["output_tail"], out
    assert workspace.is_dir()


def test_the_command_always_runs_inside_the_workspace(repo, workspace):
    out = PREPARE.run_prepare(repo, workspace, _settings(prepare_command="pwd"))
    assert out["ok"] is True, out
    tail = out["command"]["output_tail"].strip()
    assert Path(tail).resolve() == workspace.resolve(), out


def test_a_file_outside_the_repo_root_is_never_copied(repo, workspace, tmp_path):
    outside = tmp_path / "outside.env"
    outside.write_text("NOPE=1\n")
    out = PREPARE.run_prepare(
        repo,
        workspace,
        _settings(prepare_files=[str(outside)]),
    )
    assert out["ok"] is True, out
    assert out["copied"] == [], out
    assert not (workspace / "outside.env").exists()
    assert any("outside the repo root" in note for note in out["skipped"]), out


def test_the_command_is_refused_outside_the_workspace_root(repo, workspace, tmp_path):
    root = tmp_path / "elsewhere"
    root.mkdir()
    out = PREPARE.run_prepare(
        repo,
        workspace,
        _settings(prepare_command="pwd", workspace_root=str(root)),
    )
    assert out["command"]["ran"] is False, out
    assert "not inside the workspace root" in out["command"]["output_tail"], out


def test_prepare_never_raises_on_a_missing_workspace(repo, tmp_path):
    out = PREPARE.run_prepare(repo, tmp_path / "nope", _settings())
    assert out["ok"] is False, out
    assert out["error"], out


def test_the_command_output_is_tail_capped(repo, workspace):
    out = PREPARE.run_prepare(
        repo,
        workspace,
        _settings(prepare_command="for i in $(seq 1 2000); do echo line$i; done"),
    )
    assert out["ok"] is True, out
    assert len(out["command"]["output_tail"]) <= PREPARE.OUTPUT_TAIL_LIMIT + 50


# ── the module is wired into the plugin ──────────────────────────────────────


def test_the_plugin_package_exposes_the_prepare_module():
    assert PLUGIN is not None
    assert hasattr(PREPARE, "run_prepare")
    assert hasattr(PREPARE, "prepare_for_new")


class _FakeCtx:
    """The smallest thing prepare.register(ctx) needs."""

    def __init__(self):
        self.tools = {}

    def register_tool(self, name, toolset, schema, handler, emoji=""):
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler}


def test_with_prepare_adds_a_prepare_key_without_raising(monkeypatch):
    """The ws_new wiring: one expression, and a broken prepare never raises."""
    monkeypatch.setattr(
        PREPARE,
        "prepare_for_new",
        lambda repo, ws: {"ok": True, "copied": [".env"], "skipped": [],
                          "command": {"ran": False, "exit_code": None, "output_tail": ""},
                          "duration_s": 0.1},
    )
    out = PREPARE.with_prepare({"ok": True, "path": "/ws/x"}, "repo", "/ws/x")
    assert out["ok"] is True
    assert out["prepare"]["copied"] == [".env"], out


def test_with_prepare_reports_a_broken_prepare_instead_of_raising(monkeypatch):
    def _boom(repo, ws):
        raise RuntimeError("nope")

    monkeypatch.setattr(PREPARE, "prepare_for_new", _boom)
    out = PREPARE.with_prepare({"ok": True, "path": "/ws/x"}, "repo", "/ws/x")
    assert out["ok"] is True, out
    assert out["prepare"]["ok"] is False, out
    assert "nope" in out["prepare"]["error"], out


def test_register_adds_the_ws_prepare_tool():
    ctx = _FakeCtx()
    PREPARE.register(ctx)
    assert "ws_prepare" in ctx.tools, ctx.tools
    spec = ctx.tools["ws_prepare"]
    assert spec["toolset"] == "workspaces"
    assert spec["schema"]["name"] == "ws_prepare"
    assert callable(spec["handler"])


def test_a_file_git_does_not_ignore_is_never_copied(repo, workspace):
    """A .env the workspace would COMMIT is refused, not copied.

    Prevents: prepare turning the operator's real .env into a staged secret in
    a child's checkout - one `git add -A` from being pushed.
    """
    (workspace / ".gitignore").write_text("node_modules/\n")  # .env is NOT ignored
    out = PREPARE.run_prepare(repo, workspace, _settings())
    assert out["ok"] is True, out
    assert out["copied"] == [], out
    assert not (workspace / ".env").exists(), "a committable .env was copied anyway"
    assert any("not gitignored" in s for s in out["skipped"]), out
