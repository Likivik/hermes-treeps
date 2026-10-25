"""The workspace prepare step.

A workspace is not usable the moment ``jj workspace add`` returns. Every
other harness for this job (workmux, Conductor) runs a prepare step right
after creating the workspace: copy the ``.env``, symlink the dependency
directories, run the install. Without it, every real session in this plugin
started with hand-written ``GRADLE_USER_HOME`` / ``GIT_DIR`` /
``GIT_WORK_TREE`` exports typed by the agent - which is exactly the kind of
thing that should happen once, in configuration, instead of every session.

This module is deliberately dumb and total:

* it never raises - a prepare failure is reported, not raised, because the
  workspace on disk is still there and the operator can fix it by hand;
* a failing prepare never deletes the workspace;
* the command runs with ``cwd`` = the workspace, always, and never anywhere
  else - the resolved cwd is compared against the resolved workspace root
  before the process is started.

The prepare command is OPERATOR-CONFIGURED and runs arbitrary code by
design. That is why it is off by default (``prepare_command`` is empty),
behind ``prepare_enabled``, and documented in the README under "Prepare".
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# How much of the command's combined output is kept.
OUTPUT_TAIL_LIMIT = 4000


def _settings_value(settings: Mapping[str, Any], key: str, default: Any) -> Any:
    """A setting, tolerating a dict that does not have it."""
    try:
        value = settings.get(key, default)
    except AttributeError:
        return default
    return default if value is None else value


def _within(child: Path, parent: Path) -> bool:
    """True when ``child`` is ``parent`` or lives under it."""
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _is_ignored(workspace: Path, rel: str) -> bool:
    """True when git would ignore ``rel`` inside this workspace.

    WHY this exists: prepare copies the repo's real ``.env`` into a checkout
    that a child agent will work in. A child that runs ``git add -A`` would
    then commit the operator's secrets. So a file is only copied when git
    already ignores it there - otherwise it is skipped and reported, because
    a missing .env is a fixable inconvenience and a committed one is not.
    """
    env = dict(os.environ)
    if not (workspace / ".git").exists():
        # jj workspaces carry only .jj: the shared .git lives in the MAIN
        # workspace, so plain `git -C <ws>` dies with "not a git repository"
        # and every file would be skipped. `jj root` in a subworkspace returns
        # THAT workspace, not the main one - but `.jj/repo` is a relative
        # symlink to the main workspace's repo dir, and for a colocated repo
        # the git dir sits next to it. Point git at it with the child
        # workspace as the work tree; check-ignore evaluates the same rules
        # without needing a repo in the workspace itself.
        try:
            repo_link = (workspace / ".jj" / "repo")
            if repo_link.exists():
                # a plain file holding a RELATIVE path to the main workspace's
                # .jj/repo (not a symlink) - resolve it against .jj/
                target = Path(repo_link.read_text().strip())
                main_repo = (workspace / ".jj" / target).resolve()
                git_dir = main_repo.parent.parent / ".git"
                if git_dir.exists():
                    env["GIT_DIR"] = str(git_dir)
                    env["GIT_WORK_TREE"] = str(workspace)
        except Exception:
            return False  # cannot even locate the repo: refuse the copy
    try:
        proc = subprocess.run(
            ["git", "-C", str(workspace), "check-ignore", "-q", rel],
            capture_output=True, timeout=20, env=env,
        )
    except Exception:
        return False  # no git, no timeout budget: refuse the copy
    return proc.returncode == 0


def _copy_files(
    repo_root: Path, workspace: Path, files: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Copy the requested repo-relative files into the workspace.

    Returns ``(copied, skipped, errors)``. A file that does not exist in the
    repo is SKIPPED with a note, not an error - a repo without a ``.env`` is
    normal. An existing file in the workspace is never overwritten.
    """
    copied: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    for rel in files:
        rel = str(rel or "").strip()
        if not rel:
            continue
        src = (repo_root / rel).resolve()
        # A file outside the repo root is never copied - the setting is
        # repo-relative by contract.
        if not _within(src, repo_root):
            skipped.append(f"{rel}: outside the repo root")
            continue
        if not src.exists():
            skipped.append(f"{rel}: not present in the repo")
            continue
        if not src.is_file():
            skipped.append(f"{rel}: not a regular file")
            continue
        dst = workspace / rel
        if dst.exists():
            skipped.append(f"{rel}: already present in the workspace, not overwritten")
            continue
        if not _is_ignored(workspace, rel):
            # Copying a file git does not ignore turns the operator's .env into
            # a commit waiting to happen (one `git add -A` in the child).
            skipped.append(f"{rel}: not gitignored in this workspace, not copied")
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
        except OSError as exc:
            errors.append(f"{rel}: {type(exc).__name__}: {exc}")
    return copied, skipped, errors


def _run_command(
    command: str, workspace: Path, timeout: int, workspace_root: Path | None = None
) -> dict[str, Any]:
    """Run the operator's command inside the workspace.

    The cwd is resolved and compared against the resolved workspace root
    BEFORE the process starts: a prepare command must never run anywhere
    else, whatever the caller passed in. When the caller knows the
    configured ``workspace_root``, the resolved workspace must live UNDER
    it - a workspace outside the root is refused, not prepared.
    """
    result: dict[str, Any] = {"ran": False, "exit_code": None, "output_tail": ""}
    resolved = workspace.resolve()
    if workspace_root is not None:
        root = Path(workspace_root).resolve()
        if not _within(resolved, root):
            result["output_tail"] = (
                f"refused: {resolved} is not inside the workspace root {root}"
            )
            return result
    try:
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(resolved),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        result["output_tail"] = f"could not start the prepare command: {type(exc).__name__}: {exc}"
        return result

    result["ran"] = True
    try:
        out, _ = proc.communicate(timeout=timeout)
        result["exit_code"] = proc.returncode
        result["output_tail"] = (out or "")[-OUTPUT_TAIL_LIMIT:]
    except subprocess.TimeoutExpired:
        # Kill the whole process group: a prepare command that spawns
        # children must not leave them running after the timeout.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            proc.kill()
        try:
            out, _ = proc.communicate(timeout=10)
        except Exception:
            out = ""
        result["exit_code"] = None
        result["output_tail"] = (
            f"timed out after {timeout}s\n" + (out or "")[-OUTPUT_TAIL_LIMIT:]
        ).strip()
    return result


def run_prepare(
    repo_path: str | os.PathLike[str],
    workspace_path: str | os.PathLike[str],
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare a freshly created workspace.

    Returns a dict::

        {
            "ok": bool,
            "copied": [str, ...],
            "skipped": [str, ...],
            "command": {"ran": bool, "exit_code": int | None, "output_tail": str},
            "duration_s": float,
            "error": str,        # only when something went wrong
        }

    Never raises: any failure comes back as ``ok=False`` plus the reason.
    A failing prepare does NOT delete the workspace - the operator can fix
    it by hand and re-run.
    """
    started = time.monotonic()
    cfg: Mapping[str, Any] = settings or {}
    result: dict[str, Any] = {
        "ok": True,
        "copied": [],
        "skipped": [],
        "command": {"ran": False, "exit_code": None, "output_tail": ""},
        "duration_s": 0.0,
    }

    def _finish() -> dict[str, Any]:
        result["duration_s"] = round(time.monotonic() - started, 3)
        return result

    enabled = bool(_settings_value(cfg, "prepare_enabled", True))
    command = str(_settings_value(cfg, "prepare_command", "") or "")
    files_raw = _settings_value(cfg, "prepare_files", [".env"])
    files = [str(f) for f in files_raw] if isinstance(files_raw, list) else []
    try:
        timeout = int(_settings_value(cfg, "prepare_timeout_seconds", 300))
    except (TypeError, ValueError):
        timeout = 300
    timeout = max(1, min(3600, timeout))

    repo_root = Path(repo_path)
    workspace = Path(workspace_path)

    try:
        if not enabled:
            result["skipped"].append("prepare: disabled by prepare_enabled")
            return _finish()
        if not repo_root.is_dir():
            result["ok"] = False
            result["error"] = f"the repo root {repo_root} is not a directory"
            return _finish()
        if not workspace.is_dir():
            result["ok"] = False
            result["error"] = f"the workspace {workspace} is not a directory"
            return _finish()

        resolved_repo = repo_root.resolve()
        resolved_ws = workspace.resolve()
        # WHY the guard is not a self-comparison: ``_within(resolved_ws,
        # resolved_ws)`` is always True, so the guard could never fire and
        # a workspace from a DIFFERENT repo dir was prepared as if it
        # belonged to this one - with this repo's ``.env`` copied into it
        # and this repo's prepare command run in it.
        #
        # WHY the boundary is the REPO DIR and not the repo root: a
        # workspace is a SIBLING of its repo, not a child of it. The
        # layout is ``<repo_dir>/<repo>`` for the checkout and
        # ``<repo_dir>/.workspaces/<repo>/<slug>`` for the workspace, so
        # comparing the workspace against the repo root would refuse
        # every legitimate workspace. The repo dir is the boundary the
        # plugin owns: everything the plugin creates lives under it, and
        # a workspace outside it is not this plugin's workspace.
        #
        # The repo dir is resolved through the plugin's own settings, and
        # a settings failure falls back to the repo's own parent - so the
        # guard is always decidable, and never silently off.
        try:
            from . import store as _store

            resolved_root = _store.repo_dir().resolve()
        except Exception:
            resolved_root = resolved_repo.parent
        if not _within(resolved_ws, resolved_root):
            result["ok"] = False
            result["error"] = (
                f"the workspace path {workspace} is outside the plugin's repo dir "
                f"{resolved_root}"
            )
            return _finish()

        copied, skipped, errors = _copy_files(resolved_repo, resolved_ws, files)
        result["copied"] = copied
        result["skipped"].extend(skipped)
        if errors:
            result["ok"] = False
            result["error"] = "; ".join(errors)

        if command.strip():
            root_setting = cfg.get("workspace_root")
            root = Path(str(root_setting)) if root_setting else None
            result["command"] = _run_command(command, resolved_ws, timeout, root)
            exit_code = result["command"].get("exit_code")
            if result["command"].get("ran") and exit_code != 0:
                # Reported, but the workspace stays: a non-zero exit is a
                # thing the operator can fix by hand.
                result["ok"] = False
                result["error"] = (
                    f"the prepare command failed (exit {exit_code})"
                    if exit_code is not None
                    else "the prepare command timed out"
                )
        return _finish()
    except Exception as exc:  # never raise into the caller
        logger.debug("prepare failed", exc_info=True)
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        return _finish()


def prepare_for_new(repo_name: str, workspace_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Prepare a workspace that ``ws_new`` has just created.

    This is the convenience entry point for the wiring in ``__init__.py``:
    it looks the repo up and reads the live settings, so the caller needs
    one line, not five.
    """
    from . import store

    try:
        repo = store.find_repo(str(repo_name))
    except Exception as exc:
        return {
            "ok": False,
            "copied": [],
            "skipped": [],
            "command": {"ran": False, "exit_code": None, "output_tail": ""},
            "duration_s": 0.0,
            "error": f"could not find the repo {repo_name!r}: {type(exc).__name__}: {exc}",
        }
    return run_prepare(repo, workspace_path, store.settings())


def with_prepare(
    response: dict[str, Any],
    repo_name: str,
    workspace_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Add the prepare result to a ``ws_new`` response, in place.

    The wiring in ``__init__.py`` is then a single expression::

        return json.dumps(prepare.with_prepare(
            {"ok": True, "path": bound_path, "vcs": record["vcs"]}, repo, bound_path
        ))

    Never raises: a broken prepare must not turn a successful workspace
    creation into a failed one.
    """
    try:
        response["prepare"] = prepare_for_new(repo_name, workspace_path)
    except Exception as exc:
        logger.debug("prepare failed while creating the workspace", exc_info=True)
        response["prepare"] = {
            "ok": False,
            "copied": [],
            "skipped": [],
            "command": {"ran": False, "exit_code": None, "output_tail": ""},
            "duration_s": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return response


def _tool_prepare(args: dict[str, Any], **kw: Any) -> str:
    """The ``ws_prepare`` tool: re-run the prepare step on a workspace."""
    import json

    from . import store

    try:
        path = str(args.get("path") or "").strip()
        if not path:
            workspace = store.resolve_detail(kw.get("session_id", "")) or {}
            path = str(workspace.get("path") or "")
        if not path:
            return json.dumps({"ok": False, "error": "no workspace path given or bound"})
        ws = Path(path)
        record = store.read_record(ws) or {}
        repo_name = str(record.get("repo") or "")
        repo = store.find_repo(repo_name) if repo_name else ws.parent
        return json.dumps(run_prepare(repo, ws, store.settings()))
    except Exception as exc:
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def register(ctx: Any) -> None:
    """Register the ``ws_prepare`` tool.

    Wired: the plugin's ``register()`` in ``__init__.py`` calls this next to
    ``picker.register(ctx)``; the manifest declares the ``ws_prepare`` tool.
    """
    try:
        ctx.register_tool(
            name="ws_prepare",
            toolset="workspaces",
            schema={
                "name": "ws_prepare",
                "description": (
                    "Run the workspace prepare step again: copy the configured files "
                    "(e.g. .env) into the workspace and run the prepare command. "
                    "Use it when a workspace was created without it, or after fixing "
                    "a failed prepare."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "workspace path; defaults to this session's workspace",
                        }
                    },
                },
            },
            handler=_tool_prepare,
            emoji="🧰",
        )
    except Exception:  # a host without tool registration must not break the plugin
        logger.debug("could not register the ws_prepare tool", exc_info=True)
