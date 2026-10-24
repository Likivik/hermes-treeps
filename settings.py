"""Config-driven settings for hermes-treeps.

ONE accessor: :func:`settings`. It reads
``plugins.entries.hermes-treeps.settings.<key>`` from ``config.yaml`` (through
the host's read-only loader), applies environment overrides, fills in the
defaults that preserve the plugin's original behaviour, and validates the
result. Every key it reads is declared in ``plugin.yaml``'s ``config_schema``
so the Desktop settings form and the code cannot drift apart.

There is deliberately NO fallback to the old top-level
``plugins.entries.hermes-treeps.<key>`` keys: there is exactly one install of
this plugin, so a legacy reader would be dead code.

The Forgejo token is a SECRET and never lives in config.yaml. It is declared
in the manifest with ``type: secret`` (which is where the host stores it, in
``.env``) and resolved at runtime through ``agent.secret_scope``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

from .errors import WorkspaceError

logger = logging.getLogger(__name__)

PLUGIN_ID = "hermes-treeps"

# The .env name of the Forgejo token. The manifest declares it as
# `forgejo_token: {type: secret, env: FORGEJO_TOKEN}`, so the Desktop secret
# field writes HERE and forgejo_token() reads HERE. The host otherwise derives
# HERMES_TREEPS_<KEY> from the plugin id, which nothing read - the token was
# always empty.
FORGEJO_TOKEN_ENV = "FORGEJO_TOKEN"

# The name the Desktop secret field used before the manifest declared `env:`.
# Kept as a fallback so an already-written .env entry keeps working.
FORGEJO_TOKEN_ENV_LEGACY = "HERMES_TREEPS_FORGEJO_TOKEN"

# The keys settings() reads. The test suite compares this set with the
# ``config_schema`` in plugin.yaml - they must be identical.
SETTINGS_KEYS: tuple[str, ...] = (
    "workspace_root",
    "repo_roots",
    "clone_root",
    "repo_sources",
    "vcs",
    "extra_bin_dirs",
    "picker_enabled",
    "allowed_user_ids",
    "forgejo_url",
    "prepare_enabled",
    "prepare_command",
    "prepare_files",
    "prepare_timeout_seconds",
)

# Keys whose schema entry is a SECRET: they live in .env, never in config.yaml.
# They are NOT read through settings() (a secret must not travel through a
# config dict), but they ARE declared in the manifest so the Desktop form can
# render them.
SECRET_KEYS: tuple[str, ...] = ("forgejo_token",)

DEFAULTS: dict[str, Any] = {
    "workspace_root": "/Git/hermes/workspaces",
    "repo_roots": ["/Git", "/Git/hermes", "/Git/likivik", "/Git/repos"],
    "clone_root": "/Git/hermes",
    "repo_sources": ["local", "github", "forgejo"],
    "vcs": "auto",
    "extra_bin_dirs": ["/run/current-system/sw/bin", "/usr/local/bin", "/usr/bin"],
    "picker_enabled": True,
    "allowed_user_ids": [],
    # forgejo_token is deliberately NOT here: it is a SECRET, read through
    # get_secret() below, and a secret must never travel through the config
    # dict (the manifest declares it separately with `env: FORGEJO_TOKEN`).
    "forgejo_url": "",
    # The prepare step: a workspace is not usable the moment `jj workspace
    # add` returns. workmux and Conductor both run a prepare step (copy
    # .env, symlink dependencies, run installs) - we did not, which is why
    # every real session started with hand-written GRADLE_USER_HOME /
    # GIT_DIR / GIT_WORK_TREE exports.
    "prepare_enabled": True,
    # Operator-configured shell command, run with `bash -lc` INSIDE the new
    # workspace. Empty means no command runs at all. It runs arbitrary code
    # by design - see README, "Prepare".
    "prepare_command": "",
    # Repo-relative files copied into the workspace when it exists. A missing
    # file is skipped with a note, never an error.
    "prepare_files": [".env"],
    # Hard ceiling on the prepare command.
    "prepare_timeout_seconds": 300,
    # The verification command: run in the CHILD's workspace to find out
}

ALLOWED_SOURCES = ("local", "github", "forgejo")
ALLOWED_VCS = ("auto", "jj", "git")

# Environment overrides: ENV_VAR -> settings key.
ENV_OVERRIDES: dict[str, str] = {
    "HERMES_TREEPS_WORKSPACE_ROOT": "workspace_root",
    "HERMES_WS_ROOT": "workspace_root",  # the original override name
    "HERMES_TREEPS_REPO_ROOTS": "repo_roots",
    "HERMES_TREEPS_CLONE_ROOT": "clone_root",
    "HERMES_TREEPS_REPO_SOURCES": "repo_sources",
    "HERMES_TREEPS_VCS": "vcs",
    "HERMES_TREEPS_EXTRA_BIN_DIRS": "extra_bin_dirs",
    "HERMES_TREEPS_PICKER_ENABLED": "picker_enabled",
    "HERMES_TREEPS_ALLOWED_USER_IDS": "allowed_user_ids",
    "HERMES_TREEPS_FORGEJO_URL": "forgejo_url",
    "FORGEJO_URL": "forgejo_url",
}

_LIST_SEPARATORS = (":", ",")


class SettingsError(WorkspaceError):
    """A bad configuration value, named by key."""


def _load_config() -> dict[str, Any]:
    """The host's config.yaml, or {} when the host is not around (tests, CI)."""
    try:
        from hermes_cli.config import load_config_readonly  # type: ignore

        cfg = load_config_readonly() or {}
        return dict(cfg) if isinstance(cfg, Mapping) else {}
    except Exception:
        logger.debug("config.yaml is not readable; using defaults", exc_info=True)
        return {}


def _settings_entry(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``plugins.entries.hermes-treeps.settings`` - and nothing else."""
    cfg = _load_config() if config is None else dict(config)
    plugins = cfg.get("plugins")
    entries = plugins.get("entries") if isinstance(plugins, Mapping) else None
    entry = entries.get(PLUGIN_ID) if isinstance(entries, Mapping) else None
    raw = entry.get("settings") if isinstance(entry, Mapping) else None
    return dict(raw) if isinstance(raw, Mapping) else {}


def _split_list(raw: str) -> list[str]:
    parts = [raw]
    for sep in _LIST_SEPARATORS:
        out: list[str] = []
        for part in parts:
            out.extend(part.split(sep))
        parts = out
    return [p.strip() for p in parts if p.strip()]


def _env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for var, key in ENV_OVERRIDES.items():
        raw = os.environ.get(var)
        if raw is None or not raw.strip():
            continue
        if key in ("repo_roots", "repo_sources", "extra_bin_dirs"):
            out[key] = _split_list(raw)
        elif key == "allowed_user_ids":
            out[key] = [int(x) for x in _split_list(raw) if x.lstrip("-").isdigit()]
        elif key == "picker_enabled":
            out[key] = raw.strip().lower() in ("1", "true", "yes", "on")
        else:
            out[key] = raw.strip()
    return out


def _validate(values: dict[str, Any]) -> dict[str, Any]:
    """Fail loudly, naming the key, on a value that cannot work."""
    out = dict(values)

    def _abs_path(key: str) -> None:
        val = out.get(key)
        if not isinstance(val, str) or not val.strip():
            raise SettingsError(f"hermes-treeps settings.{key}: must be a non-empty path")
        if not val.startswith("/"):
            raise SettingsError(
                f"hermes-treeps settings.{key}: must be an absolute path, got {val!r}"
            )
        out[key] = val

    _abs_path("workspace_root")
    _abs_path("clone_root")

    roots = out.get("repo_roots")
    if not isinstance(roots, list) or not roots:
        raise SettingsError("hermes-treeps settings.repo_roots: must be a non-empty list of paths")
    for r in roots:
        if not isinstance(r, str) or not r.startswith("/"):
            raise SettingsError(
                f"hermes-treeps settings.repo_roots: every entry must be an absolute path, got {r!r}"
            )
    if not isinstance(out.get("extra_bin_dirs"), list):
        raise SettingsError("hermes-treeps settings.extra_bin_dirs: must be a list of paths")

    sources = out.get("repo_sources")
    if not isinstance(sources, list) or not sources:
        raise SettingsError("hermes-treeps settings.repo_sources: must be a non-empty list")
    for s in sources:
        if s not in ALLOWED_SOURCES:
            raise SettingsError(
                f"hermes-treeps settings.repo_sources: unknown source {s!r} "
                f"(allowed: {', '.join(ALLOWED_SOURCES)})"
            )

    vcs = out.get("vcs")
    if vcs not in ALLOWED_VCS:
        raise SettingsError(
            f"hermes-treeps settings.vcs: must be one of {', '.join(ALLOWED_VCS)}, got {vcs!r}"
        )

    if not isinstance(out.get("picker_enabled"), bool):
        raise SettingsError("hermes-treeps settings.picker_enabled: must be true or false")

    ids = out.get("allowed_user_ids")
    if not isinstance(ids, list):
        raise SettingsError("hermes-treeps settings.allowed_user_ids: must be a list of user ids")
    out["allowed_user_ids"] = [int(i) for i in ids]

    if not isinstance(out.get("forgejo_url"), str):
        raise SettingsError("hermes-treeps settings.forgejo_url: must be a string")

    # ── the prepare step ───────────────────────────────────────────────────
    if not isinstance(out.get("prepare_enabled"), bool):
        raise SettingsError("hermes-treeps settings.prepare_enabled: must be true or false")

    if not isinstance(out.get("prepare_command"), str):
        raise SettingsError("hermes-treeps settings.prepare_command: must be a string")

    files = out.get("prepare_files")
    if not isinstance(files, list):
        raise SettingsError("hermes-treeps settings.prepare_files: must be a list of paths")
    for f in files:
        if not isinstance(f, str) or not f.strip():
            raise SettingsError(
                "hermes-treeps settings.prepare_files: every entry must be a non-empty path"
            )

    timeout = out.get("prepare_timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int):
        raise SettingsError(
            "hermes-treeps settings.prepare_timeout_seconds: must be an integer number of seconds"
        )
    if not 1 <= timeout <= 3600:
        raise SettingsError(
            "hermes-treeps settings.prepare_timeout_seconds: must be between 1 and 3600, "
            f"got {timeout}"
        )

    return out


def settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The plugin's effective settings.

    Precedence: environment override > ``config.yaml``
    (``plugins.entries.hermes-treeps.settings``) > built-in default.

    Raises :class:`SettingsError` (a ``WorkspaceError``) naming the key when a
    value cannot work - a bad path or an unknown discovery source is a
    configuration bug, and silently falling back would hide it.
    """
    values: dict[str, Any] = dict(DEFAULTS)
    values.update(_settings_entry(config))
    values.update(_env_overrides())
    return _validate(values)


def forgejo_token() -> str:
    """The Forgejo API token, from the host's secret scope.

    Secrets live in .env and are resolved by ``agent.secret_scope``; the plain
    environment is only a fallback for hosts that do not ship the scope.

    The name ``FORGEJO_TOKEN`` is the one the manifest declares
    (``forgejo_token: {type: secret, env: FORGEJO_TOKEN}``), so the Desktop
    secret field and this reader agree. Without that ``env:`` the host derived
    ``HERMES_TREEPS_FORGEJO_TOKEN`` for the Desktop field while this function
    read ``FORGEJO_TOKEN`` - the two never met and the token was always empty.

    ``HERMES_TREEPS_FORGEJO_TOKEN`` (the name the Desktop form used to derive
    on its own) is still read as a fallback, so a token already written to
    ``.env`` under the old name keeps working.
    """
    try:
        from agent.secret_scope import get_secret  # type: ignore

        val = get_secret(FORGEJO_TOKEN_ENV)
        if val:
            return str(val)
    except Exception:
        logger.debug("agent.secret_scope is unavailable; falling back to the environment", exc_info=True)
    return os.environ.get(FORGEJO_TOKEN_ENV) or os.environ.get(FORGEJO_TOKEN_ENV_LEGACY, "")


def allowed_user_ids(config: Mapping[str, Any] | None = None) -> list[int]:
    """User ids allowed to tap the Telegram picker (empty = open)."""
    return list(settings(config)["allowed_user_ids"])
