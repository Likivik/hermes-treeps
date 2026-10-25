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
    "repo_dir",
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
    "child_immediate_reclaim",
)

# Keys whose schema entry is a SECRET: they live in .env, never in config.yaml.
# They are NOT read through settings() (a secret must not travel through a
# config dict), but they ARE declared in the manifest so the Desktop form can
# render them.
SECRET_KEYS: tuple[str, ...] = ("forgejo_token",)

DEFAULTS: dict[str, Any] = {
    # The repo directory: everything the plugin owns lives under it, and the
    # session workspaces live in its .workspaces/ dot-dir. This is the ONE
    # knob that decides where workspaces are created.
    "repo_dir": "/Git/hermes",
    # DEPRECATED: the workspace root is DERIVED from repo_dir now
    # (repo_dir/.workspaces). The key is still parsed so an old config does
    # not break the plugin, and a one-time hint names repo_dir as the
    # replacement - but the value below is only the pre-dot-dir default.
    "workspace_root": "/Git/hermes/workspaces",
    # DEPRECATED: repo discovery reads repo_roots; the dot-dir layout does
    # not change discovery in this release. A one-time hint names repo_dir.
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
    # Reclaim a child's workspace the moment the child stops.
    #
    # WHY the default is True: a finished child's tree is dead weight on
    # disk, and the reclamation preserves the work (jj snapshots the dirty
    # working copy into a commit and the plugin records the commit id). The
    # alternative - leaving the tree until the 14-day sweep - means every
    # finished child occupies disk for two weeks.
    #
    # WHY the switch exists at all: an operator who does not trust the
    # snapshot path (an unusual jj build, a repo where the snapshot lands
    # somewhere unexpected) must be able to turn the reclamation OFF
    # without losing the REPORTING. The reporting is the part that tells
    # a human what happened; the reclamation is the part that acts on it.
    # Turning this off leaves every other behaviour intact.
    "child_immediate_reclaim": True,
    # The verification command: run in the CHILD's workspace to find out
}

ALLOWED_SOURCES = ("local", "github", "forgejo")
ALLOWED_VCS = ("auto", "jj", "git")

# Environment overrides: ENV_VAR -> settings key.
ENV_OVERRIDES: dict[str, str] = {
    "HERMES_TREEPS_REPO_DIR": "repo_dir",
    "HERMES_WS_REPO_DIR": "repo_dir",  # the short alias
    "HERMES_TREEPS_WORKSPACE_ROOT": "workspace_root",  # deprecated
    "HERMES_WS_ROOT": "workspace_root",  # the original override name, deprecated
    "HERMES_TREEPS_REPO_ROOTS": "repo_roots",  # deprecated
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

# The dot-dir name of the workspaces directory, under repo_dir.
WORKSPACES_DIRNAME = ".workspaces"

# One-time deprecation hints, keyed by the deprecated setting name.
#
# WHY a module-level set: settings() is called on EVERY session start, every
# ws_doctor, every gate check. A hint that fires on each of those would
# scroll the operator's log with the same sentence all day. The hint is meant
# to say "your config still names the old key, here is the new one" ONCE.
# The set is process-local, which is the right scope: a fresh process (a new
# gateway start) says it again, because the operator may not have seen it.
_DEPRECATION_HINTS_SHOWN: set[str] = set()


def reset_deprecation_hints() -> None:
    """Forget which deprecation hints were already shown.

    WHY the tests need it: the one-time behaviour is process-global state, so
    a test that asserts "the hint fires" would poison every later test that
    asserts "the hint does NOT fire again". Resetting is the only way to
    test both halves of a one-time rule.
    """
    _DEPRECATION_HINTS_SHOWN.clear()


def deprecation_hint(key: str, message: str) -> None:
    """Log a deprecation hint for ``key`` - once per process."""
    if key in _DEPRECATION_HINTS_SHOWN:
        return
    _DEPRECATION_HINTS_SHOWN.add(key)
    logger.warning("hermes-treeps: %s", message)


def derived_workspace_root(repo_dir: str) -> str:
    """The workspace root for a repo dir: ``<repo_dir>/.workspaces``.

    WHY derived and not configured: the workspaces live NEXT TO the repos they
    belong to, in a dot-dir, so they are invisible to a plain ``ls`` of the
    repo dir and cannot be mistaken for a repo. Two knobs (workspace_root AND
    repo_dir) could disagree about where a workspace goes, and the loser
    would be whichever one a caller happened to read. One knob, derived.
    """
    return f"{str(repo_dir).rstrip('/')}/{WORKSPACES_DIRNAME}"


def legacy_workspace_root(config: Mapping[str, Any] | None = None) -> str:
    """The workspace root the operator CONFIGURED, before the derivation.

    WHY this exists: ``settings()["workspace_root"]`` now returns the
    DERIVED root, so the configured one is gone from the settings dict. The
    one-time migration needs it - it is how the old workspaces are FOUND.
    Without it, a host whose workspaces live under a custom root would
    report "nothing to move" and the operator would have to move them by
    hand.

    Returns "" when the operator set nothing (the shipped default applies).
    """
    configured = _settings_entry(config)
    if "workspace_root" in configured:
        return str(configured["workspace_root"])
    for var, key in ENV_OVERRIDES.items():
        if key == "workspace_root":
            raw = os.environ.get(var, "")
            if raw.strip():
                return raw.strip()
    return ""


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

    _abs_path("repo_dir")
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
            f"got {timeout!r}"
        )

    # ── the immediate child reclamation kill-switch ─────────────────────
    # WHY it is validated as a bool and not coerced: a config value of
    # "false" (the STRING) is truthy in Python, so coercing it would turn
    # the kill-switch the other way round - an operator who typed
    # ``child_immediate_reclaim: "false"`` would get the reclamation ON.
    if not isinstance(out.get("child_immediate_reclaim"), bool):
        raise SettingsError(
            "hermes-treeps settings.child_immediate_reclaim: must be true or false"
        )

    return out


def settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The plugin's effective settings.

    Precedence: environment override > ``config.yaml``
    (``plugins.entries.hermes-treeps.settings``) > built-in default.

    Raises :class:`SettingsError` (a ``WorkspaceError``) naming the key when a
    value cannot work - a bad path or an unknown discovery source is a
    configuration bug, and silently falling back would hide it.

    The workspace root is DERIVED: ``<repo_dir>/.workspaces``. The legacy
    ``workspace_root`` key is still parsed (so an old config does not crash
    the plugin), and when it was set explicitly to something other than the
    derived value, a one-time hint names ``repo_dir`` as the replacement.
    """
    values: dict[str, Any] = dict(DEFAULTS)
    configured = _settings_entry(config)
    values.update(configured)
    env = _env_overrides()
    values.update(env)

    # Which deprecated keys did the OPERATOR set (config.yaml or env), as
    # opposed to inherited from the defaults? Only an explicit value is worth
    # a deprecation hint: a default nobody typed is not something to nag
    # about, and the defaults still carry the old key so the manifest schema
    # test keeps its shape.
    explicit = set(configured) | {
        key for var, key in ENV_OVERRIDES.items() if os.environ.get(var, "").strip()
    }

    derived = derived_workspace_root(str(values["repo_dir"]))

    # Validate the values as they were configured BEFORE the derivation
    # overwrites the deprecated key: a stale config with a relative
    # workspace_root has to be refused and named, not silently accepted
    # because the derived value happens to be fine.
    _validate(values)

    if "workspace_root" in explicit:
        legacy_root = str(values["workspace_root"])
        if legacy_root.rstrip("/") != derived.rstrip("/"):
            deprecation_hint(
                "workspace_root",
                "settings.workspace_root is deprecated and ignored for path "
                f"derivation (configured {legacy_root!r}); the workspace root is "
                f"now derived from settings.repo_dir: {derived!r}. Set "
                "repo_dir (or HERMES_TREEPS_REPO_DIR) instead. Existing "
                "workspaces under the old root are moved by `hermes ws migrate`.",
            )
    # The derived value WINS over the legacy key: one source of truth.
    values["workspace_root"] = derived

    if "repo_roots" in explicit:
        deprecation_hint(
            "repo_roots",
            "settings.repo_roots is deprecated; repo discovery now follows "
            f"settings.repo_dir ({values['repo_dir']!r}). The list is still "
            "read for this release, but set repo_dir instead.",
        )

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
