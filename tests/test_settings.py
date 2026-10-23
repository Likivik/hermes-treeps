"""settings.py: one accessor, one namespace, loud failures.

The configuration lives in ``plugins.entries.hermes-treeps.settings``. There is
deliberately no fallback to the old top-level keys, so these tests also pin
that a legacy key is IGNORED - a silent fallback is how a plugin ends up with
two sources of truth.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from plugin_loader import SETTINGS, STORE

settings_module = SETTINGS
SettingsError = settings_module.SettingsError


def _config(**settings):
    """A config.yaml shaped dict with the plugin's settings in it."""
    return {"plugins": {"entries": {"hermes-treeps": {"settings": settings}}}}


# ── defaults ─────────────────────────────────────────────────────────────────


def test_defaults_reproduce_the_original_constants():
    """With nothing configured, the plugin behaves exactly as it did before."""
    cfg = settings_module.settings()
    assert cfg["workspace_root"] == "/Git/hermes/workspaces"
    assert cfg["repo_roots"] == ["/Git", "/Git/hermes", "/Git/likivik", "/Git/repos"]
    assert cfg["clone_root"] == "/Git/hermes"
    assert cfg["repo_sources"] == ["local", "github", "forgejo"]
    assert cfg["vcs"] == "auto"
    assert cfg["picker_enabled"] is True
    assert cfg["allowed_user_ids"] == []
    assert cfg["forgejo_url"] == ""
    # The store's own constants must agree with the settings defaults.
    assert STORE.DEFAULT_ROOT == cfg["workspace_root"]
    assert list(STORE.REPO_ROOTS) == cfg["repo_roots"]


def test_a_custom_config_is_honoured():
    """What the user configured is what the plugin uses."""
    cfg = settings_module.settings(_config(
        workspace_root="/srv/ws",
        repo_roots=["/srv/repos"],
        clone_root="/srv/new",
        repo_sources=["local"],
        vcs="jj",
        picker_enabled=False,
        allowed_user_ids=[42],
        forgejo_url="https://forgejo.example",
    ))
    assert cfg["workspace_root"] == "/srv/ws"
    assert cfg["repo_roots"] == ["/srv/repos"]
    assert cfg["clone_root"] == "/srv/new"
    assert cfg["repo_sources"] == ["local"]
    assert cfg["vcs"] == "jj"
    assert cfg["picker_enabled"] is False
    assert cfg["allowed_user_ids"] == [42]
    assert cfg["forgejo_url"] == "https://forgejo.example"


def test_a_legacy_top_level_key_is_ignored():
    """No compatibility fallback: only the settings namespace is read.

    WHY: there is exactly one install of this plugin, so a reader of the old
    top-level keys would be dead code - and dead code that reads a DIFFERENT
    value than the live one is worse than none.
    """
    cfg = settings_module.settings({"plugins": {"entries": {"hermes-treeps": {
        "workspace_root": "/legacy/root",
        "settings": {"workspace_root": "/current/root"},
    }}}})
    assert cfg["workspace_root"] == "/current/root", cfg


# ── validation: a bad value names its key ────────────────────────────────────


def test_a_relative_workspace_root_is_refused_and_names_the_key():
    with pytest.raises(SettingsError) as exc:
        settings_module.settings(_config(workspace_root="relative/path"))
    assert "workspace_root" in str(exc.value), exc.value


def test_an_unknown_repo_source_is_refused_and_names_the_key():
    with pytest.raises(SettingsError) as exc:
        settings_module.settings(_config(repo_sources=["local", "gitlab"]))
    assert "repo_sources" in str(exc.value), exc.value
    assert "gitlab" in str(exc.value), exc.value


def test_an_unknown_vcs_is_refused_and_names_the_key():
    with pytest.raises(SettingsError) as exc:
        settings_module.settings(_config(vcs="hg"))
    assert "vcs" in str(exc.value), exc.value


def test_a_relative_clone_root_is_refused_and_names_the_key():
    with pytest.raises(SettingsError) as exc:
        settings_module.settings(_config(clone_root="somewhere"))
    assert "clone_root" in str(exc.value), exc.value


# ── schema drift: the manifest and the code must agree ───────────────────────


def _manifest_config_schema() -> dict[str, str]:
    """The ``config_schema`` block of plugin.yaml, as {key: type}.

    Parsed by hand rather than with PyYAML: the CI image installs pytest and
    nothing else, and the plugin itself depends on the stdlib alone.
    """
    root = Path(__file__).resolve().parent.parent
    text = (root / "plugin.yaml").read_text()
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("config_schema:"):
            start = i + 1
            break
    assert start is not None, "plugin.yaml has no config_schema block"
    schema: dict[str, str] = {}
    key = None
    for line in lines[start:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            break  # the block ended
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 2 and stripped.endswith(":"):
            key = stripped[:-1]
            schema[key] = ""
        elif indent == 4 and key and stripped.startswith("type:"):
            schema[key] = stripped.split(":", 1)[1].strip()
    return schema


def test_plugin_yaml_declares_exactly_the_keys_settings_reads():
    """The manifest's config_schema and settings() must not drift apart.

    WHY: the Desktop settings form is generated from the manifest. A key the
    code reads but the manifest does not declare is invisible to the user, and
    a key the manifest declares but the code ignores is a lie.
    """
    schema = _manifest_config_schema()
    declared = set(schema)
    read = set(settings_module.SETTINGS_KEYS)
    secrets = set(settings_module.SECRET_KEYS)
    assert declared == read | secrets, (
        f"plugin.yaml config_schema {sorted(declared)} != "
        f"settings keys {sorted(read)} + secrets {sorted(secrets)}"
    )
    for key, stype in schema.items():
        assert stype, f"{key} declares no type"
        assert stype in ("str", "list", "bool", "int", "secret"), f"{key}: {stype}"


def test_the_manifest_declares_the_manifest_v2_fields():
    """The manifest is a v2 manifest with the fields the host knows."""
    root = Path(__file__).resolve().parent.parent
    text = (root / "plugin.yaml").read_text()
    assert re.search(r"^manifest_version:\s*2\s*$", text, re.M), text
    assert re.search(r"^api_version:\s*\d+\s*$", text, re.M), text
    assert re.search(r"^license:\s*\S+", text, re.M), text
    assert re.search(r"^homepage:\s*\S+", text, re.M), text
    assert re.search(r"^tags:", text, re.M), text
