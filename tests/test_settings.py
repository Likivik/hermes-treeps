"""settings.py: one accessor, one namespace, loud failures.

The configuration lives in ``plugins.entries.hermes-treeps.settings``. There is
deliberately no fallback to the old top-level keys, so these tests also pin
that a legacy key is IGNORED - a silent fallback is how a plugin ends up with
two sources of truth.
"""

from __future__ import annotations

import ast
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
    assert settings_module.DEFAULTS["workspace_root"] == cfg["workspace_root"]
    assert list(settings_module.DEFAULTS["repo_roots"]) == cfg["repo_roots"]


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


def _manifest_config_schema() -> dict[str, dict[str, str]]:
    """The ``config_schema`` block of plugin.yaml, as {key: {field: value}}.

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
    schema: dict[str, dict[str, str]] = {}
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
            schema[key] = {}
        elif indent == 4 and key and ":" in stripped:
            name, _, value = stripped.partition(":")
            schema[key][name.strip()] = value.strip()
    return schema


def _settings_source_keys() -> set[str]:
    """Every settings key the SOURCE of settings.py mentions.

    WHY: the old drift test compared two hand-maintained lists (the module's
    ``SETTINGS_KEYS`` and the manifest's ``config_schema``). A key missing
    from BOTH sides kept the suite green - that is exactly how
    ``inject_first_message`` ended up declared in the manifest and in
    ``SETTINGS_KEYS`` but absent from ``DEFAULTS``, so ``settings()`` never
    returned it and the feature was dead.

    So the key set is read out of the module's own source: the ``DEFAULTS``
    dict, the ``ENV_OVERRIDES`` values (each names the setting it overrides),
    the ``SETTINGS_KEYS``/``SECRET_KEYS`` tuples, and every key the module's
    accessors read out of the settings dict (a ``d["key"]`` subscript or a
    ``d.get("key")`` call).

    The accessor scan skips ``_settings_entry``/``_load_config``: those read
    the CONFIG PATH (``plugins``/``entries``/``settings``), not setting keys.

    The whole package is scanned as well: every ``settings()["key"]`` read in
    ``store.py``, ``discovery.py``, ``telegram_picker.py``, ``db.py`` and
    ``__init__.py`` names a key that must exist. That is what catches a key
    deleted from ``DEFAULTS``, ``SETTINGS_KEYS`` and the manifest at once -
    the consumer still asks for it and would get a ``KeyError``.
    """
    root = Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "settings.py").read_text())

    def _const_tuple(node: ast.AST | None) -> list[str]:
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return [e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        return []

    def _dict_keys(node: ast.AST | None) -> list[str]:
        """The keys of a dict literal (any value type)."""
        if not isinstance(node, ast.Dict):
            return []
        return [k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]

    def _dict_items(node: ast.AST | None) -> list[tuple[str, str]]:
        """(key, value) of a dict literal, both plain strings."""
        if not isinstance(node, ast.Dict):
            return []
        out = []
        for k, v in zip(node.keys, node.values):
            if (isinstance(k, ast.Constant) and isinstance(k.value, str)
                    and isinstance(v, ast.Constant) and isinstance(v.value, str)):
                out.append((k.value, v.value))
        return out

    keys: set[str] = set()

    # module-level constants (plain assignments and annotated ones)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
        else:
            continue
        if name == "DEFAULTS":
            keys.update(_dict_keys(value))
        elif name == "ENV_OVERRIDES":
            # ENV VAR -> setting key: the VALUE is the settings key.
            keys.update(v for _, v in _dict_items(value))
        elif name in ("SETTINGS_KEYS", "SECRET_KEYS"):
            keys.update(_const_tuple(value))

    # accessor reads: d["key"] and d.get("key"), outside the config-path helpers
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        if fn.name in ("_settings_entry", "_load_config"):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, str):
                    keys.add(node.slice.value)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "get" and node.args
                  and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)):
                keys.add(node.args[0].value)

    # The consumers: every settings()["key"] read anywhere in the package.
    for path in sorted(root.glob("*.py")):
        if path.name == "settings.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Constant):
                continue
            if not isinstance(node.slice.value, str):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "settings":
                keys.add(node.slice.value)
    return keys


def test_plugin_yaml_declares_exactly_the_keys_settings_reads():
    """The manifest's config_schema and settings() must not drift apart.

    WHY: the Desktop settings form is generated from the manifest. A key the
    code reads but the manifest does not declare is invisible to the user, and
    a key the manifest declares but the code ignores is a lie.

    The key set comes from the SOURCE (see :func:`_settings_source_keys`), not
    from a second hand-maintained list: a key that is missing from both the
    module's list and the manifest must FAIL this test, not pass it.
    """
    schema = _manifest_config_schema()
    declared = set(schema)
    source_keys = _settings_source_keys()
    assert declared == source_keys, (
        f"plugin.yaml config_schema {sorted(declared)} != "
        f"the keys settings.py actually uses {sorted(source_keys)}"
    )
    for key, spec in schema.items():
        stype = spec.get("type", "")
        assert stype, f"{key} declares no type"
        assert stype in ("str", "list", "bool", "int", "secret"), f"{key}: {stype}"


def test_defaults_and_settings_keys_agree():
    """``DEFAULTS`` and ``SETTINGS_KEYS`` must name the same keys.

    WHY: this is the drift that killed ``inject_first_message``. It was in
    ``SETTINGS_KEYS`` and in the manifest but NOT in ``DEFAULTS``, so
    ``settings()`` never returned it and the caller raised ``KeyError`` - which
    a bare ``except`` swallowed, leaving the feature silently dead.
    """
    defaults = set(settings_module.DEFAULTS)
    listed = set(settings_module.SETTINGS_KEYS)
    assert defaults == listed, (
        f"DEFAULTS {sorted(defaults)} != SETTINGS_KEYS {sorted(listed)}; "
        "a key in one but not the other is dead or undeclared"
    )


def test_every_declared_key_has_a_default():
    """Each settings key the manifest declares must be returned by settings().

    The cheapest guard against the KeyError class of bug: whatever the
    manifest promises, ``settings()`` must actually hand back.
    """
    cfg = settings_module.settings()
    for key in settings_module.SETTINGS_KEYS:
        assert key in cfg, f"settings() does not return the declared key {key!r}"


def test_the_gateway_injection_calls_a_function_that_exists():
    """``__init__.py`` must not call a store function that does not exist.

    WHY: the gateway first-message injection called
    ``store.session_key_for(sid)``, which store.py never defined (it has
    ``session_entry``/``session_for_chat``). The ``AttributeError`` was
    swallowed by a bare ``except``, so the branch was dead code that looked
    alive. This test walks the AST of ``__init__.py`` and checks every
    ``store.<name>(...)`` call against the real store module.
    """
    root = Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "__init__.py").read_text())
    called: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "store"):
            called.add(node.func.attr)
    assert called, "no store.* calls found - the scan is broken"
    for name in sorted(called):
        assert hasattr(STORE, name), (
            f"__init__.py calls store.{name}(...), which store.py does not define"
        )


def test_the_secret_env_name_agrees_with_the_code():
    """The manifest's secret env name must be the name the code reads.

    WHY: the manifest declared ``forgejo_token`` as ``type: secret`` with no
    ``env:``, so the host derived ``HERMES_TREEPS_FORGEJO_TOKEN`` for the
    Desktop secret field while the plugin read ``FORGEJO_TOKEN``. The two
    names never met, so the token was always empty and the Forgejo catalogue
    silently never authenticated.
    """
    schema = _manifest_config_schema()
    spec = schema.get("forgejo_token")
    assert spec is not None, "plugin.yaml does not declare forgejo_token"
    assert spec.get("type") == "secret", spec
    declared_env = spec.get("env", "")
    assert declared_env, (
        "forgejo_token declares no env: - the host would derive "
        "HERMES_TREEPS_FORGEJO_TOKEN, which the plugin never reads"
    )
    assert declared_env == settings_module.FORGEJO_TOKEN_ENV, (
        f"plugin.yaml env {declared_env!r} != the name settings.py reads "
        f"{settings_module.FORGEJO_TOKEN_ENV!r}"
    )
    # And the name really is the one handed to get_secret().
    root = Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "settings.py").read_text())
    passed = {
        node.args[0].id
        for node in ast.walk(tree)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "get_secret" and node.args
            and isinstance(node.args[0], ast.Name))
    }
    assert "FORGEJO_TOKEN_ENV" in passed, (
        "forgejo_token() must pass the shared FORGEJO_TOKEN_ENV constant to "
        "get_secret(), not a hard-coded string"
    )


# ── the prepare settings ─────────────────────────────────────────────────────


def test_a_zero_prepare_timeout_is_refused_and_names_the_key():
    with pytest.raises(SettingsError) as exc:
        settings_module.settings(_config(prepare_timeout_seconds=0))
    assert "prepare_timeout_seconds" in str(exc.value), exc.value


def test_a_prepare_timeout_above_the_ceiling_is_refused_and_names_the_key():
    with pytest.raises(SettingsError) as exc:
        settings_module.settings(_config(prepare_timeout_seconds=4000))
    assert "prepare_timeout_seconds" in str(exc.value), exc.value


def test_a_prepare_timeout_inside_the_range_is_accepted():
    cfg = settings_module.settings(_config(prepare_timeout_seconds=1))
    assert cfg["prepare_timeout_seconds"] == 1


def test_a_non_bool_prepare_enabled_is_refused_and_names_the_key():
    with pytest.raises(SettingsError) as exc:
        settings_module.settings(_config(prepare_enabled="yes"))
    assert "prepare_enabled" in str(exc.value), exc.value


def test_a_non_list_prepare_files_is_refused_and_names_the_key():
    with pytest.raises(SettingsError) as exc:
        settings_module.settings(_config(prepare_files=".env"))
    assert "prepare_files" in str(exc.value), exc.value


def test_a_non_string_prepare_command_is_refused_and_names_the_key():
    with pytest.raises(SettingsError) as exc:
        settings_module.settings(_config(prepare_command=42))
    assert "prepare_command" in str(exc.value), exc.value


def test_the_prepare_defaults_are_what_the_feature_documents():
    cfg = settings_module.settings()
    assert cfg["prepare_enabled"] is True
    assert cfg["prepare_command"] == ""
    assert cfg["prepare_files"] == [".env"]
    assert cfg["prepare_timeout_seconds"] == 300


def test_the_manifest_declares_the_manifest_v2_fields():
    """The manifest is a v2 manifest with the fields the host knows."""
    root = Path(__file__).resolve().parent.parent
    text = (root / "plugin.yaml").read_text()
    assert re.search(r"^manifest_version:\s*2\s*$", text, re.M), text
    assert re.search(r"^api_version:\s*\d+\s*$", text, re.M), text
    assert re.search(r"^license:\s*\S+", text, re.M), text
    assert re.search(r"^homepage:\s*\S+", text, re.M), text
    assert re.search(r"^tags:", text, re.M), text


def test_manifest_defaults_match_settings_defaults():
    """plugin.yaml's declared defaults must equal settings.DEFAULTS values.

    Prevents the silent third copy: the drift test compared key SETS and
    types, so the manifest could declare a different default than the code
    uses and nothing would notice until a user reset the key in the Desktop
    settings form.
    """
    declared = _manifest_config_schema()
    for key, spec in declared.items():
        raw = spec.get("default")
        if raw is None:
            continue
        # The hand parser returns strings; normalise the two shapes we declare.
        value = settings_module.DEFAULTS[key]
        if isinstance(value, bool):
            assert raw == ("true" if value else "false"), (key, raw, value)
        elif isinstance(value, int) and not isinstance(value, bool):
            assert raw == str(value), (key, raw, value)
        elif isinstance(value, (list, tuple)):
            # a flow list in the manifest: strip brackets, compare members
            inner = raw.strip()[1:-1] if raw.startswith("[") else raw
            # the hand parser keeps yaml quoting: '".env"' is .env
            got = [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()]
            assert got == [str(v) for v in value], (key, got, value)
        else:
            # the hand parser keeps the yaml quoting: '""' is the empty string
            raw = raw.strip().strip('"').strip("'")
            assert raw == value, (key, raw, value)
