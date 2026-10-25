"""Load the plugin the way the loader does and assert its public surface.

Runs in CI with no Hermes install: plugin __init__ must import cleanly against
the stdlib alone (host modules are imported lazily inside the functions that
need them). Anything that breaks that contract fails here instead of at gateway
startup, where it silently disables the plugin.

WHY the database is pointed at a tmp dir BEFORE the plugin is loaded: the
plugin opens its database lazily, but this script calls
``store.resolve_detail(...)``, which opens it for real. Without the
override that is the LIVE database under the host's plugin-data directory -
and the schema migration inside it would then CREATE tables in the live
install. A smoke test must not change the state of the thing it smokes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TOOLS = {"ws_new", "ws_chatting", "ws_rename", "ws_status", "ws_doctor"}


def main() -> int:
    # The plugin reads these on import/open. They have to be set BEFORE the
    # module is executed - the database path is resolved at open time, and
    # the legacy state dir is read by the session helpers.
    tmp = tempfile.mkdtemp(prefix="hermes-treeps-smoke-")
    os.environ.setdefault("HERMES_TREEPS_PLUGIN_DATA_DIR", os.path.join(tmp, "plugin-data"))
    os.environ.setdefault("HERMES_TREEPS_DB_PATH", os.path.join(tmp, "plugin-data", "data.db"))
    os.environ.setdefault("HERMES_TREEPS_LEGACY_STATE_DIR", os.path.join(tmp, "legacy-state"))
    os.environ.setdefault(
        "HERMES_TREEPS_LEGACY_WORKSPACE_ROOT", os.path.join(tmp, "legacy-workspaces")
    )

    spec = importlib.util.spec_from_file_location(
        "hermes_treeps", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    assert spec and spec.loader, "could not build an import spec for __init__.py"
    module = importlib.util.module_from_spec(spec)
    sys.modules["hermes_treeps"] = module
    spec.loader.exec_module(module)

    tools = {name for name, _schema, _fn in module._ws_tools()}
    missing = EXPECTED_TOOLS - tools
    assert not missing, f"missing tools: {sorted(missing)}"

    store = module.store
    for attr in ("resolve_detail", "forget", "remote_repos", "ensure_tool_path"):
        assert hasattr(store, attr), f"store.{attr} is missing"

    detail = store.resolve_detail("ci-smoke-session-that-does-not-exist")
    assert detail["state"] == "unbound", detail
    print(f"ok: {len(tools)} tools, unbound resolution verbatim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
