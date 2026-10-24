"""Load the plugin the way the loader does and assert its public surface.

Runs in CI with no Hermes install: plugin __init__ must import cleanly against
the stdlib alone (host modules are imported lazily inside the functions that
need them). Anything that breaks that contract fails here instead of at gateway
startup, where it silently disables the plugin.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TOOLS = {"ws_new", "ws_chatting", "ws_rename", "ws_status", "ws_doctor"}


def main() -> int:
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
