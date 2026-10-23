"""Load the plugin the way the loader does, and expose it to the tests.

The plugin directory IS the package (``__init__.py`` + ``store.py`` +
``telegram_picker.py``), and its directory name (``hermes-treeps``) is not a
valid Python identifier - so it cannot be imported by name. The loader loads it
by file location with a package search location; ``scripts/smoke_import.py``
does the same. The tests reuse that exact mechanism so they exercise the real
relative imports (``from . import store``), not a re-implementation.

No network, no jj repo, no Hermes install: the plugin must import against the
stdlib alone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "hermes_treeps", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    assert spec and spec.loader, "could not build an import spec for __init__.py"
    module = importlib.util.module_from_spec(spec)
    sys.modules["hermes_treeps"] = module
    spec.loader.exec_module(module)
    return module


PLUGIN = _load_plugin()
STORE = PLUGIN.store
