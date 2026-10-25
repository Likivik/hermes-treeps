"""The plugin's module layout: __init__.py is the hook surface, not the codebase.

WHY this test exists: ``__init__.py`` grew past 1500 lines because every
feature was appended to it. A file that long cannot be reviewed, and the
reviewer who skims it will miss the gate. The split keeps ``__init__.py``
small enough to read end to end, and this file is the tripwire that stops it
from growing back.

The imports here are deliberately the HOST's imports, not the test suite's:
the plugin is a directory whose name is not a valid Python identifier, so the
loader imports it by file location under ``hermes_plugins.<slug>``. A fresh
interpreter that does the same thing proves each module is importable on its
own, with no import-order luck involved.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The loader imports the plugin under hermes_plugins.<slug>; the slug is the
# manifest key with dashes turned into underscores.
IMPORT_NAME = "hermes_plugins.hermes_treeps"


def test_init_stays_small():
    """__init__.py is the register() + hook-wiring surface, and nothing more."""
    lines = (ROOT / "__init__.py").read_text().splitlines()
    assert len(lines) < 700, f"__init__.py is {len(lines)} lines; the split target is < 700"


def _fresh_interpreter(body: str) -> subprocess.CompletedProcess:
    """Run *body* in a FRESH interpreter that loads the plugin the way the host does.

    A fresh interpreter is the point: an import that only works because some
    other module happened to be imported first is an import-order accident,
    and the host does not guarantee any order.
    """
    prelude = f"""
import importlib.util
import sys
from pathlib import Path

root = Path({str(ROOT)!r})
# The host creates the hermes_plugins namespace package and imports the
# plugin directory under it by file location.
ns = type(sys)("hermes_plugins")
ns.__path__ = []
ns.__package__ = "hermes_plugins"
sys.modules["hermes_plugins"] = ns
spec = importlib.util.spec_from_file_location(
    {IMPORT_NAME!r}, root / "__init__.py", submodule_search_locations=[str(root)]
)
module = importlib.util.module_from_spec(spec)
module.__package__ = {IMPORT_NAME!r}
module.__path__ = [str(root)]
sys.modules[{IMPORT_NAME!r}] = module
spec.loader.exec_module(module)
"""
    return subprocess.run(
        [sys.executable, "-c", prelude + body],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )


def test_the_plugin_package_imports_and_exposes_register():
    """The host's import path still works and register() is callable."""
    result = _fresh_interpreter(
        f"import {IMPORT_NAME} as p\n"
        "assert callable(p.register), 'register is not callable'\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, f"the plugin does not import cleanly:\n{result.stdout}\n{result.stderr}"
    assert "ok" in result.stdout


def test_the_split_modules_import_cleanly():
    """gate.py, tools.py and cli.py each import in a fresh interpreter."""
    for name in ("gate", "tools", "cli"):
        result = _fresh_interpreter(f"importlib.import_module({IMPORT_NAME!r} + '.{name}')\n")
        assert result.returncode == 0, (
            f"{name}.py does not import cleanly in a fresh interpreter:\n"
            f"{result.stdout}\n{result.stderr}"
        )
