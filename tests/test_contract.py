"""contract.py: the rules are present, bounded, and render idempotently.

These tests assert on substrings, not on the full text: the rules are meant
to be edited, and a test that pinned the exact wording would make every
wording change a fight with CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_contract():
    spec = importlib.util.spec_from_file_location("treeps_contract", ROOT / "contract.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["treeps_contract"] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def test_rules_are_non_empty_and_bounded():
    """The block is injected into every session, so it must stay short."""
    assert contract.RULES.strip()
    assert len(contract.RULES) < 1500


def test_render_contains_the_plugin_tool_names():
    text = contract.render()
    for name in ("ws_new", "ws_chatting", "ws_doctor"):
        assert name in text


def test_render_is_idempotent():
    """Rendering twice must not duplicate the block."""
    first = contract.render()
    second = contract.render()
    assert first == second
    assert second.count("One concern per PR") == 1
    assert "{" not in second  # no placeholder left behind


def test_prompt_section_starts_with_the_title():
    section = contract.as_prompt_section()
    assert section.startswith(f"## {contract.SECTION_TITLE}")
    assert contract.SECTION_TITLE == "Delegation contract"
    assert contract.render() in section


def test_rules_name_the_live_state_paths():
    text = contract.render()
    assert "HERMES_HOME" in text
    assert "plugin-data" in text
    assert "workspace trees" in text
    assert "live db" in text
    assert "tmp dir" in text


def test_rules_state_the_failing_before_rule():
    text = contract.render()
    assert "must fail on today's code" in text
    assert "One pytest run per iteration" in text
    assert "One concern per PR" in text
    assert "Drop the P2s" in text
    assert "green suite" in text


def test_rules_carry_the_measured_facts():
    text = contract.render()
    for fact in ("52 min", "13 min", "38 rows", "60%"):
        assert fact in text
