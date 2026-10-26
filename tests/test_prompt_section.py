"""The single injected prompt section: routing only, and short.

WHY this file exists at all: the section ships inside EVERY session's system
prompt - parent and child alike. It once carried 3225 characters across two
sections (workspace discipline + a "delegation contract": live-state bans, PR
scope, a subagent primer, sqlite advice), and the operator's verdict was that
the plugin is for automatic workspaces per agent session and its children, so
all of that was noise. The one thing the gate cannot do for a session is ask
the user which workspace, because repo-touching tools are blocked until
``ws_new`` runs - so the ask is the only text worth paying for in every turn.

The budget is the point: a section that grows back into a manual has failed.
"""

from __future__ import annotations

from plugin_loader import PLUGIN, STORE

#: Hard ceiling on the section text. The old pair was 3225 chars; the routing
#: ask is ~520. Anything approaching this ceiling is a manual creeping back in.
MAX_CHARS = 900


def _section(monkeypatch) -> str:
    """Register the plugin against a fake ctx and return the prompt section."""
    registered = {}

    class Ctx:
        def register_tool(self, **kw):
            pass

        def register_cli_command(self, **kw):
            pass

        def register_command(self, *a, **kw):
            pass

        def register_hook(self, *a, **kw):
            pass

        def register_system_prompt_section(self, name, text, **kw):
            registered[name] = text if isinstance(text, str) else text()

    monkeypatch.setattr(STORE, "candidate_repos", lambda limit=6: ["repo-a", "repo-b"])
    PLUGIN.register(Ctx())
    return registered["workspace-discipline"]


def test_the_section_stays_within_budget(monkeypatch):
    text = _section(monkeypatch)
    assert len(text) < MAX_CHARS, f"prompt section is {len(text)} chars; budget {MAX_CHARS}"


def test_the_section_asks_for_the_title_and_offers_the_live_repos(monkeypatch):
    """The ask is the load-bearing part: without it the session stalls."""
    text = _section(monkeypatch)
    assert "session title" in text
    assert "repo-a, repo-b" in text, text


def test_the_section_names_both_binding_tools(monkeypatch):
    text = _section(monkeypatch)
    assert "ws_new" in text
    assert "ws_chatting" in text


def test_the_section_no_longer_ships_the_delegation_contract(monkeypatch):
    """The rules block is gone: no section heading, no per-rule bullets."""
    text = _section(monkeypatch)
    assert "Delegation contract" not in text
    assert "One concern per PR" not in text
    assert "subagent" not in text.lower()
    assert "plugin-data" not in text
