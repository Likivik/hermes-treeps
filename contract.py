"""contract.py: the delegation rules, as text that ships with the plugin.

The rules used to live in one operator's head and in a skill file. Both are
lossy: the head forgets, the skill file is not read by a session that has
never heard of it. This module turns them into prompt text that the plugin
can inject into every session's system prompt, so the rules apply to whoever
delegates, not only to whoever remembers.

The rules are derived from measured failures, not opinions:

- a three-paragraph brief ran 52 minutes, died at 60% and needed three
  follow-up PRs; the equivalent numbered checklist finished in 13 minutes,
  correct first try;
- a child wrote into the live state dir and deleted a live session record;
- another test run wrote 38 rows into the live database;
- three defects passed a green suite and appeared only when the live command
  ran.

``RULES`` is the raw text, with the plugin's own tool names left as
placeholders. ``render()`` substitutes them; ``as_prompt_section()`` wraps the
result under the section title. Rendering is idempotent - calling it twice
never duplicates the block.

This module is standalone: it imports nothing from the plugin and touches no
state.
"""

from __future__ import annotations

SECTION_TITLE = "Delegation contract"

#: Imperative rules, one per line. ``{ws_new}``-style placeholders are
#: replaced by ``render()`` with the plugin's tool names.
RULES = """\
- First action in a fresh session: run ws_doctor and cd into the workspace
  path it names. The gate binds you to your own workspace, but your shell may
  still start in the parent's tree - one cd fixes that; working from the wrong
  tree is the collision this whole contract exists to prevent.
- One concern per PR. Split anything else.
- Report defects as a numbered list with file:line, never as prose.
- New tests must fail on today's code before the fix; show the red run.
- Never touch live state: the HERMES_HOME plugin-data dir, the workspace trees,
  or the live db. Copy what you need into a tmp dir and work there.
- One pytest run per iteration. Open the PR the moment it is green.
- Drop the P2s and say plainly that you dropped them.
- Verify yourself: a green suite is a claim about CI, not about behaviour.
  Run the real command and watch it.

Measured: a three-paragraph brief ran 52 min, died at 60% and needed three
follow-up PRs, while an equivalent numbered checklist finished in 13 min,
correct first try. A child wrote into the live state dir and deleted a live
session record. Another test run wrote 38 rows into the live database. Three
defects passed a green suite and appeared only when the live command ran.

Tools: {ws_new} creates the workspace, {ws_chatting} declares a talk-only
session, {ws_doctor} checks the install."""


def render() -> str:
    """Return the rules with the plugin's tool names substituted.

    Idempotent: the result depends only on ``RULES``, so rendering twice
    yields the same string and never a doubled block.
    """
    return RULES.format(
        ws_new="ws_new",
        ws_chatting="ws_chatting",
        ws_doctor="ws_doctor",
    )


def as_prompt_section() -> str:
    """Return the rendered rules as a markdown section, ready to append."""
    return f"## {SECTION_TITLE}\n\n{render()}"
