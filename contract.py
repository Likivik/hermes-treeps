"""contract.py: the child-end report lines, as text that ships with the plugin.

This module used to hold a "delegation contract": a block of imperative rules
(live-state bans, PR scope, one-pytest-run-per-iteration, a subagent primer)
that was injected into every session's system prompt. That block is gone. The
plugin is for automatic workspaces per agent session and its children; nothing
about spawning subagents or scoping PRs needs teaching to a session that
already carries its own harness, and the block cost every session's prompt
about 1.5k characters to say so.

What survives here is the one thing a parent genuinely cannot know: what a
finished child left behind. The hooks in ``childreport`` own the DATA; this
module owns the WORDS.

``render_reports()`` is silent when there is nothing pending, so a session with
no finished children carries no empty section.

This module is standalone: it imports nothing from the plugin and touches no
state.
"""

from __future__ import annotations

from typing import Any

# ── child-end reports ────────────────────────────────────────────────────────
#
# A child can end without a word, and the parent then has no idea which tree
# the child touched, whether the work is still there, or whether it was ever
# pushed. These helpers turn the child's end state into ONE line the parent
# reads on its next turn.

#: Title of the injected section. A parent that has finished children sees
#: this heading; a parent with none sees nothing at all.
REPORTS_SECTION_TITLE = "Child-end reports"


def render_reports(reports: list[dict[str, Any]]) -> str:
    """Render pending child reports as prompt text.

    Returns the EMPTY STRING when there is nothing pending. Silent is the
    requirement: a session with no finished children must not carry an empty
    section in its prompt.

    One line per child, carrying the four facts the parent cannot otherwise
    know: the workspace (slug and path), clean-or-dirty, the number of
    unpushed commits, and the child's own status word.
    """
    if not reports:
        return ""
    lines: list[str] = []
    for report in reports:
        lines.append(_report_line(report))
    if not lines:
        return ""
    return f"## {REPORTS_SECTION_TITLE}\n\n" + "\n".join(lines)


def _report_line(report: dict[str, Any]) -> str:
    """One child, one line.

    The line reads:

        child <slug>: <clean|dirty|state-unknown>, <N unpushed commits> -
        <workspace path> (<status>)

    ``state-unknown`` and ``unpushed unknown`` are the honest answers when a
    check failed. A report that says "clean" for a workspace we could not
    inspect is a lie, and this feature exists to stop lies.
    """
    slug = str(report.get("workspace_slug") or "").strip()
    if not slug:
        slug = "(no workspace)"
    path = str(report.get("workspace_path") or "").strip()
    state = str(report.get("dirty") or "").strip() or "state-unknown"
    unpushed = report.get("unpushed")
    if unpushed is None:
        commits = "unpushed unknown"
    else:
        try:
            n = int(unpushed)
        except (TypeError, ValueError):
            n = 0
        commits = "0 unpushed commits" if n == 0 else f"{n} unpushed commit{'s' if n != 1 else ''}"
    status = str(report.get("child_status") or "").strip() or "unknown"
    role = str(report.get("child_role") or "").strip()
    role_part = f"{role} " if role else ""
    where = f" - {path}" if path else ""
    return f"child {slug}: {state}, {commits}{where} ({role_part}{status})"
