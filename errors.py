"""Shared error types for the plugin.

Kept in their own module so `vcs.py` and `settings.py` can raise the same
refusals the rest of the plugin uses without importing `store.py` (which
imports them - a cycle).
"""

from __future__ import annotations


class WorkspaceError(RuntimeError):
    """A refusal with a message meant for the user, not a traceback."""
