"""vcs.py: one adapter for jj and git, and an explicit choice between them.

The plugin used to hard-wire jj in some places and git in others, and the
choice was implicit (whatever the repo happened to contain). The adapter makes
it explicit and configurable - and a forced choice that contradicts the repo
on disk is REFUSED rather than half-applied.
"""

from __future__ import annotations

import pytest
from plugin_loader import VCS

WorkspaceError = VCS.WorkspaceError


def _plain_repo(tmp_path, name="plain"):
    """A directory that is not a jj repo."""
    d = tmp_path / name
    d.mkdir()
    return d


def _jj_repo(tmp_path, name="jjrepo"):
    """A directory that looks like a jj repo (a .jj directory is enough)."""
    d = tmp_path / name
    (d / ".jj").mkdir(parents=True)
    return d


# ── detection ────────────────────────────────────────────────────────────────


def test_detect_follows_the_repo(tmp_path):
    """A repo with .jj is jj; anything else is git."""
    assert VCS.detect(_jj_repo(tmp_path)) == VCS.JJS
    assert VCS.detect(_plain_repo(tmp_path)) == VCS.GIT


# ── the policy ───────────────────────────────────────────────────────────────


def test_auto_picks_git_for_a_plain_repo(tmp_path):
    """``auto`` follows the repo: no .jj means git."""
    assert VCS.resolve(_plain_repo(tmp_path), "auto") == VCS.GIT


def test_auto_picks_jj_for_a_jj_repo(tmp_path):
    """``auto`` follows the repo: a .jj directory means jj."""
    assert VCS.resolve(_jj_repo(tmp_path), "auto") == VCS.JJS


def test_auto_defaults_to_auto(tmp_path):
    """The default policy is ``auto``."""
    assert VCS.resolve(_plain_repo(tmp_path)) == VCS.GIT


def test_vcs_git_on_a_jj_repo_is_refused_with_a_clear_message(tmp_path):
    """A forced choice that contradicts the repo is REFUSED, not half-applied.

    WHY: a git worktree cannot be added to a jj repo. Silently using jj (or
    silently using git and failing halfway) is how a workspace ends up
    registered in one VCS and cleaned up in another.
    """
    repo = _jj_repo(tmp_path)
    with pytest.raises(WorkspaceError) as exc:
        VCS.resolve(repo, "git")
    msg = str(exc.value)
    assert "git" in msg and "jj" in msg, msg
    assert str(repo) in msg, msg


def test_vcs_jj_on_a_git_repo_is_refused_with_a_clear_message(tmp_path):
    """The mirror image: jj cannot be forced onto a plain git repo."""
    repo = _plain_repo(tmp_path)
    with pytest.raises(WorkspaceError) as exc:
        VCS.resolve(repo, "jj")
    msg = str(exc.value)
    assert "jj" in msg and "git" in msg, msg


def test_a_matching_forced_choice_is_accepted(tmp_path):
    """Forcing the VCS the repo actually uses is fine."""
    assert VCS.resolve(_jj_repo(tmp_path), "jj") == VCS.JJS
    assert VCS.resolve(_plain_repo(tmp_path), "git") == VCS.GIT


def test_an_unknown_vcs_is_refused_and_names_the_choices(tmp_path):
    """A typo in the configuration fails loudly."""
    with pytest.raises(WorkspaceError) as exc:
        VCS.resolve(_plain_repo(tmp_path), "hg")
    msg = str(exc.value)
    assert "hg" in msg, msg
    assert "auto" in msg and "jj" in msg and "git" in msg, msg


# ── the workspace lifecycle goes through the adapter ─────────────────────────


def test_create_workspace_uses_jj_for_a_jj_repo(tmp_path):
    """A jj repo gets a jj workspace - and the runner is the one we passed."""
    repo = _jj_repo(tmp_path)
    calls = []
    kind = VCS.create_workspace(
        repo, tmp_path / "ws", "ws", vcs="auto",
        runner=lambda cmd, cwd=None: calls.append((list(cmd), cwd)),
    )
    assert kind == VCS.JJS
    assert calls, "no command was run"
    assert calls[0][0][:2] == ["jj", "workspace"], calls
    assert calls[0][1] == repo, calls


def test_create_workspace_uses_git_for_a_plain_repo(tmp_path):
    """A plain repo gets a git worktree."""
    repo = _plain_repo(tmp_path)
    calls = []
    kind = VCS.create_workspace(
        repo, tmp_path / "ws", "ws", vcs="auto",
        runner=lambda cmd, cwd=None: calls.append((list(cmd), cwd)),
    )
    assert kind == VCS.GIT
    assert calls[0][0][:2] == ["git", "worktree"], calls
    assert calls[0][1] == repo, calls


def test_create_workspace_refuses_a_contradicting_policy(tmp_path):
    """The policy is checked before any command is run."""
    repo = _jj_repo(tmp_path)
    calls = []
    with pytest.raises(WorkspaceError):
        VCS.create_workspace(
            repo, tmp_path / "ws", "ws", vcs="git",
            runner=lambda cmd, cwd=None: calls.append((list(cmd), cwd)),
        )
    assert not calls, f"a command was run despite the refusal: {calls}"
