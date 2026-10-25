"""Wave B: the P2 findings from the two behavioural raids.

Every test in this file was RED on the pre-fix code and GREEN after it.
Each one names the failure it prevents, in the docstring, so the reason
the assertion exists survives the next refactor.

The findings covered here:

* F1  - ``prepare.py``: the "workspace must be inside the repo" guard was a
        SELF-comparison and could never fire.
* F2  - ``migrate_layout.py``: the live-session set was snapshotted once
        before the move loop, so a session bound mid-run lost its tree.
* F4  - ``__init__.py``: report delivery read only the literal
        ``session_id`` key.
* F5  - ``store.py``: the heartbeat rate limit read the previous stamp
        outside the lock and broke on a backwards clock step.
* F6  - ``childspace.py``: a 24-bit slug suffix collision hard-blocked the
        child.
* F9  - ``__init__.py``: the ``ws_new`` schema marked ``slug`` required
        while the code defaults it.
* F10 - ``__init__.py``: the injected prompt said the session TITLE names
        the workspace; the code says the opposite.
* F11 - ``settings.py``: the ``repo_roots`` deprecation hint claimed
        discovery follows ``repo_dir``.
* F12 - ``store.py``: ``reclaim_children`` filed a SKIPPED git worktree
        under "reclaimed".
* F15 - ``store.py``: ``gc_apply()`` had no test coverage at all.

No network, no jj repo, no Hermes install: tmp dirs and the stdlib only.
"""

from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path

import pytest
from plugin_loader import PLUGIN, STORE

MIGRATE = importlib.import_module("hermes_treeps.migrate_layout")
CHILDSPACE = importlib.import_module("hermes_treeps.childspace")
PREPARE = importlib.import_module("hermes_treeps.prepare")
SETTINGS = importlib.import_module("hermes_treeps.settings")


# ── F1: prepare must refuse a workspace outside the repo root ────────────────


def _prepare_settings(**over):
    base = {
        "prepare_enabled": True,
        "prepare_command": "",
        "prepare_files": [".env"],
        "prepare_timeout_seconds": 300,
    }
    base.update(over)
    return base


def test_prepare_refuses_a_workspace_outside_the_repo_dir(tmp_path, monkeypatch):
    """A workspace outside the plugin's repo dir is refused, not prepared.

    Prevents: the "the workspace must belong to this plugin" guard being
    DEAD CODE. The guard compared the workspace against ITSELF
    (``_within(resolved_ws, resolved_ws)``), which is always True, so it
    could never fire - and a workspace from a DIFFERENT repo dir was
    prepared as if it belonged to this one, with this repo's ``.env``
    copied into it.

    WHY the boundary is the repo DIR and not the repo root: a workspace
    is a SIBLING of its repo, not a child of it. The layout is
    ``<repo_dir>/<repo>`` for the checkout and
    ``<repo_dir>/.workspaces/<repo>/<slug>`` for the workspace, so a
    comparison against the repo root would refuse every legitimate
    workspace. The repo dir is what the plugin owns.
    """
    repo = tmp_path / "repos" / "repo"
    repo.mkdir(parents=True)
    (repo / ".env").write_text("TOKEN=secret\n")
    monkeypatch.setattr(STORE, "repo_dir", lambda: tmp_path / "repos")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    subprocess.run(["git", "init", "-q", str(elsewhere)], check=True)
    (elsewhere / ".gitignore").write_text(".env\n")

    out = PREPARE.run_prepare(repo, elsewhere, _prepare_settings())

    assert out["ok"] is False, out
    assert "outside the plugin's repo dir" in str(out["error"]), out
    assert not (elsewhere / ".env").exists(), (
        "the refused prepare still copied the repo's .env into the workspace"
    )


def test_prepare_still_runs_for_a_workspace_inside_the_repo_dir(tmp_path, monkeypatch):
    """The guard does not break the normal case.

    Prevents: the fix for F1 turning into a guard that refuses every
    workspace. The real geometry - the workspace as a SIBLING of the
    repo, both under the repo dir - is prepared as before.
    """
    repo_dir = tmp_path / "repos"
    repo = repo_dir / "repo"
    repo.mkdir(parents=True)
    (repo / ".env").write_text("TOKEN=secret\n")
    monkeypatch.setattr(STORE, "repo_dir", lambda: repo_dir)

    workspace = repo_dir / ".workspaces" / "repo" / "repo-abc123"
    workspace.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    (workspace / ".gitignore").write_text(".env\n")

    out = PREPARE.run_prepare(repo, workspace, _prepare_settings())

    assert out["ok"] is True, out
    assert out["copied"] == [".env"], out
    assert (workspace / ".env").read_text() == "TOKEN=secret\n"


# ── F2: the migration re-checks liveness at rename time ─────────────────────


def _legacy_root(tmp_path: Path) -> Path:
    return tmp_path / "repos" / "workspaces"


@pytest.fixture
def migrate_layout(tmp_path, monkeypatch):
    """A tmp repo dir with the OLD workspace root, wired like the real one."""
    repo_dir = tmp_path / "repos"
    repo_dir.mkdir()
    legacy = _legacy_root(tmp_path)
    legacy.mkdir()
    monkeypatch.setattr(STORE, "repo_dir", lambda: repo_dir)
    monkeypatch.setattr(
        STORE,
        "settings",
        lambda: {
            "repo_dir": str(repo_dir),
            "workspace_root": str(legacy),
            "repo_roots": [str(repo_dir)],
        },
    )
    monkeypatch.setattr(MIGRATE, "legacy_roots", lambda *a, **kw: [legacy])
    return {"repo_dir": repo_dir, "legacy": legacy}


def _seed_workspace(layout, repo="demo", slug="demo-abc123", sessions=()):
    ws = layout["legacy"] / repo / slug
    ws.mkdir(parents=True)
    STORE.write_record(
        ws,
        {
            "repo": repo,
            "slug": slug,
            "path": str(ws),
            "vcs": "jj",
            "sessions": list(sessions),
            "created_at": "2024-01-01T00:00:00",
        },
    )
    return ws


def test_the_migration_rechecks_liveness_at_rename_time(migrate_layout, monkeypatch):
    """A session that binds DURING the migration keeps its tree.

    Prevents: the live-session set being snapshotted ONCE, before the
    move loop. The loop runs a dirty-tree check (a subprocess) per
    workspace, so it can take minutes; a session that binds to a tree
    while it runs was not in the pre-loop snapshot, and its tree was
    renamed away underneath it - the session then resolves to a path that
    no longer exists.

    WHY the second workspace matters: the migration walks every
    workspace, and the check happens per workspace. The mock reports the
    SECOND tree as live from the moment the FIRST tree has been
    processed - which is exactly "a session bound while the migration
    was running". With the pre-loop snapshot the second tree was judged
    against a set that did not contain it, so it was moved away from
    under the session that had just bound to it.
    """
    first = _seed_workspace(migrate_layout, repo="demo", slug="demo-aaaaaa")
    second = _seed_workspace(migrate_layout, repo="demo", slug="demo-bbbbbb")

    calls: list[str] = []

    def _live_paths():
        calls.append("check")
        # The first tree has already been through the loop by the time
        # the second one is judged: the session has bound to the second
        # tree WHILE the migration was running.
        processed = [p for p in (first, second) if not p.exists()]
        if processed:
            # At least one tree was already processed: the session has
            # bound to the remaining tree mid-run.
            return {os.path.normpath(str(second))}
        return set()

    monkeypatch.setattr(MIGRATE, "_live_session_paths", _live_paths)
    # The trees are clean: the dirty check is a jj/git call, stub it out.
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    report = MIGRATE.migrate_workspaces()

    assert len(calls) >= 2, (
        f"the liveness check was called {len(calls)} time(s) - it must be "
        "re-checked per workspace, not snapshotted once"
    )
    assert second.exists(), "a tree a live session bound to was moved away"
    assert report["in_use"], report
    assert any(row["path"] == str(second) for row in report["in_use"]), report
    assert all(row["path"] != str(second) for row in report["moved"]), report


def test_the_migration_moves_a_tree_no_session_uses(migrate_layout, monkeypatch):
    """The re-check does not block the normal move.

    Prevents: the F2 fix becoming a guard that refuses everything.
    """
    ws = _seed_workspace(migrate_layout, sessions=["sid-gone"])
    monkeypatch.setattr(MIGRATE, "_live_session_paths", lambda: set())
    monkeypatch.setattr(STORE, "dirty", lambda path: False)

    report = MIGRATE.migrate_workspaces()

    assert report["in_use"] == [], report
    assert len(report["moved"]) == 1, report
    assert not ws.exists(), "the tree should have moved"


# ── F4: report delivery finds the session id under every spelling ───────────


def _seed_a_child_report(parent: str, child: str):
    """A pending child report for ``parent``."""
    STORE.db.write_child_report_at(
        STORE.db.db_path(),
        child,
        parent,
        child_role="worker",
        child_status="completed",
        workspace_path="",
        workspace_slug="",
        workspace_repo="",
        dirty="clean",
        unpushed=0,
    )


def test_the_report_is_delivered_when_the_session_id_is_under_another_key():
    """A payload that names the session differently still delivers.

    Prevents: the report being LOST. The delivery read only the literal
    ``session_id`` key, while the plugin's own ``_session_id`` helper
    accepts ``session_id``, ``session``, ``durable_session_key`` and the
    ``HERMES_SESSION_ID`` environment. A host that spells it differently
    got a silent no-op: the parent never heard what happened to its
    child.
    """
    parent = "parent-durable-1"
    _seed_a_child_report(parent, "child-durable-1")

    # The payload uses ``durable_session_key``, not ``session_id``.
    out = PLUGIN._on_pre_llm_call(durable_session_key=parent)

    assert out is not None, "the report was not delivered"
    assert "context" in out, out
    assert parent in out["context"] or "child" in out["context"], out


def test_the_report_section_is_rendered_when_the_session_id_is_under_another_key():
    """The system-prompt section uses the same helper.

    Prevents: the SAME loss in the other delivery path. A session whose
    prompt is built after its child finished gets the report through the
    section - and the section read only the literal ``session_id`` key.
    """
    parent = "parent-section-1"
    _seed_a_child_report(parent, "child-section-1")

    text = PLUGIN._reports_prompt_section({"durable_session_key": parent})

    assert text, "the report section was empty"


def test_a_payload_without_any_session_id_delivers_nothing(monkeypatch):
    """No session id at all: no report, no crash.

    WHY this matters: ``_session_id`` falls back to
    ``HERMES_SESSION_ID`` and then to the literal string ``"unknown"``.
    A delivery keyed on ``"unknown"`` would look up a report for a
    session that does not exist - harmless, but it must not raise.
    """
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    assert PLUGIN._on_pre_llm_call() is None


# ── F5: the heartbeat survives a backwards clock step ───────────────────────


def test_a_backwards_clock_step_does_not_freeze_the_heartbeat():
    """A clock that steps backwards does not silence the heartbeat.

    Prevents: a live child looking dead for 24 hours. The rate limit
    compared ``stamp - previous``, so a backwards step made the
    difference NEGATIVE - below the one-minute interval - and EVERY
    later heartbeat was rate-limited away. The stamp froze at the
    pre-step value and the sweep would reclaim a live child's tree.

    WHY ``abs()`` is the fix: a backwards step is treated as "the
    interval passed", so the new (lower) stamp is written and the
    heartbeat carries on.
    """
    sid = "child-clock-1"
    now = 1_000_000.0

    assert STORE.record_child_heartbeat(sid, now=now) is True
    # The clock steps back by an hour.
    assert STORE.record_child_heartbeat(sid, now=now - 3600.0) is True, (
        "the backwards clock step was rate-limited away - the heartbeat is frozen"
    )
    # And the stamp now tracks the new (lower) clock.
    stored = STORE.db.get_kv(STORE.child_heartbeat_key(sid))
    assert stored is not None
    assert float(stored) == pytest.approx(now - 3600.0), stored


def test_the_heartbeat_rate_limit_still_applies():
    """The rate limit is not defeated by the clock-step fix.

    Prevents: the ``abs()`` change turning the heartbeat into an
    unlimited write. Two calls inside the interval: the second is
    refused.
    """
    sid = "child-rate-1"
    now = 2_000_000.0

    assert STORE.record_child_heartbeat(sid, now=now) is True
    assert STORE.record_child_heartbeat(sid, now=now + 1.0) is False
    assert STORE.record_child_heartbeat(sid, now=now + 60.0) is True


# ── F6: a slug collision does not hard-block the child ──────────────────────


def test_a_child_slug_collision_gets_its_own_workspace(tmp_path, monkeypatch):
    """Two children whose ids share the last 6 characters both get a tree.

    Prevents: the child being hard-blocked. The slug suffix is the last
    6 characters of the child session id - 24 bits of entropy, and the
    ids are the host's, not ours. A collision made ``store.create()``
    raise "already exists", the child's workspace creation failed, and
    the gate blocked the child from doing anything at all.

    The fix: a taken slug is a name to be disambiguated, not an error.
    The suffix grows (``-2``, ``-3``, ...) until a free slug is found,
    and the child is bound to THAT one.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)
    monkeypatch.setattr(STORE, "is_jj", lambda path: True)

    created: list[str] = []

    def _fake_create(repo_name, slug, purpose=""):
        created.append(slug)
        if slug == "parent-ws-child-abc123":
            raise STORE.WorkspaceError(f"{slug} already exists")
        path = tmp_path / "ws" / slug
        path.mkdir(parents=True, exist_ok=True)
        STORE.write_record(
            path,
            {
                "repo": repo_name,
                "slug": slug,
                "path": str(path),
                "vcs": "jj",
                "purpose": purpose,
                "sessions": [],
                "created_at": "2024-01-01T00:00:00",
            },
        )
        return {
            "ok": True,
            "repo": repo_name,
            "slug": slug,
            "path": str(path),
        }

    monkeypatch.setattr(STORE, "create", _fake_create)

    parent_state = {
        "mode": "repo",
        "repo": "demo",
        "slug": "parent-ws",
        "session_id": "parent-1",
    }

    out = CHILDSPACE.ensure_child_workspace("child-session-abc123", parent_state)

    assert out["ok"] is True, out
    assert out["slug"] == "parent-ws-child-abc123-2", out
    assert created == ["parent-ws-child-abc123", "parent-ws-child-abc123-2"], created


def test_a_child_slug_collision_binds_to_the_disambiguated_slug(tmp_path, monkeypatch):
    """The child is BOUND to the slug that was actually created.

    Prevents: the disambiguation creating a workspace and the child
    still being pointed at the taken one - the child would then work in
    a tree that belongs to somebody else, which is the collision this
    whole feature exists to prevent.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(STORE, "find_repo", lambda name: repo)
    monkeypatch.setattr(STORE, "is_jj", lambda path: True)

    def _fake_create(repo_name, slug, purpose=""):
        path = tmp_path / "ws" / slug
        path.mkdir(parents=True, exist_ok=True)
        STORE.write_record(
            path,
            {
                "repo": repo_name,
                "slug": slug,
                "path": str(path),
                "vcs": "jj",
                "purpose": purpose,
                "sessions": [],
                "created_at": "2024-01-01T00:00:00",
            },
        )
        return {"ok": True, "repo": repo_name, "slug": slug, "path": str(path)}

    monkeypatch.setattr(STORE, "create", _fake_create)
    bound: list[str] = []
    monkeypatch.setattr(
        STORE, "bind_session", lambda path, sid: bound.append(f"{sid}->{path}")
    )

    parent_state = {
        "mode": "repo",
        "repo": "demo",
        "slug": "parent-ws",
        "session_id": "parent-1",
    }

    out = CHILDSPACE.ensure_child_workspace("child-session-abc123", parent_state)

    assert out["ok"] is True, out
    assert bound, "the child was never bound"
    assert "parent-ws-child-abc123" in bound[0], bound


# ── F9 / F10: the ws_new tool schema and the injected prompt ───────────────


def _ws_new_schema():
    for name, schema, _handler in PLUGIN._ws_tools():
        if name == "ws_new":
            return schema
    raise AssertionError("ws_new is not registered")


def test_the_ws_new_schema_does_not_require_a_slug():
    """``slug`` is optional: the code defaults it.

    Prevents: the tool rejecting the call the plugin's own prompt tells
    the agent to make. The schema listed ``slug`` in ``required`` while
    the handler derives it from the session id - so a well-formed call
    was rejected by the schema validator before the handler ever ran.
    """
    schema = _ws_new_schema()
    params = schema["parameters"]
    assert "slug" not in params.get("required", []), schema
    assert "slug" in params["properties"], schema
    assert "defaults to" in params["properties"]["slug"]["description"], schema


def test_the_injected_prompt_does_not_say_the_title_names_the_workspace():
    """The prompt must match the code: the title is metadata only.

    Prevents: the agent being told to derive a workspace name from the
    title. The code names the workspace after the SESSION ID (twice, and
    explicitly: "the path is derived from the id, never from the session
    title, because titles get renamed"). The prompt said the opposite -
    "it becomes the workspace name" - and the example repeated it.
    """
    prompt = PLUGIN._PROMPT_TEMPLATE
    assert "it becomes the workspace name" not in prompt, prompt
    assert "metadata only" in prompt, prompt


# ── F11: the repo_roots deprecation hint tells the truth ────────────────────


def test_the_repo_roots_deprecation_hint_does_not_claim_discovery_follows_repo_dir(
    tmp_path, monkeypatch,
):
    """The hint says discovery follows ``repo_dir``, not ``repo_roots``.

    Prevents: an operator following the hint into a broken setup. The
    hint used to claim "repo discovery still follows settings.repo_roots",
    which is now false: discovery walks ``repo_dir`` and ignores
    ``repo_roots`` entirely (the key is still PARSED, so an old config
    does not crash the plugin, but nothing reads it). An operator who
    read the old hint would put their repos under ``repo_roots``, expect
    them to be listed, and find nothing.

    WHY this test changed shape: the discovery change (issue #37) made
    ``repo_dir`` the only root discovery walks, so the hint had to
    change with it. A hint that lies in the other direction is just as
    broken - hence the assertions on the new wording.
    """
    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(tmp_path / "repos"))
    hints: list[str] = []
    monkeypatch.setattr(SETTINGS, "deprecation_hint", lambda key, msg: hints.append(msg))

    config = {
        "plugins": {
            "entries": {
                SETTINGS.PLUGIN_ID: {
                    "settings": {"repo_roots": [str(tmp_path)]},
                },
            },
        },
    }
    SETTINGS.settings(config)

    joined = " ".join(hints)
    assert joined, "no deprecation hint was raised for repo_roots"
    assert "discovery now follows settings.repo_dir" in joined, joined
    assert "IGNORED" in joined, joined
    # The old wording claimed the opposite - it must be gone.
    assert "discovery still follows" not in joined, joined


# ── F12: a skipped git worktree is not reported as reclaimed ────────────────


def test_a_skipped_git_worktree_is_bucketed_as_skipped(tmp_path, monkeypatch):
    """A skip is a skip, whatever the call's own ``ok`` flag says.

    Prevents: the summary claiming a tree was reclaimed while it is
    still on disk. ``reclaim_child_workspace`` reports a git-only
    worktree with ``ok=True`` (the call did what it was asked to do - it
    left the tree alone) and ``reason="skipped_git_worktree"``.
    Bucketing on ``ok`` alone filed it under "reclaimed".
    """
    ws = tmp_path / "repos" / ".workspaces" / "demo" / "repo-child-gwt"
    ws.mkdir(parents=True)
    monkeypatch.setattr(STORE, "repo_dir", lambda: tmp_path / "repos")
    monkeypatch.setattr(
        STORE,
        "settings",
        lambda: {
            "repo_dir": str(tmp_path / "repos"),
            "workspace_root": str(tmp_path / "repos" / ".workspaces"),
            "repo_roots": [str(tmp_path)],
        },
    )
    STORE.write_record(
        ws,
        {
            "repo": "repo",
            "slug": ws.name,
            "path": str(ws),
            "vcs": "jj",
            "purpose": STORE.CHILD_PURPOSE_PREFIX + "parent-1",
            "sessions": ["child-1"],
            "created_at": "2020-01-01T00:00:00",
        },
    )
    monkeypatch.setattr(
        STORE,
        "reclaim_child_workspace",
        lambda path, **kwargs: {
            "ok": True,
            "slug": ws.name,
            "path": str(ws),
            "repo": "repo",
            "snapshot_commit_id": "",
            "was_dirty": False,
            "already_gone": False,
            "dry_run": False,
            "reason": STORE.RECLAIM_SKIPPED_GIT_WORKTREE,
            "detail": "the workspace is a git-only worktree: left alone",
        },
    )

    out = STORE.reclaim_children([ws.name])

    assert out["reclaimed"] == [], out
    assert len(out["skipped"]) == 1, out
    assert out["skipped"][0]["reason"] == STORE.RECLAIM_SKIPPED_GIT_WORKTREE, out


# ── F15: gc_apply - the only path that can retire a USER workspace ──────────


def _user_workspace(tmp_path, monkeypatch, *, slug="my-work", sessions=("sid-1",)):
    """A USER workspace (no ``child:`` purpose) with a record."""
    root = tmp_path / "repos" / ".workspaces" / "demo"
    ws = root / slug
    ws.mkdir(parents=True)
    STORE.write_record(
        ws,
        {
            "repo": "demo",
            "slug": slug,
            "path": str(ws),
            "vcs": "jj",
            "purpose": "",
            "sessions": list(sessions),
            "created_at": "2020-01-01T00:00:00",
        },
    )
    monkeypatch.setattr(STORE, "repo_dir", lambda: tmp_path / "repos")
    monkeypatch.setattr(
        STORE,
        "settings",
        lambda: {
            "repo_dir": str(tmp_path / "repos"),
            "workspace_root": str(tmp_path / "repos" / ".workspaces"),
            "repo_roots": [str(tmp_path)],
        },
    )
    monkeypatch.setattr(STORE, "find_repo", lambda name: tmp_path / "repo")
    # The unregister step is a jj subprocess against a repo that does not
    # exist here. Stub it: what these tests are about is the GUARDS
    # around the retirement, not jj's workspace bookkeeping.
    monkeypatch.setattr(
        STORE._vcs, "forget_workspace", lambda repo, slug, path, vcs="auto": None
    )
    return ws


def test_gc_apply_retires_a_clean_ended_user_workspace(tmp_path, monkeypatch):
    """A finished, clean, pushed, PR-free user workspace is retired.

    Prevents: ``gc_apply()`` - the only path that can retire a USER
    workspace - shipping with NO test coverage at all. It is the most
    dangerous function in the plugin: it deletes a human's tree.

    WHY the retirement is safe here: the session ended, the tree is
    clean, the commits are pushed, and there is no open PR on the
    branch - so nothing is lost by retiring it.
    """
    ws = _user_workspace(tmp_path, monkeypatch)
    STORE.write_state(
        "sid-1",
        {
            "mode": "repo",
            "repo": "demo",
            "slug": ws.name,
            "path": str(ws),
            "session_ended_at": "2020-01-02T00:00:00",
        },
    )
    monkeypatch.setattr(STORE, "dirty", lambda path: False)
    monkeypatch.setattr(STORE, "open_prs", lambda path: ([], True))
    monkeypatch.setattr(STORE, "unpushed_commits", lambda path: False)
    monkeypatch.setattr(STORE, "_workspace_branches", lambda path: ["feature/x"])

    done = STORE.gc_apply()

    assert done, "nothing was retired"
    assert not ws.exists(), "the retired workspace is still on disk"
    assert STORE.db.read_workspace(str(ws)) is None, "the record was not deleted"


def test_gc_apply_refuses_an_unpushed_user_workspace(tmp_path, monkeypatch):
    """Unpushed commits are the end of the line: no retirement.

    Prevents: ``gc_apply()`` deleting work that exists only on this
    machine. The workspace is clean and finished, but its commits were
    never pushed - retiring it would take the only copy with it.
    """
    ws = _user_workspace(tmp_path, monkeypatch)
    STORE.write_state(
        "sid-1",
        {
            "mode": "repo",
            "repo": "demo",
            "slug": ws.name,
            "path": str(ws),
            "session_ended_at": "2020-01-02T00:00:00",
        },
    )
    monkeypatch.setattr(STORE, "dirty", lambda path: False)
    monkeypatch.setattr(STORE, "open_prs", lambda path: ([], True))
    monkeypatch.setattr(STORE, "unpushed_commits", lambda path: True)
    monkeypatch.setattr(STORE, "_workspace_branches", lambda path: ["feature/x"])

    done = STORE.gc_apply()

    assert not done, done
    assert ws.exists(), "an unpushed workspace was retired"
    assert STORE.db.read_workspace(str(ws)) is not None, "the record was deleted"


def test_gc_apply_refuses_a_workspace_with_a_live_session(tmp_path, monkeypatch):
    """A live session's workspace is never retired.

    Prevents: the sweep deleting the tree of a session that is still
    working in it. The workspace is clean, pushed and PR-free, but its
    session never ended - so the workspace is tier3 ("in use") and must
    not be retired.
    """
    ws = _user_workspace(tmp_path, monkeypatch)
    # No session_ended_at: the session is LIVE.
    STORE.write_state(
        "sid-1",
        {
            "mode": "repo",
            "repo": "demo",
            "slug": ws.name,
            "path": str(ws),
        },
    )
    monkeypatch.setattr(STORE, "dirty", lambda path: False)
    monkeypatch.setattr(STORE, "open_prs", lambda path: ([], True))
    monkeypatch.setattr(STORE, "unpushed_commits", lambda path: False)
    monkeypatch.setattr(STORE, "_workspace_branches", lambda path: ["feature/x"])

    done = STORE.gc_apply()

    assert not done, done
    assert ws.exists(), "a live session's workspace was retired"
    assert STORE.db.read_workspace(str(ws)) is not None, "the record was deleted"


def test_gc_apply_keeps_going_when_one_retirement_explodes(tmp_path, monkeypatch):
    """One failing retirement does not abort the whole sweep.

    Prevents: a single broken workspace stopping the sweep for every
    other workspace. ``gc_apply`` catches ``WorkspaceError`` around
    ``retire()`` - the REFUSALS - but ``retire`` can also fail with an
    ordinary exception (a sqlite error in the row deletion, an OSError
    from the archive write). That exception escaped the loop, so the
    sweep died on the first bad workspace and every workspace after it
    was never retired - the sweep silently does less and less over time.

    WHY this matters more than it looks: ``gc_apply`` is the only path
    that may retire a USER workspace. A sweep that dies halfway leaves
    exactly the stale trees the sweep exists to remove.
    """
    first = _user_workspace(tmp_path, monkeypatch, slug="aaa-work", sessions=("sid-a",))
    second = _user_workspace(tmp_path, monkeypatch, slug="bbb-work", sessions=("sid-b",))
    for sid, ws in (("sid-a", first), ("sid-b", second)):
        STORE.write_state(
            sid,
            {
                "mode": "repo",
                "repo": "demo",
                "slug": ws.name,
                "path": str(ws),
                "session_ended_at": "2020-01-02T00:00:00",
            },
        )
    monkeypatch.setattr(STORE, "dirty", lambda path: False)
    monkeypatch.setattr(STORE, "open_prs", lambda path: ([], True))
    monkeypatch.setattr(STORE, "unpushed_commits", lambda path: False)
    monkeypatch.setattr(STORE, "_workspace_branches", lambda path: ["feature/x"])

    real_retire = STORE.retire

    def _exploding_retire(slug, **kwargs):
        if slug == first.name:
            raise RuntimeError("the disk caught fire")
        return real_retire(slug, **kwargs)

    monkeypatch.setattr(STORE, "retire", _exploding_retire)

    done = STORE.gc_apply()

    assert first.exists(), "the failing workspace was retired anyway"
    assert not second.exists(), (
        "the sweep stopped at the first failure - the workspace after it was never retired"
    )
    assert done, done
