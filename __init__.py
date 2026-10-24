"""hermes-treeps: one jj workspace per session, enforced.

The gate is the point. Until a session resolves either a repo (``ws_new``) or
"just chatting" (``ws_chatting``), the tools that could touch a filesystem are
refused with a message telling the model exactly what to call. That makes
"which repo are we in?" a decision the user has to make, not something the agent
infers - which is what produced six unattributable pylon directories.

"Just chatting" is deliberately unrestricted: cwd is HERMES_HOME and every tool
stays available. The only thing it changes is the standing instruction to offer a
workspace when the conversation turns into repository work.

The Telegram picker (inline keyboard, taps, /new wiring) lives in
`telegram_picker.py`; this module owns tools, the gate, the CLI and the prompt.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from . import childspace, contract, prepare, store
from . import telegram_picker as picker

logger = logging.getLogger(__name__)

# sid of the most recent delegate_task DISPATCH (the parent), and when. Only
# dispatches move it - ordinary tool calls from other sessions cannot clobber.
_LAST_DELEGATION = {"sid": "", "at": 0.0}

# {repos} is filled at register() from the live candidate list: a hardcoded
# example here went stale the first time a repo moved.
_PROMPT_TEMPLATE = """\
## Workspace discipline (hermes-treeps)

This session starts unresolved. Repository work - anything that reads, writes or
runs commands in a checkout - requires a workspace first:

- `ws_new(repo, slug)` creates `/Git/hermes/workspaces/<repo>/<slug>` and binds
  this session to it. Use for any real work: code, docs, config, git. Omit slug
  and it defaults to `<repo>-<last 6 of the session id>` - the path is derived
  from the id, never from the session title, because titles get renamed.
- `ws_chatting()` declares a talk-only session: cwd stays HERMES_HOME and all
  tools stay available, but nothing repo-shaped should be edited.
- A repo that is not on disk but lives on GitHub/Forgejo goes through
  `ws_new(repo, slug, create_repo=true)`, and the user must approve the clone
  first - never clone on your own initiative.

On the first turn, ask BOTH questions in a single line (never two messages, never
a menu): the session TITLE (required - it becomes the workspace name), and which
repo - or just chatting. Example: "Session title? (it names the workspace) - and
which repo: {repos}, an external path, a new repo, or just chatting?"

If a chatting session drifts into repository work (a path under /Git, a clone, a
git/jj command, a PR), say one line: "this looks like repo work - want a
workspace for it?" and offer `ws_new <repo> <slug>`. Do not silently start
editing.

Never create a workspace without the user's choice, and never work in another
session's workspace.
"""


def _ws_tools() -> tuple[tuple[str, dict[str, Any], Any], ...]:
    new_schema = {
        "name": "ws_new",
        "description": (
            "Create this session's own workspace for a repo and bind the session to it. "
            "Path: /Git/hermes/workspaces/<repo>/<slug>. Call this before repo work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "repo name under /Git, e.g. hermes-pylon"},
                "slug": {"type": "string", "description": "session name, lowercase - use the session TITLE slugified"},
                "title": {"type": "string", "description": "the session title the user gave (required when known)"},
                "purpose": {"type": "string", "description": "one line: what this session is for"},
                "rebind": {"type": "boolean", "description": "true to move this session from its current workspace to the new one"},
                "path": {"type": "string", "description": "absolute path to an EXISTING repo outside /Git - binds it directly, no checkout"},
                "create_repo": {
                    "type": "boolean",
                    "description": (
                        "true to make /Git/<repo> exist first: clones it when the repo "
                        "lives on GitHub/Forgejo, otherwise creates a brand-new project. "
                        "Only pass this after the user approved the clone."
                    ),
                },
            },
            "required": ["slug"],
        },
    }
    chat_schema = {
        "name": "ws_chatting",
        "description": "Declare this a talk-only session: no workspace, cwd stays HERMES_HOME, all tools available.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string", "description": "the session title the user gave"}},
        },
    }
    rename_schema = {
        "name": "ws_rename",
        "description": "Rename this session's workspace (moves the directory, fixes records). Use when the user gives a title after a picker tap.",
        "parameters": {
            "type": "object",
            "properties": {
                "new_slug": {"type": "string", "description": "new name, lowercase"},
                "slug": {"type": "string", "description": "workspace to rename; defaults to this session's"},
            },
            "required": ["new_slug"],
        },
    }
    status_schema = {
        "name": "ws_status",
        "description": "Show this session's workspace binding and every workspace under the root.",
        "parameters": {"type": "object", "properties": {}},
    }
    doctor_schema = {
        "name": "ws_doctor",
        "description": (
            "Diagnose why workspace resolution failed: binding state (unbound / stale / bound), "
            "the recorded path and whether it still exists, candidate repos, and whether jj/gh "
            "are resolvable. forget=true drops a dead record only - the session is then unbound "
            "and still gated until ws_new or ws_chatting is called."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "forget": {"type": "boolean", "description": "drop this session's record, making it unbound"},
            },
        },
    }
    return (
        ("ws_new", new_schema, _tool_new),
        ("ws_chatting", chat_schema, _tool_chatting),
        ("ws_rename", rename_schema, _tool_rename),
        ("ws_status", status_schema, _tool_status),
        ("ws_doctor", doctor_schema, _tool_doctor),
    )


def _session_id(kw: dict[str, Any]) -> str:
    for key in ("session_id", "session", "durable_session_key"):
        if kw.get(key):
            return str(kw[key])
    import os

    return os.environ.get("HERMES_SESSION_ID", "") or "unknown"


def _match_remote(entries: list[dict[str, Any]], repo: str) -> dict[str, Any] | None:
    """The catalogue entry for `repo`, if it lives on GitHub or Forgejo.

    Only remote sources count: the catalogue's fallback lists LOCAL repos
    ("source": "local"), and matching one of those would claim a repo "exists
    remotely" for a tree that is already on disk.
    """
    want = str(repo or "").strip().lower()
    if not want:
        return None
    for entry in entries:
        if str(entry.get("source") or "").lower() not in {"github", "forgejo"}:
            continue
        if str(entry.get("name") or "").strip().lower() == want:
            return entry
    return None


def _remote_names(entries: list[dict[str, Any]]) -> list[str]:
    """The names of the remote repos the catalogue knows about."""
    names = {
        str(e.get("name") or "").strip()
        for e in entries
        if str(e.get("source") or "").lower() in {"github", "forgejo"}
    }
    names.discard("")
    return sorted(names)


def _remote_lookup(repo: str) -> tuple[dict[str, Any] | None, str, list[str]]:
    """(entry, error, remote_names) for `repo` from the remote catalogue.

    Never raises: the catalogue is a diagnostic, and a dead network must not
    turn into a traceback inside a tool answer.
    """
    try:
        entries, err = store.remote_catalogue(limit_per_source=5)
    except Exception as exc:  # defensive: the catalogue must never break ws_new
        logger.debug("remote catalogue lookup failed", exc_info=True)
        return None, f"remote catalogue unavailable: {type(exc).__name__}: {exc}", []
    return _match_remote(entries, repo), err, _remote_names(entries)


# ── session cwd pinning ───────────────────────────────────────────────────────
# The gate blocks tools while the session still sits in the gateway's own
# directory. The host can instead PIN a logical cwd per session
# (agent.runtime_cwd._SESSION_CWD, read by resolve_agent_cwd()), which is what
# Claude Code's --worktree and its jj equivalent do: the tool then physically
# runs in the workspace. The gate stays as the second line of defence.

_PIN_LOGGED = {"at": False}


def _pin_session_cwd(path: str | os.PathLike[str] | None) -> None:
    """Pin this session's logical cwd to a workspace path. Never raises."""
    try:
        from agent.runtime_cwd import set_session_cwd
    except Exception:
        if not _PIN_LOGGED["at"]:
            _PIN_LOGGED["at"] = True
            logger.debug("agent.runtime_cwd unavailable: the gate stays the only barrier")
        return
    try:
        set_session_cwd(str(path or ""))
    except Exception:
        logger.debug("could not pin the session cwd", exc_info=True)


def _unpin_session_cwd() -> None:
    try:
        from agent.runtime_cwd import clear_session_cwd
    except Exception:
        return
    try:
        clear_session_cwd()
    except Exception:
        logger.debug("could not clear the session cwd", exc_info=True)


def _pinned_workspace() -> str:
    """The pinned cwd when it really points inside the workspace root, else "".

    Deliberately narrow: a pinned path only counts as a workspace when it lives
    under the configured workspace_root. Anything else (the gateway dir, /, a
    hand-pinned tmp path) must fall through to the gate.
    """
    try:
        from agent.runtime_cwd import scoped_session_cwd

        raw = scoped_session_cwd()
        if not raw:
            return ""
        candidate = Path(raw).resolve()
        root = Path(store.settings()["workspace_root"]).resolve()
    except Exception:
        return ""
    if candidate == root or root not in candidate.parents:
        return ""
    return str(candidate) if candidate.is_dir() else ""


def _tool_new(args: dict[str, Any], **kw: Any) -> str:
    try:
        sid = _session_id(kw)
        title = str(args.get("title") or "").strip()
        slug = str(args.get("slug") or "").strip()

        existing = store.read_state(sid)
        path = str(args.get("path") or "").strip()
        repo = str(args.get("repo") or "").strip()

        if not path and not repo:
            return json.dumps({"ok": False, "error": "ws_new needs repo (or path for an external repo)"})

        if not slug and not path:
            # Stable naming: the path follows the session id, never the title.
            # Titles get renamed constantly and there is no rename hook to keep a
            # path in sync; an id-derived slug needs no syncing at all.
            slug = f"{repo}-{sid[-6:]}" if sid and sid != "unknown" else repo
        # The comparison is on (repo, slug), not the slug alone: two repos can
        # share a slug, and comparing slugs alone let a session bound to repo A
        # silently bind repo B's workspace of the same name. The requested slug
        # is sanitised the same way create() will sanitise it, so "same
        # workspace, different spelling" is not read as a re-bind.
        #
        # The confirmation covers BOTH directions: a session bound to a repo
        # workspace and then asked for an external path (or the other way
        # round) is the same silent re-binding, so `mode in {repo, external}`
        # is compared against the request in one place.
        if (
            existing
            and existing.get("mode") in {"repo", "external"}
            and not args.get("rebind")
            and _binding_key(existing) != _requested_key(args, repo, slug)
        ):
            # 1:1 by default; re-binding is explicit, never a silent append.
            return json.dumps({
                "ok": False,
                "needs_confirmation": True,
                "error": (
                    f"this session is already bound to {existing.get('repo')}/{existing.get('slug')}. "
                    f"Rebind to {args.get('repo') or args.get('path')}? The old workspace stays, just "
                    "detached. Ask the user, then call ws_new again with rebind=true."
                ),
            })

        if path:
            state = store.bind_external(path, sid, title=title)
            _detach_previous(existing, state["path"], sid, args.get("rebind"))
            _pin_session_cwd(state["path"])
            return json.dumps({"ok": True, "mode": "external", "path": state["path"]})

        # Idempotence: re-requesting the workspace this session is ALREADY
        # bound to must not try to create it a second time. create() refuses
        # an existing directory, so without this a harmless repeat of the
        # same request came back as an error ("... already exists") even
        # though the session's binding was already correct.
        requested = _requested_key(args, repo, slug)
        if existing and _binding_key(existing) == requested:
            existing_path = Path(str(existing.get("path") or ""))
            if existing_path.is_dir():
                out = {"ok": True, "path": str(existing_path)}
                if existing.get("mode") == "repo":
                    out["vcs"] = (store.read_record(existing_path) or {}).get("vcs", "jj")
                else:
                    out["mode"] = "external"
                return json.dumps(out)

        # Is the repo on disk at all? "not on disk" and "does not exist" are
        # different answers, and only the first one has a next step: the repo
        # may live on GitHub/Forgejo and need an approved clone.
        on_disk = True
        disk_error = ""
        try:
            store.find_repo(repo)
        except store.WorkspaceError as e:
            if "invalid repo name" in str(e):
                raise
            on_disk = False
            disk_error = str(e)

        if not on_disk:
            entry, remote_error, remote_names = _remote_lookup(repo)
            if entry is not None:
                remote = {
                    "name": str(entry.get("name") or repo),
                    "owner": str(entry.get("owner") or ""),
                    "url": str(entry.get("url") or ""),
                    "source": str(entry.get("source") or ""),
                }
                if not args.get("create_repo"):
                    # Stop here: cloning writes to the user's disk, so it needs
                    # the user's approval first. Prevents (P2-3): the tools
                    # only ever mentioned local /Git roots, so an agent whose
                    # repo lived on a remote cloned it on its own initiative.
                    dest = store.clone_target(repo)
                    return json.dumps({
                        "ok": False,
                        "needs_confirmation": True,
                        "remote": remote,
                        "error": (
                            f"repo {repo!r} is not on disk but exists on "
                            f"{remote['source']} as {remote['owner']}/{remote['name']}. "
                            f"Ask the user, then call ws_new again with create_repo=true "
                            f"(it clones {remote['url']} into {dest})."
                        ),
                    })
                # create_repo=true: the user approved the clone.
                store.clone_repo(repo, str(entry.get("url") or ""))
            elif args.get("create_repo"):
                # create_repo=true for a name nobody knows: keep the legacy
                # behaviour of making a brand-new, empty repo of that name.
                store.create_repo(repo)
            else:
                # Neither on disk nor on a remote: the repo does not exist.
                # Name the remotes that DO exist so the agent stops guessing
                # names, and say plainly when the catalogue could not be read.
                if remote_error:
                    tail = f" ({remote_error})."
                elif remote_names:
                    tail = " Remote repos available: " + ", ".join(remote_names) + "."
                else:
                    tail = " No remote repos found in the catalogue."
                return json.dumps({"ok": False, "error": disk_error + tail})

        record = store.create(repo, slug, purpose=str(args.get("purpose") or ""))
        # WHY the record's slug/path and not our own copy: create() SANITISES
        # the slug ("Fix The Widget" -> "fix-the-widget"). Binding the raw slug
        # pointed at a path that does not exist, so the session stayed unbound
        # and the created workspace leaked, unbound, forever.
        bound_slug = str(record.get("slug") or slug)
        bound_path = str(record.get("path") or store.workspace_path(repo, bound_slug))
        store.bind_session(Path(bound_path), sid)
        # Rebinding detaches the PREVIOUS workspace: otherwise the old tree
        # still lists this session as live, and retire() refuses it forever
        # while it also stays on disk, unowned.
        _detach_previous(existing, bound_path, sid, args.get("rebind"))
        state = {"mode": "repo", "repo": repo, "slug": bound_slug, "path": bound_path}
        if title:
            state["title"] = title
        store.write_state(sid, state)
        # The workspace exists and is bound: move the session INTO it, so the
        # next tool call runs there instead of in the gateway directory.
        _pin_session_cwd(bound_path)
        # prepare runs inside the response so a broken setup script is visible
        # in the same call that created the workspace, not on the next one.
        return json.dumps(
            prepare.with_prepare({"ok": True, "path": bound_path, "vcs": record["vcs"]}, repo, bound_path)
        )
    except store.WorkspaceError as e:
        return json.dumps({"ok": False, "error": str(e)})
    except Exception as e:  # never raise into the tool dispatcher
        logger.exception("ws_new failed")
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _detach_previous(
    existing: dict[str, Any] | None, new_path: str, sid: str, rebind: Any
) -> None:
    """Drop this session from the workspace it was bound to before.

    Prevents (P1-6): rebinding used to leave the OLD workspace still listing
    the session as live, so retire() refused it forever ("bound session still
    live") while it sat on disk, owned by nobody and impossible to clean up.
    The old tree is NOT deleted - only the session's claim on it is dropped.
    """
    if not rebind or not existing:
        return
    old_path = str(existing.get("path") or "")
    if not old_path or Path(old_path) == Path(new_path):
        return
    try:
        store.detach(Path(old_path), sid)
    except Exception:
        logger.warning("could not detach %s from %s", sid, old_path, exc_info=True)


def _binding_key(state: dict[str, Any]) -> tuple[str, str]:
    """The (repo, slug) a session is currently bound to.

    An external binding is keyed by its path, so asking for the same external
    path again is not read as a re-bind.
    """
    if state.get("mode") == "external":
        return ("external", str(state.get("path") or ""))
    return (str(state.get("repo") or ""), str(state.get("slug") or ""))


def _requested_key(args: dict[str, Any], repo: str, slug: str) -> tuple[str, str]:
    """The (repo, slug) a ws_new call is asking for, in the same shape.

    Compared against _binding_key() so "same repo, same workspace" is the only
    case that needs no confirmation. The slug is sanitised first, because
    create() sanitises it: without that, re-requesting the same workspace with
    a different spelling looked like a re-bind.
    """
    path = str(args.get("path") or "").strip()
    if path:
        # The same key shape as _binding_key(), which keys an external binding
        # by its RESOLVED path (that is what bind_external stores). Resolving
        # here too means "the same tree, spelled differently" (a trailing
        # slash, a symlink, a relative segment) is recognised as the same
        # binding instead of being read as a re-bind.
        try:
            path = str(Path(path).expanduser().resolve())
        except OSError:
            # Not fatal: compare the path verbatim. This is only the key used to
            # decide "same binding or re-bind?", and bind_external raises the
            # real error (with a far better message) when the path is used.
            pass
        return ("external", path)
    return (repo, store.sanitize_slug(slug))


def _tool_rename(args: dict[str, Any], **kw: Any) -> str:
    try:
        sid = _session_id(kw)
        slug = str(args.get("slug") or "").strip()
        if not slug:
            state = store.read_state(sid) or {}
            slug = str(state.get("slug") or "")
        if not slug:
            return json.dumps({"ok": False, "error": "no workspace bound to this session"})
        return json.dumps({"ok": True, "result": store.rename(slug, str(args.get("new_slug") or "").strip())})
    except store.WorkspaceError as e:
        return json.dumps({"ok": False, "error": str(e)})
    except Exception as e:
        logger.exception("ws_rename failed")
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _tool_chatting(args: dict[str, Any], **kw: Any) -> str:
    try:
        store.mark_chatting(_session_id(kw), title=str(args.get("title") or ""))
        # Chatting means "no workspace": drop the pin, or the session keeps
        # writing into the tree it just declared it was not using.
        _unpin_session_cwd()
        return json.dumps({"ok": True, "mode": "chatting"})
    except Exception as e:
        logger.exception("ws_chatting failed")
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _tool_status(args: dict[str, Any], **kw: Any) -> str:
    session_id = _session_id(kw)
    rows = []
    for item in store.list_workspaces():
        record = item["record"] or {}
        rows.append(
            {
                "repo": item["repo"],
                "slug": item["slug"],
                "path": item["path"],
                "orphan": item["orphan"],
                "last_synced": record.get("last_synced"),
            }
        )
    return json.dumps(
        {"session_id": session_id, "resolved": store.resolve(session_id), "workspaces": rows},
        indent=2,
    )


def _tool_doctor(args: dict[str, Any], **kw: Any) -> str:
    """Answer "why won't this session resolve?" from one place, without shell.

    Deliberately usable while unresolved: it is the repair tool for the gate, so
    it must never be behind the gate.
    """
    sid = _session_id(kw)
    try:
        detail = store.resolve_detail(sid)
        # "Where could this come from?": the local roots we actually scan, and
        # the remote repos the catalogue knows about. Without this a repo that
        # lives on GitHub/Forgejo looks exactly like a repo that does not
        # exist, and the only visible route is to invent a clone.
        local_candidates = store.candidate_repos(limit=8)
        remote_entries, remote_error = store.remote_catalogue(limit_per_source=5)
        out: dict[str, Any] = {
            "session_id": sid,
            "state": detail["state"],
            "reason": detail["reason"],
            "path": detail["path"],
            "path_exists": bool(detail["path"]) and Path(str(detail["path"])).exists(),
            "record": detail["record"],
            "repo_roots": store.repo_roots(),
            # one name, one value: "candidate_repos" and "local_candidates"
            # were both emitted with identical contents, so a consumer could
            # read either and never know which one was canonical.
            "local_candidates": local_candidates,
            "remote_candidates": [
                {
                    "name": str(e.get("name") or ""),
                    "source": str(e.get("source") or ""),
                    "owner": str(e.get("owner") or ""),
                }
                for e in remote_entries
            ],
            "remote_error": remote_error,
            "tools": {"jj": shutil.which("jj"), "gh": shutil.which("gh")},
            "db_path": str(store.db_path()),
            "workspace_root": str(store.root_dir()),
        }
        if args.get("forget"):
            out["forgot"] = store.forget(sid)
        return json.dumps(out, indent=2, default=str)
    except Exception as e:
        logger.exception("ws_doctor failed")
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _readonly_allowed(tool_name: str, kw: dict[str, Any]) -> bool:
    """Read-only diagnosis is never gated: the gate must not block its own repair.

    The invariant is about *editing* and *running commands* in a tree nobody
    chose. Blocking reads is what turned a stale binding into a dead session: the
    agent could not read the plugin's own source to say why resolution failed,
    and neither could it read the state file that held the answer. Logged so the
    relaxation stays visible.
    """
    if tool_name not in {"read_file", "search_files"}:
        return False
    logger.info("gate: allowing read-only %s while unresolved (repair path)", tool_name)
    return True


def _on_pre_tool_call(tool_name: str = "", session_id: str = "", **kw: Any) -> dict[str, Any] | None:
    """Block repo-touching tools until the session resolves a workspace."""
    # Read the recency tracker BEFORE noting this call's session id: the tracker
    # must still hold the PARENT here - noting first would overwrite it with the
    # child's own (unresolved) id and inheritance could never fire.
    parent = picker.last_session(max_age_s=600)
    picker.note_session_id(session_id)
    # Only delegate_task dispatches may set the parent pointer: a plain tool
    # call from ANY other session used to clobber it before the child's first
    # gated call (the test suite caught exactly that).
    dispatch_sid = session_id or _session_id(kw)
    if tool_name == "delegate_task" and dispatch_sid:
        _LAST_DELEGATION["sid"] = dispatch_sid
        _LAST_DELEGATION["at"] = time.time()
    if _LAST_DELEGATION["sid"] and time.time() - _LAST_DELEGATION["at"] > 600:
        _LAST_DELEGATION["sid"] = ""
    try:
        if tool_name not in store.GATED_TOOLS:
            return None
        sid = session_id or _session_id(kw)
        if store.resolve(sid):
            return None
        # Second line of defence: a pinned cwd inside the workspace root means
        # the tool physically runs in a workspace, whatever the database read
        # says - so a corrupted or unreachable binding row cannot fail open.
        if _pinned_workspace():
            return None
        # AIRTIGHT inheritance: only a real delegate_task child may inherit.
        # `is_delegated_child_context()` is a task-local ContextVar set by
        # `_run_single_child` and inherited by the child's tool dispatch - it is
        # False for every fresh session, so a new session can NEVER silently
        # grab another session's workspace. (The recency tracker alone was not
        # airtight: two concurrent delegations could cross-inherit.)
        try:
            from agent.delegation_context import is_delegated_child_context

            is_child = is_delegated_child_context()
        except Exception:
            is_child = False
        # NOTE: there is deliberately NO "recent delegation dispatch" fallback
        # here any more. The old fallback set is_child=True for ANY unresolved
        # session while `_LAST_DELEGATION` was younger than 600 s, so an
        # unrelated pre-existing session inherited another session's workspace
        # (verified live). The upstream bug it worked around (#96483,
        # `_run_single_child` not entering delegated_child_context) is FIXED in
        # the installed Hermes: children run inside delegated_child_context(),
        # so is_delegated_child_context() is authoritative. `_LAST_DELEGATION`
        # is now used ONLY to pick WHICH parent id to inherit from, never to
        # decide whether this session is a child at all.
        if is_child:
            # Parent = the dispatching session (delegation pointer first -
            # ordinary tool calls cannot move it - then the general recency
            # tracker). The durable copy written here covers the child's life.
            parent = _LAST_DELEGATION["sid"] or parent
            parent_state = store.read_state(parent) or {} if parent else {}
            if parent_state.get("mode") not in {"repo", "chatting", "external"}:
                return {
                    "action": "block",
                    "message": (
                        "Delegated child, but the parent session has no workspace binding. "
                        "Report this to the user: the parent must run ws_new/ws_chatting first."
                    ),
                }
            # Isolation first: give the child its OWN workspace in the parent's
            # repo. Two children in one tree is the collision that let an agent
            # in this repo edit another agent's uncommitted work. Inheritance is
            # only the fallback, for a parent with no repo to branch from.
            made = childspace.ensure_child_workspace(sid, parent_state)
            if made.get("ok"):
                _pin_session_cwd(str(made.get("path") or ""))
                logger.info(
                    "gate: delegated child %s got its own workspace %s", sid, made.get("slug")
                )
                return None
            logger.debug("gate: no child workspace for %s (%s)", sid, made.get("reason"))
            # Non-destructive inheritance: the child's own record wins. If the
            # child already has a record (it resolved its own workspace, or it
            # was written by an earlier hook), it is NOT overwritten - the old
            # code copied the parent's dict wholesale and clobbered it.
            if store.read_state(sid) is None:
                inherited = {
                    k: v for k, v in parent_state.items()
                    if k not in {"session_id", "inherited_from", "inherited_at"}
                }
                inherited["inherited_from"] = parent
                inherited["inherited_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                store.write_state(sid, inherited)
            logger.info("gate: delegated child %s inherits binding of %s", sid, parent)
            return None
        try:
            repos = store.candidate_repos(limit=6)
            listing = ", ".join(repos) if repos else "(none found under /Git)"
        except Exception:
            listing = "(could not scan /Git)"
        detail = store.resolve_detail(sid)
        if detail["state"] == "unbound" and _readonly_allowed(tool_name, kw):
            return None
        if detail["state"] == "stale":
            # A dead path is NOT "no choice yet". Say which tree vanished: the
            # user can usually point straight at where it moved.
            return {
                "action": "block",
                "message": (
                    "This session's workspace binding is STALE, not missing: "
                    f"{detail['reason']}\n"
                    "Tell the user the exact path that disappeared, then call ONE of:\n"
                    "  ws_new(path=<new path>, rebind=true) - bind where the tree lives now\n"
                    "  ws_chatting()                  - talk-only, if the work is no longer in a checkout\n"
                    "  ws_doctor(forget=true)         - drop the dead record ONLY, then pick one of the two\n"
                    "                                   above: forgetting alone leaves the session unbound\n"
                    "                                   and still gated\n"
                    f"Candidate repos: {listing}"
                ),
            }
        return {
            "action": "block",
            "message": (
                "No workspace resolved for this session. Ask the user, then call ONE of:\n"
                "  ws_new(repo, slug)  - repo work; creates /Git/hermes/workspaces/<repo>/<slug>\n"
                "  ws_chatting()       - talk-only session, no repo, full tools\n"
                "  repo on GitHub/Forgejo but not on disk: ws_new(repo, slug, create_repo=true) - clone it, after the user approves\n"
                f"Candidate repos: {listing}\n"
                "Ask in one short line offering these repos or 'just chatting'."
            ),
        }
    except Exception:
        logger.exception("workspace gate failed")
        return None  # fail open: a broken plugin must not brick the agent


def _opening_prompt() -> str:
    repos = store.candidate_repos()
    listing = ", ".join(repos) if repos else "(none found under /Git)"
    return (
        "New session, no workspace yet. Ask me exactly one line: which repo to work in "
        f"({listing}) - or 'just chatting' for a talk-only session with no checkout. "
        "Wait for my answer before doing anything else; do not start work, do not guess a repo."
    )


def _on_session_start(session_id: str = "", **kw: Any) -> None:
    """Seed the first turn so the choice prompt is the first thing the user sees.

    On Telegram the picker keyboard is that first thing (see telegram_picker).
    Elsewhere, inject_message() queues user-role input, which needs a
    session_key or an origin - without one there is nowhere to inject, and the
    system-prompt section already asks the question on the next turn.
    """
    try:
        sid = session_id or _session_id(kw)
        picker.note_session_id(sid)
        detail = store.resolve_detail(sid)
        if detail["state"] == "bound":
            # Resumed session: re-pin (a chatting session has no path, and
            # pinning "" would look like a workspace).
            if detail.get("path"):
                _pin_session_cwd(detail["path"])
            else:
                _unpin_session_cwd()
            return  # already resolved once: don't re-ask on every resume
        if detail["state"] == "stale":
            # A dead path must not stay pinned: the session would keep running
            # tools in a directory it no longer owns.
            _unpin_session_cwd()
            # Do NOT go quiet on a dead path: fall through to the picker/inject
            # path so the user gets a fresh choice instead of a session that
            # silently refuses every tool.
            logger.warning("session %s has a stale binding: %s", sid, detail["reason"])

        # Telegram: an inline keyboard is the whole point - one tap, no typing.
        # `platform` arrives as whatever the agent carries (str, enum, or empty),
        # so normalise before matching: a strict == "telegram" silently skipped
        # this whole branch once already.
        if picker.handle_session_start(sid, **kw):
            return

        session_key = kw.get("session_key") or kw.get("durable_session_key")
        origin = kw.get("origin") or kw.get("source")
        if origin is not None and hasattr(origin, "to_dict"):
            origin = origin.to_dict()
        kwargs: dict[str, Any] = {}
        if session_key:
            kwargs["session_key"] = str(session_key)
        elif isinstance(origin, dict) and origin:
            kwargs["origin"] = origin
        else:
            # No session_key and no origin: nothing to inject into. The system
            # prompt section already asks the question, which is why this path
            # needs no setting any more.
            return
        _CTX.inject_message(_opening_prompt(), role="user", **kwargs)
    except Exception:
        logger.debug("workspace opening prompt skipped", exc_info=True)


def _on_session_finalize(session_id: str = "", **kw: Any) -> None:
    """Stamp the session state at the /new boundary: the gateway fires this with
    the session about to end, so `ws list` can tell a normal ending from a
    vanished session (a true orphan has no state file at all)."""
    try:
        sid = session_id or _session_id(kw)
        state = store.read_state(sid)
        if state is None:
            return  # never resolved a workspace; nothing to stamp
        if isinstance(state, dict):
            state.setdefault("mode", "unknown")
            state["session_ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            store.write_state(sid, state)
    except Exception:
        logger.debug("could not stamp session end", exc_info=True)


def _register_cli(subparser) -> None:
    sub = subparser.add_subparsers(dest="ws_action")

    p_new = sub.add_parser("new", help="create a workspace + session record")
    p_new.add_argument("repo")
    p_new.add_argument("slug")
    p_new.add_argument("--purpose", default="")

    sub.add_parser("list", help="list workspaces and their records")

    p_status = sub.add_parser("status", help="detail for one workspace")
    p_status.add_argument("slug")

    p_retire = sub.add_parser("retire", help="archive + forget (refuses open PRs / dirty / unpushed; --force for unknowns)")
    p_retire.add_argument("slug")
    p_retire.add_argument("--force", action="store_true", help="retire even when PR/push checks could not run")
    p_retire.add_argument("--ultraforce", action="store_true", help="retire despite open PRs or a dirty tree (archive is still kept)")

    p_gc = sub.add_parser("gc", help="classify reclaimable workspaces (dry-run unless --apply)")
    p_gc.add_argument("--apply", action="store_true", help="retire the tier1 set (ended + clean + pushed + no open PRs)")

    p_chat = sub.add_parser("chatting", help="mark a session as talk-only")
    p_chat.add_argument("--session", default="")

    p_mig = sub.add_parser(
        "migrate",
        help="import the legacy JSON records into the plugin database (idempotent)",
    )
    p_mig.add_argument(
        "--purge",
        action="store_true",
        help="after importing, delete the in-tree .hermes/session.json files (backed up first)",
    )
    p_mig.add_argument(
        "--dry-run",
        action="store_true",
        help="with --purge: list the files that would be deleted and delete none",
    )

    subparser.set_defaults(ws_action="list")


def _cli(args) -> int:
    action = getattr(args, "ws_action", "list") or "list"
    try:
        if action == "new":
            record = store.create(args.repo, args.slug, purpose=args.purpose)
            print(record["path"])
            return 0
        if action == "list":
            items = store.list_workspaces()
            if not items:
                print(f"no workspaces under {store.root_dir()}")
                return 0
            for item in items:
                record = item["record"] or {}
                flags = []
                if item["orphan"]:
                    flags.append("ORPHAN(no session.json)")
                else:
                    # A workspace whose every session was finalised is done, not
                    # orphaned - distinguish that from a vanished session.
                    ends = []
                    for sid in record.get("sessions") or []:
                        st = store.read_state(str(sid)) or {}
                        if st.get("session_ended_at"):
                            ends.append(str(st["session_ended_at"]))
                    if ends and len(ends) == len(record.get("sessions") or []):
                        flags.append(f"session ended {max(ends)}")
                    elif record.get("last_synced"):
                        flags.append(f"synced {record['last_synced']}")
                print(f"{item['repo']}/{item['slug']}  {item['path']}  {' '.join(flags)}".rstrip())
            return 0
        if action == "status":
            ws = store.find_workspace(args.slug)
            record = store.read_record(ws)
            prs, gh_ok = store.open_prs(ws)
            print(json.dumps({"path": str(ws), "record": record, "open_prs": prs, "pr_check_ok": gh_ok}, indent=2, default=str))
            return 0
        if action == "retire":
            print(store.retire(
                args.slug,
                force=getattr(args, "force", False),
                ultraforce=getattr(args, "ultraforce", False),
            ))
            return 0
        if action == "gc":
            tiers = store.gc_tiers()
            titles = {
                "tier1": "safe to retire (session ended, clean, pushed, no open PRs)",
                "tier2": "needs a look first (unverified, orphans, open PRs, or dirty)",
                "tier3": "in use (session still live)",
            }
            total = sum(len(v) for v in tiers.values())
            if total == 0:
                print("no workspaces found - nothing to reclaim")
                return 0
            for name in ("tier1", "tier2", "tier3"):
                rows = tiers[name]
                print(f"\n{titles[name]}: {len(rows)}")
                for row in rows:
                    print(f"  {row['repo']}/{row['slug']}")
                    print(f"    {row['path']}")
                    print(f"    why: {row['why']}")
            if getattr(args, "apply", False):
                done = store.gc_apply()
                print(f"\nretired: {len(done)}" if done else "\nretired: nothing eligible")
                for path in done:
                    print(f"  {path}")
            else:
                print(
                    "\nNothing was deleted. Reclaim the 'safe to retire' set with: "
                    "hermes ws gc --apply"
                )
            return 0
        if action == "chatting":
            store.mark_chatting(args.session or "cli")
            print("session marked as chatting")
            return 0
        if action == "migrate":
            out = store.migrate_legacy(dry_run=bool(getattr(args, "dry_run", False)))
            print(json.dumps(out, indent=2, default=str))
            if not getattr(args, "purge", False):
                return 0
            purge = store.purge_legacy_records(dry_run=bool(getattr(args, "dry_run", False)))
            print(json.dumps(purge, indent=2, default=str))
            if getattr(args, "dry_run", False):
                print("dry run: nothing was deleted")
            else:
                print(f"purged {len(purge.get('purged') or [])} legacy record(s)")
            return 0
    except store.WorkspaceError as e:
        print(f"refused: {e}")
        return 2
    return 0


def register(ctx) -> None:
    global _CTX
    _CTX = ctx
    for name, schema, handler in _ws_tools():
        ctx.register_tool(name=name, toolset="workspaces", schema=schema, handler=handler, emoji="🗂️")

    ctx.register_cli_command(
        name="ws",
        help="Workspaces: one per session (new/list/status/retire/chatting)",
        setup_fn=_register_cli,
        handler_fn=_cli,
        description=(
            "Session-owned workspaces under /Git/hermes/workspaces/<repo>/<slug>. "
            "Creates the jj workspace and its record in the plugin database together."
        ),
    )

    repos = store.candidate_repos(limit=6)
    listing = ", ".join(repos) if repos else "(none found under /Git)"
    ctx.register_system_prompt_section(
        "workspace-discipline",
        # The delegation contract travels with the prompt: the rules that used to
        # live in a skill file and in the operator's head are now in every session.
        _PROMPT_TEMPLATE.format(repos=listing) + "\n\n" + contract.as_prompt_section(),
        position="after_memory",
    )

    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    picker.register(ctx)
    prepare.register(ctx)
