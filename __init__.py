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
from typing import Any, Mapping

from . import childreport, childspace, contract, migrate_layout, prepare, store
from . import telegram_picker as picker

logger = logging.getLogger(__name__)

# {repos} is filled at register() from the live candidate list: a hardcoded
# example here went stale the first time a repo moved.
_PROMPT_TEMPLATE = """\
## Workspace discipline (hermes-treeps)

This session starts unresolved. Repository work - anything that reads, writes or
runs commands in a checkout - requires a workspace first:

- `ws_new(repo, slug)` creates `<repo_dir>/.workspaces/<repo>/<slug>` and binds
  this session to it. Use for any real work: code, docs, config, git. Omit slug
  and it defaults to `<repo>-<last 6 of the session id>` - the path is derived
  from the id, never from the session title, because titles get renamed.
- `ws_chatting()` declares a talk-only session: cwd stays HERMES_HOME and all
  tools stay available, but nothing repo-shaped should be edited.
- A repo that is not on disk but lives on GitHub/Forgejo goes through
  `ws_new(repo, slug, create_repo=true)`, and the user must approve the clone
  first - never clone on your own initiative.

On the first turn, ask BOTH questions in a single line (never two messages, never
a menu): the session TITLE (required - it is metadata only; the workspace name
is derived from the session id, never from the title, because titles get
renamed), and which repo - or just chatting. Example: "Session title? (metadata
only - the workspace name comes from the session id) - and which repo: {repos},
an external path, a new repo, or just chatting?"

If a chatting session drifts into repository work (a path under a configured repo
root, a clone, a git/jj command, a PR), say one line: "this looks like repo work
- want a workspace for it?" and offer `ws_new <repo> <slug>`. Do not silently
start editing.

Never create a workspace without the user's choice, and never work in another
session's workspace.
"""


def _repo_roots_label() -> str:
    """The configured repo roots, as one readable phrase.

    WHY this exists: the user-facing strings used to hardcode "/Git", which
    was this operator's layout, not the plugin's - a host with a different
    repo_dir was told about a directory that does not exist there. The roots
    are read from the settings, so the text follows the configuration.
    """
    try:
        roots = [str(r) for r in store.settings()["repo_roots"] if str(r)]
    except Exception:
        return "the configured repo roots"
    return ", ".join(roots) if roots else "the configured repo roots"


def _ws_tools() -> tuple[tuple[str, dict[str, Any], Any], ...]:
    new_schema = {
        "name": "ws_new",
        "description": (
            "Create this session's own workspace for a repo and bind the session to it. "
            "Path: <repo_dir>/.workspaces/<repo>/<slug>. Call this before repo work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "repo name under the configured repo roots, e.g. hermes-pylon"},
                "slug": {"type": "string", "description": "workspace name, lowercase; defaults to <repo>-<last 6 of session id> - pass one only if the user asked for a specific name"},
                "title": {"type": "string", "description": "the session title the user gave (required when known)"},
                "purpose": {"type": "string", "description": "one line: what this session is for"},
                "rebind": {"type": "boolean", "description": "true to move this session from its current workspace to the new one"},
                "path": {"type": "string", "description": "absolute path to an EXISTING repo outside the configured repo roots - binds it directly, no checkout"},
                "create_repo": {
                    "type": "boolean",
                    "description": (
                        "true to make <clone_root>/<repo> exist first: clones it when the repo "
                        "lives on GitHub/Forgejo, otherwise creates a brand-new project. "
                        "Only pass this after the user approved the clone."
                    ),
                },
            },
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


def _turn_key_for_log() -> str:
    """The gateway turn key the terminal layer would read, for the pin log."""
    try:
        from tools.terminal_tool import get_current_session_key

        return str(get_current_session_key() or "")
    except Exception:
        return "<unavailable>"


def _pin_session_cwd(
    path: str | os.PathLike[str] | None, sid: str = "", task_id: str = ""
) -> None:
    """Pin this session's logical cwd to a workspace path. Never raises.

    Two layers on purpose: `agent.runtime_cwd.set_session_cwd` moves what
    context resolvers see, while `tools.terminal_tool.record_session_cwd` moves
    where the terminal tool's shell actually starts (terminal_tool resolves cwd
    from its own per-task record and TERMINAL_CWD, never from the ContextVar).
    The terminal record is keyed by session id and is only written when we know
    it: an empty key collapses to "default", which would re-point EVERY
    session without its own record.
    """
    try:
        from agent.runtime_cwd import set_session_cwd
    except Exception:
        if not _PIN_LOGGED["at"]:
            _PIN_LOGGED["at"] = True
            logger.debug("agent.runtime_cwd unavailable: the gate stays the only barrier")
        return
    try:
        set_session_cwd(str(path or ""))
        # One INFO line per pin: the terminal reads its cwd back under one of
        # three keys (agent session id, gateway turn key, child task id) and
        # the live probe showed the child's shell STILL landing in the parent
        # tree - this line is how we find out which key the write missed.
        logger.info(
            "cwd pin: sid=%r task_id=%r turn_key=%r path=%s",
            sid, task_id, _turn_key_for_log(), path,
        )
        if sid:
            from tools.terminal_tool import get_current_session_key, record_session_cwd

            record_session_cwd(sid, str(path or ""))
            # The terminal resolves its cwd from get_current_session_key() -
            # the gateway's turn key ("agent:...:<chat>") - NOT from the agent
            # session id. Record under both: the sid for the file-tools layer,
            # the turn key for the shell. An empty key collapses to "default",
            # which would re-point EVERY session, so it is only written when
            # the key is known.
            key = get_current_session_key()
            if key:
                record_session_cwd(key, str(path or ""))
            # A delegated child's tools read their cwd under the CHILD's
            # task_id ("subagent-N-..."), which the host seeds from the
            # PARENT's record at spawn. Overwrite it with the child's own
            # workspace - verified live: without this, the child's shell runs
            # in the parent's tree even though its binding points elsewhere.
            if task_id:
                record_session_cwd(task_id, str(path or ""))
    except Exception:
        logger.debug("could not pin the session cwd", exc_info=True)


def _unpin_session_cwd(sid: str = "") -> None:
    try:
        from agent.runtime_cwd import clear_session_cwd
    except Exception:
        return
    try:
        clear_session_cwd()
        if sid:
            from tools.terminal_tool import (
                clear_session_cwd as _clear_terminal_cwd,
            )
            from tools.terminal_tool import (
                get_current_session_key,
            )

            _clear_terminal_cwd(sid)
            key = get_current_session_key()
            if key:
                _clear_terminal_cwd(key)
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


def _caller_is_delegated_child() -> bool:
    """Is the CALLING session a delegated child?

    The same resolution the gate uses: ``is_delegated_child_context()`` is a
    task-local ContextVar set by the host's ``_run_single_child`` and
    inherited by the child's tool dispatch. It is False for every fresh
    session.

    WHY there is no module global here: a global can only ever name ONE
    delegation, so two children in flight would read each other's answer.
    Wave A removed ``_LAST_DELEGATION`` for exactly that reason and it must
    not come back in another shape.
    """
    try:
        from agent.delegation_context import is_delegated_child_context

        return bool(is_delegated_child_context())
    except Exception:
        # No delegation context module (a plain session, or a test): the
        # caller is NOT a child. Failing open here would be wrong - it would
        # let an unclassifiable caller mint a user workspace.
        return False


def _forced_child_shape(sid: str) -> tuple[str, str]:
    """The (purpose, slug) a delegated child's workspace MUST carry.

    WHY this exists: on 2026-10-25 a builder child called
    ``ws_new(repo='hermes-treeps', purpose='feat: picker UI - flat tagged
    list, two-step clone, background bind')`` - it passed its own goal text
    as the purpose, and the handler stamped that goal text into the record.
    The workspace then classified as a USER workspace: immediate reclaim
    refused it, ``ws gc --children`` skipped it, and ``gc_apply`` never
    retired it while the session binding existed. The tree and its record
    leaked until a human ran ``jj workspace forget`` and ``rm`` by hand.

    WHY the shape is FORCED rather than merely validated: an agent-run child
    cannot be trusted to label its own workspaces, and an unlabelled one is
    unreclaimable BY CONSTRUCTION - every reclaim path keys on the child
    prefix. So the caller's purpose AND slug are both ignored and the
    child shape is derived from the same sources the gate uses.

    Returns ``("", "")`` when the caller is not a delegated child, or when
    the child's parent cannot be resolved (the gate blocks such a child
    anyway; minting a user workspace here would be the leak again).
    """
    if not _caller_is_delegated_child():
        return "", ""
    parent_session_id = _resolve_child_parent(sid)
    if not parent_session_id:
        return "", ""
    purpose = f"{store.CHILD_PURPOSE_PREFIX}{parent_session_id}"
    # The slug shape matches childspace.ensure_child_workspace exactly:
    # "<parent slug>-child-<last 6 of the child session id>", sanitised.
    parent_slug = ""
    try:
        parent_state = store.read_state(parent_session_id) or {}
        parent_slug = str(parent_state.get("slug") or "")
    except Exception:
        logger.debug("could not read the parent's binding for %s", parent_session_id, exc_info=True)
    raw_slug = f"{parent_slug}-child-{str(sid)[-6:]}" if parent_slug else f"child-{str(sid)[-6:]}"
    slug = store.sanitize_slug(raw_slug)
    if not slug:
        return "", ""
    return purpose, slug


def _tool_new(args: dict[str, Any], **kw: Any) -> str:
    try:
        sid = _session_id(kw)
        title = str(args.get("title") or "").strip()
        slug = str(args.get("slug") or "").strip()
        purpose = str(args.get("purpose") or "")
        # LAYER 1: a delegated child may not mint a user-class workspace.
        # Its purpose and slug are IGNORED and the child shape is forced -
        # see _forced_child_shape() for the live failure this prevents.
        child_purpose, child_slug = _forced_child_shape(sid)
        forced_child = bool(child_purpose)
        if forced_child:
            purpose = child_purpose
            slug = child_slug
        if purpose.startswith("child:") and not forced_child:
            # Reserved: childspace writes this prefix into workspace records,
            # and gc classifies child workspaces by it. A user-set purpose with
            # the same prefix would make a normal workspace look like a
            # throwaway child and get collected on age alone.
            # A FORCED child purpose is the plugin's own stamp, not the
            # caller's - it is exactly what the reserved prefix is for.
            return json.dumps({"ok": False, "error": "purpose: the 'child:' prefix is reserved for delegated-child workspaces"})

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
            _pin_session_cwd(state["path"], sid=sid)
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

        record = store.create(repo, slug, purpose=purpose)
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
        _pin_session_cwd(bound_path, sid=sid)
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
        # Chatting means "no workspace": pin HERMES_HOME rather than clearing,
        # because dropping the override falls back to os.getcwd() - the
        # gateway's own directory - when TERMINAL_CWD is unset.
        _pin_session_cwd(store.hermes_home(), sid=_session_id(kw))
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
            "repo_dir": str(store.repo_dir()),
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


def _candidate_listing() -> str:
    """One line of candidate repos for the gate's block messages."""
    try:
        repos = store.candidate_repos(limit=6)
        roots = ", ".join(store.settings()["repo_roots"])
        return ", ".join(repos) if repos else f"(none found under {roots})"
    except Exception as scan_exc:
        return f"(could not scan the configured repo roots: {type(scan_exc).__name__})"


def _stale_block(detail: dict, listing: str) -> dict:
    """The stale-binding block: name the dead path and the two recovery routes."""
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


def _resolve_child_parent(child_sid: str) -> str:
    """The parent session id of THIS child, from per-child channels only.

    WHY this exists and WHY there is no global any more: the plugin used to
    keep the sid of "the most recent delegate_task dispatch" in a module
    global and read it back when the child's first gated call arrived. Two
    concurrent delegations overwrote that global, so child A resolved its
    parent as B - and A then built its own workspace inside B's repo, with
    B's parent stamped as ``inherited_from``. A module global can only ever
    name ONE parent, so it is structurally incapable of answering "who is
    MY parent" for more than one live child.

    The channels used here are all keyed by the CHILD's own session id, so
    two children can never see each other's parent:

    1. ``child_runs`` - written by the ``subagent_start`` hook, whose payload
       names BOTH the parent and the child. The host fires it synchronously
       when the child agent is constructed and BEFORE the child runs its
       first tool call, so the row is there when the child asks.
    2. the child's own binding row - ``parent_session_id``, written when the
       child's workspace is created. It survives a restart and a re-dispatch.

    WHY NOT the recency tracker: ``picker.last_session()`` is process-global
    too. It is only consulted as a LAST resort, and only when the tracker
    names a session that is the child's own recorded parent - never as a
    guess about who dispatched this child. A child whose parent cannot be
    resolved is BLOCKED, not pointed at a stranger's workspace.
    """
    if not child_sid:
        return ""
    # 1. the child_runs row (subagent_start) - the authoritative per-child map
    try:
        run = store.db.read_child_run(child_sid)
    except Exception:
        logger.debug("could not read the child run row for %s", child_sid, exc_info=True)
        run = None
    if run:
        parent = str(run.get("parent_session_id") or "")
        if parent:
            return parent
    # 2. the child's own binding row
    try:
        state = store.read_state(child_sid) or {}
    except Exception:
        logger.debug("could not read the child's binding for %s", child_sid, exc_info=True)
        state = {}
    for key in ("parent_session_id", "inherited_from"):
        parent = str(state.get(key) or "")
        if parent and parent != child_sid:
            return parent
    # 3. the recency tracker - ONLY when it names the child's own recorded
    #    parent. Never a guess: a stranger's workspace is worse than a block.
    tracked = picker.last_session(max_age_s=600)
    if tracked and tracked == str(state.get("parent_session_id") or ""):
        return tracked
    return ""


def _self_heal_child_record(sid: str, parent_session_id: str) -> bool:
    """Restamp THIS child's own workspace record to the child marker.

    LAYER 2 of the "a child may not mint a user-class workspace" fix.

    WHY this exists: before layer 1, a delegated child could call ws_new with
    its own goal text as the purpose, and the record was stamped with that
    goal text. The workspace then classified as a USER workspace: immediate
    reclaim refused it, ``ws gc --children`` skipped it, and ``gc_apply``
    never retired it while the session binding existed. The tree and its
    record leaked until a human intervened - which is exactly what we had to
    do on 2026-10-25 (``jj workspace forget`` + ``rm`` by hand).

    This is the SELF-HEAL for records that leaked BEFORE the fix: on the
    child's next gated call the record is restamped, so it rejoins the
    reclaim system instead of leaking forever.

    It is ONE UPDATE and it is IDEMPOTENT: a record that already carries the
    child prefix is left alone, so the hot path does not rewrite the record
    on every gated call.

    A user's workspace is never touched: the restamp only happens when the
    session resolves as a delegated child AND the child's own session id is
    in the workspace record's session list - i.e. this child OWNS the
    workspace. A workspace the child merely INHERITED from its parent (the
    chatting/external fallback) belongs to the parent and keeps the parent's
    classification.
    """
    if not sid or not parent_session_id:
        return False
    wanted = f"{store.CHILD_PURPOSE_PREFIX}{parent_session_id}"
    try:
        state = store.read_state(sid) or {}
        path = str(state.get("path") or "")
        if not path or state.get("mode") != "repo":
            return False
        record = store.read_record(Path(path))
        if record is None:
            return False
        if str(record.get("purpose") or "") == wanted:
            return False  # already correct: idempotent no-op
        if str(record.get("purpose") or "").startswith(store.CHILD_PURPOSE_PREFIX):
            # It is SOME child's workspace. Do not re-stamp it: the prefix
            # names a different parent, and rewriting it would move the
            # workspace into another parent's reclaim bucket.
            return False
        # The child must OWN the workspace. A workspace the child merely
        # inherited from its parent is the parent's, and restamping it would
        # put the parent's workspace into the child reclaim bucket.
        sessions = [str(s) for s in (record.get("sessions") or [])]
        if sid not in sessions:
            return False
        record["purpose"] = wanted
        store.write_record(Path(path), record)
        logger.info(
            "gate: restamped leaked workspace %s to the child marker (%s)", path, wanted
        )
        return True
    except Exception:
        # The self-heal is best-effort: a broken bookkeeping path must never
        # gate-block the child.
        logger.debug("could not self-heal the child record for %s", sid, exc_info=True)
        return False


def _on_pre_tool_call(tool_name: str = "", session_id: str = "", **kw: Any) -> dict[str, Any] | None:
    """Block repo-touching tools until the session resolves a workspace."""
    # WHY the recency tracker is NOT read here any more: it is process-global,
    # so it can only ever name ONE session, and using it as a parent source
    # is what let two concurrent delegations cross-inherit. A child's parent
    # is resolved per child, in _resolve_child_parent().
    picker.note_session_id(session_id)
    try:
        if tool_name not in store.GATED_TOOLS:
            return None
        sid = session_id or _session_id(kw)
        bound = store.resolve(sid)
        if bound:
            # A resumed child keeps its workspace and keeps proving life: the
            # heartbeat is written on EVERY gated call, not only on the first
            # one, so a long-running child is never mistaken for an orphan.
            try:
                state = store.read_state(sid) or {}
                if str(state.get("purpose") or "").startswith(store.CHILD_PURPOSE_PREFIX):
                    store.record_child_heartbeat(sid)
                else:
                    # LAYER 2 (self-heal), bound branch: a record this child
                    # created BEFORE the fix carries goal text instead of the
                    # child marker, so it classifies as a USER workspace -
                    # and a bound USER workspace is exactly what gc_apply
                    # never retires. Restamp it here, on the child's next
                    # gated call, so the already-leaked records rejoin the
                    # reclaim system instead of leaking forever.
                    if _caller_is_delegated_child():
                        _self_heal_child_record(sid, _resolve_child_parent(sid))
            except Exception:
                # Broken heartbeat bookkeeping must not gate-block the child.
                pass
            # Resumed after a gateway restart: the runtime pin is gone and
            # on_session_start does not re-fire for a resumed session, so the
            # shell silently falls back to the gateway directory. Re-arm the
            # pin lazily on the first gated call; the pin-present check makes
            # this a no-op for every later call.
            if not _pinned_workspace():
                if bound.get("mode") == "chatting":
                    _pin_session_cwd(store.hermes_home(), sid=sid)
                else:
                    _pin_session_cwd(str(bound.get("path") or ""), sid=sid)
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
        # session while the delegation pointer was younger than 600 s, so an
        # unrelated pre-existing session inherited another session's workspace
        # (verified live). The upstream bug it worked around (#96483,
        # `_run_single_child` not entering delegated_child_context) is FIXED in
        # the installed Hermes: children run inside delegated_child_context(),
        # so is_delegated_child_context() is authoritative.
        if is_child:
            # WHICH parent: resolved from per-child channels only (the
            # child's own child_runs row, its own binding row). The old
            # module global could name only ONE parent, so two concurrent
            # delegations made the second child inherit the first child's
            # parent - and build its workspace in the WRONG repo.
            parent = _resolve_child_parent(sid) or ""
            if not parent:
                # No per-child record of who dispatched this child. Guessing
                # (the recency tracker) is what let two children cross-
                # inherit, so the child is refused instead: a blocked child
                # is recoverable, a child in a stranger's tree is not.
                return {
                    "action": "block",
                    "message": (
                        "Delegated child, but this session's parent could not be "
                        "identified. The plugin records the child->parent mapping "
                        "at dispatch (subagent_start); that record is missing, so "
                        "the child is not allowed to guess a parent's workspace.\n"
                        "Report this to the user: re-dispatch the task, or run it "
                        "directly in the parent session."
                    ),
                }
            parent_state = store.read_state(parent) or {} if parent else {}
            # LAYER 2 (self-heal): a record this child created BEFORE the
            # fix carries goal text instead of the child marker, so it
            # classifies as a USER workspace and no reclaim path ever
            # touches it. Restamp it here - one idempotent UPDATE - so it
            # rejoins the reclaim system on the child's next gated call.
            _self_heal_child_record(sid, parent)
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
                _pin_session_cwd(
                    str(made.get("path") or ""), sid=sid, task_id=str(kw.get("task_id") or "")
                )
                try:
                    state = store.read_state(sid) or {}
                    purpose = str(state.get("purpose") or "")
                    if not purpose:
                        record = store.read_record(Path(str(made.get("path") or ""))) or {}
                        purpose = str(record.get("purpose") or "")
                    if purpose.startswith(store.CHILD_PURPOSE_PREFIX):
                        store.record_child_heartbeat(sid)
                except Exception:
                    # Broken heartbeat bookkeeping must not gate-block the child.
                    pass
                logger.info(
                    "gate: delegated child %s got its own workspace %s", sid, made.get("slug")
                )
                return None
            logger.debug("gate: no child workspace for %s (%s)", sid, made.get("reason"))
            if parent_state.get("mode") == "repo":
                # A repo parent with a failed childspace (git-only repo, jj
                # error) must NOT fall through to inheritance: that puts the
                # child into the PARENT'S tree - the exact two-agents-one-tree
                # collision this branch exists to prevent. Inheritance is only
                # for parents with no repo to branch from (chatting/external).
                return {
                    "action": "block",
                    "message": (
                        "Delegated child could not get its own workspace: "
                        f"{made.get('reason')}. Repo-touching tools are blocked.\n"
                        "Run the child's task directly in this session instead, or "
                        "convert the repo to jj (jj git init --colocate) and re-dispatch."
                    ),
                }
            # Non-destructive inheritance: the child's own record wins. If the
            # child already has a record (it resolved its own workspace, or it
            # was written by an earlier hook), it is NOT overwritten - the old
            # code copied the parent's dict wholesale and clobbered it.
            if store.read_state(sid) is None:
                inherited = {
                    k: v for k, v in parent_state.items()
                    if k not in {"session_id", "inherited_from", "inherited_at", "parent_session_id"}
                }
                inherited["inherited_from"] = parent
                # The explicit parent link, so the NEXT gated call of this
                # child (or of a re-dispatched one) can resolve its parent
                # from its own row instead of from any shared/global state.
                inherited["parent_session_id"] = parent
                inherited["inherited_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                store.write_state(sid, inherited)
            logger.info("gate: delegated child %s inherits binding of %s", sid, parent)
            return None
        # The state is resolved ONCE here and reused by every branch below.
        detail = store.resolve_detail(sid)
        if detail["state"] == "stale":
            # A stale binding must warn EVEN WHEN the cwd is pinned: the pin
            # means the shell runs in a workspace, but the workspace it NAMES
            # is dead - the user is never told if the pin floor wins.
            listing = _candidate_listing()
            return _stale_block(detail, listing)
        if detail["state"] == "unbound" and _readonly_allowed(tool_name, kw):
            return None
        # Second line of defence: a pinned cwd inside the workspace root means
        # the tool physically runs in a workspace, whatever the database read
        # says - so a corrupted or unreachable binding row cannot fail open.
        # Unbound (not stale) only: a stale session was already warned above.
        if _pinned_workspace():
            return None
        try:
            repos = store.candidate_repos(limit=6)
            roots = ", ".join(store.settings()["repo_roots"])
            listing = ", ".join(repos) if repos else f"(none found under {roots})"
        except Exception as scan_exc:
            listing = f"(could not scan the configured repo roots: {type(scan_exc).__name__})"
        return {
            "action": "block",
            "message": (
                "No workspace resolved for this session. Ask the user, then call ONE of:\n"
                f"  ws_new(repo, slug)  - repo work; creates {store.workspaces_base()}/<repo>/<slug>\n"
                "  ws_chatting()       - talk-only session, no repo, full tools\n"
                "  repo on GitHub/Forgejo but not on disk: ws_new(repo, slug, create_repo=true) - clone it, after the user approves\n"
                f"Candidate repos: {listing}\n"
                "Ask in one short line offering these repos or 'just chatting'."
            ),
        }
    except Exception as exc:
        logger.exception("workspace gate failed")
        # A pinned cwd inside the workspace root is independent evidence the
        # session sits in a workspace - the SAME evidence the normal path
        # accepts, so the error path accepts it too. Without this, a session
        # whose cwd is a real workspace is blocked on a db hiccup but allowed
        # the moment the db recovers - the two branches disagreeing about what
        # a pinned cwd proves.
        if _pinned_workspace():
            return None
        # Fail CLOSED for the tools this gate exists to stop. Failing open meant
        # a locked or corrupt database silently allowed `terminal` - the exact
        # hole the audits found. Repair stays possible: read-only tools are
        # allowed, and ws_new / ws_chatting / ws_doctor are not gated tools.
        if tool_name in store.GATED_TOOLS and not _readonly_allowed(tool_name, kw):
            return {
                "action": "block",
                "message": (
                    "Workspace gate unavailable: the plugin could not read its own state.\n"
                    f"  error: {type(exc).__name__}: {exc}\n"
                    "Repo-touching tools are blocked until it recovers.\n"
                    "  ws_doctor   - show the binding and the error\n"
                    "  ws_chatting - declare a talk-only session, which needs no state read\n"
                    "Report the error above to the user; do not retry the same tool."
                ),
            }
        return None


def _opening_prompt() -> str:
    repos = store.candidate_repos()
    listing = ", ".join(repos) if repos else f"(none found under {_repo_roots_label()})"
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
                _pin_session_cwd(detail["path"], sid=sid)
            else:
                _unpin_session_cwd(sid=sid)
            return  # already resolved once: don't re-ask on every resume
        if detail["state"] == "stale":
            # A dead path must not stay pinned: the session would keep running
            # tools in a directory it no longer owns. HERMES_HOME, not an
            # empty pin (see ws_chatting for why).
            _pin_session_cwd(store.hermes_home(), sid=sid)
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


# ── child-end reporting (subagent_start / subagent_stop) ────────────────────
#
# WHY these two hooks and not ``on_session_end``: the host's subagent hooks
# are the ONLY ones that know a child ENDED and which parent it belonged to.
# ``on_session_end`` fires when a session is finalized, which is a different
# moment, and it carries no parent/child relationship.
#
# WHY the parent learns about it on its NEXT TURN: the host freezes plugin
# system-prompt sections for the life of a session (they are rendered once
# and cached), so a section registered at start() cannot change mid-session.
# The per-turn channel is ``pre_llm_call``, whose context rides the user
# message of that turn. So the reports go there, and the registered section
# carries them too for the sessions whose prompt is built after the child
# stopped.


def _on_subagent_start(**kw: Any) -> None:
    """Record the child->parent mapping before the child starts working.

    WHY this row exists: ``subagent_stop``'s payload names the child and the
    parent but not the child's WORKSPACE, and the workspace lives in the
    child's binding. Writing the mapping here means the stop side never has
    to reverse-lookup a binding - which is ambiguous when a session id is
    re-dispatched or a child inherited its parent's binding.

    Never raises: the host treats a raising observer as a plugin bug, and a
    missing row is survivable (the stop side falls back to the bindings
    table).
    """
    try:
        parent_sid = str(kw.get("parent_session_id") or "")
        child_sid = str(kw.get("child_session_id") or "")
        childreport.record_child_start(
            parent_sid,
            child_sid,
            child_subagent_id=str(kw.get("child_subagent_id") or ""),
            parent_turn_id=str(kw.get("parent_turn_id") or ""),
            parent_subagent_id=str(kw.get("parent_subagent_id") or ""),
            child_role=str(kw.get("child_role") or ""),
            child_goal=str(kw.get("child_goal") or ""),
        )
    except Exception:
        # A broken reporter must never break the delegation it is watching:
        # the child is about to run, and observers cannot veto work.
        logger.warning("child report: start hook failed", exc_info=True)


def _on_subagent_stop(**kw: Any) -> None:
    """Hand the child's end state to a background thread, then return.

    WHY a background thread: the host runs ``subagent_stop`` on the CALLER
    thread - the parent's own thread, inside the delegation finalization.
    The dirty and unpushed checks are subprocess calls (``jj status``,
    ``git log``), and blocking there stalls the parent's thread pool and
    every delegation queued behind it. The callback therefore returns
    immediately and the report row is written by the daemon thread.

    WHY the fallback to the bindings table: the plugin may have been
    installed (or enabled) while a child was already running, so
    ``subagent_start`` never fired for it and there is no mapping row. The
    child's own binding still names its workspace, and the stop payload still
    names the parent - so the report is still possible.

    WHY the child's workspace is RECLAIMED from here: the child is done by
    contract, so its tree is dead weight, and the reclamation preserves the
    work (jj snapshots the dirty working copy into a commit). It happens on
    the report thread, after the report row is written, so the report still
    describes the tree as it was at stop time. The setting
    ``child_immediate_reclaim`` turns it off; the reporting is unaffected.
    """
    try:
        parent_sid = str(kw.get("parent_session_id") or "")
        child_sid = str(kw.get("child_session_id") or "")
        if not parent_sid:
            # The parent id is the one thing we cannot reconstruct from the
            # bindings table without guessing: a session can be bound to a
            # workspace but that binding does not say who spawned it.
            # Without it there is no session to deliver a report to.
            logger.debug("child report: stop hook without a parent session id")
            return
        childreport.record_child_stop(
            parent_sid,
            child_sid,
            child_status=str(kw.get("child_status") or ""),
            child_role=str(kw.get("child_role") or ""),
        )
    except Exception:
        # The report is a nicety; the delegation's result is not. A failure
        # here must never surface as a failed child.
        logger.warning("child report: stop hook failed", exc_info=True)


def _on_pre_llm_call(**kw: Any) -> Any:
    """Deliver the parent's pending child-end reports, once, on this turn.

    WHY ``pre_llm_call`` and not the system prompt: the host freezes plugin
    system-prompt sections for the life of a session, so a section rendered
    when the session started cannot carry a report about a child that has not
    finished yet. The per-turn channel is this hook: its context rides the
    user message of the turn it fires on.

    SILENT when there is nothing pending: no context, no empty section, no
    filler line. A parent with no finished children pays nothing.
    """
    try:
        # WHY ``_session_id(kw)`` and not ``kw.get("session_id")``: the host
        # names the session differently depending on the hook - the literal
        # ``session_id`` key is not the only one that carries it. The
        # plugin's own helper already knows every spelling
        # (``session_id``, ``session``, ``durable_session_key``, and the
        # ``HERMES_SESSION_ID`` environment). Reading only the one literal
        # key meant a payload keyed any other way silently delivered NO
        # report: the parent never heard what happened to its child.
        sid = _session_id(kw)
        if not sid:
            return None
        text = childreport.deliver_reports(sid)
        if not text:
            return None
        return {"context": text}
    except Exception:
        # Injection is best-effort: a failed report must not break the turn.
        logger.warning("child report: could not inject the pending reports", exc_info=True)
        return None


def _reports_prompt_section(info: Any = None) -> str:
    """The child-end reports section for the system prompt.

    Rendered ONCE per session, like every other plugin section. A session
    whose prompt is built after a child finished gets the reports here; the
    per-turn delivery (``pre_llm_call``) covers the rest.

    Returns the empty string when there is nothing to say, and the host
    skips an empty section entirely.
    """
    try:
        sid = ""
        # WHY the Mapping check and not a dict check: the host passes a
        # read-only ``types.MappingProxyType``, which is NOT a dict. An
        # ``isinstance(info, dict)`` test is False for it, and the section
        # would silently render nothing on every real session - a feature
        # that looks shipped and is not.
        #
        # WHY ``_session_id(info)`` and not ``info.get("session_id")``: the
        # same reason as the per-turn channel - the session id is not
        # always under the literal ``session_id`` key. The helper knows
        # every spelling the host uses.
        if isinstance(info, Mapping):
            sid = _session_id(info)
        if not sid:
            return ""
        return childreport.deliver_reports(sid)
    except Exception:
        logger.warning("child report: could not render the prompt section", exc_info=True)
        return ""


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
    p_gc.add_argument("--children", action="store_true", help="only child workspaces (delegated children); with --apply, safe for an unattended cron - never decides for user workspaces")

    p_forget = sub.add_parser(
        "forget",
        help="targeted removal of a stale workspace record; refuses live sessions and unforced trees",
    )
    p_forget.add_argument("--slug", required=True, help="the workspace record to forget (by slug)")
    p_forget.add_argument(
        "--force",
        action="store_true",
        help="snapshots the work first when the tree is still on disk",
    )

    # The IMMEDIATE reclamation path: a child that is done has its tree
    # reclaimed at once, and the work is preserved as a jj snapshot commit.
    p_reclaim = sub.add_parser(
        "reclaim",
        help="reclaim a finished child's workspace now (the work is preserved as a jj snapshot commit)",
    )
    p_reclaim.add_argument(
        "--slug",
        default="",
        help="the child workspace to reclaim (by slug)",
    )
    p_reclaim.add_argument(
        "--all-children",
        action="store_true",
        help="reclaim every child workspace whose sessions have all ended",
    )
    p_reclaim.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be reclaimed, reclaim nothing",
    )
    p_reclaim.add_argument(
        "--no-snapshot",
        action="store_true",
        help="do not snapshot a dirty tree into a commit (a dirty child's work would be LOST - for a tree you already know is worthless)",
    )

    p_chat = sub.add_parser("chatting", help="mark a session as talk-only")
    p_chat.add_argument("--session", default="")

    p_mig = sub.add_parser(
        "migrate",
        help="move workspaces from the old <workspace_root>/<repo>/<slug> layout "
             "to the new <repo_dir>/.workspaces/<repo>/<slug> layout (one-time)",
    )
    p_mig.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would move, move nothing",
    )

    subparser.set_defaults(ws_action="list")


def _cli_ws_forget(args) -> int:
    """``hermes ws forget --slug <slug>``: targeted removal of ONE record.

    WHY this exists next to ``retire``: retire is the LADDER (open PRs, dirty
    tree, unpushed commits, live sessions - a user's own workspace), while
    forget is the OPERATOR's targeted cleanup of a STALE record - one named
    slug, one transaction, and every guard that could delete somebody's work
    still in front of it.

    The order is deliberate:
      1. the record must exist (the slug is resolved through the DATABASE,
         not through the filesystem - a stale record's tree is gone);
      2. a LIVE session may not be bound to it;
      3. a tree still on disk needs --force, because --force is what
         authorizes the snapshot-first retirement of that tree;
      4. a user-class record needs --force too: the gc caveat for user
         workspaces ("never decided for you") applies here as well;
      5. the record and its bindings go in ONE transaction.
    """
    slug = str(getattr(args, "slug", "") or "")
    force = bool(getattr(args, "force", False))

    record = None
    for row in store.db.all_workspaces():
        if str(row.get("slug") or "") == slug:
            record = dict(row)
            break
    if record is None:
        available = sorted(
            {
                str(row.get("slug") or "")
                for row in store.db.all_workspaces()
                if str(row.get("slug") or "")
            }
        )
        print(f"refused: no workspace record for slug {slug!r}")
        print("available slugs: " + (", ".join(available) if available else "(none)"))
        return 1

    path = Path(str(record.get("path") or ""))
    purpose = str(record.get("purpose") or "")
    is_child = purpose.startswith(store.CHILD_PURPOSE_PREFIX)

    # (2) liveness: the same check the reclamation primitive uses - a LIVE
    # session bound to (or pinned at) the tree. A record whose tree is
    # already gone has nothing a live session can still be using.
    if path.is_dir() and os.path.normpath(str(path)) in store.jjlayout.live_session_paths(store.db):
        print(f"refused: {slug} is bound to (or pinned by) a live session")
        return 1

    tree_exists = path.is_dir()

    # (3) a tree on disk is never removed without --force: --force is the
    # explicit "snapshot the work first" authorization.
    if tree_exists and not force:
        print(
            f"refused: the tree for {slug} is still on disk ({path}); "
            "repeat with --force to snapshot the work first and remove it"
        )
        return 1

    # (4) a user-class record is a human's workspace: the gc caveat applies
    # here too - the operator has to say so explicitly.
    if not is_child and not force:
        print(
            f"refused: {slug} is a user workspace ({purpose or 'no purpose'}); "
            "repeat with --force to forget its record"
        )
        return 1

    snapshot_commit_id = ""
    if tree_exists:
        if is_child:
            # The child-classified dirty/unpushed protection lives inside the
            # reclamation primitive - reuse it, do NOT re-implement the checks
            # and do NOT write a second removal path.
            result = store.reclaim_child_workspace(path)
            if not result.get("ok"):
                detail = str(result.get("detail") or result.get("reason") or "retirement refused")
                print(f"refused: {slug} was not retired: {detail}")
                return 1
            snapshot_commit_id = str(result.get("snapshot_commit_id") or "")
        else:
            # A user workspace is retired by the SAME ladder `ws retire`
            # uses - --force acknowledges the unverifiable checks, and
            # --ultraforce is what lets a dirty tree go (the work is archived
            # first, so the record is still recoverable).
            try:
                store.retire(slug, force=True, ultraforce=True)
            except store.WorkspaceError as exc:
                print(f"refused: {slug} was not retired: {exc}")
                return 1

    # (5) ONE transaction: the record and its bindings go together, or
    # neither goes.
    repo = str(record.get("repo") or "")
    with store.db.transaction() as conn:
        cur_ws = conn.execute(
            "DELETE FROM workspaces WHERE repo=? AND slug=?", (repo, slug)
        )
        cur_bind = conn.execute("DELETE FROM bindings WHERE slug=?", (slug,))
        rows_removed = int(cur_ws.rowcount or 0) + int(cur_bind.rowcount or 0)

    print(f"forgot {slug} ({path}) - removed {rows_removed} row(s)")
    if snapshot_commit_id:
        print(f"snapshot commit: {snapshot_commit_id}")
    return 0


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
        if action == "forget":
            return _cli_ws_forget(args)
        if action == "gc":
            tiers = store.gc_tiers()
            if getattr(args, "children", False):
                for name in ("tier1", "tier2", "tier3"):
                    tiers[name] = [
                        row for row in tiers[name]
                        if str((row.get("record") or {}).get("purpose") or "").startswith(store.CHILD_PURPOSE_PREFIX)
                    ]
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
                done = store.gc_children_apply() if getattr(args, "children", False) else store.gc_apply()
                print(f"\nretired: {len(done)}" if done else "\nretired: nothing eligible")
                for path in done:
                    print(f"  {path}")
            else:
                print(
                    "\nNothing was deleted. Reclaim the 'safe to retire' set with: "
                    "hermes ws gc --apply"
                )
            return 0
        if action == "reclaim":
            slug = str(getattr(args, "slug", "") or "")
            all_children = bool(getattr(args, "all_children", False))
            if not slug and not all_children:
                print("nothing to reclaim: pass --slug <slug> or --all-children")
                return 2
            if slug and all_children:
                print("pass either --slug or --all-children, not both")
                return 2
            out = store.reclaim_children(
                [slug] if slug else None,
                all_children=all_children,
                snapshot=not bool(getattr(args, "no_snapshot", False)),
                dry_run=bool(getattr(args, "dry_run", False)),
            )
            # The structured report: what was reclaimed (and where its work
            # ended up) and what was skipped (and why).
            print(json.dumps(out, indent=2, default=str))
            if getattr(args, "dry_run", False):
                print("\ndry run: nothing was reclaimed")
            return 0
        if action == "chatting":
            store.mark_chatting(args.session or "cli")
            print("session marked as chatting")
            return 0
        if action == "migrate":
            # The one-time LAYOUT migration: move the workspaces that live
            # under the old root into the dot-dir layout. It never runs
            # implicitly - the operator asks for it once.
            out = migrate_layout.migrate_workspaces(dry_run=bool(getattr(args, "dry_run", False)))
            print(migrate_layout.report_text(out))
            if getattr(args, "dry_run", False):
                print("\ndry run: nothing was moved")
            else:
                print(
                    f"\nmoved {len(out.get('moved') or [])}, "
                    f"skipped {len(out.get('skipped_dirty') or [])} dirty, "
                    f"in use {len(out.get('in_use') or [])}, "
                    f"verify failed {len(out.get('verify_failed') or [])}, "
                    f"repair failed {len(out.get('repair_failed') or [])}"
                )
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
        help="Workspaces: one per session (new/list/status/retire/reclaim/chatting)",
        setup_fn=_register_cli,
        handler_fn=_cli,
        description=(
            "Session-owned workspaces under <repo_dir>/.workspaces/<repo>/<slug>. "
            "Creates the jj workspace and its record in the plugin database together."
        ),
    )

    repos = store.candidate_repos(limit=6)
    listing = ", ".join(repos) if repos else f"(none found under {_repo_roots_label()})"
    ctx.register_system_prompt_section(
        "workspace-discipline",
        # The delegation contract travels with the prompt: the rules that used to
        # live in a skill file and in the operator's head are now in every session.
        _PROMPT_TEMPLATE.format(repos=listing) + "\n\n" + contract.as_prompt_section(),
        position="after_memory",
    )

    # The child-end reports section. It is a CALLABLE, so it renders with
    # the session's own id and stays silent when the session has no pending
    # reports - the host skips an empty section entirely.
    ctx.register_system_prompt_section(
        "child-end-reports",
        _reports_prompt_section,
        position="after_memory",
    )

    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    # The host's dedicated subagent lifecycle hooks. NOT on_session_end: that
    # one fires when a session is finalized and carries no parent/child
    # relationship, so it cannot tell the parent which workspace its child
    # used.
    ctx.register_hook("subagent_start", _on_subagent_start)
    ctx.register_hook("subagent_stop", _on_subagent_stop)
    # The per-turn delivery channel for the parent's next turn. The host
    # freezes plugin system-prompt sections per session, so a section cannot
    # carry a report about a child that finishes later.
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    picker.register(ctx)
    prepare.register(ctx)
