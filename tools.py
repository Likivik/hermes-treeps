"""The workspace tools: ws_new / ws_chatting / ws_rename / ws_status / ws_doctor.

Pure code motion out of ``__init__.py``. The tool schemas, the tool handlers
and the remote-catalogue helpers live here.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from . import prepare, store
from .gate import _forced_child_shape, _pin_session_cwd, _session_id

logger = logging.getLogger(__name__)

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

def _opening_prompt() -> str:
    repos = store.candidate_repos()
    listing = ", ".join(repos) if repos else f"(none found under {_repo_roots_label()})"
    return (
        "New session, no workspace yet. Ask me exactly one line: which repo to work in "
        f"({listing}) - or 'just chatting' for a talk-only session with no checkout. "
        "Wait for my answer before doing anything else; do not start work, do not guess a repo."
    )
