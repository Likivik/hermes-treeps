"""Give a delegated child session its OWN workspace in the parent's repo.

WHY this exists: a delegated child used to inherit the parent's binding, so
two children of one parent edited one tree. That collision cost a real
incident in this project.

The gate hook calls :func:`ensure_child_workspace` once per child session:

    out = childspace.ensure_child_workspace(sid, parent_state)
    if out["ok"]:
        ... bind the child to out["path"] ...
    else:
        ... keep today's inheritance behaviour (out["reason"] says why) ...

Contract:

* NEVER raises. Every failure path returns ``ok=False`` plus the exception
  type and message, so the caller can fall back to inheritance.
* Idempotent: a session that already has a child workspace gets that one
  back instead of a second one.
* Only a parent bound to a *repo* workspace can have a child workspace. A
  ``chatting`` or ``external`` parent has no repo to branch from, and
  inventing one would be worse than inheriting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import store
from .errors import WorkspaceError

CHILD_PURPOSE_PREFIX = store.CHILD_PURPOSE_PREFIX

#: How many slug candidates are tried when the first one is already taken.
#: The suffix grows with each attempt (``-2``, ``-3``, ...), so 32
#: candidates cover a burst of same-suffix children without spinning
#: forever on a genuinely broken workspace root.
CHILD_SLUG_COLLISION_ATTEMPTS = 32


def is_child_workspace(record: dict[str, Any]) -> bool:
    """True when the workspace record is a child workspace.

    A child workspace carries the purpose marker ``child:<parent session id>``.
    """
    if not isinstance(record, dict):
        return False
    purpose = str(record.get("purpose") or "")
    return purpose.startswith(CHILD_PURPOSE_PREFIX)


def _parent_session_id(parent_state: dict[str, Any]) -> str:
    """The parent session id, as carried by the parent's binding dict."""
    for key in ("session_id", "parent_session_id", "sid"):
        value = parent_state.get(key)
        if value:
            return str(value)
    return ""


def _fail(reason: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "reason": reason, "repo": "", "slug": "", "path": ""}
    out.update(extra)
    return out


def ensure_child_workspace(session_id: str, parent_state: dict[str, Any], *, title: str = "") -> dict[str, Any]:
    """Create (or return) a child workspace for ``session_id``.

    ``parent_state`` is the parent's binding dict:
    ``{"mode": "repo"|"external"|"chatting", "repo": str, "slug": str,
    "path": str}``. The parent's session id is read from the optional
    ``session_id`` / ``parent_session_id`` / ``sid`` key.

    Returns ``{"ok": bool, "reason": str, "repo": str, "slug": str,
    "path": str}``. It never raises.
    """
    try:
        if not session_id:
            return _fail("no child session id")

        parent_state = parent_state if isinstance(parent_state, dict) else {}
        mode = str(parent_state.get("mode") or "")

        # ── the parent must have a repo to branch from ────────────────────
        if mode == "chatting":
            return _fail("parent session is chatting - no repo to branch from")
        if mode == "external":
            return _fail("parent session is bound to an external tree - no repo to branch from")
        if mode != "repo":
            return _fail(f"parent session has no usable workspace (mode={mode!r})")

        repo = str(parent_state.get("repo") or "")
        parent_slug = str(parent_state.get("slug") or "")
        if not repo:
            return _fail("parent session names no repo")

        # ── guard: a session already marked chatting stays chatting ───────
        existing = store.read_state(session_id) or {}
        if str(existing.get("mode") or "") == "chatting":
            return _fail("child session is already bound to chatting")

        # ── idempotency: one child workspace per child session ───────────
        if existing.get("inherited_from") and str(existing.get("mode") or "") == "repo":
            existing_path = str(existing.get("path") or "")
            rec = store.read_record(Path(existing_path)) if existing_path else None
            if rec is not None and is_child_workspace(rec):
                return {
                    "ok": True,
                    "reason": "child workspace already exists",
                    "repo": str(existing.get("repo") or repo),
                    "slug": str(existing.get("slug") or ""),
                    "path": existing_path,
                }

        parent_session_id = _parent_session_id(parent_state)
        purpose = f"{CHILD_PURPOSE_PREFIX}{parent_session_id}"

        raw_slug = f"{parent_slug}-child-{str(session_id)[-6:]}"
        slug = store.sanitize_slug(raw_slug)
        if not slug:
            return _fail(f"could not build a workspace slug from {raw_slug!r}")

        # ── the repo must exist and be a jj repo ──────────────────────────
        repo_path = store.find_repo(repo)
        if not store.is_jj(repo_path):
            return _fail(f"repo {repo!r} is not a jj repo - no child workspace")

        # ── create, bind, and record the child's own binding ──────────────
        #
        # WHY the collision loop: the slug's suffix is the last 6 characters
        # of the child session id - 24 bits of entropy, and session ids are
        # generated by the host, not by us. Two children whose ids share
        # those 6 characters (a re-dispatch of an old id, a host that
        # numbers its ids per-minute, a test fixture that uses the same
        # suffix) produce the SAME slug, and ``store.create()`` raises
        # "already exists". That raised exception ended the child's
        # workspace creation, and the gate then hard-blocked the child: a
        # child that cannot get a tree cannot do anything. So a taken slug
        # is not an error - it is a name to be disambiguated. The suffix
        # grows (``-2``, ``-3``, ...) until a free slug is found, and THAT
        # slug is the one the child is bound to.
        record = None
        last_error = ""
        for attempt in range(1, CHILD_SLUG_COLLISION_ATTEMPTS + 1):
            candidate = slug if attempt == 1 else f"{slug}-{attempt}"
            candidate = store.sanitize_slug(candidate)
            if not candidate:
                continue
            try:
                record = store.create(repo, candidate, purpose=purpose)
                break
            except WorkspaceError as exc:
                last_error = str(exc)
                if "already exists" not in last_error:
                    raise
                # The slug is taken: try the next one.
        if record is None:
            return _fail(
                "could not create a child workspace: every candidate slug was taken",
                detail=last_error,
            )
        bound_slug = str(record.get("slug") or slug)
        bound_path = str(record.get("path") or "")
        if not bound_path:
            return _fail(f"workspace {slug!r} was created without a path", repo=repo, slug=bound_slug)

        store.bind_session(Path(bound_path), session_id)

        state: dict[str, Any] = {
            "mode": "repo",
            "repo": repo,
            "slug": bound_slug,
            "path": bound_path,
            "inherited_from": parent_session_id,
            # WHY BOTH keys: ``parent_session_id`` is the explicit
            # "who dispatched me" field, and it is what the gate reads back
            # to resolve this child's parent after a gateway restart (the
            # ``child_runs`` row is written by the host's subagent_start
            # hook, which does not re-fire for a resumed child).
            # ``inherited_from`` is kept for the existing readers.
            "parent_session_id": parent_session_id,
            "purpose": purpose,
        }
        if title:
            state["title"] = title
        store.write_state(session_id, state)

        return {
            "ok": True,
            "reason": "",
            "repo": repo,
            "slug": bound_slug,
            "path": bound_path,
        }
    except Exception as exc:  # never raise: the caller falls back to inheritance
        return _fail(f"{type(exc).__name__}: {exc}")
