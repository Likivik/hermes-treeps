"""Repo discovery: the local repo dir, and the forges the operator uses.

The plugin offers the user ONE flat list of repos to work in. Every entry
carries a ``tag`` that says where it came from:

* ``local`` - a clone under ``repo_dir``;
* ``forge:github`` / ``forge:forgejo`` - a repo on that forge, owned by the
  configured owner.

The list is deduped by REMOTE IDENTITY - forge + owner + repo name - never
by path. A forge repo that already has a clone under ``repo_dir`` therefore
appears ONCE, tagged ``local``: the user knows one repo, and offering it
twice (once local, once remote) makes them clone a copy they already have.

Three rules, from the locked decisions in issue #37:

1. the LOCAL scan walks ``repo_dir`` and nothing else. The plugin creates
   its workspaces under ``repo_dir``, so a repo outside it is a repo the
   picker cannot bind. The deprecated ``repo_roots`` key is still PARSED
   by the settings module (an old config must not crash the plugin) but
   discovery does not read it.
2. the FORGE scan is OWNER-FILTERED. A forge catalogues every repo on the
   instance; the owner is the boundary that says "these are mine". The
   owner is DERIVED from the credentials the plugin already has - the
   Forgejo token's account, and the account ``gh`` is logged into - so
   there is no second place for the operator to name a login, and no
   way to name the wrong one. A forge whose owner cannot be derived
   (no token, no ``gh``, a dead API) is SKIPPED, quietly.
3. the result is a FLAT LIST. The picker renders one menu; two lists would
   mean two menus, or a merge at the render site - and the merge is
   exactly where a duplicate would sneak in.

The catalogue lookup is a diagnostic, not a gate: it must never raise into
a tool answer, only degrade to "this source is unavailable".
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from . import settings as _settings

logger = logging.getLogger(__name__)

#: The tag of a repo that lives on disk under ``repo_dir``.
TAG_LOCAL = "local"

#: The tag prefix of a repo that lives on a forge.
TAG_FORGE_PREFIX = "forge:"

#: The forges this module knows how to scan, and the tag each one gets.
FORGE_TAGS: dict[str, str] = {
    "forgejo": TAG_FORGE_PREFIX + "forgejo",
    "github": TAG_FORGE_PREFIX + "github",
}


def _entry(
    name: str,
    owner: str,
    clone_url: str,
    source: str,
    pushed_at: str = "",
    local_path: str = "",
) -> dict[str, Any]:
    """One row of the flat list.

    ``tag`` is the new name of ``source``; ``clone_url`` is the new name
    of ``url``. Both spellings are kept because the picker (PR 2) renders
    ``tag``/``clone_url`` while the older consumers in ``__init__.py`` and
    ``telegram_picker.py`` still read ``source``/``url``. Two names for
    one value is a debt; it is paid down in the picker PR, which is the
    only place that still reads the old ones.
    """
    tag = FORGE_TAGS.get(source, TAG_LOCAL)
    return {
        "name": name,
        "tag": tag,
        "clone_url": clone_url,
        "local_path": local_path,
        "owner": owner,
        # the legacy spellings
        "source": source,
        "url": clone_url,
        "pushed_at": pushed_at,
    }


# ── the remote identity ─────────────────────────────────────────────────────


def remote_identity(clone_url: str) -> tuple[str, str, str]:
    """``(forge, owner, name)`` parsed out of a clone URL.

    WHY this is the dedupe key and not the path: the path is a LOCAL
    accident (wherever the clone happens to sit), and two clones of the
    same repo in two directories are one repo. The remote identity is
    what decides where a clone would LAND - forge, owner and name - so
    it is what says "these two are the same repo".

    WHY the forge is part of it: the same owner can have a repo called
    ``alpha`` on GitHub AND on Forgejo, and they are two different repos
    that clone to the same local name. Collapsing them would send the
    user to the wrong one.

    Returns ``("", "", "")`` for a URL that names no forge - an
    unparseable origin simply does not dedupe against anything.
    """
    url = str(clone_url or "").strip()
    if not url:
        return ("", "", "")
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.netloc or "").lower()
    path = parsed.path
    if "://" not in url and ":" in url:
        # An SCP-like remote: git@host:owner/repo.git. urlsplit puts the
        # whole thing in ``path``, and the first segment carries the
        # host, not the owner - so the host is taken from it and the
        # owner from the segment after it.
        head, _, tail = path.partition(":")
        host = head.rsplit("@", 1)[-1].lower()
        path = tail
    # Either way the path is what carries owner and name.
    parts = [p for p in path.split("/") if p]
    if not parts:
        return ("", "", "")
    name = parts[-1]
    if name.endswith(".git"):
        name = name[: -len(".git")]
    owner = parts[-2] if len(parts) >= 2 else ""
    forge = ""
    if "github" in host:
        forge = "github"
    elif "forgejo" in host or "codeberg" in host:
        forge = "forgejo"
    if not forge:
        # A self-hosted Forgejo has its own host name; the API answers
        # with the full owner/name, so the forge is filled in from the
        # catalogue row instead of the URL.
        return ("", owner, name)
    return (forge, owner, name)


def _local_origin(path: Path) -> str:
    """The ``origin`` remote URL of a local clone, or ``""``.

    WHY read it at all: the local clone knows WHICH forge repo it is a
    clone of. Without it, the only way to pair a local clone with a
    forge row is the name - and the same name under two owners would
    pair the clone with the wrong one, hiding a repo the user actually
    has.
    """
    for cmd in (
        ["git", "-C", str(path), "remote", "get-url", "origin"],
        ["jj", "-R", str(path), "remote", "get-url", "origin"],
    ):
        if not shutil.which(cmd[0]):
            continue
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
            )
        except Exception:
            logger.debug("could not read the origin of %s", path, exc_info=True)
            continue
        if proc.returncode == 0:
            url = (proc.stdout or "").strip()
            if url:
                return url
    return ""


# ── the local scan ──────────────────────────────────────────────────────────


def local_repos(limit: int = 0) -> list[dict[str, Any]]:
    """The repos under ``repo_dir`` - and nowhere else.

    WHY ``repo_dir`` and not the deprecated ``repo_roots``: the plugin
    creates its workspaces under ``repo_dir``, so a repo outside it is a
    repo the picker would offer and then fail to bind. The old scan read
    a second list whose relationship to ``repo_dir`` was undefined.

    ``limit`` of 0 means "no limit" - the caller decides how many to
    show, and the picker shows everything the user owns.
    """
    out: list[dict[str, Any]] = []
    base = Path(str(_settings.settings()["repo_dir"]))
    if not base.is_dir():
        return out
    try:
        children = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        logger.debug("could not read the repo dir %s", base, exc_info=True)
        return out
    for p in children:
        # The dot-dirs are not repos: ``.workspaces`` holds the session
        # workspaces and would otherwise show up in the picker.
        if p.name.startswith("."):
            continue
        if (p / ".jj").exists() or (p / ".git").exists() or (p / ".git").is_file():
            out.append(_entry(p.name, "", "", "local", local_path=str(p)))
            if limit and len(out) >= limit:
                return out
    return out


def local_names(limit: int = 0) -> list[str]:
    """Just the names of the local repos."""
    return [r["name"] for r in local_repos(limit=limit)]


# ── the forge scans ─────────────────────────────────────────────────────────

#: The Forgejo API's maximum page size. The endpoint answers with one page
#: at a time, so every page is fetched to list an owner COMPLETELY.
FORGEJO_PAGE_SIZE = 50

#: How many pages the Forgejo scan fetches before it gives up.
#:
#: WHY a ceiling: the scan is a menu, not an export. An owner with a
#: thousand repos does not need all of them listed - and a server that
#: always answers with a full page would otherwise make the scan run
#: forever.
FORGEJO_MAX_PAGES = 20

#: The derived owner login per forge, cached for the life of the process.
#:
#: WHY a cache: the owner is derived from an API call or a subprocess
#: (see ``_derive_owner``), and discovery runs on every gate check. The
#: login of the authenticated user does not change while the process
#: lives, so it is derived once.
#:
#: WHY the cache stores the FAILURE too: a forge whose owner cannot be
#: derived (no token, no ``gh``, the API is down) is skipped, and
#: retrying the derivation on every call would turn a missing token
#: into a per-call network timeout.
_OWNER_CACHE: dict[str, str] = {}


def _derive_owner(forge: str) -> str:
    """The login whose repos the forge lists, or ``""``.

    WHY the owner is DERIVED and not configured: the operator already
    told the plugin who they are - the Forgejo token belongs to one
    account, and ``gh`` is logged into one account. Asking them to
    repeat the login in a setting is a second place to get it wrong,
    and a wrong login lists somebody else's repos.

    An empty result means the forge is SKIPPED: no token, no ``gh``, a
    dead API - none of those is the user's problem to solve in the
    middle of a session, so the forge is quietly left out.
    """
    cached = _OWNER_CACHE.get(forge)
    if cached is not None:
        return cached
    owner = ""
    try:
        if forge == "forgejo":
            owner = _forgejo_login()
        elif forge == "github":
            owner = _github_login()
    except Exception:
        logger.debug("could not derive the %s owner", forge, exc_info=True)
        owner = ""
    _OWNER_CACHE[forge] = owner
    return owner


def _forgejo_login() -> str:
    """The login of the account the Forgejo token belongs to."""
    base = str(_settings.settings()["forgejo_url"]).rstrip("/")
    if not base:
        raise RuntimeError("forgejo_url is not configured")
    token = _settings.forgejo_token()
    if not token:
        raise RuntimeError("no Forgejo token")
    req = urllib.request.Request(
        base + "/api/v1/user", headers={"Authorization": "token " + token},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    return str(data.get("login") or "").strip()


def _github_login() -> str:
    """The login ``gh`` is authenticated as."""
    gh = shutil.which("gh")
    if not gh:
        raise FileNotFoundError("gh is not on PATH")
    raw = subprocess.run(
        [gh, "api", "user", "--jq", ".login"],
        capture_output=True, text=True, timeout=30,
    )
    if raw.returncode != 0:
        raise RuntimeError((raw.stderr or "").strip()[:200] or f"gh exited {raw.returncode}")
    return (raw.stdout or "").strip()


def github_repos(limit: int = 0) -> tuple[list[dict[str, Any]], str]:
    """(entries, error) from ``gh repo list`` for the authenticated owner.

    WHY the owner is named explicitly: ``gh repo list`` without an owner
    lists EVERY repo the token can see - including other people's, and
    including every repo of every org the token belongs to. The owner is
    the boundary that says "these are mine".

    WHY no limit: the picker is a menu of what the user owns, not a
    search over a truncated list. A limit hides repos the user has and
    wants.
    """
    out: list[dict[str, Any]] = []
    owner = _derive_owner("github")
    if not owner:
        # No owner could be derived = this forge is not consulted. Not an
        # error: the operator may not use GitHub at all.
        return out, ""
    gh = shutil.which("gh")
    try:
        if not gh:
            raise FileNotFoundError("gh is not on PATH")
        cmd = [gh, "repo", "list", owner, "--json", "nameWithOwner,url,pushedAt"]
        raw = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if raw.returncode != 0:
            raise RuntimeError((raw.stderr or "").strip()[:200] or f"gh exited {raw.returncode}")
        for r in json.loads(raw.stdout or "[]"):
            name_with_owner = str(r.get("nameWithOwner") or "")
            row_owner, _, row_name = name_with_owner.partition("/")
            out.append(
                _entry(
                    row_name or row_owner,
                    row_owner,
                    str(r.get("url") or ""),
                    "github",
                    str(r.get("pushedAt") or ""),
                )
            )
    except Exception as exc:
        logger.debug("gh repo list failed", exc_info=True)
        return [], f"github: {type(exc).__name__}: {exc}"
    return out, ""


def forgejo_repos(limit: int = 0) -> tuple[list[dict[str, Any]], str]:
    """(entries, error) from the Forgejo API for the authenticated owner.

    WHY the user endpoint and not ``/repos/search``: the search endpoint
    answers with EVERY repo on the instance, and its only owner-ish
    parameter is ``uid`` (a numeric id). ``/users/{owner}/repos`` is the
    endpoint that means "the repos this login owns" - which is exactly
    the boundary the owner filter is about.

    WHY the paging: the endpoint returns one page (the API's maximum
    page size is 50). Every page is fetched, so an owner with 120 repos
    is listed with 120 repos - a truncated list would hide repos the
    user owns and wants.
    """
    out: list[dict[str, Any]] = []
    owner = _derive_owner("forgejo")
    if not owner:
        # No owner could be derived = this forge is not consulted.
        return out, ""
    try:
        base = _settings.settings()["forgejo_url"].rstrip("/")
        if not base:
            raise RuntimeError("forgejo_url is not configured")
        token = _settings.forgejo_token()
        if not token:
            raise RuntimeError("no Forgejo token")
        page = 1
        while True:
            url = (
                f"{base}/api/v1/users/{urllib.parse.quote(owner)}/repos"
                f"?page={page}&limit={FORGEJO_PAGE_SIZE}"
            )
            req = urllib.request.Request(
                url, headers={"Authorization": "token " + token},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            if not isinstance(data, list):
                raise RuntimeError("the Forgejo API did not answer with a list")
            for r in data:
                row_owner = (r.get("owner") or {}).get("login", "") or owner
                out.append(
                    _entry(
                        str(r.get("name") or ""),
                        row_owner,
                        str(r.get("clone_url") or ""),
                        "forgejo",
                        str(r.get("updated_at") or ""),
                    )
                )
            # A short page is the last page. The API's maximum page size
            # is 50, so a page shorter than that means there is no page
            # after it.
            if len(data) < FORGEJO_PAGE_SIZE:
                break
            page += 1
            if page > FORGEJO_MAX_PAGES:
                # A guard against a server that always answers with a
                # full page: without it the scan would never end.
                logger.warning(
                    "the Forgejo scan stopped at %s pages for owner %s",
                    FORGEJO_MAX_PAGES, owner,
                )
                break
    except Exception as exc:
        logger.debug("forgejo repo listing failed", exc_info=True)
        return [], f"forgejo: {type(exc).__name__}: {exc}"
    return out, ""


# ── the flat list ───────────────────────────────────────────────────────────


def _identity_of(entry: dict[str, Any]) -> tuple[str, str, str]:
    """The remote identity of a list row.

    For a forge row the identity is the row's own forge + owner + name -
    the catalogue already knows them, so the URL does not have to be
    parseable.

    For a local row the identity comes from its ORIGIN: the clone knows
    which forge repo it is a clone of. A clone whose origin names no
    known forge (or has no origin at all) is UN-ATTRIBUTED - it keeps
    the identity ``("local", "", name)``, which does not collide with
    any forge row.
    """
    tag = str(entry.get("tag") or "")
    name = str(entry.get("name") or "")
    owner = str(entry.get("owner") or "")
    if tag == TAG_LOCAL:
        url = str(entry.get("origin") or "")
        forge, url_owner, url_name = remote_identity(url)
        if not forge:
            return ("local", owner, name)
        return (forge, url_owner or owner, url_name or name)
    forge = tag.split(":", 1)[1] if ":" in tag else tag
    if not owner:
        forge_from_url, url_owner, url_name = remote_identity(
            str(entry.get("clone_url") or "")
        )
        owner = url_owner
        name = url_name or name
        forge = forge or forge_from_url
    return (forge, owner, name)


def _is_unattributed(identity: tuple[str, str, str]) -> bool:
    """True when a row cannot be attributed to a forge repo."""
    return identity[0] == TAG_LOCAL


def _same_repo(
    local: dict[str, Any],
    local_id: tuple[str, str, str],
    forge: dict[str, Any],
    forge_id: tuple[str, str, str],
    name_forges: dict[str, set[str]] | None = None,
) -> bool:
    """Whether a local clone and a forge row are the SAME repo.

    Two ways they can be:

    1. the identities match exactly - the clone's origin points at the
       forge row's repo;
    2. the clone is UN-ATTRIBUTED (its origin names no known forge, or
       it has no origin), the forge row's NAME is the clone's name, and
       NO OTHER FORGE claims that name.

    WHY case 2 exists: a clone of ``<owner>/<name>`` lands at
    ``<repo_dir>/<name>``, so the local NAME is what the forge row is
    called. The scan is owner-filtered - each forge lists exactly ONE
    owner - so within a forge the name is unambiguous.

    WHY the name alone is not enough: ``me/alpha`` and ``other/alpha``
    are two different repos that clone to the same local name. So case 2
    only applies to an UN-ATTRIBUTED clone, and only when the name is
    claimed by exactly one forge - otherwise the pairing is a guess, and
    a guess here hides a repo the user actually has.
    """
    if local_id == forge_id:
        return True
    if not _is_unattributed(local_id):
        return False
    local_name = str(local.get("name") or "")
    forge_name = str(forge.get("name") or "")
    if not local_name or local_name != forge_name:
        return False
    if name_forges is None:
        return True
    return len(name_forges.get(local_name, set())) <= 1


def discover(limit_per_source: int = 0) -> tuple[list[dict[str, Any]], list[str]]:
    """(entries, errors) - ONE flat, deduped, tagged list.

    The entries are deduped by REMOTE IDENTITY (forge + owner + name).
    A local clone wins over the forge row it is a clone of: the user
    already has that repo on disk, and offering the remote copy would
    make them clone a second time.

    The order is LOCAL FIRST, then the forge rows by most recently
    pushed. WHY local first: a local repo is the one the plugin can bind
    right now, and the picker offers it without a clone step - it is
    the answer the user wants most often. The forge rows keep the
    "most recently pushed first" order the picker has always shown.

    ``limit_per_source`` of 0 means "everything". A positive limit caps
    each forge scan - the local scan is never capped, because the repos
    on disk are the ones the plugin can actually work in.
    """
    cfg = _settings.settings()
    sources = cfg["repo_sources"]
    entries: list[dict[str, Any]] = []
    errors: list[str] = []

    if "local" in sources:
        entries.extend(local_repos())

    if "github" in sources:
        gh, err = github_repos(limit=limit_per_source)
        entries.extend(gh)
        if err:
            errors.append(err)

    if "forgejo" in sources:
        fj, err = forgejo_repos(limit=limit_per_source)
        entries.extend(fj)
        if err:
            errors.append(err)

    # The local rows need their origin, which is what pairs a clone with
    # the forge row it came from.
    for entry in entries:
        if entry.get("tag") == TAG_LOCAL and entry.get("local_path"):
            entry["origin"] = _local_origin(Path(str(entry["local_path"])))

    # How many forges claim each repo NAME. WHY: an unattributed clone
    # can only be paired with a forge row by name, and that is only safe
    # when the name is claimed by ONE forge - ``me/alpha`` on GitHub and
    # ``other/alpha`` on Forgejo are two different repos that happen to
    # clone to the same local name, and pairing the clone with either
    # one on a guess hides the other.
    name_forges: dict[str, set[str]] = {}
    for entry in entries:
        if entry.get("tag") == TAG_LOCAL:
            continue
        name = str(entry.get("name") or "")
        forge = str(entry.get("tag") or "").split(":", 1)[-1]
        if name:
            name_forges.setdefault(name, set()).add(forge)

    out: list[dict[str, Any]] = []
    # The identity of each entry IN ``out`` - kept in step with it, so
    # the pairing below cannot drift out of index.
    out_ids: list[tuple[str, str, str]] = []
    for entry in entries:
        identity = _identity_of(entry)
        if identity == ("", "", ""):
            out.append(entry)
            out_ids.append(identity)
            continue
        if entry.get("tag") == TAG_LOCAL:
            # A local clone: it wins against any forge row it is the
            # clone of, so the forge row is dropped.
            matched = False
            for i, other in enumerate(out):
                if other.get("tag") == TAG_LOCAL:
                    continue
                if _same_repo(entry, identity, other, out_ids[i], name_forges):
                    out[i] = entry
                    out_ids[i] = identity
                    matched = True
                    break
            if not matched:
                out.append(entry)
                out_ids.append(identity)
            continue
        # A forge row: dropped when a local clone already claims it.
        if any(
            other.get("tag") == TAG_LOCAL
            and _same_repo(other, out_ids[i], entry, identity, name_forges)
            for i, other in enumerate(out)
        ):
            continue
        out.append(entry)
        out_ids.append(identity)

    # LOCAL FIRST, then the forge rows by most recently pushed. The name
    # breaks ties, so the menu does not reshuffle between two calls that
    # saw the same repos.
    out.sort(key=lambda e: (
        0 if e.get("tag") == TAG_LOCAL else 1,
        _sort_pushed_at(e),
        str(e.get("name") or ""),
    ))
    return out, errors


def _sort_pushed_at(entry: dict[str, Any]) -> str:
    """The pushed-at stamp of a row, inverted for a newest-first sort.

    WHY the inversion: the rows are sorted ASCENDING, and the newest
    stamp has to come first. An empty stamp sorts last among the forge
    rows - a row with no stamp is older than every row that has one,
    and the stamp is a string, so "9" would sort after "10" without
    the inversion.
    """
    stamp = str(entry.get("pushed_at") or "")
    if not stamp:
        return "~"
    # The inverted string: every digit maps to its nine's complement, so
    # a lexicographic comparison of two inverted stamps is the reverse
    # of a comparison of the original ones.
    return "".join(chr(ord("9") - int(c)) if c.isdigit() else c for c in stamp)
