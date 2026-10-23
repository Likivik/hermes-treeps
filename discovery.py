"""Repo discovery: local scan, GitHub, Forgejo.

Every source returns the same shape - ``{name, owner, url, source,
pushed_at}`` - so a caller never has to know where an entry came from. Which
sources are consulted comes from the configuration
(``settings.repo_sources``); a source that is not listed is never asked.

The catalogue lookup is a diagnostic, not a gate: it must never raise into a
tool answer, only degrade to "this source is unavailable".
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import settings as _settings

logger = logging.getLogger(__name__)


def _entry(name: str, owner: str, url: str, source: str, pushed_at: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "owner": owner,
        "url": url,
        "source": source,
        "pushed_at": pushed_at,
    }


def local_repos(limit: int = 8) -> list[dict[str, Any]]:
    """Repo directories under the configured repo roots."""
    out: list[dict[str, Any]] = []
    for base in _settings.settings()["repo_roots"]:
        b = Path(base)
        if not b.is_dir():
            continue
        try:
            children = sorted(p for p in b.iterdir() if p.is_dir())
        except OSError:
            continue
        for p in children:
            if (p / ".jj").exists() or (p / ".git").exists() or (p / ".git").is_file():
                out.append(_entry(p.name, "", "", "local"))
                if len(out) >= limit:
                    return out
    return out


def local_names(limit: int = 8) -> list[str]:
    """Just the names of the local repos."""
    return [r["name"] for r in local_repos(limit=limit)]


def github_repos(limit: int = 5) -> tuple[list[dict[str, Any]], str]:
    """(entries, error) from ``gh repo list``."""
    out: list[dict[str, Any]] = []
    gh = shutil.which("gh")
    try:
        if not gh:
            raise FileNotFoundError("gh is not on PATH")
        raw = subprocess.run(
            [gh, "repo", "list", "--limit", str(limit),
             "--json", "nameWithOwner,url,pushedAt"],
            capture_output=True, text=True, timeout=30,
        )
        if raw.returncode != 0:
            raise RuntimeError((raw.stderr or "").strip()[:200] or f"gh exited {raw.returncode}")
        for r in json.loads(raw.stdout or "[]"):
            owner, _, name = str(r["nameWithOwner"]).partition("/")
            out.append(_entry(name or owner, owner, r.get("url", ""), "github", r.get("pushedAt", "")))
    except Exception as exc:
        logger.debug("gh repo list failed", exc_info=True)
        return [], f"github: {type(exc).__name__}: {exc}"
    return out, ""


def forgejo_repos(limit: int = 5) -> tuple[list[dict[str, Any]], str]:
    """(entries, error) from the Forgejo API."""
    import urllib.request

    out: list[dict[str, Any]] = []
    try:
        base = _settings.settings()["forgejo_url"].rstrip("/")
        if not base:
            raise RuntimeError("forgejo_url is not configured")
        token = _settings.forgejo_token()
        if not token:
            raise RuntimeError("no Forgejo token")
        req = urllib.request.Request(
            base + f"/api/v1/repos/search?limit={limit}&sort=updated",
            headers={"Authorization": "token " + token},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        for r in data.get("data", []):
            out.append(
                _entry(
                    r.get("name", ""),
                    (r.get("owner") or {}).get("login", ""),
                    r.get("clone_url", ""),
                    "forgejo",
                    r.get("updated_at", ""),
                )
            )
    except Exception as exc:
        logger.debug("forgejo repo search failed", exc_info=True)
        return [], f"forgejo: {type(exc).__name__}: {exc}"
    return out, ""


def discover(limit_per_source: int = 5) -> tuple[list[dict[str, Any]], list[str]]:
    """(entries, errors) from every ENABLED source.

    The entries are deduped by name (github wins on ties) and sorted by
    ``pushed_at``, newest first.
    """
    sources = _settings.settings()["repo_sources"]
    entries: list[dict[str, Any]] = []
    errors: list[str] = []

    if "github" in sources:
        gh, err = github_repos(limit_per_source)
        entries.extend(gh)
        if err:
            errors.append(err)
    if "forgejo" in sources:
        fj, err = forgejo_repos(limit_per_source)
        entries.extend(fj)
        if err:
            errors.append(err)
    if "local" in sources and not entries:
        # Local repos are the fallback, not the headline: the picker shows what
        # the user pushed, not what happens to be on disk.
        entries.extend(local_repos(limit=limit_per_source))

    seen: dict[str, dict[str, Any]] = {}
    for r in sorted(entries, key=lambda x: x.get("pushed_at", ""), reverse=True):
        seen.setdefault(r["name"], r)
    out = sorted(seen.values(), key=lambda x: x.get("pushed_at", ""), reverse=True)
    return out[: limit_per_source * 2], errors
