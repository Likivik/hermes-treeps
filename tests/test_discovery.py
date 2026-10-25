"""discovery.py: what the plugin offers the user to work in.

Three rules are pinned here, all of them from the locked decisions in
issue #37:

1. the LOCAL scan walks ``repo_dir`` and nothing else - a clone that
   lives outside the repo dir is not a candidate, because the plugin
   creates its workspaces under ``repo_dir`` and a repo it cannot
   address is a repo it cannot offer;
2. the FORGE scan is OWNER-FILTERED - a forge lists the repos of the
   login the plugin is authenticated as, and nothing else, with no
   limit. The login is DERIVED from the credentials the plugin already
   has (the Forgejo token, the ``gh`` session), so there is no setting
   to get wrong;
3. the result is ONE FLAT LIST, deduped by REMOTE IDENTITY (forge +
   owner + repo name) - never by path - so a forge repo that already
   has a local clone appears ONCE, tagged ``local``.

The tests use ``tmp_path`` only; nothing here reads or writes the live
plugin state.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from plugin_loader import PLUGIN

DISCOVERY = PLUGIN.discovery


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_repo(path: Path) -> Path:
    """A directory that looks like a repo to the scanner."""
    path.mkdir(parents=True, exist_ok=True)
    (path / ".git").mkdir(exist_ok=True)
    return path


def _entry(name: str, owner: str = "", clone_url: str = "", source: str = "forgejo"):
    """A forge catalogue row, in the shape the scan functions return.

    The scan functions already produce the flat tagged shape, so a fake
    has to produce it too - otherwise the test would be pinning a shape
    the code never sees.
    """
    return {
        "name": name,
        "tag": DISCOVERY.FORGE_TAGS.get(source, DISCOVERY.TAG_LOCAL),
        "clone_url": clone_url,
        "local_path": "",
        "owner": owner,
        "source": source,
        "url": clone_url,
        "pushed_at": "2026-10-01T00:00:00Z",
    }


@pytest.fixture(autouse=True)
def _clear_owner_cache():
    """The derived owner is cached per process - every test starts clean.

    WHY this matters: the cache is what keeps a missing token from
    becoming a network call on every gate check, and a test that left
    a cached login behind would make the next test scan a forge the
    operator never configured.
    """
    DISCOVERY._OWNER_CACHE.clear()
    yield
    DISCOVERY._OWNER_CACHE.clear()


# ── 1. the local scan is repo_dir-only ───────────────────────────────────────


def test_local_scan_lists_the_repos_under_the_repo_dir(tmp_path, monkeypatch):
    """A repo directly under ``repo_dir`` is listed."""
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    _make_repo(repo_dir / "alpha")
    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))

    names = DISCOVERY.local_names()
    assert names == ["alpha"], names


def test_local_scan_ignores_everything_outside_the_repo_dir(tmp_path, monkeypatch):
    """A clone OUTSIDE ``repo_dir`` is not a candidate.

    WHY: the plugin creates its workspaces under ``repo_dir``. A repo it
    cannot reach from there is a repo the picker would offer and then
    fail on - and the old scan read a second list (``repo_roots``) that
    had nothing to do with where the plugin actually works.
    """
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    _make_repo(repo_dir / "inside")

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    _make_repo(outside / "outside")

    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))

    names = DISCOVERY.local_names()
    assert "inside" in names, names
    assert "outside" not in names, names


def test_local_scan_ignores_the_deprecated_repo_roots(tmp_path, monkeypatch):
    """``repo_roots`` is deprecated: discovery must not read it any more.

    WHY: the key is still PARSED (so an old config does not crash the
    plugin), but a reader that still honours it would scan a second,
    unrelated tree - and the operator who set it expects it to be
    ignored.
    """
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    _make_repo(repo_dir / "inside")

    roots = tmp_path / "roots"
    roots.mkdir()
    _make_repo(roots / "from-roots")

    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))
    # The env override is a colon/comma separated list - that is the format
    # the plugin has always accepted for list settings.
    monkeypatch.setenv(
        "HERMES_TREEPS_REPO_ROOTS",
        f"{roots}:{repo_dir}",
    )

    names = DISCOVERY.local_names()
    assert "inside" in names, names
    assert "from-roots" not in names, names


def test_local_scan_does_not_descend_into_subdirectories(tmp_path, monkeypatch):
    """Only the DIRECT children of ``repo_dir`` are repos.

    WHY: a repo nested deeper is not where the plugin would put its
    workspace, and walking the tree would list a repo the picker cannot
    bind.
    """
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    _make_repo(repo_dir / "alpha" / "nested")

    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))

    names = DISCOVERY.local_names()
    assert names == [], names


# ── 2. the forge owner is derived from the credentials ──────────────────────


def test_the_github_owner_is_the_login_gh_is_authenticated_as(monkeypatch):
    """The GitHub scan asks for the repos of the login ``gh`` reports."""
    seen: dict[str, list[str]] = {}

    class _FakeProc:
        returncode = 0
        stdout = "me\n"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        return _FakeProc()

    monkeypatch.setattr(DISCOVERY.subprocess, "run", _fake_run)
    monkeypatch.setattr(DISCOVERY.shutil, "which", lambda name: "/usr/bin/gh")

    assert DISCOVERY._derive_owner("github") == "me"

    cmd = seen["cmd"]
    assert cmd[1:4] == ["api", "user", "--jq"], cmd
    assert ".login" in cmd, cmd


def test_the_forgejo_owner_is_the_login_the_token_belongs_to(monkeypatch):
    """The Forgejo scan asks for the repos of the token's own account."""
    seen: dict[str, object] = {}

    class _FakeResp:
        def read(self):
            return json.dumps({"login": "me"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = req.headers
        return _FakeResp()

    monkeypatch.setattr(DISCOVERY.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(DISCOVERY._settings, "settings", lambda: {
        "forgejo_url": "https://forgejo.example",
    })
    monkeypatch.setattr(DISCOVERY._settings, "forgejo_token", lambda: "tok")

    assert DISCOVERY._derive_owner("forgejo") == "me"

    assert seen["url"].endswith("/api/v1/user"), seen
    assert "tok" in str(seen["headers"]), seen


def test_the_derived_owner_is_cached_for_the_process(monkeypatch):
    """The login is derived ONCE, not on every scan.

    WHY: discovery runs on every gate check. Without the cache, every
    check would shell out to ``gh`` (or hit the Forgejo API) - and a
    forge whose owner cannot be derived would turn into a per-check
    network timeout.
    """
    calls: list[str] = []

    class _FakeProc:
        returncode = 0
        stdout = "me\n"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _FakeProc()

    monkeypatch.setattr(DISCOVERY.subprocess, "run", _fake_run)
    monkeypatch.setattr(DISCOVERY.shutil, "which", lambda name: "/usr/bin/gh")

    assert DISCOVERY._derive_owner("github") == "me"
    assert DISCOVERY._derive_owner("github") == "me"
    assert DISCOVERY._derive_owner("github") == "me"

    assert len(calls) == 1, calls


def test_a_github_derivation_failure_skips_the_forge_silently(monkeypatch):
    """``gh`` is not installed: GitHub is skipped, and NOTHING is raised.

    WHY: the operator may not use GitHub at all. A missing ``gh`` is
    not an error the user has to read in the middle of a session - the
    forge is simply left out of the list.
    """
    monkeypatch.setattr(DISCOVERY.shutil, "which", lambda name: None)

    entries, err = DISCOVERY.github_repos()

    assert entries == [], entries
    assert err == "", err


def test_a_forgejo_derivation_failure_skips_the_forge_silently(monkeypatch):
    """No Forgejo token: Forgejo is skipped, and NOTHING is raised."""
    def _boom(*args, **kwargs):
        raise AssertionError("the Forgejo scan must not run without an owner")

    monkeypatch.setattr(DISCOVERY.urllib.request, "urlopen", _boom)
    monkeypatch.setattr(DISCOVERY._settings, "settings", lambda: {
        "forgejo_url": "https://forgejo.example",
    })
    monkeypatch.setattr(DISCOVERY._settings, "forgejo_token", lambda: "")

    entries, err = DISCOVERY.forgejo_repos()

    assert entries == [], entries
    assert err == "", err


def test_a_forge_scan_is_skipped_when_the_login_call_fails(monkeypatch):
    """The API answers with an error: the forge is skipped, quietly."""
    class _FakeProc:
        returncode = 1
        stdout = ""
        stderr = "not logged in"

    monkeypatch.setattr(DISCOVERY.subprocess, "run", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(DISCOVERY.shutil, "which", lambda name: "/usr/bin/gh")

    entries, err = DISCOVERY.github_repos()

    assert entries == [], entries
    assert err == "", err


# ── 3. the forge scan is owner-filtered ─────────────────────────────────────


def test_the_forge_scan_uses_the_derived_owner(monkeypatch):
    """The GitHub scan asks for the repos of the DERIVED owner.

    WHY: listing every repo the token can see leaks other people's
    repos into the picker, and it is slow - an owner has tens of repos,
    an account has thousands.
    """
    seen: dict[str, list[str]] = {}

    class _FakeProc:
        returncode = 0
        stdout = "me\n"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        seen.setdefault("cmd", []).append(list(cmd))
        if cmd[1:4] == ["api", "user", "--jq"]:
            return _FakeProc()
        return type(
            "_RepoProc",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps([{"nameWithOwner": "me/alpha", "url": "u", "pushedAt": "t"}]),
                "stderr": "",
            },
        )()

    monkeypatch.setattr(DISCOVERY.subprocess, "run", _fake_run)
    monkeypatch.setattr(DISCOVERY.shutil, "which", lambda name: "/usr/bin/gh")

    entries, err = DISCOVERY.github_repos()

    assert err == "", err
    assert [e["name"] for e in entries] == ["alpha"], entries
    listing = [c for c in seen["cmd"] if c[1:2] == ["repo"]]
    assert listing, seen
    assert "me" in listing[0], listing


def test_the_forge_scan_lists_every_repo_of_the_owner(monkeypatch):
    """No limit: every repo of the owner is listed.

    WHY: a limit hides repos the user owns but did not push recently,
    and the picker is a menu, not a search.
    """
    seen: dict[str, list[str]] = {}

    class _FakeProc:
        returncode = 0
        stdout = "me\n"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        seen.setdefault("cmd", []).append(list(cmd))
        if cmd[1:4] == ["api", "user", "--jq"]:
            return _FakeProc()
        return type(
            "_RepoProc",
            (),
            {"returncode": 0, "stdout": "[]", "stderr": ""},
        )()

    monkeypatch.setattr(DISCOVERY.subprocess, "run", _fake_run)
    monkeypatch.setattr(DISCOVERY.shutil, "which", lambda name: "/usr/bin/gh")

    DISCOVERY.github_repos()

    listing = [c for c in seen["cmd"] if c[1:2] == ["repo"]]
    assert listing, seen
    assert "--limit" not in listing[0], listing


def test_the_forgejo_scan_lists_the_repos_of_the_derived_owner(monkeypatch):
    """The Forgejo scan asks the user endpoint for the DERIVED owner."""
    seen: dict[str, str] = {}

    class _FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(req, timeout=None):
        url = req.full_url
        seen["url"] = url
        if url.endswith("/api/v1/user"):
            return _FakeResp({"login": "me"})
        return _FakeResp([
            {
                "name": "alpha",
                "owner": {"login": "me"},
                "clone_url": "https://forgejo.example/me/alpha.git",
                "updated_at": "2026-10-01T00:00:00Z",
            },
        ])

    monkeypatch.setattr(DISCOVERY.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(DISCOVERY._settings, "settings", lambda: {
        "forgejo_url": "https://forgejo.example",
    })
    monkeypatch.setattr(DISCOVERY._settings, "forgejo_token", lambda: "tok")

    entries, err = DISCOVERY.forgejo_repos()

    assert err == "", err
    assert [e["name"] for e in entries] == ["alpha"], entries
    assert "/api/v1/users/me/repos" in seen["url"], seen


# ── 4. one flat list, deduped by remote identity ────────────────────────────


def test_the_result_is_one_flat_list_of_tagged_entries(tmp_path, monkeypatch):
    """Local and forge entries live in ONE list, each carrying a tag.

    WHY: the picker renders one menu. Two lists (a local one and a
    remote one) would mean two menus, or a merge at the render site -
    and the merge is exactly where the duplicate would sneak in.
    """
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    _make_repo(repo_dir / "alpha")

    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))
    monkeypatch.setattr(
        DISCOVERY,
        "github_repos",
        lambda limit=0: ([_entry("beta", "me", "https://gh/me/beta", "github")], ""),
    )
    monkeypatch.setattr(
        DISCOVERY,
        "forgejo_repos",
        lambda limit=0: ([_entry("gamma", "me", "https://fj/me/gamma", "forgejo")], ""),
    )

    entries, errors = DISCOVERY.discover()

    assert errors == [], errors
    assert isinstance(entries, list), entries
    tags = {e["name"]: e["tag"] for e in entries}
    assert tags == {
        "alpha": "local",
        "beta": "forge:github",
        "gamma": "forge:forgejo",
    }, tags


def test_every_entry_carries_the_flat_shape(tmp_path, monkeypatch):
    """Each entry has name, tag, clone_url and owner.

    WHY: the shape is the CONTRACT with the picker (Task 4). A missing
    key is a KeyError in the render path, which is where a user sees it.
    """
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    _make_repo(repo_dir / "alpha")

    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))
    monkeypatch.setattr(
        DISCOVERY,
        "github_repos",
        lambda limit=0: ([_entry("beta", "me", "https://gh/me/beta", "github")], ""),
    )
    monkeypatch.setattr(DISCOVERY, "forgejo_repos", lambda limit=0: ([], ""))

    entries, _ = DISCOVERY.discover()

    for entry in entries:
        assert set(("name", "tag", "clone_url", "owner")) <= set(entry), entry
        assert entry["tag"] in ("local", "forge:github", "forge:forgejo"), entry
        assert isinstance(entry["clone_url"], str), entry
        assert isinstance(entry["owner"], str), entry


def test_a_local_entry_carries_its_path(tmp_path, monkeypatch):
    """A local entry says WHERE it is.

    WHY: the picker binds a workspace in that path. A local entry
    without a path cannot be bound without a second lookup, and the
    second lookup is where the wrong repo gets picked.
    """
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    alpha = _make_repo(repo_dir / "alpha")

    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))
    monkeypatch.setattr(DISCOVERY, "github_repos", lambda limit=0: ([], ""))
    monkeypatch.setattr(DISCOVERY, "forgejo_repos", lambda limit=0: ([], ""))

    entries, _ = DISCOVERY.discover()

    local = [e for e in entries if e["tag"] == "local"]
    assert len(local) == 1, entries
    assert Path(local[0]["local_path"]) == alpha, local


def test_a_forge_repo_with_a_local_clone_appears_once_as_local(
    tmp_path, monkeypatch,
):
    """A repo that exists BOTH on disk and on the forge appears ONCE.

    WHY: this is the whole point of the remote-identity dedupe. The
    user knows one repo; the picker offering it twice (once "local",
    once "forge") makes them tap the wrong one and clone a copy they
    already have.
    """
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    _make_repo(repo_dir / "alpha")

    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))
    monkeypatch.setattr(
        DISCOVERY,
        "github_repos",
        lambda limit=0: ([_entry("alpha", "me", "https://gh/me/alpha", "github")], ""),
    )
    monkeypatch.setattr(DISCOVERY, "forgejo_repos", lambda limit=0: ([], ""))

    entries, _ = DISCOVERY.discover()

    assert len(entries) == 1, entries
    assert entries[0]["tag"] == "local", entries


def test_the_dedupe_matches_on_the_remote_identity_not_the_path(
    tmp_path, monkeypatch,
):
    """Two DIFFERENT repos with the same NAME are two entries.

    WHY: the old dedupe keyed on the name, so ``me/alpha`` and
    ``other/alpha`` collapsed into one. The identity is forge + owner +
    name - the thing that decides WHERE a clone would land.

    The local clone here has no origin, so it cannot be attributed to
    either of them: the name is claimed by TWO forges, and pairing the
    clone with either one would be a guess. So all three stay.
    """
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    _make_repo(repo_dir / "alpha")

    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))
    monkeypatch.setattr(
        DISCOVERY,
        "github_repos",
        lambda limit=0: ([_entry("alpha", "me", "https://gh/me/alpha", "github")], ""),
    )
    monkeypatch.setattr(
        DISCOVERY,
        "forgejo_repos",
        lambda limit=0: ([_entry("alpha", "other", "https://fj/other/alpha", "forgejo")], ""),
    )

    entries, _ = DISCOVERY.discover()

    by_tag = {e["tag"]: e for e in entries}
    assert set(by_tag) == {"local", "forge:github", "forge:forgejo"}, entries


def test_a_local_clone_with_an_origin_does_not_hide_a_same_named_forge_repo(
    tmp_path, monkeypatch,
):
    """A clone WITH an origin collapses only with ITS forge row.

    WHY: the opposite failure is just as bad as the duplicate. If the
    local clone swallowed the GitHub row too, a repo the user owns but
    has NOT cloned would disappear from the picker - and they would
    never learn it exists.
    """
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    alpha = _make_repo(repo_dir / "alpha")
    subprocess.run(
        ["git", "init", "-q", str(alpha)], capture_output=True, text=True, check=False,
    )
    subprocess.run(
        ["git", "-C", str(alpha), "remote", "add", "origin",
         "https://forgejo.example/me/alpha.git"],
        capture_output=True, text=True, check=False,
    )

    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))
    monkeypatch.setattr(
        DISCOVERY,
        "github_repos",
        lambda limit=0: ([_entry("alpha", "me", "https://gh/me/alpha", "github")], ""),
    )
    monkeypatch.setattr(
        DISCOVERY,
        "forgejo_repos",
        lambda limit=0: (
            [_entry("alpha", "me", "https://forgejo.example/me/alpha.git", "forgejo")],
            "",
        ),
    )

    entries, _ = DISCOVERY.discover()

    tags = sorted(e["tag"] for e in entries)
    assert tags == ["forge:github", "local"], entries


def test_a_local_clone_of_a_forge_repo_is_deduped_by_its_origin(
    tmp_path, monkeypatch,
):
    """The local clone is matched to the forge entry THROUGH its origin.

    WHY: the name alone is not an identity - the same name can exist
    under two owners, and a local clone's origin URL is the only thing
    that says WHICH of them it is. The clone lives at
    ``<repo_dir>/<name>``, so the origin is what ties the two together.
    """
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    alpha = _make_repo(repo_dir / "alpha")
    # A real git clone with an origin URL.
    subprocess.run(
        ["git", "init", "-q", str(alpha)], capture_output=True, text=True, check=False,
    )
    subprocess.run(
        ["git", "-C", str(alpha), "remote", "add", "origin",
         "https://forgejo.example/me/alpha.git"],
        capture_output=True, text=True, check=False,
    )

    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))
    monkeypatch.setattr(DISCOVERY, "github_repos", lambda limit=0: ([], ""))
    monkeypatch.setattr(
        DISCOVERY,
        "forgejo_repos",
        lambda limit=0: (
            [_entry("alpha", "me", "https://forgejo.example/me/alpha.git", "forgejo")],
            "",
        ),
    )

    entries, _ = DISCOVERY.discover()

    assert len(entries) == 1, entries
    assert entries[0]["tag"] == "local", entries


def test_a_forge_repo_without_a_local_copy_stays_a_forge_entry(
    tmp_path, monkeypatch,
):
    """A forge repo that is NOT on disk keeps its forge tag.

    WHY: the tag is what tells the picker whether to bind or to offer a
    clone. Tagging a remote-only repo "local" would bind a workspace in
    a directory that does not exist.
    """
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    _make_repo(repo_dir / "alpha")

    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))
    monkeypatch.setattr(DISCOVERY, "github_repos", lambda limit=0: ([], ""))
    monkeypatch.setattr(
        DISCOVERY,
        "forgejo_repos",
        lambda limit=0: (
            [_entry("remote-only", "me", "https://fj/me/remote-only.git", "forgejo")],
            "",
        ),
    )

    entries, _ = DISCOVERY.discover()

    tags = {e["name"]: e["tag"] for e in entries}
    assert tags["remote-only"] == "forge:forgejo", tags
    # The key is always there (a renderer must not KeyError on it), but
    # it is empty for a repo that is not on disk.
    remote = [e for e in entries if e["name"] == "remote-only"][0]
    assert remote["local_path"] == "", remote


def test_the_local_repos_come_first_and_the_forge_rows_by_recency(
    tmp_path, monkeypatch,
):
    """The list is ordered: local first, then the forge rows newest first.

    WHY: a local repo is the one the plugin can bind right now, and the
    forge rows keep the "most recently pushed first" order the picker
    has always shown. The name breaks ties, so the menu does not
    reshuffle between two calls that saw the same repos.
    """
    repo_dir = tmp_path / "rd"
    repo_dir.mkdir()
    _make_repo(repo_dir / "alpha")

    monkeypatch.setenv("HERMES_TREEPS_REPO_DIR", str(repo_dir))
    monkeypatch.setattr(
        DISCOVERY,
        "github_repos",
        lambda limit=0: (
            [_entry("older", "me", "https://gh/me/older", "github")],
            "",
        ),
    )
    monkeypatch.setattr(
        DISCOVERY,
        "forgejo_repos",
        lambda limit=0: ([_entry("newer", "me", "https://fj/me/newer", "forgejo")], ""),
    )
    # Give the two forge rows different recency.
    rows = {
        "older": "2026-01-01T00:00:00Z",
        "newer": "2026-10-01T00:00:00Z",
    }
    real_entry = DISCOVERY._entry

    def _entry_with_recency(name, owner="", clone_url="", source="forgejo", **kwargs):
        row = real_entry(name, owner, clone_url, source, **kwargs)
        row["pushed_at"] = rows.get(name, "")
        return row

    monkeypatch.setattr(DISCOVERY, "_entry", _entry_with_recency)

    entries, _ = DISCOVERY.discover()

    assert [e["name"] for e in entries] == ["alpha", "newer", "older"], entries


# ── the remote identity, parsed out of a URL ────────────────────────────────


def test_an_https_clone_url_yields_the_forge_the_owner_and_the_name():
    """The identity of a clone URL."""
    assert DISCOVERY.remote_identity("https://github.com/me/alpha.git") == (
        "github", "me", "alpha",
    )
    assert DISCOVERY.remote_identity("https://forgejo.example/me/alpha") == (
        "forgejo", "me", "alpha",
    )


def test_a_host_that_names_no_known_forge_has_no_forge():
    """A self-hosted forge on an unknown host: the owner and name still parse.

    WHY: the forge is filled in from the catalogue row in that case - the
    row knows which forge it came from, the URL does not.
    """
    assert DISCOVERY.remote_identity("https://git.example.com/me/alpha.git") == (
        "", "me", "alpha",
    )


def test_an_scp_like_clone_url_yields_the_owner_and_the_name():
    """``git@host:owner/repo.git`` is an origin too.

    WHY: a clone made over SSH has this shape, and the first segment
    carries the HOST, not the owner. Reading it as the owner would pair
    the clone with the wrong forge row - and the wrong row is a repo
    the user does not have.
    """
    assert DISCOVERY.remote_identity("git@github.com:me/alpha.git") == (
        "github", "me", "alpha",
    )


def test_an_empty_clone_url_has_no_identity():
    """No URL, no identity - and no dedupe against anything."""
    assert DISCOVERY.remote_identity("") == ("", "", "")
    assert DISCOVERY.remote_identity("   ") == ("", "", "")


# ── the settings the forge scan reads ───────────────────────────────────────


def test_the_forge_repos_setting_does_not_exist():
    """There is no ``forge_repos`` setting: the owner is derived.

    WHY: a setting would be a second place for the operator to name a
    login, and a wrong login lists somebody else's repos. The plugin
    already knows who the user is - from the token and the ``gh``
    session.
    """
    from plugin_loader import SETTINGS

    cfg = SETTINGS.settings()
    assert "forge_repos" not in cfg, sorted(cfg)
