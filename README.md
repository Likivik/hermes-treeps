# hermes-treeps

One jj workspace per Hermes session, enforced.

A session is either bound to a repo (its own workspace under
`/Git/hermes/workspaces/<repo>/<slug>`) or explicitly "just chatting"
(HERMES_HOME as cwd, full capabilities). Repo-touching tools are blocked until
the session resolves one of the two — so a workspace can never be created
without a session record, and a session can never quietly work in someone
else's tree.

## What's inside

| File | Purpose |
|---|---|
| `__init__.py` | tools (`ws_new`, `ws_chatting`, `ws_rename`, `ws_status`, `ws_doctor`), the tool gate, `hermes ws` CLI, system-prompt section |
| `telegram_picker.py` | Telegram inline keyboard: repo picker on `/new`, tap handling, `/wsbtn` |
| `store.py` | workspaces, session records, tokens, remote repo listing (GitHub + Forgejo), retire ladder, gc tiers |
| `db.py` | the SQLite schema and every query: sessions, bindings, workspaces, remote cache, legacy import |
| `settings.py` | one accessor for the config: defaults, env overrides, validation, the manifest schema contract |
| `discovery.py` | repo discovery: local roots plus the GitHub/Forgejo catalogue |
| `vcs.py` | the jj/git adapter: one API, two backends |
| `errors.py` | the plugin's exception types (`WorkspaceError` and friends) |

## Behavior

- **Picker on `/new`** — the gateway's `on_session_reset` hook fires a keyboard
  with your most recently pushed GitHub/Forgejo repos plus `just chatting` and
  `new repo…`. One tap binds the session; the keyboard retires itself.
- **Tool gate** — `terminal`, `write_file`, etc. are refused with a message that
  names the candidate repos until the session resolves. Fresh sessions are
  asked; delegate_task children inherit their parent's binding (see below).
- **1:1 session ↔ workspace** — `ws_new` on a bound session asks for explicit
  `rebind=true` instead of silently dual-binding. Rebinding detaches the old
  tree (the directory stays).
- **Not on disk ≠ does not exist** — `ws_new` for a repo that is not under
  the configured `repo_roots` checks the GitHub/Forgejo catalogue. If the repo
  lives there,
  the answer is `needs_confirmation: true` with the remote's name, owner, url
  and source, and nothing is written to disk; the clone happens only on the
  second call, `ws_new(repo, slug, create_repo=true)`, after the user approves.
  If the repo is nowhere, the error names the remotes that do exist, so nobody
  keeps guessing repo names. `ws_doctor` lists the same picture
  (`repo_roots`, `local_candidates`, `remote_candidates`).
- **Titles** — the greeting requires a session title; it becomes the workspace
  slug. Picker taps without a title get a temporary name and can be renamed
  with `ws_rename` (jj-safe: workspaces are keyed by working-copy commit, not
  path).
- **Retire ladder** — `hermes ws retire <slug>` refuses on open PRs (matched to
  the workspace's own branches), dirty trees, unpushed commits, and live
  sessions. `--force` overrides unverifiable checks; `--ultraforce` overrides
  everything but still archives the record.
- **gc** — `hermes ws gc` classifies (dry-run): tier1 safe-to-retire, tier2
  needs-a-look, tier3 live. `--apply` retires tier1 only. No cron is installed.

## Delegated children (airtightness layers)

1. `is_delegated_child_context()` — exact, single-path children.
2. Delegation-pointer fallback — covers **batch** children while upstream
   [#96483](https://github.com/NousResearch/hermes-agent/issues/96483) is open
   (`_run_single_child` never enters the child context). Remove this layer when
   that fix ships; until then a fresh session created within the 10-minute
   pointer window inherits too (the known residue).

## Configuration

Every setting lives in one namespace: `plugins.entries.hermes-treeps.settings`.
The same keys are editable from **Desktop → Settings → Plugins** (the form is
generated from the manifest's `config_schema`) or by hand in `config.yaml`:

```yaml
plugins:
  entries:
    hermes-treeps:
      settings:
        workspace_root: /Git/hermes/workspaces   # where <repo>/<slug> lives
        repo_roots: [/Git, /Git/hermes, /Git/likivik, /Git/repos]
        clone_root: /Git/hermes                 # where a new repo is created
        repo_sources: [local, github, forgejo]  # which catalogues the picker reads
        vcs: auto                               # auto | jj | git
        extra_bin_dirs: [/run/current-system/sw/bin, /usr/local/bin, /usr/bin]
        picker_enabled: true
        inject_first_message: false             # open gateway sessions with the question itself
        allowed_user_ids: []                    # telegram ids allowed to tap the picker
        forgejo_url: ""                         # e.g. https://forgejo.example
```

`forgejo_token` is a secret: it is declared in the manifest with
`type: secret` and `env: FORGEJO_TOKEN`, and is stored in `.env`, never in
`config.yaml`. Set it either way:

```
hermes config set FORGEJO_TOKEN <token>     # writes .env
```

or through **Desktop → Settings → Plugins → hermes-treeps → forgejo_token**,
which writes the same `.env` name. The plugin reads that name via the host's
secret scope (`agent.secret_scope.get_secret`), falling back to the process
environment.

There is no fallback to the old top-level `plugins.entries.hermes-treeps.<key>`
form — one namespace, one source of truth. A relative path, an unknown
`repo_sources` entry or an unknown `vcs` is refused with an error that names
the key.

Environment overrides exist for the same keys (`HERMES_TREEPS_WORKSPACE_ROOT`,
`HERMES_TREEPS_FORGEJO_URL`, `FORGEJO_URL`, …) and win over `config.yaml`.
`forgejo_token` is not one of them: it is a secret, and it is read from `.env`,
not from the settings namespace.

## State

All state lives in one SQLite database under the host's plugin-data
directory: `<hermes home>/plugin-data/hermes-treeps/data.db` (WAL mode). The
session bindings, the workspace records and the remote-repo cache are tables
in it, and every mutation is one transaction — the old JSON read-modify-write
lost half of eight concurrent bindings.

The plugin install directory holds no state, so `hermes plugins update` can
replace it without touching anything. Legacy JSON records (the old
`state/*.json` files and the in-tree `<workspace>/.hermes/session.json` files)
are imported into the database by an explicit `hermes ws migrate` — nothing
imports them on open. `hermes ws migrate --purge` deletes the in-tree copies
after writing a tarball backup of them.

## Requirements

- Hermes Agent with the plugin hook set used here (`on_session_reset`,
  `on_session_finalize`, `pre_tool_call`, `on_session_start`),
  `register_telegram_handler` for the picker.
- `jj` (workspaces are jj-first; git worktrees work for non-jj repos).
- Developed against Hermes 0.21.x on NixOS; no third-party pip deps (stdlib
  only — profiles with read-only venvs can load it).

## Install

```
hermes plugins install Likivik/hermes-treeps
```

## License

MIT
