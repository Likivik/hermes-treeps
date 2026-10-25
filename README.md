# hermes-treeps

One jj workspace per Hermes session, enforced.

A session is either bound to a repo (its own workspace under
`<repo_dir>/.workspaces/<repo>/<slug>`) or explicitly "just chatting"
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
| `prepare.py` | workspace prepare step: copies ignored files (`.env`), optional setup command |
| `childspace.py` | per-delegation child workspaces (one tree per child, never shared) |
| `migrate_layout.py` | the one-time move from the old workspace root to the dot-dir layout |
| `contract.py` | the delegation contract rendered into the system-prompt section |
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

Children are detected exactly, by `is_delegated_child_context()` (single-path;
no recency fallback - the upstream #96483 workaround layer is gone because the
upstream bug is fixed).

## Child-end reporting

When a delegated child finishes, the **parent** session learns what the child
left behind — on its next turn, as one line per child:

```
child parent-tree-child-abcdef: dirty, 2 unpushed commits - /Git/hermes/.workspaces/repo/parent-tree-child-abcdef (worker completed)
```

The line carries the four things the parent cannot otherwise know: the
workspace (slug and path), clean-or-dirty, how many unpushed commits the child
left, and the child's own status word (`completed`, `failed`, `interrupted`).

**How it works.** The plugin registers the host's dedicated subagent
lifecycle hooks — `subagent_start` (records which child belongs to which
parent) and `subagent_stop` (the child's end state) — plus `pre_llm_call`,
which is the per-turn channel the report rides. The dirty and unpushed checks
run on a background thread, because the host runs `subagent_stop` on the
parent's own thread and a `jj status` there would stall every delegation
behind it. That background thread is handed the database path its caller
named, so a report can never land in a different database than the one the
session was using.

**One-time delivery.** A report is delivered once and then deleted. The next
turn is silent. With nothing pending, the plugin adds nothing at all — no
empty section, no filler.

**Honest answers.** When a check cannot run (the workspace was retired
mid-run, the remote is unreachable), the line says `state-unknown` or
`unpushed unknown` — never a wrong "clean".

**Hard-death caveat.** If the gateway restarts mid-child, `subagent_stop`
never fires and no report is produced. The orphaned child workspace is still
reclaimed by age: `hermes ws gc --children --apply` retires child workspaces
once they are 14 days old, their session ended and their tree is clean.

## Automatic child collection (optional)

Child workspaces (delegated children) are reclaimable once they are 14 days
old, their session ended and their tree is clean - committed work survives
retirement in the shared repo store. Collection is manual by default:

    hermes ws gc            # classify, dry-run
    hermes ws gc --children --apply   # retire expired CHILD workspaces only

To run it unattended, add a cron line (the `--children` flag makes it safe for
an unattended run: it never retires a user workspace, never decides pushed/PR
questions):

    0 4 * * * hermes ws gc --children --apply >> ~/.hermes/ws-gc.log 2>&1

## Immediate child reclamation

A delegated child's workspace is throwaway, and once the child is **done**
the tree is dead weight on disk. `hermes ws reclaim` takes it back at once -
and the work is never lost.

    hermes ws reclaim --slug <slug>          # reclaim one finished child
    hermes ws reclaim --all-children         # reclaim every finished child
    hermes ws reclaim --slug <slug> --dry-run  # report only, change nothing

The output is a structured report:

```json
{
  "reclaimed": [
    {"slug": "app-child-a1b2c3", "snapshot_commit_id": "9f3c...", "was_dirty": true}
  ],
  "skipped": [
    {"slug": "app-child-d4e5f6", "reason": "live_session_pinned"}
  ]
}
```

### How the work survives

`jj workspace remove` (jj >= 0.46) does both halves in one step: it
**snapshots the working copy - even the dirty one - into a commit in the
shared store**, and then removes the workspace and its directory. The commit
survives in the repo; the tree does not. A clean child has nothing to
snapshot, so it is unregistered with `jj workspace forget` instead.

The plugin records the snapshot's commit id in the `child_reclaims` table.
**That table is the index**: jj never auto-gcs, so a snapshot commit stays in
the store until something evicts it, and this row is what makes it findable:

    hermes ws reclaim --slug app-child-a1b2c3   # → {"snapshot_commit_id": "9f3c..."}

Recover the work with that id:

    jj log -r 9f3c...
    jj file show -r 9f3c... path/to/the/file

### The one thing that can still evict it

A manual `jj gc --expire` is the only thing that can remove an unpushed
snapshot commit from the store. Until then the work is there, reachable
through the recorded commit id.

### What is refused, and why

| Refusal | The failure it prevents |
|---|---|
| not a `child:` purpose | deleting a **user's** workspace |
| primary (main) jj workspace | jj forgetting the main checkout with no warning (jj#5271) |
| live session pinned to it | the tree vanishing under a running session |
| git-only worktree | a dirty git worktree has no snapshot-safe removal - it is **skipped**, not lost |

The command is idempotent: reclaiming an already-reclaimed child succeeds
and keeps the **first** snapshot's commit id.

### The age-based sweep still runs

The 14-day TTL protects **unfinished** children - a child that is still
working, or was never heard from, keeps its tree until the age says it was
abandoned. A child past `subagent_stop` is finished by contract, so the
immediate path is safe to reclaim at once. The age-based sweep
(`hermes ws gc --children --apply`) keeps running for everything the
immediate path did not reclaim.

## Configuration

Every setting lives in one namespace: `plugins.entries.hermes-treeps.settings`.
The same keys are editable from **Desktop → Settings → Plugins** (the form is
generated from the manifest's `config_schema`) or by hand in `config.yaml`:

```yaml
plugins:
  entries:
    hermes-treeps:
      settings:
        repo_dir: /Git/hermes                     # the repo dir; workspaces live in its .workspaces/
        clone_root: /Git/hermes                 # where a new repo is created
        repo_sources: [local, github, forgejo]  # which catalogues the picker reads
        vcs: auto                               # auto | jj | git
        extra_bin_dirs: [/run/current-system/sw/bin, /usr/local/bin, /usr/bin]
        picker_enabled: true
        allowed_user_ids: []                    # telegram ids allowed to tap the picker
        forgejo_url: ""                         # e.g. https://forgejo.example
        prepare_enabled: true                  # run the workspace prepare step
        prepare_command: ""                    # shell command run inside the new workspace
        prepare_files: [".env"]                # repo-relative files copied into the workspace
        prepare_timeout_seconds: 300           # ceiling on the prepare command (1-3600)
```

### Deprecated settings

| key | status | replacement |
| --- | --- | --- |
| `workspace_root` | **deprecated** — still parsed, no longer decides where a workspace goes | `repo_dir` (the root is derived: `<repo_dir>/.workspaces`) |
| `repo_roots` | **deprecated** — still read for this release | `repo_dir` (repo discovery follows it) |

Both keys are still accepted so an old config does not break the plugin, and
an explicit value that differs from the new behaviour produces a **one-time**
warning naming the replacement. The warning is logged once per process — not
on every session start — because `settings()` is read on every gate check.

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

Environment overrides exist for the same keys (`HERMES_TREEPS_REPO_DIR`,
`HERMES_WS_REPO_DIR`, `HERMES_TREEPS_FORGEJO_URL`, `FORGEJO_URL`, …) and win
over `config.yaml`. `forgejo_token` is not one of them: it is a secret, and
it is read from `.env`, not from the settings namespace.

## Workspace layout

Session workspaces live in a dot-dir **next to the repos they belong to**:

```
<repo_dir>/.workspaces/<repo>/<slug>       # e.g. /Git/hermes/.workspaces/hermes-treeps/hermes-treeps-ee445923
```

The old layout was `<workspace_root>/<repo>/<slug>`
(`/Git/hermes/workspaces/hermes-treeps/…`).

**Why a dot-dir.** A plain `ls /Git/hermes` shows the repos and nothing else —
the workspaces are visible only to someone who asks for them, so a workspace
can never be mistaken for a repo, and the repo scanner never walks into one.

**Why the depth is the same.** `workspaces/<repo>/<slug>` and
`.workspaces/<repo>/<slug>` are the same number of levels below the repo dir.
That is what lets a moved jj workspace keep its **relative** `.jj/repo`
pointer valid: the pointer climbs out of the workspace root and back down
into the repo dir, and both roots sit at the same depth. The migration
verifies this after every move and reports `verify_failed` if it does not
hold.

The slug format is unchanged: `<repo>-<last 6 of the session id>` for a
session workspace, `<repo>-<sid6>-child-<hex>` for a delegated child.

### `hermes ws migrate` — the one-time move

Existing workspaces are **not** moved automatically. Run it once, by hand:

```
hermes ws migrate --dry-run    # report the plan, move nothing
hermes ws migrate              # do it
```

The migration walks the database, and for every workspace whose recorded path
is under the old root it applies these rules **in this order**:

| rule | result | why |
| --- | --- | --- |
| the tree has uncommitted changes | `skipped_dirty` — **not moved** | silently relocating uncommitted work leaves it in a tree nobody is looking at; the files are still there, but the session that owned them finds a path that is gone |
| a live session is bound to it (or has it pinned as its cwd) | `in_use` — **not moved** | moving a tree out from under a running session breaks it mid-task |
| otherwise | `moved` | the directory is renamed, and the workspace record *and* every session binding that named the old path are rewritten |
| it is a git worktree | `git -C <main checkout> worktree repair <new path>` | the main checkout's worktree list still names the old path, so git marks the entry `prunable` |
| the `.jj/repo` pointer no longer resolves to the same store | `verify_failed` — **not rolled back** | a rollback is a second move, and a second move can fail in a new place; the workspace stays moved and the failure is named |
| the worktree repair failed | `repair_failed` — **not fatal** | the move itself succeeded and the records are correct; run the repair by hand |

The command prints the five buckets and their counts. The same data is
available as a dict (`moved`, `skipped_dirty`, `in_use`, `verify_failed`,
`repair_failed`) for anything that wants to read it.

> `hermes ws migrate` used to import the legacy JSON records. That import
> still exists under the explicit name **`hermes ws migrate-legacy`**
> (`--purge` deletes the in-tree copies after writing a tarball backup).

## Prepare

A workspace is not usable the moment `jj workspace add` returns. Every other
harness for this job runs a **prepare step** right after creating the
workspace: copy the `.env`, symlink the dependency directories, run the
install. This plugin does the same, in `prepare.py`, with four settings:

| key | default | meaning |
| --- | --- | --- |
| `prepare_enabled` | `true` | master switch — when false, nothing is copied and no command runs |
| `prepare_files` | `[".env"]` | repo-relative files copied into the new workspace |
| `prepare_command` | `""` | shell command run with `bash -lc` **inside** the new workspace (empty means none) |
| `prepare_timeout_seconds` | `300` | hard ceiling on the command (1–3600) |

**What runs when.** After a workspace is created, the prepare step copies
each file in `prepare_files` from the repo root into the workspace, then runs
`prepare_command` with the workspace as the working directory.

**Files.** Only regular files that exist are copied. A missing file is
skipped with a note, not an error — a repo without a `.env` is normal. An
existing file in the workspace is **never overwritten** (your hand-edit
survives). A path that points outside the repo root is skipped.

**The command runs arbitrary code by design.** It is operator-configured:
anyone who can edit `config.yaml` can already run code as you, so the
command is not sandboxed. What the plugin does guarantee is that the command
runs with its working directory set to the workspace — never anywhere else —
and that it is killed when it exceeds `prepare_timeout_seconds`.

**On failure.** A non-zero exit code or a timeout is *reported*, not raised:
the workspace stays on disk and stays bound, and the result comes back with
`ok: false` and the tail of the command's output. Fix it by hand and re-run
the step — nothing is deleted, and no session is taken down.

The result is returned under a `prepare` key in the `ws_new` response:

```json
{
  "ok": true,
  "path": "/Git/hermes/.workspaces/hermes-pylon/fix-the-widget",
  "vcs": "jj",
  "prepare": {
    "ok": true,
    "copied": [".env"],
    "skipped": [],
    "command": {"ran": true, "exit_code": 0, "output_tail": ""},
    "duration_s": 0.412
  }
}
```

To run the step again — after fixing a failed prepare, or on a workspace
created before this feature existed — use the `ws_prepare` tool with the
workspace path (it defaults to this session's workspace).

### Example: a nix devshell project

A project that needs `GRADLE_USER_HOME` and a jj/git environment set up
before anything builds:

```yaml
plugins:
  entries:
    hermes-treeps:
      settings:
        prepare_enabled: true
        prepare_files: [".env", "gradle.properties"]
        prepare_command: >-
          nix develop --command bash -c '
            export GRADLE_USER_HOME="$PWD/.gradle-home";
            export GIT_DIR="$PWD/.jj/repo/store/git";
            export GIT_WORK_TREE="$PWD";
            gradle --version >/dev/null
          '
        prepare_timeout_seconds: 600
```

The command inherits the workspace as its working directory, so `$PWD` is
the new workspace and the exports point at the workspace's own Gradle cache
and jj store — not at the source repo.

## State

All state lives in one SQLite database under the host's plugin-data
directory: `<hermes home>/plugin-data/hermes-treeps/data.db` (WAL mode). The
session bindings, the workspace records and the remote-repo cache are tables
in it, and every mutation is one transaction — the old JSON read-modify-write
lost half of eight concurrent bindings.

The plugin install directory holds no state, so `hermes plugins update` can
replace it without touching anything. Legacy JSON records (the old
`state/*.json` files and the in-tree `<workspace>/.hermes/session.json` files)
are imported into the database by an explicit `hermes ws migrate-legacy` —
nothing imports them on open. `hermes ws migrate-legacy --purge` deletes the
in-tree copies after writing a tarball backup of them.

## Requirements

- Hermes Agent with the plugin hook set used here (`on_session_reset`,
  `on_session_finalize`, `pre_tool_call`, `on_session_start`,
  `subagent_start`, `subagent_stop`, `pre_llm_call`),
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

## Delegation contract

`contract.py` holds a compact block of imperative rules for delegating work -
one concern per PR, numbered defect lists with `file:line`, new tests that
fail on today's code, never touching live state, one pytest run per iteration,
dropping the P2s out loud, and verifying behaviour yourself rather than
trusting a green suite. The rules are derived from measured failures, not
preferences: a three-paragraph brief ran 52 minutes, died at 60% and needed
three follow-up PRs while an equivalent numbered checklist finished in 13
minutes; a child wrote into the live state dir and deleted a live session
record; a test run wrote 38 rows into the live database; three defects passed
a green suite and appeared only when the live command ran.

The text is meant to be injected into the system prompt of every session, so
it is deliberately short - `RULES` is under 1500 characters, and brevity is
the point: a long block gets skimmed. `render()` returns the block with the
plugin's own tool names (`ws_new`, `ws_chatting`, `ws_doctor`) substituted, and
`as_prompt_section()` wraps it as `## Delegation contract` for appending to a
prompt. Rendering is idempotent, so calling it twice never doubles the text.

To extend the rules, edit the `RULES` string in `contract.py` and add a
matching assertion in `tests/test_contract.py` - the tests assert on
substrings rather than the full wording, so wording changes do not need a
test rewrite, but a new rule should still be pinned by at least one
substring check.
