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
| `childreport.py` | child-end reporting: the child's end state for the parent, and the immediate reclamation that follows it |
| `migrate_layout.py` | the one-time move from the old workspace root to the dot-dir layout |
| `jjlayout.py` | jj workspace layout facts: is this a jj workspace, is it the primary one, which session paths are live |
| `contract.py` | the delegation contract rendered into the system-prompt section |
| `errors.py` | the plugin's exception types (`WorkspaceError` and friends) |

## Behavior

- **Picker on `/new`** — the gateway's `on_session_reset` hook fires a keyboard
  with your most recently pushed GitHub/Forgejo repos plus `just chatting` and
  `new repo…`. One tap binds the session; the keyboard retires itself.
  A repo that is **not on disk** takes one extra tap: the first tap replaces
  the keyboard with `clone <repo> from <forge>?`, and only the confirm tap
  clones. Cloning writes to your disk, so it is never a side effect of
  choosing a repo.
- **Clone URLs are allowlisted** — only `https://` and `ssh://` are cloned.
  git and jj both support "transport helper" URLs (`ext::sh -c …`), which make
  a clone *run a command*; a remote catalogue answer therefore cannot become
  command execution. A URL with any other scheme is refused with an error
  naming the scheme.
- **Tool gate** — `terminal`, `write_file`, etc. are refused with a message that
  names the candidate repos until the session resolves. Fresh sessions are
  asked; delegate_task children inherit their parent's binding (see below).
- **1:1 session ↔ workspace** — `ws_new` on a bound session asks for explicit
  `rebind=true` instead of silently dual-binding. Rebinding detaches the old
  tree (the directory stays).
- **Not on disk ≠ does not exist** — `ws_new` for a repo that is not under
  the configured `repo_dir` checks the GitHub/Forgejo catalogue. If the repo
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

**Which parent, per child.** A child's parent is resolved from channels keyed
by the *child's own* session id — the `child_runs` row the host's
`subagent_start` hook writes, and the child's own binding row
(`parent_session_id`). There is no module global any more: a global can name
only ONE parent, so two delegations in flight made the second child inherit the
first child's parent and build its workspace in the wrong repository. A child
whose parent cannot be resolved is **blocked**, never pointed at a stranger's
workspace.

## Child-end reporting

When a delegated child finishes, the **parent** session learns what the child
left behind — on its next turn, as one line per child:

```
child parent-tree-child-abcdef: dirty, 2 unpushed commits - <repo_dir>/.workspaces/repo/parent-tree-child-abcdef (worker completed)
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

**Delivery is versioned.** The report row is keyed by the child session id and
is rewritten in place when the child is re-dispatched, so the delivery names
the exact version it read (a per-row `report_seq`): a report that arrives
between the read and the clear is *not* deleted, and the parent still gets it
on its next turn.

**The tree is reclaimed right there.** On the *same* background thread, right
after the report row is written, the child's workspace is reclaimed: the dirty
working copy is snapshotted into a commit in the shared repo store, and the
workspace and its directory are removed. A finished child's tree is dead
weight on disk, and it is gone the moment the child ends.

The order is the point. The report says whether the tree was dirty **at stop
time**; the reclamation then snapshots that dirty tree and removes it. If the
order were reversed, the report would describe a tree that no longer exists
and would say "clean" about work that is sitting in a snapshot commit.

The snapshot commit id is recorded in the `child_reclaims` table — the same
table `hermes ws reclaim` writes. The report row and the reclaim row are
linked by the workspace slug, so "where did the child's work go?" is one
lookup:

```sql
SELECT r.snapshot_commit_id FROM child_reports r
  JOIN child_reclaims c ON c.slug = r.workspace_slug
 WHERE r.child_session_id = ?;
```

A reclamation that is **refused** (a live session is still pinned to the tree,
or the tree is not a child at all) costs the reclamation, not the report: the
parent still hears what happened, and the tree is left alone.

**The child's binding is marked ended at the stop, and then the tree goes.**
A session with no `session_ended_at` counts as *live*, and the in-use guard
refuses to remove a live session's tree — so `subagent_stop` stamps the child's
binding row as ended (the child is finished by contract at that point) and the
reclamation runs immediately afterwards. That stamp is what unlocks the
immediate reclamation: without it every freshly-stopped child looks live, the
reclaim is refused as in-use, and the tree silently stays on disk.

**The stop flow stamps ended and then reclaims — it does not refuse.** The
order is fixed: the end stamp comes first, the reclamation second. So a child
that stops is reclaimed, and the in-use guard is not what decides it.

**Refusals live in the manual reclaim path.** A refusal — a live session is
still pinned to the tree, the tree is not a child at all, it is the main jj
workspace, or it is a git-only worktree — is what `hermes ws reclaim` (the CLI,
run by a human) reports. That is the path where the in-use guard can actually
fire, because the CLI does not stamp anything ended first. The immediate stop
path never reaches it. To turn the immediate reclamation off entirely, use the
kill-switch below.

**The kill-switch.** Set `child_immediate_reclaim: false` and the stop hook
reports but does **not** reclaim — the tree stays and the age-based sweep (or
`hermes ws reclaim`) takes it later. The switch exists for an operator who
does not trust the snapshot path: the reporting is what tells a human what
happened, and turning the switch off keeps it. Nothing else changes.

```yaml
plugins:
  entries:
    hermes-treeps:
      settings:
        child_immediate_reclaim: false   # report only, keep the tree
```

**One-time delivery.** A report is delivered once and then deleted. The next
turn is silent. With nothing pending, the plugin adds nothing at all — no
empty section, no filler.

**Honest answers.** When a check cannot run (the workspace was retired
mid-run, the remote is unreachable), the line says `state-unknown` or
`unpushed unknown` — never a wrong "clean".

**Hard-death caveat.** If the gateway restarts mid-child, `subagent_stop`
never fires and no report is produced — and, because the reclamation rides on
that same hook, the tree is not reclaimed either. The orphaned child
workspace is still reclaimed by age: `hermes ws gc --children --apply` retires
child workspaces once they are 14 days old, their session ended and their
tree is clean.

## Child liveness

A delegated child has **two independent liveness markers**, and the plugin
uses both so that neither a graceful stop nor a hard death is stranded.

1. **Ended stamp (`session_ended_at`)** — written by the `subagent_stop` hook.
   A genuine stop is authoritative: the child's tree is reclaimed immediately,
   regardless of the heartbeat.
2. **Heartbeat (`child_seen:<session-id>` in the plugin's kv store)** — written
   by the plugin's gate on every gated tool call a child makes, rate-limited to
   one upsert per 60 seconds. A fresh heartbeat (< 24 hours) is proof of life:
   the age-based sweep will never reclaim that child.

**Precedence: stop > heartbeat.** A real `subagent_stop` event means the child
is finished by contract, so the ended stamp wins over any heartbeat value. The
heartbeat only matters when the stop event never arrived.

**Hard-death thresholds (both required).** If the gateway restarts mid-child,
`subagent_stop` never fires and no ended stamp is written. Such a child becomes
reclaimable only when **both** hold:

- its heartbeat **exists and** is older than **24 hours**, **and**
- its workspace is older than **24 hours**.

One stale marker alone is never enough: a brand-new child that has not called a
gated tool has no heartbeat yet, and a child with a fresh heartbeat is alive.
The in-use guard inside `reclaim_child_workspace` waives the ended-stamp
requirement under exactly the same two-threshold rule — requiring the stamp
would strand a hard-dead child forever.

Note the asymmetry: a **missing** heartbeat is not a stale heartbeat. The
hard-death backstop needs *positive evidence* that the child was alive and
then stopped reporting, so it requires a heartbeat that exists and is old. The
looser `child_heartbeat_stale()` helper answers the general liveness question
and falls back to the workspace's own age when no heartbeat was ever written —
but the reclamation's backstop does not use it.

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
the tree is dead weight on disk. It is taken back at once - automatically, the
moment the child stops - and the work is never lost.

    hermes ws reclaim --slug <slug>          # reclaim one finished child
    hermes ws reclaim --all-children         # reclaim every finished child
    hermes ws reclaim --slug <slug> --dry-run  # report only, change nothing

### The automatic path

When a delegated child stops, the plugin reclaims its workspace on the same
background thread that writes the child's end report - **report first, reclaim
second**, so the report still describes the tree as it was at stop time (see
[Child-end reporting](#child-end-reporting)).

The command above is the manual path, and it goes through the *same*
primitive: the guards, the removal and the bookkeeping are one code path, so
they cannot drift apart.

Set `child_immediate_reclaim: false` to turn the automatic path off. The
report still lands; the tree stays on disk and the age-based sweep (or
`hermes ws reclaim`) takes it later. That switch is for an operator who does
not trust the snapshot path - the reporting is what tells a human what
happened, and it survives the switch.

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

The automatic path writes the same row, so the child's work is findable the
same way whether the tree was reclaimed at stop time or by hand. The report
row and the reclaim row are linked by the workspace slug.

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

A refusal in the automatic path costs the reclamation, not the report: the
parent still gets the child's end state, and the tree is left alone.

### The age-based sweep still runs

The 14-day TTL protects **unfinished** children - a child that is still
working, or was never heard from, keeps its tree until the age says it was
abandoned. A child past `subagent_stop` is finished by contract, so the
immediate path is safe to reclaim at once. The age-based sweep
(`hermes ws gc --children --apply`) keeps running for everything the
immediate path did not reclaim: a refused reclamation, a child whose gateway
died mid-run, or every child when `child_immediate_reclaim` is off.

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
        child_immediate_reclaim: true           # reclaim a child's tree the moment it stops
```

### What discovery looks at

The plugin offers the user **one flat list** of repos. Every entry carries a
`tag` that says where it came from:

| tag | meaning | keys |
| --- | --- | --- |
| `local` | a clone under `repo_dir` | `local_path` is set |
| `forge:github` | a repo on GitHub, owned by the login `gh` is authenticated as | `clone_url`, `owner` |
| `forge:forgejo` | a repo on Forgejo, owned by the login the token belongs to | `clone_url`, `owner` |

Three rules decide what lands in the list:

1. **The local scan walks `repo_dir` and nothing else.** A clone outside it
   is not offered, because the plugin creates its workspaces under `repo_dir`
   and cannot bind a repo it cannot reach. The deprecated `repo_roots` is
   still parsed — an old config must not crash the plugin — but discovery
   does not read it.
2. **The forge scan is owner-filtered.** Each forge lists the repos of the
   login the plugin is authenticated as — and nothing else, with no limit.
   The login is **derived** from the credentials the plugin already has
   (`GET /api/v1/user` with the Forgejo token; `gh api user` for GitHub),
   not configured: there is no setting to name a login, so there is no way
   to name the wrong one. A forge whose login cannot be derived (no token,
   no `gh`, a dead API) is **skipped**, quietly — it is not an error the
   user has to read in the middle of a session. The derived login is cached
   for the life of the process.
3. **The list is deduped by remote identity** — forge + owner + repo name,
   parsed from the clone's `origin` URL — never by path. A forge repo that
   already has a clone under `repo_dir` therefore appears **once**, tagged
   `local`: the user knows one repo, and offering it twice would make them
   clone a copy they already have. Two repos with the same name under
   different owners are two different repos, and both stay.

### Deprecated settings

| key | status | replacement |
| --- | --- | --- |
| `workspace_root` | **deprecated** — still parsed, no longer decides where a workspace goes | `repo_dir` (the root is derived: `<repo_dir>/.workspaces`) |
| `repo_roots` | **deprecated** — still parsed, no longer read | `repo_dir` (repo discovery walks `repo_dir` and nothing else) |

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
