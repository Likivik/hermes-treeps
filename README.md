# hermes-workspaces

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
| `__init__.py` | tools (`ws_new`, `ws_chatting`, `ws_rename`, `ws_status`), the tool gate, `hermes ws` CLI, system-prompt section |
| `telegram_picker.py` | Telegram inline keyboard: repo picker on `/new`, tap handling, `/wsbtn` |
| `store.py` | workspaces, session records, tokens, remote repo listing (GitHub + Forgejo), retire ladder, gc tiers |

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

## Config

```yaml
plugins:
  entries:
    hermes-workspaces:
      allowed_user_ids: [<telegram ids that may tap the picker>]
      inject_first_message: false   # open gateway sessions with the question itself (runs an extra turn)
```

Environment: `FORGEJO_URL` + `FORGEJO_TOKEN` enable the Forgejo source; `gh`
must be authenticated for GitHub. With neither reachable, the picker falls back
to a stale cache, then to local repos under `REPO_ROOTS`.

## Requirements

- Hermes Agent with the plugin hook set used here (`on_session_reset`,
  `on_session_finalize`, `pre_tool_call`, `on_session_start`),
  `register_telegram_handler` for the picker.
- `jj` (workspaces are jj-first; git worktrees work for non-jj repos).
- Developed against Hermes 0.21.x on NixOS; no third-party pip deps (stdlib
  only — profiles with read-only venvs can load it).

## Install

```
hermes plugins install Likivik/hermes-1s1ws-warden
```

## License

MIT
