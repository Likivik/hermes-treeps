# Security policy

## Reporting a vulnerability

Report privately through GitHub's [security advisory
form](https://github.com/Likivik/hermes-treeps/security/advisories/new) — private
vulnerability reporting is enabled on this repository, so please do not open a
public issue for anything exploitable.

Useful in a report:

- the plugin version (`plugin.yaml`) and the Hermes build you run;
- the tool or hook involved (`ws_new`, `ws_chatting`, the gate, the Telegram picker);
- what an attacker gains — a workspace escape, a credential read, a state file
  write, a message the user did not send.

## Scope

This plugin runs inside the Hermes gateway with the gateway's own permissions and
can read and write `/Git/**`, `$HERMES_HOME/plugins/**` and `$HERMES_HOME/state.db`.
Anything that lets a *gated* session reach a tree it was not bound to, or lets a
Telegram tap bind a workspace the tapper is not authorised for
(`allowed_user_ids`), is in scope.

Out of scope: anything that requires the gateway's own config or secrets to
already be compromised, and known limitations documented in the README.

## What is automated

- Secret scanning with push protection.
- Dependency alerts and Dependabot security updates.
- CodeQL (python, `security-and-quality`) on push, PR and weekly.
- `ruff check` and a plugin import smoke test as required checks on `main`.
