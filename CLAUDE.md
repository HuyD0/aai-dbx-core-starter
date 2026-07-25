# CLAUDE.md

The operating guide for this repo is shared with Codex in **AGENTS.md** so both
tools work from one source of truth. It is imported below — read it in full
before making changes.

@AGENTS.md

## Quick reminders (the three that bite hardest)

1. **No secrets — OIDC only.** Never add a client secret, PAT, or repo/env
   *secret*. The four repo *variables* are non-secret ids.
2. **PRs stay credential-free.** Never add `id-token: write` or `azure/login`
   to a `pull_request` trigger. Keep it out of `ci.yml`.
3. **No `environment:` on deploy/smoke jobs** unless you first add a matching
   federated credential — otherwise the OIDC subject changes and login fails.
