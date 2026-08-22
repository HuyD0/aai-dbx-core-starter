# CLAUDE.md

The operating guide for this project is shared with Codex in **AGENTS.md** so
both tools work from one source of truth. It is imported below — read it in
full before making changes.

@AGENTS.md

## Quick reminders (the three that bite hardest)

1. **No secrets in Git.** Authentication is keyless; configuration carries
   secret references, never secret values.
2. **The evaluation gate decides.** `make evaluate` exit codes are a CI
   contract: `0` pass, `2` threshold failed, `1` error. Prompt, tool, and
   index changes are releases and go through it.
3. **Record non-obvious decisions** as dated files in `docs/decisions/` —
   supersede records, never rewrite them.
