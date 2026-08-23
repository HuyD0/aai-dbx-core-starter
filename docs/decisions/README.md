# Decision log

Dated, immutable records of engineering decisions whose rationale would
otherwise live only in pull-request threads. `CHANGELOG.md` says what
shipped; a decision record says why, and what the alternatives were.

## When to write one

Write a record when a change embodies a non-obvious choice: a reversal of an
earlier approach, a rejected alternative worth remembering, a constraint that
future work must not quietly undo. Routine changes need no record — the diff
and the changelog already explain them.

## Naming

`YYYY-MM-DD-short-slug.md`, for example `2026-08-22-adopt-decision-log.md`.

Dates instead of sequence numbers because downstream enterprise clones merge
this repository's release tags: the first time upstream and a clone both add
"decision 0007", the merge conflicts. Date-plus-slug names cannot collide.

For the same reason this README lists no entries. `ls docs/decisions/` is the
index, so adding a record never edits a shared file.

## Format

Every record carries a `Status:` line and three sections:

    # Title
    Status: adopted

    ## Context
    ## Decision
    ## Consequences

Name the alternatives that lost and why. One decision per file.

## Supersede, don't rewrite

A reversed decision gets a new record. The only edit an existing record ever
receives is its Status line: `Status: superseded by <newer file>`. History
stays greppable end to end.

## What may never be recorded

- Secrets or credential material of any kind (AGENTS.md section 4, rule 1).
- Personal data. Name teams and group aliases, never individuals — the same
  posture as `docs/tagging-standard.md`.
- Prompts, user content, or transcripts.
- Environment identifier values. They are owned by
  `platform-identifiers.json`, and a repository-wide test fails any markdown
  file that restates them. Non-secret does not mean recordable: a copy in
  prose is a copy a clone has to find and edit.

## Checks

`tests/test_docs_index.py` enforces the naming convention, the required
sections, and the rule that no synced file enumerates the entries.
