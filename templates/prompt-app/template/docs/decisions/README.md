# Decision log

Dated, immutable records of engineering decisions made in this project. The
changelog and pull-request history say what shipped; a decision record says
why, and what the alternatives were.

## When to write one

Write a record when a change embodies a non-obvious choice: a reversal of an
earlier approach, a rejected alternative worth remembering, a constraint
that future work must not quietly undo. Routine changes need no record — the
diff already explains them.

## Naming

`YYYY-MM-DD-short-slug.md`. Dates instead of sequence numbers so parallel
branches and template updates never collide on an entry name, and no shared
index file needs editing — `ls docs/decisions/` is the index.

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

Secrets or credential material, personal data (name teams and group aliases,
never individuals), prompts or user content, and environment identifier
values — those live in configuration, not prose.
