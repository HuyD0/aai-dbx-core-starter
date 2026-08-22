---
name: aai-log-decision
description: Capture an engineering decision as a dated, immutable record under docs/decisions/, including context, the decision, consequences, and any superseded record. Use when a change embodies a non-obvious choice, a reversal, or a rejected alternative worth remembering.
---

# Log an AAI Decision

Records are engineering memory, not status updates. `CHANGELOG.md` says what
shipped; a decision record says why, and what lost.

## When to write one

- A reversal of an earlier approach, or a rejected alternative that will be
  proposed again if the reasoning is lost.
- A constraint future work must not quietly undo.
- Not for routine changes: the diff and the changelog already explain them.
- Never for secret material, personal data, prompts, user content, or
  environment identifier values — every markdown file is scanned, and
  identifiers belong to `platform-identifiers.json` only.

## Workflow

1. Read `docs/decisions/README.md` for the current convention, then list the
   existing entries (`ls docs/decisions/`) and identify whether this decision
   supersedes one of them.
2. Gather the three sections from the change and its discussion: Context
   (what forced a choice, which alternatives were considered), Decision
   (what was chosen, stated so a reader can apply it), Consequences (what
   becomes easier, harder, or forbidden).
3. Write `docs/decisions/YYYY-MM-DD-short-slug.md` with a `Status: adopted`
   line and the `## Context`, `## Decision`, `## Consequences` sections.
   Name the losing alternatives and why they lost.
4. If superseding: edit only the old record's Status line to
   `Status: superseded by <new file>`. Never rewrite its content.
5. Verify before handing off:

   ```text
   uv run pytest -q tests/test_docs_index.py
   make check
   ```

## Handoff

Report the record path, what it supersedes (if anything), and the checks you
ran. If the decision also changed operating rules, point at the AGENTS.md
section it affects.
