# Adopt a dated decision log

Status: adopted

## Context

The repository's rules live in `AGENTS.md` and are enforced by tests, but the
reasoning behind them lived only in `CHANGELOG.md` narrative and pull-request
threads. AGENTS.md section 3 records the cost of prose that restates facts:
the SDK artifact volume once drifted from the fixture while every test stayed
green, and the durable fix was deleting the copies and machine-checking what
remained. Any memory convention added here has to follow that precedent —
curated single sources with drift checks, not a link graph.

Two alternatives were considered. Classic sequentially numbered ADRs
(`0007-...`) collide the first time upstream and a downstream clone both
record a decision in the same window, because clones merge this repository's
release tags. A single append-only `decision-log.md` has the same flaw and
additionally conflicts on every pair of concurrent entries.

## Decision

Engineering decisions are recorded as dated, immutable files under
`docs/decisions/` named `YYYY-MM-DD-short-slug.md`, each carrying a `Status:`
line plus Context, Decision, and Consequences sections. No file enumerates
the entries — the directory listing is the index. Reversals add a new record
and repoint the old record's Status line. `tests/test_docs_index.py` enforces
the convention, and AGENTS.md section 8 states the capture obligation.

## Consequences

- Rationale becomes greppable memory next to the code it explains, and it
  survives contributor and agent turnover.
- Upstream and clones can both add records with no merge conflicts, by
  construction rather than by coordination.
- Reviewers gain a concrete question for substantial pull requests: does
  this change embody a decision worth a record?
- Superseded records keep their content, so the history of a reversal stays
  readable end to end.
