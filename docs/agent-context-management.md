# Agent context management

How this repository gives coding agents — and the developers working beside
them — a persistent, shared memory. The popular "second brain" pattern for
agent workspaces layers raw capture, a wiki, generated output, and a small
always-loaded identity file, then closes the loop by saving new insights.
This repository keeps those layers but adapts the mechanics to a governed,
multi-tool, enterprise-cloned monorepo: curated single sources with
machine-checked drift instead of a link graph, and an update loop made of
tooling rather than discipline.

## Layer map

| Layer | This repository | Generated projects |
|---|---|---|
| Identity file (always loaded) | `CLAUDE.md` imports `AGENTS.md`, the single operating guide | The same pair, shipped by every template |
| Ground truth (never restated) | `platform-identifiers.json` and `compatibility.json`; a repository-wide test fails any markdown that restates identifier values | Schema defaults stamped by `make sync-templates`; provenance in `.aai-template.json` |
| Wiki (evergreen knowledge) | `docs/`, reachable from the enforced index `docs/README.md` | `README.md` plus `docs/` |
| Permanent notes (the why) | Dated records under `docs/decisions/` | `docs/decisions/`, same convention |
| Update loop | The `aai-log-decision` skill, the AGENTS.md section 8 capture rule, PR review, and drift tests | Template updates propagate through the shared scaffold |
| Retrieval | `grep` plus the index — no knowledge graph, no embedding store | Same |
| Raw capture | Deliberately absent: transcripts, screenshots, and user content do not enter a governed repository | Same |

## Why link-everything was rejected

"Everything references everything" reads well on a poster and decays in a
team repository: every link is a maintenance liability, and every restated
fact is a copy a clone has to find and edit. AGENTS.md section 3 records the
incident that set this rule — an identifier drifted in prose while every
test stayed green. The durable fix was deleting the copies, keeping one
machine-readable source, and adding drift checks. The documentation index
follows the same philosophy: one enforced map (`tests/test_docs_index.py`),
not a web of cross-references.

## What repository memory may contain

Repository memory ships to every clone, so the bar is what AGENTS.md
section 4 already sets: no secrets or credential material (rule 1); no
personal data — teams and group aliases, never individuals (rule 11); no
prompts, user content, or transcripts; no environment identifier values,
which belong to `platform-identifiers.json`. Non-secret does not mean
recordable.

## Two kinds of "decision"

`aai_core.decisions` records runtime lifecycle decisions — adopt, reject,
inconclusive — as governed MLflow evidence for experiments.
`docs/decisions/` records engineering decisions about this codebase. They
share a word, not a mechanism.

## How generated projects inherit the pattern

Every template ships `AGENTS.md` (rendered with the project name),
`CLAUDE.md`, and `docs/decisions/` through the shared scaffold
(`templates/_shared/`), so a new project starts with an identity file, a
decision log, and the same capture conventions. The render matrix and the
scaffold drift test keep the shipped copies identical to the canonical
files here.
