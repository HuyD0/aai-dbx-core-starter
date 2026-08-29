# Review-pin opted-out scaffold forks instead of composing overlays

Status: adopted

## Context

An `opt_out` entry in `templates/_shared/manifest.json` turns a shared
scaffold file into a per-template fork that the byte-for-byte sync no longer
sees. Twenty-two such forks exist, several at very high similarity to their
canonical source (the analytics-app `pyproject.toml.tmpl` differs by one
dependency line), and nothing reconciled a fork against its canonical: a fix
applied to the canonical copy could silently never reach the forks. This was
not hypothetical — a "Literal by design" comment added to the canonical
`databricks.yml.tmpl` never reached the agent-app, analytics-app, or rag-app
forks, and no check noticed.

Two remedies were considered:

1. **Composition/overlays.** Keep one canonical base per file and let each
   template contribute a declared overlay (extra Makefile variables, extra
   dependency lines) that `sync_template_shared.py` splices in and `--check`
   verifies. This eliminates the forks rather than guarding them, but it
   replaces byte-for-byte copying — trivially verifiable, trivially mergeable
   by a clone — with splice semantics that need anchor conventions in five
   file formats, and every existing fork would have to be decomposed at once
   to land it.
2. **Review pins.** Keep the forks, but record in the manifest the sha256 of
   the canonical file each fork was last reviewed against. A canonical change
   then fails `--check` for every fork it cannot reach until someone reviews
   the change fork-by-fork, ports what applies, and re-pins with
   `make acknowledge-forks`.

## Decision

Adopt review pins (`fork_reviews` in `templates/_shared/manifest.json`,
checked by `fork_review_drift()` in `scripts/sync_template_shared.py` and
`tests/test_shared_scaffold.py`). The pin is of the canonical, not the fork:
forks evolve freely, but a canonical change must be consciously dispositioned
for each fork. An opt-out without a fork on disk (a genuine absence) must not
carry a pin, so absences and forks stay distinguishable in the manifest.

Composition lost for now on risk, not on merit: overlays remove the
duplication itself, and this record does not forbid them. Introducing them
later per-file (starting with the one-line pyproject deltas) supersedes the
pin for that file naturally, because a file with no `opt_out` entry needs no
review pin.

## Consequences

- Editing a canonical file that has forks now fails `make check-templates`
  until the change is reviewed against each fork and acknowledged. That
  friction is the feature: it converts silent non-propagation into an
  explicit per-fork disposition.
- `--acknowledge-forks` is the mechanical tail of a review, not a substitute
  for one; running it without looking at the forks defeats the mechanism and
  cannot be detected by tooling.
- The 200-byte `unmanaged_duplicate_sources()` guard and this check overlap
  only at the byte-identical boundary; both stay.
- A clone that adds its own template inherits the rule automatically, since
  the check derives from `opt_out` rather than an enumerated fork list.
