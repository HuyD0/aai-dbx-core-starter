# Derive the credentialed-workflow guards instead of enumerating them

Status: adopted

## Context

Two of this repository's hardest security rules — pull requests stay
credential-free, and a credentialed job carries no GitHub `environment:` — were
enforced against hardcoded file lists. The environment check iterated a literal
three-name tuple. The credential-free check opened one file by name.

Both lists had already fallen behind. `codeql.yml` is `pull_request`-triggered and
appeared in neither, so nothing verified that it stayed credential-free, and it
holds `security-events: write`. More seriously, any workflow added later would
inherit `id-token: write` scrutiny from no test at all: it would pass the
SHA-pinning check, which globs, and escape every other rule, which did not.

This was known. AGENTS.md section 7 cited the escape as a reason not to add a
workflow file — treating a hole in the enforcement as a constraint on the
repository rather than as a defect in the test. That is the wrong direction of fit,
and it makes the rule weaker exactly where it matters: the failure mode is a future
change that looks reviewed because CI was green.

The alternative considered was keeping the lists and adding a test that they are
complete — asserting the tuple equals the set of workflows granting `id-token`.
It was rejected as strictly worse than deriving the set directly: the same scan is
needed either way, and keeping the literal preserves a second thing to forget.

## Decision

Both guards scan `.github/workflows/*.yml` and derive their subject sets.

Credentialed means the workflow grants `id-token: write` at workflow or job level.
Pull-request-triggered means the parsed triggers contain `pull_request`. Each test
asserts its derived set still contains the workflows known to belong to it, so a
regression in the detection itself fails loudly rather than quietly checking
nothing.

Two details are load-bearing. PyYAML resolves the `on:` key to the boolean `True`,
so trigger reading consults both spellings; a naive `get("on")` returns nothing and
the guard silently covers no workflow. And the credential-free assertion checks
properties — no `id-token`, no `azure/login`, no `${{ secrets.` — rather than
comparing the whole permissions block, because `codeql.yml` legitimately needs a
scope `ci.yml` does not.

The same reasoning drove `bundle_identifier_drift()` into the test suite and
`sync_template_shared.py --check` into `cloud-verify.sh`: a check that exists but
runs nowhere CI reaches is indistinguishable from an absent one.

## Consequences

Adding a workflow no longer requires remembering to update a test, and the reason
section 7 gave for not adding one is gone; that paragraph should be read as
historical.

A workflow that legitimately needs a GitHub `environment:` will now fail the guard.
That is intended — the branch-ref federated credential cannot mint an
environment-subject token, so such a workflow must not merge before the identity
owner provisions the matching credential. The test is the reminder.

The general rule: enforce over a glob, not a list. Where a list is unavoidable,
assert it against the glob so the list cannot rot silently. A guard whose subject
set is written by hand protects only the past.
