# Bootstrap intervals are opt-in and the default interval method stays normal

Status: adopted

## Context

AgentKit reports a confidence interval around every numeric scorer mean and,
against a recorded baseline, around the paired per-row improvement. Under
`enforce_confidence` those bounds feed the promotion gate through synthetic
`*/statistics/*` rules, so the interval method is not cosmetic: it decides
whether a lower bound clears a threshold.

The original method, a normal approximation to the sampling distribution of
the mean, misbehaves on exactly the scales agents are scored on. Judge
verdicts and pass/fail rates are bounded and pile up near the ceiling: a
supervisor that routes 29 of 30 evaluation cases correctly gets a symmetric
interval whose upper bound is above 100% accuracy, and the same distortion
inflates how far below the mean the lower bound sits. Percentile bootstrap —
resample the recorded rows with replacement, take the mean of each resample,
read the 2.5th and 97.5th percentiles — cannot leave the observed range and
needs no distributional assumption.

Four alternatives lost:

**Making bootstrap the default** was rejected because a project with
`enforce_confidence: true` would see its gate-feeding bounds move on an SDK
upgrade with nothing re-scored and no configuration change. The results
record carries its gate rules with it precisely so a verdict cannot change
after the fact; the interval method deserves the same discipline. A future
default flip is a deliberate, versioned platform decision, not a side effect.

**Comparing runs by whether their intervals overlap** — the folklore A/B
test — was rejected rather than added. Both versions are scored on the same
ordered rows, so their row-level noise is correlated, and two overlapping
intervals can hide a paired difference that is reliably positive. The paired
improvement interval the gate already enforces is strictly the stronger
test; offering overlap alongside it would invite weaker conclusions from the
same evidence.

**A numeric dependency (numpy/scipy, e.g. `scipy.stats.bootstrap` with BCa)**
was rejected because `aai_core.agentkit` imports with base dependencies only,
and because reproducibility here matters more than the marginal accuracy of
BCa over percentile: stdlib `random.Random` seeded with a string hashes all
of its bits, giving each `(seed, purpose, metric)` tuple an independent,
platform-stable stream, so adding one metric never shifts another metric's
draws and a recorded interval can be recomputed from its record.

**Per-scale interval families** (Wilson for pass rates, bootstrap for Likert
scales) were rejected because they would couple the statistics module to the
scorer registry's scale vocabulary. One resampling method covers every scale
a scorer can emit, and covers the paired improvement — which is a difference
of scales, not a scale — with the same machinery.

## Decision

`statistics.method` selects `normal` (default, unchanged) or `bootstrap`;
`bootstrap_resamples` (100–10,000, default 1,000) and `bootstrap_seed`
(default 0) parameterize it. The point estimate is always the observed mean —
the interval qualifies the aggregate MLflow reported, it never replaces it.
The gate surface is method-independent: both methods populate the same
`*/statistics/*` synthetic metrics, so switching methods changes bounds,
never plumbing.

Evidence is self-describing. `StatisticalEvidence` records the method, and a
bootstrap record must carry its resample count and seed (a model validator
refuses one that does not). Each estimate names its algorithm as a versioned
literal: `normal-mean-v1`, `bootstrap-percentile-v1`,
`paired-normal-mean-v1`, `paired-bootstrap-percentile-v1`. Records written
before the option existed deserialize as the normal approximation they were
computed with.

## Consequences

Routing accuracies, safety pass rates, and ceiling-heavy judge means now get
bounds that stay inside the metric's feasible range, which is what an
autonomy threshold (`enforce_confidence` against a lower bound) should be
read against. The cost is bounded and local: resamples × rows mean
computations per metric at scoring time, capped by configuration, and
nothing at gate time — the gate replays recorded numbers.

Re-running with a different `bootstrap_seed` is a supported robustness
check: if a promotion decision flips with the seed, the suite is too small
for the margin being claimed, and the honest fix is more rows, not a luckier
seed.

The rule future work must not quietly undo: a change to how bounds are
computed — resampling scheme, quantile interpolation, seeding layout — is a
new method literal (`*-v2`), never a mutation of an existing one, and the
default method never changes as a side effect of another change. An enforced
gate may only move because the project re-scored or reconfigured, not
because the SDK upgraded.
