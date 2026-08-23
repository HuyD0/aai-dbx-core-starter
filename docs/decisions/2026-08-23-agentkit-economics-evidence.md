# Record run economics as cost per successful completion, not mean cost per call

Status: adopted

## Context

The platform docs already stated the doctrine — "unknown cost is not zero
cost", cost-quality as a decision aid, no vendor price tables in `aai-core` —
but the agent-evaluation toolkit measured none of it. `ResultsRecord`
persisted no token, cost, duration, or success evidence; every aggregate was
a mean; no percentile computation existed in the SDK; and the gate engine's
`cost/coverage` hook was never fed on the agentkit path. Meanwhile the
tracing layer already preserved token and cost span attributes through every
capture mode, and pinned MLflow exposes aggregated per-trace usage and cost.

The operational failure this closes: mean cost per call prices attempts
while spend is per outcome. A cheap model that fails and retries — each
retry re-sending grown context — can cost more per delivered answer than
the larger model it undercuts on paper, and the mean hides it. The tail
(p95) and the per-outcome ratio expose it.

Alternatives considered and rejected:

- **Gating on mean cost per call.** The trap metric itself; deliberately
  not even emitted.
- **New catalog scorers for cost/tokens.** Cost per success and p95 are
  run-level aggregates and ratios, not row means, so they cannot ride the
  `<scorer>/mean` machinery; the statistics/integrity synthetic-evidence
  pattern fits and leaves the shared registry untouched.
- **Feeding `GatePolicy.minimum_cost_coverage` from `build_policy`.** A
  replayed gate reconstructs its policy from the record's `policy_rules`
  alone, so a constraint living outside the rules would be enforced at
  scoring time and silently vanish at `agentkit gate` time. An ordinary
  `thresholds` rule on `cost/coverage` persists and drift-checks instead.
- **Counting judge/scorer failures against success.** They measure the
  instrument, already fail the gate through `<scorer>/error_count`, and
  would move the routing signal whenever a judge endpoint flapped. Success
  is execution success; quality stays the gate's own concern.
- **A runtime model router in the SDK.** Routing by intent stays an
  experiment decision implemented through logical-name provider
  configuration; the toolkit supplies the per-stratum evidence, not the
  mechanism.

## Decision

A fourth evidence module, `aai_core.agentkit.economics`, harvests per-row
token, cost, duration, LLM-call, and completion readings from the traces a
live or traces run produced (trace-recorded cost first, then a
project-configured input/output price pair over trace token usage, else
unknown — never a shipped price table), and emits coverage-first synthetic
metrics: `cost/coverage`, `tokens/coverage`, `economics/success_rate`,
p50/p95 tails for cost, tokens, and latency, and per-success ratios that
appear only at complete coverage with at least one success. The frozen
`EconomicsEvidence` — including per-stratum segments driven by the existing
`strata` configuration — persists on `ResultsRecord` as an optional field,
exactly like the statistics and integrity evidence. Everything is
report-only by default; enforcement is opt-in through the existing
`thresholds`/`regression_budget` grammar, with the gate resolving economics
metric directions before the registry's higher-is-better fallback.
Answer-sheet replay records no economics: the sheet holds no agent trace.

## Consequences

- Comparisons and evidence packs now answer "what did each delivered
  outcome cost, and where is the retry tail" — the input for routing an
  intent to a different model — without a new enforcement mechanism.
- A record written by this version is not readable by pre-economics
  readers (`extra="forbid"`), the same forward-incompatibility the
  integrity evidence introduced at the toolkit's preview support tier;
  older records stay readable.
- Adopting an economics threshold is policy drift for records scored
  before it, so `agentkit gate` demands a re-run — the same behaviour as
  enabling the `integrity:` block.
- A `regression_budget` entry on an economics metric refuses to gate
  against a pre-economics baseline until the baseline is re-established,
  because the baseline carries no such metric — fail-closed, as intended.
- The `evaluation-project` template documents the block and moves to
  2.3.0.
- Known follow-up, deliberately out of scope here: the runner never
  synthesizes `predict_fn/error_count` for the gate engine (only
  `<scorer>/error_count` columns are read), so economics reads the bare
  `error_message` column itself; the missing synthesis remains open.
