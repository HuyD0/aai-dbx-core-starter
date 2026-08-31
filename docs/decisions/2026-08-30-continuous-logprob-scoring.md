# Continuous logprob-weighted scoring runs beside the discrete judges, report-only and outside the registry

Status: adopted

## Context

Discrete judge verdicts collapse into the one label the judge emits, so
answers of different quality tie whenever they round to the same label —
which makes the discrete scorers weak instruments for ranking. Following
the LLM-as-a-verifier framework, a continuous path was added that prompts
for a single-token score label, takes `exp()` of the top logprobs at the
score position, filters to valid score tokens, renormalizes by the
retained mass, and returns the probability-weighted average — with
criteria decomposition, K repeated judgments, and positional alternation
for pairwise comparisons.

Two constraints shaped where it landed. MLflow's built-in judges never
expose logprobs, so the verifier must make its own OpenAI-compatible chat
calls; and the Anthropic API exposes no top logprobs at all, so the
verifier model must be an Azure OpenAI deployment or a Databricks-served
model that supports `top_logprobs`. Whether coarse (5-point) or fine
(20-point) scales rank better on this platform's data is an open
empirical question the sweep script exists to answer.

## Decision

- The path lives in `aai_core.agentkit.continuous`, is off by default,
  and when enabled (`scorers.continuous` in `agentkit.yaml`) runs
  **beside** the discrete judges. The discrete path keeps gating.
- **Score labels are single uppercase letters, not digits**: one letter is
  one token in every common tokenizer, so the first-position distribution
  is the score distribution. Multi-digit labels split across tokens.
- **Granularity and repeats are configuration**, recorded on every run as
  MLflow params, and `scripts/sweep_continuous_scoring.py` measures
  ranking agreement (Kendall tau-b against deterministically graded
  candidates) and tie rate per (granularity, K) combination.
- **Logprob support is probed at runtime.** A backend that refuses the
  parameters or silently drops them degrades the run to the discrete path
  with a warning and the `aai.continuous_scoring: fallback-discrete` tag;
  auth/permission/rate-limit/server failures propagate.
- **Report-only and outside the shared scorer registry** until validated:
  a registry entry whose meaning varies with per-project granularity/K
  would break "0.8 means the same thing everywhere". Registration, a
  fixed platform configuration, and baseline comparability rules come
  after the sweep settles the parameters.
- The normalization mass (retained probability before renormalizing) is
  recorded per run with a low-mass flag threshold (default 0.5), because
  a low mass means the prompt is not steering the model to the scale and
  the weighted average summarizes noise.
- Verifier calls are budgeted spend: counted in the pre-run message and
  enforced under `budget.max_judge_calls` with the integrity re-scoring
  calls.

## Consequences

- Every enabled run pays `rows × 3 criteria × K` extra judge calls plus a
  probe; the estimate says so before confirmation.
- Continuous metrics (`correctness_continuous/*`, `continuous/*`) appear
  in run metrics and results records but back no thresholds; a crashing
  continuous scorer still fails the gate through its `error_count`, like
  any scorer.
- Rejected: digit score labels (multi-token); registering the scorer in
  the catalog now (meaning would vary per project); gating on the
  continuous score before validation; treating a logprob-free backend as
  an error (it would make enabling the flag unsafe across environments
  where the judge endpoint differs).
- The pivot-tournament best-of-N selection from the same framework is
  deferred until the scorer is validated.
