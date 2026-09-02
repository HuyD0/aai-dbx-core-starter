# Tool ordering policy is a deterministic registry scorer
Status: adopted

## Context

MLflow's post on evaluating agent skills makes its headline point with a
trace-based scorer: a refund skill that verified the customer's identity
*after* deciding the refund produced correct answers and a correct set of
tool calls, and only a check on span order caught it. This repository's
trajectory checks are unordered by design — the exact-multiset scorer in the
agent template and the registry's `tool_call_correctness` answer *which*
tools ran, and the lesson-08 fixture says so explicitly. Nothing expressed
"tool A must precede tool B" as a policy, and the shared-registry rule
(AGENTS.md section 5) forbids a project from writing that scorer itself.

Two shapes were on the table:

- A registry scorer that reads an ordering expectation and the trace's TOOL
  spans in start order (code, no judge).
- A judge instruction — "did the agent verify before acting?" — under
  `make_judge`, which is how the post's companion piece frames most
  behaviour checks.

## Decision

`tool_order_policy` is a code scorer in `aai_core.agentkit.catalog`. It reads
`expectations.expected_tool_order`, a list of `[before, after]` tool-name
pairs, and fails a row when any call of `after` starts without a prior call
of `before`, using the span clock (`start_time_unix_nano`, or `start_time`
on v2 envelopes) with list order as the tie-break.

Three edges are fixed here so later work does not relitigate them:

- **A tool that never ran is not out of order.** The scorer passes
  vacuously when the guarded tool is absent, because the missing call is
  already the trajectory scorers' finding; failing it twice would double
  count one defect, and passing a trace that skipped *both* guard and
  guarded tool is the honest reading of an ordering rule.
- **No readable spans means unscorable, not failed.** The row returns the
  empty feedback list, the same skip the delegation scorer uses, so a plan
  reports coverage instead of a fake 1.0 or a fake 0.0.
- **A malformed policy raises.** An expectation that cannot be parsed
  becomes a row error and fails the gate rather than reading as "no
  policy"; an ordering rule that quietly stopped applying is the failure
  the scorer exists to catch.

The post's scorer named `ToolCallRelevance` is not adopted: no such class
exists in the certified MLflow 3.15.1 or on MLflow's main branch.

## Consequences

- Ordering is a versioned platform asset with one meaning across projects;
  a project selects it and sets the threshold (default `>=1.0`), never
  redefines it.
- It costs nothing to run — no judge call, nothing to calibrate — so it
  belongs in `agentkit smoke` as much as in the full suite.
- The judge alternative lost because span order is a fact the trace
  already records; asking a model to infer it would add spend, calibration
  debt, and a second definition of "before" for no information gain.
- Ordering across *sessions* or across delegated subagents is out of scope;
  the pairs are matched within one trace's TOOL spans.
