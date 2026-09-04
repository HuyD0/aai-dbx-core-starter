# Genie is an evaluation target, not a tool recipe
Status: adopted

## Context

Databricks' "Grading Genie" write-up evaluates AI/BI Genie spaces with MLflow
judges and grounds a custom scorer in an indexed code corpus. Two of its ideas
apply here; one of them collides with an existing deferral, and the distinction
is the whole point of this record.

A Genie space is a **managed** agent: configured in the workspace, reached over
the Conversation API, with no prompt to version and no chain to trace into. Its
entire observable surface is one tuple — question, generated SQL, returned rows,
prose. That is why its characteristic failure is a clean, well-formatted, *wrong*
answer: the SQL parses, the warehouse runs it, the paragraph reads like an
analyst wrote it, and the number is not the one that was asked for. Nothing
raises, so nothing but scoring catches it.

`docs/langgraph-production.md` defers a **managed-MCP tool recipe (Genie, Vector
Search)**: the `mcp` package took a breaking 2.0 major, `databricks-mcp` is a
preview with an unbounded floor, and MCP endpoints cannot be exercised in
credential-free CI. That reasoning is about an agent *calling* Genie as a tool
inside a generated application, which puts a preview dependency into every
template's runtime.

Grading a Genie space is the opposite direction: nothing is generated, no
template gains a dependency, the call is the certified `databricks-sdk` already
in the policy, and every code path that decides anything is exercisable offline.
Treating the deferral as covering both would have blocked the cheap half on the
expensive half's reasons.

Three shapes were on the table for the grading side:

- A Genie-specific evaluation harness beside `agentkit`.
- A new `TargetKind`, so the existing comparison, gate, baseline and evidence
  machinery grades a space with no new evaluation machinery at all.
- Project-local scorers in `templates/analytics-app`, where SQL-integrity checks
  already live as generated application code.

## Decision

**Genie is a target kind.** `genie:/<space_id>` resolves to
`TargetKind.GENIE_SPACE` in `aai_core.agentkit.targets`, and `agentkit compare
--agent genie:/<space_id>` grades a space through the same plan, thresholds,
statistics, integrity checks, evidence record and exit codes as any other agent.
Nothing about comparison-first evaluation is re-implemented for it.

Three edges are fixed here so later work does not relitigate them:

- **The answer is a mapping, not a string.** A text-to-SQL answer is not just its
  prose: the statement and the rows are what make the prose checkable. The key
  `response` is already a recognised text field, so every existing prose scorer
  reads a Genie answer unchanged, while `generated_sql` is what the SQL scorers
  grade. The alternative — returning prose and discarding the statement — would
  have made the one property worth checking invisible.
- **Preflight stays offline.** A malformed `genie:/` reference is refused by
  string inspection before a client is constructed, like every other target
  shape. Whether the space exists and whether the caller may query it are
  workspace answers, reported at call time with a remediation naming the grant.
- **A fresh conversation per row.** An evaluation row must not depend on the row
  before it. Multi-turn Genie evaluation is a different scorer contract and is
  out of scope.

**The SQL-integrity checks are registry scorers, not project code.**
`sql_read_only` and `sql_claim_scope` join `aai_core.agentkit.catalog` as CODE
scorers, following the precedent set for `tool_order_policy`: a project selects
them and sets thresholds, it never defines them, so two teams reporting
`sql_read_only/mean` mean the same thing.

- `sql_read_only` — any SQL an answer shows is a single read-only statement.
  An analytics agent pointed at a warehouse it can write to is a governance
  defect visible nowhere else: the prose will not mention it and nothing will
  error. It gates by default (`>=1.0`).
- `sql_claim_scope` — a trend claim in the prose is backed by SQL that filters or
  groups on time. "Revenue grew" over a query with no time predicate asserts a
  comparison the query never made. It **reports** by default, with no threshold:
  the trigger is a heuristic over prose, so a project calibrates on its own
  answers before gating. Omitting the key is how this registry says "measure, do
  not gate, yet".

Both are code, not judges. Whether a statement scopes to time is a fact the SQL
records; asking a model to infer it would add spend, calibration debt and a
second definition of "time filter" for no information gain — the same argument
made for tool ordering.

Neither is auto-selected. Both declare no expectation and no trace need, which
keeps them out of automatic inference: they are meaningless for an agent that
never writes SQL, and a vacuous 1.0 reported for every project is noise, not
coverage. A project opts in through `scorers.add`.

They are also deliberately absent from `CODE_SCORER_FUNCTIONS`, because the
tier-1 `score_all` path hands its scorers extracted answer *text*, which discards
the structured statement they read.

**The evidence-grounded judge is not adopted here.** The write-up's headline
scorer retrieves from an indexed corpus and writes the retrieved snippets into
its rationale, so a reviewer can check the judge's homework. The idea is sound,
and this repository already has the pieces — the `Retriever` protocol with a
`TEXT` mode, and `JudgeBinding` with a UC-registered prompt. It is not adopted
now because a grounding corpus is a provisioned resource, and section 4 rule 8
reserves provisioning for the human-run platform process; adding a scorer whose
evidence source does not exist would register a scorer that cannot run. Revisit
when a corpus retriever is provisioned and can be named as a logical resource.

**This does not reverse the managed-MCP deferral.** No MCP client, no
`databricks-mcp`, no template dependency, and nothing generated. Reversing that
deferral still needs its own superseding record.

## Consequences

- A managed agent nobody here wrote is comparable with the agents that are, on
  one scale, in one experiment, under one gate.
- The `databricks` extra is required to *call* a Genie space; resolution,
  preflight, scorer selection, thresholds and the gate all work without it, so
  credential-free CI covers every path that makes a decision.
- Genie's own benchmark eval runs remain the better tool for "is my space still
  answering my curated questions right", and are unaffected. What this adds is
  what they do not do: dimension-level scores with rationales, gate-able domain
  floors, and comparability across agents.
- `sql_claim_scope` will need calibration evidence before any project gates on
  it. Reporting it first is the intended path, not an oversight.
- Multi-turn Genie conversations, result-set correctness against a gold answer,
  and grading the semantic model itself are all out of scope.
