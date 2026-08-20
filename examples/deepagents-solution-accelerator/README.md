# Deep Agents multi-agent solution accelerator

This is the repository's multi-agent reference: a supervisor agent that plans
with an explicit todo list, delegates bounded work to declarative sub-agents,
interrupts for human approval before its one irreversible side effect, and
turns every delegation into deterministic MLflow trace evidence. It is built
on `deepagents` (a LangGraph-based library), served through the approved
external Model Serving process, and closed into a continuous-evaluation loop
that synthesizes an operational `SKILL.md` under strict guardrails.

Use it when one agent with tools is no longer enough — when roles with
different tools and permissions (here: a SQL analyst and a documentation
researcher) must be composed under one supervisor without losing the
evidence chain. For the single-agent paved road, start from
`templates/agent-app`; for durable single-agent approval loops, use its
LangGraph recipes and [`docs/langgraph-production.md`](../../docs/langgraph-production.md).

## The workflow shape

```text
supervisor (plans with write_todos, owns the conversation)
  -> task delegation           one bounded sub-agent per delegated step
       sql-analyst             execute_sql_query (interrupt before execution)
       docs-researcher         search_documentation (Vector Search)
  -> deterministic span tree   deepagent.supervisor -> delegation.<role>
                                 -> <role>.turn -> TOOL
  -> answer with aggregate usage, thread_id, and hitl.status evidence
```

The graph decides what happens between the agents; each agent decides only
what happens inside its own step. Delegation, retry counts, interrupt status,
and token usage are recorded by application-owned tracing middleware, so the
span hierarchy is deterministic evidence rather than a framework side effect.

## Notebooks

| Notebook | Outcome |
|---|---|
| `notebooks/01_agent_setup_and_definition.ipynb` | Pins the runtime stack, defines strictly bounded leaf tools, composes the supervisor and sub-agents with `create_deep_agent()`, and writes the Models-from-Code module with deterministic tracing middleware. |
| `notebooks/02_deployment_and_trace_logging.ipynb` | Logs the module with the MLflow LangChain flavor, adds a version to the pre-provisioned registered model, renders the executable platform-owner serving handoff, validates the deployed version, and attaches governed user feedback to trace IDs. |
| `notebooks/03_continuous_eval_and_feedback_loop.ipynb` | Harvests deterministic, judge (`subagent_routing_accuracy`, `RelevanceToQuery`), user, and operational signals from traces, then synthesizes and promotes `SKILL.md` guardrails from normalized signal codes only. |

## Prerequisites and boundaries

This accelerator is connected-only: it runs as Databricks workspace notebooks
and is not part of the credential-free numbered curriculum. Everything it
touches is provisioned through the approved external platform process, never
by the notebooks:

- a chat model serving endpoint, a SQL warehouse, and a Vector Search
  documentation index (supplied through notebook widgets — the defaults are
  `REPLACE_*` placeholders, and identifiers are configuration, never model
  arguments);
- the pre-provisioned registered model the accelerator versions;
- the serving endpoint itself. Notebook 02 renders a complete, executable
  handoff for the platform owner and then validates the deployment; it never
  creates or mutates serving infrastructure.

The serving identity needs `CAN USE` on the warehouse, query access to the
index, and read access to the skill path. No notebook stores or asks for a
credential.

## Runtime dependency channel

The notebooks `%pip install` an exact-pinned runtime stack in the workspace.
That is a second dependency channel next to the repository's certified locks,
so it is kept honest the same way the platform console's `requirements.txt`
is: every requirement is `==`-pinned, the packages that appear in
`dependency-policy.toml` (MLflow, LangChain, LangGraph, the Databricks SDK)
pin exactly the certified versions, and
`tests/test_deepagents_accelerator.py` recomputes both claims instead of
trusting this paragraph. `deepagents` and `databricks-langchain` are
accelerator-only runtime pins; they are deliberately not part of `aai-core`'s
dependency closure.

## Durability and guardrails

- The checkpointer is `InMemorySaver` for a single-process demonstration.
  Interrupt/resume is only reliable on the same replica; configure a durable
  checkpointer (the pattern is the agent template's Lakebase recipe) before
  scaling interactive human-in-the-loop beyond one replica.
- Human approval interrupts *before* `execute_sql_query`, and the resume
  reuses the returned `thread_id` as a possession token.
- The `SKILL.md` synthesizer never sees raw prompts, responses, SQL,
  retrieved text, feedback comments, or exception messages — only normalized
  signal codes and counts. Immutable safety rules are rendered
  deterministically, every generated guardrail is validated, the prior
  document is backed up behind an optimistic-concurrency hash check, and
  promotion is an explicit job parameter.
- User ratings are bounded tags; free-text comments are first-class MLflow
  feedback assessments, so user content never enters tag metadata.
