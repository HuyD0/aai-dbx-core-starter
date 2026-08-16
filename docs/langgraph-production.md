# Production LangGraph agents

This guide explains when a generated agent application should reach for the
LangGraph recipes, and how the recipes turn human review into durable,
measurable evidence. It is the production companion to the recipe READMEs in
`templates/agent-app/template/recipes/`.

The boundary stays fixed: LangGraph owns graph state and checkpoint APIs, and
`aai-core` never wraps graphs, checkpointers, stores, tool registries, or
serving. Everything below is application-owned template code that consumes
the SDK's resource context, tracing policy, evaluation, and release evidence.

## When to reach for the recipe

The generated agent's default loop (`src/app/agent.py` behind the MLflow
Agent Server endpoint) is framework-neutral and stateless between requests.
It is the right choice while every request is self-contained and no action
needs human sign-off.

Reach for `recipes/langgraph/` when the application needs any of:

- **Durable execution.** Work that survives a process restart, a redeploy,
  or a wait measured in hours. State lives in a checkpoint, not in memory.
- **Interrupts before side effects.** A human approves the proposed action
  *before* the irreversible step, not after. Resuming can re-run a node, so
  the side effect sits behind an idempotency key.
- **Replay and safe re-entry.** A rejected proposal can be replanned with
  the reviewer's correction, bounded so rejection loops always terminate.
- **Long-term memory.** Context and decisions that persist across
  conversations in a store, scoped per user.

Prototype velocity is not the constraint these recipes optimize for —
production confidence is. The recipe's contract tests (strict boundary
models, duplicate-delivery safety, msgpack-strict checkpoints) are the
evidence that the durable path behaves.

## The review loop is trace evidence

An agent's production artifact is its behavior, and the trace is the record
of the path actually taken. The recipe treats the human review at the
interrupt as part of that record:

- **A decision has a reason.** The resume payload is a strict
  `ApprovalDecision` — approved or not, a `reason_code` from a small
  vocabulary (`approved` for approvals; `model_error`, `ambiguous_intent`,
  `policy_boundary`, `stale_context` for rejections), and an optional
  note. An override that records only the
  correction fixes one case and loses the signal; the reason decides what
  the intervention becomes. A malformed payload re-interrupts with the
  validation error instead of poisoning the durable thread.
- **The reason routes the work.** `ambiguous_intent` replans: the decision
  is fed back to the proposer and a fresh proposal is reviewed, bounded by
  an attempt cap. `model_error` labels the trace for regression promotion.
  `policy_boundary` is a guardrail gap, not a model fix. `stale_context`
  calls for a retrieval refresh.
- **Initial and resumed invocations stay joinable.** Every invocation opens
  a fresh MLflow tracing context with `session_id=thread_id`; tag each trace
  with whether it was an initial delivery or a resume. Intervention rate,
  recovery success, and replan/loop incidence then come from trace queries
  alone — no side channel.
- **The override becomes a regression case.** After a resume completes,
  attach the decision to the trace as a native MLflow assessment with the
  generated app's `src/app/feedback.py` helpers. From there the existing
  curation path applies: `curate_reviewed_expectation` promotes the labeled
  trace into the evaluation datasets that `agentkit` runs, and the next
  release must pass the case that failed review.

The measures worth watching at the workflow level — beyond per-request
quality — are task completion, recovery success after an interruption,
intervention rate, replan/loop counts against their caps, and cost per
completed workflow.

## Durable persistence on Lakebase

`recipes/langgraph-lakebase/` is the production wiring for the recipe's
checkpointer/store requirement. Lakebase is managed PostgreSQL, so the
implementation is the native LangGraph Postgres saver and store
(`langgraph-checkpoint-postgres`); the recipe adds only what Databricks
requires and production demands:

- **Credential lifecycle.** OAuth database credentials are minted through
  the workspace client, cached, refreshed ahead of expiry, and fail closed.
  Every *new* pooled connection mints a live token inside the connect path,
  because a pooled connection can be created long after the token that
  opened the pool expired. The token never appears in a DSN string,
  environment variable, log field, exception, or `repr`.
- **A validated runtime contract.** Connection coordinates arrive as the
  Databricks Apps `postgres` resource binding's environment variables and
  are validated strictly — including refusing plaintext connections to any
  non-loopback host. The binding's `value_from: "database"` yields a
  hostname; credential minting needs the endpoint *resource path*, supplied
  separately as `LAKEBASE_ENDPOINT`. The application's schema is part of the
  same contract: `LAKEBASE_SCHEMA` names the schema the app role owns, and
  every pooled connection pins `search_path` to it — the LangGraph saver and
  store issue only unqualified statements, so the search path is what keeps
  durable state out of a shared `public` schema.
- **The provisioning boundary holds.** The Lakebase instance, database,
  role, and grants come from the approved external platform process. The
  recipe connects; it never creates. The savers' one-time DDL is an explicit
  application startup decision (`run_setup=True`), not an implicit
  side effect — it first ensures the configured schema exists and is owned
  by the connected role (creating it when absent, which is what the
  binding's `CAN_CONNECT_AND_CREATE` permission covers) and fails closed
  when another principal owns it.
- **Memory carries lineage.** The user-scoped memory tools store
  `preference` memories for durable context and `decision` memories that
  must carry their `reason_code` and originating `request_id` — so a later
  session can retrieve why something was rejected, and a review can trace
  which signal changed which behavior. Handlers are defensive: not-found is
  a structured result and deletion is idempotent, so degraded memory never
  crashes the loop.

User identity for memory scoping is resolved by the serving layer — for a
Databricks App, from the forwarded identity of the authenticated user. The
platform console's rejection of on-behalf-of authorization is a
console-specific policy and does not constrain agent applications. User
identifiers never belong in resource tags, trace inputs, or tool output
metadata.

The recipe's integration test tier proves the durability claims against a
real PostgreSQL server: interrupt → resume across freshly constructed saver
instances, duplicate-delivery idempotency, and decision-reason survival
through a checkpoint round trip — under msgpack-strict serialization, which
is what keeps checkpoints portable data instead of permissively deserialized
application classes.

## Tool integration and MCP

Tools remain application-owned. The generated template's tool registry (or
its LangGraph/LangChain replacement) validates inputs strictly, bounds
execution, and emits standard `TOOL` spans; Unity Catalog functions,
vector search indexes, and Genie spaces are externally provisioned resources
declared in the application manifest (`mcp_services` records MCP-served
governance there).

A managed-MCP tool recipe (Genie, Vector Search) is deliberately deferred:
the `mcp` package took a breaking 2.0 major, `databricks-mcp` is a preview
release with an unbounded `mcp` floor, and MCP endpoints cannot be exercised
in credential-free CI, so certification now would buy churn without test
evidence. `examples/foundry-curriculum` teaches the integration pattern;
revisit certification when the client stack settles.
