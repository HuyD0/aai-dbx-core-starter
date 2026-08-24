# Optional LangGraph recipe

Use this recipe when the application needs durable graph execution,
human-in-the-loop approval, or replay. The default agent stays
framework-neutral.

Install the certified recipe dependency, then the project's optional
`langgraph` dependency group, and copy the recipe into `src/app`:

```bash
python -m pip install -r recipes/langgraph/requirements.lock
python -m pip install -e '.[langgraph]'
LANGGRAPH_STRICT_MSGPACK=true \
  python -m pytest -q recipes/langgraph/test_graph.py
```

Production construction must inject:

- A durable async-compatible `BaseCheckpointSaver`; `InMemorySaver` is
  test-only. `build_graph()` rejects checkpointers without `aget_tuple`,
  `aput`, `aput_writes`, and `alist`.
- A persistent `BaseStore` for memory shared across conversations.
- A `thread_id` for checkpoints. The baseline recipe deliberately uses the
  same opaque value for the MLflow session and application conversation so
  initial and resumed invocations remain joinable.
- An idempotent implementation of `execute_once`.

The production checkpointer/store wiring against Lakebase — including the
OAuth credential lifecycle and user-scoped memory tools — lives in
`../langgraph-lakebase/`.

The graph interrupts before its irreversible action. Resuming a graph can
re-run a node, so the action is protected with a stable idempotency key.
Call `initial_state()` before the first invocation: it strictly validates the
external Pydantic request and converts it to ordinary JSON-compatible data so
durable checkpoints never depend on importing an application model class.
Tests should cover node behavior, approval and rejection, resume, repeated
delivery, timeout/failure handling, expected tool trajectory, and the final
answer.

## The decision is evidence, not just an outcome

The resume payload crosses the same trust boundary as the original request,
so it is a strict `ApprovalDecision` — `approved`, a `reason_code` from the
recipe's small vocabulary (`approved`, `model_error`, `ambiguous_intent`,
`policy_boundary`, `stale_context`), and an optional bounded `note` — never
a bare boolean. An override that records only the correction fixes one case
and loses the signal; the reason decides what the intervention becomes:

- `model_error` → label the trace and promote it into a regression case.
- `ambiguous_intent` → the graph replans: the decision (reason and note) is
  passed to `propose` as feedback and a fresh proposal is reviewed, bounded
  by `max_proposal_attempts` so rejection loops always terminate.
- `policy_boundary` → a guardrail or escalation gap, not a model fix.
- `stale_context` → refresh retrieval/context before trusting a re-run.

A submitted resume value is durable: LangGraph replays it on every later
resume of the thread. The `approve` node therefore never raises on a
malformed payload — it re-interrupts with the validation error so the
reviewer can submit a well-formed decision and the thread stays cleanly
resumable. The rejected path carries `reason_code`, `note`, and `attempts`
into the final result so the trace records why execution did not happen.

## Tracing

Use `await graph.ainvoke(...)` or `graph.astream(...)`; do not call the
synchronous graph APIs from an async server. Enable only
`TraceIntegration.MLFLOW_LANGCHAIN` during process startup, with
`autolog_options={"run_tracer_inline": True}` so concurrent async callback
context does not merge traces. For every initial invocation and every resume,
open a fresh native `mlflow.tracing.context(session_id=thread_id)` and make one
`ainvoke()` or `astream()` call inside it. Reuse the same checkpoint
`thread_id` to link
durable state, but never keep a trace or tracing context open while waiting at
an interrupt. Do not add manual SDK trace decorators around this recipe.

Make the review loop measurable from traces alone:

- Tag each invocation's trace with whether it was an initial delivery or a
  resume (for example `mlflow.update_current_trace(tags={"invocation_kind":
  "resume"})` inside the tracing context). Intervention rate, recovery
  success, and replan/loop incidence then come straight from trace queries.
- After a resume completes, attach the decision to that trace as a native
  MLflow assessment using the generated app's `src/app/feedback.py` helpers
  (`record_intervention`, or `record_human_feedback` with the reason as
  rationale). A rejection whose reason is recorded on the trace can be
  curated with `curate_reviewed_expectation` and promoted into the
  evaluation datasets that `agentkit` runs — the override becomes a
  regression case instead of a lost correction.

LangGraph owns graph state and checkpoint APIs. `aai-core` supplies resource
context, tracing policy, provider clients, evaluation and release evidence; it
does not wrap the graph.
