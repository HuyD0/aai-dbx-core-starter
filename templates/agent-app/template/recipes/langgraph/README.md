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

The graph interrupts before its irreversible action. Resuming a graph can
re-run a node, so the action is protected with a stable idempotency key.
Call `initial_state()` before the first invocation: it strictly validates the
external Pydantic request and converts it to ordinary JSON-compatible data so
durable checkpoints never depend on importing an application model class.
Tests should cover node behavior, approval and rejection, resume, repeated
delivery, timeout/failure handling, expected tool trajectory, and the final
answer.

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

LangGraph owns graph state and checkpoint APIs. `aai-core` supplies resource
context, tracing policy, provider clients, evaluation and release evidence; it
does not wrap the graph.
