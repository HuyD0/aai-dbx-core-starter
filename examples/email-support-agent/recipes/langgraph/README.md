# Native LangGraph adapter

This adapter is deliberately optional so LangGraph does not become an
`aai-core` runtime dependency. It uses the dependency versions certified by
the generated agent template:

```bash
make email-support-langgraph-check

# Equivalent explicit command:
PYTHONPATH=examples/email-support-agent/src \
  uv run --python 3.12 --isolated --no-project \
  --with-requirements \
  templates/agent-app/template/recipes/langgraph/requirements.lock \
  --with pytest==9.1.1 python -m pytest -q \
  examples/email-support-agent/recipes/langgraph/test_graph.py
```

When running from this repository, add the accelerator package to
`PYTHONPATH`. In a generated project, copy the domain package and this adapter
under `src/app/`, retain the generated project's exact lock, and adapt imports.

Production construction must inject an async durable
`BaseCheckpointSaver`, a persistent `BaseStore`, and an outbox backed by a
transactional database. The workflow must also receive identity-backed access
and review authorizer adapters; a Pydantic claim from the resume payload is not
authorization. `InMemorySaver`, `InMemoryStore`, and
`InMemoryTransactionalOutbox` are test-only.

The initial and resumed calls use the same opaque LangGraph `thread_id`, but
each call gets a fresh trace:

```python
with mlflow.tracing.context(session_id=thread_id):
    pending = await graph.ainvoke(initial_state(email), config)

# No trace remains open while a reviewer works.

with mlflow.tracing.context(session_id=thread_id):
    completed = await graph.ainvoke(Command(resume=decision), config)
```

After installing the approved native MLflow PII masking processor, configure
one tracing owner at process startup:

```python
from aai_core.tracing import (
    TraceCaptureMode,
    TraceIntegration,
    TracePolicy,
)

context.configure_tracing(
    integration=TraceIntegration.MLFLOW_LANGCHAIN,
    policy=TracePolicy(capture_mode=TraceCaptureMode.FULL),
    autolog_options={"run_tracer_inline": True},
)
```

Native framework capture requires full trace capture, so do that only after
the ingress DLP boundary and processor are in place. If full customer-content
capture is not approved, disable framework autologging and use bounded
SDK-owned semantic spans instead. Never use a sender address as `thread_id` or
searchable trace metadata.

The graph interrupts before all review-controlled writes. The commit node
does not call SMTP or a ticket API: it writes stable idempotency keys to a
transactional outbox. Separate workers perform delivery and persist provider
receipts, retries, and terminal outcomes. The review UI must return the exact
`proposal_digest` and `application_release` from the interrupt payload plus an
opaque review-service authorization reference. It must never return a claimed
reviewer group. The injected authorizer resolves and verifies group/action
rights; commit re-resolves access and re-derives deterministic policy/actions
before accepting the decision. Invalid review text and identity claims are
rejected inside the resumed node before they can become graph state.
