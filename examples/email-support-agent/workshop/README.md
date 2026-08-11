# Email support agent workshop

This four-level curriculum teaches the reference design by executing its real
contracts, workflow, offline adapters, evaluation gate, and lifecycle evidence.
It does not fork the application into a toy implementation.

The lessons are credential-free: they make no network calls, create no remote
MLflow traces, and perform no production side effects. Every substitute is
labelled `TEST-ONLY` or `NO REMOTE` in the result. Synthetic output is useful for
learning and CI, but it is not production authorization, cost, quality, or
release evidence.

From the repository root, run the levels in order:

```bash
python examples/email-support-agent/workshop/01_graph_basics.py
python examples/email-support-agent/workshop/02_reliability_hitl_idempotency.py
python examples/email-support-agent/workshop/03_mlflow_trace_evaluation.py
python examples/email-support-agent/workshop/04_improvement_release_decision.py
```

Each command prints a short guide and ends with one stable
`WORKSHOP_RESULT=<json>` line. That line uses a strict immutable contract so a
learner, test, or CI job can verify the observations rather than relying on a
screenful of prose.

## Level 1 — graph basics

Goal: follow admission, classification, routing, retrieval, drafting, policy
gates, and action planning without executing an action.

Expected observations:

- `PreparedCase` survives a strict JSON checkpoint round trip.
- The FAQ takes the knowledge route and the safe default selects human review.
- Preparation leaves the outbox empty.
- Extra untrusted email fields fail strict admission.

Failure exercise: predict whether an unknown `raw_mime` field will be ignored.
The lesson attempts that invalid construction in isolation and reports the
expected rejection; it never places raw MIME in durable state.

## Level 2 — reliability, HITL, and idempotency

Goal: see the interrupt-before-write rule, trusted reviewer authorization, and
transactional-outbox contract at the only side-effect boundary.

Expected observations:

- Committing before the required approval is blocked with zero writes.
- A trusted approval binds the case, proposal digest, and application release.
- Replaying the approved proposal returns duplicate receipts and creates no new
  actions.
- A schema-valid but unknown authorization reference is rejected by the trusted
  authorization port.

Failure exercise: predict whether passing strict `ReviewDecision` validation is
enough to prove identity. The lesson demonstrates that schema validation is not
authentication and verifies that the forged path writes nothing.

## Level 3 — MLflow trace and evaluation contracts

Goal: inspect the MLflow GenAI row shape, retriever-document fields, safe trace
capture policy, domain metrics, and deterministic release gate.

Expected observations:

- Rows keep `inputs` and `expectations` nested for MLflow GenAI evaluation.
- Retrieval evidence exposes `page_content`, `doc_uri`, and `chunk_id` in the
  scorer-compatible shape.
- The checked-in synthetic release set passes its deterministic gate.
- Metadata-only trace capture returns payload structure without email text.
- Mutating false-auto-send to a non-zero value fails the hard safety rule.

Failure exercise: predict the gate decision after setting
`safety/false_auto_send_rate` to `1.0`. This is a local counterexample; the
lesson does not claim a remote trace ID, logged MLflow run, or judge result.

## Level 4 — improvement and release decision

Goal: turn linked review/outcome signals and evaluation evidence into an
explicit `baseline -> change -> result -> decision -> release` record.

Expected observations:

- Review edits and delivery outcomes use strict, de-identified signal contracts.
- The application release digest binds code, model, prompt, retrieval, and
  evaluation evidence.
- A passing offline fixture still yields `inconclusive` for production because
  it lacks a connected baseline, production outcomes, and provider cost evidence.
- `DecisionRecord` refuses to adopt a change whose safety gate failed.

Failure exercise: predict whether a caller can construct an `adopt` decision
with a failing gate. The isolated invalid construction must fail, and no prompt
alias, model, index, deployment, or release is mutated.

## Fake boundary legend

- `OfflineClassifier`, `OfflineDrafter`, and `OfflineKnowledgeRetriever` are
  deterministic `TEST-ONLY` substitutes. They make zero provider or judge calls.
- `OfflineAccessAuthorizer` and `OfflineReviewAuthorizer` are `TEST-ONLY`
  resolvers for synthetic references. Production must verify trusted identity
  and authorization evidence outside untrusted payloads.
- `InMemoryTransactionalOutbox` is a `TEST-ONLY` contract example, not durable
  storage and not a production transaction boundary.
- `NO REMOTE` means the lesson demonstrates an MLflow or release contract but
  deliberately does not assert that connected evidence exists.

Continue with the [reference design](../REFERENCE_DESIGN.md) and the
[accelerator guide](../README.md) before replacing these labelled fakes with
provider adapters. Use the production [agent app template](../../../templates/agent-app/)
for deployment and platform lifecycle wiring.
