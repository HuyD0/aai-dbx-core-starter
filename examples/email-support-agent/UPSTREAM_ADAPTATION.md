# Upstream review and adaptation record

Review target:

- Repository: <https://github.com/mahesh15698/AI_LangGraph>
- Reviewed revision:
  [`02146b6dc1c725334cb1e93ae5a54c86e9764a0d`](https://github.com/mahesh15698/AI_LangGraph/tree/02146b6dc1c725334cb1e93ae5a54c86e9764a0d)
- Additional input: the user-supplied “LangGraph Email Support Agent
  Workflow” picture.
- Review date: 2026-08-10.

This directory is an original clean-room AAI solution accelerator. No source,
test, prompt, data, or raster asset was copied from the upstream repository.
The common scenario and useful high-level sequence—classify, retrieve or
track, draft, review, and send—were treated as requirements and redesigned
against this repository's contracts.

## Grade of the reviewed implementation

| Dimension | Grade | Review finding |
|---|---:|---|
| Learning/demo clarity | 6/10 | The graph is small and easy to follow, and the picture communicates intent well. |
| Production architecture | 2/10 | The implementation is a local prototype with mocked search, ticketing, and sending. |
| State and reliability | 3/10 | It uses process-local `MemorySaver`; there is no durable restore, outbox, delivery receipt, or real idempotency. |
| Safety and privacy | 2/10 | Raw email moves through state, auto-send policy is weak, and prompt-injection/PII/tenant controls are absent. |
| MLflow observability | 0/10 | No trace/session design, span taxonomy, token/cost evidence, release lineage, or feedback attachment. |
| Evaluation and improvement | 0/10 | No dataset, scorer, regression baseline, gate, canary, or outcome-driven learning loop. |
| Cost control | 2/10 | A small model is selected, but there are no call/token/context/judge/human-cost budgets or quality-constrained optimization. |

Overall: a useful **learning skeleton**, not a production agent. The image is
closer to a target workflow than the repository implementation, but it also
conflates operational persistence with shared graph state and treats “send” as
an immediate terminal action instead of an outbox plus delivery outcome.

## Material gaps found

- Documentation search and bug tracking return static mock values.
- “Send reply” prints output; it has no provider result or failure state.
- The typed state and node writes drift (`messages` and `current_step`).
- The diagram advertises `execution_metadata`, raw/prompt separation, and a
  broad error strategy that the code does not implement.
- Billing is not forced onto the diagram's intended critical-review path.
- The review decision, edited draft, rejection reason, reviewer provenance,
  and downstream outcome are discarded instead of becoming feedback.
- A LangGraph `thread_id` exists, but no MLflow session or per-resume trace
  joins the lifecycle.
- There is no scorer-visible `RETRIEVER` span or document identity/version.
- There are no test assertions, side-effect fakes, duplicate tests, safety
  cases, or release criteria.

## How the accelerator responds

| Reviewed gap | Reference-design response |
|---|---|
| Raw email in graph | `RedactedEmail` admits only scanned, DLP-processed text plus opaque references; rejected model/reviewer free text is quarantined before durable state. |
| Process-local memory | Native recipe injects durable async checkpointer and persistent store; memory implementations are test-only. |
| Model-owned routing | `route_email()` is deterministic policy over a strict classification; security, critical, complex, billing, and account paths cannot auto-send. |
| Fake retrieval | Trusted access resolver, group/tenant/active/exact-release prefilter plus post-validation, bounded final context, abstention, and MLflow document shape. |
| Immediate writes | `prepare()` has zero side effects; `commit()` writes stable keys to a transactional outbox after policy/review. |
| Weak review state | Resume carries no identity claim; a trusted authorizer resolves group/action rights before approve/edit/reject, with exact proposal/release binding, deterministic re-derivation, post-edit policy, and preserved evidence. |
| No observability | Two traces per interrupted lifecycle under one opaque session plus explicit AGENT/PARSER/GUARDRAIL/CHAIN/ROUTER/RETRIEVER/TOOL/HUMAN_REVIEW/MEMORY semantics. |
| No evaluation | Synthetic nested MLflow rows, deterministic domain gate, shared AgentKit scorers, regression budgets, and a high-risk zero-false-auto-send rule. |
| No learning loop | Strict de-identified review/delivery/resolution/reopen trace signals, governed feedback, hard-case curation, judge calibration, holdout comparison, shadow/canary, and human promotion. |
| No cost model | Pre-call call/context reservations, provider output limits, top-k bounds, judge-call ceiling, risk-based sampling, reviewer-cost accounting, and cost per safely resolved case. |

The supplied image is not checked into this repository because it is a raster
artifact with unclear provenance and cannot remain synchronized with code. The
editable Mermaid diagrams in `REFERENCE_DESIGN.md` are the maintained design
record.
