# Email Support Agent solution accelerator

This is a credential-free-first **solution accelerator and reference design**
for a stateful email support agent. It turns the useful routing and
human-review idea in the reviewed LangGraph demo into a production-minded AAI
pattern: strict boundaries, redacted durable state, grounded drafting,
deterministic policy, human approval, idempotent outbox writes, MLflow spans,
release evaluation, feedback, and explicit cost budgets.

The checked-in implementation meets the accelerator's documented **10/10
learning** and **10/10 production-reference coverage** definitions. That does
not mean synthetic tests certify a live system: a deployment stays fail-closed
and `UNVERIFIED` until its real adapters and release-bound operational evidence
pass an owner-approved gate. See
[`MATURITY_SCORECARD.md`](MATURITY_SCORECARD.md) for the exact definition and
remaining external proof.

It is an example, not a deployable product and not a seventh platform
template. Production teams graduate it into the maintained
[`agent-app`](../../templates/agent-app/) template, which already supplies the
keyless CI/CD chain, exact dependency locks, nine cost tags, MLflow Agent
Server on Databricks Apps, evaluation job, and prompt-release lifecycle.

## What the accelerator proves

```text
signed webhook bytes
  -> bounded, replay-safe IngressCoordinator
  -> MIME parse + malware scan + DLP + trusted identity mapping
  -> secure raw object + RedactedEmail + opaque access reference
  -> trusted access resolver + exact KB release and group filters
  -> deterministic security preflight before model calls
  -> grounded draft or abstention
  -> durable-state admission + policy gate
  -> interrupt for every review-controlled case
  -> trusted review-service authorization
  -> atomic transactional outbox batch with stable idempotency keys
  -> leased, retry-bounded delivery/ticket worker and provider receipts
  -> MLflow feedback, comparison, gate, and release decision
```

The default `PolicyConfig` disables automatic sending. The evaluation fixture
also exercises the prospective canary policy that permits only low-urgency,
low-risk, high-confidence, grounded knowledge replies. `prepare()` never
writes anything in either mode; `commit()` is the only outbox boundary.
Known prompt-injection/security signals and stable controlled-route replies use
zero model calls. Provider calls reserve call/context budget before invocation,
and rejected model or reviewer free text never enters a checkpoint.

All cases, messages, documents, identities, and outcomes are synthetic. The
offline path calls no model, cloud, email provider, CRM, or ticket system and
does not pretend its zero calls are provider-cost evidence.

## Run it

From the repository root:

```bash
make email-support-check
make email-support-demo
make email-support-workshop
make email-support-mlops-plan
```

The complete check executes all four workshop levels, renders the mutation-free
MLflow monitoring/curation/optimization plan, validates ingress through
delivery and production-readiness evidence, runs the optional native LangGraph
adapter against the agent template's certified dependency lock, and evaluates
the full synthetic release set against the deterministic domain gate. The demo
prints a prepared case; it does not send an email.

Install the locked GenAI extras, then run the live local target through the
judge-free AgentKit smoke policy:

```bash
make examples-install
AAI_PLATFORM_CONFIG=examples/email-support-agent/config/aai-platform.example.yml \
PYTHONPATH=examples/email-support-agent/src \
  .venv/bin/agentkit smoke \
  --config examples/email-support-agent/agentkit.smoke.yaml --live
```

`--live` invokes the local preparation target and may record local traces, but
smoke deliberately runs only shared deterministic code scorers: it never buys
judge calls and does not establish a promotion baseline. Preview the governed
release plan separately:

```bash
AAI_PLATFORM_CONFIG=examples/email-support-agent/config/aai-platform.example.yml \
PYTHONPATH=examples/email-support-agent/src \
  .venv/bin/agentkit compare \
  --config examples/email-support-agent/agentkit.yaml --mode live --plan
```

The release-policy plan explicitly selects correctness, safety, guidelines, and
three retrieval judges. Its 11 rows and `top_k=4` produce a preflight ceiling
of 99 judge calls, just below the configured limit of 100. Replace the example
resource names through the normal environment configuration before a paid run.
The checked-in `target.py` always uses synthetic deterministic adapters: it
proves evaluation wiring only and must never establish an adoptable baseline.
Bind `--agent endpoints:/<candidate-endpoint>` or the generated `agent-app`
candidate for baseline/result comparisons, then use `compare`, `gate`, and
`evidence` for the release decision. AgentKit owns shared scorer semantics and
judge budgets. The email-specific route, authorization, privacy, and
idempotency checks stay deterministic in
`email_support_agent.evaluation`; they should move into the shared platform
scorer catalog only when other projects need the same versioned definition.

## Repository map

| Path | Purpose |
|---|---|
| `MATURITY_SCORECARD.md` | Exact 10/10 definitions, checked-in evidence, and the live-production proof still required. |
| `REFERENCE_DESIGN.md` | Corrected architecture, state model, span plan, evaluation, feedback, security, and cost controls. |
| `UPSTREAM_ADAPTATION.md` | Pinned review scope, grade, gap map, and clean-room relationship to the GitHub demo and supplied picture. |
| `src/email_support_agent/contracts.py` | Strict untrusted-input, review, action, and persisted-evidence boundaries. |
| `src/email_support_agent/policy.py` | Deterministic routing, evidence authorization, output gates, review policy, and action planning. |
| `src/email_support_agent/workflow.py` | Side-effect-free preparation and idempotent transactional commit. |
| `src/email_support_agent/ports.py` | Trusted access/review authorization and provider capability boundaries. |
| `src/email_support_agent/ingress.py` | Fail-closed webhook bounds, verification, replay, scan, DLP, identity, storage, and audit ordering. |
| `src/email_support_agent/delivery.py` | Active leases, provider idempotency, bounded retry, receipts, and dead-letter behavior. |
| `src/email_support_agent/offline.py` | Transparent fakes and executable offline check. |
| `src/email_support_agent/evaluation.py` | Release cases, metrics, and hard gate. |
| `src/email_support_agent/feedback.py` | De-identified review, delivery, resolution, and reopen trace signals. |
| `src/email_support_agent/mlops.py` | Dry-run-first monitoring, curation, judge alignment, optimization, lineage, cost, and human-promotion contracts. |
| `src/email_support_agent/production.py` | Release-bound evidence and fail-closed production-readiness scorecard. |
| `workshop/` | Four executable lessons using the real accelerator package. |
| `recipes/langgraph/` | Native optional graph adapter using the already-certified template dependency lock. |
| `recipes/mlflow/` | Reviewed MLflow 3/AIMLOps plan and notebook-source monitoring recipe. |
| `agentkit.smoke.yaml` | Credential-free code-scorer policy for fast local/PR feedback. |
| `agentkit.yaml` | Release-policy judge selection, thresholds, regression budgets, and spend ceiling; the default synthetic target is plan-only evidence. |

## Production graduation

1. Generate `agent-app` through the normal platform workflow. Do not copy this
   example's configuration as deployment infrastructure.
2. Move the domain package and LangGraph adapter under generated `src/app/`.
   Keep the template's pinned `aai-core` wheel and certified LangGraph lock.
3. Implement the injected capabilities behind `IngressCoordinator`,
   `EmailSupportWorkflow`, and `DeliveryWorker`: verified ingress access
   resolver, review identity authorizer, transactional outbox, provider
   delivery, and approved logical resources:
   `email-triage`, `email-draft`, and `support-knowledge`. Keep physical
   endpoints in environment configuration.
4. Have the platform process provision the email webhook, raw store, DLP,
   checkpoint/store, outbox, CRM/ticket functions, App identity, grants,
   model endpoints, and search index. Application code and CI do not create
   them.
5. Register immutable classifier and draft prompt versions, bind the complete
   model/prompt/index/embedding/chunking/tool release, establish a baseline
   against the real candidate target, require connected usage/cost coverage,
   run the full gate, deploy in shadow mode, and canary only the proven
   low-risk segment.
6. Assemble a sealed `ProductionEvidencePack` from the approved security,
   reliability, load, restore, canary, SLO, and FinOps systems. An illustrative
   policy or synthetic attestation must remain `UNVERIFIED`.
7. Start delivery workers only after idempotency, replay, provider receipt,
   bounce, lease-expiry, retry, and dead-letter tests pass. “Queued” is not
   “delivered.”

See [REFERENCE_DESIGN.md](REFERENCE_DESIGN.md) before connecting a provider.
