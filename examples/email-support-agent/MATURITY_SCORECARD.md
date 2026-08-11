# Email Support Agent maturity scorecard

This scorecard defines what “10/10” means for the two different things this
accelerator is intended to be:

1. a high-quality learning solution accelerator; and
2. a complete production reference architecture.

Those scores do **not** certify a live deployment. The repository can prove
contracts, ordering, deterministic policy, failure behavior, evaluation
wiring, and readiness logic without credentials. Only fresh, non-synthetic,
release-bound evidence from the integrating platform can make a particular
deployment production-ready.

## Solution-accelerator score: 10/10

| Point | Definition of done | Checked-in evidence |
|---:|---|---|
| 1 | The original design and its limits are explained honestly. | `UPSTREAM_ADAPTATION.md` pins the reviewed revision and records the clean-room gap analysis. |
| 2 | Learners can progress from graph basics to release decisions. | `workshop/01_*.py` through `04_*.py` form an executable four-level curriculum. |
| 3 | Lessons exercise the real package instead of a second toy agent. | `workshop.py` composes the same strict contracts, workflow, adapters, evaluation, and release objects. |
| 4 | Every lesson has expected observations and an isolated failure exercise. | `workshop/README.md` plus strict `WORKSHOP_RESULT` output and workshop tests. |
| 5 | The default path is reproducible, synthetic, credential-free, and side-effect-free. | Offline adapters are explicitly labelled `TEST-ONLY`; the demo never sends or opens a cloud connection. |
| 6 | Reliability and security are executable, not diagram annotations. | Ingress, review, authorization, atomic outbox, delivery, replay, tamper, privacy, and failure-injection tests. |
| 7 | MLflow tracing and evaluation use native contracts. | Nested `inputs`/`expectations`, scorer-visible retrieval documents, semantic spans, AgentKit smoke and release plans. |
| 8 | The improvement loop is teachable and bounded. | De-identified feedback, trace curation, sampled monitoring, judge alignment, GEPA proposal, holdout comparison, and human decision recipes. |
| 9 | Cost is treated as quality-constrained unit economics. | Pre-call budgets, top-k/output bounds, 99/100 judge-call preflight, cost coverage, reviewer minutes, and safely-resolved-case metrics. |
| 10 | There is one obvious verification and graduation path. | `make email-support-check`; production packaging graduates into the maintained `agent-app` template. |

## Production-reference score: 10/10 coverage

Each production concern has an implementation contract, a deterministic gate,
or both. The score describes architecture coverage—not live operational proof.

| Point | Required production concern | Reference implementation |
|---:|---|---|
| 1 | Authenticated, bounded, replay-safe ingress with clean durable state. | `ingress.py` orders signature verification, age/size bounds, replay reservation, MIME parsing, malware scanning, DLP, trusted identity mapping, encrypted raw storage, and content-free evidence. |
| 2 | Trusted tenant/group authorization and exact knowledge-release scope. | Access-authorizer and retriever ports plus prefilter/post-validation, entitlement digest, and re-authorization at commit. |
| 3 | Deterministic safety routing and pre-spend inference controls. | Security preflight can use zero model calls; the call/context/output ledger reserves budget before every provider invocation. |
| 4 | Durable human review with stale/tampered decision rejection. | Optional LangGraph recipe uses injected async persistence, interrupt-before-write, fresh resume trace, proposal/release binding, and trusted reviewer authorization. |
| 5 | Atomic business side effects and idempotent delivery. | One transactional outbox batch, namespaced keys, immutable payload digests, active leases, bounded retry/dead letter, and provider receipts. |
| 6 | Privacy-aware, scorer-usable observability. | Governed semantic spans honor `OFF`/metadata/bounded modes; external refs are hashed; retrieval emits MLflow document fields. |
| 7 | Deterministic, judged, adversarial, and regression evaluation. | Release cases, hard zero-false-auto-send rules, domain gates, shared AgentKit scorers, retrieval fan-out accounting, baseline/change/result evidence. |
| 8 | Signals can improve the system without training on raw traffic. | Strict review/delivery/outcome feedback, DLP-bound trace curation, risk sampling, SME judge alignment, and disjoint optimization/holdout datasets. |
| 9 | No experiment can promote itself. | Complete model/prompt/index/embedding/chunking/tool/cost lineage, under-scoped-gate rejection, proposal-only optimization, and explicit human `DecisionRecord`. |
| 10 | Operational readiness fails closed. | `production.py` checks owner-approved policy, evidence integrity/freshness/origin, dependencies, SLOs, security, load, restore, canary, unit economics, and four zero-tolerance invariants. |

## What remains external by design

The integrating team must supply real provider adapters and platform evidence:

- email signature verification, enterprise malware/DLP, encrypted raw storage,
  durable checkpoint/store, transactional outbox, email/ticket providers, and
  authenticated review service;
- provisioned logical model/search resources, identity and least-privilege
  grants through the approved platform process;
- connected MLflow traces and comparisons against the real generated app or
  endpoint, not the checked-in deterministic target;
- signed or access-controlled load, restore, security, canary, SLO, and FinOps
  attestations bound to the exact application release.

A deployment remains `UNVERIFIED` until an owner-approved
`ProductionReadinessPolicy` and a sealed, fresh, non-synthetic
`ProductionEvidencePack` make every readiness check pass. Even then, the
scorecard authorizes a human release decision; it never moves a prompt, model,
index, endpoint, or production alias automatically.

## Reproduce the local evidence

From the repository root:

```bash
make email-support-check
```

That target executes the four workshop levels, renders the dry-run MLflow
operations plan, runs every Email Support Agent test, tests the optional graph
against its certified isolated lock, and runs the deterministic release gate.
