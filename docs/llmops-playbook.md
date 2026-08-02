# LLMOps playbook

## Purpose and terminology

Industry writing calls the discipline of operating LLM applications "LLMOps":
prompt management, experiment tracking, evaluation gates, tracing, monitoring,
guardrails, governance, and cost control for GenAI systems. This repository
implements that discipline and calls it the **AI application lifecycle**
(see the naming rule in [the tagging standard](tagging-standard.md)); the
`lifecycle` tag remains a plain maturity value and never a practice name.
This playbook is the bridge: it maps each industry practice area onto the
concrete machinery this platform already ships, for two audiences:

- **Application teams** building prompts, RAG, and agents on the platform.
- **The platform team** operating models, judges, cost, and governance for
  every application. Their view is the
  [operating the platform](#operating-the-platform-platform-team-view)
  section and the platform operations notebook
  ([14](../examples/14_platform_llm_operations.ipynb)).

Vocabulary that matters everywhere: a comparison tests a **change** against a
**baseline**; after evaluation the team records an explicit **decision** of
`adopt`, `reject`, or `inconclusive`; an adopted change becomes an immutable
**release**. `candidate` is deprecated platform terminology, not a lifecycle
stage.

## The operating contract

```text
hypothesis
  → baseline
  → one deliberate change
  → result
  → decision
  → immutable release
```

[The GenAI and RAG lifecycle](genai-lifecycle.md) is the doctrine for this
contract — object glossary, run lineage, evaluation layers, reproducibility
manifest. This playbook does not restate it; each practice area below links
into it and into the code that enforces it.

## Practice map (application-team view)

Each area lists the accepted industry standard, then how this platform
implements it and where.

### Prompt lifecycle

Standard: prompts are versioned artifacts in a registry, loaded by name and
alias, never hardcoded; promotion to production follows evaluation evidence.

Here: `PromptManager` (`src/aai_core/prompts.py`) registers immutable
versions in the MLflow Prompt Registry with governed tags, loads by version
or alias, and constrains aliases to `development`, `validation`, and
`production`. `ensure_version` registers idempotently by content digest, and
`promote` refuses to move an alias without a passing gate or an `adopt`
decision bound to the target version's content digest — evidence gathered
for one template can never promote another.
Examples: [03](../examples/03_first_prompt.py),
[13](../examples/13_decision_and_promotion_lifecycle.ipynb); doctrine:
[prompt promotion](genai-lifecycle.md#prompt-promotion).

### Experiment discipline

Standard: every change is a controlled comparison; runs record purpose,
lineage, and enough context to reproduce the result.

Here: `ExperimentManager` (`src/aai_core/experiments.py`) opens governed runs
with the closed `RunPurpose` vocabulary (`baseline`, `change`, `result`,
`decision`, `monitoring`, `exploration`), refuses sensitive parameters, and
`record_reproducibility` captures commit, environment digest, and the
package freeze. Examples: [01](../examples/01_first_trace.py),
[02](../examples/02_first_experiment.py); doctrine:
[experiment identity and run lineage](genai-lifecycle.md#experiment-identity-and-run-lineage).

### Offline evaluation and release gates

Standard: a curated evaluation set plus deterministic checks and pinned LLM
judges run before every release; thresholds and regression limits block bad
changes in CI.

Here: evaluation stays native — `mlflow.genai.evaluate()` produces the
result, and `aai_core.evaluation` applies deterministic policy over it.
`MetricRule`/`GatePolicy`/`apply_gate` enforce thresholds, regression against
a baseline, minimum cost coverage, and scorer-error failure.
`evaluate_with_gate` composes the native call with the gate;
`log_gate_evidence` persists metrics and the `aai.gate_passed` tag;
`judge_model_uri` resolves the approved judge endpoint from configuration so
no scorer relies on a provider's ambient default. Deterministic scorers ship
in `src/aai_core/scorers.py`. Every generated template carries the same
two-tier gate: credential-free `evals/offline_checks.py` in PR CI and the
judge-backed `evals/evaluate.py` bound to the `release_gate` bundle job.
Examples: [04](../examples/04_first_evaluation.py),
[10](../examples/10_layered_judges.ipynb); doctrine:
[offline and automatic evaluation](genai-lifecycle.md#offline-and-automatic-evaluation).

### Decision records

Standard: evaluation results end in an explicit, recorded decision that
binds the evidence used to make it.

Here: `src/aai_core/decisions.py` provides the `Decision` vocabulary
(`adopt`, `reject`, `inconclusive`), the `DecisionRecord` contract binding
baseline run, change run, gate result, and release digest — an `adopt` with
a failing gate is rejected at the contract — and `record_decision`, which
writes the decision as a governed MLflow run with searchable
`aai.decision` tags and a `decision.json` artifact. Example:
[13](../examples/13_decision_and_promotion_lifecycle.ipynb).

### Tracing and observability

Standard: every LLM, retrieval, and tool call is traced with
OpenTelemetry-aligned spans, under an explicit content-capture policy.

Here: `src/aai_core/tracing.py` selects exactly one process-wide
`TraceIntegration`, applies a `TracePolicy` (`off`, `metadata_only`,
`redacted`, `bounded`, `full`) with payload sanitization, and projects one
`ResourceContext` onto every trace; applications cannot override controlled
fields. Provider adapters emit `LLM`, `EMBEDDING`, and `RETRIEVER` spans
with canonical token usage. Examples: [01](../examples/01_first_trace.py),
[06](../examples/06_connected_first_call.py); doctrine:
[tracking and tracing boundary](genai-lifecycle.md#tracking-and-tracing-boundary).

### Online monitoring and feedback

Standard: production traces are sampled and scored with the same scorers
used offline; human and automated feedback flows back into the evaluation
set.

Here: `src/aai_core/monitoring.py` records governed feedback
(`log_feedback` with an explicit source, never a personal email) and filters
traces by feedback (`traces_with_feedback`) so reviewed failures become
regression records via `get_or_create_evaluation_dataset`. Sampled-scorer
registration (`Scorer.register()`/`.start()`) remains a Databricks notebook
step because the service serializes notebook code; the agent template's
monitoring notebook shows the pattern. Example:
[14](../examples/14_platform_llm_operations.ipynb); doctrine:
[lifecycle](genai-lifecycle.md#lifecycle) steps 10–11.

### RAG hygiene

Standard: retrieval quality is evaluated separately from generation; index
builds, embeddings, and chunking are versioned; retrieved documents are
attributable.

Here: retriever adapters emit MLflow's document schema (`page_content`,
`doc_uri`, `chunk_id`) on `RETRIEVER` spans; `EmbeddingProfile` and
`ChunkingProfile` (`src/aai_core/rag.py`) pin the vectorization contract;
the RAG template versions its chunking job and gates on groundedness.
Doctrine: [RAG trace shape](genai-lifecycle.md#rag-trace-shape).

### Guardrails and safety

Standard: input/output policy checks run offline and online; judges measure
safety; sensitive material never reaches logs, tags, or traces.

Here: safety scorers gate every template release (`Safety` is a required
gated metric); deterministic policy scorers (`refusal_compliance`) stay
report-independent of judges; `SecretValue` never reveals secrets through
any channel; trace policies bound or redact payloads; tags are validated
non-sensitive. Rules: [AGENTS.md](../AGENTS.md) hard security rules,
[secrets and identity](secrets-and-identity.md).

### Governance and cost

Standard: one governance plane, mandatory cost attribution, identity over
secrets.

Here: Unity Catalog governs models, prompts, datasets, and volumes; the
eleven-field tag contract rides every job, cluster, and trace through one
`ResourceContext`; gateway request tags attribute per-request usage; cost
coverage is gate evidence, so missing cost telemetry cannot silently pass.
Keyless OIDC identity runs the whole CI/CD chain. Docs:
[tagging standard](tagging-standard.md),
[secrets and identity](secrets-and-identity.md).

### Release and deployment

Standard: any change to code, model, prompt, tools, index, embeddings, or
chunking is an application release and passes the same gates.

Here: `ApplicationRelease` (`src/aai_core/deployment.py`) is the immutable
release manifest with digest; templates deploy through protected branches
and a credentialed bundle workflow that runs the `release_gate` job before
the application; SDK wheels publish immutably to the artifact volume.
Docs: [platform operations](platform-operations.md),
[versioning](versioning.md).

## Operating the platform (platform-team view)

Application teams consume the lifecycle; the platform team operates the
substrate it runs on. The operator responsibilities and where each lives:

| Responsibility | Machinery |
|---|---|
| Approved model/judge catalog | Logical names in `aai-platform.yml`; provider catalog duties in [platform operations](platform-operations.md#provider-catalog) |
| Judge governance | `judge_model_uri` resolves only the approved `judge-model`; judges are pinned, versioned, and calibrated before gating |
| Endpoint and gateway management | Databricks AI Gateway usage tags via `DatabricksAIRequestTags` (`src/aai_core/tags.py`); quotas and rate limits per [operational controls](platform-operations.md#operational-controls) |
| Cost attribution | Nine mandatory cost tags on all compute; serverless usage policies; billing queries over `system.billing.usage` by tag |
| Fleet oversight | `.aai-template.json` provenance stamps, `ai-app.yaml` Hub manifests, and the platform console ([AI platform hub](ai-platform-hub.md)) |
| Evaluation-dataset governance | Unity Catalog datasets provisioned per application; retention and access policy per [operational controls](platform-operations.md#operational-controls) |
| Monitoring adoption | Sampled scorers registered per application; sampling rates and judge cost budgets are platform-set |
| SDK lifecycle | Immutable wheel publication and the three wheel paths in [platform operations](platform-operations.md#sdk-wheel-lifecycle) |
| Incident and rollback | Immutable releases and prompt aliases roll back by pointer move; revocation via [cloud setup](cloud-setup.md) |

The division of ownership for evaluation is explicit: application teams own
cases, scorer intent, and thresholds; the platform team owns judge
deployments, scorer configuration controls, sampling guardrails, dashboards,
and alert routing. The platform operations notebook
([14](../examples/14_platform_llm_operations.ipynb)) walks this loop
end-to-end: catalog governance, judge resolution, gateway usage tags,
billing-by-tag queries, fleet manifests, monitoring adoption, and rollback.

Bootstrap remains external and human-run: endpoints, catalogs, volumes,
permissions, and identities are provisioned through the approved platform
process, never by this repository's CI or application code.

## Toolkit at a glance

| Capability | Where | Maturity |
|---|---|---|
| Governed experiments and reproducibility | `src/aai_core/experiments.py` | stable |
| Trace policy and governed spans | `src/aai_core/tracing.py` | stable |
| Release gates over native evaluation | `src/aai_core/evaluation.py` | stable |
| Evaluation helpers (judge, gate evidence, datasets) | `src/aai_core/evaluation.py` | preview |
| Decision records | `src/aai_core/decisions.py` | preview |
| Deterministic scorers | `src/aai_core/scorers.py` | preview |
| Feedback and trace curation | `src/aai_core/monitoring.py` | preview |
| Evidence-gated prompt promotion | `src/aai_core/prompts.py` | preview |
| Immutable release manifests | `src/aai_core/deployment.py` | stable |
| Lifecycle readiness checks | `aai-core doctor` | preview |

Maturity is declared authoritatively in `compatibility.json`. The two
teaching notebooks demonstrate the toolkit end-to-end:
[13](../examples/13_decision_and_promotion_lifecycle.ipynb) for application
teams and [14](../examples/14_platform_llm_operations.ipynb) for the
platform team.

## Readiness checklist

A maturity ladder for any application on the platform. Each stage is
checkable; `aai-core doctor` reports the configuration-level items, and the
[examples completion rubric](../examples/README.md) covers the evidence
discipline.

**Instrumented**

- One `TraceIntegration` selected at startup; no double instrumentation.
- A `TracePolicy` states the content-capture posture explicitly.
- Retriever spans emit the MLflow document schema.

**Gated**

- An evaluation dataset of reviewed cases exists with provenance.
- Deterministic scorers cover schema, policy, and critical facts; judges
  are added only where semantic judgment is required.
- Judges resolve through the approved `judge-model`, never a default.
- A `GatePolicy` states thresholds, regression limits, and cost coverage;
  the gate runs in CI and blocks the release job.

**Decided**

- Baseline and change runs share the same ordered dataset digest.
- Every comparison ends in a recorded `adopt`, `reject`, or `inconclusive`
  decision with rationale and evidence.
- Production prompt aliases move only through evidence-gated promotion.

**Monitored**

- Sampled production traces are scored with the same scorers as CI.
- Feedback carries explicit non-personal provenance.
- Reviewed failures become regression records in the governed dataset.

**Governed**

- All eleven tag fields present; cost attribution queryable by tag.
- No secrets in code, prompts, tags, traces, or logs; keyless identity end
  to end.
- Every release is immutable and reproducible from its manifest.

## Getting started

- `make quickstart` — clone-to-running with zero credentials.
- `make local-lifecycle` — the full local trace → experiment → prompt →
  evaluation loop.
- Application teams: work through the
  [progressive examples](../examples/README.md), then generate a project
  per [the developer guide](developer-guide.md).
- Platform team: [platform operations](platform-operations.md), then the
  operations notebook
  ([14](../examples/14_platform_llm_operations.ipynb)).
- `make doctor` — configuration and lifecycle readiness diagnostics.

## Known gaps and roadmap

Stated honestly; the [platform audit](platform-audit.md) tracks these with
acceptance criteria.

- **No path to production yet.** Bundles target one dev workspace; the
  multi-target promotion path with per-environment identities is the
  largest open item (audit P1).
- **`data_classification` has no runtime effect.** It is validated but does
  not yet select a trace policy, and redaction covers credentials, not PII
  (audit P2; sequenced after redaction consolidation).
- **Strict-mode environment matching is an allowlist** (audit P3).
- **Monitoring registration is notebook-bound.** Sampled-scorer
  registration cannot ship as plain SDK code while the service serializes
  notebook code; the platform tracks the Databricks Beta.
- **Templates still carry local copies** of scorers and judge resolution;
  they adopt the shared SDK modules in a follow-up template release.

## References

- [GenAI and RAG lifecycle](genai-lifecycle.md)
- [MLflow cookbook assessment](mlflow-cookbook-assessment.md)
- [Tagging standard](tagging-standard.md)
- [Platform operations](platform-operations.md)
- [Versioning policy](versioning.md)
- [Executable curriculum](../examples/README.md)
- [MLflow GenAI evaluation and monitoring](https://mlflow.org/docs/latest/genai/eval-monitor/)
- [MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/)
- [Databricks production monitoring](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring)
- [OpenTelemetry GenAI conventions — currently Development](https://github.com/open-telemetry/semantic-conventions-genai)
- [OWASP Top 10 for LLM applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
