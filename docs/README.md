# Documentation index

Every document in `docs/`, one line each. `tests/test_docs_index.py` fails
when a top-level document is added without a line here — an unlinked
document is invisible to developers and agents.

## Start here

- [Developer onboarding](developer-onboarding.md) — access checklist before generating a project.
- [Developer guide](developer-guide.md) — the main developer workflow.
- [SDK public API](sdk-api.md) — the deliberately small public SDK surface.
- [Quality standards](quality-standards.md) — acceptance contract for the SDK, templates, and examples.

## Lifecycle guides

- [GenAI and RAG lifecycle](genai-lifecycle.md) — the evidence chain from experiment to release.
- [Agent evaluation](agent-evaluation.md) — the comparison-first paved road and its gate.
- [Analytics lifecycle](analytics-lifecycle.md) — self-service LLM analytics, from question to governed result.
- [Production LangGraph agents](langgraph-production.md) — durable state, interrupts, and long-term memory.
- [Multi-agent systems](multi-agent-systems.md) — when a second agent pays its way, delegation traces, and coordination scorers.
- [LLMOps playbook](llmops-playbook.md) — industry practice map and shared terminology.
- [UAT promotion](uat-promotion.md) — the dev-to-UAT delivery path.
- [Versioning](versioning.md) — SDK versioning and deprecation policy.

## Platform and operations

- [Platform architecture](platform-architecture.md) — where AAI Core sits in the platform.
- [Platform operations](platform-operations.md) — SDK volume bootstrap and platform controls.
- [Platform console](platform-console.md) — the guided console shipped as a Databricks App.
- [AI Platform Hub](ai-platform-hub.md) — the hub control plane and the `ai-app.yaml` contract.
- [Cost estimation](cost-estimation.md) — the console's list-price workload estimator.
- [Cloud setup](cloud-setup.md) — connecting externally provisioned resources.

## Governance and standards

- [Secrets and identity](secrets-and-identity.md) — credential preference order and keyless identity.
- [Tagging standard](tagging-standard.md) — canonical non-secret tag fields.
- [Agent context management](agent-context-management.md) — how agents and developers share this repository's memory.

## Enterprise adoption

- [Enterprise adoption guide](enterprise-adoption-guide.md) — bringing the repository into an enterprise environment.
- [Enterprise clone runbook](enterprise-clone-runbook.md) — standing up a clone in another organization and tenant.
- [Upstream release prompt](upstream-release-prompt.md) — cutting a release a downstream clone can merge cleanly.

## Reference and history

- [Platform audit](platform-audit.md) — July 2026 point-in-time audit, retained for decision history.
- [MLflow cookbook assessment](mlflow-cookbook-assessment.md) — dated relevance review of the MLflow cookbook.

## Decision log

- [Decision log](decisions/README.md) — dated records of engineering decisions and their rationale.

Individual decision records are deliberately not listed here: downstream
clones add their own records, and an enumerated list would conflict on every
upstream merge. `ls docs/decisions/` is the index.

Clones: append your own documents to this index. Merge conflicts here are
additive-line only — keep both sides.
