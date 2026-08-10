# Changelog

All notable changes to `aai-core` are documented here.

## Unreleased

## 0.4.0

Migration notes:

- `ResourceContext.data_classification` is now the closed
  `DataClassification` enum: `public`, `internal`, `confidential`, or
  `restricted`. Replace custom values before upgrading. The default trace
  policy is now bounded for public/internal data and metadata-only for
  confidential/restricted data. Explicit policies may make capture stricter;
  application code cannot enable payload capture for sensitive classes.
  Metadata-only spans retain only typed, allowlisted operational identifiers,
  lineage, and token/cost counters. Arbitrary SDK attributes are dropped,
  framework-owned values are reduced to shapes, payload shapes do not retain
  mapping keys, and caller session IDs are hashed. MLflow Agent Server root
  inputs and post-handler outputs now pass through the same export policy;
  tracing-off disables its native tracing. Agent Server payloads are sanitized
  once at export, payload replacement fails closed, arbitrary sensitive-mode
  request metadata is dropped except hashed request/correlation IDs, and
  provider/tool exception messages never enter governed spans. The AAI export
  policy now composes after existing MLflow span processors rather than
  replacing them, and traced async generators keep one parent trace/context
  for their complete streaming lifetime.
- Runtime configuration now fails safe for environment names. Only `dev`,
  `development`, `local`, and `sandbox` receive relaxed checks; every other
  name, including misspellings, must provide production-grade catalog,
  schema, experiment, identity, ownership, and lifecycle values.
- `GateResult` now records the required canonical `policy_digest` and the
  optional `baseline_digest`. Prefer `apply_gate()`, which supplies both;
  callers constructing `GateResult` directly must now provide a 64-character
  SHA-256 `policy_digest`.
- `ApplicationRelease` now defaults to schema version `2`, adds `world`,
  `tools`, and `control` evidence, and writes World/Learning/Control clock
  digests. Consumers that require the original document shape and digest must
  explicitly set `schema_version="1"` and omit the new evidence fields.
  Both schema versions now reject credential-bearing keys and values before
  release evidence can be persisted.
- `ai-platform/v1` manifests may declare the external
  `spec.costControls.budgetPolicy` reference. It remains optional in the v1
  SDK contract so existing manifests retain their canonical JSON and hashes;
  new generated projects declare `platform_standard_v1`. Keep prices,
  credentials, and enforcement state outside the manifest.
- Every generated project now runs a credential-free
  `scripts/validate_project.py` from `make check` and pull-request CI. It
  requires the budget-policy reference and rejects drift between
  `ai-app.yaml`, `aai-platform.yml`, and Databricks bundle job resources for
  application, environment, owner, team, cost center, classification,
  lifecycle, repository, preset/job/task-cluster tags, approved compute policy,
  and the evaluation job. Governed values are rendered rather than mutable
  bundle variables; persisted overrides, undeclared targets, existing or
  serverless compute bypasses, and target/resource override sections fail the
  generated contract check.
- Generated GenAI evaluation paths now consume the native Unity Catalog
  EvaluationDataset named in each application's Hub manifest, verify it against
  the reviewed repository suite, record dataset ID/digest and distinct
  target/judge identities, and associate dataset creation with the governed
  experiment rather than applying unsupported dataset tags. The non-LLM
  experiment starter follows the same dataset identity, association, drift,
  and lineage contract without inventing target/judge metadata. The agent
  starter also returns native MLflow assessments for feedback/curation and
  centralizes request, tool, output-token, and trace-capture bounds. Existing
  UC datasets associated with a different experiment fail closed; use a new
  governed, versioned dataset name instead of silently reusing them. Agent
  release evidence joins a full clean source commit, prompt, dataset, target/judge,
  tool schema, execution limits, gate policy/baseline, manifest, budget, and
  service-level contract; CI supplies the attested remote Git provenance. RAG
  releases now use the same fail-closed v2 pattern for exact prompt and
  knowledge versions, model/embedding/retrieval/index configuration digests,
  bounded RAG limits, SDK/source provenance, dataset identity and association,
  target/judge identities, and gate policy/baseline evidence.
- Added the strict, provider-neutral `AgentDecision` evidence contract and
  best-effort `record_agent_decision()` helper. Meaningful application
  decisions are native MLflow `AGENT` spans that complement, but never replace,
  authoritative TOOL/RETRIEVER/LLM execution spans. Metadata-only capture keeps
  only the validated decision type, selected action, and optional confidence;
  it suppresses goals, reasons, references, alternatives, and expected results.

- Added the `analytics-app` template: a self-service analytics agent
  implementing the published four-layer architecture (canonical data,
  semantic layer first, knowledge + runbook skills, offline-pinned
  validation) over a **neutral, repo-owned semantic layer** — a strictly
  validated YAML contract compiled to portable SQL by pure application
  code, executed behind a three-method `WarehouseExecutor` protocol
  (Databricks statement-execution adapter shipped; other warehouses are an
  application-code implementation away). Every answer carries a
  code-rendered provenance footer (tier › sources › owner › freshness ›
  SQL) and per-pass token accounting; the two-tier gate reproduces every
  golden value credential-free from a versioned snapshot, then re-judges
  against the live warehouse with cost-coverage enforcement. The platform
  console offers the template, and `docs/analytics-lifecycle.md` documents
  the lifecycle, eval-set design, tokenomics, and context engineering.
  `aai-core` itself is unchanged.
- Made `platform-identifiers.json` the only file a clone edits for environment
  identifiers. `scripts/sync_template_shared.py` now stamps the four
  platform-controlled defaults in every template schema and the identifier
  literals in `databricks.yml`; documentation sources `scripts/platform-env.sh`
  instead of restating a workspace host, and a smoke test fails on any `*.md`
  that restates one. A downstream clone's divergence from upstream is now two
  files, both marked `merge=keepours` in `.gitattributes`, so tracking upstream
  no longer re-resolves the same conflicts on every sync. See
  `docs/enterprise-clone-runbook.md` sections 3 and 3a.
- Added `sdk_pip_source` to the identifier fixture and to the values that are
  stamped and cross-checked. It is where a generated project's credential-free
  CI installs `aai-core` from; nothing previously checked it against the
  fixture, so a clone could pass every test while shipping five templates whose
  CI installed the SDK from the upstream repository.
- The platform console now refuses to generate a `bundle init` when it is
  hosted with no `template_repo` configured, and reports that as a failed
  platform-state check, instead of emitting an in-checkout relative path that a
  hosted viewer cannot use.
- Added `docs/platform-audit.md` and removed the retired `templates/agentic-rag`
  tombstone (still last renderable at tag `v0.2.0-agentic-rag-final`).

- Added the platform console (`src/platform_app`), a Databricks App that renders
  the onboarding lifecycle, generates the exact `bundle init` command for a
  chosen template with this workspace's identifiers substituted, and reports
  app-service-principal platform state. It deliberately does not verify a
  developer's personal access: on-behalf-of-user consent is irrevocable and its
  scopes do not reach compute policies, volumes or catalog grants. Served
  locally with `make app-run`; stopped by default once deployed. See
  `docs/platform-console.md`.

## 0.3.0

Migration notes:

- The default experiment scope changes from
  `/Shared/<team>-<application>-<environment>` to
  `/Shared/<team>-<project>-<application>`. Explicit `experiment_name`
  configuration is unchanged; keep one set explicitly if existing evidence
  must remain in its current experiment.
- `ResourceContext.lifecycle` now accepts the closed values `experimental`,
  `candidate`, `production`, and `retired`. Replace old `validation` or
  `active` values deliberately.
- Persisted SDK contracts now reject unknown fields and implicit type
  coercion. Pass correctly typed values or validate their JSON representation
  with the model's Pydantic API.
- The `candidate` prompt alias is deprecated in favor of `validation`; it
  remains compatible until `0.5.0`.
- `aai_core.serving` is removed. Generated applications emit deployable
  artifacts and declare native resources; approved external platform
  processes own endpoint creation, deployment, permissions, and rollback.

- Reworked the learning examples into one progressive, deterministic MLflow
  evidence path: descriptive experiments, baseline/change lineage, immutable
  prompt versions, tracing/tracking, quality/latency/token/cost measurement,
  reproducibility capture, and an explicit release decision.
- Added strict Pydantic contracts and closed enums at persisted and untrusted
  boundaries while preserving native MLflow/provider clients and result
  objects as supported escape hatches.
- Added trace capture policies, execution-local request context, one
  process-startup `TraceIntegration` owner, cost-coverage-aware
  `MetricRule`/`GatePolicy` evaluation contracts, and native MLflow result
  handling.
- Added a Databricks Apps + MLflow Agent Server deployment path to the agent
  template, with native async `@invoke`/`@stream`, application-owned async
  tools, and an optional durable async LangGraph recipe. Removed the duplicate
  synchronous models-from-code agent serving path.
- Added machine-readable SDK/template/runtime compatibility, generated exact
  transitive template locks, dependency policy, Python and provider
  compatibility CI, scheduled lower/latest-bound canaries, grouped Renovate
  updates, and manifest-last immutable publication.
- Added session-aware traces, secure opt-in MLflow OpenAI/LangChain
  autologging (including compatible LangGraph agents), bounded LLM/tool span
  inputs, outputs, and token usage, complete tool-loop response/usage
  aggregation, provider-native async/streaming clients, event-loop blocking
  protection, and assessment provenance passthrough.
- Added approved logical judge-model resolution and made every generated
  GenAI gate route judges explicitly. Gated row-level scorer errors now fail
  releases instead of disappearing from partial aggregates.
- Upgraded the agent template to preserve governed multi-turn history and gate
  exact tool names, arguments, empty trajectories, and duplicate multiplicity
  with an MLflow `ToolCallCorrectness` compatibility scorer. Responses API
  text parts are normalized while unsupported multimodal input and raw user
  identifiers are rejected or omitted from traced application requests.
- Added an executable domain-policy judge, explicit judge-metric gates,
  human-alignment guidance, and bounded failure-rationale triage to the
  evaluation template.
- Expanded the evaluation lifecycle guidance for judge alignment, held-out
  prompt optimization, conversational evaluation, and cost-quality decisions.

## 0.2.0

- Added the five-template catalog (experiment-starter, prompt-app,
  evaluation-project, rag-app, agent-app) with a shared synced scaffold,
  discovery-driven render-matrix tests, and per-template two-tier release
  gates.
- Retired `agentic-rag` into `rag-app` + `agent-app` (last renderable at tag
  `v0.2.0-agentic-rag-final`).
- Added platform conventions for experiments and evaluation runs:
  `effective_experiment_name` (`/Shared/<team>-<application>-<environment>`
  unless configured), portable `bootstrap()` config discovery
  (`AAI_PLATFORM_CONFIG` or upward search), and
  `EvaluationSuite.run_tracked` — every template gate is now a governed
  MLflow run linking the pinned prompt URI, the registered UC dataset,
  traces, params/metrics, and an `aai.gate_passed` verdict, deep-linked
  from the published report.
- Registered ground truth with the catalog across templates
  (scripts/sync_dataset.py) and added agent register-as-code
  (`deploy_serving.py --register-only`).
- Added SDK experiment logging/reproducibility helpers, `apply_thresholds`,
  `publish_report`, the tool-execution loop (`ToolRegistry`/`run_tool_loop`),
  `structured.generate_structured`, the serving adapter
  (`agent_resources`/`deploy_agent`), and scripted tool calls in
  `aai_core.testing`.

## 0.1.0

- Added the installable AI/ML platform SDK.
- Added native identity, secret-reference, redaction, and tagging foundations.
- Added Databricks and Foundry model adapters.
- Added Azure AI Search and Databricks AI Search retrievers.
- Added MLflow experiment, prompt, tracing, feedback, and evaluation helpers.
- Added RAG schemas and immutable application release manifests.
- Added the Agentic RAG Databricks bundle template.
- Added immutable Unity Catalog volume wheel publication.
