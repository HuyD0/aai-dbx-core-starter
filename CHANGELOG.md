# Changelog

All notable changes to `aai-core` are documented here.

## Unreleased

- Added `docs/llmops-playbook.md`, mapping industry LLMOps practice areas onto
  the platform's AI application lifecycle for application teams and the
  platform team, with a maturity checklist and an honest gap roadmap.
- Added `aai_core.decisions`: the `adopt`/`reject`/`inconclusive` `Decision`
  vocabulary, the strict `DecisionRecord` contract (an adopt must cite a
  passing, metrics-bearing gate whose recorded policy applied at least one
  release rule; `decided_by` rejects personal emails; `prompt_digest` and
  `release_digest` accept only sha256 hexdigests so raw prompt text, user
  content, or secrets cannot enter persisted tags; `prompt_name` and
  `prompt_version` bind the registry identity the evidence was recorded
  for), and `record_decision()`
  writing the decision as a governed run with searchable `aai.decision` tags
  and a `decision.json` artifact.
- Added thin evaluation helpers. `judge_model_uri` is restored from 0.2.0 as
  the single resolver for the approved judge endpoint (it had been duplicated
  across five template sites) and rejects setup-placeholder deployments
  (`replace-with-*`, `unset`, …) so the doctor never reports a placeholder
  judge as ready; `log_gate_evidence` standardizes the gate
  metrics and `aai.gate_passed` tag templates were hand-writing;
  `evaluate_with_gate` composes native `mlflow.genai.evaluate()` with
  `apply_gate()` through kwargs passthrough and returns the native result by
  identity — unlike the removed 0.2.0 `EvaluationSuite.run_tracked`, it owns
  no run and mirrors no native parameters; `GateResult` records the applied
  `GatePolicy` and regression baseline and re-validates its failures against
  them at construction, so gate evidence is self-describing and cannot claim
  a pass its own metrics contradict; while scorer-error enforcement is on,
  `apply_gate` refuses to produce evidence from a non-finite scorer
  error-count metric instead of silently discarding scorer health, a
  negative error count fails the gate as corrupt inside the recomputation,
  and per-row `<scorer>/error_message` failures in a native
  `mlflow.genai.evaluate()` result are counted into persisted
  `<scorer>/error_count` evidence (native results never aggregate them);
  `get_or_create_evaluation_dataset` promotes the governed dataset helper
  from `examples/notebook_setup.py` (which keeps its copy until the
  notebooks migrate) and fails locally on placeholder catalog/schema
  qualifiers instead of querying the registry.
- Added `aai_core.scorers` with the deterministic code scorers shared by
  gates and monitoring, plus a lazy `as_mlflow_scorers()` adapter that wraps
  dependency-free `registered_*` bodies (logic inlined, equivalence
  test-enforced) so registered monitoring scorers survive MLflow's
  body-only serialization in a scoring service without aai-core installed.
  `MONITORING_SCORERS` is the reference-free subset for sampled trace
  monitoring; reference-based scorers stay with offline evaluation where
  ground-truth expectations exist. Template copies are unchanged until
  each template's next version.
- Added `aai_core.monitoring`: `log_feedback()` forwarding to native MLflow
  with a required, nonblank, non-personal assessment `source_id` so no
  governed feedback lands without provenance, and `traces_with_feedback()`
  for curating reviewed production traces into the governed regression
  dataset, counting only valid feedback assessments — expectations,
  invalidated (overridden) entries, and errored scorer feedback never
  select a trace; convert selected
  traces to record dictionaries before `merge_records` (managed datasets
  reject native traces). Sampled-scorer registration remains a documented
  notebook step.
- Added evidence-gated prompt promotion: `prompt_digest()`,
  `PromptManager.ensure_version()` registering idempotently by content
  digest across every registry page (promoted from the lifecycle examples),
  and `PromptManager.promote()` moving a governed alias only on an adopt
  decision whose `prompt_digest`, qualified `prompt_name`, and immutable
  `prompt_version` were recorded at decision time and match the registry
  version's actual template and the prompt and version being promoted
  (`aai_core.prompts.promotion_blocked` otherwise) — gate evidence alone
  carries no template identity, content identity is not registry identity,
  and two versions can share a template, so evidence gathered for one
  prompt, template, or version can never promote another. `set_alias()`
  is unchanged. `PromptManager`
  fails locally on unconfigured or placeholder catalog/schema qualifiers
  instead of querying the registry for names like `unset.unset.<name>`;
  explicit `catalog.schema.name` qualification remains untouched.
  `is_missing_prompt_error()` is public so callers seeding a first version
  or first promotion can distinguish an absent prompt or alias from
  authentication, permission, and transient registry failures instead of
  catching broadly; structured non-missing codes override "does not
  exist" message wording, the common non-disclosure phrasing, and the
  same shared predicate guards the dataset helper's create path.
  `promote()` verifies the target version through `get_prompt_version()`,
  the only fetch with no lineage side effects — every `load_prompt`
  flavor, the client-level one included, links the loaded version to
  active lineage — so a rejected change is never attached to an active
  experiment, run, model, or trace.
- Added lifecycle-readiness checks to `aai-core doctor` (experiment name,
  prompt-registry catalog/schema, judge-model resolution); optional
  configuration reports skip with remediation, never fail. The
  prompt-registry preflight applies the same qualifier validation as the
  SDK helpers (placeholder vocabulary and dotted values alike), and an
  experiment name that is a placeholder — explicit, or derived from
  placeholder team/project/application components — reports skip instead
  of passing the literal through to the registry.
- Aligned the executable examples' release decision with the documented
  vocabulary: `aai.decision` and `LIFECYCLE_RESULT` now record
  `adopt`/`reject` instead of `release_change`/`keep_baseline`.
- Extended the curriculum to 00–14 with two teaching notebooks:
  `13_decision_and_promotion_lifecycle.ipynb` (score → gate → decision →
  evidence-gated promotion on the credential-free default path) and
  `14_platform_llm_operations.ipynb` (the platform team's operating loop:
  judge governance, gateway request tags, cost by tag, fleet provenance,
  monitoring adoption, and rollback levers).
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
