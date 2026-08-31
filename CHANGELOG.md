# Changelog

All notable changes to `aai-core` are documented here.

## Unreleased

- Gave the SDK's two untyped trace gaps their MLflow span semantics.
  Structured-output parsing and validation now run under a `PARSER` span
  (`structured.parse`, a sibling of the model call's `LLM` span): a schema
  failure is a real, fallible step that the provider call reports as a
  success, so the trace previously showed a green model span and no failure
  at all — the one place `generate_typed`'s sanitized error tells the
  operator to look. The span identifies the call and never restates the
  rejected content. `EMBEDDING` spans now record the provider's billed
  input tokens on `gen_ai.usage.input_tokens`, deliberately not
  `mlflow.chat.tokenUsage`: MLflow aggregates that key across every span
  type into the trace-level total that AgentKit prices at the agent chat
  model's configured rate pair, so the embedding side would be billed at
  chat rates on every retrieval row. `economics._span_usage` now skips
  `EMBEDDING` spans, matching the contract its docstring already stated,
  while spans with an unreadable type still contribute. See
  `docs/decisions/2026-08-31-embedding-tokens-stay-out-of-the-chat-aggregate.md`.
- Recorded a time-boxed dependency-audit exception for CVE-2026-71211
  (GHSA-h7x2-h6g9-p789), the MLflow AI Gateway secret/proxy SSRF. The
  advisory covers 3.13.0 through 3.15.2 with no fixed release inside the
  supported `>=3.15.1,<3.16` range, and it had been failing every
  dependency-audit step in CI since the advisory was re-scored. The
  vulnerable path is the tracking server's own HTTP handlers; neither this
  repository nor a generated project runs an MLflow server, and the
  exception names the affected symbols so it self-invalidates the moment
  any source file references one. It expires 2026-09-30 and must be
  removed once a fixed MLflow ships in range.
- Added an experimental continuous scoring path to AgentKit
  (`aai_core.agentkit.continuous`, `scorers.continuous` in
  `agentkit.yaml`): a logprob-weighted verifier that prompts for a
  single-token letter score, reads the top logprobs at the score
  position, filters to valid score tokens, renormalizes by the retained
  mass, and records the probability-weighted average beside the discrete
  judges — report-only, off by default, and never replacing the discrete
  path. Criteria-decomposed judgments, K repeated evaluations with
  positional alternation for pairwise comparisons, configurable
  granularity (letter scales, 2–26 points), runtime logprob-capability
  probing with a warned fallback to the discrete path on backends that
  return none (the Anthropic API among them), per-run instrument
  telemetry (tie rates for both instruments, normalization mass with a
  low-mass flag, call and token counts, granularity/repeats params), and
  budget-enforced verifier calls. `scripts/sweep_continuous_scoring.py`
  sweeps granularity 5/10/20 × repeats 1/2/4 over graded candidates with
  a known ordering and reports Kendall tau-b ranking agreement and tie
  rate per combination (with a credential-free `--simulate` mode). See
  `docs/continuous-scoring.md`.
- Extended the example curriculum to teach the judge-measurement and
  verified-promotion lifecycle: lesson 12 gains two runnable
  credential-free sections — per-run judge stability (self-consistency
  and frozen-anchor drift, with the "judge changed, not the agent"
  reading) and the committed kappa-vs-SME calibration record behind
  `agentkit judge calibrate` — and lesson 13 gains the
  move-the-baseline-only-after-live-verification flow
  (`agentkit baseline establish --from-run`), demonstrating the gate's
  deployed-commit binding refusal offline.
- Added lesson `06_confidence_intervals_for_release_gates` to the agentic
  operations RAG workshop: a credential-free demonstration of the AgentKit
  statistics module on the workshop's fixed cases — normal versus bootstrap
  intervals on a bounded recall scale, interval width localizing a simulated
  missing-runbook index build, paired per-row improvements deciding what
  overlapping intervals cannot, confidence-bound gating with the
  `minimum_cases` guard, and a seed-robustness exercise. Backing it,
  `agentic_ops_rag.evaluation.benchmark_samples` now exposes the per-case
  scores in dataset order (`None` for out-of-scope cases) and `benchmark`
  derives its aggregates from them unchanged.
- Added run-economics evidence to AgentKit: every live or traces run reads
  its own traces and records success rate, p50/p95 tails for cost, tokens,
  and latency, and cost per successful completion — total known spend,
  failed rows included, over the rows that completed — plus per-stratum
  segments driven by the existing `strata` configuration. Coverage-first
  throughout (`cost/coverage`/`tokens/coverage`; unknown cost is never
  zero, and per-success ratios appear only at complete coverage), with
  cost taken from trace-recorded values or an opt-in
  `economics.price_per_1m_input_tokens`/`..._output_tokens` pair — never a
  shipped price table, and deliberately no mean-cost-per-call metric.
  Report-only by default; gate through the ordinary
  `thresholds`/`regression_budget` grammar (economics directions resolve
  lower-is-better before the registry fallback). The evidence persists on
  `ResultsRecord` as an optional field, so older records stay readable
  while pre-economics readers cannot parse new ones — the same preview-tier
  trade the integrity evidence made. Rationale and rejected alternatives:
  `docs/decisions/2026-08-23-agentkit-economics-evidence.md`. The
  `evaluation-project` template documents the block and moves to 2.3.0.
- Added `statistics.method: bootstrap` to AgentKit: confidence bounds around
  scorer means and paired baseline improvements can now come from seeded
  percentile bootstrap resampling instead of the normal approximation, which
  keeps bounds inside the score's feasible range for bounded judge scales and
  pass/fail rates (a 29/30 routing accuracy no longer reports an upper bound
  above 100%). The method is a reporting policy, not a gate change — both
  methods feed the same `*/statistics/*` gate rules — and results records
  persist the method, resample count, and seed (`bootstrap-percentile-v1` /
  `paired-bootstrap-percentile-v1` per estimate), so an interval can be
  reproduced from its record. Records from before this option deserialize as
  the normal approximation they were computed with; the default is unchanged
  so enforced gates do not move on upgrade.
- Added the repository's shared knowledge layer: a dated decision log under
  `docs/decisions/` (date-prefixed entries so upstream and enterprise clones
  never collide on a name, and deliberately no enumerated index), a
  documentation map at `docs/README.md` enforced by the new
  `tests/test_docs_index.py`, a capture obligation in AGENTS.md section 8,
  and `docs/agent-context-management.md` describing how the layers fit
  together and what repository memory may never contain. The
  `aai-log-decision` skill (canonical in `.agents/skills/`, with a thin
  `.claude/skills/` shim for native Claude Code discovery) walks an agent
  through writing a record. Every generated project now starts with the same
  pattern: the shared scaffold ships `AGENTS.md` (rendered with the project
  name), a `CLAUDE.md` pointer, and `docs/decisions/`, registered in the
  template manifest and asserted through the render matrix. AGENTS.md
  section 11 now points at the decision log and the retained platform audit
  instead of a `docs/archive/` directory that no longer exists.
- Made the `project` cost-attribution tag a clone-owned identifier. It moves
  from a literal repeated in `databricks.yml` and both resource jobs into
  `platform-identifiers.json`, stamped by `make sync-templates` and required by
  the fixture-key guard. The cost anomaly watch buckets spend by that tag, so a
  clone previously attributed its own Databricks usage to this repository with
  a green deploy. Clones merging this release must add `project` to their
  fixture; `job_clusters` now also take `node_type_id` from a bundle variable.
- Documented the repository variables `deploy.yml` already read with
  placeholder fallbacks — `COST_CENTER`, `TEAM`, `OWNER_GROUP`, and
  `COST_ALERT_EMAIL` — and removed the claim in `docs/cloud-setup.md` and the
  enterprise adoption guide that those values are not repository variables.
  `AZURE_SUBSCRIPTION_ID` is no longer documented as a repository variable: no
  workflow reads it.
- Extended the enterprise clone runbook with the steps the fixture cannot
  perform: configuring the Codex Cloud environment (`AAI_CLOUD_ENV=codex` and
  the four values `scripts/cloud-verify.sh` compares — without them its
  identity and forbidden-credential checks are skipped silently), recording the
  new identity in prose, creating the Unity Catalog objects, and replacing
  `.github/CODEOWNERS`.
- Corrected the federated-credential subject documented in `auth-smoke.yml`,
  which showed the name-based form rather than the immutable-id form the
  credential actually uses, and removed this repository's clone URL from
  `README.md` and the `dbx-dev` workspace nickname from the platform console's
  onboarding content. All three are now enforced by tests.
- Replaced the enumerated credentialed-workflow and pull-request workflow
  guards with scans over `.github/workflows/*.yml`. A new credentialed workflow
  could previously add a GitHub `environment:` or a secret reference unchecked,
  and `codeql.yml` was never covered by the credential-free rule.
- Extracted the release-immutability logic from `publish-sdk.yml` into
  `scripts/publish_release.py` with unit tests for every refusal path. Rule 12
  was the only hard rule with no test, implemented as inline shell in a workflow
  that has never run.
- Closed drift gaps: `bundle_identifier_drift()` is now asserted by the test
  suite and the scaffold drift check runs inside `scripts/cloud-verify.sh` (CI
  never invoked `make check-templates`); the dependency canary's supported
  ranges are cross-checked against `dependency-policy.toml`; `deploy.yml`'s
  build job fetches full history so `validate_release.py` stops silently
  skipping its digest cross-checks; and `make check` runs that validation.
- Documentation: removed the models-from-code serving path from AGENTS.md
  section 6 (deleted in 0.3.0), added the UAT workspace to the section 3
  identity table and corrected rule 5, reconciled section 8 with the Makefile,
  added `agentkit`, `scorers`, `decisions`, `monitoring`, and `billing` to
  `docs/sdk-api.md`, and recorded why the generated-project SDK pin is held.

- Added per-run judge-integrity checks to AgentKit: an opt-in
  self-consistency flip-rate over re-judged outputs and a frozen-anchor
  drift check (`evals/judge_anchors.json`, written by judged
  `--establish-baseline` runs) that separates judge drift from agent
  regression, both enforced as gate rules and covered by the judge-call
  budget. Enabling the `integrity:` block makes older results records
  refuse with policy drift — re-run `agentkit compare` after adopting it.
- Bound the AgentKit gate to the commit it runs for: results record the
  full `AAI_RELEASE` commit (job clusters previously recorded
  `local-dev`), and `agentkit gate` refuses evidence scored for a
  different commit than the release identity in its environment.
- Added `agentkit judge calibrate`: chance-adjusted Cohen's kappa against
  SME label consensus with a pairwise human ceiling, persisted as a
  committed per-judge calibration record that evidence reports and —
  under `integrity.require_calibration` — scoring and the gate demand.
- Added `agentkit baseline establish --from-run`, moving the committed
  baseline to an already-verified run's recorded evidence after the
  deploy and post-deploy smoke pass; adopt evidence is required and may
  be recorded in the same step with `--decided-by`.
- Added a post-deploy smoke step to every generated deploy workflow
  (`scripts/smoke_deployment.py`): the Databricks App must report RUNNING
  after `bundle run agent_app`, with opt-in golden-prompt probes via
  `evals/data/live_probes.json`; a red smoke blocks UAT promotion.
  Template versions: agent-app 1.5.0, analytics-app 1.3.0,
  evaluation-project 2.2.0, experiment-starter 1.4.0, prompt-app 1.4.0,
  rag-app 1.4.0.
- Added multi-agent evaluation to the shared scorer registry:
  `delegation_structure_ok` deterministically verifies the AGENT-rooted
  delegation span hierarchy, and `subagent_routing_accuracy` judges the
  supervisor's routing against the recorded trace on a graded 0-1 rubric.
  Delegation is detected only from non-root `AGENT` spans carrying an
  `agent.role` attribute, so single-agent gates never select the new
  scorers, rows outside the convention are skipped and reported rather
  than failed, and a trace-reading prompt judge refuses the Guidelines
  fallback instead of scoring without the trace.
- Added `docs/multi-agent-systems.md`: when a second agent pays its way,
  the delegation trace convention, the coordination scorers, the failure
  modes reported by frontier multi-agent research mapped onto existing
  platform controls, and the backlog for normalizing the Deep Agents
  solution accelerator.
- Added the platform cost anomaly watch: a scheduled bundle job
  (`resources/cost_anomaly_job.yml`) evaluates the previous day's observed
  spend in `system.billing.usage` (list-priced via
  `system.billing.list_prices`) against per-series median+MAD baselines —
  account, workspace, product, and `custom_tags['project']` including an
  `untagged` bucket — plus a new-spend rule and a fail-loud stale-data guard
  (unknown cost is never reported as zero). Exit contract `0`/`2`/`1`;
  failed runs email the `COST_ALERT_EMAIL` group alias. Exactly one live
  schedule: CI's dev deployment unpauses it while laptop and UAT deployments
  stay paused. Detection math is pure stdlib in the new `aai_core.billing`
  module, unit-tested offline; only the loader touches Spark, lazily.
  Reading `system.billing` is an externally granted read documented in
  `docs/cloud-setup.md`.
- Bumped the transitive `sqlparse` pin from 0.5.5 to 0.6.0 (root `uv.lock`,
  both course locks, the classification course's exported model lock, and
  every template's regenerated `requirements.lock`) to clear four published
  advisories (CVE-2026-59893/-59894/-54284/-71491). `sqlparse` is pulled in
  by `mlflow-skinny` (`sqlparse<1,>=0.4.0`); no certified direct dependency
  changed, so `dependency-policy.toml` and `compatibility.json` needed no
  edit. Template locks were regenerated with
  `scripts/lock_template_dependencies.py`, which also picked up unrelated
  transitive patch/minor bumps already eligible under existing certified
  ranges.
- Made the Deep Agents solution accelerator discoverable: a README stating
  its supervisor/sub-agent shape, connected-only boundaries, and guardrails;
  an entry in the examples index; and credential-free contract tests that
  keep every standalone example linked with a README, keep the accelerator
  notebooks output-free and compilable, keep its workspace `%pip` stack
  exact-pinned to the certified dependency line, and keep the notebooks free
  of environment identifiers.
- Extended `docs/langgraph-production.md` with workflow-shape guidance: a
  ten-question design checklist to answer before the first node, recurring
  shapes with their guardrails (fan-out with code-owned reduction,
  independent verification, bounded loops and budgets including a deliberate
  `recursion_limit`, per-node failure policy, typed state with plain
  checkpoints), and when to move from one agent with tools to supervised
  delegation.
- Upgraded the agent template's LangGraph recipe so a review decision is
  evidence, not a bare boolean: strict `ApprovalDecision` resume payloads
  with a reason vocabulary, re-interrupt on malformed payloads instead of
  poisoning the durable thread, bounded replanning with reviewer feedback on
  `ambiguous_intent`, and rejection results that carry reason, note, and
  attempt count.
- Added the `langgraph-lakebase` agent-template recipe: production
  checkpointer/store wiring for the LangGraph recipe using the native
  Postgres saver/store against Lakebase, a fail-closed OAuth credential
  provider that mints a fresh token for every new pooled connection, and
  user-scoped memory tools with decision lineage. A required validated
  `LAKEBASE_SCHEMA` pins every pooled connection's `search_path` to the
  app-owned schema, and `run_setup` verifies schema ownership before the
  one-time DDL. Certified `langgraph-checkpoint-postgres`, `psycopg`, and
  `psycopg-pool`; CI exercises the recipe against a local PostgreSQL server
  and the dependency canary covers both resolution bounds. No Lakebase
  resource is provisioned.
- Added `docs/langgraph-production.md`: when to reach for the LangGraph
  recipes, how review decisions become trace evidence and regression cases,
  the Lakebase persistence contract, and the MCP tool-recipe deferral
  rationale.
- Removed Microsoft Foundry support: the `foundry` model provider, the
  `foundry` and `foundry-labs` extras, the Foundry notebook curriculum, and
  every Foundry template option. Model configuration now targets Databricks
  serving endpoints or an Azure APIM gateway (`azure_apim`, installable via
  the new `azure-apim` extra); external models reach judges and serving
  through governed Databricks external-model endpoints.
- Added an existing-resource-only Lakebase Autoscaling repository for the Hub,
  including checksumed schema migrations, OAuth connection pooling, bounded
  transient-connect retries, optimistic concurrency, and fail-closed hosted
  configuration. No Lakebase, App, or identity resource is provisioned.
- Added an explicit dev-to-UAT delivery contract with immutable wheel evidence,
  manual enablement, lifecycle `validation`, and no production target. UAT keeps
  the protected-main branch-ref OIDC subject and requires external workspace and
  existing-Lakebase onboarding before deployment.
- Added report-only statistical confidence evidence to AgentKit, with optional
  conservative confidence-bound and paired minimum-effect enforcement.
- Added typed retrieval modes plus Azure AI Search semantic ranking and
  Databricks AI Search hybrid reranking controls; provider-specific options stay
  explicit and validated.
- Separated the SDK under development from the SDK default offered to generated
  projects. Release candidates now use a reviewed full Git commit and content
  digest for credential-free CI; only a completed immutable publication may
  transition the default to the annotated version tag. Runtime/UAT remains
  blocked until the volume wheel, checksum, and release manifest exist.

## 0.4.0

- Hardened SDK-owned logging, secret resolution, provider caching, and resource
  cleanup. Registered secrets are redacted from formatted messages,
  tracebacks, and stack text; cold loads are single-flight; owned Databricks,
  Azure, and provider clients close deterministically under races.
- Added tag schema v2 with canonical `validation` lifecycle values while
  retaining a warning-only reader for historical schema-v1 `candidate`
  evidence. Generated templates now emit schema v2 consistently.
- Centralized the application-manifest contract in `aai_core.manifest` and made
  generated project validation fail closed on ownership, cost, lifecycle,
  compute-policy, dataset, and resource drift.
- Replaced model-authored analytics SQL with typed, allowlisted semantic query
  plans and parameterized values. Warehouse work now has bounded concurrency,
  request and statement deadlines, explicit remote cancellation, and
  deterministic executor cleanup.
- Bounded agent streaming output and deadlines, preserved native backpressure,
  and added shared-agent concurrency, slow-consumer, cancellation, provider
  failure, and cleanup coverage.
- Upgraded agent and RAG release evidence to fail-closed schema-v2 joins across
  clean source, exact prompt and dataset versions, gate status, target/judge,
  tool or retrieval configuration, control limits, and policy digests.
- Added full rendered-project quality gates for every supported template
  combination, strict SDK/template typing, branch-coverage ratchets, exact dev
  locks, dependency auditing, and SHA-pinned CodeQL Python/Actions workflows.
- Expanded the credential-free Foundry and lifecycle curriculum with executable
  offline labs, typed support modules, a separate learner workshop, and
  repository-local Codex skills for SDK, template, example, and release work.
- Added `docs/llmops-playbook.md`, mapping industry LLMOps practice areas onto
  the platform's AI application lifecycle for application teams and the
  platform team, with a maturity checklist and an honest gap roadmap.
- Added `aai_core.decisions`: the `adopt`/`reject`/`inconclusive` `Decision`
  vocabulary, the strict `DecisionRecord` contract (an adopt must cite a
  passing, metrics-bearing gate whose recorded policy actually enforced at
  least one substantive release rule — a zero cost-coverage threshold
  gates nothing, and neither do regression-only rules whose baseline
  values were absent under a waived-baseline policy, since
  `_evaluate_policy` skips those checks entirely;
  `decided_by` rejects personal emails; `prompt_digest` and
  `release_digest` accept only sha256 hexdigests so raw prompt text, user
  content, or secrets cannot enter persisted tags; run ids and
  `change_id` are bounded opaque identifiers because they become searchable
  tags; `change_summary` remains bounded prose but, like `rationale`, is
  stored only in the decision artifact; the free-text
  `change_summary`, `rationale`, and `decided_by` are trimmed and must
  stay nonblank, so no decision is persisted with a whitespace-only
  artifact summary or rationale stating no reason; `prompt_name`
  accepts only the qualified
  `catalog.schema.name` shape with no placeholder components and, with
  `prompt_version`, binds the registry identity the evidence was recorded
  for), `decision_digest()` binding the strict artifact to its governed run,
  `load_decision()` refusing unfinished runs or evidence whose identity,
  purpose, searchable tags, gate metrics, or artifact digest disagree, and
  `record_decision()`
  writing the decision as a governed run with searchable `aai.decision` tags
  and a `decision.json` artifact — the free-form change summary and rationale
  persist only inside that artifact, never as run metadata or a run description
  (MLflow stores
  descriptions as the `mlflow.note.content` tag), and the run name derives
  exclusively from the bounded `change_id` (MLflow persists run names as
  the `mlflow.runName` tag, so a free-form override would bypass the
  record's bounded fields).
- Added thin evaluation helpers. `judge_model_uri` is restored from 0.2.0 as
  the single resolver for the approved judge endpoint (it had been duplicated
  across five template sites); the agent, prompt, RAG, and analytics evaluation
  entry points and the agent monitoring notebook now call it before prompt,
  warehouse, tracing, agent, or judge work. It rejects setup-placeholder
  deployments
  (`replace-with-*`, `unset`, `<angle-bracket>` markers, …) as well as
  values outside the serving-endpoint name character set (alphanumerics,
  dashes, underscores — a pasted URI or display label would otherwise fail
  only inside the later evaluation request) so the doctor
  never reports an unusable judge as ready; `log_gate_evidence` standardizes the gate
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
  negative error count fails the gate as corrupt inside the recomputation
  (as does an observed cost coverage outside the `[0, 1]` unit interval,
  which would otherwise satisfy any threshold),
  and per-row failures in a native `mlflow.genai.evaluate()` result —
  `<scorer>/error_message` scorer failures and bare `error_message`
  `predict_fn` failures alike — are counted into persisted
  `*/error_count` evidence (native results never aggregate them), with
  the larger of the aggregate and observed row counts kept so a mapping
  reporting zero cannot erase failing rows;
  `get_or_create_evaluation_dataset` promotes the governed dataset helper
  from `examples/notebook_setup.py` (which keeps its copy until the
  notebooks migrate) and fails locally on placeholder, dotted, or
  invalid-character catalog/schema qualifiers and logical dataset names
  instead of querying the registry. Every identifier crossing into the
  registry — dataset name, catalog and schema qualifiers (in the dataset
  helper and the prompt registry alike), prompt names, and the experiment
  id — must be an actual string, since `str()` coercion turns `None` and
  `123` into the valid-looking names `"None"` and `"123"` that would
  address a real but unintended resource; the experiment id is also
  normalized and placeholder-checked before the first request, so it
  cannot misreport an association the backend reports normalized.
- Added `aai_core.scorers` with the deterministic code scorers shared by
  gates and monitoring, plus a lazy `as_mlflow_scorers()` adapter that wraps
  dependency-free `registered_*` bodies (logic inlined, equivalence
  test-enforced) so registered monitoring scorers survive MLflow's
  body-only serialization in a scoring service without aai-core installed.
  `MONITORING_SCORERS` is the reference-free subset for sampled trace
  monitoring; reference-based scorers stay with offline evaluation where
  ground-truth expectations exist. `refusal_compliance` derives the
  expectation direction from the same refusal-marker vocabulary applied
  to outputs, so a refusal case worded without the word "refuse" still
  gates an unsafe compliant answer, and both reference-based scorers fail
  a missing, blank, null, or non-string expected response — or an
  entirely absent expectations mapping — outright in pure and registered
  forms alike. Pure, AgentKit, and registered forms extract legitimate
  strings and common provider response shapes, while missing or non-text
  outputs — Python/Decimal/NumPy/pandas null and NaN/NA/NaT sentinels,
  numeric scalars, and empty mappings or sequences — fail closed instead of
  being stringified. An absent answer exhibits no refusal behavior to verify
  and covers no keywords, while `str(None)` would otherwise read as a
  compliant non-refusal, take the nothing-to-cover branch, or even match an
  expected keyword "none". A dataset defect or absent answer must never inflate
  a release gate. Template copies are unchanged until each template's next
  version. AgentKit publishes `keyword_coverage`, `refusal_compliance`, and
  `response_length_ok` as scorer version 2 because these output semantics
  materially changed; a baseline carrying their version-1 scores is now
  correctly incomparable and must be re-established rather than mixed into a
  delta.
- Added `aai_core.monitoring`: `log_feedback()` forwarding to native MLflow
  with a required assessment `source_id` namespaced by source kind
  (`group:` for human review, `judge:`/`code:` for automated scorers) so
  no governed feedback lands without provenance and no personal identity —
  username, employee id, or email — can pass as provenance; the trace id,
  assessment name, and span id must be strings and are normalized before
  the native request, so neither a coerced `str(None)` nor an untrimmed
  id can address the wrong trace and no untrimmed name can record
  feedback under a label later lookups miss. Plus `traces_with_feedback()`
  for curating reviewed production traces into the governed regression
  dataset, counting only valid feedback assessments — expectations,
  invalidated (overridden) entries, and errored scorer feedback never
  select a trace; convert selected
  traces to record dictionaries and merge only rows whose expectations
  carry a nonblank string expected response (managed datasets reject
  native traces, and the reference-based scorers fail anything else as a
  dataset defect). Sampled-scorer registration remains a documented
  notebook step.
- Added evidence-gated prompt promotion: `prompt_digest()`,
  `PromptManager.ensure_version()` registering idempotently by content
  digest across every registry page (promoted from the lifecycle examples),
  and `PromptManager.promote()` moving a governed alias only after loading a
  finished persisted decision run and verifying an adopt decision whose
  `prompt_digest`, qualified `prompt_name`, and immutable `prompt_version`
  were recorded at decision time and match the registry version's actual
  template and the prompt and version being promoted
  (`aai_core.prompts.promotion_blocked` otherwise) — gate evidence alone
  carries no template identity, content identity is not registry identity,
  and two versions can share a template, so evidence gathered for one
  prompt, template, or version can never promote another. `set_alias()`
  is unchanged. `PromptManager`
  fails locally on unconfigured or placeholder catalog/schema qualifiers
  instead of querying the registry for names like `unset.unset.<name>`,
  and refuses blank or malformed names (`main.app.`, spaces, punctuation
  outside the recordable `catalog.schema.name` shape) before any registry
  call, so every name it accepts can also receive promotion evidence;
  well-formed explicit `catalog.schema.name` qualification remains
  untouched.
  `is_missing_prompt_error()` is public so callers seeding a first version
  or first promotion can distinguish an absent prompt or alias from
  authentication, permission, and transient registry failures instead of
  catching broadly; structured non-missing codes override "does not
  exist" message wording, the common non-disclosure phrasing. Built-in and
  provider authentication, permission, connection, timeout, transport, and
  non-file OSError types are likewise authoritative even when their message
  deliberately says a protected prompt was not found. The classifier also
  inspects wrapped exceptions and HTTP response status: a genuine 404 remains
  absence, while 401, 403, 429, every 5xx, and other non-404 responses
  propagate. Precedence is evaluated over the bounded, cycle-safe exception
  chain: an explicit `NOT_FOUND`, `RESOURCE_DOES_NOT_EXIST`, 404, exact
  provider `NotFound`, or MLflow missing-alias shape is not erased merely
  because its own class also inherits `OSError`/`HTTPError`, while any nested
  authentication, credentials, quota, HTTP/request, RPC/API, or transport
  failure still propagates. An exact MLflow alias shape accepts only its
  expected HTTP 400 (or no status); 401, 403, 429, and 5xx responses still
  propagate. The bounded walk fails closed when an unseen exception remains,
  and code-less API-key, credential, token, network, host, TLS, DNS, and
  connection failures are recognized from their strong message signals. The
  same shared predicate guards the dataset helper's create path.
  `promote()` verifies the target version through `get_prompt_version()`,
  the only fetch with no lineage side effects — every `load_prompt`
  flavor, the client-level one included, links the loaded version to
  active lineage — so a rejected change is never attached to an active
  experiment, run, model, or trace. Every fetch on that path — the
  decision run, its artifact, and the prompt version — converts absence
  into the guarded refusal with remediation, so invalid promotion input
  never escapes as a raw registry error, while permission and transport
  failures still propagate as themselves.
  The prompt, agent, and RAG templates now persist the exact release-gate
  decision, print its run id, and require that id when their promotion scripts
  call the guarded `promote()` path; their template versions are 1.2.0.
- Added lifecycle-readiness checks to `aai-core doctor` (experiment name,
  prompt-registry catalog/schema, judge-model resolution); optional
  configuration reports skip with remediation, never fail. The
  prompt-registry preflight applies the same qualifier validation as the
  SDK helpers (placeholder vocabulary, dotted values, and invalid
  identifier characters alike), and an
  experiment name that is a placeholder — explicit, derived from
  placeholder team/project/application components, or carrying a
  placeholder path component such as `/Shared/replace-with-experiment`
  (the bare markers match exactly and `replace-with-` is anchored, so a
  whole-string test misses them mid-path) — reports skip instead
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
- Added `agentkit`, the agent-evaluation paved road: a second console script
  in the same wheel (`aai_core.agentkit`) built around one idea — an
  experiment is a comparison, not a log. `agentkit compare` scores this
  version of an agent against the recorded baseline on the same dataset with
  the same scorers; the MLflow run, the lineage tags, the dataset digest and
  the scorer/prompt versions are byproducts the toolkit generates rather than
  things a developer types. `agentkit smoke` is the seconds-long,
  credential-free, judge-free gate for pull requests; `agentkit eval` runs the
  full suite locally or as the bundle's `release_gate` job; `agentkit gate`
  is the promotion check; `agentkit evidence` writes the release record; and
  `agentkit scorers ls` browses the registry. Exit codes are a stable CI
  contract: `0` passed, `2` ran but a threshold failed, `1` configuration or
  runtime error.

  Scorers are versioned platform assets in `aai_core.agentkit.catalog`, with
  judge instructions in the Unity Catalog Prompt Registry — a project selects
  scorers and sets thresholds but never redefines one, so a `correctness/mean`
  of 0.8 means the same thing on two teams. Which scorers apply is inferred
  from the dataset's shape, and scorers whose contract the data cannot satisfy
  (retrieval judges over recorded answers, for instance) are excluded with the
  reason printed rather than silently skipped. Judge spend is estimated and
  confirmed before a run, never reported after it. The gate refuses an empty
  answer to "what did you compare against" and fails closed when a thresholded
  metric never appeared. `agentkit.yaml` is three required lines; everything
  else is an escape hatch. Targets resolve by shape — a local callable, a
  serving endpoint, a Unity Catalog model, or any HTTP/JSON endpoint — so
  execution can move without the record moving. See `docs/agent-evaluation.md`.

  A dataset that carries traces is scored as traces: MLflow replaces a row's
  recorded trace when a `predict_fn` is supplied, so calling the agent again
  would evaluate freshly generated behaviour while reporting it against a
  dataset of production traces. Judge cost accounts for scorer fan-out —
  MLflow judges retrieval relevance once per retrieved chunk and groundedness
  once per retriever span — counted from the rows' traces where they exist and
  assumed (`budget.retrieved_chunks_per_row`) where they do not, so
  `budget.max_judge_calls` is the ceiling it claims to be. Retrieval and tool
  scorers that a live plan cannot decide are named in the plan with the line
  that enables them rather than dropped in silence. Each recorded run attaches
  its results record to its MLflow run, so `agentkit evidence --run <id>`
  works from a machine that never saw the job cluster the gate ran on, and
  every `approval*` tag on a model version is reported rather than only the
  first.

  A results record is self-contained evidence: it carries the gate rules the
  run was judged by and the lineage of the baseline it was compared against,
  so reopening it cannot change the verdict and evidence cannot pair one
  run's deltas with another checkout's baseline. `agentkit gate` refuses a
  record whose rules have since changed instead of judging old numbers by new
  ones. Approval reporting takes the required task names from `approvals:` in
  `agentkit.yaml` — a set discovered from the tags that exist cannot detect a
  required approval whose tag is absent — and says so when they are not
  configured. Trace coverage is per row, so a `trace: null` column or a
  partially traced dataset no longer selects the traces mode.

  A comparison against a baseline that measured something else is refused
  before any judge call: a changed dataset digest, scope, scorer version, or
  judge model stops the run and asks for a new baseline, and
  `--allow-baseline-drift` records the override in the results and the
  evidence rather than removing the control. The dataset digest now covers
  the questions a dataset asks, not the answers under test, so re-recording
  an answer sheet no longer reads as a different dataset. A scorer that
  raised on some rows fails the gate — MLflow reports those failures in its
  result table rather than its metrics, so an aggregate over the surviving
  rows would otherwise pass. Retrieval fan-out is counted from traces
  serialized as JSON strings (what MLflow puts in a dataframe's `trace`
  column) as well as from mappings and objects, and only from top-level
  retriever spans, matching what MLflow actually judges. An evaluation plan
  that selects no scorers is refused instead of recording a run that
  evaluated nothing, and `--mode traces` on a dataset without a trace on
  every row is an error rather than a warning.

  Dataset identity excludes answer behaviour — `outputs`, trace ids,
  timestamps, and responses — so two sets of production traces over the same
  questions stay comparable while an edited case does not; a trace-only row
  takes its identity from the request the trace recorded. In traces mode the
  effective digest additionally binds the trace expectation assessments that
  MLflow substitutes for authored ground truth, so changing what is judged
  cannot reuse the old baseline. A run records the scope it scored at, so a
  sampled baseline fetched by run id is still a sampled baseline. HTTP request
  mapping builds arrays for
  numeric path segments, so the documented `messages.0.content` produces a
  real messages list instead of `{"0": ...}` and preserves what
  `extra_body` already placed there.

  A cancelled run exits 1 rather than 0 — the usual cause is a CI job on a
  non-interactive stream with no `--yes`, and exit 0 there reported success
  for an evaluation that never happened. Comparability covers the scorer set
  as well as scorer versions, since removing a scorer also removes its
  threshold from the policy; a judge-free run such as `agentkit smoke` is not
  treated as a mismatch for skipping judges. A judge prompt whose alias has
  moved is a different judge, so it now blocks the comparison, and the check
  runs before any judge call rather than being reported alongside the
  results.

  The comparability refusal belongs to the promotion-grade commands.
  `agentkit smoke` scores a deterministic sample, so its scope is narrower
  than the committed baseline by design; blocking there stopped the
  credential-free pull-request gate working as soon as a suite outgrew
  `smoke.rows`. `smoke` now sets an incomparable baseline aside, prints every
  reason, and gates on absolute thresholds, while `compare` and `eval` still
  refuse. A sample also carries the digest of the dataset it was drawn from,
  so it is reported as a narrower scope rather than as changed data — which
  is both the accurate reason and the one a developer can act on. Judge
  prompts are compared as a set as well as by version: an alias that stops
  resolving swaps in bundled instructions without raising, and a judge that
  gains a registered prompt is the same change reversed. A baseline fetched
  by run id now restores its recorded prompt versions, without which that
  check could never fire on that path. Trace span kinds are read from the
  spans rather than by scanning the serialized trace, so an answer that
  mentions a retriever or a tool no longer buys judges that cannot score it.
  An HTTP target configured with `request_mapping.auth_env` must use
  `https://` (or loopback): the bearer token would otherwise travel in
  cleartext.

  A scorer whose input contract is a choice is satisfied per row: correctness
  reads `expected_response` or `expected_facts`, so a suite that mixes the two
  keeps the scorer and its default threshold instead of losing both to an
  empty intersection. A scorer needing one specific field still needs it on
  every row. Rows an agent did not retrieve for are skipped by the retrieval
  scorers rather than raising — MLflow raises there, and since scorer errors
  fail the gate, an agent that retrieves only when a question needs it could
  not pass at all. A skipped row is left out of the mean rather than scored
  zero, and the run reports how many rows each scorer actually judged so a
  subset mean is never read as a whole-dataset one.

  A recorded run whose results record cannot be attached to it now fails
  instead of warning. The deployment-job gate scores on an ephemeral job
  cluster, so the run is the only durable copy and the approval task would
  otherwise ask a human to approve evidence `agentkit evidence --run` cannot
  retrieve. That approval task also receives the evaluation task's exact run
  id as a Databricks task value rather than searching MLflow for the newest
  run against the model version, which a concurrent or manual evaluation
  could win; when it does fall back to the search, it says so. The judge-cost
  estimate counts the retrieved context and response a trace carries, so a
  trace-backed retrieval run no longer reports a near-zero token estimate for
  the runs that cost the most. And a suite whose rows are split between
  `expected_response` and `expected_facts` no longer reads as "no
  expectations", which had been adding a thresholded relevance judge nobody
  asked for.

  Approval is reported only for a run that named a version of the configured
  registered model. An endpoint, a local callable, an alias, or a different
  model identifies no version, and reading the newest version's tags instead
  let `"status": "approved"` describe a run that evaluated something else —
  a caveat in the identity string does not stop a machine reading the
  status. An alias stays unresolved on purpose: it may have moved since the
  run. Span outputs now have one reader, so the token estimate sees the
  documents `Span.to_dict()` stores in `attributes["mlflow.spanOutputs"]`
  rather than only a plain `outputs` key — the chunk count already read
  both. And a Unity Catalog dataset's `NaN`/`NaT`/`pd.NA` are recognised as
  absent: a nullable `trace` column had made every row look traced, which
  selects a mode that supplies no predict_fn.

  An authenticated HTTP target no longer follows a redirect to another
  origin. urllib copies every header onto the redirected request, so a
  301/302/303 elsewhere hands the bearer token to whoever answers there;
  the run refuses instead, because a target that redirects away is no
  longer the target the project named. `scorers.remove` now conflicts with
  `regression_budget` as well as `thresholds` — a budget is a gate rule
  too, so removing its scorer meant paying for every judge and then failing
  on a metric that was never going to appear. Gate policy-drift folds in
  the current `scorers.add`/`remove`, so adding a thresholded scorer no
  longer lets a record that predates it exit 0. The judge-cost estimate
  assumes retrieval only for rows whose trace it could not read: a traced
  row that retrieved nothing is a counted zero, matching the scorers that
  skip it, so a conditionally retrieving agent is not refused by
  `budget.max_judge_calls` for calls the run never makes. And dataset
  identity is taken from a trace's root-span inputs before its
  `request_preview`, which MLflow documents as truncatable — two different
  long questions sharing a prefix could otherwise share a digest and pass
  comparability. Trace-only baselines recorded before this change will
  report a digest difference once and need re-establishing.

  Trace reading goes through one helper for both layouts. The identity
  lookup read only `data.spans`, so a payload carrying `spans` at the top
  level still fell through to the truncated preview — the same collision
  the previous fix set out to remove, reachable through the other shape.
  The token estimate likewise prefers the full response and the root span's
  output over `response_preview`, and a plain-text span output is no longer
  dropped for failing to parse as JSON. A comparison also pins *what the
  judge endpoint serves*, not just its name: a governed
  `endpoints:/judge` can be repointed or have a new version promoted behind
  it, and two runs would otherwise look comparable while being scored by
  different models. The identity is read best-effort — a least-privilege CI
  principal may hold `CAN_QUERY` without `CAN_VIEW`, and widening that
  grant to make a check work is not the answer — so an unreadable endpoint
  is reported rather than enforced, and a baseline that pinned one says
  plainly when the current run could not verify it.

  Dataset validation checks a row's `expectations` before excusing it for
  carrying a trace: a trace exempts a row from needing `inputs`, not from
  being well formed, and a malformed value reads as *absent* to shape
  inference — which silently drops the scorers and thresholds that depend
  on it. The evidence pack now records which model the judge endpoint
  actually served, so an approver reading it later can tell. And
  `--rows 0` is refused rather than read as "flag not given", which had
  meant scoring the default scope — every configured judge call — on a
  `--yes` run that asked for the smallest possible one.

  A stored trace now reaches MLflow only in the mode that scores it. A live
  run's answers come from the agent and an answer-sheet run's from the
  file, so the recorded trace is a different run's answer — and MLflow does
  not ignore a trace column it was not asked about: it rewrites inputs,
  outputs and expectations from it, and calls `trace.data` on every value,
  so a single null raised before the agent was ever called. That null is
  how a nullable Unity Catalog column arrives, which the null-sentinel fix
  had made reachable by correctly routing those rows to a live run. A
  missing value is likewise dropped rather than passed, so an absent field
  reads as absent instead of as a float. Because an answer-sheet run no
  longer carries the trace, retrieval and tool-call scorers are excluded
  there and name `--mode traces` — pairing one run's recorded answers with
  another run's retrieval was not evidence about either. In `traces` mode
  the trace's own expectation assessments still win over the dataset's, as
  MLflow intends; the run now says which expectations that applies to
  instead of letting the substitution pass unseen.

  The plan, the cost estimate and the payload are all built from the rows
  MLflow will actually score rather than the rows on disk, and comparison
  identity is recomputed from those effective questions and expectations.
  The authored rows had been deciding all three, which is how a plan could
  promise a scorer whose field the run would not have: one expectation
  assessment replaces MLflow's whole expectations column, so a row whose
  trace carries none loses the curated `expected_response` entirely, and
  `keyword_coverage` reads an absent expected response as a vacuous 1.0 —
  a gate passing on evidence that never evaluated its contract. It is how
  a live run was priced from the recorded agent's retrieval fan-out, so
  `budget.max_judge_calls` authorised it against a number the new agent
  could exceed; the configured `retrieved_chunks_per_row` assumption now
  applies instead, while the scorers the suite was recorded for stay
  selected. And it is how dropping the stored trace could take the
  question with it: a trace-only row now keeps the request recovered from
  its trace, so re-running production traces with `--mode live` still has
  something to send the agent, and a row whose request cannot be recovered
  is named and refused before the run rather than failing inside MLflow.

- Updated the `evaluation-project` template to 2.0.0: it now generates a real,
  runnable agent (`src/app/example_agent.py`) whose gate passes immediately,
  an `agentkit.yaml` carrying a regression budget, and opt-in Unity Catalog
  registered-model and deployment-job-gate resources with a linking script
  (the bundle schema cannot express the model-to-deployment-job link). The
  `evals/` scripts became thin shims over the toolkit and `gate_config.json`
  was removed — thresholds live in `agentkit.yaml` and a scorer's kind now
  comes from the registry, so a report-only judge can no longer be promoted
  into a release threshold by editing a list. Generated projects on 1.1.0 keep
  working; migration is documented, not automated.

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
  Its 1.1.0 template release uses the SDK's canonical, strict judge resolver
  before constructing the live warehouse target. The `experiment-starter`
  template advances to 1.2.0 so its provenance and certified SDK projection
  record the 0.4.0 contract consistently.
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
