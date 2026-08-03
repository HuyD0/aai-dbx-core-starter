# Changelog

All notable changes to `aai-core` are documented here.

## Unreleased

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

  The dataset digest covers the questions a dataset asks and excludes both
  answer fields — `outputs` and `trace` — so two sets of production traces
  over the same questions stay comparable while an edited case does not; a
  trace-only row takes its identity from the request the trace recorded. A
  run records the scope it scored at, so a sampled baseline fetched by run
  id is still a sampled baseline. HTTP request mapping builds arrays for
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
