# Developer guide

Complete the [developer onboarding checklist](developer-onboarding.md) before
generating a project. The platform team prepares group-based access; the
generated setup command verifies it without granting permissions.

## 0. Start locally, then use the workspace

From a fresh checkout, create the locked environment and run the zero-credential
example:

```bash
make quickstart
make local-start
make local-ui  # open http://127.0.0.1:5000; Ctrl-C stops it
```

`local-start` records a trace in `.aai/local/mlflow.db`; it does not use cloud
credentials or the legacy root `mlflow.db`. Once the local trace is visible,
create and preflight the keyless workspace configuration:

```bash
make workspace-connect
# Complete the reported keyless authentication/configuration actions.
make workspace-example EXAMPLE=first_trace
```

The second run sends the same example to the configured Databricks experiment.
View it in the workspace UI. Application deployment comes later through a
generated template's `make bundle-validate bundle-deploy` targets.

## 1. Generate a project

Pick the template that matches what you are building (each wizard asks only
non-secret configuration; the README's template table has the decision
guide):

- `experiment-starter` — reproducible MLflow experiments (LLM-free)
- `prompt-app` — governed prompt lifecycle with judged, pinned-version evals
- `evaluation-project` — standalone eval harness for an existing app/endpoint
- `rag-app` — governed retrieval-augmented generation
- `agent-app` — tool-using agents with gated serving
- `analytics-app` — self-service analytics agent over a versioned semantic
  layer (see [the analytics lifecycle](analytics-lifecycle.md))

From your own machine (the normal case), point `bundle init` at this
repository's Git URL:

```bash
az login
source scripts/platform-env.sh
databricks bundle init "$AAI_TEMPLATE_REPO" \
  --template-dir templates/rag-app --output-dir my-project
cd my-project
python3.12 scripts/setup_dev.py
```

(Inside a checkout of this monorepo, `databricks bundle init
./templates/<template-name>` works too.)

## 2. Authenticate keylessly

```bash
az login
source scripts/platform-env.sh
```

Do not create a PAT or client secret. The generated `scripts/setup_dev.py`
checks authentication, SDK volume access, and compute-policy visibility before
installing anything. Run `aai-core doctor --cloud` when a later provider
identity or permission error is unclear.

## 3. Explore deliberately

Use the generated notebook to inspect data and test an idea. Once behavior is
reusable, move it into `src/app`, give it a typed interface, and add tests.
Production jobs invoke packaged Python rather than using a notebook as the
application boundary.

## 4. Record experiments

Every meaningful comparison should have:

- A question or hypothesis.
- A descriptive run name that says what changed.
- Dataset or trace-set reference.
- Parameters and application release.
- Quality, latency, and cost measurements.
- A baseline link and an explicit conclusion.

Use `ExperimentManager.run()` with `ExperimentRunMetadata` so standard
ownership, purpose, change id, hypothesis, and baseline lineage are searchable
tags. One experiment is the durable comparison space for an application:
`/Shared/<team>-<project>-<application>` unless explicitly configured.
Environment remains a run tag, so evidence is comparable without scattering
it across environment-named experiments. Strict environments require an
explicit experiment name.

The normal evidence sequence is:

```text
baseline -> change -> result -> adopt | reject | inconclusive
```

The word `change` means only “the controlled difference under test.” It is not
a new deployable object or wrapper around an MLflow run.

Call `record_reproducibility()` inside the run to capture source commit/state,
SDK version, seed, and an installed-package freeze. Use native MLflow APIs for
datasets, metrics, artifacts, logged models, and features the helper does not
cover; `ExperimentManager.native_client` is the deliberate escape hatch.

## 5. Develop prompts and retrieval

Register prompts rather than copying prompt strings between notebooks.
Reference immutable prompt versions during evaluation and production; use the
controlled `development`, `validation`, and `production` aliases for
promotion. The old `candidate` alias remains temporarily compatible but is
deprecated.

For RAG, version the source snapshot, parser, chunking profile, embedding
profile, search index, filters, and reranker behavior. A change to any of these
is an application change.

## 6. Evaluate

Begin with manually reviewed golden cases and known safety/failure cases.
Add representative production traces as the application is used. Evaluate:

- Retrieval relevance and recall.
- Access-filter correctness.
- Groundedness and citation correctness.
- Answer relevance and safe abstention.
- Tool selection and arguments.
- Latency and provider cost.

The change must pass absolute thresholds and allowed regression limits against
the deployed baseline. A gate result is evidence, not deployment permission by
itself: persist an `adopt`, `reject`, or `inconclusive` decision with its
rationale and evidence identifiers.

Measure quality, latency, tokens, cost, and **cost coverage** on the same cases.
Unknown cost is not zero. It either blocks the gate or is reported explicitly,
according to the versioned gate policy. Begin with deterministic scorers;
introduce an LLM judge only after comparing it with held-out human labels.

Tracing has an equally explicit choice:

- `TraceIntegration.SDK` for stable `model.generate()` calls and a bounded,
  provider-neutral span shape.
- `TraceIntegration.MLFLOW_OPENAI` for direct native sync, async, or streaming
  OpenAI calls after full-content capture is approved.
- `TraceIntegration.MLFLOW_LANGCHAIN` for LangChain/LangGraph.
- `TraceIntegration.MLFLOW_AGENT_SERVER` when Agent Server owns the root trace
  and the application uses one bounded child instrumentation path.
- No tracing only when a declared policy selects `OFF`.

Configure this selection once at process startup; identical configuration is
idempotent and conflicts fail. Never toggle autologging in a request handler.
Do not enable an autologger on top of SDK spans for the same call; duplicate
spans also duplicate token and cost evidence.

### Keep trace ownership and assurance evidence explicit

MLflow is the authoritative assurance evidence plane for applications. Its
traces, runs, versioned EvaluationDatasets, and Feedback / Assessments are the
records evaluated by the release gate. Foundry and Application Insights provide
a complementary service-operational view; neither backend copies verdicts or
reviewed datasets into the other.

Tracing ownership determines which privacy controls actually apply:

| Trace source | Owner and privacy boundary | What it proves |
|---|---|---|
| SDK-owned application tracing | The application selects one provider/integration, exporters, attributes, and bounded inputs/outputs. Its declared MLflow capture policy applies to these spans. | Client behavior the application directly observed. |
| Foundry client instrumentation | The application owns the client OTel provider, but the preview `AIProjectInstrumentor` has its own content, trace-context, and baggage switches. Content capture stays off by default. | Foundry SDK calls and correlated client activity. |
| Foundry-native or hosted telemetry | The Foundry protocol runtime owns server-side instrumentation and the platform-connected Application Insights export. Its runtime policy—not the SDK trace policy—controls those payloads. | What the managed service observed, subject to sampling and ingestion. |

Do not assume that disabling content in one owner redacts spans produced by
another. Exception messages, identifiers, tool definitions, and custom span
attributes can still be sensitive even when prompt/response capture is off.
Never request or store hidden model chain-of-thought to fill an observability
gap.

Native OTel GenAI conventions give MLflow a useful execution mapping after
ingestion: `invoke_agent` is agent activity, `chat` is a model call, and
`execute_tool` is actual tool execution. A tool call represented only inside a
model response is intent, not proof that application tool code ran. Native
Agent Framework 1.12.1 emits repeated `chat` and `execute_tool` spans as direct
sibling children of `invoke_agent`; do not teach a TOOL span as nested under the
model span that requested it. A hosted protocol runtime can add another parent,
so validate the deployed platform root and managed-backend translation live.
Native instrumentation has no portable decision-reason, retry/recovery, or human-
approval span contract, so add small application spans only for those meaningful
events. Record failures on execution spans and keep evaluator verdicts in
Feedback / Assessments.

Debug from the producer outward:

1. Prove the expected span tree and parent IDs with an in-memory exporter.
2. Confirm exactly one instrumentation owner and the intended content policy.
3. Confirm exporter flush, endpoint, protocol, routing headers, and renewable
   authentication without printing credentials.
4. Confirm ingestion in Application Insights or the MLflow receiver.
5. Confirm Foundry correlation fields, viewer RBAC, sampling, and ingestion
   delay.
6. Confirm MLflow translated the native spans, then run independent scorers.

The production improvement loop is similarly reviewed: sample useful traces,
redact/minimize them, obtain human review and expert Feedback, promote approved cases into a
versioned MLflow EvaluationDataset, run deterministic checks before calibrated
judges, and require the normal regression gate. Do not treat raw production
traffic as benchmark truth or imply that an agent recovered unless an observed
span or application event records that recovery.

## 7. Deploy and monitor

The platform team must first onboard the generated repository with a
repository-specific main-branch FIC, dev-only Databricks service principal,
constrained compute permission, and non-secret repository variables. Do not
reuse this template hub's identity.

Validate and deploy the generated bundle:

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

For agent HTTP serving, the generated agent template uses MLflow Agent Server
on Databricks Apps as its native deployment path, with async `@invoke` and
`@stream` handlers. LangGraph is an optional application-owned recipe for
durable graph execution; inject an async persistent checkpointer/store and put
an interrupt before irreversible work. When to reach for the recipe, how
review decisions become trace evidence, and the Lakebase persistence wiring
are covered in [Production LangGraph agents](langgraph-production.md).

Production applications emit sampled asynchronous traces. Attach end-user and
expert feedback to the originating trace, monitor quality and operations, and
promote real failures into the regression dataset.
