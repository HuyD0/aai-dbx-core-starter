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

From your own machine (the normal case), point `bundle init` at this
repository's Git URL:

```bash
az login
export DATABRICKS_HOST=https://adb-7405609799238491.11.azuredatabricks.net
export DATABRICKS_AUTH_TYPE=azure-cli
databricks bundle init https://github.com/HuyD0/aai-dbx-core-starter \
  --template-dir templates/rag-app --output-dir my-project
cd my-project
python3.12 scripts/setup_dev.py
```

(Inside a checkout of this monorepo, `databricks bundle init
./templates/<template-name>` works too.)

## 2. Authenticate keylessly

```bash
az login
export DATABRICKS_HOST=https://adb-7405609799238491.11.azuredatabricks.net
export DATABRICKS_AUTH_TYPE=azure-cli
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
- Dataset or trace-set reference.
- Parameters and application release.
- Quality, latency, and cost measurements.
- A conclusion.

Use `ExperimentManager.run()` so the standard ownership and release tags are
attached automatically. Experiments follow the platform naming convention
(`/Shared/<team>-<application>-<environment>`) unless `experiment_name` is
set explicitly; strict environments require an explicit name. Evaluation
gates run through `EvaluationSuite.run_tracked(...)`, so every gate is a
governed MLflow run carrying the pinned prompt URI, the registered Unity
Catalog dataset name, gate metrics, an `aai.gate_passed` verdict tag, and
the evaluation traces.

## 5. Develop prompts and retrieval

Register prompts rather than copying prompt strings between notebooks.
Reference immutable prompt versions during evaluation and production; use the
controlled `development`, `candidate`, and `production` aliases for promotion.

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

The candidate must pass absolute thresholds and allowed regression limits
against the deployed baseline.

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

Production applications emit sampled asynchronous traces. Attach end-user and
expert feedback to the originating trace, monitor quality and operations, and
promote real failures into the regression dataset.
