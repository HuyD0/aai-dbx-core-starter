# Developer guide

## 1. Generate a project

Run the Agentic RAG bundle template and answer its non-secret configuration
questions:

```bash
databricks bundle init ./templates/agentic-rag --output-dir ../my-agent
cd ../my-agent
```

## 2. Authenticate keylessly

```bash
az login
export DATABRICKS_HOST=https://adb-7405609799238491.11.azuredatabricks.net
export DATABRICKS_AUTH_TYPE=azure-cli
```

Do not create a PAT or client secret. Run `aai-core doctor --cloud` when an
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
attached automatically.

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
