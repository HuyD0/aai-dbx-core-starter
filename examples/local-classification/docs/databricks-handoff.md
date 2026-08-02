# From this Mac project to Databricks

Moving the workflow does not mean rewriting the statistical contract. Keep the
prediction time, feature exclusions, split policy, seed, candidate comparison,
threshold rule, test gate, signature, and `adopt`/`reject`/`inconclusive`
decision intact. Change the execution, storage, identity, and governance layers.

| Local course component | Governed Databricks counterpart | What must be added |
|---|---|---|
| Generated CSV + SHA-256 manifest | Versioned Delta table in Unity Catalog | Table ownership, schema, grants, lineage, retention, and a reproducible version reference |
| SQLite MLflow backend and local artifact folder | Hosted MLflow tracking | Workspace experiment policy, access control, retention, and cost/ownership tags |
| `mlflow.data.from_pandas` input | Delta/UC dataset input logged with `mlflow.log_input` | Exact table name/version and governed table-to-model lineage |
| Local registered model name | `<catalog>.<schema>.<model>` in Models in Unity Catalog | `USE CATALOG`, `USE SCHEMA`, and least-privilege model grants through the approved platform process |
| Local `champion` alias | Unity Catalog model alias | Review/approval policy; aliases replace deprecated stages |
| Local Python command | A Databricks job packaged with production code under `src/` | Compute policy, environment configuration, retries, notifications, run-as identity, and mandatory cost tags |
| Local Make/config files | Declarative Automation Bundle | Target-specific variables and reviewed deployment configuration; no credentials in source |
| Local predictor | Batch job or Model Serving endpoint pinned to a concrete version | Capacity, latency/error SLOs, request schema, rollback, and access policy |
| Simulated drift report | AI Gateway inference table normalized to governed tables, plus data profiling | Retention/redaction, delayed labels, quality alerts, ownership, and response process |
| Laptop lock and tests | Certified Databricks runtime lock plus credential-free PR tests | Target-Linux artifact validation and authenticated post-merge bundle validation |

## Minimal MLflow configuration change

In an approved Databricks environment, the logical change is:

```python
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

registered_name = "<catalog>.<schema>.subscription_churn_classifier"
```

Do not copy a personal access token, client secret, or workspace password into
the project. This repository's connected path uses approved keyless identity;
local study deliberately needs no identity at all.

Unity Catalog model versions require a model signature. This project already
logs one, plus an input example and dataset lineage, so the artifact contract is
designed to travel. Registration still needs the externally provisioned catalog,
schema, privileges, and platform controls; application code must not create or
broaden those resources to make a demo pass.

## Alias and deployment are separate actions

An alias is a mutable discovery pointer. Record the concrete version resolved
from `champion` for each batch or deployment. When updating a serving endpoint,
resolve the approved alias and configure the endpoint with that concrete model
version. Do not assume moving an alias necessarily updates a running deployment.

## Monitoring after deployment

Log a request identifier, event time, concrete model version, input schema
version, probability, threshold, prediction, latency, and error state. Apply the
organization's privacy and retention rules before capturing inputs or outputs.

Monitor three layers separately:

1. Service health: latency, throughput, availability, and errors.
2. Data and score health: schema, missingness, categories, feature/score drift,
   and predicted-positive rate.
3. Outcome quality: precision, recall, calibration, cost, and slices after true
   labels arrive.

Input drift is a reason to investigate. It is not, by itself, evidence that
outcome quality changed. Conversely, performance can degrade without a large
univariate drift signal.

## Release checklist

- Training, validation, and test sources are immutable and reproducible.
- Transformations fit only inside train/CV folds.
- The positive label, metrics, threshold costs, and gates are reviewed.
- The exact logged model passes reload/parity and target-runtime checks.
- The registered name is three-part and the version carries a signature.
- The model version—not merely the run—links to its dataset and metrics.
- Promotion verifies the exact dataset, model ID, threshold, code/dependency
  policy, gate policy, and evaluation run that passed.
- The alias moves only after approval, and rollback points to a known version.
- Serving and monitoring identities have least privilege.
- The job and serving resources carry required ownership and cost attribution.
- A response owner and action exist for every alert.
