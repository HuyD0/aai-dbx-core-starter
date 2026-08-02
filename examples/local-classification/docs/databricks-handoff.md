# Move the local lifecycle to Databricks

Read this after lesson 09. It assumes you can already explain the local dataset
splits, MLflow experiment and runs, logged model, registered version, `champion`
alias, inference output, and monitoring signals. The
[glossary](glossary.md) defines those local terms.

Moving to Databricks changes **where** data, execution, evidence, models, and
predictions live. It does not make the statistical rules optional. Keep the
prediction time, target, feature exclusions, train/validation/test boundary,
candidate-selection metric, threshold procedure, final gate, and model
input/output contract.

## Five Databricks ideas to learn first

1. **Unity Catalog** governs named data and AI assets, their permissions, and
   lineage. A three-part name such as `dev.ml.churn_features` means
   `catalog.schema.object`.
2. A **Delta table** is a versioned table format. It replaces the course's local
   CSVs with governed, queryable data and a reproducible version reference.
3. **Hosted MLflow** replaces the laptop's SQLite database and artifact folder.
   Experiments, runs, inputs, metrics, and logged models retain the same roles.
4. A **Declarative Automation Bundle** describes jobs and their environment as
   reviewed configuration. Packaged Python under `src/` remains the execution
   unit; a notebook is not the production job.
5. **Model Serving** or a batch job loads a concrete approved model version.
   Inference tables and governed data profiling help observe requests, outputs,
   schemas, and distribution changes.

These are platform capabilities, not permission to create infrastructure from
application code. Catalogs, schemas, identities, compute policies, endpoints,
and grants are provided through the approved external platform process.

## Concept map

| Local concept you already used | Databricks counterpart | What changes | What must remain true |
|---|---|---|---|
| Generated CSV files plus `manifest.json` | Versioned Delta tables in Unity Catalog | Data is stored and governed centrally; jobs read a table/version instead of a laptop path | Row meaning, feature timing, split rules, schema checks, and immutable version evidence |
| SHA-256 dataset digest | Table identity/version plus MLflow dataset input and governed lineage | Platform lineage can connect tables, runs, and models | Record an exact reproducible source; do not log only a friendly table name |
| Local SQLite MLflow backend and artifact directory | Databricks-hosted MLflow tracking | The workspace stores run metadata and artifacts with access control | Parameters, metrics, inputs, artifacts, tags, and model IDs keep their meanings |
| Local experiment name | Workspace MLflow experiment | Naming, ownership, retention, and access become shared platform concerns | One run still represents one declared execution |
| Local logged sklearn pipeline | Logged model in hosted MLflow | Artifact storage moves; target-runtime dependencies must be validated on Linux | Preprocessing and estimator remain one fitted artifact with a signature/input example |
| `subscription_churn_classifier` | `<catalog>.<schema>.subscription_churn_classifier` in Models in Unity Catalog | The registered name is governed and three-part | Registration occurs only after the exact artifact passes the release decision |
| Local numbered model version | Unity Catalog model version | Permissions and lineage are governed centrally | Consumers record the concrete version they actually use |
| Local `champion` alias | Unity Catalog model alias | Review policy controls who may move it | The alias remains a mutable pointer, not immutable deployment evidence |
| Local Python/Make command | Databricks job from packaged code in a bundle | Compute, retries, notifications, environment, and run-as identity are configured | The job calls tested source code and preserves the same data/evaluation boundaries |
| Local `predictor.predict(...)` | Batch inference job or Model Serving endpoint | Capacity, latency, errors, access, and rollback become operational concerns | Input signature, positive class, threshold, output schema, and concrete version remain explicit |
| Simulated reference/current comparison | Inference table plus governed tables and data profiling | Production requests/scores and later outcomes can be joined under retention/privacy rules | Input or score drift is diagnostic; delayed labels are required for outcome quality |
| `.venv`, `uv.lock`, exported model requirement closure, and Mac tests | Certified Databricks runtime/dependency resolution plus CI tests | The target is Linux and may use a platform runtime | Validate the exact supported environment before release |

## The smallest conceptual MLflow change

After a platform owner has provided the workspace resources and permissions,
the tracking and registry destinations change conceptually like this:

```python
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

registered_name = "<catalog>.<schema>.subscription_churn_classifier"
```

Do not paste a personal access token, client secret, storage key, or workspace
password into source, a notebook, GitHub, or environment files. This repository's
connected path uses approved keyless identity. Local study needs no cloud
identity at all.

Unity Catalog model versions require a signature. The local course already
teaches a signature, input example, dataset input, and reproducibility artifacts
so the model contract can travel. Databricks still needs externally provisioned
catalogs, schemas, privileges, compute policy, and identities.

## Promotion is not deployment

The course's `champion` alias answers “which registered version is currently
approved for this role?” A serving endpoint or batch job has its own deployment
configuration. A safe release therefore:

1. resolves `champion` to a concrete version;
2. records that version in the deployment change;
3. validates the model in the target runtime;
4. updates the job or endpoint to that concrete version; and
5. retains a known prior version for rollback.

Moving an alias must not be assumed to update a running endpoint automatically.

## Monitoring after deployment

Keep the four questions separate:

| Question | Example signals | What the signal can establish |
|---|---|---|
| Is the service working? | Latency, throughput, availability, error rate | Operational health |
| Do current inputs still satisfy the contract? | Missing columns, types, null rates, new categories | Schema/data health |
| Have inputs or scores changed? | Feature distributions, predicted-positive rate, score distribution | A reason to investigate, not proof of quality loss |
| Is outcome quality still acceptable? | Precision, recall, calibration, cost, slices after labels arrive | Real model/action quality for the measured labelled cohort |

When policy permits, record a request identifier, event time, concrete model
version, input schema version, probability, threshold, prediction, latency, and
error state. Apply privacy, minimization, access, and retention rules before
capturing any input or output. Join delayed labels using governed identifiers;
never infer real recall from unlabeled score drift.

## Recommended migration order

1. **Reproduce data:** publish immutable train, validation, and test versions as
   governed tables and verify counts, dates, schema, and digests.
2. **Reproduce training:** run the packaged pipeline with hosted MLflow and
   compare local/Databricks metrics within justified numeric tolerances.
3. **Validate the artifact:** reload the exact logged model on the target Linux
   runtime and confirm signature and prediction parity.
4. **Reproduce the gate:** verify the same selected run/model, threshold, costs,
   checks, and decision evidence.
5. **Register conditionally:** register the passing artifact under a three-part
   Unity Catalog name; do not register a refit substitute.
6. **Deploy a concrete version:** configure a batch job or endpoint, test the
   request/response contract, permissions, rollback, and service limits.
7. **Connect monitoring:** capture permitted operational evidence, establish
   reference windows, join delayed labels, assign owners, and test response
   playbooks.

## Release checklist

- Data sources are immutable, reproducible, access-controlled, and correctly
  divided into training, validation, and final test roles.
- Preprocessing fits only with training data or inside training folds.
- The positive label, metric meanings, threshold costs, capacity assumptions,
  and release rules have owners and review evidence.
- The exact logged model reloads successfully in the target runtime and retains
  its signature and input example.
- Model version evidence links to the exact dataset, training run, selection
  result, threshold, final-test run, and gate decision.
- Registration and alias movement occur only for an adopted artifact; reject
  leaves the current alias unchanged, while infeasible selection stops before a
  test gate exists.
- Jobs and endpoints use least-privilege identities and approved compute policy.
- Required ownership, environment, team, cost, lifecycle, and data-classification
  attributes are present on supported resources.
- Monitoring respects privacy and retention policy, and every actionable alert
  has an owner, response, and safe rollback/escalation path.

For this repository's identity and provisioning boundaries, continue with the
root [cloud setup](../../../docs/cloud-setup.md),
[platform operations](../../../docs/platform-operations.md), and
[tagging standard](../../../docs/tagging-standard.md). Do not add credentials or
infrastructure creation to this learning project.
