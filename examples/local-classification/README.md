# Local classification with MLflow

This is a small, notebook-led classical machine-learning course that runs on a
Mac without cloud credentials, a GPU, or a data download. It trains a binary
subscription-churn classifier and uses local MLflow tracking plus a local Model
Registry to teach the parts that matter around `fit()`.

The data is synthetic and deterministic. It is useful for learning workflow
discipline, not for making real retention decisions or estimating real-world
model performance.

## Start here

From this directory:

```bash
make install
make check
make notebook
```

Or from the repository root:

```bash
make classification-install
make classification-check
make classification-notebook
```

`make notebook` opens `notebooks/00_start_here.ipynb` with the exact locked
kernel. No notebook needs a network connection or Databricks access. Run the
lessons in order; each one is independently restartable and says which evidence
it creates.

To see a completed lifecycle before studying it cell by cell:

```bash
make pipeline
make mlflow-ui
```

Open <http://127.0.0.1:5000>. The server binds only to localhost and uses the
ignored `.aai/mlflow/mlflow.db` SQLite database. Stop it with `Ctrl-C`.

## What you learn

The project follows one evidence chain:

```text
problem and action contract
  -> deterministic data and lineage manifest
  -> time-ordered train / validation / frozen-test split
  -> no-skill baseline
  -> leakage-safe preprocessing Pipeline
  -> fair candidate comparison on validation only
  -> cost-aware threshold chosen on validation only
  -> one frozen-test evaluation and slice gate
  -> adopt / reject / inconclusive decision
  -> register the exact passing artifact and assign champion
  -> reload, infer, and simulate monitoring
```

The notebooks make the modeling work visible, while repeatable logic lives
under `src/aai_local_classification/`. That separation is intentional: notebooks
are excellent for exploration and teaching; tested packaged code is what jobs
and releases should execute.

## Standards deliberately practiced

- Define the prediction time, target horizon, action, positive label, error
  costs, and gate before model comparison.
- Split by time before fitting imputers, scalers, encoders, or models.
- Keep identifiers, timestamps, the target, and post-outcome fields out of the
  feature contract.
- Establish a `DummyClassifier` result before claiming improvement.
- Select candidates with a threshold-independent validation metric and a
  predeclared simplicity tolerance, then choose an action threshold using
  validation data and declared costs.
- Treat the test partition as frozen release evidence, not a tuning resource.
- Report precision, recall, F1, PR-AUC, ROC-AUC, Brier score, log loss,
  confusion counts, action cost, and operational slices; accuracy alone is not
  sufficient for an imbalanced task.
- Log dataset inputs, a SHA-256 manifest, seed, source state, parameters,
  metrics, artifacts, model signature, input example, the exact lock, and a
  digest binding evidence to code, dependencies, and declared policy.
- Register only an adopted model and use a `champion` alias; deprecated model
  stages are not used.
- Refuse to reuse a consumed frozen-test version for a different model or
  policy, and verify exact dataset/run/model/policy linkage before promotion.
- Reload the registered artifact before inference and keep the chosen threshold
  with its model version evidence.
- Treat drift as a diagnostic signal. Only delayed labels can establish
  performance or calibration drift.

## Project map

```text
configs/project.yaml               problem, split, cost, and gate policy
data/README.md                     generated-data and lineage contract
docs/curriculum.md                 lesson outcomes and completion rubric
docs/resources.md                  curated primary-source reading path
docs/databricks-handoff.md         exact local-to-Databricks concept map
notebooks/                         ordered executable lessons
src/aai_local_classification/      reusable data, model, tracking, gate code
tests/                             fast behavioral and lifecycle checks
uv.lock                            exact cross-platform dependency resolution
```

Generated CSVs, MLflow databases/artifacts, workflow state, and notebook outputs
stay out of Git. The source generator and manifest make the dataset reproducible.

## Important limits

This is a learning system, not a production reference architecture. Synthetic
data cannot validate business utility, fairness, privacy, or representativeness.
SQLite is suitable for one local learner, not a concurrent team registry. A Mac
environment is not proof that an artifact will work on a Databricks Linux
runtime; test the locked target runtime before release. The final notebook and
[Databricks handoff](docs/databricks-handoff.md) explain what changes—and what
must stay the same—when moving the workflow to a governed workspace.
