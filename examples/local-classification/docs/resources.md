# Curated learning resources

This list favors primary documentation and foundational papers over short-form
tutorials. Read the “core path” in order while doing the matching notebooks;
use the rest when a lesson raises a question.

The project intentionally pins exact versions, including MLflow 3.14.0 and
scikit-learn 1.9.0. MLflow aligns with the repository's certified range, while
the scikit-learn documentation matches this standalone course's lock. Versioned
MLflow links keep the executable course stable even when rolling documentation
changes.

## Core path

1. [scikit-learn: common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
   Read before modeling. It explains inconsistent preprocessing, leakage, and
   why pipelines must learn transformations only from training data.

2. [scikit-learn: cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
   Learn what train, validation/CV, and held-out test evidence can legitimately
   tell you. This course uses a time holdout rather than assuming exchangeable
   rows.

3. [scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
   Use this as the metric reference. Pair it with [classification decision
   thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html)
   and [probability calibration](https://scikit-learn.org/stable/modules/calibration.html).

4. [MLflow tracking quickstart (3.14.0)](https://mlflow.org/docs/3.14.0/ml/getting-started/quickstart/)
   Experiments, runs, parameters, metrics, models, and the UI in one short flow.

5. [MLflow local database tutorial (3.14.0)](https://mlflow.org/docs/3.14.0/ml/tracking/tutorials/local-database/)
   Explains the SQLite topology used here. The file backend is legacy; SQLite is
   a sensible single-user local backend but not a production team store.

6. [MLflow sklearn guide (3.14.0)](https://mlflow.org/docs/3.14.0/ml/traditional-ml/sklearn/)
   Covers autologging and explicit sklearn model logging. This course logs the
   intended datasets, metrics, and model explicitly so the evidence is visible
   and duplicate exploratory model versions are avoided.

7. [MLflow dataset tracking (3.14.0)](https://mlflow.org/docs/3.14.0/dataset/)
   `mlflow.log_input` records source, schema, profile, and digest metadata. It
   does not turn a mutable source into durable, immutable raw-data storage; the
   course therefore records its own SHA-256 split manifest too.

8. [MLflow classic model evaluation (3.14.0)](https://mlflow.org/docs/3.14.0/ml/evaluation/)
   This is the correct evaluator for classifiers and regressors. Do not mix it
   with `mlflow.genai.evaluate`, whose scorer system solves a different problem.

9. [MLflow model signatures (3.14.0)](https://mlflow.org/docs/3.14.0/ml/model/signatures/)
   A signature defines input/output names and types; a representative input
   example supports validation and later serving. Models newly registered in
   Unity Catalog require signatures.

10. [MLflow Model Registry workflow (3.14.0)](https://mlflow.org/docs/3.14.0/ml/model-registry/workflow/)
    Learn registered versions, tags, and aliases. Stages are deprecated; this
    course moves `champion` only after the gate passes.

## Modeling depth

- [DummyClassifier reference](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html): establish a no-skill comparison.
- [Average precision reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): understand the course's primary selection metric.
- [Precision-recall plots for imbalanced data](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432): Saito and Rehmsmeier's explanation of why ROC plots can look optimistic on heavily imbalanced tasks.
- [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) and [Cawley & Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html): understand selection bias when tuning is extensive.
- [Model persistence](https://scikit-learn.org/stable/model_persistence.html): serialization security, `skops`, environment compatibility, and why artifacts must come from trusted sources.

## MLflow depth

- [MLflow Tracking concepts (3.14.0)](https://mlflow.org/docs/3.14.0/ml/tracking/): backend versus artifact stores, dataset inputs, runs, and Logged Models.
- [MLflow model dependencies (3.14.0)](https://mlflow.org/docs/3.14.0/ml/model/dependencies/): inspect and validate the model environment. Keep the exact `uv.lock` as the tested project contract.
- [MLflow pickle-free models (3.14.0)](https://mlflow.org/docs/3.14.0/ml/tracking/pickle-free-models/): the course explicitly uses `skops` and trusts only the required NumPy dtype type.
- [MLflow Model Registry aliases (3.14.0)](https://mlflow.org/docs/3.14.0/ml/model-registry/workflow/#deploy-and-organize-models-with-aliases-and-tags): load by a human-readable role while recording the concrete version used.

## Databricks path

- [Train sklearn models on Databricks](https://docs.databricks.com/aws/en/machine-learning/train-model/scikit-learn): the closest hosted equivalent to these local pipelines.
- [MLflow experiment tracking on Databricks](https://docs.databricks.com/aws/en/mlflow/tracking): move metadata/artifacts from local SQLite to hosted tracking while retaining run structure.
- [MLOps workflow](https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-workflow): development, staging, deployment, and monitoring responsibilities.
- [Manage model lifecycle in Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/): three-part model names, required signatures, lineage, privileges, aliases, and version loading.
- [Declarative Automation Bundles](https://docs.databricks.com/aws/en/dev-tools/bundles/): package jobs, resources, variables, and deployment configuration as reviewed code.
- [Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints): deploy a concrete approved model version for online inference.
- [Unity AI Gateway inference tables](https://docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints): capture serving requests and responses for governed observability.
- [Data profiling](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling): current table/profile monitoring path. Older `quality_monitors` APIs and legacy inference tables should not anchor a new design.

## Production thinking

- [Google's Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml): practical system and iteration rules; especially useful before adding model complexity.
- [Hidden Technical Debt in Machine Learning Systems](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/): why data and system dependencies dominate the model file.
- [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/): a rubric for testing data, models, infrastructure, and monitoring rather than judging only offline accuracy.

## A useful warning about tutorials

Tutorials often optimize hyperparameters on a variable named `X_test` and then
report that same score as final evidence. Treat that partition as validation,
create a separate untouched test set, and name the objects for their real role.
The variable name does not create the statistical boundary; team behavior does.
