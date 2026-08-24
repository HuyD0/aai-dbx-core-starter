# Official learning resources

You do not need to read everything before starting. Use the **Read now** item
beside each lesson, then return to the **Go deeper** links when you want more
detail. The course [glossary](glossary.md) gives shorter definitions.

This path favors official documentation and foundational papers. The executable
project pins MLflow 3.15.1 and scikit-learn 1.9.0; versioned MLflow links match
the code you run.

## Before lesson 00: Mac, Python, and notebooks

**Read now**

- [uv installation](https://docs.astral.sh/uv/getting-started/installation/) —
  official Mac installation and shell-path help.
- [uv Python versions](https://docs.astral.sh/uv/concepts/python-versions/) —
  explains managed Python and `uv python install`.
- [JupyterLab: notebooks](https://jupyterlab.readthedocs.io/en/stable/user/notebook.html) —
  cells, kernels, execution, restart, and run-all.

**Go deeper**

- [Python virtual environments](https://docs.python.org/3/tutorial/venv.html) —
  why `.venv` isolates project packages.

## Lessons 01–02: classification and data

**Read now**

- [scikit-learn getting started](https://scikit-learn.org/stable/getting_started.html) —
  estimator basics, `fit`, `predict`, preprocessing, and evaluation.
- [scikit-learn glossary](https://scikit-learn.org/stable/glossary.html) — precise
  definitions of feature, target, sample, estimator, and related terms.
- [pandas missing-data guide](https://pandas.pydata.org/docs/user_guide/missing_data.html) —
  how pandas represents and inspects missing values.

The course data is synthetic. Before using a real dataset, add its owner,
license, purpose, collection process, privacy constraints, known gaps, and an
immutable version; a digest alone does not answer those questions.

## Lesson 03: splitting and leakage

**Read now**

- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) —
  inconsistent preprocessing, leakage, and safe pipelines.
- [scikit-learn cross-validation and held-out data](https://scikit-learn.org/stable/modules/cross_validation.html) —
  why evaluation data must be separate from fitting.
- [scikit-learn time-series splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) —
  the general rule that training should precede evaluation in time-ordered
  problems. The course uses explicit date boundaries rather than this class.

## Lesson 04: baseline and classification metrics

**Read now**

- [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) —
  the official no-skill estimator reference.
- [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) —
  confusion matrices, precision, recall, F1, average precision, and ROC-AUC.
- [precision-recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html) —
  the plotting API used to see threshold trade-offs on imbalanced data.

**Go deeper**

- [Saito and Rehmsmeier (2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) —
  foundational explanation of precision-recall versus ROC plots for imbalanced
  data.

## Lesson 05: preprocessing, pipelines, and recorded model evidence

**Read now**

- [ColumnTransformer with mixed types](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html) —
  numeric and categorical preprocessing in one pipeline.
- [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) —
  the precise sklearn API contract.
- [MLflow tracking quickstart 3.15.1](https://mlflow.org/docs/3.15.1/ml/getting-started/quickstart/) —
  experiments, runs, parameters, metrics, and models.
- [MLflow local database tutorial 3.15.1](https://mlflow.org/docs/3.15.1/ml/tracking/tutorials/local-database/) —
  the SQLite layout used by this course.

**Go deeper**

- [MLflow sklearn guide 3.15.1](https://mlflow.org/docs/3.15.1/ml/traditional-ml/sklearn/) —
  explicit logging, autologging, model loading, and evaluation.
- [MLflow dataset tracking 3.15.1](https://mlflow.org/docs/3.15.1/ml/dataset/) —
  what a logged dataset input records. It does not make a mutable source durable
  or copy all raw data into MLflow.

## Lesson 06: model choice, thresholds, and calibration

**Read now**

- [classification decision thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html) —
  why probability estimation and the action decision are separate.
- [probability calibration](https://scikit-learn.org/stable/modules/calibration.html) —
  calibration curves, Brier score, and calibrated probabilities.
- [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) —
  the course's candidate-selection metric.

**Go deeper**

- [nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) —
  selection bias when model tuning becomes extensive.
- [Cawley and Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html) —
  foundational treatment of over-fitting during model selection.

## Lesson 07: final evaluation and release evidence

**Read now**

- [MLflow model evaluation 3.15.1](https://mlflow.org/docs/3.15.1/ml/evaluation/) —
  classic classifier evaluation. It is different from MLflow GenAI evaluation.
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) —
  metric definitions and scoring behavior.

**Go deeper**

- [scikit-learn permutation-test example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_permutation_tests_for_classification.html) —
  one introduction to distinguishing apparent score from evidence above chance.
  Confidence intervals and statistical release policies are extensions, not
  beginner prerequisites.

## Lesson 08: model contract and registry

**Read now**

- [MLflow model signatures 3.15.1](https://mlflow.org/docs/3.15.1/ml/model/signatures/) —
  input/output names and types plus input examples.
- [MLflow Model Registry workflow 3.15.1](https://mlflow.org/docs/3.15.1/ml/model-registry/workflow/) —
  registered models, numbered versions, tags, and aliases.
- [MLflow model aliases 3.15.1](https://mlflow.org/docs/3.15.1/ml/model-registry/workflow/#deploy-and-organize-models-with-aliases-and-tags) —
  movable aliases versus concrete versions.

**Go deeper**

- [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html) —
  environment compatibility and serialization security. Load model artifacts
  only from trusted sources.
- [MLflow model dependencies 3.15.1](https://mlflow.org/docs/3.15.1/ml/model/dependencies/) —
  recorded model environments and reproducible loading.

## Lesson 09: monitoring, then Databricks

Complete the local monitoring part before reading the cloud resources.

**Read now**

- [Google Rules of ML: monitoring](https://developers.google.com/machine-learning/guides/rules-of-ml#monitoring) —
  practical production signals and response thinking.
- [MLflow on Azure Databricks](https://learn.microsoft.com/en-us/azure/databricks/mlflow/) —
  hosted counterpart to the local SQLite experiment and Registry.
- [Use scikit-learn on Azure Databricks](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/train-model/scikit-learn) —
  the closest managed execution equivalent.
- [Manage model lifecycle in Unity Catalog](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/) —
  governed names, versions, signatures, permissions, and aliases.

**Continue with the platform pieces only after those concepts are familiar**

- [Declarative Automation Bundles](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/bundles/) —
  reviewed job/resource configuration and deployment.
- [Azure Databricks Model Serving](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/) —
  managed online inference at an approved concrete version.
- [AI Gateway-enabled inference tables](https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/inference-tables-serving-endpoints) —
  governed request/response observability.
- [Azure Databricks data quality monitoring](https://learn.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/data-quality-monitoring/) —
  data profiling, drift, model-performance metrics, and alerts.
- [Azure Databricks MLOps workflow](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/mlops/mlops-workflow) —
  development, deployment, monitoring, and ownership responsibilities.

The [Databricks handoff](databricks-handoff.md) maps these resources to the exact
local objects created by the course.

## Broader production perspective

These are valuable after the ten lessons:

- [Google's Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml) —
  practical principles for starting simple and operating real systems.
- [Hidden Technical Debt in Machine Learning Systems](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/) —
  why data and system dependencies often dominate model code.
- [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) —
  a readiness rubric spanning data, models, infrastructure, and monitoring.

## A warning when reading tutorials

Some tutorials tune repeatedly on an object named `X_test` and then describe the
same score as final evidence. The variable name does not make data untouched.
If its labels influenced a feature, model, hyperparameter, threshold, or rule,
it served as validation data. A final claim needs a separate holdout that did
not choose the system.
