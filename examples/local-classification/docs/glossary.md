# Beginner glossary

These definitions use the course's subscription example. They are intentionally
plain-language; the [resource guide](resources.md) links to precise reference
documentation.

## Model and data words

| Term | Plain-language meaning | Course example |
|---|---|---|
| Binary classification | Predicting which of two classes an example belongs to. | `churned_30d` is either `0` (did not churn) or `1` (churned). |
| Row / example | One item presented to the model. | One account at one monthly snapshot. |
| Feature | Information available to the model when it makes a prediction. | `monthly_fee` and `contract_type`. |
| Target / label | The outcome the model is trained to predict. | Whether the account churned in the following 30 days. |
| Positive class | The event represented by label `1`. “Positive” means the event of interest, not something desirable. | Churn is the positive class. |
| Prediction time | The moment at which all model inputs must genuinely be available. | The monthly `snapshot_date`. |
| Prediction horizon | The future period covered by the target. | The 30 days after the snapshot. |
| Model / estimator | A learned rule that converts features into an output. scikit-learn calls trainable model objects estimators. | Logistic regression and random forest are two estimators. |
| Fit / training | Learning model values from labelled examples. | `model.fit(x_train, y_train)`. |
| Inference | Using an already-fitted model on new feature rows. | Estimating churn risk for next month's accounts. |
| Probability / score | A number used to rank examples by estimated likelihood of the positive class. | `0.72` means the model assigns a higher churn chance than `0.18`. |
| Threshold | The boundary that turns a score into a yes/no action. | At `0.60`, a score of `0.72` triggers review and `0.18` does not. |
| Hyperparameter | A setting chosen before fitting, rather than learned from rows. | A forest's number of trees. |

## Splitting and preprocessing

| Term | Plain-language meaning | Course example |
|---|---|---|
| Training data | Rows allowed to teach the preprocessor and model. | The earliest account snapshots. |
| Validation data | Separate rows used to compare candidates and choose a threshold. | Later snapshots that do not fit the model. |
| Test data / frozen test | Rows held back until every model and threshold choice is fixed. | The final time period opened in lesson 07. |
| Leakage | Information used during development that would not honestly be available for the claimed prediction. | A cancellation reason recorded after churn. |
| Preprocessing | Converting raw columns into values a model can use. | Filling a missing usage value or encoding a plan name. |
| Imputation | Filling a missing value using a declared rule. | Replace a missing numeric value with the training median. |
| Scaling | Putting numeric features on comparable numeric scales. | Transform fees and ticket counts using training statistics. |
| One-hot encoding | Turning one categorical column into yes/no indicator columns. | `plan_tier=pro` becomes a `plan_tier_pro` indicator. |
| Pipeline | A single object that applies preprocessing and the model in the same order for training and inference. | The course's sklearn preprocessing-plus-classifier object. |
| Deterministic / seed | A repeatable process; a random seed fixes the pseudo-random sequence. | The same generator code, configuration, and seed create the same CSVs. |

## Evaluation words

| Term | Plain-language meaning | Course example |
|---|---|---|
| Baseline | A deliberately simple comparison that a useful model should improve upon. | Predict using only the training churn rate. |
| Prevalence / positive rate | The share of rows whose true label is positive. | 20 positives among 100 rows means 20% prevalence. |
| True positive (TP) | A positive event correctly predicted positive. | Churned and was sent to review. |
| False positive (FP) | A negative event incorrectly predicted positive. | Did not churn but was sent to review. |
| False negative (FN) | A positive event incorrectly predicted negative. | Churned but was not sent to review. |
| True negative (TN) | A negative event correctly predicted negative. | Did not churn and was not sent to review. |
| Confusion matrix | A table containing TP, FP, FN, and TN counts. | It makes the two kinds of mistake visible. |
| Accuracy | The share of all predictions that are correct. | It can look high when almost every row is negative. |
| Precision | Of the rows predicted positive, the share that truly are positive: `TP / (TP + FP)`. | How concentrated the review queue is with real churn. |
| Recall | Of all truly positive rows, the share found: `TP / (TP + FN)`. | How much churn the review queue catches. |
| F1 | A combined precision-and-recall score; it is high only when both are reasonably high. | Useful as one summary, but it does not encode business costs. |
| Average precision (AP) | A ranking summary across many thresholds; higher is better. | Used to compare candidate models on validation data. |
| ROC-AUC | Another ranking measure: how often a random positive ranks above a random negative. | Reported as a diagnostic, not used as the course's primary selector. |
| Calibration | Agreement between predicted probabilities and observed frequencies. | Among many rows scored near `0.30`, roughly 30% should be positive if well calibrated. |
| Brier score | The mean squared difference between probability and 0/1 outcome; lower is better. | Confident wrong probabilities are penalized heavily. |
| Log loss | A probability error measure that strongly penalizes confident wrong predictions; lower is better. | A score near 1.0 for a true negative is costly. |
| Slice | A meaningful subgroup examined separately. | Recall for each `plan_tier`. |

## MLOps and MLflow words

| Term | Plain-language meaning | Course example |
|---|---|---|
| MLOps | Practices for making model work repeatable, reviewable, releasable, and observable—not just fitting a model once. | Data versions, tests, MLflow evidence, a release gate, and monitoring. |
| Experiment | A named collection of related MLflow runs. | All local churn-development runs. |
| Run | One recorded execution with inputs, settings, results, and artifacts. | Training one candidate on one declared dataset. |
| Parameter | A run setting recorded as evidence. | Model family or random seed. |
| Metric | A numeric result recorded for comparison. | Validation average precision. |
| Artifact | A file saved by a run. | A model, manifest, lockfile, or evaluation table. |
| Tracking | Recording experiments and runs so they can be compared and reproduced. | Local MLflow stores metadata in SQLite and files in an artifact directory. |
| Lineage | Links showing where an output came from. | Model version → run → dataset digest and code state. |
| Manifest | A small file describing data files, row counts, time ranges, and fingerprints. | `manifest.json` beside the generated splits. |
| Digest / SHA-256 | A fixed-length fingerprint of bytes. It detects change but is not a copy of the data. | Editing a CSV changes its SHA-256 digest. |
| Logged Model | A model artifact recorded by MLflow with metadata and a model ID/URI. | The fitted logistic pipeline logged during training. |
| Model signature | The expected names and types of model inputs and outputs. | Numeric and categorical input columns plus probability output. |
| Input example | A small representative input saved with a model. | Five training-feature rows used to validate the contract. |
| Model Registry | A catalogue of named models and numbered versions. | `subscription_churn_classifier`, version 1. |
| Model version | One immutable numbered registration of a model artifact. | Version `3` remains version `3`. |
| Alias | A movable human-readable pointer to a model version. | `champion` may point to version 2 today and version 3 later. |
| Release gate | Rules evaluated before a model may be promoted. | Minimum recall and maximum cost on the frozen test. |
| Adopt / reject | The two release-gate outcomes implemented here: approve or do not approve. | A failed cost check produces `reject` without crashing. |
| Inconclusive selection | Development stops because no validation threshold meets the declared constraints; no test gate is run. | The threshold helper reports infeasibility before test access. |
| Drift | A change between reference and current data or scores. | Current monthly fees have shifted upward. |
| Delayed label | A true outcome that becomes known after prediction. | Thirty days must pass before churn outcome quality can be measured. |

## Local tool and Databricks words

| Term | Plain-language meaning | Course example |
|---|---|---|
| Virtual environment | An isolated directory containing one project's Python and packages. | `.venv` created by `make install`. |
| Lockfile | Exact dependency versions used to recreate an environment. | `uv.lock`. |
| Kernel | The Python process that executes notebook cells. | **AAI Local Classification**. |
| SQLite | A database stored in one local file, suitable here for one learner. | The local MLflow backend. |
| Delta table | A versioned table format used for reliable data workloads on Databricks. | The governed replacement for course CSVs. |
| Unity Catalog | Databricks governance for data and AI assets, names, permissions, and lineage. | Catalogued Delta tables and registered models. |
| Three-part name | A Unity Catalog identifier written `catalog.schema.object`. | `dev.ml.subscription_churn_classifier`. |
| Model Serving | A managed Databricks endpoint that serves model predictions. | An online counterpart to the local predictor. |
| Declarative Automation Bundle | Reviewed files describing Databricks jobs/resources and deployment settings. | The cloud counterpart to manually running local commands. |
| Inference table | Governed records of serving requests and responses used for observability. | Model version, time, score, prediction, and permitted request fields. |

Read the [Databricks handoff](databricks-handoff.md) only after the local terms
through model version, alias, inference, and drift are familiar.
