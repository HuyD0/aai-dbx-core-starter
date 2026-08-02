"""Render the deterministic local-classification notebook curriculum."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import black
import nbformat

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT / "notebooks"


def _cell_id(notebook: str, position: int, text: str) -> str:
    digest = hashlib.sha256(f"{notebook}:{position}:{text}".encode()).hexdigest()[:12]
    return f"aai-{position:02d}-{digest}"


def markdown(notebook: str, position: int, text: str):
    return nbformat.v4.new_markdown_cell(
        source=text.strip() + "\n",
        id=_cell_id(notebook, position, text),
    )


def code(notebook: str, position: int, text: str):
    source = text.strip()
    try:
        formatted = black.format_cell(
            source,
            fast=True,
            mode=black.Mode(line_length=88),
        )
    except black.NothingChanged:
        formatted = source
    return nbformat.v4.new_code_cell(
        source=formatted.rstrip() + "\n",
        execution_count=None,
        outputs=[],
        id=_cell_id(notebook, position, text),
    )


def common_imports(extra: str = "") -> str:
    return f"""
import pandas as pd

{extra}
from aai_local_classification.learning import study_root
from aai_local_classification.settings import load_settings

settings = load_settings()
root = study_root()
print(f"Course state: {{root}}")
print(f"Experiment: {{settings.experiment_name}}")
"""


LESSONS: dict[str, list[tuple[str, str]]] = {
    "00_start_here.ipynb": [
        (
            "markdown",
            """
# 00 — Start here: from `fit()` to release evidence

**Objectives**

- See the complete `baseline -> change -> result -> decision` lifecycle.
- Confirm the exact local environment and isolated MLflow topology.
- Separate notebook exploration from reusable production logic under `src/`.

**Prerequisites:** Python 3.11 or 3.12, `uv`, and `make install`. No cloud
credentials, data download, or GPU is used.
""",
        ),
        ("code", common_imports("import mlflow\nimport sklearn")),
        (
            "code",
            """
environment = pd.Series(
    {
        "mlflow": mlflow.__version__,
        "scikit_learn": sklearn.__version__,
        "random_seed": settings.random_seed,
        "dataset": settings.data.dataset_name,
        "positive_label": 1,
        "primary_metric": settings.selection.primary_metric,
    },
    name="value",
)
environment.to_frame()
""",
        ),
        (
            "markdown",
            """
## The evidence chain

A reproducible model is more than fitted coefficients. We need an answer to six
questions: what problem was defined, which data and code were used, what changed,
how it performed on data that did not choose it, what decision the gate permits,
and which exact artifact receives traffic.
""",
        ),
        (
            "code",
            """
stages = pd.DataFrame(
    [
        ("problem", "target, horizon, action, costs, gates"),
        ("baseline", "no-skill result on validation"),
        ("change", "two declared pipelines on the same train/validation data"),
        ("result", "one evaluation of the selected artifact on frozen test"),
        ("decision", "adopt, reject, or inconclusive"),
        ("release", "registered version plus mutable champion alias"),
        ("operate", "reload, infer, observe, and respond"),
    ],
    columns=["stage", "required evidence"],
)
stages
""",
        ),
        (
            "markdown",
            """
### Exercise

Write one sentence describing why “the notebook ran without an exception” is
not a model-quality claim.

**Hint:** successful computation says nothing about leakage, the comparator,
the test boundary, or the action threshold.

**Checkpoint:** you can point to the local SQLite backend, artifact directory,
locked environment, source package, and notebooks, and explain each one's role.

Next: **01_problem_and_data_contract.ipynb**.
""",
        ),
    ],
    "01_problem_and_data_contract.ipynb": [
        (
            "markdown",
            """
# 01 — Problem and data contract

**Objectives**

- Define prediction time, unit, target horizon, positive label, and action.
- Make false-positive and false-negative assumptions visible.
- Review allowed and forbidden features before looking at model scores.

**Prerequisite:** lesson 00.
""",
        ),
        ("code", common_imports()),
        (
            "markdown",
            """
## Prediction contract

At a monthly account snapshot, predict whether a synthetic subscription will
churn within 30 days. `churned_30d = 1` is the positive event. A positive
prediction would send the account to a fictional retention review—not directly
take an action. Missing a churn is assigned five times the illustrative cost of
an unnecessary review.

These are authored learning assumptions. A real team must derive them with the
people who own the action, capacity, risk, and customer impact.
""",
        ),
        (
            "code",
            """
contract = pd.Series(
    {
        "prediction_unit": "one account at one monthly snapshot",
        "prediction_time": "snapshot_date",
        "target": settings.data.target_column,
        "horizon": "30 days after the snapshot",
        "positive_label": 1,
        "primary_selection_metric": settings.selection.primary_metric,
        "false_negative_cost": settings.selection.false_negative_cost,
        "false_positive_cost": settings.selection.false_positive_cost,
    }
)
contract.to_frame("declared value")
""",
        ),
        (
            "code",
            """
pd.DataFrame(
    {
        "role": (
            ["numeric feature"] * len(settings.features.numeric)
            + ["categorical feature"] * len(settings.features.categorical)
            + ["forbidden"] * len(settings.features.forbidden)
        ),
        "column": (
            list(settings.features.numeric)
            + list(settings.features.categorical)
            + list(settings.features.forbidden)
        ),
    }
)
""",
        ),
        (
            "markdown",
            """
`cancellation_reason` and `closed_account_at` occur after the prediction event;
using them would make offline results impressive and the deployed model useless.
Identifiers and timestamps are excluded because they are lineage/context fields,
not learned signals in this contract.

### Exercise

Suppose the retention team can review only 100 accounts per week. Which metric
or threshold constraint would you add, and why?

**Hint:** consider precision among the top `k` scores or a predicted-positive
capacity constraint. Accuracy does not encode capacity.

**Checkpoint:** you can state the positive event, action, horizon, primary
ranking metric, threshold costs, and forbidden information without opening the
test set.

Next: **02_data_quality_and_eda.ipynb**.
""",
        ),
    ],
    "02_data_quality_and_eda.ipynb": [
        (
            "markdown",
            """
# 02 — Data quality and exploratory analysis

**Objectives**

- Generate data from a versioned, deterministic source.
- Check schema, identity uniqueness, label meaning, missingness, and prevalence.
- Inspect training/validation cohort change without opening frozen-test labels.

**Prerequisite:** lesson 01.
""",
        ),
        (
            "code",
            common_imports(
                "from aai_local_classification.contracts import SplitName\n"
                "from aai_local_classification.data import (load_split, prepare_dataset, validate_dataset)\n"
                "from aai_local_classification.tracking import local_paths"
            ),
        ),
        (
            "code",
            """
paths = local_paths(root)
manifest = prepare_dataset(settings, paths.data_root)
pd.DataFrame([item.model_dump(mode="json") for item in manifest.artifacts])
""",
        ),
        (
            "markdown",
            """
The frozen-test file is hashed and its dates/row count are known, but its label
rate is deliberately withheld from the manifest until release evaluation. A
digest detects change; it does not preserve the raw dataset for you.
""",
        ),
        (
            "code",
            """
train = load_split(settings, SplitName.TRAIN, paths.data_root)
validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
pd.DataFrame(
    {
        "train": validate_dataset(train, settings),
        "validation": validate_dataset(validation, settings),
    }
)
""",
        ),
        (
            "code",
            """
missingness = pd.concat(
    {
        "train": train[list(settings.features.model_columns)].isna().mean(),
        "validation": validation[list(settings.features.model_columns)].isna().mean(),
    },
    axis=1,
)
missingness.sort_values("train", ascending=False)
""",
        ),
        (
            "code",
            """
development = pd.concat([train, validation], ignore_index=True)
monthly = development.groupby(development.snapshot_date.dt.to_period("M")).agg(
    rows=(settings.data.target_column, "size"),
    positive_rate=(settings.data.target_column, "mean"),
    average_fee=("monthly_fee", "mean"),
)
monthly
""",
        ),
        (
            "markdown",
            """
### Exercise

Add one check that should fail before training if a category suddenly dominates
validation. Decide whether it is a hard failure or a warning and explain why.

**Hint:** compare normalized value counts, but remember that distribution change
can be legitimate production behavior rather than corrupt data.

**Checkpoint:** the generator is deterministic, feature missingness is within
contract, IDs are unique, both labels exist, and only train/validation labels
have been inspected.

Next: **03_leakage_safe_splits.ipynb**.
""",
        ),
    ],
    "03_leakage_safe_splits.ipynb": [
        (
            "markdown",
            """
# 03 — Leakage-safe time splits and preprocessing

**Objectives**

- Verify temporal and identity separation.
- Watch the feature contract reject a post-outcome field.
- Fit imputation, scaling, and encoding only through a training Pipeline.

**Prerequisite:** lesson 02 has prepared the dataset.
""",
        ),
        (
            "code",
            common_imports(
                "from aai_local_classification.contracts import SplitName\n"
                "from aai_local_classification.data import add_intentional_leakage, load_split, prepare_dataset, validate_feature_contract\n"
                "from aai_local_classification.modeling import build_candidate, candidate_specs, feature_frame\n"
                "from aai_local_classification.tracking import local_paths"
            ),
        ),
        (
            "code",
            """
paths = local_paths(root)
prepare_dataset(settings, paths.data_root)
train = load_split(settings, SplitName.TRAIN, paths.data_root)
validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
pd.DataFrame(
    [
        ("train", train.snapshot_date.min(), train.snapshot_date.max(), len(train)),
        ("validation", validation.snapshot_date.min(), validation.snapshot_date.max(), len(validation)),
    ],
    columns=["split", "first_snapshot", "last_snapshot", "rows"],
)
""",
        ),
        (
            "code",
            """
assert train.snapshot_date.max() < validation.snapshot_date.min()
assert set(train.account_id).isdisjoint(validation.account_id)
print("Time order and account identity separation verified.")
""",
        ),
        (
            "code",
            """
leaked = add_intentional_leakage(train)
unsafe_features = settings.features.model_copy(
    update={"categorical": settings.features.categorical + ("cancellation_reason",)}
)
unsafe_settings = settings.model_copy(update={"features": unsafe_features})
try:
    validate_feature_contract(unsafe_settings)
except ValueError as error:
    print(f"Blocked as intended: {error}")
else:
    raise AssertionError("The leakage demonstration should have been blocked")
assert "cancellation_reason" in leaked
""",
        ),
        (
            "code",
            """
spec = candidate_specs()[0]
pipeline = build_candidate(spec, settings)
x_train = feature_frame(train, settings)
y_train = train[settings.data.target_column]
pipeline.fit(x_train, y_train)
transformed_columns = pipeline.named_steps["preprocess"].get_feature_names_out()
print(f"Declared input columns: {x_train.shape[1]}")
print(f"Post-encoding columns learned from train: {len(transformed_columns)}")
print(transformed_columns[:12])
""",
        ),
        (
            "markdown",
            """
Because the transformers and estimator are one Pipeline, a fit on a training
fold also fits the imputer/scaler/encoder only on that fold. Running preprocessing
once on all rows before a split would leak distribution information.

### Exercise

Why is a random stratified split not automatically “safer” than this time split?

**Hint:** the correct split imitates how the model will encounter unseen data.
If production predicts later cohorts, random mixing can hide temporal change.

**Checkpoint:** no test file was loaded, the prediction-time audit blocks the
teaching leakage field, and all learned preprocessing lives inside the Pipeline.

Next: **04_baseline.ipynb**.
""",
        ),
    ],
    "04_baseline.ipynb": [
        (
            "markdown",
            """
# 04 — Establish the no-skill baseline

**Objectives**

- Log a prior-only classifier before comparing real estimators.
- See why accuracy can reward a useless minority-class model.
- Inspect the run's dataset lineage, parameters, model, and metrics.

**Prerequisite:** lesson 03.
""",
        ),
        (
            "code",
            common_imports(
                "from aai_local_classification.workflow import run_baseline"
            ),
        ),
        ("code", "baseline = run_baseline(settings, root)\nbaseline"),
        (
            "code",
            """
metrics = pd.Series(baseline["metrics"], name="baseline")
metrics[[
    "positive_rate",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "average_precision",
    "roc_auc",
    "cost_per_1000",
]].to_frame()
""",
        ),
        (
            "markdown",
            """
The dummy model predicts no churn at the default threshold. Its accuracy looks
high because most rows are negative, while recall and F1 correctly show that it
finds none of the positive events. Its average precision equals prevalence and
ROC-AUC is 0.5—the ranking has no skill.

### Exercise

If the baseline predicted every account as positive, which metrics would improve
and which operational cost would worsen?

**Hint:** recall becomes 1.0, but precision falls to prevalence and every
negative account becomes a false-positive review.

**Checkpoint:** a candidate must improve ranking and support a useful action
threshold; merely exceeding baseline accuracy is not the experiment goal.

Open the run in `make mlflow-ui`, then continue to
**05_pipeline_and_training.ipynb**.
""",
        ),
    ],
    "05_pipeline_and_training.ipynb": [
        (
            "markdown",
            """
# 05 — Controlled candidate training and MLflow evidence

**Objectives**

- Fit two declared changes on the exact same training rows.
- Compare them on the exact same validation rows.
- Log explicit inputs, model IDs, signatures, input examples, and the lock.

**Prerequisite:** lesson 04.
""",
        ),
        (
            "code",
            common_imports(
                "from aai_local_classification.learning import short_digest\n"
                "from aai_local_classification.workflow import run_candidate_selection"
            ),
        ),
        (
            "code",
            """
selection = run_candidate_selection(settings, root)
comparison = pd.DataFrame(
    [
        {
            "candidate": item.candidate_name,
            "run_id": item.run_id,
            "model_id": item.model_id,
            "validation_average_precision": item.threshold_selection.validation_metrics.average_precision,
            "validation_roc_auc": item.threshold_selection.validation_metrics.roc_auc,
            "validation_brier": item.threshold_selection.validation_metrics.brier_score,
        }
        for item in selection.candidates
    ]
).sort_values("validation_average_precision", ascending=False)
comparison
""",
        ),
        (
            "code",
            """
print(f"Selected candidate: {selection.selected_candidate}")
print(f"Selected model ID: {selection.selected_model_id}")
print(f"Dataset digest: {short_digest(selection.dataset_sha256)}")
print(f"Selection rule: {selection.selection_rule}")
assert selection.primary_metric == "average_precision"
""",
        ),
        (
            "markdown",
            """
The change is controlled: only the estimator family differs. Both candidates
receive the same declared features and train/validation partitions. A
predeclared tolerance prefers the simpler model when average precision is
practically tied, avoiding needless complexity for a tiny validation advantage.
The selected model is a first-class MLflow Logged Model (`models:/<model-id>`),
not a path we guess from the run layout.

The course logs explicitly rather than registering inside `fit()`: exploratory
training should not create registry versions. The model uses a representative
input example, inferred signature, and `skops` serialization; load artifacts
only from trusted sources even with a safer format.

### Exercise

Add a hyperparameter change to one candidate. Which facts must remain fixed for
the comparison to support a causal explanation of the score change?

**Hint:** data/split, feature contract, preprocessing, seed policy, metric
definition, and threshold procedure are experimental controls.

**Checkpoint:** candidate selection used validation average precision only; the
test file was never loaded by the training workflow.

Next: **06_model_selection_and_threshold.ipynb**.
""",
        ),
    ],
    "06_model_selection_and_threshold.ipynb": [
        (
            "markdown",
            """
# 06 — Model selection is not threshold selection

**Objectives**

- Distinguish ranking quality from a binary action rule.
- Reproduce the validation-only cost/constraint threshold search.
- Explain calibration, discrimination, and thresholded decisions separately.

**Prerequisite:** lesson 05 created selection evidence.
""",
        ),
        (
            "code",
            common_imports(
                "import mlflow.sklearn\n"
                "from aai_local_classification.contracts import SplitName\n"
                "from aai_local_classification.data import load_split\n"
                "from aai_local_classification.evaluation import evaluate_probabilities\n"
                "from aai_local_classification.learning import state_exists\n"
                "from aai_local_classification.modeling import feature_frame\n"
                "from aai_local_classification.policy import selection_policy_sha256\n"
                "from aai_local_classification.tracking import configure_mlflow\n"
                "from aai_local_classification.workflow import ensure_prepared, load_selection, run_candidate_selection"
            ),
        ),
        (
            "code",
            """
paths = configure_mlflow(settings, root)
manifest = ensure_prepared(settings, root)
if not state_exists("selection.json"):
    run_candidate_selection(settings, root)
selection = load_selection(root)
if (
    selection.dataset_sha256 != manifest.dataset_sha256
    or selection.selection_policy_sha256 != selection_policy_sha256(settings)
):
    selection = run_candidate_selection(settings, root)
selected = next(item for item in selection.candidates if item.run_id == selection.selected_run_id)
model = mlflow.sklearn.load_model(selection.selected_model_uri)
validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
x_validation = feature_frame(validation, settings)
y_validation = validation[settings.data.target_column]
probability = model.predict_proba(x_validation)[:, list(model.classes_).index(1)]
""",
        ),
        (
            "code",
            """
threshold_rows = []
chosen = selected.threshold_selection
thresholds = sorted(
    {
        round(max(0.01, min(0.99, chosen.threshold + offset)), 3)
        for offset in (-0.04, -0.02, 0.0, 0.02, 0.06, 0.12)
    }
)
for threshold in thresholds:
    metrics = evaluate_probabilities(
        y_validation,
        probability,
        threshold,
        false_negative_cost=settings.selection.false_negative_cost,
        false_positive_cost=settings.selection.false_positive_cost,
    )
    threshold_rows.append(
        {
            "threshold": threshold,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "predicted_positive_rate": metrics.predicted_positive_rate,
            "cost_per_1000": metrics.cost_per_1000,
            "average_precision": metrics.average_precision,
        }
    )
pd.DataFrame(threshold_rows)
""",
        ),
        (
            "code",
            """
pd.Series(
    {
        "selected_candidate": selected.candidate_name,
        "threshold": chosen.threshold,
        "feasible_thresholds": chosen.feasible_threshold_count,
        "validation_precision": chosen.validation_metrics.precision,
        "validation_recall": chosen.validation_metrics.recall,
        "validation_cost_per_1000": chosen.validation_metrics.cost_per_1000,
        "selection_rule": chosen.selection_rule,
    }
).to_frame("value")
""",
        ),
        (
            "markdown",
            """
Average precision and ROC-AUC stay constant as the threshold changes because
they assess ranking. Precision, recall, review volume, and action cost change.
Brier score/log loss assess probability quality and are also threshold-free.
If no threshold satisfies both declared operating constraints, selection stops
as inconclusive; it never silently drops a requirement.

### Exercise

Change only the false-negative cost in a scratch copy of the settings and
predict which direction the chosen threshold should move.

**Hint:** making missed churn more expensive usually favors a lower threshold
and higher recall, at the price of more reviews.

**Checkpoint:** the candidate and threshold are fixed using validation evidence.
No final-test metric has influenced either choice.

Next: **07_frozen_test_gate.ipynb**.
""",
        ),
    ],
    "07_frozen_test_gate.ipynb": [
        (
            "markdown",
            """
# 07 — One frozen-test release gate

**Objectives**

- Evaluate the exact selected Logged Model on the later-time test partition.
- Combine classic MLflow classifier diagnostics with an explicit business gate.
- Make one `adopt`, `reject`, or `inconclusive` decision.

**Prerequisite:** lesson 06 fixed the candidate and threshold.

This function is idempotent for the same selected artifact and policy: rerunning
returns the existing decision rather than repeatedly opening the test. If code,
dependencies, model, or policy differ after this dataset version is consumed,
the gate refuses to reuse it. Create a new frozen-test version before making a
new release claim.
""",
        ),
        (
            "code",
            common_imports(
                "from aai_local_classification.learning import short_digest, state_exists\n"
                "from aai_local_classification.workflow import run_candidate_selection, run_frozen_test_gate"
            ),
        ),
        (
            "code",
            """
if not state_exists("selection.json"):
    selection = run_candidate_selection(settings, root)
else:
    selection = None
decision = run_frozen_test_gate(settings, root, selection)
pd.Series(
    {
        "decision": decision.decision.value,
        "selected_candidate": decision.selected_candidate,
        "selected_run_id": decision.selected_run_id,
        "test_run_id": decision.test_run_id,
        "threshold": decision.threshold,
        "dataset": short_digest(decision.dataset_sha256),
    }
).to_frame("value")
""",
        ),
        (
            "code",
            """
pd.DataFrame(
    [
        {"check": name, "passed": passed}
        for name, passed in decision.checks.model_dump().items()
    ]
)
""",
        ),
        (
            "code",
            """
important = [
    "test_average_precision",
    "test_roc_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_brier_score",
    "test_cost_per_1000",
    "test_maximum_slice_recall_gap",
]
pd.Series({name: decision.metrics[name] for name in important}).to_frame("test value")
""",
        ),
        (
            "markdown",
            """
The native MLflow classic evaluator logs standard classifier diagnostics in the
result run. The explicit gate remains authoritative because it uses the
validation-selected business threshold, declared costs, and operational slices.
This is intentionally not MLflow GenAI evaluation.

### Exercise

Which result should be `inconclusive` rather than `reject`? Give one example
involving missing evidence and one involving statistical uncertainty.

**Hint:** an invalid test extract or a confidence interval spanning the minimum
effect cannot establish that the model is bad—or good.

**Checkpoint:** the exact selected artifact either passed every declared check
or remains unpromoted. The decision links the dataset digest, model/run IDs,
threshold, metrics, and test run.

Next: **08_registry_and_inference.ipynb**.
""",
        ),
    ],
    "08_registry_and_inference.ipynb": [
        (
            "markdown",
            """
# 08 — Registry, alias, and inference contract

**Objectives**

- Register only an adopted exact artifact.
- Attach decision evidence and move the `champion` alias conditionally.
- Reload by alias and apply the versioned threshold to representative inputs.

**Prerequisite:** lesson 07 produced a decision.
""",
        ),
        (
            "code",
            common_imports(
                "from aai_local_classification.contracts import SplitName\n"
                "from aai_local_classification.data import load_split\n"
                "from aai_local_classification.inference import load_champion\n"
                "from aai_local_classification.learning import state_exists\n"
                "from aai_local_classification.tracking import local_paths\n"
                "from aai_local_classification.workflow import ensure_prepared, load_decision, promote_if_approved, run_candidate_selection, run_frozen_test_gate"
            ),
        ),
        (
            "code",
            """
if not state_exists("decision.json"):
    selected = run_candidate_selection(settings, root)
    decision = run_frozen_test_gate(settings, root, selected)
else:
    decision = load_decision(root)
promotion = promote_if_approved(settings, decision, root)
promotion
""",
        ),
        (
            "code",
            """
if not promotion["registered"]:
    raise RuntimeError("The learning model did not pass; inspect lesson 07 instead of forcing registration.")
predictor = load_champion(settings, root)
paths = local_paths(root)
ensure_prepared(settings, root)
validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
predictions = predictor.predict(validation.head(8), settings)
pd.concat([validation[["account_id"]].head(8), predictions], axis=1)
""",
        ),
        (
            "code",
            """
assert predictor.model_version == str(promotion["model_version"])
assert predictor.threshold == decision.threshold
assert predictions["churn_probability"].between(0, 1).all()
print("Alias resolution, concrete version, threshold, and output schema agree.")
""",
        ),
        (
            "markdown",
            """
`champion` is a mutable pointer, while the resolved version is concrete evidence.
Record that version in every batch/deployment. The threshold is stored both in
the logged model metadata and the registered version tags; the inference helper
uses the approved version tag and tests parity with decision evidence.

### Exercise

Why should an online endpoint update to a concrete resolved version instead of
assuming that moving an alias changes a running deployment?

**Hint:** discovery/promotion state and deployment configuration have different
lifecycles, permissions, and rollback behavior.

**Checkpoint:** registration is conditional, the model has a signature and input
example, and a fresh loader reproduces bounded probabilities and binary actions.
Promotion also rechecks the dataset, selected run/model ID, policy digests, and
threshold before moving the alias.

Next: **09_monitoring_and_databricks.ipynb**.
""",
        ),
    ],
    "09_monitoring_and_databricks.ipynb": [
        (
            "markdown",
            """
# 09 — Monitoring simulation and Databricks handoff

**Objectives**

- Detect schema, missingness, numeric, and categorical input change.
- Separate data/score drift from delayed-label performance evidence.
- Map every local artifact to a governed Databricks counterpart.

**Prerequisite:** lesson 08 registered the adopted artifact.
""",
        ),
        (
            "code",
            common_imports(
                "from aai_local_classification.contracts import SplitName\n"
                "from aai_local_classification.data import load_split\n"
                "from aai_local_classification.inference import load_champion\n"
                "from aai_local_classification.learning import state_exists\n"
                "from aai_local_classification.monitoring import compare_batches, shifted_batch\n"
                "from aai_local_classification.tracking import local_paths\n"
                "from aai_local_classification.workflow import ensure_prepared, promote_if_approved, run_candidate_selection, run_frozen_test_gate"
            ),
        ),
        (
            "code",
            """
if not state_exists("promotion.json"):
    selected = run_candidate_selection(settings, root)
    decision = run_frozen_test_gate(settings, root, selected)
    promote_if_approved(settings, decision, root, selected)
paths = local_paths(root)
ensure_prepared(settings, root)
reference = load_split(settings, SplitName.VALIDATION, paths.data_root)
current = shifted_batch(reference, settings.random_seed + 1)
report = compare_batches(reference, current, settings)
pd.Series(
    {
        "maximum_numeric_psi": report.maximum_numeric_psi,
        "maximum_categorical_total_variation": report.maximum_categorical_total_variation,
        "largest_missing_rate_delta": max(abs(v) for v in report.missing_rate_delta.values()),
    }
).to_frame("diagnostic")
""",
        ),
        (
            "code",
            """
pd.DataFrame(
    {
        "numeric_psi": pd.Series(report.numeric_psi),
        "missing_rate_delta": pd.Series(report.missing_rate_delta),
    }
).sort_values("numeric_psi", ascending=False)
""",
        ),
        (
            "code",
            """
predictor = load_champion(settings, root)
reference_scores = predictor.predict(reference, settings)
current_scores = predictor.predict(current, settings)
pd.Series(
    {
        "reference_predicted_positive_rate": reference_scores.churn_prediction.mean(),
        "current_predicted_positive_rate": current_scores.churn_prediction.mean(),
        "reference_mean_score": reference_scores.churn_probability.mean(),
        "current_mean_score": current_scores.churn_probability.mean(),
    }
).to_frame("value")
""",
        ),
        (
            "markdown",
            """
The current batch is explicitly simulated. PSI and total variation are
diagnostics with context-dependent thresholds, not universal tests. A shifted
score or input distribution warrants investigation; it does not prove recall or
calibration changed. Those require correctly joined delayed outcomes.

## Local to governed platform

| Here | Databricks |
|---|---|
| CSV + digest | versioned Unity Catalog Delta table |
| SQLite MLflow | hosted MLflow tracking |
| local model name | `<catalog>.<schema>.<model>` |
| local `champion` | Models in Unity Catalog alias |
| Python/Make execution | packaged job in a Declarative Automation Bundle |
| local predictor | batch inference or Model Serving at a concrete version |
| simulated drift report | AI Gateway inference table + governed data profiling and delayed labels |

Read [the complete handoff](../docs/databricks-handoff.md) before adapting the
project. Moving to cloud does not authorize infrastructure creation or secrets;
use the approved keyless identity and external platform process.

### Exercise

Design one alert for service health, one for data health, and one for outcome
quality. For each, name an owner and a safe action.

**Hint:** an alert without a response owner and playbook is telemetry, not an
operating control.

**Checkpoint:** you can trace one prediction back to a concrete model
version, threshold, signature, dataset version, training/selection/test runs,
source state, dependency lock, release decision, and monitoring plan.
""",
        ),
    ],
}


def render_lesson(filename: str, cells: list[tuple[str, str]]) -> str:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        (
            markdown(filename, index, text)
            if kind == "markdown"
            else code(filename, index, text)
        )
        for index, (kind, text) in enumerate(cells)
    ]
    notebook.metadata = {
        "kernelspec": {
            "display_name": "AAI Local Classification",
            "language": "python",
            "name": "aai-local-classification",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "aai_course": {
            "schema_version": 1,
            "network_required": False,
            "cloud_credentials_required": False,
        },
    }
    return nbformat.writes(notebook, version=4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for filename, cells in LESSONS.items():
        rendered = render_lesson(filename, cells)
        path = NOTEBOOKS / filename
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                failures.append(filename)
        else:
            path.write_text(rendered, encoding="utf-8")
    if failures:
        raise SystemExit(f"Notebook sources are stale: {', '.join(failures)}")
    print(
        f"Verified {len(LESSONS)} notebooks"
        if args.check
        else f"Rendered {len(LESSONS)} notebooks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
