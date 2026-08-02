"""Teaching content for the local classification notebooks.

This file is deliberately plain Python data. ``render_notebooks.py`` turns the
cells below into deterministic, output-free notebooks with stable cell IDs.
"""

from __future__ import annotations

Cell = tuple[str, str]


def m(text: str) -> Cell:
    return ("markdown", text)


def c(text: str) -> Cell:
    return ("code", text)


def preflight() -> str:
    return r"""
import importlib.util
import sys

required = ("mlflow", "pandas", "sklearn", "aai_local_classification")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError(
        "This notebook is using the wrong Python kernel. Close this Jupyter "
        "server, run `make notebook` from examples/local-classification, or "
        "select the 'AAI Local Classification' kernel. Missing: "
        + ", ".join(missing)
    )

import pandas as pd

from aai_local_classification.learning import study_root
from aai_local_classification.settings import load_settings
from aai_local_classification.tracking import local_paths

settings = load_settings()
root = study_root()
paths = local_paths(root)
print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}: {sys.executable}")
print("✓ Course imports are available")
print(f"✓ Learner state: {root}")
"""


LESSONS: dict[str, list[Cell]] = {
    "00_start_here.ipynb": [
        m("""
# 00 — Start here: make one prediction understandable

**Plain-language question:** Can my Mac run the course, and what will the model
eventually do?

**Why this matters:** before training anything, you should know how to run a
notebook safely and what question a classification model answers.

**Estimated time:** 25–35 minutes.
**Prerequisite:** you can assign a Python variable and you launched Jupyter with
`make notebook`. No ML or MLflow knowledge is assumed.

This course uses fictional subscription accounts. Eventually, the model will
estimate the chance that an account cancels in the 30 days after a monthly
snapshot. Nothing here contacts Databricks or acts on a real customer.
"""),
        m("""
## How a notebook works

A notebook contains two kinds of cells:

- A **Markdown cell** is explanatory text like this.
- A **code cell** runs Python and shows its result underneath.

Click the next code cell and press **Shift+Enter**. The number at its left tells
you when it ran. `[*]` means it is still running. A **kernel** is the Python
process that remembers variables between cells.

Run cells from top to bottom. If results ever seem impossible, choose
**Kernel → Restart Kernel and Run All Cells**. If setup fails, return to a
Terminal in `examples/local-classification` and run `make doctor`.

### Words introduced

| Word | Plain meaning | Example here |
|---|---|---|
| notebook | A document mixing explanation, Python, and results | This lesson |
| kernel | The Python process executing cells | `AAI Local Classification` |
| classification | Choosing between named outcomes | churn `1` or no churn `0` |
"""),
        m("""
## Preflight

**Before you run this:** predict which three things a useful environment check
should report. Then run the cell.
"""),
        c(preflight()),
        m("""
### What you should see

Three lines beginning with `✓`: a Python 3.11 or 3.12 executable inside this
course's `.venv`, available imports, and a learner-state directory ending in
`.aai/course-v2`.

If you see those lines, the notebook is using the supported environment. That
proves only that the tools run—not that a future model is accurate or useful.
"""),
        m("""
## Meet the data before the model

The next cell creates the same small synthetic dataset every time. It then loads
only the **training** rows. Later lessons explain why validation and test rows
have different jobs.

**Before you run this:** the configured course has 18 months of training data
and 120 rows per month. Predict the row count.
"""),
        c("""
from aai_local_classification.contracts import SplitName
from aai_local_classification.data import load_split
from aai_local_classification.workflow import ensure_prepared

manifest = ensure_prepared(settings, root)
train = load_split(settings, SplitName.TRAIN, paths.data_root)
print(f"Training shape: {train.shape}")
print(
    "Possible labels:",
    sorted(int(value) for value in train.churned_30d.unique()),
)
"""),
        m("""
### What you should see

`Training shape: (2160, 12)` and labels `[0, 1]`. There are 2,160 examples and
12 stored columns. Not all 12 columns will be model inputs.

### Words introduced

| Word | Plain meaning | Concrete example |
|---|---|---|
| row / example | One case the model can learn from | one account snapshot |
| feature | A fact available when predicting | monthly fee |
| label / target | The answer learned later | `churned_30d` |
"""),
        m("""
Now look at three rows. The columns on the left describe what was known at the
snapshot. The final column is the later answer.
"""),
        c("""
visible_columns = [
    "account_id",
    "snapshot_date",
    "monthly_fee",
    "contract_type",
    "autopay",
    "churned_30d",
]
train.loc[:2, visible_columns]
"""),
        m("""
### How to read the output

Each row asks: “Using facts known on `snapshot_date`, did this fictional account
churn during the next 30 days?” A model learns a pattern across many rows; it
does not memorize a rule from the three rows shown here.
"""),
        m("""
## From a probability to a yes/no prediction

A classifier can produce a score between 0 and 1. We choose a **threshold** to
turn that score into an action. This tiny example is not a trained model; it
only makes the final operation visible.

**Before you run this:** with a threshold of `0.40`, predict which scores become
`1`.
"""),
        c("""
toy = pd.DataFrame({"churn_probability": [0.08, 0.41, 0.76]})
toy_threshold = 0.40
toy["prediction"] = (toy.churn_probability >= toy_threshold).astype(int)
toy
"""),
        m("""
### What you should see

`0.08` becomes `0`; `0.41` and `0.76` become `1`. A probability and a binary
prediction are different objects. Lesson 06 will choose a threshold using
validation data and explicit error costs.

### Misconception check

“The notebook ran” is an environment claim. It is not evidence that the data is
appropriate, the model beats a baseline, or the model should be released.
"""),
        m("""
## The course roadmap

Each lesson answers one question:

1. What decision are we supporting?
2. Is the data trustworthy?
3. How do we avoid learning from the future?
4. What does “better” mean for an uncommon event?
5. How does a real training pipeline work?
6. Which model and threshold should we choose?
7. Did the fixed choice pass an honest final check?
8. What exact artifact gets released?
9. How do we notice changed behavior, and what maps to Databricks?

MLflow appears only after you understand the thing it records.
"""),
        m("""
### Guided exercise

Find the minimum and maximum monthly fee observed in training. The starter cell
already selects the column; replace the two method calls if you want to explore.
"""),
        c("""
exercise_fee_range = train["monthly_fee"].agg(["min", "max"])
exercise_fee_range
"""),
        m("""
**Self-check:** the minimum must be smaller than the maximum, and both must be
positive. This describes the observed synthetic training rows; it does not set
valid limits for future data.

<details><summary>Solution explanation</summary>

`Series.agg(["min", "max"])` applies both summaries to one column. A production
quality rule would come from a reviewed data contract, not merely these extrema.
</details>
"""),
        c("""
# Reference solution — run after your attempt
assert 0 < exercise_fee_range["min"] < exercise_fee_range["max"]
print("✓ Fee range is ordered and positive")
"""),
        m("""
## Recap

- A kernel runs code and remembers state; top-to-bottom execution matters.
- Classification predicts a named outcome; here the label is churn `1` or no
  churn `0`.
- A score needs a threshold before it becomes a binary action.

**Evidence created:** deterministic CSV files and a manifest under the printed
learner-state directory. Rerunning this lesson reuses them after checking that
their fingerprints still match the course.

**Ready for 01?** You can explain the difference between a feature, a label, a
probability, and a prediction.
"""),
    ],
    "01_problem_and_data_contract.ipynb": [
        m("""
# 01 — Frame the prediction before training

**Plain-language question:** What decision are we trying to support, and what
may the model know at prediction time?

**Why this matters:** a technically good model can still solve the wrong
problem. We define the question, timing, action, and errors before comparing
algorithms.

**Estimated time:** 35–45 minutes.
**Prerequisite:** lesson 00; you know row, feature, label, probability, and
threshold.
"""),
        m("""
## Preflight

Run this first. It checks the kernel and shows where this lesson reads and
writes course state.
"""),
        c(preflight()),
        c("""
from aai_local_classification.contracts import SplitName
from aai_local_classification.data import load_split
from aai_local_classification.workflow import ensure_prepared

ensure_prepared(settings, root)
train = load_split(settings, SplitName.TRAIN, paths.data_root)
print(f"✓ Loaded {len(train):,} training rows")
"""),
        m("""
### What you should see

The environment checks and `✓ Loaded 2,160 training rows`. If this notebook was
opened directly, the setup cell safely created the deterministic prerequisite
data rather than training a hidden model.
"""),
        m("""
## Put the question on a timeline

```text
facts available now                 answer learned later
monthly snapshot ── predict ── optional review ── churn within 30 days?
        ↑ prediction time                         ↑ prediction horizon
```

### Words introduced

| Word | Plain meaning | Here |
|---|---|---|
| prediction time | The instant inputs must be available | monthly snapshot |
| prediction horizon | How far ahead the answer covers | the next 30 days |
| positive class | The event represented by `1` | churn |

The fictional action is to send a positive prediction to a human retention
review. The model does not contact a customer and does not make an autonomous
decision.
"""),
        m("""
Look at one example vertically. `churned_30d` is shown so we can learn from
historical training data, but it cannot be an input when making the prediction.
"""),
        c("""
example = train.loc[0, [*settings.features.model_columns, "churned_30d"]]
example.to_frame("value")
"""),
        m("""
### How to interpret what you should notice

Numbers such as `monthly_fee` and categories such as `contract_type` describe
the account. The label is the later answer. A data contract prevents code from
quietly moving the answer—or facts created after the answer—into the feature
list.
"""),
        m("""
## Which columns play which role?

A **numeric feature** has meaningful numeric magnitude. A **categorical
feature** names a group. A context field helps identify or order a row without
being learned directly. A forbidden field is unavailable or unsafe at
prediction time.
"""),
        c("""
roles = pd.DataFrame(
    [
        ("tenure_months", "numeric feature", "months already subscribed"),
        ("monthly_fee", "numeric feature", "current monthly charge"),
        ("contract_type", "categorical feature", "contract arrangement"),
        ("autopay", "categorical feature", "automatic payment enabled"),
        ("account_id", "context", "row identifier, not a learned signal"),
        ("snapshot_date", "context", "defines prediction time and split"),
        ("churned_30d", "target", "answer during the next 30 days"),
        ("cancellation_reason", "forbidden", "only known after cancellation"),
    ],
    columns=["column", "role", "meaning"],
)
roles
"""),
        m("""
The full reviewed feature lists live in `configs/project.yaml`. Raw IDs are
usually poor model inputs. Dates can support legitimate prediction-time
features (for example, month of year), but this course deliberately uses the
raw snapshot date only for lineage and splitting.
"""),
        m("""
## How uncommon is the positive class?

**Before you run this:** if churn is uncommon, which count do you expect to be
larger: `0` or `1`?
"""),
        c("""
label_counts = train.churned_30d.value_counts().sort_index()
label_summary = pd.DataFrame(
    {
        "meaning": ["no churn", "churn"],
        "rows": [label_counts[0], label_counts[1]],
        "share": [label_counts[0] / len(train), label_counts[1] / len(train)],
    },
    index=pd.Index([0, 1], name="label"),
)
label_summary
"""),
        m("""
### What you should see

1,833 non-churn rows and 327 churn rows: about 15.1% are positive. This
**prevalence** matters because a model that always says “no churn” would look
about 84.9% accurate while missing every churn. Lesson 04 makes that failure
visible with a confusion matrix.
"""),
        m("""
## Name the two error types before assigning costs

### Words introduced

| Error | Plain meaning | Fictional consequence |
|---|---|---|
| false positive | predict churn, but no churn occurs | unnecessary review |
| false negative | predict no churn, but churn occurs | missed review opportunity |
| action cost | a teaching weight used to compare those errors | FP = 1, FN = 5 |

The numbers are units for a worked example—not dollars and not researched
business impact. A real team would agree them with decision owners, capacity
owners, risk specialists, and affected users.
"""),
        c("""
cost_example = pd.DataFrame(
    {
        "false_positives": [20, 0],
        "false_negatives": [0, 20],
    },
    index=["20 unnecessary reviews", "20 missed churns"],
)
cost_example["teaching_cost"] = (
    cost_example.false_positives * settings.selection.false_positive_cost
    + cost_example.false_negatives * settings.selection.false_negative_cost
)
cost_example
"""),
        m("""
### What you should see

Twenty false positives cost 20 teaching units; twenty false negatives cost 100.
This makes recall important and will usually move the chosen threshold below
0.5. It does not mean “predict everyone positive”: reviews still have a cost.

### Misconception check

The target is never a prediction-time input. A column being stored in the same
historical table does not make it available when the real prediction occurs.
"""),
        m("""
### Guided exercise

Classify `cancellation_reason` as `allowed` or `forbidden`, then explain why.
Change the starter value if needed.
"""),
        c("""
exercise_role = "forbidden"
exercise_reason = "It is only known after the customer has cancelled."
pd.Series({"role": exercise_role, "reason": exercise_reason})
"""),
        m("""
**Self-check:** a feature must exist at the monthly snapshot. If knowing its
value requires waiting for the outcome, it leaks the answer.

<details><summary>Solution explanation</summary>

`cancellation_reason` is forbidden. It describes an event that happens after
the prediction time, so it would make historical scores unrealistically good.
</details>
"""),
        c("""
# Reference solution — run after your attempt
assert exercise_role == "forbidden"
assert "after" in exercise_reason.lower()
print("✓ The post-outcome field is excluded")
"""),
        m("""
## MLOps bridge

The prediction contract belongs in reviewed configuration and run evidence so
that “good score” cannot silently replace “right decision.” MLflow will later
record the target, positive label, costs, and selection policy beside each run.

## Recap

- We predict churn within 30 days using only facts available at the snapshot.
- The positive class is uncommon, and the two error types have different
  illustrative costs.
- Context, target, and post-outcome fields stay outside the model features.

**Evidence created:** none beyond lesson 00's deterministic data. This lesson
reads the authored contract from `configs/project.yaml`.

**Ready for 02?** You can state the prediction time, target, positive class,
fictional action, and why `cancellation_reason` is forbidden.
"""),
    ],
    "02_data_quality_and_eda.ipynb": [
        m("""
# 02 — Inspect and validate the data before fitting

**Plain-language question:** Is the data trustworthy enough to train on?

**Why this matters:** an algorithm cannot repair a missing column, duplicated
identity, impossible label, or unexplained change in the dataset.

**Estimated time:** 45–55 minutes.
**Prerequisite:** lessons 00–01; you know rows, features, labels, and prediction
time.
"""),
        m("## Preflight\n\nRun the environment check before touching the data."),
        c(preflight()),
        c("""
from aai_local_classification.contracts import SplitName
from aai_local_classification.data import (
    load_split,
    validate_dataset,
)
from aai_local_classification.workflow import ensure_prepared

manifest = ensure_prepared(settings, root)
train = load_split(settings, SplitName.TRAIN, paths.data_root)
validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
print(f"✓ Train {train.shape}; validation {validation.shape}")
"""),
        m("""
### What you should see

Train `(2160, 12)` and validation `(360, 12)`. We do not load test data in this
lesson. The later final exam remains sealed.

### Words introduced

| Word | Plain meaning | Example check |
|---|---|---|
| schema | Expected column names and data types | `monthly_fee` is numeric |
| missing value | An unknown entry, represented as `NaN`/`<NA>` | missing usage |
| duplicate | A row identity appearing more than once | repeated `account_id` |
"""),
        m("## Start with shape, sample rows, and data types"),
        c("""
train.head(3)
"""),
        c("""
schema_view = pd.DataFrame(
    {
        "dtype": train.dtypes.astype(str),
        "missing_rows": train.isna().sum(),
        "distinct_values": train.nunique(dropna=True),
    }
)
schema_view
"""),
        m("""
### How to interpret this

The table is a first inspection, not proof of quality. Numeric features should
load as numbers, the snapshot should parse as a date, and only the two columns
designed with missingness should have missing rows. A type can be technically
valid yet semantically wrong, so meanings still require a data contract.
"""),
        m("""
## Make important checks visible

**Before you run this:** predict the correct duplicate count and allowed label
set for this course.
"""),
        c("""
visible_checks = pd.Series(
    {
        "duplicate_account_ids": int(train.account_id.duplicated().sum()),
        "missing_account_ids": int(train.account_id.isna().sum()),
        "invalid_snapshot_dates": int(
            pd.to_datetime(train.snapshot_date, errors="coerce").isna().sum()
        ),
        "label_values": str(sorted(train.churned_30d.unique())),
        "maximum_feature_missing_rate": train[
            list(settings.features.model_columns)
        ].isna().mean().max(),
    },
    name="observed",
)
visible_checks.to_frame()
"""),
        m("""
### What you should see

Zero duplicate/missing IDs, zero invalid dates, labels `[0, 1]`, and maximum
feature missingness around 3%. The course contract permits at most 10%.

These checks fail fast because training on structurally unexpected data would
create misleading evidence.
"""),
        m("""
## Look for change across time

**Prevalence** is the share of rows whose label is positive. It can change even
when the schema stays identical. We can inspect training and validation because
both may guide development; test stays unopened.
"""),
        c("""
cohort_summary = pd.DataFrame(
    {
        "rows": [len(train), len(validation)],
        "positive_rate": [train.churned_30d.mean(), validation.churned_30d.mean()],
        "usage_missing_rate": [
            train.usage_hours_30d.isna().mean(),
            validation.usage_hours_30d.isna().mean(),
        ],
        "channel_missing_rate": [
            train.signup_channel.isna().mean(),
            validation.signup_channel.isna().mean(),
        ],
    },
    index=["train", "validation"],
)
cohort_summary
"""),
        m("""
### What you should see

Training prevalence is about 15.1%; validation is about 19.4%. A later cohort
already looks different. That observation motivates a time-based split; it does
not by itself explain why churn changed.
"""),
        c("""
import matplotlib.pyplot as plt

monthly = train.assign(month=train.snapshot_date.dt.to_period("M").astype(str))
monthly_rate = monthly.groupby("month").churned_30d.mean()
ax = monthly_rate.plot(figsize=(9, 3), marker="o", title="Training churn rate by month")
ax.set_ylabel("share with churned_30d = 1")
ax.set_xlabel("snapshot month")
plt.xticks(rotation=45)
plt.tight_layout()
"""),
        m("""
The plot is **exploratory data analysis (EDA)**: a way to notice patterns worth
investigating. Association is not causation. A higher monthly rate does not tell
us which feature caused it or whether it will continue.
"""),
        m("""
## Version the exact dataset

### Words introduced

| Word | Plain meaning | Here |
|---|---|---|
| manifest | A small inventory describing data files | `manifest.json` |
| digest | A fingerprint that changes when bytes change | SHA-256 text |
| lineage | Evidence of where an input came from | generator + config + split |

A digest detects change; it does not contain the data, prove correctness, or
make the data representative.
"""),
        c("""
manifest_table = pd.DataFrame(
    [
        {
            "split": item.split.value,
            "rows": item.row_count,
            "dates": f"{item.start_date} to {item.end_date}",
            "positive_rate": item.positive_rate,
            "sha256": item.sha256[:12] + "…",
        }
        for item in manifest.artifacts
    ]
)
manifest_table
"""),
        m("""
Notice that the test row count and dates are visible, but its `positive_rate` is
blank. The manifest lets us verify the sealed file without exposing its label
summary during development.
"""),
        c("""
import hashlib

first = hashlib.sha256(b"same text").hexdigest()[:12]
second = hashlib.sha256(b"same text!").hexdigest()[:12]
pd.Series({"same text": first, "one-character change": second})
"""),
        m("""
### What you should see

The two short fingerprints differ completely. The course stores full 64-character
digests; we shorten them only for display.

Now compare our visible checks with the packaged validator used by repeatable
jobs.
"""),
        c("""
packaged_quality = validate_dataset(train, settings)
pd.Series(packaged_quality, name="observed").to_frame()
"""),
        m("""
The helper repeats enforceable schema and quality rules. We inspected the
important operations first, so the function is now a reusable safety boundary
rather than a black box.

### Misconception check

Passing these checks means “matches this authored contract.” It does not mean
the synthetic sample represents real customers, is fair, or supports a useful
business decision.
"""),
        m("""
### Guided exercise

Duplicate one row in a scratch DataFrame. Do not alter `train`. Predict which
quality rule should reject the scratch copy.
"""),
        c("""
exercise_with_duplicate = pd.concat([train, train.iloc[[0]]], ignore_index=True)
exercise_duplicate_count = int(
    exercise_with_duplicate.account_id.duplicated().sum()
)
print(f"Duplicate IDs in scratch data: {exercise_duplicate_count}")
"""),
        m("""
**Self-check:** the scratch copy should contain exactly one duplicated ID. The
solution catches the expected validation error so Restart-and-Run-All still
finishes successfully.

<details><summary>Solution explanation</summary>

The validator rejects duplicated account identities before any model fit. The
original `train` object and persisted data remain unchanged.
</details>
"""),
        c("""
# Reference solution — run after your attempt
assert exercise_duplicate_count == 1
try:
    validate_dataset(exercise_with_duplicate, settings)
except ValueError as error:
    print(f"✓ Expected rejection: {error}")
else:
    raise AssertionError("The duplicate should have been rejected")
"""),
        m("""
## MLOps bridge

Later MLflow runs log the dataset input and manifest digest. On Databricks, the
equivalent source would normally be a versioned Unity Catalog Delta table rather
than local CSV files.

## Recap

- Inspect shape, rows, types, missingness, identities, labels, and time cohorts
  before fitting.
- A manifest and digest make change detectable and traceable.
- Quality rules are necessary boundaries, not proof of real-world usefulness.

**Evidence created:** the same manifest and split files from lesson 00; no model
has been trained. Reruns verify rather than silently replace them.

**Ready for 03?** You can explain why validation may be inspected while the
frozen test label summary remains sealed.
"""),
    ],
    "03_leakage_safe_splits.ipynb": [
        m("""
# 03 — Learn from the past without learning from the future

**Plain-language question:** Which rows may teach and choose the model, and
which rows must remain untouched?

**Why this matters:** using future information can make an offline model look
excellent even though that information will not exist when predictions are
needed.

**Estimated time:** 40–50 minutes.
**Prerequisite:** lessons 00–02; you understand prediction time, labels, data
quality checks, and time cohorts.
"""),
        m(
            "## Preflight\n\nRun the environment check, then load development data only."
        ),
        c(preflight()),
        c("""
from aai_local_classification.contracts import SplitName
from aai_local_classification.data import (
    add_intentional_leakage,
    load_split,
)
from aai_local_classification.modeling import feature_frame
from aai_local_classification.workflow import ensure_prepared

manifest = ensure_prepared(settings, root)
train = load_split(settings, SplitName.TRAIN, paths.data_root)
validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
print("✓ Loaded train and validation; test labels remain unopened")
"""),
        m("""
### What you should see

A success message naming only train and validation. The manifest can describe
the test file without loading its rows.

### Words introduced

| Word | Plain meaning | Single job in this course |
|---|---|---|
| training split | Rows used to learn model parameters | fit preprocessing/model |
| validation split | Later rows used to choose among fixed options | model + threshold choice |
| frozen test split | Still-later rows reserved as a final exam | one release decision |
"""),
        m("""
## See the time boundaries before writing model code

```text
Jan 2023 ───────── Jun 2024 | Jul ─ Sep 2024 | Oct ─ Dec 2024
          TRAIN             |   VALIDATION   |   FROZEN TEST
      learn parameters      | choose once    | judge fixed choice
```

**Before you run this:** predict whether any split's dates should overlap.
"""),
        c("""
split_calendar = pd.DataFrame(
    [
        {
            "split": item.split.value,
            "rows": item.row_count,
            "start": item.start_date,
            "end": item.end_date,
            "label_rate_visible": item.positive_rate is not None,
        }
        for item in manifest.artifacts
    ]
)
split_calendar
"""),
        m("""
### How to interpret the output

Training has 2,160 rows, validation 360, and test 360. Dates are ordered and do
not overlap. The final row says the test label rate is not visible. Think of the
test split as a sealed final exam, not extra practice questions.
"""),
        c("""
ordered_boundaries = pd.Series(
    {
        "train_ends_before_validation": (
            train.snapshot_date.max() < validation.snapshot_date.min()
        ),
        "validation_ends_before_test": (
            validation.snapshot_date.max()
            < pd.Timestamp(settings.data.test_start)
        ),
    },
    name="passed",
)
ordered_boundaries.to_frame()
"""),
        m("""
### What you should see

Both checks are `True`. A random split is not automatically safer: the split
should imitate how the model will encounter future data. This generator has one
snapshot per synthetic account; a real dataset with repeated accounts would
also need an entity-aware design.
"""),
        m("""
## Make the model inputs and labels explicit

Uppercase `X` conventionally means the feature table; lowercase `y` means the
target series. They are ordinary pandas objects.

**Before you run this:** there are 9 declared model features. Predict the two
shapes.
"""),
        c("""
X_train = feature_frame(train, settings)
y_train = train[settings.data.target_column]
X_validation = feature_frame(validation, settings)
y_validation = validation[settings.data.target_column]

pd.Series(
    {
        "X_train": X_train.shape,
        "y_train": y_train.shape,
        "X_validation": X_validation.shape,
        "y_validation": y_validation.shape,
    }
).to_frame("shape")
"""),
        m("""
### What you should see

`X_train` is `(2160, 9)` while `y_train` is `(2160,)`; validation has 360 rows
in both objects. `account_id`, `snapshot_date`, and `churned_30d` are absent
from `X` by contract.
"""),
        m("""
## Leakage: when the answer sneaks into the inputs

### Words introduced

| Word | Plain meaning | Example |
|---|---|---|
| target leakage | An input directly reveals the later answer | cancellation reason |
| temporal leakage | Training uses information from after prediction time | future support tickets |
| preprocessing leakage | A cleanup step learns from validation/test | scaling on all rows |

The next function creates a teaching-only column. It does not alter persisted
data.
"""),
        c("""
leaked_train = add_intentional_leakage(train)
pd.crosstab(
    leaked_train.cancellation_reason,
    leaked_train.churned_30d,
    rownames=["post-outcome value"],
    colnames=["true label"],
)
"""),
        m("""
### How to interpret the output

`account_closed` occurs only when the label is `1`; `not_applicable` occurs only
when it is `0`. A model could appear perfect by reading the answer after it
happened. At real prediction time the column would not exist, so the model would
fail operationally.
"""),
        m("""
The configuration explicitly lists forbidden fields. This readable rule runs
before any expensive fit.
"""),
        c("""
forbidden_attempt = "cancellation_reason"
contract_result = pd.Series(
    {
        "attempted_feature": forbidden_attempt,
        "listed_as_forbidden": forbidden_attempt in settings.features.forbidden,
        "allowed_to_train": forbidden_attempt
        not in settings.features.forbidden,
    }
)
contract_result.to_frame("value")
"""),
        m("""
### What you should see

The attempted field is listed as forbidden and `allowed_to_train` is `False`.
Packaged training also calls `validate_feature_contract`, so a source change
cannot silently bypass this display.

### Misconception check

“The column improves validation” is not enough. First ask whether its value
exists, with the same meaning, at the exact prediction time.
"""),
        m("""
### Guided exercise

Classify “support tickets opened during the 30 days after the snapshot.” Is it
available or forbidden for this prediction? Change the starter answer if needed.
"""),
        c("""
exercise_feature_timing = "forbidden"
exercise_explanation = "Those tickets happen after prediction time."
pd.Series(
    {
        "classification": exercise_feature_timing,
        "explanation": exercise_explanation,
    }
)
"""),
        m("""
**Self-check:** only tickets observed before the monthly snapshot could be
eligible. A similarly named 90-day history feature is allowed because its
window ends at the snapshot.

<details><summary>Solution explanation</summary>

The future 30-day ticket count is forbidden temporal leakage. The existing
`support_tickets_90d` feature is a backward-looking value available now.
</details>
"""),
        c("""
# Reference solution — run after your attempt
assert exercise_feature_timing == "forbidden"
assert "after" in exercise_explanation.lower()
print("✓ Future-window support data is excluded")
"""),
        m("""
## MLOps bridge

The exact date predicates and dataset fingerprints become run evidence. On
Databricks, a reviewed job would read versioned Delta data using the same time
boundaries; changing model or policy after a test result requires a new frozen
test version.

## Recap

- Train learns parameters, validation chooses, and frozen test judges the fixed
  choice once.
- A production-like time split is more meaningful than a convenient random
  split for this scenario.
- Feature availability is evaluated at prediction time; future facts are
  leakage even when present in historical storage.

**Evidence created:** explicit `X_train`, `y_train`, `X_validation`, and
`y_validation` in memory only. No test rows were loaded and no model was fit.

**Ready for 04?** You can give one sentence for each split's job and identify
the three leakage types above.
"""),
    ],
    "04_baseline.ipynb": [
        m("""
# 04 — Build a baseline and make classification metrics concrete

**Plain-language question:** Can a model look accurate while finding no
churners?

**Why this matters:** “85% accurate” sounds impressive until you compare it
with a rule that always predicts the common class.

**Estimated time:** 50–60 minutes.
**Prerequisite:** lessons 00–03; you know the positive class and the roles of
train and validation.
"""),
        m("## Preflight\n\nRun this check, then load only train and validation."),
        c(preflight()),
        c("""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

from aai_local_classification.contracts import SplitName
from aai_local_classification.data import load_split
from aai_local_classification.workflow import ensure_prepared, run_baseline

ensure_prepared(settings, root)
train = load_split(settings, SplitName.TRAIN, paths.data_root)
validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
y_validation = validation.churned_30d
print(f"✓ Validation rows: {len(validation):,}")
"""),
        m("""
### What you should see

360 validation rows. It is legitimate to inspect validation labels because
validation's declared job is model development. Test labels remain sealed.

### Words introduced

| Word | Plain meaning | Question answered |
|---|---|---|
| baseline | A deliberately simple comparator | Do we beat no skill? |
| confusion matrix | Counts of four correct/error outcomes | What mistakes occur? |
| recall | Share of true positives that we found | How many churners were caught? |
"""),
        m("""
## Start without scikit-learn: predict every row as no churn

**Before you run this:** because churn is the uncommon class, predict whether
accuracy will be high or low—and what recall will be.
"""),
        c("""
all_negative = np.zeros(len(y_validation), dtype=int)
tn, fp, fn, tp = confusion_matrix(
    y_validation, all_negative, labels=[0, 1]
).ravel()

pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["actual 0", "actual 1"],
    columns=["predicted 0", "predicted 1"],
)
"""),
        m("""
### What you should see

290 true negatives, 0 false positives, 70 false negatives, and 0 true
positives. The rule is correct on all 290 non-churn rows and misses all 70 churn
rows.
"""),
        m("""
## Calculate accuracy, precision, and recall from those counts

- **Accuracy** = all correct predictions / all rows.
- **Precision** = true positives / all predicted positives.
- **Recall** = true positives / all actual positives.

When there are no predicted positives, this course reports precision as zero.
"""),
        c("""
baseline_arithmetic = pd.Series(
    {
        "accuracy": (tn + tp) / (tn + fp + fn + tp),
        "precision": 0.0 if tp + fp == 0 else tp / (tp + fp),
        "recall": tp / (tp + fn),
    },
    name="value",
)
baseline_arithmetic.to_frame()
"""),
        m("""
### How to interpret the output

Accuracy is about 80.6%, but recall and precision are 0%. Accuracy is not
mathematically wrong; by itself it answers the wrong practical question for an
uncommon positive event.

### Misconception check

A high accuracy percentage does not imply that the model finds any churners.
Always inspect the error counts and metrics connected to the intended action.
"""),
        m("""
## A probability baseline and a ranking metric

`DummyClassifier(strategy="prior")` assigns every row the training churn rate.
At threshold 0.5, every prediction is still negative. **Average precision (AP)**
asks whether positive rows tend to receive higher scores than negative rows; a
constant-score baseline has AP near the positive prevalence.
"""),
        c("""
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="prior")
dummy.fit(np.zeros((len(train), 1)), train.churned_30d)
dummy_scores = dummy.predict_proba(np.zeros((len(validation), 1)))[:, 1]

pd.Series(
    {
        "constant_score": dummy_scores[0],
        "validation_prevalence": y_validation.mean(),
        "average_precision": average_precision_score(y_validation, dummy_scores),
    }
).to_frame("value")
"""),
        m("""
### What you should see

One constant training-prior score (about 0.151), validation prevalence about
0.194, and average precision about 0.194. The baseline cannot rank one
validation account above another.

Now record the same concept as a repeatable MLflow baseline run.
"""),
        c("""
baseline_evidence = run_baseline(settings, root)
baseline_metrics = baseline_evidence["metrics"]
pd.Series(
    {
        "run_id": baseline_evidence["run_id"],
        "validation_accuracy": baseline_metrics["accuracy"],
        "validation_recall": baseline_metrics["recall"],
        "validation_average_precision": baseline_metrics["average_precision"],
    }
).to_frame("recorded value")
"""),
        m("""
### MLflow words introduced

| Word | Plain meaning | Baseline example |
|---|---|---|
| experiment | A collection of related attempts | subscription-churn course |
| run | One recorded execution | this baseline fit/evaluation |
| parameter / metric / artifact | input choice / measured number / saved file | strategy / recall / model |

The long run ID is a lookup key, not a result to interpret. MLflow also stores
the dataset fingerprint, model artifact, dependency evidence, and source state.
Lesson 05 will make the trained pipeline itself visible.
"""),
        m("""
### Guided exercise

Predict every validation account as churn. Compute the four confusion counts
and teaching cost. A false negative costs 5; a false positive costs 1.
"""),
        c("""
exercise_all_positive = np.ones(len(y_validation), dtype=int)
exercise_tn, exercise_fp, exercise_fn, exercise_tp = confusion_matrix(
    y_validation, exercise_all_positive, labels=[0, 1]
).ravel()
exercise_cost = (
    exercise_fn * settings.selection.false_negative_cost
    + exercise_fp * settings.selection.false_positive_cost
)
pd.Series({"tn": exercise_tn, "fp": exercise_fp, "fn": exercise_fn, "tp": exercise_tp, "cost": exercise_cost})
"""),
        m("""
**Self-check:** all-positive should find every churner but create many
unnecessary reviews.

<details><summary>Solution explanation</summary>

The counts are TN=0, FP=290, FN=0, TP=70. Recall is 100%, but precision and
accuracy are only 19.4%. Its cost is 290, compared with 350 for all-negative
under the fictional weights. Neither rule ranks accounts or balances capacity.
</details>
"""),
        c("""
# Reference solution — run after your attempt
assert (exercise_tn, exercise_fp, exercise_fn, exercise_tp) == (0, 290, 0, 70)
assert exercise_cost == 290
print("✓ All-positive trades perfect recall for 290 false positives")
"""),
        m("""
## MLOps bridge

A candidate should never be called an improvement without a baseline recorded
on the same validation population. MLflow makes that comparator discoverable
and binds its metrics to exact data and code evidence.

## Recap

- A confusion matrix exposes the exact correct and incorrect outcomes.
- The all-negative baseline looks accurate but has zero recall.
- Average precision evaluates ranking; the no-skill value is near prevalence.

**Evidence created:** one reusable MLflow baseline run plus `baseline.json` in
course state. Rerunning replaces no selection or test evidence; it records a
fresh baseline result on the same verified inputs.

**Ready for 05?** You can calculate accuracy and recall from TN/FP/FN/TP and
explain why the baseline is not useful.
"""),
    ],
    "05_pipeline_and_training.ipynb": [
        m("""
# 05 — Preprocess and train a real model, step by step

**Plain-language question:** How can training and future prediction apply
exactly the same cleanup?

**Why this matters:** if preprocessing is learned separately or repeated by
hand, training and inference can transform the same row differently.

**Estimated time:** 60–75 minutes.
**Prerequisite:** lessons 00–04; you know missing values, train/validation, a
confusion matrix, and the no-skill baseline.
"""),
        m("## Preflight\n\nCheck the kernel, then load the two development splits."),
        c(preflight()),
        c("""
import numpy as np

from aai_local_classification.contracts import SplitName
from aai_local_classification.data import load_split
from aai_local_classification.evaluation import evaluate_probabilities
from aai_local_classification.modeling import feature_frame
from aai_local_classification.workflow import ensure_prepared

ensure_prepared(settings, root)
train = load_split(settings, SplitName.TRAIN, paths.data_root)
validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
X_train = feature_frame(train, settings)
X_validation = feature_frame(validation, settings)
y_train = train.churned_30d
y_validation = validation.churned_30d
print(f"✓ X_train {X_train.shape}; X_validation {X_validation.shape}")
"""),
        m("""
### What you should see

Nine raw input columns for 2,160 training rows and 360 validation rows.

### Words introduced

| Word | Plain meaning | Example |
|---|---|---|
| imputation | Fill a missing value using a learned rule | training median |
| scaling | Re-express numbers on comparable scales | mean 0, spread 1 |
| one-hot encoding | Turn each category into indicator columns | plan_basic 0/1 |
"""),
        m("""
## Find real missing inputs

**Before you run this:** predict whether a missing value should be replaced
using information from train alone or train plus validation.
"""),
        c("""
missing_examples = X_train.loc[
    X_train.isna().any(axis=1),
    ["usage_hours_30d", "signup_channel", "plan_tier"],
].head(5)
missing_examples
"""),
        m("""
### How to interpret the output

`NaN` means the value is unknown; it is not the number zero or a category named
“None.” The imputer must learn replacement values from training only. Learning
them from validation would let validation influence the fitted pipeline.
"""),
        m("""
## Build the numeric and categorical paths visibly

An sklearn **transformer** learns a data transformation with `fit` and applies
it with `transform`. A **Pipeline** chains steps so they are fit and applied in
the same order.
"""),
        c("""
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_pipeline = Pipeline(
    [
        ("impute_median", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
)
categorical_pipeline = Pipeline(
    [
        ("impute_most_common", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)
"""),
        m("""
`handle_unknown="ignore"` means a future category that did not occur in
training will not crash inference. It does not refit or invent evidence from the
future row.

Now combine the two paths. `remainder="drop"` excludes undeclared columns.
"""),
        c("""
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, list(settings.features.numeric)),
        (
            "categorical",
            categorical_pipeline,
            list(settings.features.categorical),
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)
"""),
        m("""
## Fit on train; transform validation

**Before you run this:** nine raw columns will expand because each category gets
its own indicator. Predict whether the transformed matrix may contain missing
values.
"""),
        c("""
X_train_encoded = preprocessor.fit_transform(X_train)
X_validation_encoded = preprocessor.transform(X_validation)
encoded_names = preprocessor.get_feature_names_out()

pd.Series(
    {
        "raw_columns": X_train.shape[1],
        "encoded_columns": X_train_encoded.shape[1],
        "training_missing_after_transform": int(np.isnan(X_train_encoded).sum()),
        "validation_missing_after_transform": int(np.isnan(X_validation_encoded).sum()),
    }
).to_frame("value")
"""),
        m("""
### What you should see

Nine raw features become 16 numeric columns and both missing counts are zero.
The fitted categorical names must not include a fake `signup_channel_None`
category—the missing channel was genuinely imputed.
"""),
        c("""
encoded_preview = pd.DataFrame(
    X_train_encoded[:3], columns=encoded_names
)
print(encoded_names.tolist())
encoded_preview
"""),
        m("""
### How to interpret the output

The five numeric inputs appear once each; categories expand into 0/1 columns.
Scaled numeric values can be negative because zero now represents the training
mean—not because the original fee or tenure was negative.
"""),
        m("""
## Put preprocessing and logistic regression in one artifact

### Words introduced

| Word | Plain meaning | Here |
|---|---|---|
| estimator | An object that learns from examples | logistic regression |
| `fit` | Learn preprocessing and model parameters | training rows only |
| `predict_proba` | Return a probability for each class | churn score in column 1 |

Logistic regression learns a weighted linear combination of the encoded
features. It is a useful interpretable first candidate, not proof of causality.
"""),
        c("""
from sklearn.linear_model import LogisticRegression

logistic_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000, random_state=settings.random_seed),
        ),
    ]
)
logistic_pipeline.fit(X_train, y_train)
print("✓ Fitted one complete preprocessing + model Pipeline")
"""),
        m("""
**Before you run this:** predict whether the first five probabilities must all
be between 0 and 1. The `classes_` lookup avoids assuming class-column order.
"""),
        c("""
positive_index = list(logistic_pipeline.classes_).index(1)
validation_probability = logistic_pipeline.predict_proba(X_validation)[
    :, positive_index
]
pd.DataFrame(
    {
        "true_label": y_validation.head().to_numpy(),
        "churn_probability": validation_probability[:5],
        "prediction_at_0.5": (validation_probability[:5] >= 0.5).astype(int),
    }
)
"""),
        m("""
### What you should see

Five different probabilities in `[0, 1]`. At threshold 0.5, many rows remain
negative. A probability is not a promise that a particular account will churn;
it is the model's score under this fitted data and specification.
"""),
        c("""
metrics_at_half = evaluate_probabilities(
    y_validation,
    validation_probability,
    0.5,
    false_negative_cost=settings.selection.false_negative_cost,
    false_positive_cost=settings.selection.false_positive_cost,
)
pd.Series(
    {
        "accuracy": metrics_at_half.accuracy,
        "precision": metrics_at_half.precision,
        "recall": metrics_at_half.recall,
        "average_precision": metrics_at_half.average_precision,
    }
).to_frame("validation value")
"""),
        m("""
### How to interpret the output

Accuracy is around 81%, but recall at 0.5 is only about 11%. The model ranks
positives much better than the baseline (AP around 0.46 versus 0.19), yet the
default threshold is a poor action policy. Lesson 06 separates ranking choice
from threshold choice.

### Misconception check

A Pipeline is not just notebook cells in order. It is one fitted object that
carries the learned imputation, scaling, encoding, and classifier together into
future inference.
"""),
        m("""
## MLOps bridge: what must be recorded

The baseline run from lesson 04 recorded a model artifact. The controlled
comparison in lesson 06 will record this whole Pipeline, an input example,
input/output signature, parameters, metrics, dataset fingerprints, and exact
dependency evidence. Keeping preprocessing inside the artifact prevents
training/serving mismatch.
"""),
        m("""
### Guided exercise

Copy one validation row, set `signup_channel` to a category never seen during
training, and ask the existing fitted Pipeline for a probability. Do not refit.
"""),
        c("""
exercise_future = X_validation.head(1).copy()
exercise_future.loc[:, "signup_channel"] = "brand_new_channel"
exercise_probability = logistic_pipeline.predict_proba(exercise_future)[0, positive_index]
print(f"Probability for unseen category: {exercise_probability:.3f}")
"""),
        m("""
**Self-check:** the prediction should succeed and remain between 0 and 1.
`handle_unknown="ignore"` reuses the training vocabulary; it does not learn the
new category.

<details><summary>Solution explanation</summary>

The complete Pipeline accepts the raw row, ignores the unknown category's
one-hot indicators, applies all other learned transformations, and scores it.
</details>
"""),
        c("""
# Reference solution — run after your attempt
assert 0 <= exercise_probability <= 1
assert "brand_new_channel" not in encoded_names
print("✓ Unseen category handled without refitting")
"""),
        m("""
## Recap

- Imputation, scaling, and encoding learn rules from training only.
- One sklearn Pipeline carries those learned transformations with the model.
- A strong ranking score does not make 0.5 the correct action threshold.

**Evidence created:** a logistic Pipeline in this kernel's memory. No candidate
selection or test decision was persisted, so lesson 06 remains the first place
that chooses and records a candidate.

**Ready for 06?** You can explain why validation uses `transform`, not
`fit_transform`, and why the preprocessing belongs inside the model artifact.
"""),
    ],
    "06_model_selection_and_threshold.ipynb": [
        m("""
# 06 — Choose a model, then choose an action threshold

**Plain-language question:** Is choosing the model the same as choosing when it
should say “yes”?

**Why this matters:** model selection compares ranking quality; threshold
selection decides who receives an action. Mixing the two makes evaluation hard
to reason about and easy to bias.

**Estimated time:** 60–75 minutes.
**Prerequisite:** lessons 00–05; you understand probabilities, confusion counts,
the baseline, and a fitted sklearn Pipeline.
"""),
        m(
            "## Preflight\n\nCheck the kernel, then rebuild the visible development objects."
        ),
        c(preflight()),
        c("""
from sklearn.metrics import average_precision_score

from aai_local_classification.contracts import SplitName
from aai_local_classification.data import load_split
from aai_local_classification.evaluation import (
    evaluate_probabilities,
    select_threshold,
)
from aai_local_classification.learning import state_exists
from aai_local_classification.modeling import (
    build_candidate,
    candidate_specs,
    feature_frame,
)
from aai_local_classification.workflow import (
    ensure_prepared,
    get_or_run_candidate_selection,
)
"""),
        m("""
The import cell names the reusable operations; it has not fit a model. Now load
only train and validation.
"""),
        c("""
ensure_prepared(settings, root)
train = load_split(settings, SplitName.TRAIN, paths.data_root)
validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
X_train = feature_frame(train, settings)
y_train = train.churned_30d
X_validation = feature_frame(validation, settings)
y_validation = validation.churned_30d
print("✓ Development data ready; test remains unopened")
"""),
        m("""
### What you should see

A success message confirming that the test is still unopened.

### Words introduced

| Word | Plain meaning | Here |
|---|---|---|
| ranking | Ordering rows from lower to higher model score | AP evaluates it |
| candidate | One declared model specification being compared | logistic or forest |
| fair comparison | Hold data and evaluation rule constant | same train/validation |
"""),
        m("""
## Fit both candidates on exactly the same rows

The two candidates use the same preprocessing and random seed. Their classifier
families are the controlled change.

**Before you run this:** the more complex random forest may score slightly
higher. Predict whether “slightly higher” must always win.
"""),
        c("""
manual_models = {}
manual_probabilities = {}
comparison_records = []

for spec in candidate_specs():
    model = build_candidate(spec, settings).fit(X_train, y_train)
    positive_index = list(model.classes_).index(1)
    probability = model.predict_proba(X_validation)[:, positive_index]
    average_precision = average_precision_score(y_validation, probability)
    manual_models[spec.name] = model
    manual_probabilities[spec.name] = probability
    comparison_records.append(
        {
            "candidate": spec.name,
            "complexity_rank": spec.complexity_rank,
            "validation_average_precision": average_precision,
        }
    )

manual_comparison = pd.DataFrame(comparison_records)
manual_comparison
"""),
        m("""
### What you should see

Logistic regression AP is about 0.466 and random forest AP about 0.475. Both
rank much better than the roughly 0.194 no-skill baseline. The difference
between the candidates is only about 0.009.

### How to interpret the output

AP ignores a particular threshold and evaluates score ranking across possible
operating points. Higher is better, but tiny differences may not justify a more
complex model.
"""),
        m("""
## Apply the declared simplicity tolerance visibly

The policy says: find the best validation AP, then consider a simpler candidate
eligible when it is within `0.02` of that best score. Among eligible candidates,
prefer lower complexity.
"""),
        c("""
best_ap = manual_comparison.validation_average_precision.max()
manual_comparison["gap_from_best"] = (
    best_ap - manual_comparison.validation_average_precision
)
manual_comparison["within_tolerance"] = (
    manual_comparison.gap_from_best
    <= settings.selection.simpler_model_tolerance
)
eligible_manual = manual_comparison.loc[manual_comparison.within_tolerance]
manual_selected_name = eligible_manual.sort_values("complexity_rank").iloc[0].candidate
selected_probability = manual_probabilities[manual_selected_name]
manual_comparison
"""),
        m("""
### What you should see

Both rows are within `0.02`; the selected name is
`logistic-regression` because it has complexity rank 1. This is a predeclared
preference, not a story invented after seeing test data.

### Misconception check

“More complex” does not mean “more production-ready.” Complexity can increase
maintenance and explanation cost. A different project may declare a different
tolerance or no simplicity preference at all.
"""),
        m("""
## A model score is not yet an action

### Words introduced

| Word | Plain meaning | Example |
|---|---|---|
| threshold | Score at or above which prediction becomes 1 | `0.12` |
| operating point | Precision/recall/workload at one threshold | one table row |
| constraint | A minimum/maximum an option must satisfy | recall at least 0.75 |

For a score of 0.20, threshold 0.12 says positive while threshold 0.50 says
negative. The fitted model and its AP have not changed—only the action rule has.

The teaching **cost per 1,000** is
`(5 × false negatives + 1 × false positives) / row count × 1,000`. The cost
units are fictional; the calculation makes an authored trade-off comparable
across batches of different sizes.
"""),
        m("""
**Before you run this:** as the threshold rises, predict what generally happens
to the number of positive predictions and to recall.
"""),
        c("""
def threshold_record(threshold):
    result = evaluate_probabilities(
        y_validation,
        selected_probability,
        threshold,
        false_negative_cost=settings.selection.false_negative_cost,
        false_positive_cost=settings.selection.false_positive_cost,
    )
    feasible = (
        result.precision >= settings.selection.minimum_validation_precision
        and result.recall >= settings.selection.minimum_validation_recall
    )
    return {
        "threshold": threshold,
        "precision": result.precision,
        "recall": result.recall,
        "predicted_positive_rate": result.predicted_positive_rate,
        "cost_per_1000": result.cost_per_1000,
        "feasible": feasible,
    }
"""),
        m("""
The helper above evaluates one visible threshold; it does not choose anything.
Now apply it to five illustrative operating points.
"""),
        c("""
sample_thresholds = [0.10, 0.12, 0.13, 0.20, 0.50]
threshold_table = pd.DataFrame(
    threshold_record(value) for value in sample_thresholds
)
threshold_table
"""),
        m("""
### How to interpret the output

At 0.50, recall is very low. At 0.12, recall is about 0.829 and precision
about 0.305; roughly 53% of rows receive a positive action. The lower threshold
accepts more false positives to avoid expensive false negatives.

The exact policy searches a finer grid, keeps rows meeting minimum precision
0.30 and recall 0.75, then chooses minimum teaching cost.
"""),
        c("""
manual_threshold = select_threshold(
    y_validation,
    selected_probability,
    settings,
)
pd.Series(
    {
        "chosen_threshold": manual_threshold.threshold,
        "precision": manual_threshold.validation_metrics.precision,
        "recall": manual_threshold.validation_metrics.recall,
        "cost_per_1000": manual_threshold.validation_metrics.cost_per_1000,
        "feasible_thresholds": manual_threshold.feasible_threshold_count,
    }
).to_frame("validation result")
"""),
        m("""
### What you should see

Threshold `0.12`, precision about 0.305, recall about 0.829, and cost about
533.3 per 1,000. If no threshold met both constraints, selection would stop as
inconclusive before touching test data.
"""),
        c("""
import matplotlib.pyplot as plt

plot_table = threshold_table.set_index("threshold")
ax = plot_table[["precision", "recall", "predicted_positive_rate"]].plot(
    marker="o", figsize=(8, 4), title="Validation trade-offs at sample thresholds"
)
ax.axhline(settings.selection.minimum_validation_precision, linestyle="--", color="gray")
ax.axhline(settings.selection.minimum_validation_recall, linestyle=":", color="gray")
ax.set_ylabel("share")
plt.tight_layout()
"""),
        m("""
The chart makes the trade-off visible. Dashed guide lines are constraints, not
universal standards. The action owner must also consider review capacity and
whether the assumed costs are defensible.

## Record the controlled comparison in MLflow

Only now do we call the packaged helper. It repeats the operations just shown,
logs each candidate as a nested run, stores each complete Pipeline, and persists
the selection evidence. If compatible evidence already exists, it reuses it so
a later frozen-test decision cannot be disconnected from its chosen model.
"""),
        c("""
selection_already_existed = state_exists("selection.json")
selection = get_or_run_candidate_selection(settings, root)
chosen = next(
    item for item in selection.candidates
    if item.run_id == selection.selected_run_id
)
print("Reused compatible selection" if selection_already_existed else "Created selection")
pd.Series(
    {
        "selected_candidate": selection.selected_candidate,
        "threshold": chosen.threshold_selection.threshold,
        "validation_average_precision": chosen.threshold_selection.validation_metrics.average_precision,
        "test_data_accessed": False,
    }
).to_frame("recorded value")
"""),
        m("""
### What you should see

The recorded candidate and threshold match the manual result, and
`test_data_accessed` remains false. The helper also logged parameters, metrics,
dataset inputs, model signature, input example, configuration, manifest, and
dependency lock.
"""),
        c("""
import mlflow

mlflow.set_tracking_uri(paths.tracking_uri)
client = mlflow.MlflowClient()
recorded_run = client.get_run(selection.selected_run_id)
pd.Series(
    {
        "run_name": recorded_run.data.tags.get("mlflow.runName"),
        "model_family": recorded_run.data.params.get("model_family"),
        "validation_average_precision": recorded_run.data.metrics.get(
            "validation_average_precision"
        ),
        "artifact_names": [item.path for item in client.list_artifacts(selection.selected_run_id)],
    }
).to_frame("MLflow value")
"""),
        m("""
### MLOps bridge

The MLflow run makes the comparison auditable: same data inputs, explicit
parameters, linked metrics, a loadable model, and reproducibility artifacts.
The model's restore environment uses the exact runtime dependency closure
exported from `uv.lock`, not only a few top-level package pins.
Run `make mlflow-ui` in a second Terminal to explore the same local experiment;
stop that server with Ctrl-C.
"""),
        m("""
### Guided exercise

In a copied setting, double the false-positive cost from 1 to 2 and rerun only
threshold selection on the already-computed validation probabilities. This is
safe scratch work; it does not replace the persisted selection.
"""),
        c("""
exercise_selection_policy = settings.selection.model_copy(
    update={"false_positive_cost": 2.0}
)
exercise_settings = settings.model_copy(
    update={"selection": exercise_selection_policy}
)
exercise_threshold = select_threshold(
    y_validation,
    selected_probability,
    exercise_settings,
)
pd.Series(
    {
        "original_threshold": manual_threshold.threshold,
        "new_threshold": exercise_threshold.threshold,
        "original_precision": manual_threshold.validation_metrics.precision,
        "new_precision": exercise_threshold.validation_metrics.precision,
    }
)
"""),
        m("""
**Self-check:** making false alarms more expensive raises the chosen threshold
in this deterministic exercise. It trades some recall for higher precision.
Discrete thresholds and constraints mean the exact move is a policy result,
not a universal mathematical guarantee.

<details><summary>Solution explanation</summary>

The selected threshold moves from 0.12 to 0.14. Precision rises from about
0.305 to about 0.321 while recall falls from about 0.829 to about 0.757. Raising
the threshold flags fewer accounts when false alarms cost more.
</details>
"""),
        c("""
# Reference solution — run after your attempt
assert abs(exercise_threshold.threshold - 0.14) < 1e-9
assert (
    exercise_threshold.validation_metrics.precision
    >= manual_threshold.validation_metrics.precision
)
print("✓ Higher false-positive cost produced a higher-threshold policy")
"""),
        m("""
## Recap

- Validation AP chooses ranking behavior; it does not choose an action threshold.
- A predeclared tolerance selects the simpler logistic candidate here.
- Constraints and error costs choose threshold 0.12 on validation only.

**Evidence created:** MLflow selection and candidate runs plus `selection.json`.
A compatible rerun reuses this selection once later test evidence exists.

**Ready for 07?** You can explain why AP stays fixed when only the threshold
changes and why test data has not yet been opened.
"""),
    ],
    "07_frozen_test_gate.ipynb": [
        m("""
# 07 — Evaluate the frozen test for one release decision

**Plain-language question:** Does the exact chosen model work on later data it
did not use?

**Why this matters:** the test set is useful only if the model, threshold, and
release rules are fixed before its labels are examined.

**Estimated time:** 50–65 minutes.
**Prerequisite:** lesson 06; you understand the selected candidate, threshold,
average precision, precision, recall, and confusion counts.
"""),
        m(
            "## Preflight\n\nCheck the kernel and locate or visibly recreate lesson 06 evidence."
        ),
        c(preflight()),
        c("""
from aai_local_classification.contracts import SplitName
from aai_local_classification.data import load_split
from aai_local_classification.evaluation import recall_slices
from aai_local_classification.learning import state_exists
from aai_local_classification.modeling import feature_frame
from aai_local_classification.workflow import (
    get_or_run_candidate_selection,
    run_frozen_test_gate,
)

selection_was_missing = not state_exists("selection.json")
selection = get_or_run_candidate_selection(settings, root)
chosen = next(
    item for item in selection.candidates
    if item.run_id == selection.selected_run_id
)
print("Recreated lesson 06 selection" if selection_was_missing else "✓ Reused lesson 06 selection")
"""),
        m("""
### What you should see

Normally, `✓ Reused lesson 06 selection`. If this notebook was opened directly,
it says that it recreated the prerequisite—nothing is silently hidden.

### Words introduced

| Word | Plain meaning | Here |
|---|---|---|
| frozen test | Final rows not used to choose model or threshold | Oct–Dec 2024 |
| release gate | Prewritten pass/fail rules | six checks below |
| adopt / reject | Release may proceed / must not proceed | gate decision |
"""),
        m("""
## Freeze the choice before opening the final exam

**Before you run this:** verify that the selected model, threshold, and all gate
limits come from persisted configuration/evidence—not from test results.
"""),
        c("""
fixed_choice = pd.Series(
    {
        "candidate": selection.selected_candidate,
        "model_uri": selection.selected_model_uri,
        "threshold_selected_on_validation": chosen.threshold_selection.threshold,
        "minimum_test_average_precision": settings.promotion_gate.minimum_test_average_precision,
        "minimum_test_average_precision_lift": settings.promotion_gate.minimum_test_average_precision_lift,
        "minimum_test_recall": settings.promotion_gate.minimum_test_recall,
        "maximum_test_brier_score": settings.promotion_gate.maximum_test_brier_score,
        "maximum_test_cost_per_1000": settings.promotion_gate.maximum_test_cost_per_1000,
        "maximum_slice_recall_gap": settings.promotion_gate.maximum_slice_recall_gap,
    }
)
fixed_choice.to_frame("fixed before test")
"""),
        m("""
### How to interpret the output

This is the test protocol: one exact logged model, threshold 0.12, and authored
limits. A model change, threshold change, or gate change after viewing test
results would require a new frozen-test dataset version.
"""),
        m("""
## Run the one-time, idempotent gate

**Before you run this:** “adopt” and “reject” are both valid outcomes. Neither
should crash the notebook. Predict which outcome the deterministic course was
designed to demonstrate.
"""),
        c("""
decision = run_frozen_test_gate(settings, root, selection)
print(f"Decision: {decision.decision.value.upper()}")
print(decision.rationale)
"""),
        m("""
### What you should see

`Decision: ADOPT` and a sentence saying all predeclared checks passed. That means
the artifact passed this teaching gate; it does not prove business value,
fairness, privacy, or universal production readiness.
"""),
        m("""
## Show actual value, rule, and result—not just `True`

**Brier score** is the average squared distance between predicted probability
and a 0/1 outcome. Lower is better; a confidently wrong probability is penalized
more than a cautious one. It is a simple calibration-sensitive diagnostic.
"""),
        c("""
test_metrics = decision.metrics
gate = settings.promotion_gate
checks = decision.checks.model_dump()

actual_gate_values = pd.Series(
    {
        "average precision": test_metrics["test_average_precision"],
        "AP lift over prevalence": (
            test_metrics["test_average_precision"]
            - test_metrics["test_positive_rate"]
        ),
        "recall": test_metrics["test_recall"],
        "Brier score": test_metrics["test_brier_score"],
        "cost per 1,000": test_metrics["test_cost_per_1000"],
        "maximum slice recall gap": test_metrics[
            "test_maximum_slice_recall_gap"
        ],
    }
)
"""),
        m("""
The actual values above cover ranking lift, captured positives, probability
quality, operational cost, and concentration by slice. For a no-skill ranking,
AP is near the positive-class prevalence; **AP lift over prevalence** subtracts
that reference rate from observed AP. Next pair each value with the rule that
was fixed before test access.
"""),
        c("""
required_gate_values = pd.Series(
    {
        "average precision": gate.minimum_test_average_precision,
        "AP lift over prevalence": gate.minimum_test_average_precision_lift,
        "recall": gate.minimum_test_recall,
        "Brier score": gate.maximum_test_brier_score,
        "cost per 1,000": gate.maximum_test_cost_per_1000,
        "maximum slice recall gap": gate.maximum_slice_recall_gap,
    }
)
gate_rules = pd.Series(
    [">=", ">=", ">=", "<=", "<=", "<="],
    index=actual_gate_values.index,
)
"""),
        m("""
The first three metrics have minimums (`>=`); the last three have maximums
(`<=`). Finally, attach the gate's recorded booleans without recalculating or
changing the evidence.
"""),
        c("""
passed_gate_values = pd.Series(
    list(checks.values()),
    index=actual_gate_values.index,
)
gate_table = pd.DataFrame(
    {
        "actual": actual_gate_values,
        "rule": gate_rules,
        "required": required_gate_values,
        "passed": passed_gate_values,
    }
)
gate_table
"""),
        m("""
### What you should see

Approximately AP 0.471, recall 0.757, Brier 0.135, cost 588.9 per 1,000,
and maximum slice recall gap 0.327. Every row passes its authored comparison.

The margin matters: cost passes 600 but is not far below it. A boolean alone
would hide that operational context.
"""),
        m("## Inspect the final confusion counts"),
        c("""
test_confusion = pd.DataFrame(
    [
        [test_metrics["test_true_negatives"], test_metrics["test_false_positives"]],
        [test_metrics["test_false_negatives"], test_metrics["test_true_positives"]],
    ],
    index=["actual 0", "actual 1"],
    columns=["predicted 0", "predicted 1"],
)
test_confusion
"""),
        m("""
### How to interpret the output

The selected threshold catches most positive rows but also flags many negative
rows. The counts make the release trade-off concrete and should be reviewed
with action capacity, not only model developers.
"""),
        m("""
## Compare validation and test without tuning again

Validation chose the model and threshold; test judges that fixed choice. Small
differences are expected on different rows.
"""),
        c("""
validation_metrics = chosen.threshold_selection.validation_metrics
validation_vs_test = pd.DataFrame(
    {
        "validation": {
            "average_precision": validation_metrics.average_precision,
            "precision": validation_metrics.precision,
            "recall": validation_metrics.recall,
            "brier_score": validation_metrics.brier_score,
            "cost_per_1000": validation_metrics.cost_per_1000,
        },
        "test": {
            "average_precision": test_metrics["test_average_precision"],
            "precision": test_metrics["test_precision"],
            "recall": test_metrics["test_recall"],
            "brier_score": test_metrics["test_brier_score"],
            "cost_per_1000": test_metrics["test_cost_per_1000"],
        },
    }
)
validation_vs_test
"""),
        m("""
The test result is somewhat different but remains inside every fixed gate. We
do not now change the threshold to make the test table prettier.

An **operational slice** groups rows by a field such as plan or signup channel
to reveal concentrated errors. This diagnostic is not a fairness certification.
"""),
        c("""
import mlflow.sklearn

test = load_split(settings, SplitName.TEST, paths.data_root)
test_model = mlflow.sklearn.load_model(selection.selected_model_uri)
X_test = feature_frame(test, settings)
positive_index = list(test_model.classes_).index(1)
test_probability = test_model.predict_proba(X_test)[:, positive_index]
slice_table = recall_slices(
    test,
    test.churned_30d,
    test_probability,
    decision.threshold,
)
slice_table
"""),
        m("""
### What you should see

Recall for sufficiently large plan/channel groups, including row and positive
counts. A gap can flag investigation, but synthetic categories, limited sample
sizes, and omitted human-impact analysis prevent any fairness conclusion.

### Misconception check

Test is not “validation round two.” If a test result causes a model or policy
change, that new choice has learned from the test and needs new final evidence.
"""),
        m("""
## Prove rerunning does not consume the test twice

The helper is idempotent: compatible evidence is returned rather than creating
a second result run.
"""),
        c("""
decision_again = run_frozen_test_gate(settings, root, selection)
pd.Series(
    {
        "first_test_run_id": decision.test_run_id,
        "second_test_run_id": decision_again.test_run_id,
        "same_evidence": decision.test_run_id == decision_again.test_run_id,
    }
).to_frame("value")
"""),
        m("""
### Guided exercise

Without reevaluating test rows, apply a hypothetical stricter cost limit of 500
to the already-recorded cost. What release outcome follows?
"""),
        c("""
exercise_cost_limit = 500.0
exercise_recorded_cost = test_metrics["test_cost_per_1000"]
exercise_passed = exercise_recorded_cost <= exercise_cost_limit
exercise_decision = "adopt" if exercise_passed else "reject"
pd.Series(
    {
        "recorded_cost": exercise_recorded_cost,
        "hypothetical_limit": exercise_cost_limit,
        "hypothetical_decision": exercise_decision,
    }
)
"""),
        m("""
**Self-check:** 588.9 is greater than 500, so the hypothetical outcome is a
valid `reject`. This thought experiment does not rewrite the real gate, create
new test evidence, or support a new release claim. Adopting a stricter gate for
a future release would require new final evidence for the changed policy.

<details><summary>Solution explanation</summary>

A release gate can legitimately reject a statistically promising model because
the selected action policy violates an operational limit.
</details>
"""),
        c("""
# Reference solution — run after your attempt
assert exercise_decision == "reject"
assert not exercise_passed
print("✓ The stricter hypothetical gate rejects without crashing")
"""),
        m("""
## MLOps bridge

The result run binds the exact logged model, dataset fingerprint, selection
policy, gate policy, threshold, metrics, slices, and decision. In a Databricks
workflow, this gate belongs in a job or CI task—not an informal UI click.

## Recap

- The selected model, threshold, and gate were fixed before test access.
- Actual values, rules, margins, confusion counts, and slices explain the gate.
- Both adopt and reject are valid outcomes; compatible reruns reuse evidence.

**Evidence created:** one MLflow frozen-test result and `decision.json`. The
frozen test is now consumed for this exact model and policy.

**Ready for 08?** You can explain why the passing test is release evidence, not
permission to tune again.
"""),
    ],
    "08_registry_and_inference.ipynb": [
        m("""
# 08 — Register, reload, and use the approved artifact

**Plain-language question:** Which exact artifact receives the name `champion`,
and can a fresh loader reload it safely?

**Why this matters:** a release should point to the model that passed the gate,
not a convenient refit or an object still living only in one notebook kernel.

**Estimated time:** 45–60 minutes.
**Prerequisite:** lesson 07; you understand the fixed test decision and its
approved threshold.
"""),
        m(
            "## Preflight\n\nCheck the kernel and visibly locate or rebuild prerequisite evidence."
        ),
        c(preflight()),
        c("""
import mlflow

from aai_local_classification.contracts import SplitName
from aai_local_classification.data import load_split
from aai_local_classification.inference import load_champion
from aai_local_classification.learning import state_exists
from aai_local_classification.modeling import feature_frame
from aai_local_classification.workflow import (
    get_or_run_candidate_selection,
    promote_if_approved,
    run_frozen_test_gate,
)

selection_missing = not state_exists("selection.json")
selection = get_or_run_candidate_selection(settings, root)
decision_missing = not state_exists("decision.json")
decision = run_frozen_test_gate(settings, root, selection)
print(f"Selection recreated: {selection_missing}; decision recreated: {decision_missing}")
"""),
        m("""
### What you should see

After lessons 06–07, both values are `False`. If opened directly, this notebook
explicitly reports which prerequisite evidence it recreated.

### Words introduced

| Word | Plain meaning | Example |
|---|---|---|
| logged model | The saved model artifact attached to a run | selected Pipeline |
| registered version | An immutable numbered registry entry | version 1, 2, … |
| alias | A movable human-friendly pointer | `champion` |
"""),
        m("""
## Follow the release chain

```text
candidate run
    └── logged Pipeline (fixed model ID / URI)
          └── frozen-test decision: ADOPT or REJECT
                └── registered version (only after ADOPT)
                      └── alias: champion
```

The concrete version is the auditable artifact. The alias helps consumers find
the current approved version and may move during a future release.
"""),
        m("""
## Promote only an adopted model

**Before you run this:** predict what should happen if the gate says `reject`.
The safe behavior is a clear result with `registered=False`, not an exception
and not a moved alias.
"""),
        c("""
promotion = promote_if_approved(
    settings,
    decision,
    root,
    selection,
)
pd.Series(promotion).to_frame("promotion value")
"""),
        m("""
### What you should see

For the deterministic course, `registered=True`, a concrete model version, and
alias `champion`. If a legitimate reject occurred, the table would instead
explain that the alias remains unchanged and the notebook would continue.

### How to interpret the output

Registration does not train anything. It records the exact selected model URI
that already passed the gate and adds release metadata around that artifact.
"""),
        m("""
## Inspect the registry rather than trusting a success message

**Before you run this:** an alias is a pointer. Predict whether resolving it
should return the same concrete version shown in `promotion`.
"""),
        c("""
mlflow.set_tracking_uri(paths.tracking_uri)
client = mlflow.MlflowClient()

if promotion.get("registered"):
    registry_version = client.get_model_version_by_alias(
        settings.registered_model_name, "champion"
    )
    tags = registry_version.tags
    registry_view = pd.Series(
        {
            "registered_name": registry_version.name,
            "concrete_version": registry_version.version,
            "selected_candidate": tags.get("selected_candidate"),
            "decision_threshold": tags.get("decision_threshold"),
            "test_run_id": tags.get("test_run_id"),
        }
    )
else:
    registry_version = None
    registry_view = pd.Series({"status": promotion["reason"]})

registry_view.to_frame("registry value")
"""),
        m("""
### What you should see

`champion` resolves to the same version number returned by promotion. Its tags
carry the selected candidate, threshold 0.12, and final test run ID. These fields
connect discovery (`champion`) back to release evidence.
"""),
        m("""
## Inspect the input/output contract saved with the model

A **signature** describes expected input and output columns/types. An **input
example** is a small valid batch stored with the model to make that contract
concrete. Neither replaces live data validation.
"""),
        c("""
model_info = mlflow.models.get_model_info(selection.selected_model_uri)
signature_view = pd.Series(
    {
        "model_id": model_info.model_id,
        "input_schema": str(model_info.signature.inputs),
        "output_schema": str(model_info.signature.outputs),
        "decision_threshold_metadata": model_info.metadata.get("decision_threshold"),
        "positive_class_metadata": model_info.metadata.get("positive_class"),
    }
)
signature_view.to_frame("logged model contract")
"""),
        m("""
### How to interpret the output

The nine raw feature names/types are the input contract. The model emits two
class probabilities. The approved binary action still requires the threshold
stored in registry evidence.

## Inspect the saved input example

The signature describes types; the input example shows five concrete rows that
match those types. It is documentation and a deployment check—not a substitute
for the complete training dataset or proof that all future rows will be valid.
"""),
        c("""
input_example_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"{model_info.artifact_path}/input_example.json"
)
saved_input_example = pd.read_json(input_example_path, orient="split")
saved_input_example
"""),
        m("""
### What you should see

Five rows with the same nine feature columns named by the signature. The target
column is absent because inference does not require the future outcome.

### Misconception check

Moving an alias does not necessarily update a process that already loaded an
older model. Production jobs/endpoints should record the concrete version they
actually used.
"""),
        m("""
## Reload into a new object and run inference

**Inference** means applying a fitted model to rows whose labels are not needed
at prediction time. We intentionally keep only declared feature columns.

**Before you run this:** predict the four output fields that a traceable
predictor should return.
"""),
        c("""
validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
inference_batch = feature_frame(validation.tail(8), settings)

if promotion.get("registered"):
    predictor = load_champion(settings, root)
    prediction_output = predictor.predict(inference_batch, settings)
else:
    predictor = None
    prediction_output = pd.DataFrame(
        {"status": ["No approved champion; inference correctly skipped."]}
    )

prediction_output
"""),
        m("""
### What you should see

Eight rows with `churn_probability`, `churn_prediction`, `model_name`, and the
concrete `model_version`. Probabilities are between 0 and 1, predictions are 0
or 1, and every row names the same loaded version.
"""),
        c("""
if predictor is not None:
    inference_preview = pd.concat(
        [
            inference_batch[["monthly_fee", "usage_hours_30d", "contract_type"]],
            prediction_output[["churn_probability", "churn_prediction"]],
        ],
        axis=1,
    )
else:
    inference_preview = prediction_output

inference_preview
"""),
        m("""
### How to interpret the output

The feature values provide context for each score, but they do not establish a
causal explanation. The binary column is exactly
`churn_probability >= approved threshold`; it is not sklearn's default 0.5
classification.
"""),
        c("""
if predictor is not None:
    threshold_check = pd.DataFrame(
        {
            "probability": prediction_output.churn_probability,
            "approved_threshold": predictor.threshold,
            "recomputed_prediction": (
                prediction_output.churn_probability >= predictor.threshold
            ).astype(int),
            "returned_prediction": prediction_output.churn_prediction,
        }
    )
else:
    threshold_check = prediction_output

threshold_check
"""),
        m("""
### Guided exercise

Classify a probability of 0.20 using thresholds 0.12 and 0.50. This isolates
why threshold evidence must travel with the model.
"""),
        c("""
exercise_probability = 0.20
exercise_actions = pd.Series(
    {
        "action_at_0.12": int(exercise_probability >= 0.12),
        "action_at_0.50": int(exercise_probability >= 0.50),
    }
)
exercise_actions.to_frame("binary action")
"""),
        m("""
**Self-check:** the same model score becomes positive at 0.12 and negative at
0.50. Losing the approved threshold changes who receives the action.

<details><summary>Solution explanation</summary>

The comparisons are `0.20 >= 0.12` (true) and `0.20 >= 0.50` (false). The
threshold is release policy, not a hidden model default.
</details>
"""),
        c("""
# Reference solution — run after your attempt
assert exercise_actions.to_dict() == {
    "action_at_0.12": 1,
    "action_at_0.50": 0,
}
if predictor is not None:
    assert (threshold_check.recomputed_prediction == threshold_check.returned_prediction).all()
print("✓ Version and approved threshold produce reproducible actions")
"""),
        m("""
## MLOps bridge

Local SQLite Registry teaches the objects; Databricks uses Models in Unity
Catalog with a three-part name (`catalog.schema.model`) and governed
permissions. Use aliases for discovery and concrete versions for auditable jobs
or endpoints.

## Recap

- Only an adopted logged model is registered and assigned `champion`.
- A registered version is concrete evidence; an alias is a mutable pointer.
- Reloaded inference carries model name, version, and approved threshold.

**Evidence created:** a registered model version, version tags, `champion`
alias, and `promotion.json`. Compatible reruns resolve the same version rather
than registering duplicates.

**Ready for 09?** You can trace one prediction from alias to concrete version,
logged model, test decision, and threshold.
"""),
    ],
    "09_monitoring_and_databricks.ipynb": [
        m("""
# 09 — Monitor changed behavior and map the workflow to Databricks

**Plain-language question:** What can monitoring tell us before true outcomes
arrive—and what must remain the same when we move to Databricks?

**Why this matters:** production data and system behavior change. Monitoring
should trigger investigation without claiming more than the available evidence
supports.

**Estimated time:** 60–75 minutes.
**Prerequisite:** lessons 00–08; you can load the approved version and explain
its score, prediction, and release evidence.
"""),
        m(
            "## Preflight\n\nCheck the kernel and visibly ensure the approved local release exists."
        ),
        c(preflight()),
        c("""
from aai_local_classification.contracts import SplitName
from aai_local_classification.data import load_split
from aai_local_classification.inference import load_champion
from aai_local_classification.monitoring import compare_batches, shifted_batch
from aai_local_classification.workflow import (
    get_or_run_candidate_selection,
    promote_if_approved,
    run_frozen_test_gate,
)

selection = get_or_run_candidate_selection(settings, root)
decision = run_frozen_test_gate(settings, root, selection)
promotion = promote_if_approved(settings, decision, root, selection)
print(f"Release decision: {decision.decision.value}; registered: {promotion.get('registered')}")
"""),
        m("""
### What you should see

`Release decision: adopt; registered: True`. If a valid reject occurred, no
champion would be loaded and inference monitoring would be skipped safely.

### Words introduced

| Word | Plain meaning | Example signal |
|---|---|---|
| input drift | Feature distribution changed | higher monthly fees |
| score drift | Model scores/actions changed | more positive predictions |
| delayed labels | Outcomes arriving after predictions | later churn truth |
"""),
        m("""
## Create a transparent simulated current batch

This course has no live traffic. The helper resamples validation rows, raises
monthly fees by 10%, lowers usage by about four hours, and changes some signup
channels. It is a deliberate scenario, not evidence about real customers.

**Before you run this:** predict which mean will rise and which will fall.
"""),
        c("""
reference = load_split(settings, SplitName.VALIDATION, paths.data_root)
current = shifted_batch(reference, settings.random_seed + 99)

raw_comparison = pd.DataFrame(
    {
        "reference": {
            "monthly_fee_mean": reference.monthly_fee.mean(),
            "usage_hours_mean": reference.usage_hours_30d.mean(),
            "paid_search_share": (reference.signup_channel == "paid_search").mean(),
            "usage_missing_rate": reference.usage_hours_30d.isna().mean(),
        },
        "current": {
            "monthly_fee_mean": current.monthly_fee.mean(),
            "usage_hours_mean": current.usage_hours_30d.mean(),
            "paid_search_share": (current.signup_channel == "paid_search").mean(),
            "usage_missing_rate": current.usage_hours_30d.isna().mean(),
        },
    }
)
raw_comparison
"""),
        m("""
### What you should see

Monthly fee rises (roughly 68.5 to the mid-70s), usage falls (roughly 28.4 to
the mid-20s), and category share changes. Exact resampling values are
deterministic for the course seed.

### How to interpret the output

Raw summaries are usually more understandable than a single drift score. They
identify what moved and in which direction; they do not tell us whether model
recall or calibration changed.
"""),
        c("""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
reference.monthly_fee.plot(kind="hist", alpha=0.5, bins=20, ax=axes[0], label="reference")
current.monthly_fee.plot(kind="hist", alpha=0.5, bins=20, ax=axes[0], label="current")
reference.usage_hours_30d.plot(kind="hist", alpha=0.5, bins=20, ax=axes[1], label="reference")
current.usage_hours_30d.plot(kind="hist", alpha=0.5, bins=20, ax=axes[1], label="current")
axes[0].set_title("Monthly fee")
axes[1].set_title("Usage hours")
for axis in axes:
    axis.legend()
plt.tight_layout()
"""),
        m("""
The overlapping histograms show the reference and current distributions rather
than hiding change behind one number. In production, compare appropriate
seasonal cohorts and investigate upstream product/data changes.
"""),
        m("""
## Add compact diagnostics after inspecting raw values

**Population Stability Index (PSI)** summarizes binned numeric distribution
change. **Total variation (TV)** summarizes categorical-share change from 0
(same shares) to 1 (no overlap). Neither has a universal pass/fail threshold.
"""),
        c("""
report = compare_batches(reference, current, settings)
numeric_drift = pd.DataFrame(
    {
        "psi": pd.Series(report.numeric_psi),
        "missing_rate_change": pd.Series(report.missing_rate_delta),
    }
).sort_values("psi", ascending=False)
numeric_drift
"""),
        m("""
### What you should see

Monthly fee is the largest numeric PSI value; maximum PSI is around 0.32
for this simulation. The exact number is a diagnostic to investigate alongside
the raw distributions, not an automatic declaration of failure.
"""),
        c("""
categorical_drift = pd.Series(
    report.categorical_total_variation,
    name="total_variation",
).sort_values(ascending=False)
categorical_drift.to_frame()
"""),
        m("""
### How to interpret the output

A larger TV value means category shares moved more. It does not identify a root
cause or impact. Monitoring should link the signal to an owner and a safe
investigation playbook.
"""),
        m("""
## Did model scores and actions move too?

**Before you run this:** because the simulated batch has higher fees and lower
usage, predict whether the average churn score and positive-action rate rise or
fall.
"""),
        c("""
def score_summary(output):
    return pd.Series(
        {
            "mean_score": output.churn_probability.mean(),
            "predicted_positive_rate": output.churn_prediction.mean(),
        }
    )

if promotion.get("registered"):
    predictor = load_champion(settings, root)
    reference_scores = predictor.predict(reference, settings)
    current_scores = predictor.predict(current, settings)
    score_comparison = pd.concat(
        [score_summary(reference_scores), score_summary(current_scores)], axis=1
    )
    score_comparison.columns = ["reference", "current"]
else:
    score_comparison = pd.DataFrame(
        {"status": ["No approved champion; score monitoring skipped."]}
    )

score_comparison
"""),
        m("""
### What you should see

Mean score rises from roughly 0.17 to around 0.20, and the positive-action rate
rises from about 53% to around 59%. These values are deterministic for the
course seed.

### Misconception check

Changed inputs and scores do **not** prove degraded accuracy, recall, or
calibration. Those require correctly joined delayed churn labels. No drift also
would not prove that the system is safe or useful.
"""),
        m("""
## Separate monitoring questions and owners

| Layer | Signal available now | Conclusion allowed | Not established yet |
|---|---|---|---|
| service | errors, latency, throughput | endpoint/job health changed | model quality |
| schema/data | missing/invalid fields | input contract failed | causal impact |
| inputs/scores | distributions and actions moved | investigate drift | recall/calibration loss |
| outcomes | delayed labels joined correctly | performance/calibration changed | business causality |

An alert without an owner and safe response is only telemetry.
"""),
        m("""
## Map understood local objects to Databricks

The concepts stay the same; storage, identity, orchestration, and governance
become shared platform services.

| Local object you used | Databricks equivalent | New term in plain language |
|---|---|---|
| generated CSV + manifest | versioned Unity Catalog Delta table | governed table with auditable versions |
| local SQLite MLflow | hosted MLflow experiment | shared tracking service |
| local registered model | `<catalog>.<schema>.<model>` | three-part governed model name |
| local `champion` alias | Models in Unity Catalog alias | pointer on a governed model |
| Python/Make workflow | Databricks job in a Declarative Automation Bundle | reviewed declarative deployment |
| local batch predictor | job or Model Serving at a concrete version | scheduled or online inference |
| local drift report | governed tables, profiles, dashboards, alerts | shared monitoring evidence |

Read `docs/databricks-handoff.md` before adapting this project. Cloud migration
does not authorize creating infrastructure or storing credentials; use the
approved keyless identity and external platform process.
"""),
        m("""
### Guided exercise

Complete one safe alert for each layer. The starter table contains a reasonable
reference answer; change the wording to match how you would explain it to an
operations partner.
"""),
        c("""
exercise_alerts = pd.DataFrame(
    {
        "layer": ["service", "data", "outcome"],
        "signal": [
            "elevated errors/latency",
            "required feature missing",
            "labeled recall below floor",
        ],
        "owner": ["serving owner", "data owner", "model owner"],
        "safe_action": [
            "investigate; roll back if unsafe",
            "stop batch and repair upstream",
            "review, disable, or retrain",
        ],
    }
)
exercise_alerts
"""),
        m("""
**Self-check:** every row needs a measurable signal, accountable owner, and
bounded action. “Retrain automatically whenever PSI is high” is not a safe
default because drift is not proof of failure.

<details><summary>Solution explanation</summary>

Service problems belong to serving operations, broken fields to a data owner,
and labeled quality changes to model/decision owners. Each response preserves
evidence and limits harm while the cause is investigated.
</details>
"""),
        c("""
# Reference solution — run after your attempt
assert set(exercise_alerts.layer) == {"service", "data", "outcome"}
assert exercise_alerts.owner.str.len().gt(0).all()
assert exercise_alerts.safe_action.str.len().gt(0).all()
print("✓ Each alert has a signal, owner, and safe action")
"""),
        m("""
## Recap

- Inspect raw shifts before compact diagnostics; drift prompts investigation,
  not an unsupported quality claim.
- Delayed labels are required to measure post-release recall and calibration.
- Databricks changes the platform implementation, not the evidence chain you
  practiced locally.

**Evidence used:** registered model version, alias, threshold, reference batch,
simulated current batch, input/score diagnostics, and an operating plan. The
shifted batch is created in memory and is not production data.

**Course completion check:** you can trace one prediction back to a concrete
model version, threshold, input contract, selected candidate, validation
evidence, frozen-test decision, dataset fingerprint, source/dependency evidence,
and monitoring owner.

Continue with `docs/glossary.md`, `docs/resources.md`, and
`docs/databricks-handoff.md`. Run `make check` only when you want the full
contributor verification gate.
"""),
    ],
}
