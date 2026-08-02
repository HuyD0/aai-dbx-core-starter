# Notebook guide

These ten notebooks are one course, not ten unrelated demos. Start JupyterLab
from the project directory so it uses the locked environment and isolated course
state:

```bash
make doctor
make notebook
```

The kernel shown in the top-right corner must be **AAI Local Classification**.
Run a cell with **Shift+Enter**. If execution order becomes confusing, use
**Kernel → Restart Kernel and Run All Cells**.

## Lesson order

1. `00_start_here.ipynb` — verify the environment, learn notebook mechanics,
   and see the complete classification journey in plain language.
2. `01_problem_and_data_contract.ipynb` — learn feature, target, positive class,
   probability, threshold, prediction time, action, and error cost.
3. `02_data_quality_and_eda.ipynb` — meet the rows and columns, then check types,
   labels, duplicates, missing values, class balance, and data provenance.
4. `03_leakage_safe_splits.ipynb` — inspect and verify the time-based train,
   validation, and test split, then see why future information makes an offline
   score dishonest.
5. `04_baseline.ipynb` — calculate a no-skill result, read a confusion matrix,
   and understand accuracy, precision, recall, and average precision.
6. `05_pipeline_and_training.ipynb` — preprocess numeric and categorical data,
   then fit and inspect one in-memory logistic-regression Pipeline.
7. `06_model_selection_and_threshold.ipynb` — choose a model by ranking quality,
   then separately choose the probability threshold that triggers an action.
8. `07_frozen_test_gate.ipynb` — make one release evaluation on untouched test
   data and compare every observed value with a predeclared rule.
9. `08_registry_and_inference.ipynb` — register only an approved model, resolve
   its version and alias, reload it, and score representative feature-only rows.
10. `09_monitoring_and_databricks.ipynb` — compare a reference and current batch,
    separate input change from outcome quality, and then map the local lifecycle
    to Databricks.

Each lesson follows the same learning loop:

```text
explain → predict → run → interpret → practice → self-check → recap
```

The notebooks are stored without outputs, so blank output areas are normal.
Every important cell states the expected result or range. Avoid running ahead:
later lessons reuse evidence created earlier.

## Course state and starting over

Generated data, MLflow runs, registered models, and lesson evidence live under
`.aai/course-v2/` in the project directory. They are not committed to Git.

To start a new attempt, stop Jupyter and run:

```bash
make course-reset
make notebook
```

The reset moves the previous state to a backup instead of permanently deleting
it. For setup errors, wrong-kernel help, and expected success messages, return to
the [main course README](../README.md#troubleshooting). Definitions are in the
[glossary](../docs/glossary.md).
