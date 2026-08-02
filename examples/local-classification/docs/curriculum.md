# Beginner-first curriculum

## Starting point

This course assumes basic Python and pandas, plus the ability to run a command
in Terminal. It does not assume machine-learning, statistics, scikit-learn,
MLflow, MLOps, or Databricks knowledge.

A first pass takes about eight to ten hours. Complete it over several sessions
if useful. Each lesson ends by naming the evidence it saved and the prerequisite
for the next lesson.

The course uses one recurring question:

> Given information available at a monthly account snapshot, can a model
> estimate the chance that a fictional subscriber will cancel within 30 days,
> well enough to decide which accounts receive a retention review?

The dataset and costs are teaching assumptions. They are not claims about real
customers or universal release standards.

## Learning sequence

| Lesson | New ideas, in learning order | What you do | Evidence or skill you leave with | Time |
|---:|---|---|---|---:|
| 00 | Environment, kernel, course state, model lifecycle | Run a preflight, make a first example prediction, and locate local course state | A known-good environment and a plain-language lifecycle map | 25–35 min |
| 01 | Feature, target, class, probability, threshold, prediction time, action, FP/FN | Work through small customer examples and define what the model may know at prediction time | A problem, action, feature, target, and cost contract | 35–45 min |
| 02 | Schema, missingness, prevalence, provenance, manifest, digest | Inspect real rows and types; implement visible quality checks before using the reusable validator | A checked train/validation dataset and lineage manifest | 45–55 min |
| 03 | Train/validation/test roles, time split, leakage, preprocessing leakage | Inspect and verify the declared split; compare a safe feature set with a post-outcome leak | Time partitions and a leakage audit without opening test labels | 40–50 min |
| 04 | Baseline, confusion matrix, accuracy, precision, recall, average precision, MLflow run | Compute a majority baseline, derive metrics from actual error counts, then record the same comparator | A benchmark run that candidate models must beat | 50–60 min |
| 05 | Imputation, scaling, one-hot encoding, Pipeline, fit, probability | Transform rows visibly, build one complete Pipeline, and train logistic regression | An in-memory fitted Pipeline whose operations are understood | 60–75 min |
| 06 | Ranking versus action, model selection, threshold trade-off, cost constraint | Compare candidates with the baseline, move a threshold, then log the controlled comparison | One model and one threshold fixed using validation data only | 60–75 min |
| 07 | Frozen test, predeclared gate, slice check, adopt/reject | Restate the rules, make one final evaluation, and compare observed values with each rule | A non-crashing release decision linked to the exact model and data | 50–65 min |
| 08 | Logged model, Registry, version, signature, input example, alias, inference | Register only an adopted artifact, inspect its contract and input example, reload it, and score representative rows | A concrete registered version and reproducible prediction schema | 45–60 min |
| 09 | Reference/current batch, input and score drift, delayed outcomes, platform mapping | Simulate a known shift, interpret diagnostics, design responses, then map local components to Databricks | A monitoring plan and local-to-Databricks concept map | 60–75 min |

Terms are defined in the [glossary](glossary.md). The
[resource path](resources.md) follows this same order.

## Why the data is split three ways

The course protects three different questions:

```text
training data    → what patterns can the model learn?
validation data  → which model and action threshold should we choose?
test data        → did those already-fixed choices pass the release rules?
```

Training data may fit preprocessors and models. Validation data may compare
models and choose the threshold. Test labels stay unopened until lesson 07.
After test results influence a feature, model, threshold, or gate, that test set
is no longer untouched evidence for the changed system.

The packaged workflow also records dataset, code/dependency, selection-policy,
and gate-policy digests. Those links prevent a release decision for one model or
policy from being silently reused for another. The notebooks introduce this
only after the learner understands the three data roles.

## The model question and the action question are different

The model first produces a probability-like score. Candidate selection asks
whether positive examples tend to rank above negative examples; this course
uses validation average precision for that comparison because churn is the less
common class.

The retention action needs a yes/no boundary. The threshold is chosen separately
on validation data using illustrative costs plus minimum precision and recall.
A model can rank well and still have a poor action threshold, or produce poorly
calibrated probabilities. Lesson 06 makes those distinctions visible before the
test set is opened.

## Completion checks

You have completed the core course when you can explain, in your own words:

1. What one output probability means and how a threshold turns it into an action.
2. Which column is the target, which value is positive, and which information is
   unavailable at prediction time.
3. Why training, validation, and test data answer different questions.
4. How fitting preprocessing before the split leaks information.
5. Why an all-negative baseline can have high accuracy and zero recall.
6. How to read the four cells of a confusion matrix and calculate precision and
   recall from them.
7. Why the model-selection metric does not itself choose the action threshold.
8. What an MLflow experiment, run, parameter, metric, artifact, and logged model
   preserve—and why MLflow does not preserve your raw source data for you.
9. Why the final test rules are declared before test labels are examined.
10. Why a model alias is a movable label while a model version is concrete.
11. Why input drift is a reason to investigate but cannot prove outcome quality
    changed before true labels arrive.
12. Which local data, tracking, registry, execution, serving, and monitoring
    components have Databricks counterparts.

## Practice policy

Practice cells operate on in-memory copies or a clearly named scratch area.
They must not silently replace release evidence. If you intentionally change
the generator, split, feature contract, candidate, threshold policy, or gate,
start a new course attempt first:

```bash
make course-reset
```

The command moves the old `.aai/course-v2/` state to a recoverable backup.

After the core course, suitable extensions—in increasing difficulty—are:

1. Add one derived feature that is genuinely available at prediction time and
   rerun the validation comparison in a fresh course attempt.
2. Add a third estimator while holding the data, features, seed, metric, and
   threshold procedure fixed.
3. Plot confidence intervals for one test metric and design a reviewed policy
   for insufficient evidence before changing the implemented gate vocabulary.
4. Replace the synthetic source only after writing a dataset card covering
   ownership, license, privacy, time semantics, and an immutable version.

Grouped splitting, probability calibration with out-of-fold predictions, and
bootstrap release policies are advanced topics linked from
[resources.md](resources.md); they are not assumed beginner knowledge.
