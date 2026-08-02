# Curriculum and completion rubric

The course is designed for short, focused sessions. A first pass takes roughly
four to six hours; the exercises and source-code reading can extend it into a
two-day workshop.

| Lesson | Main question | Evidence produced | Typical time |
|---:|---|---|---:|
| 00 | What makes training an MLOps lifecycle rather than one `fit()` call? | Local topology and learning hypothesis | 20 min |
| 01 | What exactly is predicted, when, and for what action? | Problem, feature, target, cost, and gate contract | 30 min |
| 02 | Can the data be trusted for this experiment? | Quality report, prevalence, missingness, and time drift | 35 min |
| 03 | How do we prevent preprocessing and future information from leaking? | Time partitions, manifest, forbidden-feature check | 35 min |
| 04 | Is a trained model better than doing almost nothing? | Logged dummy baseline | 30 min |
| 05 | How do pipelines make training and inference consistent? | Logged logistic and forest candidate runs | 45 min |
| 06 | How are model selection and decision thresholds different? | Validation-only selection and threshold evidence | 40 min |
| 07 | Did the exact selected artifact pass the untouched release test? | Metrics, classic MLflow evaluation, slices, and gate | 45 min |
| 08 | What is promoted, and how is it reloaded safely? | Registered version, tags, signature, and `champion` alias | 35 min |
| 09 | What changes after release and when moving to Databricks? | Inference contract, drift report, and handoff map | 35 min |

## Experiment question

The controlled question is:

> On the same historical training and validation periods, does either declared
> sklearn pipeline rank likely 30-day churn better than a no-skill prior, while
> supporting a validation-selected action threshold that passes the frozen test
> quality, cost, calibration, and slice checks?

`average_precision` is the primary candidate-selection metric because churn is
the minority class and the experiment asks about ranking likely positives.
Thresholded recall, precision, and expected action cost answer a different
question: which accounts should receive an intervention under the illustrative
false-negative and false-positive costs in `configs/project.yaml`.

The numbers in that file are teaching assumptions, not universal standards.
In a real project they must come from product, operations, risk, and affected
stakeholders before the team evaluates a model.

## Evidence boundaries

- Training data fits preprocessors and estimators.
- Validation data chooses the model and threshold.
- Test data is opened only in lesson 07, after choices are fixed.
- The exact candidate artifact selected in lesson 06 is loaded for lesson 07;
  it is not silently refit.
- Selection and gate evidence carry digests of their code, dependency lock, and
  declared policy. Promotion fails closed if those links do not match.
- A failed gate leaves the registry alias unchanged.
- An existing decision consumes that frozen-test dataset version. A different
  candidate or policy requires a new frozen-test version rather than another
  look at the same labels.
- Once a learner uses test results to change the generator, features, model,
  threshold, or gate, that partition has become development data. Create a new
  frozen test version before making another release claim.

## Completion rubric

You are finished when you can explain, without referring to the notebook text:

1. Why the split occurs before preprocessing and why a sklearn `Pipeline`
   enforces that rule inside a fit.
2. Why the positive label is `1` and why accuracy hides the no-skill baseline's
   zero recall.
3. Why PR-AUC selects a candidate but does not choose an operating threshold.
4. Why calibration, ranking, and thresholded action quality are related but
   distinct properties.
5. Which data, code, environment, model, metric, and decision evidence MLflow
   preserves—and which raw data it does not preserve for you.
6. Why a model signature and input example are release contracts, not UI
   decoration.
7. Why registry aliases are mutable pointers and a deployment should record the
   concrete resolved version.
8. Why input drift alone cannot prove that the model's real outcome quality fell.
9. Which local components map to Unity Catalog tables, hosted MLflow tracking,
   Models in Unity Catalog, jobs, serving, inference tables, and data profiling.

## Extension exercises

- In a disposable copy *before lesson 07*, tighten one gate and confirm that the
  decision becomes `reject` and `champion` does not move. Do not reuse a test
  partition whose result you already saw as fresh release evidence.
- Add a third estimator without changing any other experimental variable.
- Replace the time split with an entity-grouped split and explain which
  production assumption changed.
- Add probability calibration using training-fold out-of-fold predictions;
  never fit the calibrator on the frozen test set.
- Add bootstrap confidence intervals and decide what result should be
  `inconclusive` rather than `reject`.
- Replace the synthetic generator with a reviewed dataset card, immutable source
  URI, license, checksum, and prediction-time feature audit.
