# Learning model card: subscription churn classifier

## Intended use

Demonstrate a local binary-classification MLOps lifecycle. The model estimates a
synthetic account's probability of churn within 30 days so a fictional team can
study ranking, thresholding, and release decisions.

## Not intended for

Real customer decisions, automated targeting, financial impact estimates,
fairness claims, or deployment. The data is synthetic and does not represent a
population, business process, legal context, or intervention effect.

## Data

The deterministic generator creates 24 monthly cohorts with mixed numeric and
categorical features, modest missingness, minority positive labels, and later
cohort shift. Data is split in time: 18 months train, 3 months validation, and 3
months frozen test. A manifest records each CSV's rows, dates, and SHA-256 digest
while withholding test prevalence until the release gate.

Identifiers, snapshot time, target, cancellation reason, and account-closure
time are forbidden model features. `cancellation_reason` is available only as an
intentional leakage demonstration and is never generated into the real model
matrix.

## Modeling and decision policy

A prior-only dummy baseline is compared with logistic regression and a small
random forest. Both models use a sklearn `Pipeline` containing
train-fitted imputation, scaling, and one-hot encoding. Candidate selection uses
validation average precision, with a predeclared tolerance that prefers the
simpler logistic model when its score is practically tied with a more complex
candidate. A threshold is then selected on validation data using illustrative
false-negative/false-positive costs plus minimum precision and recall.

## Evaluation

The exact selected artifact is evaluated on a frozen later-time test set once.
The gate includes average precision, recall, Brier score, expected cost, and
recall gaps across plan and acquisition-channel slices. Precision, F1, ROC-AUC,
log loss, confusion counts, and predicted-positive rate are supporting evidence.

The slice check is an operational diagnostic, not a fairness assessment. The
synthetic categories are not asserted to be protected or harmless proxies in a
real context.

## Known limitations

- Synthetic relationships and gate values are authored for learning.
- No causal effect of a retention action is modeled.
- Temporal shift is simple and not representative of real drift.
- Metrics are point estimates; the base course does not add confidence intervals.
- No privacy, consent, legal, capacity, or human-review policy is validated.
- Local SQLite and a Mac environment do not establish production readiness.

## Release rule

Register the exact logged artifact and move `champion` only if every predeclared
test check passes. Dataset, selected run, model ID, code/dependency policy, gate
policy, threshold, and test run must agree before promotion. Any change
influenced by test results requires a newly frozen test version before making
another release claim.
