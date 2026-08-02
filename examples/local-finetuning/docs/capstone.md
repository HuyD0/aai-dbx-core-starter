# Capstone: application production-readiness reviewer

The capstone is domain-specific and independent of Kaggle. A versioned policy
engine generates every critical label, severity, remediation ID, and readiness
decision. The hybrid lab lets a language model vary explanation wording, but it
never originates or changes ground truth.

## Input schema 1.0.0

The simplified manifest contains:

- application name, non-personal owner group, and business domain;
- lifecycle: `development`, `staging`, or `production`;
- immutable model revision and pin flag;
- versioned evaluation dataset, last-run timestamp, and threshold result;
- required cost tags and budget policy;
- production-support owner, rollback, and monitoring controls;
- data classifications and approvals;
- framework and known high-severity finding count;
- explicit flags for an authorized external lookup or human risk judgment.

Unknown fields are rejected by the strict validated schema. The untrusted-input
review path still accepts a raw mapping so the frozen test set can record an
explainable schema failure. `candidate` appears only as a legacy-invalid test
value; it is not a supported lifecycle stage.

## Output schema 1.0.0

```json
{
  "schema_version": "1.0.0",
  "manifest_schema_version": "1.0.0",
  "policy_version": "1.0.0",
  "status": "not_ready",
  "as_of": "2026-07-31T12:00:00Z",
  "checks": [
    {
      "name": "evaluation_dataset",
      "result": "fail",
      "severity": "high",
      "evidence": "No evaluation dataset is registered.",
      "remediation_id": "evaluation.register_dataset",
      "remediation_text": "Add a versioned regression evaluation dataset.",
      "provenance": {
        "policy_version": "1.0.0",
        "rule_id": "evaluation_dataset",
        "rule_kind": "deterministic",
        "source_fields": ["evaluation_dataset"],
        "facts_origin": "manifest"
      }
    }
  ]
}
```

Every check records the generating rule and fact origin. External lookups and
human judgment return `review`, route the application, and never assert an
unobserved registry or risk fact.

## Compact model output 1.0.0

The full deterministic review averages roughly 1,560 tokenizer tokens because
it preserves all 19 checks and their provenance. That is the immutable ground
truth, but it is a poor generation target for a tiny local model. The
model-facing projection contains the overall status and only actionable
non-pass checks:

```json
{
  "schema_version": "1.0.0",
  "status": "not_ready",
  "checks": [
    {
      "name": "evaluation_dataset",
      "result": "fail",
      "severity": "high",
      "remediation_id": "evaluation.register_dataset"
    }
  ]
}
```

Pass checks are forbidden in this compact schema. The evaluator derives the
expected projection from the full policy output and measures JSON/schema
validity, status accuracy, check-result and severity accuracy, missing and
invented checks, exact-review rate, performance, slices, and bounded errors.
This compression does not give the model authority to change policy decisions.

## Policy catalog 1.0.0

| Rule | Kind | Failure severity |
|---|---|---|
| Closed manifest schema | Deterministic | High |
| Ownership | Deterministic | High |
| Business domain | Deterministic | Medium |
| Supported lifecycle | Deterministic | High |
| Pinned model revision | Deterministic | High |
| Evaluation dataset | Deterministic | High |
| Evaluation no older than 30 days | Policy | High |
| Evaluation thresholds passed | Policy | Critical |
| Cost tags | Deterministic | High |
| Supported budget policy | Policy | High |
| Production-support owner | Deterministic | High |
| Rollback plan | Deterministic | High |
| Monitoring | Deterministic | High |
| Non-conflicting data classification | Policy | High |
| Required approvals | Policy | High |
| Supported framework | Policy | High |
| No known high-severity findings | Deterministic | Critical |
| Registry verification | External lookup | Review route |
| Risk judgment | Human judgment | Review route |

This classification is part of the lesson: not every readiness question is a
model task, and the model must not invent external state.

## Deterministic dataset

`make capstone` writes:

- 400 training records;
- 100 validation records;
- 150 frozen test records;
- SHA-256 hashes for content and ordered example IDs.

Seed 42 produces stable content IDs. The frozen test set contains fully ready,
missing ownership/evaluation/budget/tags/model revision/rollback, stale
evaluation, one critical failure, multiple failures, conflicting metadata,
unknown fields, invalid lifecycle, long and minimal manifests, unexpected
nulls, and unseen combinations of known failures. It also covers every policy,
external, and human-review rule.

Application-context families are assigned to one split before slice variants
are expanded. Removing the record-specific application name therefore leaves
no duplicate manifest in another split. This group-style partition matters:
randomly naming otherwise identical synthetic rows would make a frozen test
look independent while mostly measuring memorization.

Each expected output comes from the exact policy engine. No LLM creates labels.
Any language variation added later must be validated back against those
immutable results and reviewed before it becomes training data.

## Baselines and architecture decision

Compare:

1. The deterministic policy engine.
2. Untouched tiny model with a basic prompt.
3. Untouched model with a constrained prompt.
4. Untouched model with training-only few-shot examples.
5. Fine-tuned model.
6. Hybrid policy checks plus optionally model-rendered explanations.

The complete comparison is executable offline:

```bash
make capstone
make capstone-train-smoke
make capstone-train
make capstone-evaluate
```

The final command uses all 150 frozen examples. A LoRA decision remains
`inconclusive` for a debug subset or missing prompt baseline, and is `adopt`
only if the change beats basic, strong, and training-only few-shot untouched
model evidence while passing absolute structure and invented-check gates.
The complete tiny-model comparison is intentionally a longer study run. For a
quick report-only rehearsal, use
`aai-finetune --offline capstone-evaluate --limit 5 --max-tokens 256`.

For fully deterministic rules, the policy engine is the accuracy ceiling. The
hybrid helper passes a frozen check to a renderer and gives it no channel to
change status, result, severity, rule identity, or remediation ID. Empty or
failed rendering falls back to policy text.

The runnable hybrid comparison exercises the local tiny model only for
explanation wording. Those explanations are saved as report-only evidence for
human review; deterministic decisions are scored separately and stay at the
policy ceiling.

A likely production shape is therefore:

```text
deterministic validation and policy decisions
                    +
tiny model for bounded normalization or explanation wording
```

The capstone should reject a model-only architecture when it adds uncertainty
without measurable value.

## Later platform mapping

The local files intentionally require neither Delta Lake nor Unity Catalog.
Their immutable raw inputs, SHA-256 hashes, stable IDs, versioned configuration,
split manifest, MLflow artifacts, and dataset cards can later map to governed
Delta/Unity Catalog tables and volumes, managed evaluation datasets, and
Databricks lineage without changing the logical records or policy boundary.
