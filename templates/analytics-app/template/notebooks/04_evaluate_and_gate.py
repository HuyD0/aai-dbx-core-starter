# Databricks notebook source
# Layer 4, validation: what the golden set covers, what the deterministic
# scorers grade, and what the release gate enforces. Everything here runs
# offline; the credentialed twin of this notebook is evals/evaluate.py.

# COMMAND ----------

import json
from collections import Counter
from pathlib import Path

from app.scorers import score_all

ROOT = next(
    parent
    for parent in [Path.cwd(), *Path.cwd().parents]
    if (parent / "semantics" / "semantic_model.yml").exists()
)
golden = json.loads((ROOT / "evals" / "data" / "golden_cases.json").read_text())
sheet = json.loads((ROOT / "evals" / "data" / "answer_sheet.json").read_text())
answers = {record["question"]: record["answer"] for record in sheet}

# COMMAND ----------

# Eval-set design: two families of questions. Stakeholder lookups pin exact
# values to the seed snapshot; long-tail cases cover the published failure
# modes — encoding traps (concept-to-entity ambiguity), clarification,
# freshness, sanctioned raw fallback, and out-of-scope refusal. The shipped
# set is a floor: harvest corrections into new cases and grow toward dozens
# per domain (returns diminish beyond that).
print(Counter(case["expectations"]["category"] for case in golden))

# COMMAND ----------

# Ground truth is pinned to the snapshot, never to live data: the offline
# gate recomputes every expected_value through the semantic compiler over
# seed_data.json, so a drifting seed or definition fails CI immediately.
example = golden[0]
print(example["inputs"]["question"])
print(example["expectations"]["expected_query"])
print({"pinned_value": example["expectations"]["expected_value"]})

# COMMAND ----------

# The five deterministic scorers grade provenance discipline per answer.
totals = Counter()
for case in golden:
    scores = score_all(answers[case["inputs"]["question"]], case["expectations"])
    totals.update(scores)
means = {name: value / len(golden) for name, value in totals.items()}
print(means)

# COMMAND ----------

# The gate: ~90% floors on accuracy-style metrics (the per-domain launch
# bar this architecture was published with), hard 1.0 on provenance and
# read-only, a monitored floor on semantic share, plus judge metrics,
# baseline regression, and cost coverage in the credentialed tier.
gate = json.loads((ROOT / "evals" / "gate_config.json").read_text())
for rule in gate["thresholds"]:
    print(rule)

# COMMAND ----------

# Telemetry recorded by evals/evaluate.py per governed run: semantic-model
# version, knowledge digest, model + judge identity, git provenance, token
# usage per answer (tokens/mean_total, tokens/mean_review, cost/coverage),
# per-case pass/fail in the native result, and the aai.gate_passed verdict.
# That is enough to chart accuracy and cost over time in MLflow.
print("run `python evals/evaluate.py` on the credentialed path")
