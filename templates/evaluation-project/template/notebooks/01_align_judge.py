# Databricks notebook source
# Judge calibration (human-run): before trusting an LLM judge in the gate,
# measure it against human labels. The auditable claim is never "the agent
# scores 0.87" — it is "0.87 under a judge that agrees with our reviewers at
# Cohen's kappa >= 0.60 on a named label set, against a measured human
# ceiling". `agentkit judge calibrate` computes exactly that record; this
# notebook collects the labels it needs and runs it.
#
# Calibrate on the calibration split only; measure once on the held-out
# validation split. Judge releases move in their own change, never in the
# same commit as an agent change.

# COMMAND ----------

import json
from random import Random

from mlflow.entities import AssessmentSource, AssessmentSourceType

from aai_core.runtime import find_platform_config
from app.judges import judge_model_uri, judge_scorers

# The judges the shared registry selects for this project's dataset.
judge_uri = judge_model_uri()
scorers = judge_scorers()
print({"judge_model": judge_uri, "scorers": [judge.name for judge in scorers]})

# COMMAND ----------

project_root = find_platform_config().parent
cases = json.loads((project_root / "evals" / "data" / "golden_cases.json").read_text())
if len(cases) < 2:
    raise ValueError("Judge calibration needs at least two labeled cases")

# A deterministic split makes reruns comparable. Do not tune judge
# instructions or memories against validation_cases.
shuffled = list(cases)
Random(20260726).shuffle(shuffled)
split_index = max(1, min(len(shuffled) - 1, round(len(shuffled) * 0.7)))
calibration_cases = shuffled[:split_index]
validation_cases = shuffled[split_index:]
print(
    {
        "calibration_cases": len(calibration_cases),
        "validation_cases": len(validation_cases),
        "minimum_labels_for_calibrate": 20,
        "target_kappa": 0.60,
    }
)

# COMMAND ----------

# Human feedback must carry explicit provenance and use the SAME assessment
# name as the judge being calibrated. Use a non-personal reviewer group id;
# never place reviewer email addresses or case content in tags.
human_source = AssessmentSource(
    source_type=AssessmentSourceType.HUMAN,
    source_id="group:domain-reviewers",
)
print(
    {
        "human_source_type": human_source.source_type,
        "human_source_id": human_source.source_id,
    }
)

# Label collection:
#
# 1. Run `agentkit compare` (or mlflow.genai.evaluate directly) over
#    calibration_cases with the `scorers` above, so every trace carries the
#    judge's own assessment.
# 2. Have reviewers label the resulting traces — in the review app or here —
#    using the judge's exact registry name and its categorical value:
#
#    mlflow.log_feedback(
#        trace_id=trace_id,
#        name="pension_domain_policy",
#        value="yes",  # or "no"
#        rationale="No private contact data; official support path offered.",
#        source=human_source,
#    )
#
# Two or more reviewers labelling an overlap slice is what makes the human
# ceiling measurable: a judge cannot be more consistent than the people
# defining the target, and a low ceiling means the rubric is the problem.

# COMMAND ----------

# Export the labelled traces into the file `agentkit judge calibrate` reads:
# one entry per example, the judge's verdict plus every human verdict.

import mlflow

JUDGE_NAME = "pension_domain_policy"
labels = []
traces = mlflow.search_traces(max_results=500, return_type="list")
for trace in traces:
    assessments = getattr(trace.info, "assessments", None) or []
    named = [a for a in assessments if getattr(a, "name", None) == JUDGE_NAME]
    judge_values = [
        a
        for a in named
        if getattr(getattr(a, "source", None), "source_type", None) != "HUMAN"
    ]
    human_values = [
        a
        for a in named
        if getattr(getattr(a, "source", None), "source_type", None) == "HUMAN"
    ]
    if not judge_values or not human_values:
        continue
    labels.append(
        {
            "example_id": trace.info.trace_id,
            "judge_value": judge_values[0].feedback.value,
            "annotations": [
                {
                    "annotator": str(a.source.source_id),
                    "value": a.feedback.value,
                }
                for a in human_values
            ],
        }
    )

labels_path = project_root / "evals" / "data" / "calibration_labels.json"
labels_path.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")
print({"labelled_examples": len(labels), "written_to": str(labels_path)})

# COMMAND ----------

# Measure and record. The command computes percent agreement, chance-adjusted
# Cohen's kappa against the reviewer consensus, and the pairwise human
# ceiling, then writes the committed record `agentkit gate` can demand:
#
#   agentkit judge calibrate \
#       --scorer pension_domain_policy \
#       --labels evals/data/calibration_labels.json \
#       --decided-by group:domain-reviewers
#
# Exit 0 records a PASSING calibration (kappa >= 0.60); exit 2 records the
# failure honestly — most low-kappa judges are under-specified rubrics, so
# fix the rubric before touching the model. Commit
# evals/judges/pension_domain_policy.json with the judge release, and set
# `integrity.require_calibration: true` in agentkit.yaml once every judge
# in use carries a passing record.
#
# The judge's instructions are a versioned platform asset in the Unity
# Catalog Prompt Registry, and its threshold lives in the shared scorer
# registry — so calibration evidence goes to the platform team, who publish
# a new judge prompt version and give the scorer a gating threshold. A
# project never edits a judge's instructions locally; that is what keeps one
# team's 0.8 comparable with another's. After any judge release,
# re-establish the baseline and the judge anchors
# (`agentkit compare --establish-baseline`) in a change of its own.

# COMMAND ----------

# OPTIONAL / EXPERIMENTAL: MemAlign remains deliberately opt-in. It requires
# DSPy plus an approved, keyless embedding model; neither is added by this
# template. Run it only in a controlled calibration experiment, never on the
# validation split, and re-run `agentkit judge calibrate` on held-out labels
# before the aligned judge is released.
#
# from mlflow.genai.judges.optimizers import MemAlignOptimizer
#
# domain_policy = next(s for s in scorers if "domain_policy" in s.name)
# optimizer = MemAlignOptimizer(
#     reflection_lm=judge_uri,
#     embedding_model="databricks:/<approved-embedding-model>",
# )
# aligned_domain_policy = domain_policy.align(
#     traces=calibration_traces,
#     optimizer=optimizer,
# )
#
# `calibration_traces` must contain HUMAN assessments named
# "pension_domain_policy" from `human_source`. Do not add API keys to make
# this optional path work.
