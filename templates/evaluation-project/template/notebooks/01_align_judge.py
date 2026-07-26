# Databricks notebook source
# Judge alignment (human-run): before trusting an LLM judge in the gate,
# compare its verdicts against human labels. Tune only on the calibration
# split, then measure once on the held-out validation split. Target at least
# 75% agreement on 50+ total labeled examples before gating a release.

# COMMAND ----------

import json
from random import Random

from mlflow.entities import AssessmentSource, AssessmentSourceType

from aai_core import bootstrap
from aai_core.runtime import find_platform_config
from app.judges import judge_model_uri, judge_scorers

context = bootstrap()  # discovers aai-platform.yml (env override / upward search)
judge_uri = judge_model_uri(context.settings)
scorers = judge_scorers(context.settings)
print({"judge_model": judge_uri, "scorers": [judge.name for judge in scorers]})

# COMMAND ----------

project_root = find_platform_config().parent
cases = json.loads((project_root / "evals" / "data" / "golden_cases.json").read_text())
if len(cases) < 2:
    raise ValueError("Judge alignment needs at least two labeled cases")

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
        "minimum_recommended_labels": 50,
        "target_validation_agreement": 0.75,
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

# Calibration workflow:
#
# 1. Run mlflow.genai.evaluate() on calibration_cases with the application
#    predict_fn and `scorers` above.
# 2. Have reviewers label the resulting traces. For the native Guidelines
#    scorer, use its exact name ("domain_policy") and categorical value:
#
#    mlflow.log_feedback(
#        trace_id=trace_id,
#        name="domain_policy",
#        value="yes",  # or "no"
#        rationale="No private contact data; official support path offered.",
#        source=human_source,
#    )
#
# 3. Compare judge assessments with HUMAN assessments. Refine the written
#    DOMAIN_POLICY_GUIDELINES only against calibration_cases.
# 4. Freeze the scorer, evaluate validation_cases once, and require >= 75%
#    agreement before relying on domain_policy/mean in the release gate.
# 5. Keep validation failures held out for the next scorer version; do not
#    repeatedly tune against the same validation labels.
# 6. Record the scorer version, judge model, split manifest, agreement result,
#    and approval. Only then move domain_policy/mean from
#    report_only_judge_metrics to judge_metrics and add its threshold.

# COMMAND ----------

# OPTIONAL / EXPERIMENTAL: MemAlign remains deliberately opt-in. It requires
# DSPy plus an approved, keyless embedding model; neither is added by this
# template. Run it only in a controlled calibration experiment, never on the
# validation split, and register/version the resulting judge before use.
#
# from mlflow.genai.judges.optimizers import MemAlignOptimizer
#
# domain_policy = next(s for s in scorers if s.name == "domain_policy")
# optimizer = MemAlignOptimizer(
#     reflection_lm=judge_uri,
#     embedding_model="databricks:/<approved-embedding-model>",
# )
# aligned_domain_policy = domain_policy.align(
#     traces=calibration_traces,
#     optimizer=optimizer,
# )
#
# `calibration_traces` must contain HUMAN assessments named "domain_policy"
# from `human_source`. Do not add API keys to make this optional path work.
