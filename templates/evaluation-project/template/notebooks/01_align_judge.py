# Databricks notebook source
# Judge alignment (human-run): before trusting an LLM judge in the gate,
# compare its verdicts against human labels. Target ~75% agreement on 50+
# labeled examples; below that, tighten the judge guidelines and re-check.

# COMMAND ----------

import json
from pathlib import Path

from aai_core import bootstrap
from app.judges import judge_model_uri

context = bootstrap(Path.cwd().parent / "aai-platform.yml")
print({"judge_model": judge_model_uri(context.settings)})

# COMMAND ----------

# 1. Score a sample with the judge (mlflow.genai.evaluate on golden cases).
# 2. Have a human label the same sample (correct / incorrect).
# 3. Record each human label as feedback on the evaluation trace, using the
#    JUDGE'S name so agreement is queryable:
#
#    import mlflow
#    mlflow.log_feedback(
#        trace_id=trace_id, name="correctness_human",
#        value=True, rationale="matches documented policy",
#    )
#
# 4. Compare judge vs human agreement before raising gate thresholds.

cases = json.loads(
    (Path.cwd().parent / "evals" / "data" / "golden_cases.json").read_text()
)
print(f"{len(cases)} golden cases available for the alignment sample")
