# Databricks notebook source
# Judge alignment (human-run): before trusting an LLM judge in the gate,
# compare its verdicts against human labels. Target ~75% agreement on 50+
# labeled examples; below that, tighten the judge guidelines and re-check.

# COMMAND ----------

import json

from aai_core import bootstrap
from aai_core.runtime import find_platform_config
from app.judges import judge_model_uri

context = bootstrap()  # discovers aai-platform.yml (env override / upward search)
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

project_root = find_platform_config().parent
cases = json.loads((project_root / "evals" / "data" / "golden_cases.json").read_text())
print(f"{len(cases)} golden cases available for the alignment sample")
