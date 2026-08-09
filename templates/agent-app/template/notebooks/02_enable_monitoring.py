# Databricks notebook source
# Production monitoring (Beta): register sampled scorers on the deployed
# agent's traces. scorer.register()/.start() must run FROM A NOTEBOOK (the
# service serializes notebook code), which is why this is not wrapped in
# aai-core or automated in CI. Mind the per-experiment scorer cap and keep
# sample rates low — judges cost money on every sampled trace.

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import Safety, ScorerSamplingConfig

from aai_core import bootstrap
from aai_core.evaluation import judge_model_uri

context = bootstrap()  # discovers aai-platform.yml (env override / upward search)
judge_model = judge_model_uri(context.settings)
print({"experiment": context.settings.effective_experiment_name})


# COMMAND ----------

mlflow.set_experiment(context.settings.effective_experiment_name)

safety = Safety(model=judge_model).register(name="production_safety")
safety.start(sampling_config=ScorerSamplingConfig(sample_rate=0.1))
print("registered production_safety at 10% sampling")

# COMMAND ----------

# Attach end-user/expert feedback to the originating trace so production
# failures can be curated into evals/data/release_cases.json:
#
#   mlflow.log_feedback(trace_id=..., name="user_helpful", value=False,
#                       rationale="answer ignored the order id")
