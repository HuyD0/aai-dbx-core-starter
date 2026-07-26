# Databricks notebook source
# Exploration only — production logic lives in src/app and runs as a job.

# COMMAND ----------

from aai_core import bootstrap
from app.experiment import DEFAULT_SEED, dataset_rows, evaluate_rows, load_dataset

context = bootstrap()  # discovers aai-platform.yml (env override / upward search)
print({"application": context.tags.application})
print({"experiment": context.settings.effective_experiment_name})

# COMMAND ----------

data = load_dataset()
print(data.head())

# COMMAND ----------

# Try seeds interactively; promote anything reusable into src/app with tests.
for seed in (DEFAULT_SEED, DEFAULT_SEED + 1):
    print(seed, evaluate_rows(dataset_rows(data), seed=seed))
