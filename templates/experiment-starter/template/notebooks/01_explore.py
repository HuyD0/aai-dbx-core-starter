# Databricks notebook source
# Exploration only — production logic lives in src/app and runs as a job.

# COMMAND ----------

from pathlib import Path

from aai_core import bootstrap
from app.experiment import DEFAULT_SEED, dataset_rows, evaluate_rows, load_dataset

context = bootstrap(Path.cwd().parent / "aai-platform.yml")
print({"application": context.tags.application})
print({"experiment": context.settings.experiment_name})

# COMMAND ----------

data = load_dataset()
print(data.head())

# COMMAND ----------

# Try seeds interactively; promote anything reusable into src/app with tests.
for seed in (DEFAULT_SEED, DEFAULT_SEED + 1):
    print(seed, evaluate_rows(dataset_rows(data), seed=seed))
