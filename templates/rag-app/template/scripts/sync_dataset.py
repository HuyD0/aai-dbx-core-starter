"""Publish the release cases to the governed Unity Catalog dataset.

The repo file (evals/data/release_cases.json) is the reviewed source of
truth the gate evaluates; the UC dataset is its governed, queryable,
shareable registration. Run on the credentialed path after case changes
merge — evaluation runs link the dataset by name.
"""

import json
from pathlib import Path

import mlflow

from aai_core import bootstrap
from app.config import DATASET_NAME

ROOT = Path(__file__).resolve().parents[1]

context = bootstrap(ROOT / "aai-platform.yml")
records = json.loads(
    (ROOT / "evals" / "data" / "release_cases.json").read_text("utf-8")
)
qualified = f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
dataset = mlflow.genai.datasets.create_dataset(
    name=qualified,
    tags=context.tags.for_mlflow(),
)
dataset.merge_records(records)
print({"dataset": qualified, "records": len(records)})
