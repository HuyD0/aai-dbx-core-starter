"""Publish the repo's evaluation cases to the Unity Catalog dataset.

The repo files (evals/data/*.json) are the reviewed source of truth; the UC
dataset is the governed, queryable copy used by scheduled evaluations and
shared with other teams. Run on the credentialed path (human or main-branch
job) after case changes merge.
"""

import json
from pathlib import Path

from aai_core import bootstrap
from aai_core.evaluation import EvaluationDatasetManager
from app.config import DATASET_NAME

ROOT = Path(__file__).resolve().parents[1]

context = bootstrap(ROOT / "aai-platform.yml")
records = json.loads((ROOT / "evals" / "data" / "golden_cases.json").read_text("utf-8"))
manager = EvaluationDatasetManager(context=context.tags)
qualified = f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
dataset = manager.create(qualified, records=records)
print({"dataset": qualified, "records": len(records)})
