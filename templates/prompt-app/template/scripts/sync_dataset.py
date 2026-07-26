"""Publish the release cases to the governed Unity Catalog dataset.

The repo file (evals/data/release_cases.json) is the reviewed source of
truth the gate evaluates; the UC dataset is its governed, queryable,
shareable registration. Run on the credentialed path after case changes
merge — evaluation runs link the dataset by name.
"""

import json
from pathlib import Path

from aai_core import bootstrap
from aai_core.evaluation import EvaluationDatasetManager
from app.config import DATASET_NAME

ROOT = Path(__file__).resolve().parents[1]

context = bootstrap(ROOT / "aai-platform.yml")
records = json.loads(
    (ROOT / "evals" / "data" / "release_cases.json").read_text("utf-8")
)
manager = EvaluationDatasetManager(context=context.tags)
qualified = f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
manager.create(qualified, records=records)
print({"dataset": qualified, "records": len(records)})
