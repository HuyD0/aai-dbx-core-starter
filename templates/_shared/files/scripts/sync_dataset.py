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


def main() -> None:
    context = bootstrap(ROOT / "aai-platform.yml")
    records = json.loads(
        (ROOT / "evals" / "data" / "release_cases.json").read_text("utf-8")
    )
    qualified = f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
    experiment = mlflow.set_experiment(context.settings.effective_experiment_name)

    try:
        dataset = mlflow.genai.datasets.get_dataset(name=qualified)
    except Exception as error:
        if not _is_missing_dataset(error):
            raise
        # Unity Catalog evaluation datasets do not support tags. Ownership,
        # lifecycle, and cost metadata live on the associated experiment/run.
        dataset = mlflow.genai.datasets.create_dataset(
            name=qualified,
            experiment_id=experiment.experiment_id,
        )

    associated = {str(value) for value in (dataset.experiment_ids or [])}
    if str(experiment.experiment_id) not in associated:
        raise RuntimeError(
            f"Dataset {qualified!r} is not associated with experiment "
            f"{experiment.experiment_id!r}; create a new governed dataset name "
            "for this application instead of silently crossing experiments."
        )
    dataset.merge_records(records)
    # Databricks may return a wrapper whose digest has not refreshed yet.
    # Re-read the governed object before reporting identity for release evidence.
    dataset = mlflow.genai.datasets.get_dataset(name=qualified)
    print(
        {
            "dataset": qualified,
            "dataset_id": dataset.dataset_id,
            "dataset_digest": dataset.digest,
            "records_merged": len(records),
            "experiment_id": experiment.experiment_id,
        }
    )


def _is_missing_dataset(error: Exception) -> bool:
    error_code = str(getattr(error, "error_code", "")).upper()
    message = str(error).upper()
    return error_code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"} or any(
        marker in message
        for marker in ("NOT_FOUND", "RESOURCE_DOES_NOT_EXIST", "DOES NOT EXIST")
    )


if __name__ == "__main__":
    main()
