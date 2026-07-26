"""One governed, reproducible experiment.

Everything a rerun needs is recorded: the dataset digest, the seed, the
source commit, the environment freeze, and deterministic metrics. Replace
`score_row` and the sample dataset with your real model and data — keep the
recording calls.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aai_core import PlatformContext
from aai_core.experiments import ExperimentManager, record_reproducibility

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "sample.csv"
DEFAULT_SEED = 7


def load_dataset(path: Path = DATA_PATH) -> Any:
    """Load the experiment dataset as a pandas DataFrame (lazy import so
    hermetic unit tests can inject plain records instead)."""

    import pandas

    return pandas.read_csv(path)


def dataset_rows(data: Any) -> list[dict[str, Any]]:
    if hasattr(data, "to_dict"):
        return list(data.to_dict("records"))
    return [dict(row) for row in data]


def score_row(row: Mapping[str, Any], *, seed: int) -> int:
    """Stand-in 'model': a deterministic threshold rule. The seed feeds the
    threshold so reruns with the same seed reproduce identical metrics."""

    threshold = 40 + (seed % 10)
    return 1 if float(row["feature_signal"]) >= threshold else 0


def evaluate_rows(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, float]:
    predictions = [score_row(row, seed=seed) for row in rows]
    labels = [int(row["label"]) for row in rows]
    pairs = list(zip(predictions, labels, strict=True))
    correct = sum(1 for predicted, actual in pairs if predicted == actual)
    positives = sum(predictions) or 1
    true_positives = sum(
        1 for predicted, actual in pairs if predicted == 1 and actual == 1
    )
    return {
        "accuracy": correct / len(rows),
        "precision": true_positives / positives,
        "example_count": float(len(rows)),
    }


def run_experiment(
    context: PlatformContext,
    *,
    seed: int = DEFAULT_SEED,
    data: Any | None = None,
    mlflow_module: Any | None = None,
) -> dict[str, float]:
    """Run the tracked experiment and return its metrics."""

    data = data if data is not None else load_dataset()
    rows = dataset_rows(data)
    manager = ExperimentManager(
        experiment_name=context.settings.effective_experiment_name,
        context=context.tags,
        mlflow_module=mlflow_module,
    )
    with manager.run(run_name=f"experiment-seed-{seed}", parameters={"seed": seed}):
        record_reproducibility(seed=seed, mlflow_module=mlflow_module)
        manager.log_dataset(data, name="sample", source=str(DATA_PATH))
        metrics = evaluate_rows(rows, seed=seed)
        manager.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as scratch:
            summary = Path(scratch) / "summary.json"
            summary.write_text(
                json.dumps({"seed": seed, "metrics": metrics}, indent=2, sort_keys=True)
            )
            manager.log_artifact(summary, artifact_path="reports")
    return metrics
