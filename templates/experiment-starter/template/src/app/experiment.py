"""One governed, reproducible experiment.

Everything a rerun needs is recorded: the dataset digest, the seed, the
source commit, the environment freeze, and deterministic metrics. Replace
`score_row` and the sample dataset with your real model and data — keep the
recording calls.
"""

from __future__ import annotations

import csv
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aai_core import PlatformContext
from aai_core.evaluation import GatePolicy, apply_gate
from aai_core.experiments import ExperimentManager, record_reproducibility
from app.config import DATASET_NAME

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "sample.csv"
DEFAULT_SEED = 7


class ExperimentRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    example_id: int = Field(ge=0)
    feature_signal: float = Field(allow_inf_nan=False)
    label: Literal[0, 1]


class EvaluationInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    example_id: int = Field(ge=0)
    feature_signal: float = Field(allow_inf_nan=False)


class EvaluationExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    label: Literal[0, 1]


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    inputs: EvaluationInputs
    expectations: EvaluationExpectations


def load_dataset(path: Path = DATA_PATH) -> Any:
    """Load the experiment dataset as a pandas DataFrame (lazy import so
    hermetic unit tests can inject plain records instead)."""

    import pandas

    return pandas.read_csv(path)


def load_csv_rows(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    """Parse CSV scalars explicitly before applying the strict row contract."""

    with path.open(encoding="utf-8", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    parsed = []
    for index, raw in enumerate(raw_rows, start=2):
        row: dict[str, Any] = dict(raw)
        try:
            if "example_id" in row:
                row["example_id"] = int(row["example_id"])
            if "feature_signal" in row:
                row["feature_signal"] = float(row["feature_signal"])
            if "label" in row:
                row["label"] = int(row["label"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid scalar in CSV row {index}") from error
        parsed.append(row)
    return dataset_rows(parsed)


def dataset_rows(data: Any) -> list[dict[str, Any]]:
    if hasattr(data, "to_dict"):
        data = data.to_dict("records")
    if not isinstance(data, Sequence) or isinstance(data, str | bytes):
        raise TypeError("experiment data must be a sequence of row mappings")
    rows = [ExperimentRow.model_validate(dict(row), strict=True) for row in data]
    identifiers = [row.example_id for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("experiment data contains duplicate example_id values")
    if not rows:
        raise ValueError("experiment data must contain at least one row")
    return [row.model_dump(mode="python") for row in rows]


def evaluation_records(data: Any) -> list[dict[str, Any]]:
    """Convert the tabular sample into native EvaluationDataset records."""

    return [
        {
            "inputs": {
                "example_id": int(row["example_id"]),
                "feature_signal": float(row["feature_signal"]),
            },
            "expectations": {"label": int(row["label"])},
        }
        for row in dataset_rows(data)
    ]


def rows_from_evaluation_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten governed EvaluationDataset records for the sample scorer."""

    validated = [
        EvaluationRecord.model_validate(dict(record), strict=True) for record in records
    ]
    return dataset_rows(
        [
            {
                "example_id": record.inputs.example_id,
                "feature_signal": record.inputs.feature_signal,
                "label": record.expectations.label,
            }
            for record in validated
        ]
    )


def score_row(row: Mapping[str, Any], *, seed: int) -> int:
    """Stand-in 'model': a deterministic threshold rule. The seed feeds the
    threshold so reruns with the same seed reproduce identical metrics."""

    threshold = 40 + (seed % 10)
    return 1 if float(row["feature_signal"]) >= threshold else 0


def evaluate_rows(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, float]:
    rows = dataset_rows(rows)
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
    evaluation_dataset: Any | None = None,
    evaluation_dataset_name: str | None = None,
    gate_policy: GatePolicy | None = None,
    baseline_metrics: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Run the tracked experiment and return its metrics."""

    data = data if data is not None else load_dataset()
    rows = dataset_rows(data)
    manager = ExperimentManager(
        experiment_name=context.settings.effective_experiment_name,
        context=context.tags,
        mlflow_module=mlflow_module,
    )
    mlflow = manager.native_client
    with manager.run(
        run_name=f"signal-threshold-seed-{seed}-validation",
        parameters={"seed": seed},
    ):
        record_reproducibility(seed=seed, mlflow_module=mlflow)
        dataset = evaluation_dataset
        if dataset is None:
            dataset = mlflow.data.from_pandas(
                data,
                name=DATASET_NAME,
                source=str(DATA_PATH),
            )
        mlflow.log_input(dataset, context="evaluation")
        dataset_parameters = {
            "dataset_name": evaluation_dataset_name
            or str(getattr(dataset, "name", DATASET_NAME)),
            "dataset_digest": str(dataset.digest),
        }
        dataset_id = getattr(dataset, "dataset_id", None)
        if dataset_id is not None:
            dataset_parameters["dataset_id"] = str(dataset_id)
        mlflow.log_params(dataset_parameters)
        metrics = evaluate_rows(rows, seed=seed)
        mlflow.log_metrics(metrics)
        if gate_policy is not None:
            gate = apply_gate(
                metrics,
                policy=gate_policy,
                baseline_metrics=baseline_metrics,
            )
            mlflow.log_params(
                {
                    "gate_policy_digest": gate.policy_digest,
                    "gate_baseline_digest": gate.baseline_digest or "none",
                }
            )
            mlflow.set_tags({"aai.gate_passed": str(gate.passed).lower()})
        with tempfile.TemporaryDirectory() as scratch:
            summary = Path(scratch) / "summary.json"
            summary.write_text(
                json.dumps({"seed": seed, "metrics": metrics}, indent=2, sort_keys=True)
            )
            mlflow.log_artifact(str(summary), artifact_path="reports")
    return metrics
