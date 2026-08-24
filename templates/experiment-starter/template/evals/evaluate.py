"""Release gate: run the experiment and apply deterministic thresholds.

No LLM judges here — this template's gate is plain metrics. It still runs on
the credentialed path (the experiment writes to the workspace tracking
server); pull-request CI runs only evals/offline_checks.py.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import mlflow

from aai_core import bootstrap
from aai_core.evaluation import GatePolicy, MetricRule, apply_gate
from app.config import DATASET_NAME
from app.experiment import (
    evaluation_records,
    load_dataset,
    rows_from_evaluation_records,
    run_experiment,
)

ROOT = Path(__file__).resolve().parents[1]
GATE_CONFIG = ROOT / "evals" / "gate_config.json"
BASELINE = ROOT / "evals" / "baseline.json"


def load_thresholds() -> list[MetricRule]:
    config = json.loads(GATE_CONFIG.read_text(encoding="utf-8"))
    return [MetricRule(**threshold) for threshold in config["thresholds"]]


def load_baseline() -> dict[str, float]:
    if not BASELINE.exists():
        print(
            "No evals/baseline.json yet; regression checks activate after the "
            "first release records one with --update-baseline."
        )
        return {}
    metrics = json.loads(BASELINE.read_text(encoding="utf-8"))["metrics"]
    return {name: float(value) for name, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    context = bootstrap(ROOT / "aai-platform.yml")
    baseline = load_baseline()
    policy = GatePolicy(
        rules=tuple(load_thresholds()),
        allow_missing_regression_baseline=args.update_baseline and not baseline,
    )
    dataset_name = (
        f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
    )
    reviewed_records = evaluation_records(load_dataset())
    dataset, registered_records = _load_release_dataset(
        dataset_name,
        reviewed_records,
    )
    metrics = run_experiment(
        context,
        data=rows_from_evaluation_records(registered_records),
        evaluation_dataset=dataset,
        evaluation_dataset_name=dataset_name,
        gate_policy=policy,
        baseline_metrics=baseline,
    )
    report = apply_gate(
        metrics,
        baseline_metrics=baseline,
        policy=policy,
    )
    report.require_passed()
    if args.update_baseline:
        BASELINE.write_text(
            json.dumps({"metrics": dict(report.metrics)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(
        {
            "application": context.tags.application,
            "metrics": report.metrics,
            "baseline_updated": args.update_baseline,
        }
    )


def _load_release_dataset(
    name: str,
    reviewed_records: list[dict],
):
    """Load the governed suite and fail closed on any unreviewed row drift."""

    dataset = mlflow.genai.datasets.get_dataset(name=name)
    registered_records = dataset.to_df().to_dict(orient="records")
    expected = Counter(_case_key(record) for record in reviewed_records)
    actual = Counter(_case_key(record) for record in registered_records)
    if actual != expected:
        missing = sum((expected - actual).values())
        extra = sum((actual - expected).values())
        raise RuntimeError(
            f"Unity Catalog dataset {name!r} differs from the reviewed release "
            f"suite (missing={missing}, extra={extra}). Run "
            "scripts/sync_dataset.py; if stale records remain, use a new "
            "versioned DATASET_NAME rather than evaluating unreviewed rows."
        )
    return dataset, registered_records


def _case_key(record: Mapping) -> str:
    return json.dumps(
        {
            "inputs": record.get("inputs"),
            "expectations": record.get("expectations"),
            "tags": _review_tags(record),
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _review_tags(record: Mapping) -> dict:
    tags = record.get("tags") or {}
    if not isinstance(tags, Mapping):
        raise TypeError("Evaluation case tags must be an object")
    return {
        str(key): value
        for key, value in tags.items()
        if not str(key).casefold().startswith("mlflow.")
    }


if __name__ == "__main__":
    main()
