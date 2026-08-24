"""Deterministic, credential-free release-gate checks for pull-request CI.

Validates gate configuration, the checked-in dataset, and metric determinism
without any workspace access. The full gate (evals/evaluate.py) additionally
records the run in MLflow on the credentialed path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aai_core.evaluation import MetricRule

ROOT = Path(__file__).resolve().parents[1]
MIN_ROWS = 10
REQUIRED_GATED_METRICS = ("accuracy",)


def main() -> int:  # noqa: C901 - linear, independent contract assertions
    sys.path.insert(0, str(ROOT / "src"))
    from app.experiment import DEFAULT_SEED, evaluate_rows, load_csv_rows

    failures: list[str] = []

    config = json.loads((ROOT / "evals" / "gate_config.json").read_text("utf-8"))
    thresholds = [MetricRule(**threshold) for threshold in config["thresholds"]]
    gated = {threshold.metric for threshold in thresholds}
    for metric in REQUIRED_GATED_METRICS:
        if metric not in gated:
            failures.append(f"gate_config.json does not gate {metric}")

    rows = load_csv_rows(ROOT / "data" / "sample.csv")
    if len(rows) < MIN_ROWS:
        failures.append(f"data/sample.csv has {len(rows)} rows; keep >= {MIN_ROWS}")
    for column in ("feature_signal", "label"):
        if rows and column not in rows[0]:
            failures.append(f"data/sample.csv is missing the {column} column")

    if not failures:
        first = evaluate_rows(rows, seed=DEFAULT_SEED)
        second = evaluate_rows(rows, seed=DEFAULT_SEED)
        if first != second:
            failures.append("evaluate_rows is not deterministic for a fixed seed")
        for metric in gated:
            if metric not in first and metric != "example_count":
                failures.append(f"gated metric {metric} is not produced by the run")

    baseline = ROOT / "evals" / "baseline.json"
    if baseline.exists():
        metrics = json.loads(baseline.read_text("utf-8")).get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            failures.append("baseline.json exists but has no metrics mapping")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"offline release-gate checks passed ({len(rows)} dataset rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
