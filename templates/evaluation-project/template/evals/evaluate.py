"""Tier-2 gate: full evaluation with code scorers AND LLM judges.

Scores the target (recorded answer sheet by default, or a live endpoint with
--mode endpoint) against the golden suite, applies every threshold in
gate_config.json including baseline regression, and publishes the report to
the CI step summary. It also prints bounded per-row scorer failures without
raw input/output columns. Rationale/error details require
--show-triage-details and remain subject to normal log data-handling rules.
Runs on the credentialed path only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.genai.scorers import scorer

from aai_core import bootstrap
from aai_core.evaluation import (
    GatePolicy,
    MetricRule,
    apply_gate,
)
from app import judges, targets
from app.config import DATASET_NAME
from app.scorers import CODE_SCORERS
from app.triage import print_failure_triage

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evals" / "baseline.json"


def load_thresholds() -> list[MetricRule]:
    config = json.loads((ROOT / "evals" / "gate_config.json").read_text("utf-8"))
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


def wrapped_code_scorers() -> list:
    wrapped = []
    for fn in CODE_SCORERS:

        def make(inner):
            @scorer(name=inner.__name__)
            def code_scorer(outputs, expectations):
                return inner(str(outputs), dict(expectations or {}))

            return code_scorer

        wrapped.append(make(fn))
    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["answer-sheet", "endpoint"],
        default="answer-sheet",
        help="answer-sheet replays recorded answers; endpoint calls the "
        "target-model logical endpoint from aai-platform.yml.",
    )
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument(
        "--show-triage-details",
        action="store_true",
        help="Print bounded judge rationale/error text. Enable only when CI "
        "logs are approved for the evaluation data classification.",
    )
    args = parser.parse_args()

    context = bootstrap(ROOT / "aai-platform.yml")
    if args.mode == "endpoint":
        predict_fn = targets.endpoint_predict_fn(context)
    else:
        predict_fn = targets.answer_sheet_predict_fn(
            ROOT / "evals" / "data" / "answer_sheet.json"
        )

    cases = json.loads(
        (ROOT / "evals" / "data" / "golden_cases.json").read_text(encoding="utf-8")
    )
    baseline = load_baseline()
    policy = GatePolicy(
        rules=tuple(load_thresholds()),
        allow_missing_regression_baseline=args.update_baseline and not baseline,
    )
    dataset_name = (
        f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
    )
    # A governed MLflow run: aai.* tags, the registered dataset identity,
    # gate metrics, verdict tag, and evaluation traces attached.
    with context.experiments.run(
        run_name=f"evaluation-{args.mode}-validation-gate",
        parameters={
            "mode": args.mode,
            "evaluation_dataset": dataset_name,
            "case_count": len(cases),
        },
    ):
        native_result = mlflow.genai.evaluate(
            data=cases,
            predict_fn=predict_fn,
            scorers=[*wrapped_code_scorers(), *judges.judge_scorers(context.settings)],
        )
        report = apply_gate(
            native_result,
            policy=policy,
            baseline_metrics=baseline,
        )
        mlflow.log_metrics(dict(report.metrics))
        mlflow.set_tag("aai.gate_passed", str(report.passed).lower())
    print_failure_triage(
        native_result,
        include_details=args.show_triage_details,
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
            "mode": args.mode,
            "metrics": report.metrics,
            "baseline_updated": args.update_baseline,
        }
    )


if __name__ == "__main__":
    main()
