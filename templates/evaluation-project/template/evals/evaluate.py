"""Tier-2 gate: full evaluation with code scorers AND LLM judges.

Scores the target (recorded answer sheet by default, or a live endpoint with
--mode endpoint) against the golden suite, applies every threshold in
gate_config.json including baseline regression, and publishes the report to
the CI step summary. Runs on the credentialed path only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlflow.genai.scorers import scorer

from aai_core import bootstrap
from aai_core.evaluation import EvaluationSuite, QualityThreshold, publish_report
from app import judges, targets
from app.scorers import CODE_SCORERS

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evals" / "baseline.json"


def load_thresholds() -> list[QualityThreshold]:
    config = json.loads((ROOT / "evals" / "gate_config.json").read_text("utf-8"))
    return [QualityThreshold(**threshold) for threshold in config["thresholds"]]


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
    suite = EvaluationSuite(
        scorers=[*wrapped_code_scorers(), *judges.judge_scorers(context.settings)],
        thresholds=load_thresholds(),
    )
    baseline = load_baseline()
    report = suite.run(data=cases, predict_fn=predict_fn, baseline_metrics=baseline)
    publish_report(
        report,
        title=f"Evaluation gate — {context.tags.application} ({args.mode})",
        baseline=baseline,
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
