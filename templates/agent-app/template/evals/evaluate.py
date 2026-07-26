"""Release gate: LLM judges plus deterministic trajectory scoring.

Runs the real agent per case (pinned prompt version), records which tools it
used, scores answers with Correctness/Safety judges, computes
tool_call_accuracy from the recorded trajectories, and applies every
threshold plus baseline regression. Publishes the report to the CI summary.
Credentialed path only; pull-request CI runs evals/offline_checks.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlflow.genai.scorers import Correctness, Safety

from aai_core import bootstrap
from aai_core.agents import AgentRequest
from aai_core.evaluation import (
    EvaluationSuite,
    QualityThreshold,
    apply_thresholds,
    publish_report,
)
from app.agent import ToolAgent
from app.config import PROMPT_NAME
from app.scoring import tool_call_accuracy

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


def resolve_version(context, requested: int | None) -> int:
    if requested is not None:
        return requested
    development = context.prompts.load(PROMPT_NAME, alias="development")
    return int(development.version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-version", type=int, default=None)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    context = bootstrap(ROOT / "aai-platform.yml")
    version = resolve_version(context, args.prompt_version)
    agent = ToolAgent(context, prompt_version=version)
    cases = json.loads(
        (ROOT / "evals" / "data" / "release_cases.json").read_text(encoding="utf-8")
    )

    trajectories: dict[str, list[str]] = {}

    def predict_fn(question: str) -> str:
        response = agent.invoke(
            AgentRequest(messages=[{"role": "user", "content": question}])
        )
        trajectories[question] = list(response.metadata.get("tools_used", []))
        return response.content

    thresholds = load_thresholds()
    baseline = load_baseline()
    suite = EvaluationSuite(scorers=[Correctness(), Safety()], thresholds=[])
    judged = suite.run(data=cases, predict_fn=predict_fn)

    accuracy_values = [
        tool_call_accuracy(
            trajectories.get(case["inputs"]["question"], []),
            case["expectations"]["expected_tools"],
        )
        for case in cases
    ]
    metrics = {
        **judged.metrics,
        "tool_call_accuracy/mean": sum(accuracy_values) / len(accuracy_values),
    }
    report = apply_thresholds(metrics, thresholds, baseline_metrics=baseline)
    publish_report(
        report,
        title=f"Agent gate — {context.tags.application} (prompt v{version})",
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
            "prompt_version": version,
            "metrics": report.metrics,
            "baseline_updated": args.update_baseline,
        }
    )


if __name__ == "__main__":
    main()
