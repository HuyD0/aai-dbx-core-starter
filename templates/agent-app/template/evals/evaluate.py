"""Release gate: LLM judges plus exact MLflow tool-call scoring.

Runs the real agent per case (pinned prompt version), scores its traced TOOL
spans against expected names and arguments, applies Correctness/Safety judges,
and enforces every threshold plus baseline regression. Publishes the report
to the CI summary. Credentialed path only; pull-request CI runs
evals/offline_checks.py.
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
    judge_model_uri,
    publish_report,
    workspace_run_url,
)
from app.agent import ToolAgent
from app.config import DATASET_NAME, PROMPT_NAME
from app.tool_scoring import exact_tool_call_scorer

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

    def predict_fn(question: str) -> str:
        response = agent.invoke(
            AgentRequest(messages=[{"role": "user", "content": question}])
        )
        return response.content

    thresholds = load_thresholds()
    baseline = load_baseline()
    judge_model = judge_model_uri(context.settings)
    suite = EvaluationSuite(
        scorers=[
            exact_tool_call_scorer(),
            Correctness(model=judge_model),
            Safety(model=judge_model),
        ],
        thresholds=thresholds,
    )
    prompt_uri = f"prompts:/{context.prompts.qualify(PROMPT_NAME)}/{version}"
    dataset_name = (
        f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
    )

    # One governed MLflow run holds the judged evaluation, exact tool-call
    # score, pinned prompt/dataset identity, and verdict.
    manager = context.experiments
    run_id = None
    with manager.run(
        run_name=f"agent-gate-v{version}",
        parameters={
            "prompt_version": version,
            "prompt_uri": prompt_uri,
            "evaluation_dataset": dataset_name,
            "case_count": len(cases),
        },
    ) as active_run:
        run_id = active_run.info.run_id
        report = suite.run(
            data=cases,
            predict_fn=predict_fn,
            baseline_metrics=baseline,
        )
        manager.log_metrics(
            {
                name: value
                for name, value in report.metrics.items()
                if isinstance(value, (int, float))
            }
        )
        import mlflow

        mlflow.set_tags({"aai.gate_passed": str(report.passed).lower()})

    publish_report(
        report,
        title=f"Agent gate — {context.tags.application} (prompt v{version})",
        baseline=baseline,
        run_link=workspace_run_url(run_id),
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
