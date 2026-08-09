"""Release gate: LLM judges plus exact MLflow tool-call scoring.

Runs the real agent per case (pinned prompt version), scores its traced TOOL
spans against expected names and arguments, applies Correctness/Safety judges,
and enforces every threshold plus baseline regression. Publishes the report
to the CI summary. Credentialed path only; pull-request CI runs
evals/offline_checks.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import mlflow
from mlflow.genai.scorers import Correctness, Safety

from aai_core import bootstrap
from aai_core.agents import AgentRequest
from aai_core.decisions import Decision, DecisionRecord, record_decision
from aai_core.evaluation import (
    GatePolicy,
    MetricRule,
    apply_gate,
    judge_model_uri,
)
from aai_core.prompts import prompt_digest
from aai_core.tracing import TraceIntegration
from app.agent import ToolAgent
from app.config import DATASET_NAME, PROMPT_NAME
from app.tool_scoring import exact_tool_call_scorer

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-version",
        type=int,
        required=True,
        help="Exact immutable Prompt Registry version to evaluate.",
    )
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    if args.prompt_version < 1:
        parser.error("--prompt-version must be a positive integer")

    context = bootstrap(ROOT / "aai-platform.yml")
    judge_model = judge_model_uri(context.settings)
    context.configure_tracing(integration=TraceIntegration.SDK)
    version = args.prompt_version
    cases = json.loads(
        (ROOT / "evals" / "data" / "release_cases.json").read_text(encoding="utf-8")
    )

    thresholds = load_thresholds()
    baseline = load_baseline()
    policy = GatePolicy(
        rules=tuple(thresholds),
        allow_missing_regression_baseline=args.update_baseline and not baseline,
    )
    prompt_uri = f"prompts:/{context.prompts.qualify(PROMPT_NAME)}/{version}"
    dataset_name = (
        f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
    )

    # One governed MLflow run holds the judged evaluation, exact tool-call
    # score, pinned prompt/dataset identity, and verdict.
    manager = context.experiments
    with asyncio.Runner() as runner:
        agent = ToolAgent(context, prompt_version=version)

        def predict_fn(question: str) -> str:
            response = runner.run(
                agent.ainvoke(
                    AgentRequest(messages=[{"role": "user", "content": question}])
                )
            )
            return response.content

        try:
            with manager.run(
                run_name=f"agent-prompt-v{version}-validation-gate",
                parameters={
                    "prompt_version": version,
                    "prompt_uri": prompt_uri,
                    "evaluation_dataset": dataset_name,
                    "case_count": len(cases),
                },
            ) as evaluation_run:
                registered = context.prompts.load(
                    PROMPT_NAME, version=version, cache_ttl_seconds=0
                )
                native_result = mlflow.genai.evaluate(
                    data=cases,
                    predict_fn=predict_fn,
                    scorers=[
                        exact_tool_call_scorer(),
                        Correctness(model=judge_model),
                        Safety(model=judge_model),
                    ],
                )
                report = apply_gate(
                    native_result,
                    policy=policy,
                    baseline_metrics=baseline,
                )
                mlflow.log_metrics(dict(report.metrics))
                mlflow.set_tag("aai.gate_passed", str(report.passed).lower())
                evaluation_run_id = str(evaluation_run.info.run_id)
        finally:
            runner.run(agent.aclose())

    template = getattr(registered, "template", None)
    if not isinstance(template, (str, list)):
        raise TypeError(
            "The evaluated prompt version exposes no template for decision evidence."
        )
    decision = Decision.ADOPT if report.passed else Decision.REJECT
    decision_run_id = record_decision(
        DecisionRecord(
            decision=decision,
            change_id=f"prompt-v{version}",
            change_summary=f"Evaluate pinned prompt version {version} for release.",
            rationale=(
                "The release gate passed for the exact registered prompt version."
                if report.passed
                else "The release gate failed for the exact registered prompt version."
            ),
            change_run_id=evaluation_run_id,
            gate=report,
            prompt_name=context.prompts.qualify(PROMPT_NAME),
            prompt_version=version,
            prompt_digest=prompt_digest(template),
            decided_by="code:release-gate",
        ),
        experiments=context.experiments,
    )
    if report.passed and args.update_baseline:
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
            "decision": decision.value,
            "decision_run_id": decision_run_id,
        }
    )
    report.require_passed()


if __name__ == "__main__":
    main()
