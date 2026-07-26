"""Release gate: LLM-judge evaluation against configured quality thresholds.

This run needs model access (the judges call an LLM), so it executes on the
credentialed path — locally with keyless auth, or in the workspace job — and
must pass BEFORE a release is promoted. Pull-request CI runs only the
deterministic checks in evals/offline_checks.py.

Thresholds live in evals/gate_config.json. Regression checks activate once
evals/baseline.json exists; refresh it from a passing release run with
``python evals/evaluate.py --update-baseline``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, Safety

from aai_core import bootstrap
from aai_core.agents import AgentRequest
from aai_core.evaluation import (
    EvaluationSuite,
    QualityThreshold,
    judge_model_uri,
    publish_report,
    workspace_run_url,
)
from app.config import DATASET_NAME
from app.rag import RAGAgent

ROOT = Path(__file__).resolve().parents[1]
GATE_CONFIG = ROOT / "evals" / "gate_config.json"
BASELINE = ROOT / "evals" / "baseline.json"


def load_thresholds() -> list[QualityThreshold]:
    config = json.loads(GATE_CONFIG.read_text(encoding="utf-8"))
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
    """Pin the exact prompt version under evaluation (never a mutable alias),
    so the version that passed this gate is the one promote_prompt.py and
    create_release.py record."""

    if requested is not None:
        return requested
    development = context.prompts.load("agent-system", alias="development")
    return int(development.version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-version",
        type=int,
        default=None,
        help="Exact version to evaluate; defaults to the version the "
        "development alias currently points at (resolved once, then pinned).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="After a passing run, write the metrics to evals/baseline.json "
        "so future runs regression-check against this release.",
    )
    args = parser.parse_args()

    context = bootstrap(ROOT / "aai-platform.yml")
    version = resolve_version(context, args.prompt_version)
    agent = RAGAgent(context, prompt_version=version)

    def predict_fn(question: str) -> str:
        response = agent.invoke(
            AgentRequest(messages=[{"role": "user", "content": question}])
        )
        return response.content

    cases = json.loads(
        (ROOT / "evals" / "data" / "release_cases.json").read_text(encoding="utf-8")
    )
    judge_model = judge_model_uri(context.settings)
    suite = EvaluationSuite(
        scorers=[
            RetrievalGroundedness(model=judge_model),
            RelevanceToQuery(model=judge_model),
            Safety(model=judge_model),
        ],
        thresholds=load_thresholds(),
    )
    baseline = load_baseline()
    prompt_uri = f"prompts:/{context.prompts.qualify('agent-system')}/{version}"
    dataset_name = (
        f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
    )
    # A governed MLflow run: aai.* tags, pinned prompt URI + dataset as
    # params, gate metrics, verdict tag, and the evaluation traces attached.
    report, run_id = suite.run_tracked(
        experiments=context.experiments,
        run_name=f"rag-gate-v{version}",
        data=cases,
        predict_fn=predict_fn,
        baseline_metrics=baseline,
        prompt_uri=prompt_uri,
        dataset_name=dataset_name,
        parameters={"prompt_version": version},
    )
    publish_report(
        report,
        title=f"RAG gate — {context.tags.application} (prompt v{version})",
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
            "application": context.tags.application,
            "release": context.tags.release,
            "prompt_version": version,
            "metrics": report.metrics,
            "baseline_updated": args.update_baseline,
        }
    )


if __name__ == "__main__":
    main()
