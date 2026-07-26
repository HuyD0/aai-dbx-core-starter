"""Release gate: LLM-judge evaluation of a PINNED prompt version.

Evaluation never uses a mutable alias — it resolves an exact version (from
--prompt-version, or the version the development alias currently points at)
so results are reproducible and comparable. Runs on the credentialed path;
pull-request CI runs only evals/offline_checks.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety

from aai_core import bootstrap
from aai_core.evaluation import (
    EvaluationSuite,
    QualityThreshold,
    publish_report,
    workspace_run_url,
)
from app.assistant import Assistant
from app.config import DATASET_NAME, PROMPT_NAME

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
    if requested is not None:
        return requested
    development = context.prompts.load(PROMPT_NAME, alias="development")
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
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    context = bootstrap(ROOT / "aai-platform.yml")
    version = resolve_version(context, args.prompt_version)
    assistant = Assistant(context, prompt_version=version)
    prompt_uri = f"prompts:/{context.prompts.qualify(PROMPT_NAME)}/{version}"
    dataset_name = (
        f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
    )
    cases = json.loads(
        (ROOT / "evals" / "data" / "release_cases.json").read_text(encoding="utf-8")
    )
    suite = EvaluationSuite(
        scorers=[Correctness(), RelevanceToQuery(), Safety()],
        thresholds=load_thresholds(),
    )
    baseline = load_baseline()
    # A governed MLflow run: aai.* tags, pinned prompt URI + dataset as
    # params, gate metrics, verdict tag, and the evaluation traces attached.
    report, run_id = suite.run_tracked(
        experiments=context.experiments,
        run_name=f"prompt-gate-v{version}",
        data=cases,
        predict_fn=assistant.ask,
        baseline_metrics=baseline,
        prompt_uri=prompt_uri,
        dataset_name=dataset_name,
        parameters={"prompt_version": version},
    )
    publish_report(
        report,
        title=f"Prompt gate — {PROMPT_NAME} v{version}",
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
            "prompt": PROMPT_NAME,
            "prompt_version": version,
            "metrics": report.metrics,
            "baseline_updated": args.update_baseline,
        }
    )


if __name__ == "__main__":
    main()
