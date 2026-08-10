"""Release gate: LLM-judge evaluation of a PINNED prompt version.

Evaluation never uses a mutable alias — it resolves an exact version (from
--prompt-version, or the version the development alias currently points at)
so results are reproducible and comparable. Runs on the credentialed path;
pull-request CI runs only evals/offline_checks.py.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety

from aai_core import bootstrap
from aai_core.decisions import Decision, DecisionRecord, record_decision
from aai_core.evaluation import (
    GatePolicy,
    MetricRule,
    apply_gate,
)
from aai_core.prompts import prompt_digest
from aai_core.providers.types import ProviderConfigurationError
from app.assistant import Assistant
from app.config import DATASET_NAME, PROMPT_NAME

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
    # Registry versions start at 1, so a typo must fail here rather than
    # during the credentialed load inside Assistant.
    if args.prompt_version is not None and args.prompt_version < 1:
        parser.error("--prompt-version must be a positive integer")

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
    judge_model = _judge_model_uri(context.settings)
    baseline = load_baseline()
    policy = GatePolicy(
        rules=tuple(load_thresholds()),
        allow_missing_regression_baseline=args.update_baseline and not baseline,
    )
    # A governed MLflow run: aai.* tags, pinned prompt URI + dataset as
    # params, gate metrics, verdict tag, and the evaluation traces attached.
    with context.experiments.run(
        run_name=f"prompt-v{version}-validation-gate",
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
            predict_fn=assistant.ask,
            scorers=[
                Correctness(model=judge_model),
                RelevanceToQuery(model=judge_model),
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
            "prompt": PROMPT_NAME,
            "prompt_version": version,
            "metrics": report.metrics,
            "baseline_updated": args.update_baseline,
            "decision": decision.value,
            "decision_run_id": decision_run_id,
        }
    )
    report.require_passed()


def _judge_model_uri(settings) -> str:
    config = settings.models.get("judge-model")
    if not isinstance(config, Mapping) or config.get("provider") != "databricks":
        raise ProviderConfigurationError(
            "judge-model must resolve to a governed Databricks serving endpoint"
        )
    deployment = config.get("deployment")
    if not isinstance(deployment, str) or not deployment.strip():
        raise ProviderConfigurationError("judge-model requires a deployment")
    return f"endpoints:/{deployment.strip()}"


if __name__ == "__main__":
    main()
