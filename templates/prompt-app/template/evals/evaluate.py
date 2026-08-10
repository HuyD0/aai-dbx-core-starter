"""Release gate: LLM-judge evaluation of a PINNED prompt version.

Evaluation never uses a mutable alias — it resolves an exact version (from
--prompt-version, or the version the development alias currently points at)
so results are reproducible and comparable. Runs on the credentialed path;
pull-request CI runs only evals/offline_checks.py.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety

from aai_core import bootstrap
from aai_core.evaluation import (
    GatePolicy,
    MetricRule,
    apply_gate,
)
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
    judge_model, target_identity, judge_identity = _evaluation_models(context.settings)
    baseline = load_baseline()
    policy = GatePolicy(
        rules=tuple(load_thresholds()),
        allow_missing_regression_baseline=args.update_baseline and not baseline,
    )
    dataset = _load_release_dataset(dataset_name, cases)
    # A governed MLflow run: aai.* tags, pinned prompt URI + dataset as
    # params, gate metrics, verdict tag, and the evaluation traces attached.
    with context.experiments.run(
        run_name=f"prompt-v{version}-validation-gate",
        parameters={
            "prompt_version": version,
            "prompt_uri": prompt_uri,
            "evaluation_dataset": dataset_name,
            "evaluation_dataset_id": dataset.dataset_id,
            "evaluation_dataset_digest": dataset.digest,
            "case_count": len(cases),
            "target_model": target_identity,
            "judge_model": judge_identity,
        },
    ):
        native_result = mlflow.genai.evaluate(
            data=dataset,
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
        mlflow.log_params(
            {
                "gate_policy_digest": report.policy_digest,
                "gate_baseline_digest": report.baseline_digest or "none",
            }
        )
        mlflow.set_tags(
            {
                "aai.gate_passed": str(report.passed).lower(),
                "aai.target_model": target_identity,
                "aai.judge_model": judge_identity,
            }
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


def _evaluation_models(settings) -> tuple[str, str, str]:
    target = _model_config(settings, "general-chat")
    judge = _model_config(settings, "judge-model")
    if judge["provider"] != "databricks":
        raise ProviderConfigurationError(
            "judge-model must resolve to a governed Databricks serving endpoint"
        )
    if (
        judge["provider"].casefold() == target["provider"].casefold()
        and judge["deployment"].casefold() == target["deployment"].casefold()
    ):
        raise ProviderConfigurationError(
            "judge-model must use a deployment distinct from general-chat; "
            "a release gate cannot rely on the target judging itself"
        )
    target_identity = f"{target['provider']}:{target['deployment']}"
    judge_identity = f"{judge['provider']}:{judge['deployment']}"
    return f"endpoints:/{judge['deployment']}", target_identity, judge_identity


def _model_config(settings, logical_name: str) -> dict[str, str]:
    config = settings.models.get(logical_name)
    if not isinstance(config, Mapping):
        raise ProviderConfigurationError(f"{logical_name} must be configured")
    provider = config.get("provider")
    deployment = config.get("deployment")
    if not isinstance(provider, str) or not provider.strip():
        raise ProviderConfigurationError(f"{logical_name} requires a provider")
    if not isinstance(deployment, str) or not deployment.strip():
        raise ProviderConfigurationError(f"{logical_name} requires a deployment")
    return {
        "provider": provider.strip(),
        "deployment": deployment.strip(),
    }


def _load_release_dataset(name: str, reviewed_cases: list[dict]):
    dataset = mlflow.genai.datasets.get_dataset(name=name)
    registered_cases = dataset.to_df().to_dict(orient="records")
    expected = Counter(_case_key(record) for record in reviewed_cases)
    actual = Counter(_case_key(record) for record in registered_cases)
    if actual != expected:
        missing = sum((expected - actual).values())
        extra = sum((actual - expected).values())
        raise RuntimeError(
            f"Unity Catalog dataset {name!r} differs from the reviewed release "
            f"suite (missing={missing}, extra={extra}). Run "
            "scripts/sync_dataset.py; if stale records remain, use a new "
            "versioned DATASET_NAME rather than evaluating unreviewed rows."
        )
    return dataset


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
