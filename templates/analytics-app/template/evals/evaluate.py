"""Release gate: the real agent against the live warehouse, judged and gated.

Runs every golden case through the runbook agent (semantic-first tools over
the deployed warehouse), scores answers with the deterministic provenance
scorers plus Correctness/Safety judges, and applies every threshold, the
regression baseline, and the cost-coverage policy. Telemetry follows the
published pattern for eval storage: the governed MLflow run records the
semantic-model version, a knowledge-docs digest, model and judge identity,
git provenance, per-answer token usage, and the verdict — enough to chart
accuracy and cost over time. Credentialed path only; pull-request CI runs
evals/offline_checks.py.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from contextvars import copy_context
from pathlib import Path

import mlflow
from mlflow.genai.scorers import Correctness, Safety, scorer

from aai_core import bootstrap
from aai_core.evaluation import GatePolicy, MetricRule, apply_gate, judge_model_uri
from aai_core.experiments import record_reproducibility
from aai_core.tracing import TraceIntegration, traced
from app.agent import AnalyticsAgent
from app.config import DATASET_NAME, DEMO_CATALOG, DEMO_SCHEMA, resolve_warehouse_id
from app.scorers import CODE_SCORERS
from app.semantics.executor import DatabricksWarehouseExecutor

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evals" / "baseline.json"
MINIMUM_COST_COVERAGE = 0.9


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


def knowledge_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "knowledge").glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse-id", default=None)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    context = bootstrap(ROOT / "aai-platform.yml")
    judge_model = judge_model_uri(context.settings)
    context.configure_tracing(integration=TraceIntegration.SDK)
    warehouse_id = resolve_warehouse_id(args.warehouse_id)
    cases = json.loads(
        (ROOT / "evals" / "data" / "golden_cases.json").read_text(encoding="utf-8")
    )
    dataset_name = (
        f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
    )
    dataset = _load_release_dataset(dataset_name, cases)

    thresholds = load_thresholds()
    baseline = load_baseline()
    target_identity, judge_identity = _evaluation_model_identities(context.settings)
    policy = GatePolicy(
        rules=tuple(thresholds),
        # Tokenomics is part of the gate: answers must carry token accounting
        # (cost/coverage) so cost trends stay chartable run over run.
        minimum_cost_coverage=MINIMUM_COST_COVERAGE,
        allow_missing_regression_baseline=args.update_baseline and not baseline,
    )

    executor = DatabricksWarehouseExecutor(
        warehouse_id=warehouse_id, catalog=DEMO_CATALOG, schema=DEMO_SCHEMA
    )
    # Fail the cheap capability check before any model spend: a missing
    # warehouse grant should not cost a single judge token.
    executor.execute("SELECT 1", row_limit=1)

    manager = context.experiments
    usages: list[dict[str, int | bool]] = []
    with asyncio.Runner() as runner:
        agent = AnalyticsAgent.from_project(ROOT, context, executor=executor)
        predict_fn = _build_predict_fn(agent, runner, usages)

        try:
            with manager.run(
                run_name="analytics-runbook-validation-gate",
                parameters={
                    "warehouse_id": warehouse_id,
                    "evaluation_dataset": dataset_name,
                    "evaluation_dataset_id": dataset.dataset_id,
                    "evaluation_dataset_digest": dataset.digest,
                    "case_count": len(cases),
                    "adversarial_review": agent.enable_review,
                    "semantic_model_version": _semantic_model_version(),
                    "knowledge_digest": knowledge_digest(),
                    "target_model": target_identity,
                    "judge_model": judge_identity,
                },
            ) as evaluation_run:
                _validate_dataset_association(
                    dataset, evaluation_run.info.experiment_id
                )
                record_reproducibility()
                native_result = mlflow.genai.evaluate(
                    data=dataset,
                    predict_fn=predict_fn,
                    scorers=[
                        *wrapped_code_scorers(),
                        Correctness(model=judge_model),
                        Safety(model=judge_model),
                    ],
                )
                metrics = _finite_metrics(native_result)
                metrics.update(_token_metrics(usages))
                report = apply_gate(
                    metrics,
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
        finally:
            try:
                runner.run(agent.aclose())
            finally:
                runner.run(executor.aclose())

    report.require_passed()
    if args.update_baseline:
        BASELINE.write_text(
            json.dumps({"metrics": dict(report.metrics)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(
        {
            "warehouse_id": warehouse_id,
            "metrics": report.metrics,
            "baseline_updated": args.update_baseline,
        }
    )


def _build_predict_fn(
    agent: AnalyticsAgent,
    runner: asyncio.Runner,
    usages: list[dict[str, int | bool]],
):
    """Keep the complete multi-step agent trajectory in one MLflow trace."""

    @traced(name="agent.evaluate", span_type="AGENT")
    def predict_fn(question: str) -> str:
        answer = runner.run(agent.aanswer(question), context=copy_context())
        usages.append(
            {
                "captured": answer.usage.captured,
                "total": answer.usage.total_tokens,
                "review": answer.usage.review_tokens,
            }
        )
        return answer.answer

    return predict_fn


def _finite_metrics(native_result) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in dict(getattr(native_result, "metrics", {}) or {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        metrics[name] = float(value)
    return metrics


def _token_metrics(usages: list[dict[str, int | bool]]) -> dict[str, float]:
    if not usages:
        return {"cost/coverage": 0.0}
    captured = [usage for usage in usages if usage["captured"]]
    coverage = len(captured) / len(usages)
    metrics = {"cost/coverage": coverage}
    if captured:
        metrics["tokens/mean_total"] = sum(
            int(usage["total"]) for usage in captured
        ) / len(captured)
        metrics["tokens/mean_review"] = sum(
            int(usage["review"]) for usage in captured
        ) / len(captured)
    return metrics


def _semantic_model_version() -> str:
    import yaml

    payload = yaml.safe_load(
        (ROOT / "semantics" / "semantic_model.yml").read_text(encoding="utf-8")
    )
    info = payload.get("semantic_model", {})
    return f"{info.get('name', 'unknown')}-v{info.get('version', 0)}"


def _evaluation_model_identities(settings) -> tuple[str, str]:
    target = _model_config(settings, "general-chat")
    judge = _model_config(settings, "judge-model")
    if (
        judge["provider"].casefold() == target["provider"].casefold()
        and judge["deployment"].casefold() == target["deployment"].casefold()
    ):
        raise ValueError(
            "judge-model must use a deployment distinct from general-chat; "
            "a release gate cannot rely on the target judging itself"
        )
    target_identity = f"{target['provider']}:{target['deployment']}"
    judge_identity = f"{judge['provider']}:{judge['deployment']}"
    return target_identity, judge_identity


def _model_config(settings, logical_name: str) -> dict[str, str]:
    config = settings.models.get(logical_name)
    if not isinstance(config, Mapping):
        raise ValueError(f"{logical_name} must be configured")
    provider = config.get("provider")
    deployment = config.get("deployment")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(f"{logical_name} requires a provider")
    if not isinstance(deployment, str) or not deployment.strip():
        raise ValueError(f"{logical_name} requires a deployment")
    return {
        "provider": provider.strip(),
        "deployment": deployment.strip(),
    }


def _load_release_dataset(name: str, reviewed_cases: list[dict]):
    """Load the governed suite and fail closed on any unreviewed row drift."""

    dataset = mlflow.genai.datasets.get_dataset(name=name)
    if not isinstance(dataset.dataset_id, str) or not dataset.dataset_id.strip():
        raise RuntimeError(f"Unity Catalog dataset {name!r} has no stable dataset ID")
    if not isinstance(dataset.digest, str) or not dataset.digest.strip():
        raise RuntimeError(f"Unity Catalog dataset {name!r} has no stable digest")
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


def _validate_dataset_association(dataset, experiment_id: str) -> None:
    associated = {str(value) for value in (dataset.experiment_ids or [])}
    if str(experiment_id) not in associated:
        raise RuntimeError(
            "The release dataset is not associated with this application's "
            "configured experiment"
        )


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
