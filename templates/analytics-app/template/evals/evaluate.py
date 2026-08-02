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
from collections.abc import Mapping
from pathlib import Path

import mlflow
from mlflow.genai.scorers import Correctness, Safety, scorer

from aai_core import bootstrap
from aai_core.evaluation import GatePolicy, MetricRule, apply_gate
from aai_core.experiments import record_reproducibility
from aai_core.providers.types import ProviderConfigurationError
from aai_core.tracing import TraceIntegration
from app.agent import AnalyticsAgent
from app.config import DEMO_CATALOG, DEMO_SCHEMA, resolve_warehouse_id
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
    context.configure_tracing(integration=TraceIntegration.SDK)
    warehouse_id = resolve_warehouse_id(args.warehouse_id)
    cases = json.loads(
        (ROOT / "evals" / "data" / "golden_cases.json").read_text(encoding="utf-8")
    )

    thresholds = load_thresholds()
    baseline = load_baseline()
    judge_model = _judge_model_uri(context.settings)
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

        def predict_fn(question: str) -> str:
            answer = runner.run(agent.aanswer(question))
            usages.append(
                {
                    "captured": answer.usage.captured,
                    "total": answer.usage.total_tokens,
                    "review": answer.usage.review_tokens,
                }
            )
            return answer.answer

        try:
            with manager.run(
                run_name="analytics-runbook-validation-gate",
                parameters={
                    "warehouse_id": warehouse_id,
                    "case_count": len(cases),
                    "adversarial_review": agent.enable_review,
                    "semantic_model_version": _semantic_model_version(),
                    "knowledge_digest": knowledge_digest(),
                    "judge_model": judge_model,
                },
            ):
                record_reproducibility()
                native_result = mlflow.genai.evaluate(
                    data=cases,
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
                mlflow.set_tag("aai.gate_passed", str(report.passed).lower())
        finally:
            runner.run(agent.aclose())

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
