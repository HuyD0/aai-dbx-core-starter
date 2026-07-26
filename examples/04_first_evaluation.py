"""Lifecycle step 4: run a deterministic MLflow GenAI comparison and release gate.

The custom scorers are native MLflow APIs and require no LLM judge. A connected
project may add an approved judge only after calibrating it against held-out
human labels; deterministic policy checks remain the first gate.
All model, latency, token, and cost measurements in this credential-free stage
are simulated by the deterministic offline fixture.
"""

from __future__ import annotations

import mlflow
from lifecycle_support import (
    BASELINE_NAME,
    BASELINE_PROMPT,
    CHANGE_ID,
    CHANGE_NAME,
    CHANGE_PROMPT,
    CHANGE_SUMMARY,
    DATASET_NAME,
    DECISION_RULE,
    HYPOTHESIS,
    PROMPT_NAME,
    citation_score,
    critical_case_pass,
    dataset_digest,
    emit_result,
    ensure_prompt_version,
    evaluation_data,
    fact_coverage,
    generate_response,
    load_context,
    prepare_mlflow,
    prompt_digest,
    recommendation_policy_score,
)
from mlflow import MlflowClient
from mlflow.genai.scorers import scorer

from aai_core.evaluation import GatePolicy, MetricDirection, MetricRule, apply_gate
from aai_core.experiments import (
    ExperimentManager,
    ExperimentRunMetadata,
    RunPurpose,
    record_reproducibility,
)
from aai_core.prompts import PromptManager
from aai_core.tracing import TraceIntegration, configure_tracing, provider_span, traced


@scorer(name="quality_score", aggregations=["mean", "min"])
def quality_score(outputs: dict, expectations: dict) -> float:
    fact_score = fact_coverage(outputs, expectations)
    cited = citation_score(outputs, expectations)
    return (fact_score + cited) / 2


@scorer(name="critical_case_pass", aggregations=["mean", "min"])
def critical_pass(outputs: dict, expectations: dict) -> float:
    return critical_case_pass(outputs, expectations)


@scorer(name="latency_ms", aggregations=["mean", "max"])
def latency_ms(outputs: dict) -> float:
    return float(outputs["latency_ms"])


@scorer(name="total_tokens", aggregations=["mean", "max"])
def total_tokens(outputs: dict) -> float:
    return float(outputs["total_tokens"])


@scorer(name="cost_usd", aggregations=["mean", "max"])
def cost_usd(outputs: dict) -> float:
    if outputs["cost_usd"] is None:
        raise ValueError("Cost is unknown; it must never be coerced to zero")
    return float(outputs["cost_usd"])


@scorer(name="cost_coverage", aggregations=["mean", "min"])
def cost_coverage(outputs: dict) -> float:
    return float(outputs["cost_usd"] is not None)


@scorer(name="recommendation_policy_compliance", aggregations=["mean", "min"])
def recommendation_compliance(outputs: dict, expectations: dict) -> float:
    return recommendation_policy_score(outputs, expectations)


SCORERS = (
    quality_score,
    critical_pass,
    latency_ms,
    total_tokens,
    cost_usd,
    cost_coverage,
    recommendation_compliance,
)


def _predict(role: str):
    @traced(name=f"earnings_summary.{role}", span_type="CHAIN")
    def predict(
        *,
        case_id: str,
        question: str,
        earnings_excerpt: str,
        source_id: str,
    ) -> dict:
        with provider_span(
            "deterministic.generate",
            span_type="LLM",
            attributes={
                "aai.model": "offline-deterministic-v1",
                "aai.provider": "simulated_offline_fixture",
                "aai.experiment_role": role,
            },
        ) as span:
            output = generate_response(
                role,
                case_id=case_id,
                question=question,
                earnings_excerpt=earnings_excerpt,
                source_id=source_id,
            )
            span.set_outputs(output)
            span.set_attribute(
                "mlflow.chat.tokenUsage",
                {
                    "input_tokens": output["input_tokens"],
                    "output_tokens": output["output_tokens"],
                    "total_tokens": output["total_tokens"],
                },
            )
            return output

    return predict


def _numeric_metrics(result) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in result.metrics.items()
        if isinstance(value, (int, float))
    }


def _require_no_scorer_errors(result) -> None:
    frame = result.result_df
    for column in frame.columns:
        if str(column).endswith("/error_message") and frame[column].notna().any():
            raise RuntimeError(
                f"Evaluation scorer failed for one or more rows: {column}"
            )


def main() -> None:
    ctx = load_context()
    experiment_name = prepare_mlflow(ctx)
    configure_tracing(
        ctx.tags,
        experiment_name=experiment_name,
        integration=TraceIntegration.SDK,
    )
    experiments = ExperimentManager(
        experiment_name=experiment_name,
        context=ctx.tags,
    )
    prompts = PromptManager(
        context=ctx.tags,
        catalog=ctx.settings.catalog,
        schema=ctx.settings.schema,
    )
    baseline_prompt = ensure_prompt_version(
        prompts,
        role="baseline",
        template=BASELINE_PROMPT,
    )
    change_prompt = ensure_prompt_version(
        prompts,
        role="change",
        template=CHANGE_PROMPT,
    )

    data = evaluation_data()
    exact_digest = dataset_digest()

    with experiments.run(
        run_name="evaluate-baseline-earnings-summary-prompt-v1",
        parameters={
            "hypothesis": HYPOTHESIS,
            "baseline": BASELINE_NAME,
            "change": CHANGE_NAME,
            "decision_rule": DECISION_RULE,
            "prompt_uri": baseline_prompt.uri,
            "evaluation_dataset": f"inline:{DATASET_NAME}",
            "dataset_digest_sha256": exact_digest,
            "prompt_digest_sha256": prompt_digest(BASELINE_PROMPT),
            "evaluation_repetitions": 1,
            "determinism": "simulated_offline_fixture",
        },
        metadata=ExperimentRunMetadata(
            purpose=RunPurpose.BASELINE,
            change_id=CHANGE_ID,
            change_summary=CHANGE_SUMMARY,
            hypothesis=HYPOTHESIS,
        ),
    ) as active_run:
        baseline_run_id = active_run.info.run_id
        MlflowClient().link_prompt_version_to_run(baseline_run_id, baseline_prompt)
        baseline_result = mlflow.genai.evaluate(
            data=data,
            predict_fn=_predict("baseline"),
            scorers=list(SCORERS),
        )
        _require_no_scorer_errors(baseline_result)
        baseline_metrics = _numeric_metrics(baseline_result)
        mlflow.log_metrics(baseline_metrics)
        record_reproducibility(
            seed=0,
            extra={
                "dataset_digest_sha256": exact_digest,
                "prompt_digest_sha256": prompt_digest(BASELINE_PROMPT),
                "experiment_role": "baseline",
            },
        )
        mlflow.set_tags(
            {
                "aai.gate_passed": "false",
                "aai.result": "baseline_measured",
                "aai.decision": "evaluate_change",
                "aai.release": "blocked_until_change_passes",
            }
        )

    with experiments.run(
        run_name="evaluate-change-cited-earnings-summary-prompt-v2",
        parameters={
            "hypothesis": HYPOTHESIS,
            "baseline": BASELINE_NAME,
            "change": CHANGE_NAME,
            "decision_rule": DECISION_RULE,
            "baseline_run_id": baseline_run_id,
            "prompt_uri": change_prompt.uri,
            "evaluation_dataset": f"inline:{DATASET_NAME}",
            "dataset_digest_sha256": exact_digest,
            "prompt_digest_sha256": prompt_digest(CHANGE_PROMPT),
            "evaluation_repetitions": 1,
            "determinism": "simulated_offline_fixture",
        },
        metadata=ExperimentRunMetadata(
            purpose=RunPurpose.CHANGE,
            change_id=CHANGE_ID,
            change_summary=CHANGE_SUMMARY,
            hypothesis=HYPOTHESIS,
            baseline_run_id=baseline_run_id,
        ),
    ) as active_run:
        change_run_id = active_run.info.run_id
        MlflowClient().link_prompt_version_to_run(change_run_id, change_prompt)
        change_result = mlflow.genai.evaluate(
            data=data,
            predict_fn=_predict("change"),
            scorers=list(SCORERS),
        )
        _require_no_scorer_errors(change_result)
        change_metrics = _numeric_metrics(change_result)
        mlflow.log_metrics(change_metrics)
        record_reproducibility(
            seed=0,
            extra={
                "dataset_digest_sha256": exact_digest,
                "prompt_digest_sha256": prompt_digest(CHANGE_PROMPT),
                "experiment_role": "change",
            },
        )

    summary_baseline = {
        "quality_score": baseline_metrics["quality_score/mean"],
        "minimum_row_quality": baseline_metrics["quality_score/min"],
        "critical_case_pass_rate": baseline_metrics["critical_case_pass/mean"],
        "recommendation_policy_compliance": baseline_metrics[
            "recommendation_policy_compliance/min"
        ],
        "latency_ms_mean": baseline_metrics["latency_ms/mean"],
        "total_tokens": baseline_metrics["total_tokens/mean"] * len(data),
        "cost_usd_total": baseline_metrics["cost_usd/mean"] * len(data),
        "cost_coverage": baseline_metrics["cost_coverage/mean"],
    }
    summary_change = {
        "quality_score": change_metrics["quality_score/mean"],
        "minimum_row_quality": change_metrics["quality_score/min"],
        "critical_case_pass_rate": change_metrics["critical_case_pass/mean"],
        "recommendation_policy_compliance": change_metrics[
            "recommendation_policy_compliance/min"
        ],
        "latency_ms_mean": change_metrics["latency_ms/mean"],
        "total_tokens": change_metrics["total_tokens/mean"] * len(data),
        "cost_usd_total": change_metrics["cost_usd/mean"] * len(data),
        "cost_coverage": change_metrics["cost_coverage/mean"],
    }
    gate_policy = GatePolicy(
        rules=(
            MetricRule(
                metric="quality_score",
                direction=MetricDirection.HIGHER,
                required=0.90,
                max_regression=0.0,
            ),
            MetricRule(
                metric="minimum_row_quality",
                direction=MetricDirection.HIGHER,
                required=0.90,
            ),
            MetricRule(
                metric="critical_case_pass_rate",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
            MetricRule(
                metric="recommendation_policy_compliance",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
            MetricRule(
                metric="cost_coverage",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
            MetricRule(
                metric="latency_ms_mean",
                direction=MetricDirection.LOWER,
                required=summary_baseline["latency_ms_mean"] * 1.25,
            ),
            MetricRule(
                metric="total_tokens",
                direction=MetricDirection.LOWER,
                required=summary_baseline["total_tokens"] * 1.30,
            ),
            MetricRule(
                metric="cost_usd_total",
                direction=MetricDirection.LOWER,
                required=summary_baseline["cost_usd_total"] * 1.30,
            ),
        )
    )
    gate = apply_gate(
        summary_change,
        policy=gate_policy,
        baseline_metrics=summary_baseline,
    )
    gate_passed = gate.passed
    decision = "release_change" if gate_passed else "keep_baseline"
    release = "earnings-summary-prompt-v2" if gate_passed else "blocked"

    MlflowClient().set_tag(
        change_run_id,
        "aai.decision",
        decision,
    )
    MlflowClient().set_tag(
        change_run_id,
        "aai.release",
        release,
    )
    MlflowClient().set_tag(
        change_run_id,
        "aai.gate_passed",
        str(gate_passed).lower(),
    )
    MlflowClient().set_tag(
        change_run_id,
        "aai.result",
        "change_evaluated",
    )
    gate.require_passed()
    # An alias is a deployment pointer, not evaluation evidence. Move it only
    # after the exact change version has passed every release-grade gate.
    prompts.set_alias(
        PROMPT_NAME,
        alias="production",
        version=int(change_prompt.version),
    )
    MlflowClient().set_tag(
        change_run_id,
        "aai.prompt_alias_promoted",
        "production",
    )

    emit_result(
        {
            "stage": "evaluation",
            "experiment": experiment_name,
            "hypothesis": HYPOTHESIS,
            "baseline": {
                "name": BASELINE_NAME,
                "run_id": baseline_run_id,
                "prompt_uri": baseline_prompt.uri,
                "metrics": baseline_metrics,
            },
            "change": {
                "name": CHANGE_NAME,
                "run_id": change_run_id,
                "prompt_uri": change_prompt.uri,
                "metrics": change_metrics,
            },
            "result": {
                "gate_passed": gate_passed,
                "checks": {
                    rule.metric: all(
                        failure.metric != rule.metric for failure in gate.failures
                    )
                    for rule in gate_policy.rules
                },
                "failures": [
                    failure.model_dump(mode="json") for failure in gate.failures
                ],
            },
            "decision": decision,
            "release": release,
            "dataset_digest_sha256": exact_digest,
        }
    )


if __name__ == "__main__":
    main()
