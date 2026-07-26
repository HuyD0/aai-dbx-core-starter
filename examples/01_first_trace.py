"""Lifecycle step 1: observe one fictional earnings-summary trace, entirely offline.

This example chooses SDK-managed tracing. OpenAI and LangChain autologging stay
off so the same call cannot produce duplicate spans or duplicate token counts.
The inputs are synthetic and approved for capture; a real application must
select its trace capture policy from the data classification.
"""

from __future__ import annotations

import mlflow
from lifecycle_support import (
    BASELINE_NAME,
    CASES,
    CHANGE_ID,
    CHANGE_NAME,
    CHANGE_SUMMARY,
    DECISION_RULE,
    HYPOTHESIS,
    dataset_digest,
    emit_result,
    generate_response,
    load_context,
    prepare_mlflow,
    quality_score,
)

from aai_core.experiments import (
    ExperimentManager,
    ExperimentRunMetadata,
    RunPurpose,
    record_reproducibility,
)
from aai_core.tracing import TraceIntegration, configure_tracing, provider_span, traced


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

    @traced(name="earnings_summary.baseline", span_type="CHAIN")
    def answer(
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
                "aai.experiment_role": "baseline",
            },
        ) as span:
            output = generate_response(
                "baseline",
                case_id=case_id,
                question=question,
                earnings_excerpt=earnings_excerpt,
                source_id=source_id,
            )
            if span is not None:
                span.set_inputs(
                    {
                        "case_id": case_id,
                        "question": question,
                        "source_id": source_id,
                    }
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

    case = CASES[0]
    with experiments.run(
        run_name="observe-baseline-earnings-summary",
        parameters={
            "hypothesis": HYPOTHESIS,
            "baseline": BASELINE_NAME,
            "change": CHANGE_NAME,
            "decision_rule": DECISION_RULE,
            "dataset_digest_sha256": dataset_digest(),
            "trace_capture_policy": "synthetic_inputs_allowed",
            "autolog_mode": "disabled_manual_spans_only",
        },
        metadata=ExperimentRunMetadata(
            purpose=RunPurpose.BASELINE,
            change_id=CHANGE_ID,
            change_summary=CHANGE_SUMMARY,
            hypothesis=HYPOTHESIS,
        ),
    ) as active_run:
        baseline_run_id = active_run.info.run_id
        output = answer(
            case_id=case.case_id,
            question=case.question,
            earnings_excerpt=case.earnings_excerpt,
            source_id=case.source_id,
        )
        result = {
            "quality_score": quality_score(
                output,
                case.evaluation_record()["expectations"],
            ),
            "latency_ms": output["latency_ms"],
            "total_tokens": output["total_tokens"],
            "cost_usd": output["cost_usd"],
            "cost_coverage": float(output["cost_usd"] is not None),
        }
        mlflow.log_metrics(result)
        record_reproducibility(
            seed=0,
            extra={
                "dataset_digest_sha256": dataset_digest(),
                "experiment_role": "baseline",
            },
        )
        mlflow.set_tags(
            {
                "aai.result": "baseline_observed",
                "aai.decision": "measure_change",
                "aai.release": "blocked_until_evaluated",
            }
        )

    emit_result(
        {
            "stage": "trace",
            "experiment": experiment_name,
            "hypothesis": HYPOTHESIS,
            "baseline": {
                "name": BASELINE_NAME,
                "run_id": baseline_run_id,
            },
            "change": CHANGE_NAME,
            "result": result,
            "decision": "measure_change",
            "release": "blocked_until_evaluated",
            "dataset_digest_sha256": dataset_digest(),
        }
    )


if __name__ == "__main__":
    main()
