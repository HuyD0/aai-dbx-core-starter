"""Step 2: compare a baseline and one deliberate change.

Both runs use the same ordered cases and dataset digest. The SDK owns governed
run context; native MLflow APIs log the input dataset, metrics, and decision.
All model, latency, token, and cost measurements are explicitly simulated by
the deterministic offline fixture.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from lifecycle_support import (
    BASELINE_NAME,
    CHANGE_ID,
    CHANGE_NAME,
    CHANGE_SUMMARY,
    DATASET_NAME,
    DECISION_RULE,
    HYPOTHESIS,
    dataset_digest,
    emit_result,
    evaluation_data,
    generate_response,
    load_context,
    metrics_for,
    prepare_mlflow,
    release_decision,
)

from aai_core.experiments import (
    ExperimentManager,
    ExperimentRunMetadata,
    RunPurpose,
    record_reproducibility,
)
from aai_core.tracing import TraceIntegration, configure_tracing, provider_span, traced


def _dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": record["inputs"]["case_id"],
                "question": record["inputs"]["question"],
                "source_id": record["inputs"]["source_id"],
                "required_facts": "|".join(record["expectations"]["required_facts"]),
            }
            for record in evaluation_data()
        ]
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

    @traced(name="earnings_summary.experiment", span_type="CHAIN")
    def predict(role: str, inputs: dict) -> dict:
        with provider_span(
            "deterministic.generate",
            span_type="LLM",
            attributes={
                "aai.model": "offline-deterministic-v1",
                "aai.provider": "simulated_offline_fixture",
                "aai.experiment_role": role,
            },
        ):
            return generate_response(role, **inputs)

    frame = _dataset_frame()
    native_dataset = mlflow.data.from_pandas(
        frame,
        name=DATASET_NAME,
        source=Path(__file__).with_name("lifecycle_support.py").resolve().as_uri(),
    )
    exact_digest = dataset_digest()
    results: dict[str, dict[str, float]] = {}
    run_ids: dict[str, str] = {}

    for role, run_name in (
        ("baseline", BASELINE_NAME),
        ("change", CHANGE_NAME),
    ):
        with experiments.run(
            run_name=run_name,
            parameters={
                "hypothesis": HYPOTHESIS,
                "baseline": BASELINE_NAME,
                "change": CHANGE_NAME,
                "decision_rule": DECISION_RULE,
                "dataset_digest_sha256": exact_digest,
                "mlflow_dataset_digest": native_dataset.digest,
                "dataset_case_count": len(frame),
                "model": "offline-deterministic-v1",
                "measurement_source": "simulated_offline_fixture",
                "trace_mode": "sdk_manual",
            },
            tags={"experiment_role": role},
            metadata=ExperimentRunMetadata(
                purpose=(
                    RunPurpose.BASELINE if role == "baseline" else RunPurpose.CHANGE
                ),
                change_id=CHANGE_ID,
                change_summary=CHANGE_SUMMARY,
                hypothesis=HYPOTHESIS,
                baseline_run_id=run_ids.get("baseline"),
            ),
        ) as active_run:
            run_ids[role] = active_run.info.run_id
            mlflow.log_input(native_dataset, context="evaluation")
            for record in evaluation_data():
                predict(role, record["inputs"])
            metrics = metrics_for(role)
            mlflow.log_metrics(metrics)
            record_reproducibility(
                seed=0,
                extra={
                    "dataset_digest_sha256": exact_digest,
                    "experiment_role": role,
                },
            )
            results[role] = metrics
            if role == "change":
                comparison = release_decision(results["baseline"], metrics)
                mlflow.set_tags(
                    {
                        "aai.result": "change_measured",
                        "aai.decision": "record_exact_prompt_lineage",
                        "aai.release": "blocked_until_evaluated",
                    }
                )

    emit_result(
        {
            "stage": "experiment",
            "experiment": experiment_name,
            "hypothesis": HYPOTHESIS,
            "baseline": {
                "name": BASELINE_NAME,
                "run_id": run_ids["baseline"],
                "metrics": results["baseline"],
            },
            "change": {
                "name": CHANGE_NAME,
                "run_id": run_ids["change"],
                "baseline_run_id": run_ids["baseline"],
                "metrics": results["change"],
            },
            "result": comparison["checks"],
            "decision": "record_exact_prompt_lineage",
            "release": "blocked_until_evaluated",
            "dataset_digest_sha256": exact_digest,
        }
    )


if __name__ == "__main__":
    main()
