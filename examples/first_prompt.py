"""Step 3: register immutable prompts and bind exact versions to a run.

No mutable alias is used as evaluation or release evidence. Re-running the
example reuses an identical version by digest instead of creating duplicates.
Framework autologgers are deliberately not enabled: this script makes no model
or framework call for them to instrument. Instead, the governed MLflow run, a
manual ``PROMPT`` span, native prompt-version links, and a safe synthetic render
track the registry operations without inventing an autologging use case.
"""

from __future__ import annotations

import mlflow
from lifecycle_support import (
    BASELINE_NAME,
    BASELINE_PROMPT,
    CASES,
    CHANGE_ID,
    CHANGE_NAME,
    CHANGE_PROMPT,
    CHANGE_SUMMARY,
    DECISION_RULE,
    HYPOTHESIS,
    PROMPT_NAME,
    dataset_digest,
    emit_result,
    ensure_prompt_version,
    load_context,
    prepare_mlflow,
    prompt_digest,
    render_prompt,
)
from mlflow import MlflowClient

from aai_core.experiments import (
    ExperimentManager,
    ExperimentRunMetadata,
    RunPurpose,
    record_reproducibility,
)
from aai_core.prompts import PromptManager
from aai_core.tracing import TraceIntegration, configure_tracing, provider_span


def main() -> None:
    ctx = load_context()
    experiment_name = prepare_mlflow(ctx)
    configure_tracing(
        ctx.tags,
        experiment_name=experiment_name,
        integration=TraceIntegration.SDK,
    )
    prompts = PromptManager(
        context=ctx.tags,
        catalog=ctx.settings.catalog,
        schema=ctx.settings.schema,
    )
    experiments = ExperimentManager(
        experiment_name=experiment_name,
        context=ctx.tags,
    )

    with experiments.run(
        run_name="bind-exact-prompt-lineage",
        parameters={
            "hypothesis": HYPOTHESIS,
            "baseline": BASELINE_NAME,
            "change": CHANGE_NAME,
            "decision_rule": DECISION_RULE,
            "dataset_digest_sha256": dataset_digest(),
            "autolog_mode": "not_applicable_no_model_or_framework_call",
            "trace_capture_policy": "synthetic_render_allowed",
        },
        metadata=ExperimentRunMetadata(
            purpose=RunPurpose.CHANGE,
            change_id=CHANGE_ID,
            change_summary=CHANGE_SUMMARY,
            hypothesis=HYPOTHESIS,
        ),
    ) as active_run:
        run_id = active_run.info.run_id
        with provider_span(
            "prompt.register_load_render",
            span_type="PROMPT",
            attributes={
                "aai.prompt.name": prompts.qualify(PROMPT_NAME),
                "aai.autolog_mode": ("not_applicable_no_model_or_framework_call"),
                "aai.trace_capture_policy": "synthetic_render_allowed",
            },
        ) as span:
            baseline = ensure_prompt_version(
                prompts,
                role="baseline",
                template=BASELINE_PROMPT,
            )
            change = ensure_prompt_version(
                prompts,
                role="change",
                template=CHANGE_PROMPT,
            )
            loaded_baseline = prompts.load(
                PROMPT_NAME,
                version=int(baseline.version),
            )
            loaded_change = prompts.load(
                PROMPT_NAME,
                version=int(change.version),
            )

            baseline_digest = prompt_digest(loaded_baseline.template)
            change_digest = prompt_digest(loaded_change.template)
            if baseline_digest != prompt_digest(BASELINE_PROMPT):
                raise RuntimeError(
                    "Loaded baseline prompt does not match its expected digest"
                )
            if change_digest != prompt_digest(CHANGE_PROMPT):
                raise RuntimeError(
                    "Loaded change prompt does not match its expected digest"
                )

            case = CASES[0]
            rendered_change = render_prompt(
                loaded_change.template,
                question=case.question,
                earnings_excerpt=case.earnings_excerpt,
                source_id=case.source_id,
            )
            if span is not None:
                span.set_inputs(
                    {
                        "prompt_name": prompts.qualify(PROMPT_NAME),
                        "baseline_prompt_digest": baseline_digest,
                        "change_prompt_digest": change_digest,
                    }
                )
                span.set_outputs(
                    {
                        "baseline_prompt_uri": baseline.uri,
                        "change_prompt_uri": change.uri,
                        "rendered_case_id": case.case_id,
                    }
                )

        client = MlflowClient()
        client.link_prompt_version_to_run(run_id, baseline)
        client.link_prompt_version_to_run(run_id, change)
        mlflow.log_params(
            {
                "baseline_prompt_uri": baseline.uri,
                "baseline_prompt_digest": baseline_digest,
                "change_prompt_uri": change.uri,
                "change_prompt_digest": change_digest,
            }
        )
        record_reproducibility(
            extra={
                "dataset_digest_sha256": dataset_digest(),
                "baseline_prompt_digest": baseline_digest,
                "change_prompt_digest": change_digest,
            }
        )
        mlflow.log_text(rendered_change, "prompt/rendered_synthetic_example.txt")
        mlflow.set_tags(
            {
                "aai.result": "exact_versions_bound",
                "aai.decision": "evaluate_change",
                "aai.release": "blocked_until_evaluated",
                "aai.autolog_explanation": (
                    "No model or framework call; manual PROMPT span used"
                ),
            }
        )

    emit_result(
        {
            "stage": "prompt",
            "experiment": experiment_name,
            "hypothesis": HYPOTHESIS,
            "run_id": run_id,
            "baseline": {
                "name": BASELINE_NAME,
                "prompt_uri": baseline.uri,
                "prompt_digest": baseline_digest,
            },
            "change": {
                "name": CHANGE_NAME,
                "prompt_uri": change.uri,
                "prompt_digest": change_digest,
            },
            "result": {
                "exact_versions_linked": True,
                "manual_prompt_span": True,
                "autolog": "not_applicable_no_model_or_framework_call",
                "measurement_status": {
                    "quality": "pending_full_evaluation",
                    "latency": "pending_model_execution",
                    "tokens": "pending_model_execution",
                    "cost": "pending_model_execution",
                    "cost_coverage": "pending_model_execution",
                },
            },
            "decision": "evaluate_change",
            "release": "blocked_until_evaluated",
            "dataset_digest_sha256": dataset_digest(),
        }
    )


if __name__ == "__main__":
    main()
