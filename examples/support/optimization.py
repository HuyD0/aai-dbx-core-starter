"""Dataset isolation and guarded connected optimization for lesson 12."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping
from functools import partial
from typing import Any

from aai_core.contracts import ContractModel

JsonRecord = dict[str, Any]

SPLIT_MANIFEST: dict[str, list[str]] = {
    "judge_calibration": [
        "cal-revenue-01",
        "cal-margin-01",
        "cal-guidance-01",
        "cal-cash-01",
        "cal-risk-01",
        "cal-policy-01",
    ],
    "optimizer_training": [
        "train-revenue-01",
        "train-margin-01",
        "train-guidance-01",
        "train-cash-01",
        "train-risk-01",
        "train-policy-01",
    ],
    "held_out_release": [
        "holdout-revenue-01",
        "holdout-margin-01",
        "holdout-guidance-01",
        "holdout-cash-01",
        "holdout-risk-01",
        "holdout-policy-01",
    ],
}

TOPIC_FIXTURES: dict[str, JsonRecord] = {
    "revenue": {
        "question": "What was fictional quarterly revenue?",
        "earnings_excerpt": "Revenue was $128.4 million, up 12%.",
        "source_id": "ARS-FY25-Q2-RESULTS",
        "required_fact": "$128.4 million",
    },
    "margin": {
        "question": "What was fictional operating margin?",
        "earnings_excerpt": "Operating margin was 18.6% versus 16.9%.",
        "source_id": "ARS-FY25-Q2-RESULTS",
        "required_fact": "18.6%",
    },
    "guidance": {
        "question": "What fictional revenue guidance was supplied?",
        "earnings_excerpt": "Revenue guidance was $132 million to $136 million.",
        "source_id": "ARS-FY25-Q2-GUIDANCE",
        "required_fact": "$132 million to $136 million",
    },
    "cash": {
        "question": "What was fictional free cash flow?",
        "earnings_excerpt": "Free cash flow was $21.7 million.",
        "source_id": "ARS-FY25-Q2-CASH-RISK",
        "required_fact": "$21.7 million",
    },
    "risk": {
        "question": "What fictional supplier risk was disclosed?",
        "earnings_excerpt": "Single-source supplier concentration is a risk.",
        "source_id": "ARS-FY25-Q2-CASH-RISK",
        "required_fact": "single-source supplier concentration",
    },
    "policy": {
        "question": "Should I buy shares based on this fictional excerpt?",
        "earnings_excerpt": "The excerpt contains historical fictional results only.",
        "source_id": "ARS-FY25-Q2-RESULTS",
        "required_fact": "cannot provide investment advice",
    },
}


def split_record(case_id: str) -> JsonRecord:
    fixture = TOPIC_FIXTURES[case_id.split("-")[1]]
    expected = f"{fixture['required_fact']} [source: {fixture['source_id']}]"
    return {
        "inputs": {
            "question": fixture["question"],
            "earnings_excerpt": fixture["earnings_excerpt"],
            "source_id": fixture["source_id"],
        },
        "outputs": expected,
        "expectations": {
            "required_facts": [fixture["required_fact"]],
            "source_id": fixture["source_id"],
            "no_investment_recommendation": True,
        },
    }


SPLIT_RECORDS: dict[str, list[JsonRecord]] = {
    split: [split_record(case_id) for case_id in case_ids]
    for split, case_ids in SPLIT_MANIFEST.items()
}
SPLIT_MANIFEST_DIGEST = hashlib.sha256(
    json.dumps(
        SPLIT_MANIFEST,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
).hexdigest()
OPTIMIZATION_BUDGET = {
    "max_metric_calls": 30,
    "max_training_cases": len(SPLIT_MANIFEST["optimizer_training"]),
    "evaluation_concurrency": 1,
    "request_timeout_seconds": 60,
}


def split_contract_summary() -> JsonRecord:
    split_sets = {name: set(case_ids) for name, case_ids in SPLIT_MANIFEST.items()}
    if any(
        left & right
        for index, left in enumerate(split_sets.values())
        for right in list(split_sets.values())[index + 1 :]
    ):
        raise ValueError("Optimization evidence splits must be disjoint")
    if sum(map(len, split_sets.values())) != len(set().union(*split_sets.values())):
        raise ValueError("Optimization case IDs must be globally unique")
    return {
        "split_manifest_digest": SPLIT_MANIFEST_DIGEST,
        "case_counts": {name: len(case_ids) for name, case_ids in split_sets.items()},
    }


def experimental_dependency_status() -> JsonRecord:
    dependencies = {
        "dspy": importlib.util.find_spec("dspy") is not None,
        "gepa": importlib.util.find_spec("gepa") is not None,
    }
    return {
        "optimization_budget": OPTIMIZATION_BUDGET,
        "experimental_dependencies": dependencies,
        "ready": all(dependencies.values()),
    }


class AlignmentConfig(ContractModel):
    persist_evidence: bool = False
    run_optimization: bool = False
    judge_name: str = "uncertainty_explanation"
    judge_experiment_id: str | None = None
    seed_prompt_uri: str | None = None
    reflection_model_uri: str | None = None
    aligned_judge_version: int | None = None
    judge_validation_run_id: str | None = None
    judge_validation_agreement: float | None = None
    judge_validation_label_count: int | None = None


def _require_experimental_readiness(config: AlignmentConfig) -> None:
    required = {
        "JUDGE_EXPERIMENT_ID": config.judge_experiment_id,
        "SEED_PROMPT_URI": config.seed_prompt_uri,
        "REFLECTION_MODEL_URI": config.reflection_model_uri,
        "ALIGNED_JUDGE_VERSION": config.aligned_judge_version,
        "JUDGE_VALIDATION_RUN_ID": config.judge_validation_run_id,
        "JUDGE_VALIDATION_AGREEMENT": config.judge_validation_agreement,
        "JUDGE_VALIDATION_LABEL_COUNT": config.judge_validation_label_count,
    }
    missing = [name for name, value in required.items() if value in (None, "")]
    if missing:
        raise ValueError(f"Configure the connected lab first: {missing}")
    if not experimental_dependency_status()["ready"]:
        raise RuntimeError(
            "DSPy and GEPA are not in the certified locks; complete the dependency "
            "policy and compatibility workflow first"
        )
    if (
        config.judge_validation_agreement is None
        or config.judge_validation_label_count is None
        or config.judge_validation_agreement < 0.75
        or config.judge_validation_label_count < 50
    ):
        raise RuntimeError(
            "The aligned judge has not passed held-out agreement and sample-size "
            "requirements"
        )


def _build_predictor(
    *,
    mlflow: Any,
    client: Any,
    connected: Any,
    trace_ids_by_prompt: dict[str, list[str]],
) -> Any:
    from aai_core.tracing import set_trace_resource_context

    def predict_with_prompt_uri(
        prompt_uri: str,
        question: str,
        earnings_excerpt: str,
        source_id: str,
    ) -> str:
        active_prompt = mlflow.genai.load_prompt(prompt_uri)
        rendered = active_prompt.format(
            question=question,
            earnings_excerpt=earnings_excerpt,
            source_id=source_id,
        )
        with mlflow.start_span(
            name="earnings_summary.optimization_prediction",
            span_type="CHAIN",
        ) as application_span:
            set_trace_resource_context(connected.context.tags)
            application_span.set_attribute("mlflow.message.format", "openai")
            application_span.set_inputs(
                {"messages": [{"role": "user", "content": rendered}]}
            )
            mlflow.update_current_trace(request_preview=question)
            response = connected.model.generate(
                [{"role": "user", "content": rendered}],
                temperature=0.0,
                max_tokens=400,
            )
            application_span.set_outputs({"content": response.content})
            mlflow.update_current_trace(response_preview=response.content)
        mlflow.flush_trace_async_logging()
        trace_id = mlflow.get_last_active_trace_id()
        if trace_id is None:
            raise RuntimeError("Prediction did not produce a trace")
        client.link_prompt_versions_to_trace(
            prompt_versions=[active_prompt],
            trace_id=trace_id,
        )
        trace_ids_by_prompt.setdefault(prompt_uri, []).append(trace_id)
        return str(response.content)

    return predict_with_prompt_uri


def _evaluate_heldout_prompts(
    *,
    mlflow: Any,
    client: Any,
    experiments: Any,
    dataset: Any,
    aligned_judge: Any,
    seed_prompt: Any,
    optimized_prompt: Any,
    predict_with_prompt_uri: Any,
) -> dict[str, Any]:
    from aai_core.experiments import ExperimentRunMetadata, RunPurpose

    results = {}
    for role, prompt_version in (
        ("baseline", seed_prompt),
        ("optimized", optimized_prompt),
    ):
        with experiments.run(
            run_name=f"agent-alignment-heldout-{role}",
            description=(
                f"Observed held-out release evidence for the exact {role} prompt "
                "version; no alias is moved by this run."
            ),
            nested=True,
            metadata=ExperimentRunMetadata(
                purpose=(
                    RunPurpose.BASELINE if role == "baseline" else RunPurpose.CHANGE
                ),
                change_id="agent-alignment-optimization-v1",
                change_summary="Evaluate optimized prompt on held-out data.",
            ),
        ) as heldout_run:
            mlflow.log_input(dataset, context="held_out_release")
            client.link_prompt_version_to_run(
                heldout_run.info.run_id,
                prompt_version,
            )
            results[role] = mlflow.genai.evaluate(
                data=dataset,
                predict_fn=partial(predict_with_prompt_uri, prompt_version.uri),
                scorers=[aligned_judge],
            )
    return results


def _run_experimental_optimization(
    *,
    config: AlignmentConfig,
    mlflow: Any,
    client: Any,
    experiments: Any,
    connected: Any,
    datasets: Mapping[str, Any],
    optimization_run: Any,
) -> JsonRecord:
    from mlflow.genai.optimize import GepaPromptOptimizer
    from mlflow.genai.scorers import get_scorer

    from aai_core.tracing import (
        TraceCaptureMode,
        TraceIntegration,
        TracePolicy,
        configure_tracing,
    )

    _require_experimental_readiness(config)
    configure_tracing(
        connected.context.tags,
        experiment_name=connected.experiment_name,
        integration=TraceIntegration.SDK,
        policy=TracePolicy(capture_mode=TraceCaptureMode.FULL),
    )
    aligned_judge = get_scorer(
        name=config.judge_name,
        experiment_id=config.judge_experiment_id,
        version=config.aligned_judge_version,
    )
    seed_prompt = mlflow.genai.load_prompt(config.seed_prompt_uri)
    client.link_prompt_version_to_run(optimization_run.info.run_id, seed_prompt)
    trace_ids_by_prompt: dict[str, list[str]] = {}
    predictor = _build_predictor(
        mlflow=mlflow,
        client=client,
        connected=connected,
        trace_ids_by_prompt=trace_ids_by_prompt,
    )
    optimization_result = mlflow.genai.optimize_prompts(
        predict_fn=partial(predictor, config.seed_prompt_uri),
        train_data=datasets["optimizer_training"],
        prompt_uris=[config.seed_prompt_uri],
        optimizer=GepaPromptOptimizer(
            reflection_model=config.reflection_model_uri,
            max_metric_calls=OPTIMIZATION_BUDGET["max_metric_calls"],
            display_progress_bar=True,
        ),
        scorers=[aligned_judge],
    )
    optimized_prompt = optimization_result.optimized_prompts[0]
    client.link_prompt_version_to_run(optimization_run.info.run_id, optimized_prompt)
    heldout = _evaluate_heldout_prompts(
        mlflow=mlflow,
        client=client,
        experiments=experiments,
        dataset=datasets["held_out_release"],
        aligned_judge=aligned_judge,
        seed_prompt=seed_prompt,
        optimized_prompt=optimized_prompt,
        predict_with_prompt_uri=predictor,
    )
    metric_name = f"{aligned_judge.name}/mean"
    scores = {role: result.metrics.get(metric_name) for role, result in heldout.items()}
    if any(score is None for score in scores.values()):
        raise RuntimeError(f"Held-out results lack required metric {metric_name!r}")
    decision = "adopt" if scores["optimized"] >= scores["baseline"] else "reject"
    client.set_tag(optimization_run.info.run_id, "aai.decision", decision)
    return {
        "initial_score": optimization_result.initial_eval_score,
        "final_score": optimization_result.final_eval_score,
        "optimized_prompt_uri": optimized_prompt.uri,
        "heldout_scores": scores,
        "linked_trace_ids": trace_ids_by_prompt,
        "decision": decision,
        "alias_moved": False,
    }


def run_alignment_workflow(config: AlignmentConfig) -> JsonRecord:
    """Register isolated datasets and optionally run the guarded optimizer."""

    import mlflow
    from mlflow import MlflowClient

    from aai_core.experiments import (
        ExperimentManager,
        ExperimentRunMetadata,
        RunPurpose,
    )
    from examples.notebook_setup import (
        get_or_create_uc_evaluation_dataset,
        preflight_databricks,
        preflight_databricks_evidence,
        prepare_notebook_environment,
    )

    environment = prepare_notebook_environment(evidence_destination="databricks")
    evidence = preflight_databricks_evidence(environment)
    connected = preflight_databricks(environment) if config.run_optimization else None
    datasets = {
        split: get_or_create_uc_evaluation_dataset(
            evidence=evidence,
            dataset_name=f"fictional_{split}_v1",
            records=SPLIT_RECORDS[split],
            mlflow_module=mlflow,
        )
        for split in SPLIT_MANIFEST
    }
    experiments = ExperimentManager(
        experiment_name=evidence.experiment_name,
        context=evidence.context.tags,
    )
    client = MlflowClient()
    with experiments.run(
        run_name="agent-alignment-governed-evidence",
        description=(
            "Governed registration of disjoint judge-calibration, optimizer-"
            "training, and held-out release datasets. Optimization remains "
            "experimental and cannot move a production alias."
        ),
        parameters={
            "split_manifest_digest_sha256": SPLIT_MANIFEST_DIGEST,
            "optimization_enabled": config.run_optimization,
        },
        metadata=ExperimentRunMetadata(
            purpose=RunPurpose.RESULT,
            change_id="agent-alignment-optimization-v1",
            change_summary="Optimize one prompt with disjoint governed evidence.",
        ),
    ) as optimization_run:
        for split, dataset in datasets.items():
            mlflow.log_input(dataset, context=split)
        if config.run_optimization:
            if connected is None:
                raise RuntimeError("Connected model preflight did not complete")
            return _run_experimental_optimization(
                config=config,
                mlflow=mlflow,
                client=client,
                experiments=experiments,
                connected=connected,
                datasets=datasets,
                optimization_run=optimization_run,
            )
        return {
            "run_id": optimization_run.info.run_id,
            "datasets": {split: dataset.name for split, dataset in datasets.items()},
            "optimization": "skipped",
        }


def optimization_plan() -> JsonRecord:
    return {
        "stage": "optimization_plan",
        "split_manifest_digest": SPLIT_MANIFEST_DIGEST,
        "experimental_dependencies_ready": experimental_dependency_status()["ready"],
        "decision": "inconclusive",
        "release": "blocked",
        "reason": (
            "optimization is disabled and cannot authorize release; register any "
            "proposed prompt and run the final held-out gate"
        ),
    }


__all__ = [
    "AlignmentConfig",
    "OPTIMIZATION_BUDGET",
    "SPLIT_MANIFEST",
    "SPLIT_MANIFEST_DIGEST",
    "SPLIT_RECORDS",
    "TOPIC_FIXTURES",
    "experimental_dependency_status",
    "optimization_plan",
    "run_alignment_workflow",
    "split_contract_summary",
    "split_record",
]
