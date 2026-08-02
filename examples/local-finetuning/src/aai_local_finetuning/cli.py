"""Command-line workflow for the offline local fine-tuning curriculum."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

from .capstone import (
    CAPSTONE_SYSTEM_PROMPT,
    CapstoneEvaluationReport,
    CapstonePrediction,
    CapstoneRecord,
    ReadinessStatus,
    build_hybrid_review,
    compact_expected,
    compact_review,
    deterministic_capstone_predictions,
    evaluate_capstone_predictions,
    generate_capstone_dataset,
    load_capstone_records,
    render_capstone_mlx_dataset,
    rule_catalog,
)
from .data import (
    DatasetIntegrityError,
    PreparationConfig,
    assert_no_leakage,
    check_split_files,
    prepare_dataset,
    require_valid_manifest,
)
from .evaluation import (
    BaselineEvaluation,
    DeterministicInferenceConfig,
    EvaluationRecord,
    EvaluationReport,
    EvaluationSession,
    Evaluator,
    InferenceConfig,
    KeywordRuleBaseline,
    LocalMLXInferenceConfig,
    MajorityBaseline,
    Prediction,
    PromotionAssessment,
    build_local_mlx_inference_config,
    decide_lora_promotion,
    format_error_analysis,
    load_records_jsonl,
    recheck_evaluation_session,
    start_evaluation_session,
    write_predictions_jsonl,
    write_report_json,
)
from .modeling import LocalMLXPredictor, PromptStrategy, build_messages
from .offline import (
    OfflineAssetError,
    apple_silicon_status,
    deny_network,
    enable_offline_environment,
    prove_socket_denial,
    require_assets,
    verify_flight_manifest,
    write_flight_manifest,
)
from .settings import PROJECT_ROOT, ProjectSettings, load_settings
from .training import (
    TRAINING_MANIFEST_NAME,
    TrainingManifestError,
    ValidatedTrainingSnapshot,
    exclusive_adapter_lock,
    recheck_training_snapshot,
    require_valid_training_snapshot,
    run_lora,
    shared_adapter_lock,
)


class StudyCommandError(RuntimeError):
    """A study command could not satisfy its documented contract."""


def _preparation_config(settings: ProjectSettings) -> PreparationConfig:
    counts = settings.dataset.per_intent
    return PreparationConfig(
        seed=settings.dataset.split_seed,
        train_per_intent=counts.train,
        validation_per_intent=counts.validation,
        test_per_intent=counts.test,
        expected_intent_count=27,
        near_duplicate_threshold=0.9,
        source_dataset="bitext-customer-support",
        source_version=f"kaggle-v{settings.dataset.version}",
        source_provider="kaggle",
        source_owner=settings.dataset.owner,
        source_url=settings.dataset.url,
        source_license=settings.dataset.license,
        date_accessed=settings.dataset.accessed_on,
        processing_config_path="configs/project.yaml",
        output_version="1.0.0",
    )


def _prepare_data(settings: ProjectSettings) -> None:
    result = prepare_dataset(
        settings.archive_path,
        settings.processed_dir,
        _preparation_config(settings),
        related_raw_paths=(settings.csv_path,),
    )
    counts = {
        name: descriptor.record_count
        for name, descriptor in result.manifest.splits.items()
    }
    print(
        "Prepared leakage-safe Bitext splits: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    print(f"Dataset fingerprint: {result.manifest.dataset_fingerprint}")


def _load_splits(
    settings: ProjectSettings,
) -> tuple[list[EvaluationRecord], list[EvaluationRecord], list[EvaluationRecord]]:
    root = settings.processed_dir
    return (
        load_records_jsonl(root / "train.jsonl"),
        load_records_jsonl(root / "valid.jsonl"),
        load_records_jsonl(root / "test.jsonl"),
    )


def _require_prepared_split_integrity(processed_dir: Path) -> None:
    """Fail before model work when prepared dataset evidence no longer matches."""

    try:
        require_valid_manifest(processed_dir)
    except DatasetIntegrityError as error:
        raise StudyCommandError(
            "prepared dataset integrity check failed; rerun `make prepare-data`:\n"
            f"{error}"
        ) from error


def _require_current_flight_preparation(settings: ProjectSettings) -> None:
    """Fail promotion-capable work when plane-preparation evidence has drifted."""

    try:
        verify_flight_manifest(settings)
    except (OfflineAssetError, OSError, ValueError) as error:
        raise StudyCommandError(
            "flight preparation evidence is missing, stale, or mismatched; "
            "rerun `make prepare-flight` while online before training or evaluation:\n"
            f"{error}"
        ) from error


def _require_trained_adapter(
    adapter_dir: Path,
    *,
    config_path: Path,
    train_command: str,
) -> ValidatedTrainingSnapshot:
    """Reject adapter bytes without matching successful-training evidence."""

    try:
        return require_valid_training_snapshot(
            adapter_dir,
            config_path=config_path,
        )
    except (TrainingManifestError, OSError, ValueError) as error:
        raise StudyCommandError(
            "LoRA adapter training evidence is missing, stale, or mismatched; "
            f"rerun `{train_command}` before evaluation:\n{error}"
        ) from error


def _adapter_evidence_present(adapter_dir: Path) -> bool:
    evidence_paths = (
        adapter_dir / "adapters.safetensors",
        adapter_dir / TRAINING_MANIFEST_NAME,
    )
    return any(path.exists() or path.is_symlink() for path in evidence_paths)


def _support_evidence_paths(name: str) -> tuple[Path, Path]:
    output_dir = PROJECT_ROOT / "artifacts" / "evaluation"
    return (
        output_dir / f"{name}-predictions.jsonl",
        output_dir / f"{name}-report.json",
    )


def _invalidate_support_evidence(name: str) -> tuple[Path, Path]:
    """Remove same-name evidence before a new persisted attempt starts."""

    paths = _support_evidence_paths(name)
    for path in paths:
        path.unlink(missing_ok=True)
    return paths


def _invalidate_support_promotion() -> None:
    """A new evaluation attempt makes any prior promotion decision stale."""

    (PROJECT_ROOT / "artifacts" / "evaluation" / "promotion.json").unlink(
        missing_ok=True
    )


def _write_evaluation(
    *,
    name: str,
    records: Sequence[EvaluationRecord],
    predictions: Sequence[Prediction],
    report: EvaluationReport,
) -> tuple[Path, Path]:
    prediction_path, report_path = _invalidate_support_evidence(name)
    try:
        write_predictions_jsonl(prediction_path, predictions)
        write_report_json(report_path, report)
    except BaseException:
        prediction_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    return prediction_path, report_path


def _write_support_promotion(
    path: Path,
    assessment: PromotionAssessment,
    training_snapshot: ValidatedTrainingSnapshot,
    evaluation_sessions: Sequence[EvaluationSession],
) -> None:
    """Commit only while every inference-wide and adapter snapshot stays current."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(assessment.model_dump_json(indent=2) + "\n", encoding="utf-8")
    try:
        for evaluation_session in evaluation_sessions:
            recheck_evaluation_session(evaluation_session)
        recheck_training_snapshot(training_snapshot)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_capstone_decision(
    path: Path,
    payload: dict[str, Any],
    decision_session: EvaluationSession,
    training_snapshot: ValidatedTrainingSnapshot | None,
    evaluation_sessions: Sequence[EvaluationSession] = (),
) -> None:
    """Commit only while capstone inference and adapter snapshots remain current."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        recheck_evaluation_session(decision_session)
        for evaluation_session in evaluation_sessions:
            recheck_evaluation_session(evaluation_session)
        if training_snapshot is not None:
            recheck_training_snapshot(training_snapshot)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _track_report(
    settings: ProjectSettings,
    *,
    name: str,
    role: str,
    records: Sequence[EvaluationRecord],
    report: EvaluationReport,
    evaluation_session: EvaluationSession,
    training_snapshot: ValidatedTrainingSnapshot | None = None,
) -> str:
    from .tracking import log_evaluation

    return log_evaluation(
        settings,
        run_name=name,
        role=role,
        method=name,
        metrics=report.flat_metrics(),
        report=report.model_dump(mode="json"),
        records=(record.model_dump(mode="json") for record in records),
        manifest_path=settings.processed_dir / "manifest.json",
        prediction_path=(
            PROJECT_ROOT / "artifacts" / "evaluation" / f"{name}-predictions.jsonl"
        ),
        model_based=name not in {"majority", "keyword-rule"},
        evaluation_session=evaluation_session,
        training_snapshot=training_snapshot,
    )


def _score_predictions(
    *,
    name: str,
    records: Sequence[EvaluationRecord],
    predictions: Sequence[Prediction],
    supported_intents: Sequence[str],
    evaluation_session: EvaluationSession,
    inference_config: InferenceConfig,
    training_snapshot: ValidatedTrainingSnapshot | None = None,
) -> EvaluationReport:
    _invalidate_support_evidence(name)
    report = Evaluator(supported_intents=supported_intents).evaluate(
        records,
        predictions,
        evaluation_session=evaluation_session,
        inference_config=inference_config,
    )
    if training_snapshot is not None:
        report = EvaluationReport.model_validate(
            report.model_dump(mode="python")
            | {
                "training_manifest_sha256": training_snapshot.manifest_sha256,
                "training_execution_contract_sha256": (
                    training_snapshot.manifest.execution_contract_sha256
                ),
            }
        )
    prediction_path, report_path = _write_evaluation(
        name=name,
        records=records,
        predictions=predictions,
        report=report,
    )
    try:
        recheck_evaluation_session(evaluation_session)
        if training_snapshot is not None:
            recheck_training_snapshot(training_snapshot)
    except BaseException:
        prediction_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    return report


def _baseline_reports(
    settings: ProjectSettings,
    *,
    records: Sequence[EvaluationRecord] | None = None,
    track: bool = False,
) -> dict[str, EvaluationReport]:
    baseline_types = (
        ("majority", MajorityBaseline),
        ("keyword-rule", KeywordRuleBaseline),
    )
    _invalidate_support_promotion()
    for name, _baseline_type in baseline_types:
        _invalidate_support_evidence(name)
    train, _, test = _load_splits(settings)
    evaluation_records = list(records or test)
    supported = tuple(sorted({record.target.intent for record in train}))
    reports: dict[str, EvaluationReport] = {}
    for name, baseline_type in baseline_types:
        evaluation_session = start_evaluation_session()
        method = baseline_type.fit(train)
        predictions = method.predict_many(evaluation_records)
        report = _score_predictions(
            name=name,
            records=evaluation_records,
            predictions=predictions,
            supported_intents=supported,
            evaluation_session=evaluation_session,
            inference_config=DeterministicInferenceConfig(method=name),
        )
        reports[name] = report
        if track:
            run_id = _track_report(
                settings,
                name=name,
                role="baseline",
                records=evaluation_records,
                report=report,
                evaluation_session=evaluation_session,
            )
            print(f"  local MLflow run: {run_id}")
        print(
            f"{name}: macro-F1={report.classification.macro_f1:.3f}, "
            f"schema={report.output_quality.json_schema_validity_rate:.3f}, "
            f"policy={report.output_quality.response_policy_compliance_rate:.3f}"
        )
    return reports


def _intent_categories(
    train: Sequence[EvaluationRecord],
) -> tuple[list[str], dict[str, str]]:
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    for record in train:
        categories[record.target.intent][record.target.category] += 1
    intents = sorted(categories)
    mapping = {
        intent: min(
            counts,
            key=lambda category: (-counts[category], category),
        )
        for intent, counts in categories.items()
    }
    return intents, mapping


def _few_shot_examples(
    train: Sequence[EvaluationRecord], count: int = 3
) -> list[tuple[str, dict[str, Any]]]:
    first_by_intent: dict[str, EvaluationRecord] = {}
    for record in train:
        first_by_intent.setdefault(record.target.intent, record)
    choices = [first_by_intent[key] for key in sorted(first_by_intent)]
    if len(choices) > count:
        positions = [
            round(index * (len(choices) - 1) / (count - 1)) for index in range(count)
        ]
        choices = [choices[position] for position in positions]
    return [
        (record.input_text, record.target.model_dump(mode="json")) for record in choices
    ]


def _model_predictions(
    predictor: LocalMLXPredictor,
    records: Sequence[EvaluationRecord],
    *,
    strategy: PromptStrategy,
    train: Sequence[EvaluationRecord],
    inference_config: LocalMLXInferenceConfig,
) -> tuple[Prediction, ...]:
    intents, categories = _intent_categories(train)
    shots = _few_shot_examples(train) if strategy == "few_shot" else None
    if inference_config.prompt_recipe != strategy:
        raise ValueError(
            "inference prompt recipe does not match the requested strategy"
        )
    if inference_config.few_shot_examples != len(shots or ()):
        raise ValueError("inference few-shot count does not match the rendered prompt")
    predictions: list[Prediction] = []
    for index, record in enumerate(records, start=1):
        messages = build_messages(
            record.input_text,
            strategy=strategy,
            allowed_intents=intents,
            category_by_intent=categories,
            few_shot=shots,
        )
        generated = predictor.generate(
            messages,
            max_tokens=inference_config.generation.max_tokens,
        )
        predictions.append(
            Prediction(
                example_id=record.example_id,
                raw_text=generated.text,
                latency_ms=generated.latency_ms,
                output_tokens=generated.output_tokens,
                peak_memory_mb=generated.peak_memory_mb,
            )
        )
        if index == len(records) or index % 10 == 0:
            print(f"  {strategy}: {index}/{len(records)}")
    return tuple(predictions)


def _balanced_subset(
    records: Sequence[EvaluationRecord], limit: int | None
) -> list[EvaluationRecord]:
    if limit is None or limit >= len(records):
        return list(records)
    if limit < 1:
        raise StudyCommandError("--limit must be positive")
    by_intent: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        by_intent[record.target.intent].append(record)
    selected: list[EvaluationRecord] = []
    round_number = 0
    while len(selected) < limit:
        added = False
        for intent in sorted(by_intent):
            values = by_intent[intent]
            if round_number < len(values):
                selected.append(values[round_number])
                added = True
                if len(selected) == limit:
                    return selected
        if not added:
            break
        round_number += 1
    return selected


def _run_local_probe(settings: ProjectSettings) -> None:
    train, _, _ = _load_splits(settings)
    intents, categories = _intent_categories(train)
    predictor = LocalMLXPredictor(settings.model_dir)
    messages = build_messages(
        "I forgot my password and need help signing in.",
        strategy="strong",
        allowed_intents=intents,
        category_by_intent=categories,
    )
    generated = predictor.generate(messages, max_tokens=96)
    if not generated.text.strip():
        raise StudyCommandError("the local model returned an empty readiness probe")
    print(
        "Local MLX inference passed: "
        f"{generated.latency_ms:.0f} ms, {generated.output_tokens} output tokens, "
        f"{generated.peak_memory_mb:.0f} MB process peak"
    )


def _check_notebook_runtime() -> None:
    required = {"jupyterlab": "4.6.2", "ipykernel": "7.3.0"}
    problems = []
    for package, expected in required.items():
        if importlib.util.find_spec(package) is None:
            problems.append(f"{package} is missing")
            continue
        actual = importlib.metadata.version(package)
        if actual != expected:
            problems.append(f"{package} is {actual}, expected {expected}")
    if problems:
        raise StudyCommandError(
            "notebook runtime is incomplete: " + "; ".join(problems)
        )
    print("Locked JupyterLab notebook runtime passed.")


def _generate_capstone() -> None:
    output_dir = PROJECT_ROOT / "data" / "processed" / "capstone-readiness-v1"
    manifest = generate_capstone_dataset(output_dir)
    counts = {
        artifact.split.value: artifact.record_count for artifact in manifest.artifacts
    }
    expected = {"train": 400, "validation": 100, "test": 150}
    if counts != expected or not manifest.frozen_test:
        raise StudyCommandError(
            "capstone generation violated its frozen split contract"
        )
    training_manifest = render_capstone_mlx_dataset(
        output_dir,
        PROJECT_ROOT / "data" / "processed" / "capstone-mlx-v1",
    )
    test_records = load_capstone_records(output_dir / "test.jsonl")
    policy_session = start_evaluation_session()
    policy_predictions = deterministic_capstone_predictions(test_records)
    policy_report = _score_capstone_predictions(
        name="deterministic-policy",
        records=test_records,
        predictions=policy_predictions,
        evaluation_session=policy_session,
        inference_config=DeterministicInferenceConfig(method="deterministic-policy"),
    )
    hybrid_session = start_evaluation_session()
    hybrid_predictions = tuple(
        CapstonePrediction(
            example_id=record.example_id,
            raw_text=compact_review(
                build_hybrid_review(record.manifest).deterministic_review
            ).model_dump_json(),
            latency_ms=0.0,
            output_tokens=0,
            peak_memory_mb=0.0,
        )
        for record in test_records
    )
    _score_capstone_predictions(
        name="hybrid-policy-text",
        records=test_records,
        predictions=hybrid_predictions,
        evaluation_session=hybrid_session,
        inference_config=DeterministicInferenceConfig(method="hybrid-policy-text"),
    )
    if policy_report.aggregate.exact_review_rate != 1.0:
        raise StudyCommandError("deterministic capstone policy failed its own ceiling")
    print(
        "Generated policy-derived capstone: "
        f"train=400, validation=100, frozen-test=150; {manifest.dataset_sha256}"
    )
    print(
        "Rendered compact MLX capstone data: "
        f"{training_manifest.dataset_fingerprint}"
    )
    print("Policy and deterministic-hybrid frozen-test ceiling: exact=1.000")


def _capstone_evidence_paths(name: str) -> tuple[Path, Path]:
    output = PROJECT_ROOT / "artifacts" / "capstone-evaluation"
    return (
        output / f"{name}-predictions.jsonl",
        output / f"{name}-report.json",
    )


def _invalidate_capstone_evidence(name: str) -> tuple[Path, Path]:
    """Remove same-name capstone evidence before a persisted attempt starts."""

    paths = _capstone_evidence_paths(name)
    for path in paths:
        path.unlink(missing_ok=True)
    return paths


def _write_capstone_evaluation(
    name: str,
    records: Sequence[CapstoneRecord],
    predictions: Sequence[CapstonePrediction],
    report: CapstoneEvaluationReport,
) -> tuple[Path, Path]:
    output = PROJECT_ROOT / "artifacts" / "capstone-evaluation"
    output.mkdir(parents=True, exist_ok=True)
    prediction_path, report_path = _invalidate_capstone_evidence(name)
    try:
        prediction_path.write_text(
            "".join(prediction.model_dump_json() + "\n" for prediction in predictions),
            encoding="utf-8",
        )
        report_path.write_text(
            report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        prediction_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    return prediction_path, report_path


def _score_capstone_predictions(
    *,
    name: str,
    records: Sequence[CapstoneRecord],
    predictions: Sequence[CapstonePrediction],
    evaluation_session: EvaluationSession,
    inference_config: InferenceConfig,
    training_snapshot: ValidatedTrainingSnapshot | None = None,
) -> CapstoneEvaluationReport:
    _invalidate_capstone_evidence(name)
    report = evaluate_capstone_predictions(
        records,
        predictions,
        evaluation_session=evaluation_session,
        inference_config=inference_config,
    )
    if training_snapshot is not None:
        report = CapstoneEvaluationReport.model_validate(
            report.model_dump(mode="python")
            | {
                "training_manifest_sha256": training_snapshot.manifest_sha256,
                "training_execution_contract_sha256": (
                    training_snapshot.manifest.execution_contract_sha256
                ),
            }
        )
    prediction_path, report_path = _write_capstone_evaluation(
        name,
        records,
        predictions,
        report,
    )
    try:
        recheck_evaluation_session(evaluation_session)
        if training_snapshot is not None:
            recheck_training_snapshot(training_snapshot)
    except BaseException:
        prediction_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    return report


def _capstone_inference_comparability_reasons(
    *,
    reports: dict[str, CapstoneEvaluationReport],
    lora_report: CapstoneEvaluationReport,
    decision_session: EvaluationSession,
    training_snapshot: ValidatedTrainingSnapshot,
) -> tuple[str, ...]:
    """Return fair-comparison gaps; reject stale or contradictory lineage."""

    current_model = decision_session.base_model_execution_contract
    if current_model is None:
        raise StudyCommandError(
            "capstone model promotion requires a model-aware decision session"
        )
    manifest = training_snapshot.manifest
    if (
        manifest.model_path != current_model.model_path
        or manifest.model_revision != current_model.model_revision
        or manifest.model_files != current_model.model_files
    ):
        raise StudyCommandError(
            "capstone training evidence and current base-model files do not match"
        )
    lora_inference = lora_report.inference_config
    if not isinstance(lora_inference, LocalMLXInferenceConfig):
        return ("the LoRA report lacks local model inference evidence",)
    if lora_inference.base_model != current_model:
        raise StudyCommandError(
            "capstone LoRA report does not match the current base-model files"
        )
    if lora_inference.adapter_manifest_sha256 != training_snapshot.manifest_sha256:
        raise StudyCommandError(
            "capstone LoRA inference does not match its training manifest"
        )

    reasons: list[str] = []
    baseline_configs: dict[str, LocalMLXInferenceConfig] = {}
    for name in ("basic", "strong", "few-shot"):
        config = reports[name].inference_config
        if not isinstance(config, LocalMLXInferenceConfig):
            reasons.append(f"{name} lacks local model inference evidence")
            continue
        if config.method != name:
            reasons.append(f"{name} report carries a different inference method")
        if config.adapter_manifest_sha256 is not None:
            raise StudyCommandError(
                f"capstone untouched-model baseline {name} carries adapter lineage"
            )
        if config.base_model != current_model:
            raise StudyCommandError(
                f"capstone baseline {name} does not match current base-model files"
            )
        baseline_configs[name] = config

    if baseline_configs and any(
        config.generation != lora_inference.generation
        for config in baseline_configs.values()
    ):
        reasons.append(
            "capstone untouched-model baselines and LoRA used different "
            "generation settings"
        )
    basic = baseline_configs.get("basic")
    if (
        basic is None
        or basic.prompt_recipe != lora_inference.prompt_recipe
        or basic.few_shot_examples != lora_inference.few_shot_examples
    ):
        reasons.append(
            "capstone LoRA lacks an untouched-model control with the same prompt "
            "recipe and few-shot count"
        )
    return tuple(reasons)


def _capstone_shots(
    train: Sequence[CapstoneRecord],
) -> list[tuple[str, str]]:
    selected: list[CapstoneRecord] = []
    for status in (
        ReadinessStatus.READY,
        ReadinessStatus.NOT_READY,
        ReadinessStatus.REVIEW_REQUIRED,
    ):
        match = next(
            (record for record in train if record.expected_output.status is status),
            None,
        )
        if match is not None:
            selected.append(match)
    return [
        (
            json.dumps(record.manifest, sort_keys=True, separators=(",", ":")),
            compact_expected(record).model_dump_json(),
        )
        for record in selected
    ]


def _capstone_messages(
    record: CapstoneRecord,
    *,
    strategy: str,
    shots: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    system = CAPSTONE_SYSTEM_PROMPT
    if strategy != "basic":
        rules = [
            {
                "name": rule.name,
                "kind": rule.kind.value,
                "failure_severity": rule.failure_severity.value,
                "remediation_id": rule.remediation_id,
            }
            for rule in rule_catalog()
        ]
        system += (
            " Valid statuses: ready, not_ready, review_required. Valid non-pass "
            "results: fail, review. Use only these rule definitions: "
            + json.dumps(rules, separators=(",", ":"), sort_keys=True)
        )
    messages = [{"role": "system", "content": system}]
    if strategy == "few-shot":
        for manifest, target in shots:
            messages.extend(
                [
                    {"role": "user", "content": manifest},
                    {"role": "assistant", "content": target},
                ]
            )
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                record.manifest,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    )
    return messages


def _capstone_model_predictions(
    predictor: LocalMLXPredictor,
    records: Sequence[CapstoneRecord],
    *,
    strategy: str,
    train: Sequence[CapstoneRecord],
    inference_config: LocalMLXInferenceConfig,
    display_name: str | None = None,
) -> tuple[CapstonePrediction, ...]:
    shots = _capstone_shots(train) if strategy == "few-shot" else []
    if inference_config.prompt_recipe != strategy:
        raise ValueError(
            "inference prompt recipe does not match the requested strategy"
        )
    if inference_config.few_shot_examples != len(shots):
        raise ValueError("inference few-shot count does not match the rendered prompt")
    predictions: list[CapstonePrediction] = []
    for index, record in enumerate(records, start=1):
        generated = predictor.generate(
            _capstone_messages(record, strategy=strategy, shots=shots),
            max_tokens=inference_config.generation.max_tokens,
        )
        predictions.append(
            CapstonePrediction(
                example_id=record.example_id,
                raw_text=generated.text,
                latency_ms=generated.latency_ms,
                output_tokens=generated.output_tokens,
                peak_memory_mb=generated.peak_memory_mb,
            )
        )
        if index == len(records) or index % 10 == 0:
            print(f"  capstone {display_name or strategy}: {index}/{len(records)}")
    return tuple(predictions)


def _capstone_hybrid_predictions(
    predictor: LocalMLXPredictor,
    records: Sequence[CapstoneRecord],
    *,
    inference_config: LocalMLXInferenceConfig,
) -> tuple[tuple[CapstonePrediction, ...], list[dict[str, Any]]]:
    if inference_config.prompt_recipe != "hybrid_explanation":
        raise ValueError("hybrid inference must use the hybrid explanation recipe")
    if inference_config.few_shot_examples != 0:
        raise ValueError("hybrid explanation inference does not use few-shot examples")
    cache: dict[tuple[str, ...], str] = {}
    evidence: list[dict[str, Any]] = []
    predictions: list[CapstonePrediction] = []
    for index, record in enumerate(records, start=1):
        generated_latency = 0.0
        generated_tokens = 0
        generated_peak = 0.0

        def renderer(check) -> str:
            nonlocal generated_latency, generated_tokens, generated_peak
            key = (
                check.name,
                check.result.value,
                check.severity.value,
                check.evidence,
                check.remediation_text or "",
            )
            if key not in cache:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the supplied deterministic check as one concise "
                            "sentence. Preserve every fact and never change the result."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "name": check.name,
                                "result": check.result.value,
                                "severity": check.severity.value,
                                "evidence": check.evidence,
                                "remediation": check.remediation_text,
                            },
                            separators=(",", ":"),
                        ),
                    },
                ]
                generated = predictor.generate(
                    messages,
                    max_tokens=inference_config.generation.max_tokens,
                )
                cache[key] = generated.text.strip()
                generated_latency += generated.latency_ms
                generated_tokens += generated.output_tokens
                generated_peak = max(generated_peak, generated.peak_memory_mb)
            return cache[key]

        hybrid = build_hybrid_review(
            record.manifest,
            renderer=renderer,
            renderer_name="local-qwen-explanation",
        )
        evidence.append(
            {
                "example_id": record.example_id,
                "explanations": [
                    explanation.model_dump(mode="json")
                    for explanation in hybrid.explanations
                ],
            }
        )
        predictions.append(
            CapstonePrediction(
                example_id=record.example_id,
                raw_text=compact_review(hybrid.deterministic_review).model_dump_json(),
                latency_ms=generated_latency,
                output_tokens=generated_tokens,
                peak_memory_mb=generated_peak,
            )
        )
        if index == len(records) or index % 10 == 0:
            print(f"  capstone hybrid: {index}/{len(records)}")
    return tuple(predictions), evidence


def _cmd_prepare_flight(_args: argparse.Namespace, settings: ProjectSettings) -> None:
    from .acquisition import acquire_bitext, acquire_model

    _invalidate_support_promotion()
    for name in ("majority", "keyword-rule"):
        _invalidate_support_evidence(name)
    print("Acquiring and verifying pinned public study assets...")
    acquire_bitext(settings)
    acquire_model(settings)
    _prepare_data(settings)
    _generate_capstone()
    _baseline_reports(settings)
    _check_notebook_runtime()

    enable_offline_environment()
    with deny_network():
        prove_socket_denial()
        _run_local_probe(settings)
        from .tracking import tracking_smoke

        run_id = tracking_smoke(settings)
        print(f"Local MLflow write passed: {run_id}")

    print("Running one real MLX-LM LoRA iteration...")
    evidence = run_lora(
        iterations=1,
        adapter_path=settings.preflight_adapter_dir,
        log_name="preflight-smoke",
    )
    if not (settings.preflight_adapter_dir / "adapters.safetensors").is_file():
        raise StudyCommandError("MLX-LM did not write the preflight adapter")
    print(
        f"MLX-LM training passed: return={evidence.return_code}, "
        f"peak={evidence.peak_memory_gb or 0.0:.2f} GB"
    )
    manifest_path = write_flight_manifest(settings)
    print(f"Flight manifest written: {manifest_path.relative_to(PROJECT_ROOT)}")
    print("Preparation complete. Run `make flight-check` before departure.")


def _cmd_flight_check(_args: argparse.Namespace, settings: ProjectSettings) -> None:
    machine = apple_silicon_status()
    if not machine.ready:
        raise StudyCommandError(f"MLX requires Apple silicon; found {machine.detail}")
    checks = require_assets(settings)
    verify_flight_manifest(settings)
    prove_socket_denial()
    leakage = check_split_files(settings.processed_dir)
    assert_no_leakage(leakage)
    _check_notebook_runtime()
    _generate_capstone()
    _run_local_probe(settings)
    from .tracking import tracking_smoke

    run_id = tracking_smoke(settings)
    verify_flight_manifest(settings)
    print(f"Verified {len(checks)} local asset groups and the locked manifest.")
    print(f"Local MLflow write passed: {run_id}")
    print("READY FOR OFFLINE STUDY")


def _cmd_smoke(_args: argparse.Namespace, settings: ProjectSettings) -> None:
    _invalidate_support_promotion()
    for name in ("majority", "keyword-rule"):
        _invalidate_support_evidence(name)
    require_assets(settings)
    leakage = check_split_files(settings.processed_dir)
    assert_no_leakage(leakage)
    _baseline_reports(settings)
    _generate_capstone()
    print("Offline deterministic study smoke passed.")


def _cmd_prepare_data(_args: argparse.Namespace, settings: ProjectSettings) -> None:
    _prepare_data(settings)


def _cmd_baselines(args: argparse.Namespace, settings: ProjectSettings) -> None:
    _invalidate_support_promotion()
    for name in ("majority", "keyword-rule"):
        _invalidate_support_evidence(name)
    _require_prepared_split_integrity(settings.processed_dir)
    _baseline_reports(settings, track=args.track)


def _cmd_train(args: argparse.Namespace, settings: ProjectSettings) -> None:
    _require_prepared_split_integrity(settings.processed_dir)
    _require_current_flight_preparation(settings)
    require_assets(settings)
    smoke_adapter = (
        settings.adapter_dir.with_name(f"{settings.adapter_dir.name}-smoke")
        if args.iterations is not None
        else None
    )
    evidence = run_lora(
        iterations=args.iterations,
        adapter_path=smoke_adapter,
        log_name="support-smoke" if smoke_adapter is not None else "latest",
    )
    print(evidence.model_dump_json(indent=2))


def _cmd_evaluate(args: argparse.Namespace, settings: ProjectSettings) -> None:
    if args.max_tokens < 1:
        raise StudyCommandError("--max-tokens must be positive")
    requested = {"basic", "strong", "few-shot", "lora"}
    if args.methods != "all":
        requested = set(args.methods.split(","))
        unknown = requested.difference({"basic", "strong", "few-shot", "lora"})
        if unknown:
            raise StudyCommandError("unknown evaluation methods: " + ", ".join(unknown))

    promotion_path = PROJECT_ROOT / "artifacts" / "evaluation" / "promotion.json"
    _invalidate_support_promotion()
    for name in ("majority", "keyword-rule", *sorted(requested - {"lora"})):
        _invalidate_support_evidence(name)
    if "lora" in requested:
        _invalidate_support_evidence("lora-change")

    _require_prepared_split_integrity(settings.processed_dir)
    _require_current_flight_preparation(settings)
    require_assets(settings)

    lora_snapshot: ValidatedTrainingSnapshot | None = None
    if "lora" in requested:
        if _adapter_evidence_present(settings.adapter_dir):
            lora_snapshot = _require_trained_adapter(
                settings.adapter_dir,
                config_path=PROJECT_ROOT / "configs" / "training" / "lora.yaml",
                train_command="make train",
            )
        else:
            print("LoRA adapter is absent; run `make train` before LoRA evaluation.")

    train, _, test = _load_splits(settings)
    records = _balanced_subset(test, args.limit)
    supported, _ = _intent_categories(train)
    deterministic = _baseline_reports(settings, records=records, track=args.track)

    reports: dict[str, EvaluationReport] = dict(deterministic)
    base_session: EvaluationSession | None = None
    base_methods = [
        name for name in ("basic", "strong", "few-shot") if name in requested
    ]
    if base_methods:
        for name in base_methods:
            _invalidate_support_evidence(name)
        base_session = start_evaluation_session(settings)
        predictor = LocalMLXPredictor(settings.model_dir)
        for name in base_methods:
            strategy: PromptStrategy = "few_shot" if name == "few-shot" else name  # type: ignore[assignment]
            inference_config = build_local_mlx_inference_config(
                base_session,
                method=name,
                prompt_recipe=strategy,
                max_tokens=args.max_tokens,
                few_shot_examples=(
                    len(_few_shot_examples(train)) if strategy == "few_shot" else 0
                ),
            )
            predictions = _model_predictions(
                predictor,
                records,
                strategy=strategy,
                train=train,
                inference_config=inference_config,
            )
            report = _score_predictions(
                name=name,
                records=records,
                predictions=predictions,
                supported_intents=supported,
                evaluation_session=base_session,
                inference_config=inference_config,
            )
            reports[name] = report
            if args.track:
                run_id = _track_report(
                    settings,
                    name=name,
                    role="baseline",
                    records=records,
                    report=report,
                    evaluation_session=base_session,
                )
                print(f"  local MLflow run: {run_id}")
            print(
                f"{name}: macro-F1={report.classification.macro_f1:.3f}; "
                f"schema={report.output_quality.json_schema_validity_rate:.3f}"
            )

    lora_report: EvaluationReport | None = None
    lora_session: EvaluationSession | None = None
    lora_context = (
        shared_adapter_lock(settings.adapter_dir)
        if lora_snapshot is not None
        else nullcontext()
    )
    with lora_context:
        if "lora" in requested and lora_snapshot is not None:
            lora_session = start_evaluation_session(settings)
            lora_inference_config = build_local_mlx_inference_config(
                lora_session,
                method="lora-change",
                prompt_recipe="strong",
                max_tokens=args.max_tokens,
                adapter_manifest_sha256=lora_snapshot.manifest_sha256,
            )
            recheck_training_snapshot(lora_snapshot)
            predictor = LocalMLXPredictor(
                settings.model_dir,
                adapter_path=settings.adapter_dir,
            )
            predictions = _model_predictions(
                predictor,
                records,
                strategy="strong",
                train=train,
                inference_config=lora_inference_config,
            )
            recheck_training_snapshot(lora_snapshot)
            lora_report = _score_predictions(
                name="lora-change",
                records=records,
                predictions=predictions,
                supported_intents=supported,
                evaluation_session=lora_session,
                inference_config=lora_inference_config,
                training_snapshot=lora_snapshot,
            )
            reports["lora-change"] = lora_report
            if args.track:
                run_id = _track_report(
                    settings,
                    name="lora-change",
                    role="change",
                    records=records,
                    report=lora_report,
                    evaluation_session=lora_session,
                    training_snapshot=lora_snapshot,
                )
                print(f"  local MLflow run: {run_id}")
            print(format_error_analysis(lora_report))

        complete_promotion_evidence = (
            lora_report is not None
            and args.limit is None
            and requested == {"basic", "strong", "few-shot", "lora"}
        )
        if (
            complete_promotion_evidence
            and lora_report is not None
            and lora_snapshot is not None
        ):
            if base_session is None or lora_session is None:
                raise StudyCommandError(
                    "complete promotion evidence requires retained model sessions"
                )
            recheck_evaluation_session(base_session)
            recheck_evaluation_session(lora_session)
            recheck_training_snapshot(lora_snapshot)
            baselines = [
                BaselineEvaluation(
                    name=name,
                    report=report,
                    meaningful=name != "majority",
                )
                for name, report in reports.items()
                if name != "lora-change"
            ]
            promotion_session = start_evaluation_session(settings)
            assessment = decide_lora_promotion(
                change_name="bitext-lora-v1",
                evaluation_session=promotion_session,
                training_snapshot=lora_snapshot,
                change_report=lora_report,
                baselines=baselines,
            )
            _write_support_promotion(
                promotion_path,
                assessment,
                lora_snapshot,
                (base_session, lora_session, promotion_session),
            )
            print(f"Decision: {assessment.decision.value}")
        elif lora_report is not None:
            print(
                "Decision: inconclusive (partial or debug evaluation is report-only; "
                "run the full frozen set with all methods for promotion evidence)"
            )


def _cmd_capstone(_args: argparse.Namespace, _settings: ProjectSettings) -> None:
    _generate_capstone()


def _cmd_capstone_train(args: argparse.Namespace, settings: ProjectSettings) -> None:
    _require_current_flight_preparation(settings)
    require_assets(settings)
    _generate_capstone()
    smoke_adapter = (
        settings.capstone_adapter_dir.with_name(
            f"{settings.capstone_adapter_dir.name}-smoke"
        )
        if args.iterations is not None
        else None
    )
    evidence = run_lora(
        iterations=args.iterations,
        config_path=PROJECT_ROOT / "configs" / "training" / "capstone-lora.yaml",
        adapter_path=smoke_adapter,
        log_name="capstone-smoke" if smoke_adapter is not None else "capstone-latest",
    )
    print(evidence.model_dump_json(indent=2))


def _cmd_capstone_evaluate(args: argparse.Namespace, settings: ProjectSettings) -> None:
    if args.limit is not None and args.limit < 1:
        raise StudyCommandError("--limit must be positive")
    if args.max_tokens < 1:
        raise StudyCommandError("--max-tokens must be positive")
    allowed = {"policy", "basic", "strong", "few-shot", "lora", "hybrid"}
    requested = allowed if args.methods == "all" else set(args.methods.split(","))
    unknown = requested - allowed
    if unknown:
        raise StudyCommandError(
            "unknown capstone methods: " + ", ".join(sorted(unknown))
        )

    decision_path = PROJECT_ROOT / "artifacts" / "capstone-evaluation" / "decision.json"
    artifact_names = {
        "policy": "deterministic-policy",
        "basic": "basic",
        "strong": "strong",
        "few-shot": "few-shot",
        "lora": "capstone-lora-change",
        "hybrid": "hybrid",
    }
    for method in sorted(requested):
        _invalidate_capstone_evidence(artifact_names[method])
    hybrid_evidence_path = (
        PROJECT_ROOT / "artifacts" / "capstone-evaluation" / "hybrid-explanations.json"
    )
    if "hybrid" in requested:
        hybrid_evidence_path.unlink(missing_ok=True)
    decision_path.unlink(missing_ok=True)

    _require_current_flight_preparation(settings)
    require_assets(settings)
    _generate_capstone()
    lora_snapshot: ValidatedTrainingSnapshot | None = None
    if "lora" in requested:
        if _adapter_evidence_present(settings.capstone_adapter_dir):
            lora_snapshot = _require_trained_adapter(
                settings.capstone_adapter_dir,
                config_path=(
                    PROJECT_ROOT / "configs" / "training" / "capstone-lora.yaml"
                ),
                train_command="make capstone-train",
            )
        else:
            print(
                "Capstone LoRA adapter is absent; " "run `make capstone-train` first."
            )

    source = PROJECT_ROOT / "data" / "processed" / "capstone-readiness-v1"
    train = load_capstone_records(source / "train.jsonl")
    all_test = load_capstone_records(source / "test.jsonl")
    records = all_test if args.limit is None else all_test[: args.limit]

    reports: dict[str, CapstoneEvaluationReport] = {}

    def score(
        name: str,
        predictions: Sequence[CapstonePrediction],
        *,
        evaluation_session: EvaluationSession,
        inference_config: InferenceConfig,
        training_snapshot: ValidatedTrainingSnapshot | None = None,
    ) -> CapstoneEvaluationReport:
        report = _score_capstone_predictions(
            name=name,
            records=records,
            predictions=predictions,
            evaluation_session=evaluation_session,
            inference_config=inference_config,
            training_snapshot=training_snapshot,
        )
        reports[name] = report
        print(
            f"{name}: exact={report.aggregate.exact_review_rate:.3f}, "
            f"status={report.aggregate.status_accuracy:.3f}, "
            f"checks={report.aggregate.check_result_accuracy:.3f}, "
            f"schema={report.aggregate.schema_validity_rate:.3f}"
        )
        return report

    if "policy" in requested:
        policy_session = start_evaluation_session()
        score(
            "deterministic-policy",
            deterministic_capstone_predictions(records),
            evaluation_session=policy_session,
            inference_config=DeterministicInferenceConfig(
                method="deterministic-policy"
            ),
        )

    needs_base = bool(requested & {"basic", "strong", "few-shot", "hybrid"})
    base_session = start_evaluation_session(settings) if needs_base else None
    base_predictor = LocalMLXPredictor(settings.model_dir) if needs_base else None
    for method in ("basic", "strong", "few-shot"):
        if (
            method in requested
            and base_predictor is not None
            and base_session is not None
        ):
            inference_config = build_local_mlx_inference_config(
                base_session,
                method=method,
                prompt_recipe=method,
                max_tokens=args.max_tokens,
                few_shot_examples=(
                    len(_capstone_shots(train)) if method == "few-shot" else 0
                ),
            )
            score(
                method,
                _capstone_model_predictions(
                    base_predictor,
                    records,
                    strategy=method,
                    train=train,
                    inference_config=inference_config,
                ),
                evaluation_session=base_session,
                inference_config=inference_config,
            )

    if (
        "hybrid" in requested
        and base_predictor is not None
        and base_session is not None
    ):
        hybrid_inference_config = build_local_mlx_inference_config(
            base_session,
            method="hybrid",
            prompt_recipe="hybrid_explanation",
            max_tokens=80,
        )
        predictions, explanation_evidence = _capstone_hybrid_predictions(
            base_predictor,
            records,
            inference_config=hybrid_inference_config,
        )
        hybrid_evidence_path.write_text(
            json.dumps(explanation_evidence, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            score(
                "hybrid",
                predictions,
                evaluation_session=base_session,
                inference_config=hybrid_inference_config,
            )
        except BaseException:
            hybrid_evidence_path.unlink(missing_ok=True)
            raise
        print(
            "Hybrid decisions remain deterministic; generated explanations are "
            "report-only evidence for human review."
        )

    lora_report: CapstoneEvaluationReport | None = None
    lora_session: EvaluationSession | None = None
    lora_context = (
        shared_adapter_lock(settings.capstone_adapter_dir)
        if lora_snapshot is not None
        else nullcontext()
    )
    with lora_context:
        if "lora" in requested and lora_snapshot is not None:
            lora_session = start_evaluation_session(settings)
            lora_inference_config = build_local_mlx_inference_config(
                lora_session,
                method="capstone-lora-change",
                prompt_recipe="basic",
                max_tokens=args.max_tokens,
                adapter_manifest_sha256=lora_snapshot.manifest_sha256,
            )
            recheck_training_snapshot(lora_snapshot)
            predictor = LocalMLXPredictor(
                settings.model_dir,
                adapter_path=settings.capstone_adapter_dir,
            )
            lora_predictions = _capstone_model_predictions(
                predictor,
                records,
                strategy="basic",
                train=train,
                inference_config=lora_inference_config,
                display_name="lora",
            )
            recheck_training_snapshot(lora_snapshot)
            lora_report = score(
                "capstone-lora-change",
                lora_predictions,
                evaluation_session=lora_session,
                inference_config=lora_inference_config,
                training_snapshot=lora_snapshot,
            )

        required_baselines = {"basic", "strong", "few-shot"}
        complete = (
            lora_report is not None
            and args.limit is None
            and required_baselines <= requested
            and len(records) == 150
        )
        retained_model_sessions = tuple(
            session for session in (base_session, lora_session) if session is not None
        )
        for retained_session in retained_model_sessions:
            recheck_evaluation_session(retained_session)
        has_model_reports = any(
            isinstance(report.inference_config, LocalMLXInferenceConfig)
            for report in reports.values()
        )
        decision_session = start_evaluation_session(
            settings if has_model_reports or lora_snapshot is not None else None
        )
        decision_execution_contract_sha256 = decision_session.execution_contract_sha256
        if any(
            report.evaluation_execution_contract_sha256
            != decision_execution_contract_sha256
            for report in reports.values()
        ):
            raise StudyCommandError(
                "capstone reports do not match the current evaluation "
                "source/runtime contract"
            )
        current_decision_model = decision_session.base_model_execution_contract
        for name, report in reports.items():
            inference = report.inference_config
            if isinstance(inference, LocalMLXInferenceConfig) and (
                current_decision_model is None
                or inference.base_model != current_decision_model
            ):
                raise StudyCommandError(
                    f"capstone report {name} does not match current base-model files"
                )
        inference_reasons: tuple[str, ...] = ()
        if complete and lora_report is not None and lora_snapshot is not None:
            recheck_training_snapshot(lora_snapshot)
            if (
                lora_report.training_execution_contract_sha256
                != lora_snapshot.manifest.execution_contract_sha256
            ):
                raise StudyCommandError(
                    "capstone LoRA report does not match the current training "
                    "source/runtime contract"
                )
            inference_reasons = _capstone_inference_comparability_reasons(
                reports=reports,
                lora_report=lora_report,
                decision_session=decision_session,
                training_snapshot=lora_snapshot,
            )
            strongest_name = max(
                required_baselines,
                key=lambda name: (
                    reports[name].aggregate.exact_review_rate,
                    reports[name].aggregate.check_result_accuracy,
                    name,
                ),
            )
            strongest = reports[strongest_name]
            change_score = lora_report.aggregate.exact_review_rate
            baseline_score = strongest.aggregate.exact_review_rate
            passed_gates = (
                lora_report.aggregate.schema_validity_rate >= 0.98
                and lora_report.aggregate.extra_check_rate == 0.0
                and lora_report.aggregate.status_accuracy
                >= strongest.aggregate.status_accuracy
            )
            if inference_reasons:
                decision = "inconclusive"
                reason = "; ".join(inference_reasons)
            else:
                decision = (
                    "adopt"
                    if change_score > baseline_score and passed_gates
                    else "reject"
                )
                reason = (
                    "the compact LoRA change beat the strongest untouched-model "
                    "baseline and passed the absolute gates"
                    if decision == "adopt"
                    else "the compact LoRA change did not beat the strongest complete "
                    "untouched-model evidence and every absolute gate"
                )
            payload: dict[str, Any] = {
                "baseline": {
                    "name": strongest_name,
                    "exact_review_rate": baseline_score,
                    "inference_config": strongest.inference_config.model_dump(
                        mode="json"
                    ),
                    "evaluation_execution_contract_sha256": (
                        decision_execution_contract_sha256
                    ),
                },
                "change": {
                    "name": "capstone-lora-v1",
                    "exact_review_rate": change_score,
                    "inference_config": lora_report.inference_config.model_dump(
                        mode="json"
                    ),
                    "training_manifest_sha256": lora_snapshot.manifest_sha256,
                    "training_execution_contract_sha256": (
                        lora_snapshot.manifest.execution_contract_sha256
                    ),
                    "evaluation_execution_contract_sha256": (
                        decision_execution_contract_sha256
                    ),
                },
                "result": {
                    "inference_configs_comparable": not inference_reasons,
                    "passed_absolute_gates": passed_gates,
                    "reason": reason,
                },
                "decision": decision,
            }
        else:
            payload = {
                "baseline": None,
                "change": {
                    "name": "capstone-lora-v1",
                    "training_manifest_sha256": (
                        lora_snapshot.manifest_sha256
                        if lora_snapshot is not None
                        else None
                    ),
                    "training_execution_contract_sha256": (
                        lora_snapshot.manifest.execution_contract_sha256
                        if lora_snapshot is not None
                        else None
                    ),
                    "evaluation_execution_contract_sha256": (
                        decision_execution_contract_sha256
                    ),
                    "inference_config": (
                        lora_report.inference_config.model_dump(mode="json")
                        if lora_report is not None
                        else None
                    ),
                },
                "result": {
                    "inference_configs_comparable": False,
                    "reason": (
                        "full frozen LoRA and basic/strong/few-shot evidence "
                        "is required"
                    ),
                },
                "decision": "inconclusive",
            }
        payload["evaluated_inference_configs"] = {
            name: report.inference_config.model_dump(mode="json")
            for name, report in sorted(reports.items())
        }
        payload["architecture_boundary"] = (
            "Deterministic checks remain authoritative; a model may only supply "
            "bounded normalization or explanation wording."
        )
        _write_capstone_decision(
            decision_path,
            payload,
            decision_session,
            lora_snapshot,
            retained_model_sessions,
        )
        print(f"Capstone model-change decision: {payload['decision']}")


def _remove_generated_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _clean_artifacts_preserving_adapter_locks() -> bool:
    """Remove generated artifacts without ever unlinking stable lock inodes."""

    artifacts = PROJECT_ROOT / "artifacts"
    if not artifacts.exists():
        return False
    if artifacts.is_symlink() or not artifacts.is_dir():
        raise StudyCommandError("generated artifacts path must be a real directory")

    removed = False
    adapter_root = artifacts / "adapters"
    for child in tuple(artifacts.iterdir()):
        if child == adapter_root and child.is_dir() and not child.is_symlink():
            for adapter_child in tuple(child.iterdir()):
                persistent_lock = (
                    adapter_child.name.startswith(".")
                    and adapter_child.name.endswith(".lock")
                    and adapter_child.is_file()
                    and not adapter_child.is_symlink()
                )
                if persistent_lock:
                    continue
                _remove_generated_path(adapter_child)
                removed = True
            continue
        _remove_generated_path(child)
        removed = True
    return removed


def _clean_study_adapter_targets(settings: ProjectSettings) -> tuple[Path, ...]:
    """Return every adapter directory created by a supported CLI workflow."""

    candidates = (
        settings.adapter_dir,
        settings.adapter_dir.with_name(f"{settings.adapter_dir.name}-smoke"),
        settings.capstone_adapter_dir,
        settings.capstone_adapter_dir.with_name(
            f"{settings.capstone_adapter_dir.name}-smoke"
        ),
        settings.preflight_adapter_dir,
    )
    resolved = {candidate.resolve() for candidate in candidates}
    return tuple(sorted(resolved, key=lambda path: path.as_posix()))


def _cmd_clean_study(_args: argparse.Namespace, settings: ProjectSettings) -> None:
    with ExitStack() as adapter_locks:
        for adapter_target in _clean_study_adapter_targets(settings):
            adapter_locks.enter_context(exclusive_adapter_lock(adapter_target))
        targets = (
            settings.processed_dir,
            PROJECT_ROOT / "data" / "processed" / "capstone-readiness-v1",
            PROJECT_ROOT / "data" / "processed" / "capstone-mlx-v1",
        )
        for target in targets:
            if target.exists() or target.is_symlink():
                _remove_generated_path(target)
                print(
                    "Removed generated output: " f"{target.relative_to(PROJECT_ROOT)}"
                )
        if _clean_artifacts_preserving_adapter_locks():
            print("Removed generated output: artifacts")
        tracking_root = PROJECT_ROOT / ".aai"
        if tracking_root.exists() or tracking_root.is_symlink():
            _remove_generated_path(tracking_root)
            print("Removed generated output: .aai")
    print("Raw downloads, the local model, and the locked environment were retained.")


Command = Callable[[argparse.Namespace, ProjectSettings], None]


def _parser() -> tuple[argparse.ArgumentParser, dict[str, Command]]:
    parser = argparse.ArgumentParser(
        prog="aai-finetune",
        description="Offline-first local fine-tuning curriculum",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="set library offline controls and deny Python socket connections",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    handlers: dict[str, Command] = {}

    def command(name: str, help_text: str, handler: Command) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name, help=help_text)
        handlers[name] = handler
        return child

    command(
        "prepare-flight", "acquire and verify all travel assets", _cmd_prepare_flight
    )
    command(
        "flight-check", "prove the prepared project works offline", _cmd_flight_check
    )
    command("smoke", "run fast deterministic offline checks", _cmd_smoke)
    command("prepare-data", "rebuild immutable portable splits", _cmd_prepare_data)
    baseline_parser = command(
        "baselines", "evaluate deterministic baselines", _cmd_baselines
    )
    baseline_parser.add_argument("--track", action="store_true")

    train_parser = command("train", "run the configured MLX-LM LoRA change", _cmd_train)
    train_parser.add_argument("--iterations", type=int)

    evaluate_parser = command(
        "evaluate", "score frozen local model predictions", _cmd_evaluate
    )
    evaluate_parser.add_argument("--track", action="store_true")
    evaluate_parser.add_argument(
        "--methods",
        default="all",
        help="all or a comma-separated subset of basic,strong,few-shot,lora",
    )
    evaluate_parser.add_argument(
        "--limit",
        type=int,
        help="balanced debug subset; omit for the complete frozen test set",
    )
    evaluate_parser.add_argument("--max-tokens", type=int, default=160)

    command("capstone", "generate the policy-derived capstone", _cmd_capstone)
    capstone_train_parser = command(
        "capstone-train",
        "train the compact capstone LoRA change",
        _cmd_capstone_train,
    )
    capstone_train_parser.add_argument("--iterations", type=int)
    capstone_evaluate_parser = command(
        "capstone-evaluate",
        "compare policy, prompt, LoRA, and hybrid capstone methods",
        _cmd_capstone_evaluate,
    )
    capstone_evaluate_parser.add_argument(
        "--methods",
        default="all",
        help="all or policy,basic,strong,few-shot,lora,hybrid",
    )
    capstone_evaluate_parser.add_argument("--limit", type=int)
    capstone_evaluate_parser.add_argument("--max-tokens", type=int, default=512)
    command(
        "clean-study", "remove generated outputs but retain assets", _cmd_clean_study
    )
    return parser, handlers


def run(argv: Sequence[str] | None = None) -> int:
    """Parse and execute one command, returning a process status."""

    parser, handlers = _parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    strict_offline = args.offline or args.command in {"flight-check", "smoke"}
    if strict_offline:
        enable_offline_environment()
    context = deny_network() if strict_offline else nullcontext()
    try:
        with context:
            handlers[args.command](args, settings)
    except (OfflineAssetError, StudyCommandError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


def main() -> None:
    """Console-script entry point."""

    raise SystemExit(run())


if __name__ == "__main__":
    main()
