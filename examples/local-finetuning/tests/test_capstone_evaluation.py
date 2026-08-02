"""Tests for the compact, framework-independent capstone evaluator."""

from __future__ import annotations

import json
from types import SimpleNamespace

from aai_local_finetuning import cli, training
from aai_local_finetuning.capstone import (
    CapstonePrediction,
    CheckOutcome,
    CompactReadinessReview,
    DatasetSplit,
    ReadinessStatus,
    build_records,
    compact_expected,
    deterministic_capstone_predictions,
    evaluate_capstone_predictions,
    generate_capstone_dataset,
    load_capstone_records,
    render_capstone_mlx_dataset,
)
from aai_local_finetuning.evaluation import (
    DeterministicInferenceConfig,
    GenerationConfig,
    LocalMLXInferenceConfig,
    start_evaluation_session,
)


def _base_model_contract() -> training.BaseModelExecutionContract:
    files = (
        training.TrainingFileEvidence(
            path="LOCAL_REVISION", sha256="1" * 64, size_bytes=41
        ),
        training.TrainingFileEvidence(
            path="config.json", sha256="2" * 64, size_bytes=10
        ),
    )
    return training.BaseModelExecutionContract(
        repository="local/capstone-model",
        model_path="models/capstone-model",
        model_revision="3" * 40,
        model_files=files,
        model_files_sha256=training._evidence_sequence_sha256(files),
    )


def test_compact_contract_contains_only_actionable_checks() -> None:
    records = build_records(DatasetSplit.TEST, 40)

    for record in records:
        compact = compact_expected(record)
        assert all(check.result is not CheckOutcome.PASS for check in compact.checks)
        assert compact.status is record.expected_output.status
        assert len(compact.model_dump_json()) < len(
            record.expected_output.model_dump_json()
        )


def test_policy_ceiling_scores_perfectly_and_exposes_slices() -> None:
    records = build_records(DatasetSplit.TEST, 40)
    evaluation_session = start_evaluation_session()
    predictions = deterministic_capstone_predictions(records)

    report = evaluate_capstone_predictions(
        records,
        predictions,
        evaluation_session=evaluation_session,
        inference_config=DeterministicInferenceConfig(method="deterministic-policy"),
    )

    assert report.total_examples == 40
    assert report.inference_config == DeterministicInferenceConfig(
        method="deterministic-policy"
    )
    assert report.aggregate.json_parse_rate == 1.0
    assert report.aggregate.schema_validity_rate == 1.0
    assert report.aggregate.status_accuracy == 1.0
    assert report.aggregate.check_result_accuracy == 1.0
    assert report.aggregate.check_severity_accuracy == 1.0
    assert report.aggregate.exact_review_rate == 1.0
    assert report.aggregate.missing_check_rate == 0.0
    assert report.aggregate.extra_check_rate == 0.0
    assert report.error_analysis.total_errors == 0
    assert report.by_slice


def test_evaluator_detects_invalid_missing_and_invented_checks() -> None:
    evaluation_session = start_evaluation_session()
    records = list(build_records(DatasetSplit.TEST, 40))
    actionable = next(record for record in records if compact_expected(record).checks)
    ready = next(
        record
        for record in records
        if compact_expected(record).status is ReadinessStatus.READY
    )
    expected = compact_expected(actionable).model_dump(mode="json")
    expected["checks"] = expected["checks"][1:]
    invented = compact_expected(ready).model_dump(mode="json")
    invented["checks"] = [
        {
            "name": "invented_registry_fact",
            "result": "review",
            "severity": "high",
            "remediation_id": None,
        }
    ]
    selected = (actionable, ready, records[-1])
    predictions = (
        _prediction(actionable.example_id, json.dumps(expected)),
        _prediction(ready.example_id, json.dumps(invented)),
        _prediction(records[-1].example_id, "not-json"),
    )

    report = evaluate_capstone_predictions(
        selected,
        predictions,
        evaluation_session=evaluation_session,
        inference_config=DeterministicInferenceConfig(method="invalid-fixture"),
    )

    assert report.aggregate.exact_review_rate < 1.0
    assert report.aggregate.missing_check_rate > 0.0
    assert report.aggregate.extra_check_rate > 0.0
    assert report.aggregate.json_parse_rate < 1.0
    issues = {
        issue for error in report.error_analysis.examples for issue in error.issues
    }
    assert any(issue.startswith("missing_checks:") for issue in issues)
    assert any(issue.startswith("extra_checks:") for issue in issues)
    assert "json_parse" in issues


def test_capstone_promotion_detects_inference_config_mismatch() -> None:
    records = build_records(DatasetSplit.TEST, 1)
    report = evaluate_capstone_predictions(
        records,
        deterministic_capstone_predictions(records),
        evaluation_session=start_evaluation_session(),
        inference_config=DeterministicInferenceConfig(method="fixture"),
    )
    base_model = _base_model_contract()

    def model_config(
        method: str,
        max_tokens: int,
        adapter: str | None = None,
        *,
        few_shot_examples: int = 0,
    ):
        return LocalMLXInferenceConfig(
            method=method,
            prompt_recipe="basic" if method != "strong" else "strong",
            few_shot_examples=few_shot_examples,
            generation=GenerationConfig(max_tokens=max_tokens),
            base_model=base_model,
            adapter_manifest_sha256=adapter,
        )

    reports = {
        "basic": report.model_copy(
            update={"inference_config": model_config("basic", 257)}
        ),
        "strong": report.model_copy(
            update={"inference_config": model_config("strong", 257)}
        ),
        "few-shot": report.model_copy(
            update={"inference_config": model_config("few-shot", 258)}
        ),
    }
    manifest_digest = "4" * 64
    lora_report = report.model_copy(
        update={
            "inference_config": model_config(
                "capstone-lora-change",
                257,
                manifest_digest,
            )
        }
    )
    snapshot = SimpleNamespace(
        manifest_sha256=manifest_digest,
        manifest=SimpleNamespace(
            model_path=base_model.model_path,
            model_revision=base_model.model_revision,
            model_files=base_model.model_files,
        ),
    )
    decision_session = SimpleNamespace(base_model_execution_contract=base_model)

    reasons = cli._capstone_inference_comparability_reasons(
        reports=reports,
        lora_report=lora_report,
        decision_session=decision_session,  # type: ignore[arg-type]
        training_snapshot=snapshot,  # type: ignore[arg-type]
    )

    assert "different generation settings" in " ".join(reasons)

    reports["few-shot"] = report.model_copy(
        update={"inference_config": model_config("few-shot", 257)}
    )
    reports["basic"] = report.model_copy(
        update={
            "inference_config": model_config(
                "basic",
                257,
                few_shot_examples=1,
            )
        }
    )
    reasons = cli._capstone_inference_comparability_reasons(
        reports=reports,
        lora_report=lora_report,
        decision_session=decision_session,  # type: ignore[arg-type]
        training_snapshot=snapshot,  # type: ignore[arg-type]
    )

    assert "few-shot count" in " ".join(reasons)


def test_mlx_renderer_preserves_ids_and_compact_targets(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "mlx"
    generate_capstone_dataset(source)

    manifest = render_capstone_mlx_dataset(source, output)

    assert {
        name: artifact.record_count for name, artifact in manifest.splits.items()
    } == {"train": 400, "validation": 100, "test": 150}
    assert len(manifest.dataset_fingerprint) == 64
    source_records = load_capstone_records(source / "train.jsonl")
    rendered = [
        json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()
    ]
    assert [record.example_id for record in source_records] == [
        record["example_id"] for record in rendered
    ]
    target = CompactReadinessReview.model_validate_json(
        rendered[0]["messages"][-1]["content"]
    )
    assert all(check.result is not CheckOutcome.PASS for check in target.checks)
    assert (output / "valid.jsonl").is_file()
    assert (output / "manifest.json").is_file()


def _prediction(example_id: str, raw_text: str) -> CapstonePrediction:
    return CapstonePrediction(
        example_id=example_id,
        raw_text=raw_text,
        latency_ms=1.0,
        output_tokens=4,
        peak_memory_mb=10.0,
    )
