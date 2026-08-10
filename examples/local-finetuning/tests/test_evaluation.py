from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_local_finetuning import training
from aai_local_finetuning.evaluation import (
    BaselineEvaluation,
    DeterministicInferenceConfig,
    EvaluationDataError,
    EvaluationRecord,
    Evaluator,
    GenerationConfig,
    KeywordRuleBaseline,
    LocalMLXInferenceConfig,
    MajorityBaseline,
    Prediction,
    PromotionDecision,
    PromotionThresholds,
    SupportOutput,
    decide_lora_promotion,
    evaluate_predictions,
    format_error_analysis,
    load_predictions_jsonl,
    load_records_jsonl,
    parse_portable_record,
    start_evaluation_session,
    write_predictions_jsonl,
    write_records_jsonl,
    write_report_json,
)
from aai_local_finetuning.evaluation import metrics as evaluation_metrics
from aai_local_finetuning.evaluation import promotion as promotion_module


def test_default_promotion_contract_requires_a_minimum_useful_gain():
    thresholds = PromotionThresholds()

    assert thresholds.minimum_macro_f1_gain == 0.01


def _deterministic_config(method: str) -> DeterministicInferenceConfig:
    return DeterministicInferenceConfig(method=method)


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
        repository="local/test-model",
        model_path="models/test-model",
        model_revision="3" * 40,
        model_files=files,
        model_files_sha256=training._evidence_sequence_sha256(files),
    )


def _model_config(
    method: str,
    base_model: training.BaseModelExecutionContract,
    *,
    adapter_manifest_sha256: str | None = None,
    max_tokens: int = 37,
    few_shot_examples: int = 0,
) -> LocalMLXInferenceConfig:
    return LocalMLXInferenceConfig(
        method=method,
        prompt_recipe="strong",
        few_shot_examples=few_shot_examples,
        generation=GenerationConfig(max_tokens=max_tokens),
        base_model=base_model,
        adapter_manifest_sha256=adapter_manifest_sha256,
    )


@pytest.mark.parametrize("mutation", ("source", "package"))
def test_evaluator_rejects_source_or_package_change_while_scoring(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    record = _record("one", "forgot password", "recover_password", "account")
    session = SimpleNamespace(execution_contract_sha256="a" * 64)
    checks = 0

    def recheck(_session: object) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError(f"{mutation} changed during the evaluation session")

    monkeypatch.setattr(evaluation_metrics, "recheck_evaluation_session", recheck)

    with pytest.raises(RuntimeError, match="changed during"):
        evaluate_predictions(
            [record],
            _perfect_predictions([record]),
            evaluation_session=session,  # type: ignore[arg-type]
            inference_config=_deterministic_config("unit-test-drift"),
        )


def _record(
    example_id: str,
    text: str,
    intent: str,
    category: str,
    *,
    escalation: bool = False,
    response: str = "I can help with that request.",
    flags: tuple[str, ...] = (),
    difficulty: str = "unspecified",
) -> EvaluationRecord:
    return EvaluationRecord(
        example_id=example_id,
        input_text=text,
        target=SupportOutput(
            intent=intent,
            category=category,
            requires_escalation=escalation,
            response=response,
        ),
        source_dataset="synthetic-support",
        source_version="1.0",
        system_prompt="Return valid JSON only.",
        flags=flags,
        difficulty=difficulty,
        metadata={"intent": intent, "category": category},
    )


def _prediction(
    record: EvaluationRecord,
    output: SupportOutput | str,
    *,
    latency_ms: float = 10.0,
    output_tokens: int = 20,
    peak_memory_mb: float = 128.0,
) -> Prediction:
    raw_text = output.model_dump_json() if isinstance(output, SupportOutput) else output
    return Prediction(
        example_id=record.example_id,
        raw_text=raw_text,
        latency_ms=latency_ms,
        output_tokens=output_tokens,
        peak_memory_mb=peak_memory_mb,
    )


def _perfect_predictions(records: list[EvaluationRecord]) -> tuple[Prediction, ...]:
    return tuple(_prediction(record, record.target) for record in records)


def _mutated_execution_contract(
    contract: training.ExecutionContract,
    mutation: str,
) -> training.ExecutionContract:
    source_files = contract.source_files
    packages = contract.runtime_packages
    if mutation == "source":
        first = source_files[0]
        changed_digest = "0" * 64 if first.sha256 != "0" * 64 else "1" * 64
        source_files = (
            first.model_copy(update={"sha256": changed_digest}),
            *source_files[1:],
        )
    else:
        packages = tuple(
            sorted(
                (
                    *packages,
                    training.RuntimePackageEvidence(
                        name="zz-runtime-mutation-test",
                        version="1.0.0",
                        payload_file_count=1,
                        payload_size_bytes=10,
                        payload_files_sha256="f" * 64,
                    ),
                ),
                key=lambda package: package.name,
            )
        )
    return training.ExecutionContract(
        python_version=contract.python_version,
        python_implementation=contract.python_implementation,
        operating_system=contract.operating_system,
        machine=contract.machine,
        source_files=source_files,
        source_files_sha256=training._evidence_sequence_sha256(source_files),
        runtime_packages=packages,
        runtime_packages_sha256=training._evidence_sequence_sha256(packages),
    )


def test_support_output_is_strict_and_forbids_extras() -> None:
    valid = {
        "intent": "recover_password",
        "category": "account",
        "requires_escalation": False,
        "response": "I can help you reset your password.",
    }

    assert SupportOutput.model_validate(valid).intent == "recover_password"
    with pytest.raises(ValidationError):
        SupportOutput.model_validate({**valid, "confidence": 0.99})
    with pytest.raises(ValidationError):
        SupportOutput.model_validate({**valid, "requires_escalation": "false"})


def test_portable_record_and_prediction_jsonl_round_trip(tmp_path: Path) -> None:
    target = {
        "intent": "recover_password",
        "category": "account",
        "requires_escalation": False,
        "response": "I can help you reset your password.",
    }
    payload = {
        "example_id": "bitext-000001",
        "source_dataset": "bitext-customer-support",
        "source_version": "2026-07-31",
        "messages": [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "I cannot remember my password."},
            {"role": "assistant", "content": json.dumps(target)},
        ],
        "metadata": {
            "intent": "recover_password",
            "category": "account",
            "flags": ["ambiguous", "short"],
            "difficulty": "hard",
        },
    }
    parsed = parse_portable_record(payload)

    assert parsed.input_text == payload["messages"][1]["content"]
    assert parsed.target == SupportOutput.model_validate(target)
    assert parsed.flags == ("ambiguous", "short")
    assert parsed.difficulty == "hard"

    records_path = tmp_path / "records.jsonl"
    write_records_jsonl(records_path, [parsed])
    assert load_records_jsonl(records_path) == [parsed]

    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        json.dumps(
            {
                "example_id": parsed.example_id,
                "output": json.dumps(target),
                "latency_seconds": 0.025,
                "generated_tokens": 17,
                "peak_memory_gb": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prediction = load_predictions_jsonl(prediction_path)[0]
    assert prediction.latency_ms == 25.0
    assert prediction.output_tokens == 17
    assert prediction.peak_memory_mb == 512.0

    canonical_path = tmp_path / "predictions-canonical.jsonl"
    write_predictions_jsonl(canonical_path, [prediction])
    assert load_predictions_jsonl(canonical_path) == [prediction]

    payload["metadata"]["intent"] = "wrong_label"
    with pytest.raises(EvaluationDataError, match="does not match"):
        parse_portable_record(payload)


def test_train_only_deterministic_baselines_are_useful() -> None:
    train = [
        _record("train-1", "I forgot my password", "recover_password", "account"),
        _record("train-2", "Reset my login password", "recover_password", "account"),
        _record("train-3", "Cash withdrawal at an ATM", "cash_withdrawal", "cash"),
        _record("train-4", "ATM cash machine withdrawal", "cash_withdrawal", "cash"),
        _record("train-5", "My password will not work", "recover_password", "account"),
    ]
    test = [
        _record("test-1", "Forgot password", "recover_password", "account"),
        _record("test-2", "ATM withdrawal", "cash_withdrawal", "cash"),
    ]

    majority = MajorityBaseline.fit(train)
    keyword = KeywordRuleBaseline.fit(train)
    majority_predictions = majority.predict_many(test)
    keyword_predictions = keyword.predict_many(test)

    assert majority.intent == "recover_password"
    assert majority.meaningful is False
    assert keyword.meaningful is True
    assert keyword.training_example_ids == {record.example_id for record in train}
    assert not keyword.training_example_ids.intersection(
        record.example_id for record in test
    )
    assert keyword.predict_intent("Please reset my password") == "recover_password"
    assert keyword.predict_intent("Cash from the ATM") == "cash_withdrawal"
    assert [prediction.raw_text for prediction in keyword.predict_many(test)] == [
        prediction.raw_text for prediction in keyword_predictions
    ]

    evaluation_session = start_evaluation_session()
    majority_report = evaluate_predictions(
        test,
        majority_predictions,
        evaluation_session=evaluation_session,
        inference_config=_deterministic_config("majority"),
    )
    keyword_report = evaluate_predictions(
        test,
        keyword_predictions,
        evaluation_session=evaluation_session,
        inference_config=_deterministic_config("keyword-rule"),
    )
    assert majority_report.classification.intent_accuracy == 0.5
    assert keyword_report.classification.intent_accuracy == 1.0
    assert keyword_report.classification.macro_f1 > (
        majority_report.classification.macro_f1
    )
    assert keyword.keywords_by_intent["cash_withdrawal"]


def test_evaluator_tracks_structure_policy_performance_and_slices(
    tmp_path: Path,
) -> None:
    records = [
        _record(
            "eval-1",
            "Forgot password",
            "recover_password",
            "account",
            flags=("ambiguous",),
            difficulty="hard",
        ),
        _record(
            "eval-2",
            "Reset password",
            "recover_password",
            "account",
            difficulty="easy",
        ),
        _record(
            "eval-3",
            "ATM cash",
            "cash_withdrawal",
            "cash",
            escalation=True,
            flags=("safety",),
            difficulty="hard",
        ),
        _record(
            "eval-4",
            "Cash withdrawal",
            "cash_withdrawal",
            "cash",
            difficulty="easy",
        ),
    ]
    predictions = [
        _prediction(records[0], records[0].target, latency_ms=10, output_tokens=10),
        _prediction(records[1], "not json", latency_ms=20, output_tokens=2),
        _prediction(
            records[2],
            SupportOutput(
                intent="invented_intent",
                category="cash",
                requires_escalation=True,
                response="Send me your password so I can investigate.",
            ),
            latency_ms=30,
            output_tokens=30,
            peak_memory_mb=256,
        ),
        _prediction(
            records[3],
            json.dumps(
                {
                    "intent": "cash_withdrawal",
                    "category": "cash",
                    "requires_escalation": False,
                    "response": "I can help.",
                    "extra": "not allowed",
                }
            ),
            latency_ms=40,
            output_tokens=40,
        ),
    ]

    report = Evaluator(
        supported_intents=("recover_password", "cash_withdrawal"),
        error_limit=2,
    ).evaluate(
        records,
        predictions,
        evaluation_session=start_evaluation_session(),
        inference_config=_deterministic_config("metrics-fixture"),
    )

    assert report.classification.intent_accuracy == pytest.approx(0.25)
    assert report.inference_config == _deterministic_config("metrics-fixture")
    assert report.classification.macro_precision == pytest.approx(0.5)
    assert report.classification.macro_recall == pytest.approx(0.25)
    assert report.classification.macro_f1 == pytest.approx(1 / 3)
    assert report.classification.weighted_f1 == pytest.approx(1 / 3)
    assert report.classification.per_intent_f1 == {
        "cash_withdrawal": 0.0,
        "recover_password": pytest.approx(2 / 3),
    }
    assert report.classification.category_accuracy == 0.5
    assert report.classification.escalation_accuracy == 0.5
    assert report.output_quality.json_parse_rate == 0.75
    assert report.output_quality.json_schema_validity_rate == 0.5
    assert report.output_quality.unsupported_intent_rate == 0.25
    assert report.output_quality.response_policy_compliance_rate == 0.25
    assert report.performance.latency_ms.mean == 25.0
    assert report.performance.latency_ms.p95 == pytest.approx(38.5)
    assert report.performance.output_tokens.maximum == 40.0
    assert report.performance.peak_memory_mb.maximum == 256.0
    assert report.by_intent["recover_password"].count == 2
    assert report.by_flag["ambiguous"].count == 1
    assert report.by_flag["__none__"].count == 2
    assert report.by_difficulty["hard"].count == 2
    assert report.error_analysis.total_errors == 3
    assert report.error_analysis.truncated is True
    rendered = format_error_analysis(report)
    assert "3/4 examples with errors" in rendered
    assert "additional examples omitted" in rendered
    assert report.flat_metrics()["response/policy_compliance_rate"] == 0.25

    report_path = tmp_path / "report.json"
    write_report_json(report_path, report)
    assert json.loads(report_path.read_text(encoding="utf-8"))["total_examples"] == 4


def test_promotion_requires_best_meaningful_baseline_and_absolute_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _record("one", "forgot password", "recover_password", "account"),
        _record("two", "cash at atm", "cash_withdrawal", "cash"),
    ]
    evaluation_session = start_evaluation_session()
    perfect_report = evaluate_predictions(
        records,
        _perfect_predictions(records),
        evaluation_session=evaluation_session,
        inference_config=_deterministic_config("perfect-fixture"),
    )
    weak_predictions = (
        _prediction(records[0], records[0].target),
        _prediction(records[1], records[0].target),
    )
    weak_report = evaluate_predictions(
        records,
        weak_predictions,
        evaluation_session=evaluation_session,
        inference_config=_deterministic_config("weak-fixture"),
    )
    base_model = _base_model_contract()
    strong_report = weak_report.model_copy(
        update={"inference_config": _model_config("strong", base_model)}
    )
    baselines = [
        BaselineEvaluation(name="majority", report=perfect_report, meaningful=False),
        BaselineEvaluation(name="keyword-rule", report=weak_report, meaningful=True),
        BaselineEvaluation(name="strong", report=strong_report, meaningful=True),
    ]
    monkeypatch.setattr(
        promotion_module,
        "recheck_training_snapshot",
        lambda snapshot: snapshot,
    )
    monkeypatch.setattr(
        promotion_module,
        "recheck_evaluation_session",
        lambda session: session,
    )

    def snapshot(digest: str):
        return SimpleNamespace(
            manifest_sha256=digest,
            manifest=SimpleNamespace(
                execution_contract_sha256=(
                    perfect_report.evaluation_execution_contract_sha256
                ),
                model_path=base_model.model_path,
                model_revision=base_model.model_revision,
                model_files=base_model.model_files,
            ),
        )

    def change_report(report, digest: str):
        return report.model_copy(
            update={
                "inference_config": _model_config(
                    "lora-change",
                    base_model,
                    adapter_manifest_sha256=digest,
                ),
                "training_manifest_sha256": digest,
                "training_execution_contract_sha256": (
                    report.evaluation_execution_contract_sha256
                ),
            }
        )

    promotion_session = SimpleNamespace(
        execution_contract_sha256=(perfect_report.evaluation_execution_contract_sha256),
        base_model_execution_contract=base_model,
    )

    adopted = decide_lora_promotion(
        change_name="support-lora-v1",
        evaluation_session=promotion_session,  # type: ignore[arg-type]
        training_snapshot=snapshot("a" * 64),
        change_report=change_report(perfect_report, "a" * 64),
        baselines=baselines,
        thresholds=PromotionThresholds(
            minimum_schema_validity_rate=1.0,
            minimum_policy_compliance_rate=1.0,
            minimum_macro_f1_gain=0.0,
        ),
    )
    assert adopted.decision is PromotionDecision.ADOPT
    assert adopted.baseline is not None
    assert adopted.baseline.name == "keyword-rule"
    assert adopted.change.method == "lora_fine_tune"
    assert adopted.change.training_manifest_sha256 == "a" * 64
    assert (
        adopted.change.evaluation_execution_contract_sha256
        == perfect_report.evaluation_execution_contract_sha256
    )
    assert adopted.result.beats_strongest_meaningful_baseline is True
    assert adopted.change.inference_config.generation.max_tokens == 37

    tied = decide_lora_promotion(
        change_name="support-lora-v2",
        evaluation_session=promotion_session,  # type: ignore[arg-type]
        training_snapshot=snapshot("b" * 64),
        change_report=change_report(perfect_report, "b" * 64),
        baselines=[
            BaselineEvaluation(
                name="strong",
                report=perfect_report.model_copy(
                    update={"inference_config": _model_config("strong", base_model)}
                ),
                meaningful=True,
            )
        ],
    )
    assert tied.decision is PromotionDecision.REJECT

    unsafe_predictions = (
        _prediction(
            records[0],
            SupportOutput(
                intent="recover_password",
                category="account",
                requires_escalation=False,
                response="Send me your password.",
            ),
        ),
        _prediction(records[1], records[1].target),
    )
    unsafe_report = evaluate_predictions(
        records,
        unsafe_predictions,
        evaluation_session=evaluation_session,
        inference_config=_deterministic_config("unsafe-fixture"),
    )
    rejected = decide_lora_promotion(
        change_name="support-lora-unsafe",
        evaluation_session=promotion_session,  # type: ignore[arg-type]
        training_snapshot=snapshot("c" * 64),
        change_report=change_report(unsafe_report, "c" * 64),
        baselines=baselines,
        thresholds=PromotionThresholds(minimum_policy_compliance_rate=1.0),
    )
    assert rejected.decision is PromotionDecision.REJECT
    assert rejected.result.passes_policy_threshold is False

    inconclusive = decide_lora_promotion(
        change_name="support-lora-no-baseline",
        evaluation_session=promotion_session,  # type: ignore[arg-type]
        training_snapshot=snapshot("d" * 64),
        change_report=change_report(perfect_report, "d" * 64),
        baselines=[
            BaselineEvaluation(name="majority", report=weak_report, meaningful=False),
            BaselineEvaluation(name="strong", report=strong_report, meaningful=False),
        ],
    )
    assert inconclusive.decision is PromotionDecision.INCONCLUSIVE
    assert inconclusive.baseline is None

    with pytest.raises(ValueError, match="must carry the supplied"):
        decide_lora_promotion(
            change_name="support-lora-mismatched-lineage",
            evaluation_session=promotion_session,  # type: ignore[arg-type]
            training_snapshot=snapshot("e" * 64),
            change_report=change_report(perfect_report, "a" * 64),
            baselines=baselines,
        )


def test_promotion_is_inconclusive_when_generation_budgets_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _record("one", "forgot password", "recover_password", "account"),
        _record("two", "cash at atm", "cash_withdrawal", "cash"),
    ]
    scoring_session = start_evaluation_session()
    report = evaluate_predictions(
        records,
        _perfect_predictions(records),
        evaluation_session=scoring_session,
        inference_config=_deterministic_config("fixture"),
    )
    base_model = _base_model_contract()
    training_digest = "9" * 64
    change = report.model_copy(
        update={
            "inference_config": _model_config(
                "lora-change",
                base_model,
                adapter_manifest_sha256=training_digest,
                max_tokens=37,
            ),
            "training_manifest_sha256": training_digest,
            "training_execution_contract_sha256": (
                report.evaluation_execution_contract_sha256
            ),
        }
    )
    baseline = report.model_copy(
        update={
            "inference_config": _model_config(
                "strong",
                base_model,
                max_tokens=38,
            )
        }
    )
    snapshot = SimpleNamespace(
        manifest_sha256=training_digest,
        manifest=SimpleNamespace(
            execution_contract_sha256=report.evaluation_execution_contract_sha256,
            model_path=base_model.model_path,
            model_revision=base_model.model_revision,
            model_files=base_model.model_files,
        ),
    )
    promotion_session = SimpleNamespace(
        execution_contract_sha256=report.evaluation_execution_contract_sha256,
        base_model_execution_contract=base_model,
    )
    monkeypatch.setattr(
        promotion_module,
        "recheck_evaluation_session",
        lambda session: session,
    )
    monkeypatch.setattr(
        promotion_module,
        "recheck_training_snapshot",
        lambda value: value,
    )

    assessment = decide_lora_promotion(
        change_name="support-lora-budget-mismatch",
        evaluation_session=promotion_session,  # type: ignore[arg-type]
        training_snapshot=snapshot,  # type: ignore[arg-type]
        change_report=change,
        baselines=[BaselineEvaluation(name="strong", report=baseline)],
    )

    assert assessment.decision is PromotionDecision.INCONCLUSIVE
    assert assessment.result.comparable is False
    assert "different generation settings" in " ".join(assessment.result.reasons)
    assert assessment.change.inference_config.generation.max_tokens == 37
    assert assessment.baseline is not None
    assert assessment.baseline.inference_config.generation.max_tokens == 38


def test_promotion_is_inconclusive_when_control_few_shot_counts_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _record("one", "forgot password", "recover_password", "account"),
        _record("two", "cash at atm", "cash_withdrawal", "cash"),
    ]
    scoring_session = start_evaluation_session()
    report = evaluate_predictions(
        records,
        _perfect_predictions(records),
        evaluation_session=scoring_session,
        inference_config=_deterministic_config("fixture"),
    )
    base_model = _base_model_contract()
    training_digest = "8" * 64
    change = report.model_copy(
        update={
            "inference_config": _model_config(
                "lora-change",
                base_model,
                adapter_manifest_sha256=training_digest,
            ),
            "training_manifest_sha256": training_digest,
            "training_execution_contract_sha256": (
                report.evaluation_execution_contract_sha256
            ),
        }
    )
    baseline = report.model_copy(
        update={
            "inference_config": _model_config(
                "strong",
                base_model,
                few_shot_examples=1,
            )
        }
    )
    snapshot = SimpleNamespace(
        manifest_sha256=training_digest,
        manifest=SimpleNamespace(
            execution_contract_sha256=report.evaluation_execution_contract_sha256,
            model_path=base_model.model_path,
            model_revision=base_model.model_revision,
            model_files=base_model.model_files,
        ),
    )
    promotion_session = SimpleNamespace(
        execution_contract_sha256=report.evaluation_execution_contract_sha256,
        base_model_execution_contract=base_model,
    )
    monkeypatch.setattr(
        promotion_module,
        "recheck_evaluation_session",
        lambda session: session,
    )
    monkeypatch.setattr(
        promotion_module,
        "recheck_training_snapshot",
        lambda value: value,
    )

    assessment = decide_lora_promotion(
        change_name="support-lora-shot-mismatch",
        evaluation_session=promotion_session,  # type: ignore[arg-type]
        training_snapshot=snapshot,  # type: ignore[arg-type]
        change_report=change,
        baselines=[BaselineEvaluation(name="strong", report=baseline)],
    )

    assert assessment.decision is PromotionDecision.INCONCLUSIVE
    assert assessment.result.comparable is False
    assert "few-shot count" in " ".join(assessment.result.reasons)


@pytest.mark.parametrize("mutation", ("source", "package"))
def test_promotion_rejects_reports_after_source_or_package_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    records = [
        _record("one", "forgot password", "recover_password", "account"),
        _record("two", "cash at atm", "cash_withdrawal", "cash"),
    ]
    report = evaluate_predictions(
        records,
        _perfect_predictions(records),
        evaluation_session=start_evaluation_session(),
        inference_config=_deterministic_config("promotion-drift-fixture"),
    )
    training_digest = report.evaluation_execution_contract_sha256
    base_model = _base_model_contract()
    change_report = report.model_copy(
        update={
            "inference_config": _model_config(
                "lora-change",
                base_model,
                adapter_manifest_sha256="a" * 64,
            ),
            "training_manifest_sha256": "a" * 64,
            "training_execution_contract_sha256": training_digest,
        }
    )
    baseline_report = report.model_copy(
        update={"inference_config": _model_config("strong", base_model)}
    )
    snapshot = SimpleNamespace(
        manifest_sha256="a" * 64,
        manifest=SimpleNamespace(
            execution_contract_sha256=training_digest,
            model_path=base_model.model_path,
            model_revision=base_model.model_revision,
            model_files=base_model.model_files,
        ),
    )
    monkeypatch.setattr(
        promotion_module,
        "recheck_training_snapshot",
        lambda value: value,
    )
    monkeypatch.setattr(
        promotion_module,
        "recheck_evaluation_session",
        lambda _session: (_ for _ in ()).throw(
            RuntimeError(f"{mutation} changed during promotion")
        ),
    )
    promotion_session = SimpleNamespace(
        execution_contract_sha256=training_digest,
        base_model_execution_contract=base_model,
    )

    with pytest.raises(RuntimeError, match="changed during promotion"):
        decide_lora_promotion(
            change_name="support-lora-drifted",
            evaluation_session=promotion_session,  # type: ignore[arg-type]
            training_snapshot=snapshot,  # type: ignore[arg-type]
            change_report=change_report,
            baselines=[BaselineEvaluation(name="strong", report=baseline_report)],
        )


def test_evaluator_rejects_misaligned_prediction_identifiers() -> None:
    record = _record("expected", "forgot password", "recover_password", "account")
    prediction = Prediction(
        example_id="different",
        raw_text=record.target.model_dump_json(),
        latency_ms=1.0,
        output_tokens=1,
        peak_memory_mb=1.0,
    )

    with pytest.raises(ValueError, match="identifiers do not match"):
        evaluate_predictions(
            [record],
            [prediction],
            evaluation_session=start_evaluation_session(),
            inference_config=_deterministic_config("misalignment-fixture"),
        )
