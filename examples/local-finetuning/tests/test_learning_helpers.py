"""Tests for the notebook-facing Python seam."""

from __future__ import annotations

from pathlib import Path

import pytest

from aai_local_finetuning import training
from aai_local_finetuning.data import DatasetIntegrityError
from aai_local_finetuning.evaluation import (
    DeterministicInferenceConfig,
    EvaluationRecord,
    GenerationConfig,
    LocalMLXInferenceConfig,
    Prediction,
    SupportOutput,
    evaluate_predictions,
    evaluation_fingerprint,
    start_evaluation_session,
)
from aai_local_finetuning.learning import (
    COMPLETE_EVALUATION_SCOPE,
    generate_support_predictions,
    load_support_splits,
    lora_parameter_budget,
    report_row,
    select_few_shots,
    stratified_evaluation_scope,
    stratified_subsample,
    support_contract,
)
from aai_local_finetuning.modeling import LocalGeneration


def _model_config(*, max_tokens: int, few_shot_examples: int):
    files = (
        training.TrainingFileEvidence(
            path="LOCAL_REVISION",
            sha256="a" * 64,
            size_bytes=41,
        ),
        training.TrainingFileEvidence(
            path="config.json",
            sha256="b" * 64,
            size_bytes=10,
        ),
    )
    base_model = training.BaseModelExecutionContract(
        repository="local/test-model",
        model_path="models/test-model",
        model_revision="c" * 40,
        model_files=files,
        model_files_sha256=training._evidence_sequence_sha256(files),
    )
    return LocalMLXInferenceConfig(
        method="few-shot",
        prompt_recipe="few_shot",
        few_shot_examples=few_shot_examples,
        generation=GenerationConfig(max_tokens=max_tokens),
        base_model=base_model,
    )


def test_support_loader_fails_closed_before_reading_unverified_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Settings:
        processed_dir = Path("prepared")

    def reject_manifest(_path: Path) -> None:
        raise DatasetIntegrityError("frozen test hash mismatch")

    monkeypatch.setattr(
        "aai_local_finetuning.learning.require_valid_manifest",
        reject_manifest,
    )

    with pytest.raises(DatasetIntegrityError, match="frozen test hash mismatch"):
        load_support_splits(Settings())  # type: ignore[arg-type]


def _record(ordinal: int, intent: str, category: str = "account") -> EvaluationRecord:
    return EvaluationRecord(
        example_id=f"example-{ordinal}",
        input_text=f"safe input {ordinal}",
        target=SupportOutput(
            intent=intent,
            category=category,
            requires_escalation=False,
            response="Use the documented support flow.",
        ),
    )


class _Predictor:
    def generate(self, messages, *, max_tokens=160):
        assert messages[-1]["role"] == "user"
        return LocalGeneration(
            text=(
                '{"intent":"intent-a","category":"account",'
                '"requires_escalation":false,'
                '"response":"Use the documented support flow."}'
            ),
            latency_ms=1.5,
            output_tokens=min(12, max_tokens),
            peak_memory_mb=32.0,
        )


def test_contract_and_few_shots_are_train_derived_and_deterministic():
    train = tuple(
        _record(index, f"intent-{letter}") for index, letter in enumerate("abcd")
    )

    intents, categories = support_contract(train)
    shots = select_few_shots(tuple(reversed(train)), limit=3)

    assert intents == ("intent-a", "intent-b", "intent-c", "intent-d")
    assert categories == {intent: "account" for intent in intents}
    assert [shot[1]["intent"] for shot in shots] == [
        "intent-a",
        "intent-b",
        "intent-c",
    ]


def test_generation_helper_returns_measured_strict_predictions():
    train = tuple(
        _record(index, f"intent-{letter}") for index, letter in enumerate("abcd")
    )
    records = (_record(10, "intent-a"),)

    predictions = generate_support_predictions(
        _Predictor(),
        records,
        strategy="few_shot",
        train_records=train,
        inference_config=_model_config(max_tokens=20, few_shot_examples=3),
        few_shot_limit=3,
    )

    assert len(predictions) == 1
    assert predictions[0].example_id == "example-10"
    assert predictions[0].output_tokens == 12
    assert predictions[0].peak_memory_mb == 32.0


def test_support_contract_rejects_inconsistent_categories():
    records = (
        _record(1, "recover_password", "account"),
        _record(2, "recover_password", "payments"),
    )

    try:
        support_contract(records)
    except ValueError as error:
        assert "multiple categories" in str(error)
    else:  # pragma: no cover - protects the explicit teaching contract
        raise AssertionError("inconsistent categories were accepted")


def test_stratified_subsample_keeps_first_n_per_intent_in_frozen_order():
    records = (
        _record(0, "intent-a"),
        _record(1, "intent-b"),
        _record(2, "intent-a"),
        _record(3, "intent-c"),
        _record(4, "intent-a"),
        _record(5, "intent-b"),
        _record(6, "intent-b"),
        _record(7, "intent-c"),
    )

    selected = stratified_subsample(records, per_intent=2)

    assert [record.example_id for record in selected] == [
        "example-0",
        "example-1",
        "example-2",
        "example-3",
        "example-5",
        "example-7",
    ]
    # Every present intent keeps support, so its macro-F1 term is defined.
    counts: dict[str, int] = {}
    for record in selected:
        counts[record.target.intent] = counts.get(record.target.intent, 0) + 1
    assert counts == {"intent-a": 2, "intent-b": 2, "intent-c": 2}
    # Deterministic: the same frozen order always yields the same selection.
    assert stratified_subsample(records, per_intent=2) == selected
    # A rarer intent keeps whatever support it has rather than being dropped.
    assert len(stratified_subsample(records[:4], per_intent=2)) == 4


def test_stratified_subsample_and_scope_reject_invalid_per_intent():
    records = (_record(0, "intent-a"),)

    for invalid in (0, -1, True, "2", None):
        with pytest.raises(ValueError):
            stratified_subsample(records, per_intent=invalid)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            stratified_evaluation_scope(invalid)  # type: ignore[arg-type]


def test_evaluation_scope_names_are_stable_file_prefixes():
    assert COMPLETE_EVALUATION_SCOPE == "complete"
    assert stratified_evaluation_scope(2) == "stratified-subsample-2-per-intent"
    assert stratified_evaluation_scope(5) == "stratified-subsample-5-per-intent"


def test_evaluation_fingerprint_is_order_insensitive_and_content_bound():
    records = tuple(
        _record(index, f"intent-{letter}") for index, letter in enumerate("abc")
    )

    fingerprint = evaluation_fingerprint(records)

    assert len(fingerprint) == 64
    assert evaluation_fingerprint(tuple(reversed(records))) == fingerprint
    assert evaluation_fingerprint(records[:2]) != fingerprint
    with pytest.raises(ValueError, match="must not be empty"):
        evaluation_fingerprint(())


def _qwen2_config(**overrides):
    config = {
        "model_type": "qwen2",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 16,
        "vocab_size": 10,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "tie_word_embeddings": True,
    }
    config.update(overrides)
    return config


def _lora_config(**overrides):
    config = {
        "num_layers": 1,
        "lora_parameters": {
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
            "rank": 2,
            "scale": 16.0,
            "dropout": 0.0,
        },
    }
    config.update(overrides)
    return config


def test_lora_parameter_budget_arithmetic_on_a_small_fake_config():
    budget = lora_parameter_budget(_qwen2_config(), _lora_config())

    # Per layer: q (8*8+8) + k,v (2*(8*4+4)) + o (8*8) + MLP (3*8*16)
    # + two RMSNorms (2*8) = 72 + 72 + 64 + 384 + 16 = 608.
    # Total: embeddings 10*8 + 2*608 + final norm 8 = 1304 (tied output).
    assert budget.total_parameters == 1304
    # LoRA pairs: q rank*(8+8)=32, v rank*(8+4)=24 -> 56 on one layer.
    assert budget.lora_trainable_parameters == 56
    assert budget.trainable_fraction == pytest.approx(56 / 1304)
    assert budget.adapted_layers == 1
    assert budget.rank == 2
    assert budget.adapted_projections == ("self_attn.q_proj", "self_attn.v_proj")

    untied = lora_parameter_budget(
        _qwen2_config(tie_word_embeddings=False), _lora_config()
    )
    assert untied.total_parameters == 1304 + 80

    all_layers = lora_parameter_budget(_qwen2_config(), _lora_config(num_layers=-1))
    assert all_layers.adapted_layers == 2
    assert all_layers.lora_trainable_parameters == 112


def test_lora_parameter_budget_matches_the_pinned_course_model_shape():
    model_config = {
        "model_type": "qwen2",
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "tie_word_embeddings": True,
    }
    lora_config = {
        "num_layers": 8,
        "lora_parameters": {
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
            "rank": 8,
            "scale": 16.0,
            "dropout": 0.0,
        },
    }

    budget = lora_parameter_budget(model_config, lora_config)

    assert budget.total_parameters == 494_032_768
    assert budget.lora_trainable_parameters == 180_224
    assert budget.trainable_fraction == pytest.approx(180_224 / 494_032_768)


def test_lora_parameter_budget_fails_closed_on_unsupported_shapes():
    with pytest.raises(ValueError, match="qwen2 layout only"):
        lora_parameter_budget(_qwen2_config(model_type="llama"), _lora_config())
    with pytest.raises(ValueError, match="unsupported LoRA projection key"):
        lora_parameter_budget(
            _qwen2_config(),
            _lora_config(lora_parameters={"keys": ["self_attn.qkv_proj"], "rank": 2}),
        )
    with pytest.raises(ValueError, match="num_layers"):
        lora_parameter_budget(_qwen2_config(), _lora_config(num_layers=99))
    with pytest.raises(ValueError, match="positive integer"):
        lora_parameter_budget(
            _qwen2_config(hidden_size=0),
            _lora_config(),
        )
    with pytest.raises(ValueError, match="head_dim"):
        lora_parameter_budget(
            _qwen2_config(hidden_size=9),
            _lora_config(),
        )


def test_report_row_keeps_quality_and_performance_separate():
    record = _record(1, "intent-a")
    evaluation_session = start_evaluation_session()
    prediction = Prediction(
        example_id=record.example_id,
        raw_text=record.target.model_dump_json(),
        latency_ms=1.0,
        output_tokens=10,
        peak_memory_mb=20.0,
    )
    report = evaluate_predictions(
        (record,),
        (prediction,),
        evaluation_session=evaluation_session,
        inference_config=DeterministicInferenceConfig(method="baseline"),
    )

    row = report_row("baseline", report)

    assert row["method"] == "baseline"
    assert "macro_f1" in row
    assert "schema_validity" in row
    assert "response_policy" in row
    assert "mean_latency_ms" in row
