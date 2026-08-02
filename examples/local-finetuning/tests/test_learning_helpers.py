"""Tests for the notebook-facing Python seam."""

from __future__ import annotations

from pathlib import Path

import pytest

from aai_local_finetuning.data import DatasetIntegrityError
from aai_local_finetuning.evaluation import (
    EvaluationRecord,
    Prediction,
    SupportOutput,
    evaluate_predictions,
    start_evaluation_session,
)
from aai_local_finetuning.learning import (
    generate_support_predictions,
    load_support_splits,
    report_row,
    select_few_shots,
    support_contract,
)
from aai_local_finetuning.modeling import LocalGeneration


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
        max_tokens=20,
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
    )

    row = report_row("baseline", report)

    assert row["method"] == "baseline"
    assert "macro_f1" in row
    assert "schema_validity" in row
    assert "response_policy" in row
    assert "mean_latency_ms" in row
