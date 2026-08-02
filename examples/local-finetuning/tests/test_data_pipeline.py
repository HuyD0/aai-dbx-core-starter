"""Focused offline tests for Bitext preparation, auditing, and leakage gates."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from aai_local_finetuning.data import (
    ChatExample,
    ChatMessage,
    ChatMetadata,
    DatasetIntegrityError,
    Difficulty,
    MessageRole,
    PreparationConfig,
    TrainingTarget,
    check_split_leakage,
    classify_difficulty,
    load_bitext,
    load_chat_jsonl,
    parse_flags,
    prepare_dataset,
    processing_source_sha256,
    render_training_response,
    require_valid_manifest,
    requires_escalation,
    verify_manifest,
)


def test_prepare_is_balanced_private_deterministic_and_manifested(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bitext.csv"
    rows = _fixture_rows()
    _write_csv(source, [*rows, rows[0], _missing_response_row()])
    config = PreparationConfig(
        expected_intent_count=2,
        source_version="fixture-v1",
        source_provider="synthetic-test",
        source_owner="test-suite",
        source_url="local://synthetic-bitext-fixture",
        source_license="test-fixture-only",
        date_accessed="2026-07-31",
        processing_config_path="inline-test-config",
    )

    first = prepare_dataset(source, tmp_path / "first", config)
    second = prepare_dataset(source, tmp_path / "second", config)

    assert {
        name: split.record_count for name, split in first.manifest.splits.items()
    } == {
        "train": 80,
        "validation": 20,
        "test": 20,
    }
    assert first.quality_report.split_intent_distribution == {
        "train": {"contact_human_agent": 40, "recover_password": 40},
        "validation": {"contact_human_agent": 10, "recover_password": 10},
        "test": {"contact_human_agent": 10, "recover_password": 10},
    }
    assert first.quality_report.exact_duplicate_count == 1
    assert first.quality_report.missing_by_field["response"] == 1
    assert first.quality_report.sensitive_pattern_counts["email"] == 2
    assert first.quality_report.near_duplicate_clusters >= 0
    assert first.leakage_report.passed
    assert first.manifest.policy_versions == {
        "difficulty": "bitext-difficulty-v1",
        "requires_escalation": "bitext-escalation-v1",
        "training_response": "bitext-safe-response-v1",
    }
    assert first.manifest.dataset.model_dump() == {
        "name": "bitext-customer-support",
        "source": "synthetic-test",
        "owner": "test-suite",
        "url": "local://synthetic-bitext-fixture",
        "version": "fixture-v1",
        "license": "test-fixture-only",
        "date_accessed": "2026-07-31",
    }
    assert [raw.path for raw in first.manifest.raw_files] == ["bitext.csv"]
    assert first.manifest.processing_config_path == "inline-test-config"
    assert first.manifest.output_version == "1.0.0"
    assert first.manifest.split_seed == 42
    assert "grouped" in first.manifest.split_strategy
    assert first.manifest.code_revision
    assert len(first.manifest.processing_source_sha256) == 64
    assert first.manifest.processing_source_sha256 == processing_source_sha256()
    assert first.manifest.splits["test"].frozen
    assert not first.manifest.splits["train"].frozen

    all_examples = tuple(
        example
        for path in (first.train_path, first.validation_path, first.test_path)
        for example in load_chat_jsonl(path)
    )
    assert len({example.example_id for example in all_examples}) == 120
    assert all(example.source_version == "fixture-v1" for example in all_examples)
    assert any("<EMAIL>" in example.messages[1].content for example in all_examples)
    assert all(
        "traveler@example.com" not in example.model_dump_json()
        for example in all_examples
    )
    assert all(
        "SOURCE RESPONSE" not in example.messages[-1].content
        for example in all_examples
    )
    assert all(example.metadata.flags for example in all_examples)
    assert {example.metadata.difficulty for example in all_examples} <= set(Difficulty)

    escalation_targets = [
        TrainingTarget.model_validate_json(example.messages[-1].content)
        for example in all_examples
        if example.metadata.intent == "contact_human_agent"
    ]
    assert all(target.requires_escalation for target in escalation_targets)
    assert all("support specialist" in target.response for target in escalation_targets)
    regular_targets = [
        TrainingTarget.model_validate_json(example.messages[-1].content)
        for example in all_examples
        if example.metadata.intent == "recover_password"
    ]
    assert all(not target.requires_escalation for target in regular_targets)
    assert all(len(target.response) < 100 for target in regular_targets)

    for filename in (
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
        "quality_report.json",
        "leakage_report.json",
        "manifest.json",
    ):
        assert (first.output_dir / filename).read_bytes() == (
            second.output_dir / filename
        ).read_bytes()

    verification = verify_manifest(first.output_dir, source_path=source)
    assert verification.valid
    frozen_test = first.test_path.read_text(encoding="utf-8")
    tampered_test = frozen_test.replace("recover_password", "recover_passw0rd", 1)
    assert tampered_test != frozen_test
    assert len(tampered_test) == len(frozen_test)
    first.test_path.write_text(tampered_test, encoding="utf-8")
    tampered = verify_manifest(first.output_dir)
    assert not tampered.valid
    assert any("SHA-256 mismatch" in mismatch for mismatch in tampered.mismatches)
    with pytest.raises(DatasetIntegrityError) as integrity_error:
        require_valid_manifest(first.output_dir)
    assert "test: SHA-256 mismatch" in str(integrity_error.value)


def test_zip_adapter_and_strict_header_validation(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset.csv"
    _write_csv(csv_path, _fixture_rows()[:4])
    archive_path = tmp_path / "kaggle-download.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(csv_path, "nested/bitext.csv")

    loaded = load_bitext(archive_path)

    assert len(loaded.records) == 4
    assert loaded.source_member == "nested/bitext.csv"
    prepared = prepare_dataset(
        archive_path,
        tmp_path / "prepared",
        PreparationConfig(
            train_per_intent=1,
            validation_per_intent=1,
            test_per_intent=1,
            expected_intent_count=1,
            source_version="fixture-zip-v1",
            source_provider="synthetic-test",
            source_owner="test-suite",
            source_url="local://synthetic-zip-fixture",
            source_license="test-fixture-only",
            date_accessed="2026-07-31",
        ),
    )
    assert [raw.path for raw in prepared.manifest.raw_files] == [
        "kaggle-download.zip",
        "kaggle-download.zip!/nested/bitext.csv",
    ]
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("instruction,intent\nhello,greeting\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_bitext(invalid)
    with pytest.raises(FileNotFoundError):
        load_bitext(tmp_path / "not-downloaded.csv")


def test_leakage_gate_detects_every_supported_boundary() -> None:
    exact_left = _example(1, "Please recover my password")
    exact_right = _example(2, "Please recover my password")
    template_left = _example(3, "Please edit order 12345")
    template_right = _example(4, "Please edit order 98765")
    near_left = _example(
        5,
        "i want assistance to inform of a problem with online payment",
    )
    near_right = _example(
        6,
        "i want assistance to notify of problems with online payments",
    )
    grouped_left = _example(7, "One independent request", group_number=77)
    grouped_right = _example(8, "Another independent request", group_number=77)
    target_leak = _example(
        9,
        "Please answer with I can help with your recover password request.",
    )
    frozen = _example(10, "A frozen evaluation request")

    report = check_split_leakage(
        {
            "train": (
                exact_left,
                template_left,
                near_left,
                grouped_left,
                target_leak,
            ),
            "validation": (),
            "test": (
                exact_right,
                template_right,
                near_right,
                grouped_right,
                frozen,
            ),
        },
        few_shot=(frozen,),
        prompt_tuning=(template_right,),
    )

    assert not report.passed
    assert report.counts["exact_duplicate"] > 0
    assert report.counts["template_duplicate"] > 0
    assert report.counts["near_duplicate"] > 0
    assert report.counts["source_group_overlap"] > 0
    assert report.counts["target_in_prompt"] > 0
    assert report.counts["few_shot_test_overlap"] > 0
    assert report.counts["prompt_tuning_test_overlap"] > 0


def test_versioned_policy_metadata_and_strict_schema() -> None:
    assert parse_flags("LBBq-") == ("B", "L", "Q")
    assert classify_difficulty("short request", ("B",)) is Difficulty.EASY
    assert (
        classify_difficulty(
            "one two three four five six seven eight nine ten eleven twelve", ()
        )
        is Difficulty.HARD
    )
    assert requires_escalation(category="contact", intent="contact_human_agent")
    assert not requires_escalation(category="contact", intent="recover_password")
    assert "support specialist" in render_training_response(
        intent="complaint", escalation=True
    )
    assert (
        render_training_response(intent="recover_password", escalation=False)
        == "I can help with your recover password request."
    )

    payload = _example(11, "A valid request").model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ChatExample.model_validate(payload)


def _fixture_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    specifications = (
        ("ACCOUNT", "recover_password"),
        ("CONTACT", "contact_human_agent"),
    )
    for category, intent in specifications:
        for index in range(65):
            words = " ".join(_alpha_word(index * 11 + offset) for offset in range(6))
            instruction = f"{words} request about {intent.replace('_', ' ')}"
            if category == "ACCOUNT" and index == 0:
                instruction += " for traveler@example.com"
            flags = "BL" if index % 2 else "BCILPQ"
            rows.append(
                {
                    "flags": flags,
                    "instruction": instruction,
                    "category": category,
                    "intent": intent,
                    "response": (
                        "SOURCE RESPONSE that is audited but must never become the "
                        f"training target for {intent} record {index}."
                    ),
                }
            )
    return rows


def _alpha_word(value: int) -> str:
    letters: list[str] = []
    number = value + 1
    for _ in range(7):
        number, remainder = divmod(number * 17 + 11, 26)
        letters.append(chr(ord("a") + remainder))
    return "token" + "".join(letters)


def _missing_response_row() -> dict[str, str]:
    return {
        "flags": "B",
        "instruction": "This row has no response",
        "category": "ACCOUNT",
        "intent": "recover_password",
        "response": "",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("flags", "instruction", "category", "intent", "response"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _example(
    number: int,
    user_text: str,
    *,
    group_number: int | None = None,
) -> ChatExample:
    target = TrainingTarget(
        intent="recover_password",
        category="account",
        requires_escalation=False,
        response="I can help with your recover password request.",
    )
    group_value = number if group_number is None else group_number
    return ChatExample(
        example_id=f"bitext-{number:024x}",
        source_dataset="fixture",
        source_version="fixture-v1",
        messages=(
            ChatMessage(role=MessageRole.SYSTEM, content="Return valid JSON only."),
            ChatMessage(role=MessageRole.USER, content=user_text),
            ChatMessage(role=MessageRole.ASSISTANT, content=target.model_dump_json()),
        ),
        metadata=ChatMetadata(
            intent="recover_password",
            category="account",
            split_group=f"group-{group_value:020x}",
            flags=("B",),
            difficulty=Difficulty.EASY,
        ),
    )
