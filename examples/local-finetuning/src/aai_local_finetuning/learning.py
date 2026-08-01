"""Small, explicit Python helpers for the narrative notebook curriculum.

The command-line interface remains useful for repeatable automation, but the
notebooks need inspectable Python objects at every stage.  This module exposes
that seam without importing MLflow or MLX until a learner chooses to use them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .evaluation import (
    EvaluationRecord,
    EvaluationReport,
    Prediction,
    load_records_jsonl,
)
from .modeling import LocalGeneration, PromptStrategy, build_messages
from .settings import ProjectSettings, load_settings


class SupportPredictor(Protocol):
    """The local generation behavior used by the teaching helpers."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 160,
    ) -> LocalGeneration: ...


@dataclass(frozen=True)
class SupportSplits:
    """In-memory views of the three portable evidence boundaries."""

    train: tuple[EvaluationRecord, ...]
    validation: tuple[EvaluationRecord, ...]
    test: tuple[EvaluationRecord, ...]


def load_support_splits(
    settings: ProjectSettings | None = None,
    *,
    include_test: bool = True,
) -> SupportSplits:
    """Load development records and optionally cross the frozen-test boundary."""

    project = settings or load_settings()
    paths = {
        "train": project.processed_dir / "train.jsonl",
        "validation": project.processed_dir / "valid.jsonl",
        "test": project.processed_dir / "test.jsonl",
    }
    required = (
        ("train", "validation", "test")
        if include_test
        else (
            "train",
            "validation",
        )
    )
    missing = [str(paths[name]) for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(
            "prepared splits are missing; run `make prepare-flight` while online: "
            + ", ".join(missing)
        )
    return SupportSplits(
        train=tuple(load_records_jsonl(paths["train"])),
        validation=tuple(load_records_jsonl(paths["validation"])),
        test=(tuple(load_records_jsonl(paths["test"])) if include_test else ()),
    )


def support_contract(
    train_records: Sequence[EvaluationRecord],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Derive the supported labels and intent/category mapping from train only."""

    if not train_records:
        raise ValueError("train_records must not be empty")
    categories: dict[str, str] = {}
    for record in train_records:
        intent = record.target.intent
        category = record.target.category
        previous = categories.setdefault(intent, category)
        if previous != category:
            raise ValueError(
                f"training records map intent {intent!r} to multiple categories"
            )
    return tuple(sorted(categories)), dict(sorted(categories.items()))


def select_few_shots(
    train_records: Sequence[EvaluationRecord],
    *,
    limit: int = 4,
) -> list[tuple[str, dict[str, object]]]:
    """Select deterministic demonstrations from train without touching evaluation."""

    if limit < 1:
        raise ValueError("limit must be positive")
    selected: list[tuple[str, dict[str, object]]] = []
    seen_intents: set[str] = set()
    for record in sorted(train_records, key=lambda item: item.example_id):
        if record.target.intent in seen_intents:
            continue
        selected.append(
            (
                record.input_text,
                record.target.model_dump(mode="json"),
            )
        )
        seen_intents.add(record.target.intent)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise ValueError(
            f"only {len(selected)} distinct training intents are available"
        )
    return selected


def generate_support_predictions(
    predictor: SupportPredictor,
    records: Sequence[EvaluationRecord],
    *,
    strategy: PromptStrategy,
    train_records: Sequence[EvaluationRecord],
    max_tokens: int = 96,
    few_shot_limit: int = 4,
) -> tuple[Prediction, ...]:
    """Generate measured predictions while keeping prompt evidence train-derived."""

    if not records:
        raise ValueError("records must not be empty")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    allowed_intents, category_by_intent = support_contract(train_records)
    demonstrations = (
        select_few_shots(train_records, limit=few_shot_limit)
        if strategy == "few_shot"
        else None
    )
    predictions = []
    for record in records:
        messages = build_messages(
            record.input_text,
            strategy=strategy,
            allowed_intents=list(allowed_intents),
            category_by_intent=category_by_intent,
            few_shot=demonstrations,
        )
        generated = predictor.generate(messages, max_tokens=max_tokens)
        predictions.append(
            Prediction(
                example_id=record.example_id,
                raw_text=generated.text,
                latency_ms=generated.latency_ms,
                output_tokens=generated.output_tokens,
                peak_memory_mb=generated.peak_memory_mb,
            )
        )
    return tuple(predictions)


def report_row(name: str, report: EvaluationReport) -> dict[str, object]:
    """Return the compact comparison row used across evaluation notebooks."""

    return {
        "method": name,
        "examples": report.total_examples,
        "macro_f1": round(report.classification.macro_f1, 3),
        "weighted_f1": round(report.classification.weighted_f1, 3),
        "category_accuracy": round(report.classification.category_accuracy, 3),
        "json_parse_rate": round(report.output_quality.json_parse_rate, 3),
        "schema_validity": round(report.output_quality.json_schema_validity_rate, 3),
        "unsupported_intent_rate": round(
            report.output_quality.unsupported_intent_rate, 3
        ),
        "response_policy": round(
            report.output_quality.response_policy_compliance_rate, 3
        ),
        "mean_latency_ms": round(report.performance.latency_ms.mean, 1),
        "mean_output_tokens": round(report.performance.output_tokens.mean, 1),
        "peak_memory_mb": round(report.performance.peak_memory_mb.maximum, 1),
    }


def load_report(path: str | Path) -> EvaluationReport:
    """Load one persisted evaluation report through its strict schema."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return EvaluationReport.model_validate_json(source.read_text(encoding="utf-8"))


def report_inventory(directory: str | Path) -> Mapping[str, Path]:
    """List available report artifacts without assuming every experiment ran."""

    source = Path(directory)
    if not source.is_dir():
        return {}
    return {
        path.name.removesuffix("-report.json"): path
        for path in sorted(source.glob("*-report.json"))
    }
