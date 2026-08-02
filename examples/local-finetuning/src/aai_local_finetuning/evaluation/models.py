"""Strict, framework-independent contracts for local evaluation evidence."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictEvidenceModel(BaseModel):
    """Immutable evidence that rejects accidental or misspelled fields."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
    )


class SupportOutput(StrictEvidenceModel):
    """The complete output contract for the customer-support exercise."""

    intent: str = Field(min_length=1)
    category: str = Field(min_length=1)
    requires_escalation: bool
    response: str = Field(min_length=1)

    @field_validator("intent", "category", "response")
    @classmethod
    def reject_surrounding_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        if value != value.strip():
            raise ValueError("value must not contain surrounding whitespace")
        return value


class EvaluationRecord(StrictEvidenceModel):
    """Normalized view of one portable conversational JSONL record."""

    example_id: str = Field(min_length=1)
    input_text: str = Field(min_length=1)
    target: SupportOutput
    source_dataset: str | None = Field(default=None, min_length=1)
    source_version: str | None = Field(default=None, min_length=1)
    system_prompt: str | None = None
    flags: tuple[str, ...] = ()
    difficulty: str = Field(default="unspecified", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("example_id", "input_text", "difficulty")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("flags")
    @classmethod
    def validate_flags(cls, flags: tuple[str, ...]) -> tuple[str, ...]:
        if any(not flag.strip() for flag in flags):
            raise ValueError("flags must not contain blank values")
        if len(set(flags)) != len(flags):
            raise ValueError("flags must not contain duplicates")
        return flags


class Prediction(StrictEvidenceModel):
    """Persistable model output and local resource-use evidence."""

    example_id: str = Field(min_length=1)
    raw_text: str
    latency_ms: float = Field(ge=0.0)
    output_tokens: int = Field(ge=0)
    peak_memory_mb: float = Field(ge=0.0)


class DistributionSummary(StrictEvidenceModel):
    """Small deterministic summary suitable for JSON and MLflow metrics."""

    count: int = Field(ge=0)
    minimum: float
    mean: float
    median: float
    p95: float
    maximum: float


class ClassificationMetrics(StrictEvidenceModel):
    intent_accuracy: float = Field(ge=0.0, le=1.0)
    macro_precision: float = Field(ge=0.0, le=1.0)
    macro_recall: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    weighted_f1: float = Field(ge=0.0, le=1.0)
    per_intent_f1: dict[str, float]
    category_accuracy: float = Field(ge=0.0, le=1.0)
    escalation_accuracy: float = Field(ge=0.0, le=1.0)


class OutputQualityMetrics(StrictEvidenceModel):
    json_parse_rate: float = Field(ge=0.0, le=1.0)
    json_schema_validity_rate: float = Field(ge=0.0, le=1.0)
    unsupported_intent_rate: float = Field(ge=0.0, le=1.0)
    response_policy_compliance_rate: float = Field(ge=0.0, le=1.0)


class SliceMetrics(StrictEvidenceModel):
    """Metrics that remain interpretable for a potentially small slice."""

    count: int = Field(ge=1)
    intent_accuracy: float = Field(ge=0.0, le=1.0)
    category_accuracy: float = Field(ge=0.0, le=1.0)
    escalation_accuracy: float = Field(ge=0.0, le=1.0)
    json_parse_rate: float = Field(ge=0.0, le=1.0)
    json_schema_validity_rate: float = Field(ge=0.0, le=1.0)
    unsupported_intent_rate: float = Field(ge=0.0, le=1.0)
    response_policy_compliance_rate: float = Field(ge=0.0, le=1.0)


class PerformanceMetrics(StrictEvidenceModel):
    latency_ms: DistributionSummary
    output_tokens: DistributionSummary
    peak_memory_mb: DistributionSummary


class ErrorKind(StrEnum):
    JSON_PARSE = "json_parse"
    JSON_SCHEMA = "json_schema"
    UNSUPPORTED_INTENT = "unsupported_intent"
    INTENT = "intent"
    CATEGORY = "category"
    ESCALATION = "escalation"
    RESPONSE_POLICY = "response_policy"


class ErrorExample(StrictEvidenceModel):
    example_id: str = Field(min_length=1)
    kinds: tuple[ErrorKind, ...] = Field(min_length=1)
    expected_intent: str = Field(min_length=1)
    predicted_intent: str | None = None
    input_preview: str
    output_preview: str
    policy_issues: tuple[str, ...] = ()


class ErrorAnalysis(StrictEvidenceModel):
    total_errors: int = Field(ge=0)
    counts: dict[ErrorKind, int]
    examples: tuple[ErrorExample, ...] = ()
    truncated: bool = False


class EvaluationReport(StrictEvidenceModel):
    total_examples: int = Field(ge=1)
    evaluation_fingerprint: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    training_manifest_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    supported_intents: tuple[str, ...] = Field(min_length=1)
    classification: ClassificationMetrics
    output_quality: OutputQualityMetrics
    performance: PerformanceMetrics
    by_intent: dict[str, SliceMetrics]
    by_flag: dict[str, SliceMetrics]
    by_difficulty: dict[str, SliceMetrics]
    error_analysis: ErrorAnalysis

    def flat_metrics(self) -> dict[str, float]:
        """Return stable scalar names for trackers without importing MLflow."""

        values = {
            "intent/accuracy": self.classification.intent_accuracy,
            "intent/macro_precision": self.classification.macro_precision,
            "intent/macro_recall": self.classification.macro_recall,
            "intent/macro_f1": self.classification.macro_f1,
            "intent/weighted_f1": self.classification.weighted_f1,
            "category/accuracy": self.classification.category_accuracy,
            "escalation/accuracy": self.classification.escalation_accuracy,
            "output/json_parse_rate": self.output_quality.json_parse_rate,
            "output/json_schema_validity_rate": (
                self.output_quality.json_schema_validity_rate
            ),
            "output/unsupported_intent_rate": (
                self.output_quality.unsupported_intent_rate
            ),
            "response/policy_compliance_rate": (
                self.output_quality.response_policy_compliance_rate
            ),
            "latency/mean_ms": self.performance.latency_ms.mean,
            "latency/p95_ms": self.performance.latency_ms.p95,
            "tokens/mean": self.performance.output_tokens.mean,
            "memory/peak_max_mb": self.performance.peak_memory_mb.maximum,
        }
        values.update(
            {
                f"intent/{intent}/f1": score
                for intent, score in self.classification.per_intent_f1.items()
            }
        )
        return values
