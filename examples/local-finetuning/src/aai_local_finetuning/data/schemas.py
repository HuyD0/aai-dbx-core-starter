"""Strict, framework-neutral schemas for the local Bitext data pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model used at every persisted or untrusted-data boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RawBitextRow(StrictModel):
    """One row from the Bitext CSV, before filtering and normalization."""

    source_row: int = Field(ge=2)
    flags: str
    instruction: str
    category: str
    intent: str
    response: str


class BitextLoadResult(StrictModel):
    records: tuple[RawBitextRow, ...]
    headers: tuple[str, ...]
    invalid_csv_rows: int = Field(ge=0)
    source_member: str | None


class CanonicalRecord(StrictModel):
    """Sanitized logical record used for auditing, grouping, and splitting."""

    example_id: str = Field(pattern=r"^bitext-[0-9a-f]{24}$")
    flags: str
    instruction: str = Field(min_length=1)
    category: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    response: str = Field(min_length=1)
    template_id: str = Field(pattern=r"^template-[0-9a-f]{20}$")
    template_text: str = Field(min_length=1)


class CanonicalizationResult(StrictModel):
    records: tuple[CanonicalRecord, ...]
    missing_by_field: dict[str, int]
    invalid_record_count: int = Field(ge=0)
    sensitive_pattern_counts: dict[str, int]
    records_with_sensitive_patterns: int = Field(ge=0)


class RecordGroup(StrictModel):
    group_id: str = Field(pattern=r"^group-[0-9a-f]{20}$")
    records: tuple[CanonicalRecord, ...]
    label_conflict: bool


class GroupingResult(StrictModel):
    groups: tuple[RecordGroup, ...]
    near_duplicate_pair_count: int = Field(ge=0)
    near_duplicate_group_ids: tuple[str, ...]
    inferred_template_groups: int = Field(ge=0)
    repeated_template_groups: int = Field(ge=0)


class GroupedRecord(StrictModel):
    record: CanonicalRecord
    group_id: str = Field(pattern=r"^group-[0-9a-f]{20}$")


class CuratedSplits(StrictModel):
    train: tuple[GroupedRecord, ...]
    validation: tuple[GroupedRecord, ...]
    test: tuple[GroupedRecord, ...]
    conflicting_group_count: int = Field(ge=0)
    excluded_conflicting_records: int = Field(ge=0)


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(StrictModel):
    role: MessageRole
    content: str = Field(min_length=1)


class Difficulty(StrEnum):
    EASY = "easy"
    STANDARD = "standard"
    HARD = "hard"


class TrainingTarget(StrictModel):
    intent: str = Field(min_length=1)
    category: str = Field(min_length=1)
    requires_escalation: bool
    response: str = Field(min_length=1)


class ChatMetadata(StrictModel):
    intent: str = Field(min_length=1)
    category: str = Field(min_length=1)
    split_group: str = Field(pattern=r"^group-[0-9a-f]{20}$")
    flags: tuple[str, ...]
    difficulty: Difficulty


class ChatExample(StrictModel):
    """Portable chat JSONL record consumable by MLX-LM or future trainers."""

    example_id: str = Field(pattern=r"^bitext-[0-9a-f]{24}$")
    source_dataset: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    messages: tuple[ChatMessage, ChatMessage, ChatMessage]
    metadata: ChatMetadata

    @model_validator(mode="after")
    def _validate_chat_order(self) -> ChatExample:
        expected = (
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        )
        actual = tuple(message.role for message in self.messages)
        if actual != expected:
            raise ValueError("messages must be ordered system, user, assistant")
        target = TrainingTarget.model_validate_json(self.messages[-1].content)
        if target.intent != self.metadata.intent:
            raise ValueError("assistant intent must match metadata intent")
        if target.category != self.metadata.category:
            raise ValueError("assistant category must match metadata category")
        return self


class PreparationConfig(StrictModel):
    """Versioned settings for deterministic curation and splitting."""

    seed: int = 42
    train_per_intent: int = Field(default=40, ge=0)
    validation_per_intent: int = Field(default=10, ge=0)
    test_per_intent: int = Field(default=10, ge=0)
    expected_intent_count: int = Field(default=27, ge=1)
    near_duplicate_threshold: float = Field(default=0.9, ge=0.8, le=1.0)
    source_dataset: str = "bitext-customer-support"
    source_version: str = "local-unversioned"
    source_provider: str = "kaggle"
    source_owner: str = "bitext"
    source_url: str = (
        "https://www.kaggle.com/datasets/bitext/"
        "bitext-gen-ai-chatbot-customer-support-dataset"
    )
    source_license: str = "unverified"
    date_accessed: str = "not-recorded"
    processing_config_path: str = "inline"
    output_version: str = "1.0.0"
    split_strategy: str = "intent-balanced-grouped-exact-template-near-v1"
    system_prompt: str = (
        "Classify the customer request and return valid JSON only with intent, "
        "category, requires_escalation, and response fields."
    )

    @model_validator(mode="after")
    def _require_a_split(self) -> PreparationConfig:
        if not (
            self.train_per_intent + self.validation_per_intent + self.test_per_intent
        ):
            raise ValueError("at least one per-intent split target must be positive")
        return self


class LengthSummary(StrictModel):
    minimum: int = Field(ge=0)
    maximum: int = Field(ge=0)
    mean: float = Field(ge=0)
    median: float = Field(ge=0)
    p95: float = Field(ge=0)


class QualityReport(StrictModel):
    schema_version: str
    source_records: int = Field(ge=0)
    valid_records: int = Field(ge=0)
    unique_records: int = Field(ge=0)
    curated_records: int = Field(ge=0)
    invalid_record_count: int = Field(ge=0)
    missing_by_field: dict[str, int]
    exact_duplicate_count: int = Field(ge=0)
    exact_duplicate_rate: float = Field(ge=0, le=1)
    inferred_template_groups: int = Field(ge=0)
    repeated_template_groups: int = Field(ge=0)
    near_duplicate_pairs: int = Field(ge=0)
    near_duplicate_clusters: int = Field(ge=0)
    conflicting_group_count: int = Field(ge=0)
    excluded_conflicting_records: int = Field(ge=0)
    intent_distribution: dict[str, int]
    category_distribution: dict[str, int]
    split_intent_distribution: dict[str, dict[str, int]]
    flag_distribution: dict[str, int]
    difficulty_distribution: dict[str, int]
    instruction_characters: LengthSummary
    response_characters: LengthSummary
    instruction_words: LengthSummary
    response_words: LengthSummary
    sensitive_pattern_counts: dict[str, int]
    records_with_sensitive_patterns: int = Field(ge=0)
    warnings: tuple[str, ...]


class FileDigest(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class DatasetProvenance(StrictModel):
    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    url: str = Field(min_length=1)
    version: str = Field(min_length=1)
    license: str = Field(min_length=1)
    date_accessed: str = Field(min_length=1)


class SplitDescriptor(StrictModel):
    logical_name: str
    path: str
    record_count: int = Field(ge=0)
    record_ids: tuple[str, ...]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen: bool


class DatasetManifest(StrictModel):
    schema_version: str
    dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset: DatasetProvenance
    source: FileDigest
    raw_files: tuple[FileDigest, ...]
    source_member: str | None
    processing: PreparationConfig
    code_revision: str = Field(min_length=1)
    processing_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processing_config_path: str = Field(min_length=1)
    output_version: str = Field(min_length=1)
    split_strategy: str = Field(min_length=1)
    split_seed: int
    policy_versions: dict[str, str]
    artifacts: dict[str, FileDigest]
    splits: dict[str, SplitDescriptor]


class ManifestVerification(StrictModel):
    valid: bool
    checked_files: int = Field(ge=0)
    mismatches: tuple[str, ...]


class LeakageKind(StrEnum):
    EXACT_DUPLICATE = "exact_duplicate"
    TEMPLATE_DUPLICATE = "template_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    SOURCE_GROUP_OVERLAP = "source_group_overlap"
    TARGET_IN_PROMPT = "target_in_prompt"
    FEW_SHOT_TEST_OVERLAP = "few_shot_test_overlap"
    PROMPT_TUNING_TEST_OVERLAP = "prompt_tuning_test_overlap"


class LeakageFinding(StrictModel):
    kind: LeakageKind
    split_left: str
    split_right: str | None
    example_id_left: str
    example_id_right: str | None
    similarity: float | None = Field(default=None, ge=0, le=1)
    detail: str


class LeakageReport(StrictModel):
    passed: bool
    counts: dict[str, int]
    findings: tuple[LeakageFinding, ...]
    truncated: bool


class PreparationResult(StrictModel):
    output_dir: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    quality_report_path: Path
    leakage_report_path: Path
    manifest_path: Path
    manifest: DatasetManifest
    quality_report: QualityReport
    leakage_report: LeakageReport
