"""Offline data preparation and leakage-audit API."""

from .bitext import BITEXT_COLUMNS, load_bitext
from .duplicates import group_related_records, text_similarity
from .leakage import (
    DatasetLeakageError,
    assert_no_leakage,
    check_split_files,
    check_split_leakage,
    load_chat_jsonl,
)
from .manifests import sha256_file, verify_manifest
from .pipeline import audit_dataset, prepare_dataset, processing_source_sha256
from .policies import (
    DIFFICULTY_POLICY_VERSION,
    ESCALATION_POLICY_VERSION,
    ESCALATION_RULES,
    RESPONSE_POLICY_VERSION,
    classify_difficulty,
    parse_flags,
    render_training_response,
    requires_escalation,
)
from .schemas import (
    ChatExample,
    ChatMessage,
    ChatMetadata,
    DatasetManifest,
    DatasetProvenance,
    Difficulty,
    LeakageFinding,
    LeakageKind,
    LeakageReport,
    ManifestVerification,
    MessageRole,
    PreparationConfig,
    PreparationResult,
    QualityReport,
    TrainingTarget,
)
from .token_analysis import summarize_instruction_tokens

__all__ = [
    "BITEXT_COLUMNS",
    "ChatExample",
    "ChatMessage",
    "ChatMetadata",
    "DatasetLeakageError",
    "DatasetManifest",
    "DatasetProvenance",
    "Difficulty",
    "DIFFICULTY_POLICY_VERSION",
    "ESCALATION_POLICY_VERSION",
    "ESCALATION_RULES",
    "RESPONSE_POLICY_VERSION",
    "LeakageFinding",
    "LeakageKind",
    "LeakageReport",
    "ManifestVerification",
    "MessageRole",
    "PreparationConfig",
    "PreparationResult",
    "QualityReport",
    "TrainingTarget",
    "assert_no_leakage",
    "audit_dataset",
    "check_split_files",
    "check_split_leakage",
    "classify_difficulty",
    "group_related_records",
    "load_bitext",
    "load_chat_jsonl",
    "parse_flags",
    "prepare_dataset",
    "processing_source_sha256",
    "render_training_response",
    "sha256_file",
    "summarize_instruction_tokens",
    "text_similarity",
    "requires_escalation",
    "verify_manifest",
]
