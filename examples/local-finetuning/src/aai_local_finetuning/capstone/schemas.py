"""Versioned, strict schemas for the production-readiness capstone.

The policy engine accepts an untrusted JSON mapping so it can explain malformed
manifests.  Callers that need a validated manifest should use
``ApplicationManifest.model_validate_json``.  The latter distinction is
intentional: invalid lifecycle values and unknown fields are useful frozen-test
cases, but they are not valid ``ApplicationManifest`` instances.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_SCHEMA_VERSION = "1.0.0"
OUTPUT_SCHEMA_VERSION = "1.0.0"
POLICY_VERSION = "1.0.0"
DATASET_SCHEMA_VERSION = "1.0.0"
DATASET_VERSION = "1.0.0"


class StrictFrozenModel(BaseModel):
    """Common boundary contract used by all persisted capstone evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Lifecycle(StrEnum):
    """Repository-supported lifecycle values.

    ``candidate`` is deliberately absent.  It is legacy platform terminology,
    not an application lifecycle stage.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class RuleKind(StrEnum):
    DETERMINISTIC = "deterministic"
    POLICY = "policy"
    EXTERNAL_LOOKUP = "external_lookup"
    HUMAN_JUDGMENT = "human_judgment"


class CheckOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReadinessStatus(StrEnum):
    READY = "ready"
    NOT_READY = "not_ready"
    REVIEW_REQUIRED = "review_required"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ApplicationManifest(StrictFrozenModel):
    """A simplified, versioned application manifest.

    Optional fields model incomplete submissions.  Omitting one is valid JSON
    shape but will normally produce a failed readiness check.  Values arriving
    from JSON are still type-strict; no strings are coerced into booleans or
    numbers.
    """

    schema_version: Literal["1.0.0"]
    application_name: str = Field(min_length=1, max_length=200)
    owner: str | None = Field(default=None, max_length=200)
    business_domain: str | None = Field(default=None, max_length=200)
    lifecycle: Lifecycle | None = None
    model_revision: str | None = Field(default=None, max_length=300)
    model_revision_pinned: bool | None = None
    evaluation_dataset: str | None = Field(default=None, max_length=500)
    evaluation_last_run: datetime | None = None
    evaluation_thresholds_passed: bool | None = None
    cost_tags_present: bool | None = None
    budget_policy: str | None = Field(default=None, max_length=100)
    production_support_owner: str | None = Field(default=None, max_length=200)
    rollback_plan: bool | None = None
    monitoring_configured: bool | None = None
    data_classifications: tuple[str, ...] | None = None
    required_approvals_complete: bool | None = None
    framework: str | None = Field(default=None, max_length=100)
    known_high_severity_findings: int | None = Field(default=None, ge=0)
    external_registry_lookup_required: bool = False
    human_risk_judgment_required: bool = False
    description: str | None = Field(default=None, max_length=10_000)
    declared_controls: tuple[str, ...] = ()


class RuleDefinition(StrictFrozenModel):
    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    kind: RuleKind
    failure_severity: Severity
    source_fields: tuple[str, ...]
    remediation_id: str
    remediation_text: str


class CheckProvenance(StrictFrozenModel):
    policy_version: Literal["1.0.0"]
    rule_id: str
    rule_kind: RuleKind
    source_fields: tuple[str, ...]
    facts_origin: Literal["manifest", "policy", "external_system", "human_review"]


class ReadinessCheck(StrictFrozenModel):
    name: str
    result: CheckOutcome
    severity: Severity
    evidence: str
    remediation_id: str | None = None
    remediation_text: str | None = None
    provenance: CheckProvenance


class ReadinessReview(StrictFrozenModel):
    schema_version: Literal["1.0.0"]
    manifest_schema_version: Literal["1.0.0"]
    policy_version: Literal["1.0.0"]
    status: ReadinessStatus
    as_of: datetime
    checks: tuple[ReadinessCheck, ...]


class CapstoneRecordMetadata(StrictFrozenModel):
    split: DatasetSplit
    slices: tuple[str, ...]
    generator_seed: int
    policy_version: Literal["1.0.0"]


class CapstoneRecord(StrictFrozenModel):
    """Portable logical record consumed by any future training framework."""

    schema_version: Literal["1.0.0"]
    example_id: str = Field(pattern=r"^capstone-[a-f0-9]{16}$")
    source_dataset: Literal["aai-application-production-readiness"]
    source_version: Literal["1.0.0"]
    manifest: dict[str, Any]
    expected_output: ReadinessReview
    metadata: CapstoneRecordMetadata


class SplitArtifact(StrictFrozenModel):
    split: DatasetSplit
    path: str
    record_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    example_ids_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class SplitManifest(StrictFrozenModel):
    schema_version: Literal["1.0.0"]
    dataset_name: Literal["aai-application-production-readiness"]
    dataset_version: Literal["1.0.0"]
    policy_version: Literal["1.0.0"]
    seed: int
    strategy: Literal["controlled_policy_slices"]
    frozen_test: Literal[True]
    artifacts: tuple[SplitArtifact, ...]
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class HybridExplanation(StrictFrozenModel):
    check_name: str
    text: str
    renderer: str


class HybridReview(StrictFrozenModel):
    """A deterministic decision accompanied by optional rendered language."""

    schema_version: Literal["1.0.0"]
    deterministic_review: ReadinessReview
    explanations: tuple[HybridExplanation, ...]
