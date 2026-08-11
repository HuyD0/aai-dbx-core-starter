"""Strict, immutable domain contracts for the AI Platform Hub."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
SnakeCaseKey = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-fA-F]{7,64}$")]
RowVersion = Annotated[int, Field(ge=1)]
Timestamp = AwareDatetime


class HubModel(BaseModel):
    """Validation defaults for untrusted input and persisted Hub evidence."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )


class Role(StrEnum):
    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    OWNER = "owner"
    PLATFORM_VIEWER = "platform_viewer"
    PLATFORM_ADMINISTRATOR = "platform_administrator"
    AUDITOR = "auditor"


class PrincipalType(StrEnum):
    USER = "USER"
    GROUP = "GROUP"


class HealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class DeploymentStatus(StrEnum):
    RUNNING = "RUNNING"
    DEPLOYING = "DEPLOYING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class IncidentSeverity(StrEnum):
    NONE = "NONE"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


class CostAttributionClass(StrEnum):
    DIRECT = "DIRECT"
    APPLICATION = "APPLICATION"
    ALLOCATED = "ALLOCATED"
    UNATTRIBUTED = "UNATTRIBUTED"


class CostAttributionQuality(StrEnum):
    EXACT = "EXACT"
    ESTIMATED = "ESTIMATED"
    UNKNOWN = "UNKNOWN"


class ReadinessSeverity(StrEnum):
    BLOCKING = "BLOCKING"
    WARNING = "WARNING"
    INFORMATIONAL = "INFORMATIONAL"


class ReadinessStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EvaluationStatus(StrEnum):
    REQUESTED = "REQUESTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def active(self) -> bool:
        return self in {
            EvaluationStatus.REQUESTED,
            EvaluationStatus.QUEUED,
            EvaluationStatus.RUNNING,
        }


class PromotionStatus(StrEnum):
    PENDING_REVIEW = "PENDING_REVIEW"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    REJECTED = "REJECTED"
    APPROVED = "APPROVED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def active(self) -> bool:
        return self in {
            PromotionStatus.PENDING_REVIEW,
            PromotionStatus.CHANGES_REQUESTED,
            PromotionStatus.APPROVED,
            PromotionStatus.EXECUTING,
        }


class ActionEntityType(StrEnum):
    APPLICATION = "APPLICATION"
    EVALUATION = "EVALUATION"
    PROMOTION = "PROMOTION"
    OPTIMIZATION_RECOMMENDATION = "OPTIMIZATION_RECOMMENDATION"


class ActionEventType(StrEnum):
    APPLICATION_REGISTERED = "APPLICATION_REGISTERED"
    EVALUATION_REQUESTED = "EVALUATION_REQUESTED"
    EVALUATION_STATUS_CHANGED = "EVALUATION_STATUS_CHANGED"
    PROMOTION_REQUESTED = "PROMOTION_REQUESTED"
    PROMOTION_APPROVED = "PROMOTION_APPROVED"
    PROMOTION_REJECTED = "PROMOTION_REJECTED"
    PROMOTION_CHANGES_REQUESTED = "PROMOTION_CHANGES_REQUESTED"
    PROMOTION_EXECUTION_STARTED = "PROMOTION_EXECUTION_STARTED"
    PROMOTION_SUCCEEDED = "PROMOTION_SUCCEEDED"
    PROMOTION_FAILED = "PROMOTION_FAILED"
    PROMOTION_CANCELLED = "PROMOTION_CANCELLED"


class Tag(HubModel):
    key: SnakeCaseKey
    value: NonEmptyStr


class AuthorizationContext(HubModel):
    """Identity asserted by a trusted delivery adapter, never by a request body."""

    principal: NonEmptyStr
    groups: tuple[NonEmptyStr, ...] = ()
    platform_roles: tuple[Role, ...] = ()

    @field_validator("groups", "platform_roles")
    @classmethod
    def unique_values(cls, value: tuple) -> tuple:
        if len(value) != len(set(value)):
            raise ValueError("values must be unique")
        return value

    def has_platform_role(self, *roles: Role) -> bool:
        return bool(set(self.platform_roles).intersection(roles))


class CostRecord(HubModel):
    amount: Annotated[Decimal, Field(ge=0)]
    currency: Annotated[
        str,
        StringConstraints(strip_whitespace=True, pattern=r"^[A-Z]{3}$"),
    ]
    attribution_class: CostAttributionClass
    attribution_quality: CostAttributionQuality
    source: NonEmptyStr
    freshness_at: Timestamp


class HealthProfile(HubModel):
    profile_id: NonEmptyStr
    version: NonEmptyStr
    maximum_error_rate: Annotated[float, Field(ge=0, le=1)]
    critical_error_rate: Annotated[float, Field(ge=0, le=1)]
    maximum_p95_latency_ms: Annotated[float, Field(gt=0)]
    critical_p95_latency_ms: Annotated[float, Field(gt=0)]
    maximum_telemetry_age_seconds: Annotated[int, Field(gt=0)] = 3600
    evaluation_failure_affects_health: bool = False

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> HealthProfile:
        if self.critical_error_rate < self.maximum_error_rate:
            raise ValueError("critical_error_rate must be at least maximum_error_rate")
        if self.critical_p95_latency_ms < self.maximum_p95_latency_ms:
            raise ValueError(
                "critical_p95_latency_ms must be at least maximum_p95_latency_ms"
            )
        return self


class HealthEvidence(HubModel):
    application_id: NonEmptyStr
    environment: NonEmptyStr
    deployment_status: DeploymentStatus
    deployment_expected_active: bool = True
    telemetry_observed_at: Timestamp | None = None
    request_count: Annotated[int, Field(ge=0)] | None = None
    error_count: Annotated[int, Field(ge=0)] | None = None
    p95_latency_ms: Annotated[float, Field(ge=0)] | None = None
    last_successful_trace_at: Timestamp | None = None
    latest_evaluation_passed: bool | None = None
    incident_severity: IncidentSeverity = IncidentSeverity.NONE

    @model_validator(mode="after")
    def counts_are_consistent(self) -> HealthEvidence:
        if self.error_count is not None and self.request_count is None:
            raise ValueError("error_count requires request_count")
        if (
            self.error_count is not None
            and self.request_count is not None
            and self.error_count > self.request_count
        ):
            raise ValueError("error_count cannot exceed request_count")
        return self


class HealthSnapshot(HubModel):
    application_id: NonEmptyStr
    environment: NonEmptyStr
    profile_id: NonEmptyStr
    profile_version: NonEmptyStr
    status: HealthStatus
    reasons: tuple[NonEmptyStr, ...]
    evidence_at: Timestamp | None
    evaluated_at: Timestamp


class ReadinessRuleResult(HubModel):
    rule_id: SnakeCaseKey
    rule_version: NonEmptyStr
    description: NonEmptyStr
    severity: ReadinessSeverity
    status: ReadinessStatus
    evidence: tuple[NonEmptyStr, ...]
    evaluated_at: Timestamp
    remediation: NonEmptyStr | None = None


class ReadinessSnapshot(HubModel):
    application_id: NonEmptyStr
    environment: NonEmptyStr
    application_version_id: NonEmptyStr
    profile_id: NonEmptyStr
    profile_version: NonEmptyStr
    evaluated_at: Timestamp
    ready: bool
    results: tuple[ReadinessRuleResult, ...]

    @model_validator(mode="after")
    def ready_matches_blocking_results(self) -> ReadinessSnapshot:
        blocked = any(
            result.severity is ReadinessSeverity.BLOCKING
            and result.status in {ReadinessStatus.FAIL, ReadinessStatus.UNKNOWN}
            for result in self.results
        )
        if self.ready == blocked:
            raise ValueError(
                "ready must be false when a blocking rule fails or is unknown"
            )
        return self

    def decision_signature(self) -> str:
        """Return a timestamp-independent signature of promotion evidence."""

        payload = {
            "application_id": self.application_id,
            "environment": self.environment,
            "application_version_id": self.application_version_id,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "ready": self.ready,
            "results": [
                {
                    "rule_id": result.rule_id,
                    "rule_version": result.rule_version,
                    "description": result.description,
                    "severity": result.severity.value,
                    "status": result.status.value,
                    "evidence": list(result.evidence),
                    "remediation": result.remediation,
                }
                for result in sorted(
                    self.results,
                    key=lambda item: (item.rule_id, item.rule_version),
                )
            ],
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


class ApplicationRecord(HubModel):
    application_id: NonEmptyStr
    name: NonEmptyStr
    description: str = ""
    owner_principal: NonEmptyStr
    support_group: NonEmptyStr
    business_domain: NonEmptyStr
    cost_center: NonEmptyStr
    risk_tier: NonEmptyStr
    lifecycle_state: NonEmptyStr
    tags: tuple[Tag, ...] = ()
    created_at: Timestamp
    updated_at: Timestamp
    row_version: RowVersion = 1

    @field_validator("tags")
    @classmethod
    def unique_tag_keys(cls, value: tuple[Tag, ...]) -> tuple[Tag, ...]:
        keys = [tag.key for tag in value]
        if len(keys) != len(set(keys)):
            raise ValueError("tag keys must be unique")
        return value

    @model_validator(mode="after")
    def timestamps_are_ordered(self) -> ApplicationRecord:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        return self


class ApplicationVersionRecord(HubModel):
    version_id: NonEmptyStr
    application_id: NonEmptyStr
    environment: NonEmptyStr
    git_repository: NonEmptyStr
    git_commit_sha: GitSha
    manifest_version: NonEmptyStr
    manifest_hash: Sha256
    manifest_json: NonEmptyStr
    original_manifest_json: NonEmptyStr | None = None
    registered_by: NonEmptyStr
    registered_at: Timestamp
    deployment_target: NonEmptyStr
    is_current: bool = True

    @field_validator("manifest_json", "original_manifest_json")
    @classmethod
    def manifest_is_a_json_object(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            document = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("manifest_json must be valid JSON") from error
        if not isinstance(document, dict):
            raise ValueError("manifest_json must encode a JSON object")
        return value


class ApplicationPrincipalRecord(HubModel):
    application_id: NonEmptyStr
    principal_type: PrincipalType
    principal_name: NonEmptyStr
    application_role: Role

    @model_validator(mode="after")
    def role_is_application_scoped(self) -> ApplicationPrincipalRecord:
        if self.application_role in {
            Role.PLATFORM_VIEWER,
            Role.PLATFORM_ADMINISTRATOR,
            Role.AUDITOR,
        }:
            raise ValueError("platform roles cannot be assigned to one application")
        return self


class ResourceBindingRecord(HubModel):
    binding_id: NonEmptyStr
    application_id: NonEmptyStr
    environment: NonEmptyStr
    resource_type: NonEmptyStr
    resource_id: NonEmptyStr
    resource_name: NonEmptyStr
    workspace_id: NonEmptyStr
    tags: tuple[Tag, ...] = ()
    discovered_at: Timestamp
    last_verified_at: Timestamp | None = None

    @model_validator(mode="after")
    def verification_follows_discovery(self) -> ResourceBindingRecord:
        if (
            self.last_verified_at is not None
            and self.last_verified_at < self.discovered_at
        ):
            raise ValueError("last_verified_at cannot precede discovered_at")
        return self


class EvaluationSummary(HubModel):
    """Bounded, sanitized evidence accepted from an evaluation reconciler."""

    dataset_exists: bool | None = None
    dataset_case_count: Annotated[int, Field(ge=0, le=100_000_000)] | None = None
    blocking_thresholds_passed: bool | None = None
    metrics: Mapping[SnakeCaseKey, int | float] = Field(default_factory=dict)

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError("metrics must be an object")
        if len(value) > 50:
            raise ValueError("metrics may contain at most 50 values")
        for metric_value in value.values():
            if isinstance(metric_value, bool) or not isinstance(
                metric_value, int | float
            ):
                raise ValueError("metric values must be numbers")
            if not math.isfinite(float(metric_value)):
                raise ValueError("metric values must be finite")
        return value

    @field_validator("metrics")
    @classmethod
    def freeze_metrics(
        cls, value: Mapping[str, int | float]
    ) -> Mapping[str, int | float]:
        return MappingProxyType(dict(value))

    @field_serializer("metrics")
    def serialize_metrics(
        self, value: Mapping[str, int | float]
    ) -> dict[str, int | float]:
        return dict(value)


class EvaluationRunRecord(HubModel):
    evaluation_run_id: NonEmptyStr
    application_id: NonEmptyStr
    environment: NonEmptyStr
    application_version_id: NonEmptyStr
    evaluation_profile: NonEmptyStr
    dataset_name: NonEmptyStr
    dataset_version: NonEmptyStr
    job_id: NonEmptyStr
    job_run_id: NonEmptyStr | None = None
    mlflow_run_id: NonEmptyStr | None = None
    requested_by: NonEmptyStr
    status: EvaluationStatus
    requested_at: Timestamp
    started_at: Timestamp | None = None
    completed_at: Timestamp | None = None
    summary_json: str | None = None
    failure_message: str | None = None
    row_version: RowVersion = 1

    @field_validator("summary_json")
    @classmethod
    def summary_is_json_object(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value.encode("utf-8")) > 16_384:
            raise ValueError("summary_json must be at most 16 KiB")
        try:
            document = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("summary_json must be valid JSON") from error
        if not isinstance(document, dict):
            raise ValueError("summary_json must encode a JSON object")
        summary = EvaluationSummary.model_validate(document)
        return json.dumps(
            summary.model_dump(mode="json", exclude_none=True),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @model_validator(mode="after")
    def timestamps_and_terminal_state_are_consistent(self) -> EvaluationRunRecord:
        if self.started_at is not None and self.started_at < self.requested_at:
            raise ValueError("started_at cannot precede requested_at")
        if self.completed_at is not None:
            if self.completed_at < self.requested_at:
                raise ValueError("completed_at cannot precede requested_at")
            if self.started_at is not None and self.completed_at < self.started_at:
                raise ValueError("completed_at cannot precede started_at")
        terminal = self.status in {
            EvaluationStatus.SUCCEEDED,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        }
        if terminal != (self.completed_at is not None):
            raise ValueError("terminal evaluation status requires completed_at")
        if self.status is EvaluationStatus.FAILED and not self.failure_message:
            raise ValueError("failed evaluation requires failure_message")
        return self


class PromotionRequestRecord(HubModel):
    promotion_request_id: NonEmptyStr
    application_id: NonEmptyStr
    source_environment: NonEmptyStr
    target_environment: NonEmptyStr
    application_version_id: NonEmptyStr
    requested_by: NonEmptyStr
    requested_at: Timestamp
    status: PromotionStatus
    # Immutable request-time evidence. Approval revalidation is stored separately so
    # the decision remains reproducible even when readiness changes between the two.
    readiness_snapshot: ReadinessSnapshot
    approval_readiness_snapshot: ReadinessSnapshot | None = None
    reviewed_by: NonEmptyStr | None = None
    reviewed_at: Timestamp | None = None
    review_comment: str | None = None
    promotion_job_id: NonEmptyStr
    promotion_job_run_id: NonEmptyStr | None = None
    row_version: RowVersion = 1

    @model_validator(mode="after")
    def request_is_consistent(self) -> PromotionRequestRecord:
        if self.source_environment == self.target_environment:
            raise ValueError("source and target environments must differ")
        if self.readiness_snapshot.application_id != self.application_id:
            raise ValueError("readiness snapshot belongs to another application")
        if (
            self.readiness_snapshot.application_version_id
            != self.application_version_id
        ):
            raise ValueError("readiness snapshot belongs to another version")
        if self.approval_readiness_snapshot is not None and (
            self.approval_readiness_snapshot.application_id != self.application_id
            or self.approval_readiness_snapshot.application_version_id
            != self.application_version_id
        ):
            raise ValueError(
                "approval readiness snapshot belongs to another application version"
            )
        if self.reviewed_at is not None and self.reviewed_at < self.requested_at:
            raise ValueError("reviewed_at cannot precede requested_at")
        if (self.reviewed_by is None) != (self.reviewed_at is None):
            raise ValueError("reviewed_by and reviewed_at must be set together")
        if (
            self.status
            in {
                PromotionStatus.APPROVED,
                PromotionStatus.EXECUTING,
                PromotionStatus.SUCCEEDED,
                PromotionStatus.FAILED,
            }
            and self.approval_readiness_snapshot is None
        ):
            raise ValueError(
                "approved or executing promotion requires approval readiness evidence"
            )
        return self


class ActionEvent(HubModel):
    event_id: NonEmptyStr
    entity_type: ActionEntityType
    entity_id: NonEmptyStr
    event_type: ActionEventType
    actor_principal: NonEmptyStr
    actor_request_id: NonEmptyStr
    event_time: Timestamp
    previous_state: str | None = None
    new_state: str | None = None
    comment: str | None = None
    details_json: str = "{}"

    @field_validator("details_json")
    @classmethod
    def details_are_json_object(cls, value: str) -> str:
        try:
            document = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("details_json must be valid JSON") from error
        if not isinstance(document, dict):
            raise ValueError("details_json must encode a JSON object")
        return value


class RegistrationResult(HubModel):
    application: ApplicationRecord
    version: ApplicationVersionRecord
    created: bool
