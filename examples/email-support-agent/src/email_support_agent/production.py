"""Fail-closed production-readiness evidence for the solution accelerator.

This module evaluates already-collected attestations. It does not run a load
test, inspect a cloud resource, validate an identity, or certify a deployment.
Non-synthetic evidence references must be resolved and authorized by the
platform that assembles the evidence pack. The local digest detects accidental
or unreviewed mutation; it is not a replacement for a signed attestation.

The thresholds returned by :func:`illustrative_policy` are examples only. That
policy is deliberately unapproved, so neither it nor synthetic evidence can
produce a production-ready decision.
"""

from __future__ import annotations

import hmac
import json
import re
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from aai_core.contracts import ContractModel

_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
_GROUP = re.compile(r"^group:[A-Za-z0-9._-]{1,64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


class EvidenceOrigin(StrEnum):
    """Bounded vocabulary for the system that produced an attestation."""

    PLATFORM_CONTROL = "platform_control"
    SECURITY_CONTROL = "security_control"
    PERFORMANCE_HARNESS = "performance_harness"
    RESTORE_HARNESS = "restore_harness"
    CANARY_TELEMETRY = "canary_telemetry"
    FINOPS_LEDGER = "finops_ledger"
    SYNTHETIC = "synthetic"


class OperationalCapability(StrEnum):
    """Capabilities required by the connected production architecture."""

    INGRESS_STORE = "ingress_store"
    ACCESS_AUTHORIZER = "access_authorizer"
    DLP_SCANNER = "dlp_scanner"
    MALWARE_SCANNER = "malware_scanner"
    MODEL_GATEWAY = "model_gateway"
    KNOWLEDGE_RETRIEVER = "knowledge_retriever"
    DURABLE_CHECKPOINTER = "durable_checkpointer"
    TRANSACTIONAL_OUTBOX = "transactional_outbox"
    REVIEW_SERVICE = "review_service"
    TELEMETRY_SINK = "telemetry_sink"
    FINOPS_LEDGER = "finops_ledger"


class ReadinessStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    MISSING = "missing"
    STALE = "stale"
    MISMATCH = "mismatch"
    UNVERIFIED = "unverified"
    TAMPERED = "tampered"


class EvidenceProvenance(ContractModel):
    """Release-bound pointer to evidence held by an approved external system."""

    origin: EvidenceOrigin
    application_id: str = Field(min_length=3, max_length=128)
    environment: str = Field(min_length=3, max_length=64)
    evidence_ref: str = Field(min_length=8, max_length=512)
    attestation_digest: str
    target_release_digest: str
    observed_at: AwareDatetime
    valid_until: AwareDatetime

    @field_validator("attestation_digest", "target_release_digest")
    @classmethod
    def require_digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("evidence digests must use sha256:<hex>")
        return value

    @field_validator("application_id", "environment")
    @classmethod
    def require_opaque_identifier(cls, value: str) -> str:
        if not _OPAQUE_ID.fullmatch(value) or "@" in value:
            raise ValueError("evidence provenance identifiers must be opaque")
        return value

    @field_validator("evidence_ref")
    @classmethod
    def require_opaque_reference(cls, value: str) -> str:
        if any(character in value for character in ("@", "?", "#")):
            raise ValueError("evidence_ref must be opaque and query-free")
        return value

    @model_validator(mode="after")
    def validate_reference_and_window(self) -> Self:
        expected_prefix = (
            "synthetic://evidence/"
            if self.origin is EvidenceOrigin.SYNTHETIC
            else "secure://evidence/"
        )
        if not self.evidence_ref.startswith(expected_prefix):
            raise ValueError(f"evidence_ref must start with {expected_prefix}")
        if self.valid_until <= self.observed_at:
            raise ValueError("valid_until must be later than observed_at")
        return self


class IngressVerificationEvidence(ContractModel):
    provenance: EvidenceProvenance
    messages_observed: int = Field(ge=1)
    authenticated_messages: int = Field(ge=0)
    malware_scanned_messages: int = Field(ge=0)
    dlp_processed_messages: int = Field(ge=0)
    access_context_verified_messages: int = Field(ge=0)
    raw_payload_checkpoint_count: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_cannot_exceed_observations(self) -> Self:
        counts = (
            self.authenticated_messages,
            self.malware_scanned_messages,
            self.dlp_processed_messages,
            self.access_context_verified_messages,
        )
        if any(count > self.messages_observed for count in counts):
            raise ValueError("ingress verification count exceeds messages_observed")
        return self


class DependencyObservation(ContractModel):
    capability: OperationalCapability
    configured: bool
    health_check_passed: bool
    authorization_check_passed: bool
    least_privilege_reviewed: bool
    failure_policy_configured: bool


class OperationalEvidence(ContractModel):
    provenance: EvidenceProvenance
    dependencies: tuple[DependencyObservation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def capabilities_are_unique(self) -> Self:
        capabilities = [item.capability for item in self.dependencies]
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("dependency capabilities must be unique")
        return self


class ReleaseComponents(ContractModel):
    application_release: str = Field(min_length=1, max_length=128)
    git_commit: str
    artifact_digest: str
    prompt_release: str = Field(min_length=1, max_length=128)
    model_release: str = Field(min_length=1, max_length=256)
    tool_release: str = Field(min_length=1, max_length=128)
    knowledge_release: str = Field(min_length=1, max_length=128)
    embedding_release: str = Field(min_length=1, max_length=128)
    chunking_release: str = Field(min_length=1, max_length=128)
    policy_release: str = Field(min_length=1, max_length=128)
    schema_release: str = Field(min_length=1, max_length=128)

    @field_validator("git_commit")
    @classmethod
    def require_git_sha(cls, value: str) -> str:
        if not _GIT_SHA.fullmatch(value):
            raise ValueError("git_commit must be a full lowercase Git SHA")
        return value

    @field_validator("artifact_digest")
    @classmethod
    def require_artifact_digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("artifact_digest must use sha256:<hex>")
        return value


class ReleaseLineageEvidence(ContractModel):
    provenance: EvidenceProvenance
    components: ReleaseComponents
    declared_release_digest: str
    evaluated_release_digest: str
    deployed_release_digest: str
    canary_release_digest: str

    @field_validator(
        "declared_release_digest",
        "evaluated_release_digest",
        "deployed_release_digest",
        "canary_release_digest",
    )
    @classmethod
    def require_release_digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("release lineage fields must use sha256:<hex>")
        return value


class SloTargets(ContractModel):
    minimum_window_seconds: int = Field(default=3_600, ge=60)
    minimum_request_count: int = Field(default=1_000, ge=1)
    maximum_error_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    minimum_availability: float = Field(default=0.999, ge=0.0, le=1.0)
    maximum_p95_latency_ms: float = Field(default=4_000.0, gt=0.0)
    maximum_p95_review_wait_seconds: float = Field(default=3_600.0, gt=0.0)


class SloObservationEvidence(ContractModel):
    provenance: EvidenceProvenance
    window_seconds: int = Field(ge=1)
    request_count: int = Field(ge=1)
    error_count: int = Field(ge=0)
    available_request_count: int = Field(ge=0)
    p95_latency_ms: float = Field(ge=0.0)
    p95_review_wait_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.error_count > self.request_count:
            raise ValueError("error_count exceeds request_count")
        if self.available_request_count > self.request_count:
            raise ValueError("available_request_count exceeds request_count")
        return self


class ReliabilityEvidence(ContractModel):
    provenance: EvidenceProvenance
    scenarios_executed: int = Field(ge=1)
    scenarios_passed: int = Field(ge=0)
    delivery_attempts: int = Field(ge=1)
    successful_deliveries: int = Field(ge=0)
    idempotency_replay_cases: int = Field(ge=1)
    idempotency_replay_passes: int = Field(ge=0)
    duplicate_send_count: int = Field(ge=0)
    checkpoint_resume_passed: bool
    outbox_recovery_passed: bool
    retry_budget_enforced: bool
    dead_letter_recovery_passed: bool

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.scenarios_passed > self.scenarios_executed:
            raise ValueError("scenarios_passed exceeds scenarios_executed")
        if self.successful_deliveries > self.delivery_attempts:
            raise ValueError("successful_deliveries exceeds delivery_attempts")
        if self.idempotency_replay_passes > self.idempotency_replay_cases:
            raise ValueError("idempotency replay passes exceed replay cases")
        return self


class SecurityEvidence(ContractModel):
    provenance: EvidenceProvenance
    security_test_cases: int = Field(ge=1)
    cross_tenant_test_cases: int = Field(ge=1)
    unauthorized_send_test_cases: int = Field(ge=1)
    high_risk_routing_cases: int = Field(ge=1)
    cross_tenant_leakage_count: int = Field(ge=0)
    unauthorized_send_count: int = Field(ge=0)
    high_risk_false_auto_send_count: int = Field(ge=0)
    dlp_admission_passed: bool
    trusted_access_context_passed: bool
    reviewer_authorization_passed: bool
    trace_policy_passed: bool
    prompt_injection_precheck_passed: bool
    least_privilege_review_passed: bool

    @model_validator(mode="after")
    def validate_violation_counts(self) -> Self:
        bounds = (
            (self.cross_tenant_leakage_count, self.cross_tenant_test_cases),
            (self.unauthorized_send_count, self.unauthorized_send_test_cases),
            (
                self.high_risk_false_auto_send_count,
                self.high_risk_routing_cases,
            ),
        )
        if any(violations > cases for violations, cases in bounds):
            raise ValueError("security violation count exceeds tested cases")
        return self


class LoadTargets(ContractModel):
    minimum_duration_seconds: int = Field(default=900, ge=60)
    minimum_request_count: int = Field(default=5_000, ge=1)
    minimum_target_rps: float = Field(default=10.0, gt=0.0)
    maximum_error_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    maximum_p95_latency_ms: float = Field(default=4_000.0, gt=0.0)
    maximum_resource_saturation: float = Field(default=0.80, gt=0.0, le=1.0)


class LoadEvidence(ContractModel):
    provenance: EvidenceProvenance
    duration_seconds: int = Field(ge=1)
    request_count: int = Field(ge=1)
    error_count: int = Field(ge=0)
    target_rps: float = Field(gt=0.0)
    achieved_rps: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    peak_resource_saturation: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_error_count(self) -> Self:
        if self.error_count > self.request_count:
            raise ValueError("load error_count exceeds request_count")
        return self


class RestoreTargets(ContractModel):
    maximum_rto_seconds: float = Field(default=1_800.0, gt=0.0)
    maximum_rpo_seconds: float = Field(default=300.0, ge=0.0)


class RestoreEvidence(ContractModel):
    provenance: EvidenceProvenance
    restored_release_digest: str
    restore_succeeded: bool
    observed_rto_seconds: float = Field(ge=0.0)
    observed_rpo_seconds: float = Field(ge=0.0)
    checkpoint_integrity_passed: bool
    outbox_reconciled: bool
    duplicate_send_count: int = Field(ge=0)

    @field_validator("restored_release_digest")
    @classmethod
    def require_release_digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("restored_release_digest must use sha256:<hex>")
        return value


class CanaryTargets(ContractModel):
    minimum_duration_seconds: int = Field(default=1_800, ge=60)
    minimum_request_count: int = Field(default=500, ge=1)
    minimum_traffic_fraction: float = Field(default=0.01, gt=0.0, le=1.0)
    maximum_traffic_fraction: float = Field(default=0.10, gt=0.0, le=1.0)
    maximum_error_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    maximum_p95_latency_ms: float = Field(default=4_000.0, gt=0.0)
    minimum_trace_coverage: float = Field(default=0.99, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_traffic_bounds(self) -> Self:
        if self.minimum_traffic_fraction > self.maximum_traffic_fraction:
            raise ValueError("minimum canary traffic exceeds maximum traffic")
        return self


class CanaryEvidence(ContractModel):
    provenance: EvidenceProvenance
    duration_seconds: int = Field(ge=1)
    request_count: int = Field(ge=1)
    error_count: int = Field(ge=0)
    traffic_fraction: float = Field(gt=0.0, le=1.0)
    p95_latency_ms: float = Field(ge=0.0)
    traced_request_count: int = Field(ge=0)
    rollback_ready: bool
    cross_tenant_leakage_count: int = Field(ge=0)
    unauthorized_send_count: int = Field(ge=0)
    duplicate_send_count: int = Field(ge=0)
    high_risk_false_auto_send_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        counts = (
            self.error_count,
            self.traced_request_count,
            self.cross_tenant_leakage_count,
            self.unauthorized_send_count,
            self.duplicate_send_count,
            self.high_risk_false_auto_send_count,
        )
        if any(count > self.request_count for count in counts):
            raise ValueError("canary count exceeds request_count")
        return self


class CostTargets(ContractModel):
    minimum_request_count: int = Field(default=500, ge=1)
    minimum_cost_coverage: float = Field(default=0.99, ge=0.0, le=1.0)
    maximum_cost_per_request_usd: float = Field(default=0.10, gt=0.0)
    maximum_cost_per_resolved_case_usd: float = Field(default=0.50, gt=0.0)
    maximum_p95_model_calls: float = Field(default=2.0, ge=0.0)


class CostEvidence(ContractModel):
    provenance: EvidenceProvenance
    requests_observed: int = Field(ge=1)
    requests_costed: int = Field(ge=0)
    resolved_cases: int = Field(ge=0)
    inference_cost_usd: float = Field(ge=0.0)
    operational_cost_usd: float = Field(ge=0.0)
    p95_model_calls: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.requests_costed > self.requests_observed:
            raise ValueError("requests_costed exceeds requests_observed")
        if self.resolved_cases > self.requests_observed:
            raise ValueError("resolved_cases exceeds requests_observed")
        return self

    @property
    def coverage(self) -> float:
        return self.requests_costed / self.requests_observed

    @property
    def total_cost_usd(self) -> float:
        return self.inference_cost_usd + self.operational_cost_usd

    @property
    def cost_per_request_usd(self) -> float:
        return self.total_cost_usd / self.requests_observed

    @property
    def cost_per_resolved_case_usd(self) -> float | None:
        if self.resolved_cases == 0:
            return None
        return self.total_cost_usd / self.resolved_cases


_DEFAULT_CAPABILITIES = tuple(OperationalCapability)


class ProductionReadinessPolicy(ContractModel):
    """Owner-controlled policy; defaults are illustrative, not approved."""

    application_id: str = Field(min_length=3, max_length=128)
    environment: str = Field(min_length=3, max_length=64)
    profile_id: str = Field(min_length=3, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    owner_group: str
    approved_for_production: bool = False
    approval_ref: str | None = None
    maximum_evidence_age_hours: int = Field(default=24, ge=1)
    minimum_ingress_samples: int = Field(default=1_000, ge=1)
    minimum_ingress_verification_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    required_capabilities: tuple[OperationalCapability, ...] = Field(
        default=_DEFAULT_CAPABILITIES,
        min_length=1,
    )
    minimum_reliability_scenarios: int = Field(default=5, ge=1)
    minimum_delivery_success_rate: float = Field(default=0.995, ge=0.0, le=1.0)
    minimum_security_test_cases: int = Field(default=100, ge=1)
    slo: SloTargets = Field(default_factory=SloTargets)
    load: LoadTargets = Field(default_factory=LoadTargets)
    restore: RestoreTargets = Field(default_factory=RestoreTargets)
    canary: CanaryTargets = Field(default_factory=CanaryTargets)
    cost: CostTargets = Field(default_factory=CostTargets)

    @field_validator("application_id", "environment", "profile_id")
    @classmethod
    def require_opaque_identifier(cls, value: str) -> str:
        if not _OPAQUE_ID.fullmatch(value) or "@" in value:
            raise ValueError("policy identifiers must be opaque")
        return value

    @field_validator("version")
    @classmethod
    def require_version(cls, value: str) -> str:
        if not _VERSION.fullmatch(value):
            raise ValueError("policy version must be an opaque version identifier")
        return value

    @field_validator("owner_group")
    @classmethod
    def require_owner_group(cls, value: str) -> str:
        if not _GROUP.fullmatch(value):
            raise ValueError("owner_group must use group:<identifier>")
        return value

    @field_validator("approval_ref")
    @classmethod
    def require_approval_reference(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.startswith("secure://policy-approval/")
            or any(character in value for character in ("@", "?", "#"))
        ):
            raise ValueError("approval_ref must be an opaque secure reference")
        return value

    @model_validator(mode="after")
    def validate_approval_and_capabilities(self) -> Self:
        if self.approved_for_production and self.approval_ref is None:
            raise ValueError("approved policy requires approval_ref")
        if len(self.required_capabilities) != len(set(self.required_capabilities)):
            raise ValueError("required_capabilities must be unique")
        return self


class ProductionEvidencePack(ContractModel):
    """One immutable, release-bound snapshot assembled from external evidence."""

    application_id: str = Field(min_length=3, max_length=128)
    environment: str = Field(min_length=3, max_length=64)
    target_release_digest: str
    assembled_at: AwareDatetime
    ingress: IngressVerificationEvidence | None = None
    operations: OperationalEvidence | None = None
    lineage: ReleaseLineageEvidence | None = None
    slo: SloObservationEvidence | None = None
    reliability: ReliabilityEvidence | None = None
    security: SecurityEvidence | None = None
    load: LoadEvidence | None = None
    restore: RestoreEvidence | None = None
    canary: CanaryEvidence | None = None
    cost: CostEvidence | None = None
    integrity_digest: str | None = None

    @field_validator("application_id", "environment")
    @classmethod
    def require_opaque_identifier(cls, value: str) -> str:
        if not _OPAQUE_ID.fullmatch(value) or "@" in value:
            raise ValueError("evidence identifiers must be opaque")
        return value

    @field_validator("target_release_digest", "integrity_digest")
    @classmethod
    def require_digest(cls, value: str | None) -> str | None:
        if value is not None and not _DIGEST.fullmatch(value):
            raise ValueError("pack digests must use sha256:<hex>")
        return value


class ReadinessCheck(ContractModel):
    check_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    status: ReadinessStatus
    summary: str = Field(min_length=1, max_length=1_000)
    evidence_ref: str | None = Field(default=None, max_length=512)


class ProductionReadinessScorecard(ContractModel):
    application_id: str
    environment: str
    target_release_digest: str
    policy_id: str
    policy_version: str
    policy_digest: str
    evidence_digest: str
    evaluated_at: AwareDatetime
    ready: bool
    passed_checks: int = Field(ge=0)
    required_checks: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)
    checks: tuple[ReadinessCheck, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def decision_is_derived_from_checks(self) -> Self:
        passed = sum(item.status is ReadinessStatus.PASS for item in self.checks)
        if self.passed_checks != passed or self.required_checks != len(self.checks):
            raise ValueError("scorecard counts must match checks")
        expected_score = passed / len(self.checks)
        if abs(self.score - expected_score) > 1e-12:
            raise ValueError("score must be the unweighted pass ratio")
        if self.ready is not all(
            item.status is ReadinessStatus.PASS for item in self.checks
        ):
            raise ValueError("ready must be derived from all required checks")
        return self


def illustrative_policy(
    *,
    application_id: str = "email-support-agent",
    environment: str = "production",
    owner_group: str = "group:solution-owners",
) -> ProductionReadinessPolicy:
    """Return visible example thresholds that cannot authorize production."""

    return ProductionReadinessPolicy(
        application_id=application_id,
        environment=environment,
        profile_id="illustrative-email-support-v1",
        version="1",
        owner_group=owner_group,
        approved_for_production=False,
    )


def release_digest(components: ReleaseComponents) -> str:
    """Bind every application release dimension into one deterministic digest."""

    return _canonical_digest(components.model_dump(mode="json"))


def evidence_digest(evidence: ProductionEvidencePack) -> str:
    """Calculate the local integrity digest, excluding the digest field itself."""

    payload = evidence.model_dump(mode="json", exclude={"integrity_digest"})
    return _canonical_digest(payload)


def seal_evidence(evidence: ProductionEvidencePack) -> ProductionEvidencePack:
    """Return a copy sealed against later accidental or unreviewed mutation."""

    return evidence.model_copy(update={"integrity_digest": evidence_digest(evidence)})


def evaluate_production_readiness(
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
    *,
    evaluated_at: datetime,
) -> ProductionReadinessScorecard:
    """Evaluate a fixed, deterministic set of blocking readiness rules.

    A ``ready`` result means only that this supplied evidence pack satisfies the
    owner-approved policy. The function performs no external validation itself.
    """

    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        raise ValueError("evaluated_at must include a timezone")

    checks: list[ReadinessCheck] = []
    add = checks.append
    add(_policy_check(evidence, policy))

    calculated_digest = evidence_digest(evidence)
    if evidence.integrity_digest is None:
        add(_check("evidence_integrity", ReadinessStatus.MISSING, "pack is unsealed"))
    elif not hmac.compare_digest(evidence.integrity_digest, calculated_digest):
        add(
            _check(
                "evidence_integrity",
                ReadinessStatus.TAMPERED,
                "pack content does not match its integrity digest",
            )
        )
    elif evidence.assembled_at > evaluated_at:
        add(
            _check(
                "evidence_integrity",
                ReadinessStatus.UNVERIFIED,
                "pack assembly timestamp is in the future",
            )
        )
    else:
        add(_check("evidence_integrity", ReadinessStatus.PASS, "pack digest matches"))

    add(
        _lineage_check(
            evidence.lineage,
            evidence=evidence,
            policy=policy,
            evaluated_at=evaluated_at,
        )
    )
    add(
        _ingress_check(
            evidence.ingress,
            evidence=evidence,
            policy=policy,
            evaluated_at=evaluated_at,
        )
    )
    add(
        _operations_check(
            evidence.operations,
            evidence=evidence,
            policy=policy,
            evaluated_at=evaluated_at,
        )
    )
    add(
        _slo_check(
            evidence.slo,
            evidence=evidence,
            policy=policy,
            evaluated_at=evaluated_at,
        )
    )
    add(
        _reliability_check(
            evidence.reliability,
            evidence=evidence,
            policy=policy,
            evaluated_at=evaluated_at,
        )
    )
    add(
        _security_check(
            evidence.security,
            evidence=evidence,
            policy=policy,
            evaluated_at=evaluated_at,
        )
    )
    add(
        _load_check(
            evidence.load,
            evidence=evidence,
            policy=policy,
            evaluated_at=evaluated_at,
        )
    )
    add(
        _restore_check(
            evidence.restore,
            evidence=evidence,
            policy=policy,
            evaluated_at=evaluated_at,
        )
    )
    add(
        _canary_check(
            evidence.canary,
            evidence=evidence,
            policy=policy,
            evaluated_at=evaluated_at,
        )
    )
    add(
        _cost_check(
            evidence.cost,
            evidence=evidence,
            policy=policy,
            evaluated_at=evaluated_at,
        )
    )
    add(
        _zero_tolerance_check(
            "zero_cross_tenant_leakage",
            evidence,
            values=(
                (
                    None
                    if evidence.security is None
                    else evidence.security.cross_tenant_leakage_count
                ),
                (
                    None
                    if evidence.canary is None
                    else evidence.canary.cross_tenant_leakage_count
                ),
            ),
        )
    )
    add(
        _zero_tolerance_check(
            "zero_unauthorized_sends",
            evidence,
            values=(
                (
                    None
                    if evidence.security is None
                    else evidence.security.unauthorized_send_count
                ),
                (
                    None
                    if evidence.canary is None
                    else evidence.canary.unauthorized_send_count
                ),
            ),
        )
    )
    add(
        _zero_tolerance_check(
            "zero_duplicate_sends",
            evidence,
            values=(
                (
                    None
                    if evidence.reliability is None
                    else evidence.reliability.duplicate_send_count
                ),
                (
                    None
                    if evidence.restore is None
                    else evidence.restore.duplicate_send_count
                ),
                (
                    None
                    if evidence.canary is None
                    else evidence.canary.duplicate_send_count
                ),
            ),
        )
    )
    add(
        _zero_tolerance_check(
            "zero_high_risk_false_auto_sends",
            evidence,
            values=(
                (
                    None
                    if evidence.security is None
                    else evidence.security.high_risk_false_auto_send_count
                ),
                (
                    None
                    if evidence.canary is None
                    else evidence.canary.high_risk_false_auto_send_count
                ),
            ),
        )
    )

    ready = all(item.status is ReadinessStatus.PASS for item in checks)
    passed = sum(item.status is ReadinessStatus.PASS for item in checks)
    return ProductionReadinessScorecard(
        application_id=evidence.application_id,
        environment=evidence.environment,
        target_release_digest=evidence.target_release_digest,
        policy_id=policy.profile_id,
        policy_version=policy.version,
        policy_digest=_canonical_digest(policy.model_dump(mode="json")),
        evidence_digest=calculated_digest,
        evaluated_at=evaluated_at,
        ready=ready,
        passed_checks=passed,
        required_checks=len(checks),
        score=passed / len(checks),
        checks=tuple(checks),
    )


def _policy_check(
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
) -> ReadinessCheck:
    if (
        evidence.application_id != policy.application_id
        or evidence.environment != policy.environment
    ):
        return _check(
            "policy_approval",
            ReadinessStatus.MISMATCH,
            "policy application or environment does not match the evidence pack",
            policy.approval_ref,
        )
    if not policy.approved_for_production:
        return _check(
            "policy_approval",
            ReadinessStatus.UNVERIFIED,
            "illustrative or unapproved policy cannot authorize production",
            policy.approval_ref,
        )
    return _check(
        "policy_approval",
        ReadinessStatus.PASS,
        "owner-approved policy is bound to this application and environment",
        policy.approval_ref,
    )


def _lineage_check(
    item: ReleaseLineageEvidence | None,
    *,
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
    evaluated_at: datetime,
) -> ReadinessCheck:
    preflight = _preflight(
        "release_lineage",
        item,
        EvidenceOrigin.PLATFORM_CONTROL,
        evidence=evidence,
        policy=policy,
        evaluated_at=evaluated_at,
    )
    if preflight is not None:
        return preflight
    assert item is not None
    expected = evidence.target_release_digest
    observed = (
        release_digest(item.components),
        item.declared_release_digest,
        item.evaluated_release_digest,
        item.deployed_release_digest,
        item.canary_release_digest,
    )
    if any(value != expected for value in observed):
        return _check(
            "release_lineage",
            ReadinessStatus.MISMATCH,
            "declared, evaluated, deployed, or canary release lineage differs",
            item.provenance.evidence_ref,
        )
    return _check(
        "release_lineage",
        ReadinessStatus.PASS,
        "all release dimensions match the target release digest",
        item.provenance.evidence_ref,
    )


def _ingress_check(
    item: IngressVerificationEvidence | None,
    *,
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
    evaluated_at: datetime,
) -> ReadinessCheck:
    preflight = _preflight(
        "ingress_verification",
        item,
        EvidenceOrigin.PLATFORM_CONTROL,
        evidence=evidence,
        policy=policy,
        evaluated_at=evaluated_at,
    )
    if preflight is not None:
        return preflight
    assert item is not None
    failures: list[str] = []
    _minimum(
        failures, "messages", item.messages_observed, policy.minimum_ingress_samples
    )
    for name, count in (
        ("authentication", item.authenticated_messages),
        ("malware scan", item.malware_scanned_messages),
        ("DLP", item.dlp_processed_messages),
        ("access context", item.access_context_verified_messages),
    ):
        coverage = count / item.messages_observed
        if coverage < policy.minimum_ingress_verification_coverage:
            failures.append(f"{name} coverage {coverage:.4f} below threshold")
    if item.raw_payload_checkpoint_count != 0:
        failures.append("raw payloads were observed in durable checkpoints")
    return _metric_result(
        "ingress_verification", failures, item.provenance.evidence_ref
    )


def _operations_check(
    item: OperationalEvidence | None,
    *,
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
    evaluated_at: datetime,
) -> ReadinessCheck:
    preflight = _preflight(
        "operational_capabilities",
        item,
        EvidenceOrigin.PLATFORM_CONTROL,
        evidence=evidence,
        policy=policy,
        evaluated_at=evaluated_at,
    )
    if preflight is not None:
        return preflight
    assert item is not None
    by_capability = {entry.capability: entry for entry in item.dependencies}
    failures = [
        f"missing capability {capability.value}"
        for capability in policy.required_capabilities
        if capability not in by_capability
    ]
    for capability in policy.required_capabilities:
        observation = by_capability.get(capability)
        if observation is None:
            continue
        checks = (
            observation.configured,
            observation.health_check_passed,
            observation.authorization_check_passed,
            observation.least_privilege_reviewed,
            observation.failure_policy_configured,
        )
        if not all(checks):
            failures.append(f"capability {capability.value} is not production-ready")
    return _metric_result(
        "operational_capabilities", failures, item.provenance.evidence_ref
    )


def _slo_check(
    item: SloObservationEvidence | None,
    *,
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
    evaluated_at: datetime,
) -> ReadinessCheck:
    preflight = _preflight(
        "service_levels",
        item,
        EvidenceOrigin.PLATFORM_CONTROL,
        evidence=evidence,
        policy=policy,
        evaluated_at=evaluated_at,
    )
    if preflight is not None:
        return preflight
    assert item is not None
    target = policy.slo
    failures: list[str] = []
    _minimum(
        failures,
        "SLO window seconds",
        item.window_seconds,
        target.minimum_window_seconds,
    )
    _minimum(failures, "SLO requests", item.request_count, target.minimum_request_count)
    _maximum(
        failures,
        "error rate",
        item.error_count / item.request_count,
        target.maximum_error_rate,
    )
    availability = item.available_request_count / item.request_count
    if availability < target.minimum_availability:
        failures.append(f"availability {availability:.4f} below threshold")
    _maximum(
        failures,
        "p95 latency ms",
        item.p95_latency_ms,
        target.maximum_p95_latency_ms,
    )
    _maximum(
        failures,
        "p95 review wait seconds",
        item.p95_review_wait_seconds,
        target.maximum_p95_review_wait_seconds,
    )
    return _metric_result("service_levels", failures, item.provenance.evidence_ref)


def _reliability_check(
    item: ReliabilityEvidence | None,
    *,
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
    evaluated_at: datetime,
) -> ReadinessCheck:
    preflight = _preflight(
        "reliability",
        item,
        EvidenceOrigin.PLATFORM_CONTROL,
        evidence=evidence,
        policy=policy,
        evaluated_at=evaluated_at,
    )
    if preflight is not None:
        return preflight
    assert item is not None
    failures: list[str] = []
    _minimum(
        failures,
        "reliability scenarios",
        item.scenarios_executed,
        policy.minimum_reliability_scenarios,
    )
    if item.scenarios_passed != item.scenarios_executed:
        failures.append("not every reliability scenario passed")
    delivery_rate = item.successful_deliveries / item.delivery_attempts
    if delivery_rate < policy.minimum_delivery_success_rate:
        failures.append(f"delivery success rate {delivery_rate:.4f} below threshold")
    if item.idempotency_replay_passes != item.idempotency_replay_cases:
        failures.append("not every idempotency replay passed")
    for passed, label in (
        (item.checkpoint_resume_passed, "checkpoint resume"),
        (item.outbox_recovery_passed, "outbox recovery"),
        (item.retry_budget_enforced, "retry budget"),
        (item.dead_letter_recovery_passed, "dead-letter recovery"),
    ):
        if not passed:
            failures.append(f"{label} did not pass")
    return _metric_result("reliability", failures, item.provenance.evidence_ref)


def _security_check(
    item: SecurityEvidence | None,
    *,
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
    evaluated_at: datetime,
) -> ReadinessCheck:
    preflight = _preflight(
        "security_controls",
        item,
        EvidenceOrigin.SECURITY_CONTROL,
        evidence=evidence,
        policy=policy,
        evaluated_at=evaluated_at,
    )
    if preflight is not None:
        return preflight
    assert item is not None
    failures: list[str] = []
    _minimum(
        failures,
        "security cases",
        item.security_test_cases,
        policy.minimum_security_test_cases,
    )
    for passed, label in (
        (item.dlp_admission_passed, "DLP admission"),
        (item.trusted_access_context_passed, "trusted access context"),
        (item.reviewer_authorization_passed, "reviewer authorization"),
        (item.trace_policy_passed, "trace policy"),
        (item.prompt_injection_precheck_passed, "prompt-injection precheck"),
        (item.least_privilege_review_passed, "least-privilege review"),
    ):
        if not passed:
            failures.append(f"{label} did not pass")
    return _metric_result("security_controls", failures, item.provenance.evidence_ref)


def _load_check(
    item: LoadEvidence | None,
    *,
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
    evaluated_at: datetime,
) -> ReadinessCheck:
    preflight = _preflight(
        "load_test",
        item,
        EvidenceOrigin.PERFORMANCE_HARNESS,
        evidence=evidence,
        policy=policy,
        evaluated_at=evaluated_at,
    )
    if preflight is not None:
        return preflight
    assert item is not None
    target = policy.load
    failures: list[str] = []
    _minimum(
        failures,
        "load duration",
        item.duration_seconds,
        target.minimum_duration_seconds,
    )
    _minimum(
        failures, "load requests", item.request_count, target.minimum_request_count
    )
    _minimum(failures, "target RPS", item.target_rps, target.minimum_target_rps)
    if item.achieved_rps < item.target_rps:
        failures.append("achieved RPS is below the test target")
    _maximum(
        failures,
        "load error rate",
        item.error_count / item.request_count,
        target.maximum_error_rate,
    )
    _maximum(
        failures,
        "load p95 latency ms",
        item.p95_latency_ms,
        target.maximum_p95_latency_ms,
    )
    _maximum(
        failures,
        "resource saturation",
        item.peak_resource_saturation,
        target.maximum_resource_saturation,
    )
    return _metric_result("load_test", failures, item.provenance.evidence_ref)


def _restore_check(
    item: RestoreEvidence | None,
    *,
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
    evaluated_at: datetime,
) -> ReadinessCheck:
    preflight = _preflight(
        "restore_test",
        item,
        EvidenceOrigin.RESTORE_HARNESS,
        evidence=evidence,
        policy=policy,
        evaluated_at=evaluated_at,
    )
    if preflight is not None:
        return preflight
    assert item is not None
    failures: list[str] = []
    if item.restored_release_digest != evidence.target_release_digest:
        failures.append("restored release differs from target release")
    if not item.restore_succeeded:
        failures.append("restore did not succeed")
    _maximum(
        failures,
        "RTO seconds",
        item.observed_rto_seconds,
        policy.restore.maximum_rto_seconds,
    )
    _maximum(
        failures,
        "RPO seconds",
        item.observed_rpo_seconds,
        policy.restore.maximum_rpo_seconds,
    )
    if not item.checkpoint_integrity_passed:
        failures.append("restored checkpoint integrity failed")
    if not item.outbox_reconciled:
        failures.append("restored outbox was not reconciled")
    return _metric_result("restore_test", failures, item.provenance.evidence_ref)


def _canary_check(
    item: CanaryEvidence | None,
    *,
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
    evaluated_at: datetime,
) -> ReadinessCheck:
    preflight = _preflight(
        "canary",
        item,
        EvidenceOrigin.CANARY_TELEMETRY,
        evidence=evidence,
        policy=policy,
        evaluated_at=evaluated_at,
    )
    if preflight is not None:
        return preflight
    assert item is not None
    target = policy.canary
    failures: list[str] = []
    _minimum(
        failures,
        "canary duration",
        item.duration_seconds,
        target.minimum_duration_seconds,
    )
    _minimum(
        failures, "canary requests", item.request_count, target.minimum_request_count
    )
    if (
        not target.minimum_traffic_fraction
        <= item.traffic_fraction
        <= target.maximum_traffic_fraction
    ):
        failures.append("canary traffic fraction is outside the approved range")
    _maximum(
        failures,
        "canary error rate",
        item.error_count / item.request_count,
        target.maximum_error_rate,
    )
    _maximum(
        failures,
        "canary p95 latency ms",
        item.p95_latency_ms,
        target.maximum_p95_latency_ms,
    )
    trace_coverage = item.traced_request_count / item.request_count
    if trace_coverage < target.minimum_trace_coverage:
        failures.append(f"trace coverage {trace_coverage:.4f} below threshold")
    if not item.rollback_ready:
        failures.append("canary rollback is not ready")
    return _metric_result("canary", failures, item.provenance.evidence_ref)


def _cost_check(
    item: CostEvidence | None,
    *,
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
    evaluated_at: datetime,
) -> ReadinessCheck:
    preflight = _preflight(
        "unit_economics",
        item,
        EvidenceOrigin.FINOPS_LEDGER,
        evidence=evidence,
        policy=policy,
        evaluated_at=evaluated_at,
    )
    if preflight is not None:
        return preflight
    assert item is not None
    target = policy.cost
    failures: list[str] = []
    _minimum(
        failures,
        "costed population",
        item.requests_observed,
        target.minimum_request_count,
    )
    if item.coverage < target.minimum_cost_coverage:
        failures.append(f"cost coverage {item.coverage:.4f} below threshold")
    _maximum(
        failures,
        "cost per request USD",
        item.cost_per_request_usd,
        target.maximum_cost_per_request_usd,
    )
    if item.cost_per_resolved_case_usd is None:
        failures.append("no resolved cases are available for unit economics")
    else:
        _maximum(
            failures,
            "cost per resolved case USD",
            item.cost_per_resolved_case_usd,
            target.maximum_cost_per_resolved_case_usd,
        )
    _maximum(
        failures,
        "p95 model calls",
        item.p95_model_calls,
        target.maximum_p95_model_calls,
    )
    return _metric_result("unit_economics", failures, item.provenance.evidence_ref)


def _preflight(
    check_id: str,
    item: object | None,
    expected_origin: EvidenceOrigin,
    *,
    evidence: ProductionEvidencePack,
    policy: ProductionReadinessPolicy,
    evaluated_at: datetime,
) -> ReadinessCheck | None:
    if item is None:
        return _check(check_id, ReadinessStatus.MISSING, "required evidence is missing")
    provenance = item.provenance
    if provenance.origin is EvidenceOrigin.SYNTHETIC:
        return _check(
            check_id,
            ReadinessStatus.UNVERIFIED,
            "synthetic evidence cannot establish production readiness",
            provenance.evidence_ref,
        )
    if provenance.origin is not expected_origin:
        return _check(
            check_id,
            ReadinessStatus.UNVERIFIED,
            f"expected {expected_origin.value} attestation",
            provenance.evidence_ref,
        )
    if (
        provenance.application_id != evidence.application_id
        or provenance.environment != evidence.environment
    ):
        return _check(
            check_id,
            ReadinessStatus.MISMATCH,
            "attestation application or environment differs from the evidence pack",
            provenance.evidence_ref,
        )
    if provenance.target_release_digest != evidence.target_release_digest:
        return _check(
            check_id,
            ReadinessStatus.MISMATCH,
            "attestation targets a different release",
            provenance.evidence_ref,
        )
    age = evaluated_at - provenance.observed_at
    if age < timedelta(0):
        return _check(
            check_id,
            ReadinessStatus.UNVERIFIED,
            "attestation timestamp is in the future",
            provenance.evidence_ref,
        )
    if evaluated_at > provenance.valid_until or age > timedelta(
        hours=policy.maximum_evidence_age_hours
    ):
        return _check(
            check_id,
            ReadinessStatus.STALE,
            "attestation is outside the approved freshness window",
            provenance.evidence_ref,
        )
    return None


def _zero_tolerance_check(
    check_id: str,
    evidence: ProductionEvidencePack,
    *,
    values: tuple[int | None, ...],
) -> ReadinessCheck:
    if any(value is None for value in values):
        return _check(
            check_id,
            ReadinessStatus.MISSING,
            "all required invariant evidence must be present",
        )
    violations = sum(value for value in values if value is not None)
    if violations:
        return _check(
            check_id,
            ReadinessStatus.FAIL,
            f"observed {violations} zero-tolerance violation(s)",
        )
    return _check(
        check_id,
        ReadinessStatus.PASS,
        "zero violations observed across required evidence",
    )


def _metric_result(
    check_id: str,
    failures: list[str],
    evidence_ref: str,
) -> ReadinessCheck:
    if failures:
        return _check(
            check_id,
            ReadinessStatus.FAIL,
            "; ".join(failures),
            evidence_ref,
        )
    return _check(
        check_id,
        ReadinessStatus.PASS,
        "all required thresholds passed",
        evidence_ref,
    )


def _minimum(
    failures: list[str],
    label: str,
    observed: float,
    required: float,
) -> None:
    if observed < required:
        failures.append(f"{label} {observed:g} below minimum {required:g}")


def _maximum(
    failures: list[str],
    label: str,
    observed: float,
    maximum: float,
) -> None:
    if observed > maximum:
        failures.append(f"{label} {observed:g} exceeds maximum {maximum:g}")


def _check(
    check_id: str,
    status: ReadinessStatus,
    summary: str,
    evidence_ref: str | None = None,
) -> ReadinessCheck:
    return ReadinessCheck(
        check_id=check_id,
        status=status,
        summary=summary,
        evidence_ref=evidence_ref,
    )


def _canonical_digest(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + sha256(serialized).hexdigest()
