"""Deterministic health and production-readiness evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated

from pydantic import Field, model_validator

from .models import (
    DeploymentStatus,
    HealthEvidence,
    HealthProfile,
    HealthSnapshot,
    HealthStatus,
    HubModel,
    IncidentSeverity,
    NonEmptyStr,
    ReadinessRuleResult,
    ReadinessSeverity,
    ReadinessSnapshot,
    ReadinessStatus,
    SnakeCaseKey,
    Timestamp,
)


class ManifestEvidence(HubModel):
    schema_valid: bool
    git_sha_matches_registered_version: bool | None
    resources_resolve: bool | None


class OwnershipEvidence(HubModel):
    owner_registered: bool
    support_group_registered: bool


class TagEvidence(HubModel):
    observed_tag_keys: tuple[SnakeCaseKey, ...]
    data_classification_declared: bool

    @model_validator(mode="after")
    def keys_are_unique(self) -> TagEvidence:
        if len(self.observed_tag_keys) != len(set(self.observed_tag_keys)):
            raise ValueError("observed tag keys must be unique")
        return self


class RiskEvidence(HubModel):
    risk_tier: NonEmptyStr | None


class AuthenticationEvidence(HubModel):
    mode_declared: bool
    approved_ai_gateway: bool | None
    rate_limit_configured: bool | None


class TracingEvidence(HubModel):
    configured: bool | None
    recent_trace_metadata_complete: bool | None


class JobEvidence(HubModel):
    evaluation_job_configured: bool
    promotion_job_configured: bool


class EvaluationEvidence(HubModel):
    dataset_exists: bool | None
    dataset_case_count: Annotated[int, Field(ge=0)] | None
    targets_current_application_version: bool | None
    evaluated_at: Timestamp | None
    blocking_thresholds_passed: bool | None


class TelemetryEvidence(HubModel):
    monitoring_configured: bool | None
    service_levels_passed: bool | None
    approved_new_application_exception: bool = False
    required_request_tags_observed: bool | None
    load_test_required: bool = False
    current_load_test_evidence: bool | None = None


class AdministratorEvidence(HubModel):
    unresolved_blocking_actions: Annotated[int, Field(ge=0)] | None = None


class ReadinessEvidence(HubModel):
    application_id: NonEmptyStr
    environment: NonEmptyStr
    application_version_id: NonEmptyStr
    manifest: ManifestEvidence
    ownership: OwnershipEvidence
    tags: TagEvidence
    risk: RiskEvidence
    authentication: AuthenticationEvidence
    tracing: TracingEvidence
    jobs: JobEvidence
    evaluation: EvaluationEvidence
    telemetry: TelemetryEvidence
    administrator: AdministratorEvidence


class ReadinessProfile(HubModel):
    profile_id: NonEmptyStr
    version: NonEmptyStr
    required_tag_keys: tuple[SnakeCaseKey, ...] = (
        "team",
        "domain",
        "cost_center",
        "environment",
        "application_id",
    )
    minimum_evaluation_cases: Annotated[int, Field(gt=0)] = 30
    maximum_evaluation_age_hours: Annotated[int, Field(gt=0)] = 168
    require_ai_gateway: bool = True
    require_request_tags: bool = True
    require_rate_limit: bool = False
    require_monitoring: bool = True

    @model_validator(mode="after")
    def required_tags_are_unique(self) -> ReadinessProfile:
        if len(self.required_tag_keys) != len(set(self.required_tag_keys)):
            raise ValueError("required tag keys must be unique")
        return self


def calculate_health(
    evidence: HealthEvidence,
    profile: HealthProfile,
    *,
    evaluated_at: datetime,
) -> HealthSnapshot:
    """Calculate one explainable health status from bounded evidence.

    Precedence is deterministic: critical platform state, critical SLOs,
    incomplete/stale evidence, degraded state, then healthy.  A fresh zero-traffic
    interval is healthy rather than being treated as an outage.
    """

    critical: list[str] = []
    degraded: list[str] = []
    unknown: list[str] = []
    healthy: list[str] = []

    if evidence.deployment_status in {
        DeploymentStatus.FAILED,
        DeploymentStatus.UNAVAILABLE,
    }:
        critical.append(
            f"active deployment is {evidence.deployment_status.value.lower()}"
        )
    elif (
        evidence.deployment_expected_active
        and evidence.deployment_status is DeploymentStatus.STOPPED
    ):
        critical.append("active deployment is stopped")
    elif evidence.deployment_status in {
        DeploymentStatus.UNKNOWN,
        DeploymentStatus.DEPLOYING,
    }:
        unknown.append(
            f"deployment status is {evidence.deployment_status.value.lower()}"
        )

    if evidence.incident_severity is IncidentSeverity.CRITICAL:
        critical.append("a critical production incident is active")
    elif evidence.incident_severity is IncidentSeverity.DEGRADED:
        degraded.append("a degrading production incident is active")

    evidence_at = evidence.telemetry_observed_at
    if evidence_at is None:
        unknown.append("telemetry freshness is unknown")
    else:
        telemetry_age = evaluated_at - evidence_at
        if telemetry_age < timedelta(0):
            unknown.append("telemetry timestamp is in the future")
        elif telemetry_age > timedelta(seconds=profile.maximum_telemetry_age_seconds):
            unknown.append("telemetry is stale")

    if evidence.request_count is None:
        unknown.append("request and error counts are unavailable")
    elif evidence.request_count == 0:
        healthy.append("no requests were observed in the fresh evidence window")
    else:
        if evidence.error_count is None:
            unknown.append("error count is unavailable")
        else:
            error_rate = evidence.error_count / evidence.request_count
            if error_rate > profile.critical_error_rate:
                critical.append(
                    f"error rate {error_rate:.4f} exceeds critical threshold "
                    f"{profile.critical_error_rate:.4f}"
                )
            elif error_rate > profile.maximum_error_rate:
                degraded.append(
                    f"error rate {error_rate:.4f} exceeds threshold "
                    f"{profile.maximum_error_rate:.4f}"
                )
            else:
                healthy.append("error rate is within the configured service level")

    if evidence.p95_latency_ms is not None:
        if evidence.p95_latency_ms > profile.critical_p95_latency_ms:
            critical.append(
                f"p95 latency {evidence.p95_latency_ms:g} ms exceeds critical "
                f"threshold {profile.critical_p95_latency_ms:g} ms"
            )
        elif evidence.p95_latency_ms > profile.maximum_p95_latency_ms:
            degraded.append(
                f"p95 latency {evidence.p95_latency_ms:g} ms exceeds threshold "
                f"{profile.maximum_p95_latency_ms:g} ms"
            )
        else:
            healthy.append("p95 latency is within the configured service level")
    elif evidence.request_count:
        unknown.append("p95 latency is unavailable for observed traffic")

    if (
        profile.evaluation_failure_affects_health
        and evidence.latest_evaluation_passed is False
    ):
        degraded.append("the latest evaluation failed this health profile")

    if critical:
        status = HealthStatus.CRITICAL
        reasons = critical + degraded + unknown
    elif unknown:
        status = HealthStatus.UNKNOWN
        reasons = unknown + degraded
    elif degraded:
        status = HealthStatus.DEGRADED
        reasons = degraded
    else:
        status = HealthStatus.HEALTHY
        reasons = healthy or ["all available health evidence is within policy"]

    return HealthSnapshot(
        application_id=evidence.application_id,
        environment=evidence.environment,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        status=status,
        reasons=tuple(reasons),
        evidence_at=evidence_at,
        evaluated_at=evaluated_at,
    )


class ReadinessEvaluator:
    """Versioned, deterministic readiness rules over explicit evidence."""

    def __init__(self, profile: ReadinessProfile) -> None:
        self.profile = profile

    def evaluate(
        self,
        evidence: ReadinessEvidence,
        *,
        evaluated_at: datetime,
    ) -> ReadinessSnapshot:
        results = self._evaluate_rules(evidence, evaluated_at=evaluated_at)
        severity_order = {
            ReadinessSeverity.BLOCKING: 0,
            ReadinessSeverity.WARNING: 1,
            ReadinessSeverity.INFORMATIONAL: 2,
        }
        status_order = {
            ReadinessStatus.FAIL: 0,
            ReadinessStatus.UNKNOWN: 1,
            ReadinessStatus.PASS: 2,
            ReadinessStatus.NOT_APPLICABLE: 3,
        }
        ordered = tuple(
            sorted(
                results,
                key=lambda result: (
                    severity_order[result.severity],
                    status_order[result.status],
                ),
            )
        )
        ready = not any(
            result.severity is ReadinessSeverity.BLOCKING
            and result.status in {ReadinessStatus.FAIL, ReadinessStatus.UNKNOWN}
            for result in ordered
        )
        return ReadinessSnapshot(
            application_id=evidence.application_id,
            environment=evidence.environment,
            application_version_id=evidence.application_version_id,
            profile_id=self.profile.profile_id,
            profile_version=self.profile.version,
            evaluated_at=evaluated_at,
            ready=ready,
            results=ordered,
        )

    def _result(
        self,
        *,
        rule_id: str,
        description: str,
        status: ReadinessStatus,
        evidence: str,
        evaluated_at: datetime,
        remediation: str | None = None,
        severity: ReadinessSeverity = ReadinessSeverity.BLOCKING,
    ) -> ReadinessRuleResult:
        return ReadinessRuleResult(
            rule_id=rule_id,
            rule_version=self.profile.version,
            description=description,
            severity=severity,
            status=status,
            evidence=(evidence,),
            evaluated_at=evaluated_at,
            remediation=remediation,
        )

    @staticmethod
    def _boolean_status(value: bool | None) -> ReadinessStatus:
        if value is None:
            return ReadinessStatus.UNKNOWN
        return ReadinessStatus.PASS if value else ReadinessStatus.FAIL

    def _evaluate_rules(
        self,
        evidence: ReadinessEvidence,
        *,
        evaluated_at: datetime,
    ) -> list[ReadinessRuleResult]:
        result: list[ReadinessRuleResult] = []
        add = result.append

        add(
            self._result(
                rule_id="manifest_schema",
                description="The registered manifest conforms to its versioned schema.",
                status=self._boolean_status(evidence.manifest.schema_valid),
                evidence=(
                    "manifest schema is valid"
                    if evidence.manifest.schema_valid
                    else "manifest schema validation failed"
                ),
                evaluated_at=evaluated_at,
                remediation="Register a manifest that passes its declared JSON Schema.",
            )
        )

        ownership_ok = (
            evidence.ownership.owner_registered
            and evidence.ownership.support_group_registered
        )
        missing_ownership = []
        if not evidence.ownership.owner_registered:
            missing_ownership.append("owner")
        if not evidence.ownership.support_group_registered:
            missing_ownership.append("support group")
        add(
            self._result(
                rule_id="ownership",
                description="An owner and support group are registered.",
                status=self._boolean_status(ownership_ok),
                evidence=(
                    "owner and support group are registered"
                    if ownership_ok
                    else f"missing: {', '.join(missing_ownership)}"
                ),
                evaluated_at=evaluated_at,
                remediation="Register non-personal owner and support groups.",
            )
        )

        missing_tags = sorted(
            set(self.profile.required_tag_keys) - set(evidence.tags.observed_tag_keys)
        )
        add(
            self._result(
                rule_id="required_tags",
                description="All platform-required application tags are present.",
                status=(
                    ReadinessStatus.PASS if not missing_tags else ReadinessStatus.FAIL
                ),
                evidence=(
                    "all required tags are present"
                    if not missing_tags
                    else f"missing tags: {', '.join(missing_tags)}"
                ),
                evaluated_at=evaluated_at,
                remediation="Add the missing controlled tags to the manifest.",
            )
        )

        risk_ok = bool(evidence.risk.risk_tier)
        classification_ok = evidence.tags.data_classification_declared
        add(
            self._result(
                rule_id="risk_and_classification",
                description="Risk tier and data classification are declared.",
                status=self._boolean_status(risk_ok and classification_ok),
                evidence=(
                    "risk tier and data classification are declared"
                    if risk_ok and classification_ok
                    else "risk tier or data classification is missing"
                ),
                evaluated_at=evaluated_at,
                remediation="Declare risk tier and data classification.",
            )
        )

        for rule_id, description, value, passed, failed, remediation in (
            (
                "registered_git_sha",
                "The deployed Git SHA matches the registered version.",
                evidence.manifest.git_sha_matches_registered_version,
                "deployed and registered Git SHAs match",
                "deployed Git SHA differs from the registered version",
                "Register or deploy the intended immutable application version.",
            ),
            (
                "resource_resolution",
                "Required Databricks resources resolve successfully.",
                evidence.manifest.resources_resolve,
                "all registered resources resolved",
                "one or more registered resources did not resolve",
                "Correct resource IDs or request the missing platform grant.",
            ),
            (
                "authentication_mode",
                "The application authentication mode is declared.",
                evidence.authentication.mode_declared,
                "authentication mode is declared",
                "authentication mode is missing",
                "Declare the supported authentication mode in the manifest.",
            ),
            (
                "tracing_configuration",
                "MLflow tracing is configured.",
                evidence.tracing.configured,
                "MLflow tracing is configured",
                "MLflow tracing is not configured",
                "Configure governed MLflow tracing for the application.",
            ),
            (
                "trace_metadata",
                "Recent traces carry application, environment, and version metadata.",
                evidence.tracing.recent_trace_metadata_complete,
                "recent trace metadata is complete",
                "required metadata is absent from recent traces",
                "Emit the controlled application, environment, and version metadata.",
            ),
            (
                "evaluation_target",
                "The latest evaluation targets the current application version.",
                evidence.evaluation.targets_current_application_version,
                "latest evaluation targets the current version",
                "latest evaluation targets another version",
                "Evaluate the currently registered application version.",
            ),
            (
                "evaluation_thresholds",
                "All blocking evaluation thresholds pass.",
                evidence.evaluation.blocking_thresholds_passed,
                "all blocking evaluation thresholds passed",
                "one or more blocking evaluation thresholds failed",
                "Address the failed evaluation gates and run the suite again.",
            ),
        ):
            status = self._boolean_status(value)
            add(
                self._result(
                    rule_id=rule_id,
                    description=description,
                    status=status,
                    evidence=passed if status is ReadinessStatus.PASS else failed,
                    evaluated_at=evaluated_at,
                    remediation=remediation,
                )
            )

        dataset_status = ReadinessStatus.PASS
        if evidence.evaluation.dataset_exists is None:
            dataset_status = ReadinessStatus.UNKNOWN
            dataset_detail = "evaluation dataset existence is unknown"
        elif not evidence.evaluation.dataset_exists:
            dataset_status = ReadinessStatus.FAIL
            dataset_detail = "evaluation dataset does not exist"
        elif evidence.evaluation.dataset_case_count is None:
            dataset_status = ReadinessStatus.UNKNOWN
            dataset_detail = "evaluation dataset size is unknown"
        elif (
            evidence.evaluation.dataset_case_count
            < self.profile.minimum_evaluation_cases
        ):
            dataset_status = ReadinessStatus.FAIL
            dataset_detail = (
                f"dataset has {evidence.evaluation.dataset_case_count} cases; "
                f"{self.profile.minimum_evaluation_cases} required"
            )
        else:
            dataset_detail = (
                f"dataset has {evidence.evaluation.dataset_case_count} cases"
            )
        add(
            self._result(
                rule_id="evaluation_dataset",
                description="The evaluation dataset exists and has enough cases.",
                status=dataset_status,
                evidence=dataset_detail,
                evaluated_at=evaluated_at,
                remediation="Register a governed dataset with enough reviewed cases.",
            )
        )

        if evidence.evaluation.evaluated_at is None:
            age_status = ReadinessStatus.UNKNOWN
            age_detail = "latest evaluation time is unknown"
        else:
            age = evaluated_at - evidence.evaluation.evaluated_at
            if age < timedelta(0):
                age_status = ReadinessStatus.FAIL
                age_detail = "latest evaluation timestamp is in the future"
            elif age > timedelta(hours=self.profile.maximum_evaluation_age_hours):
                age_status = ReadinessStatus.FAIL
                age_detail = (
                    f"latest evaluation is older than "
                    f"{self.profile.maximum_evaluation_age_hours} hours"
                )
            else:
                age_status = ReadinessStatus.PASS
                age_detail = "latest evaluation is within the permitted age"
        add(
            self._result(
                rule_id="evaluation_freshness",
                description="The latest evaluation is sufficiently recent.",
                status=age_status,
                evidence=age_detail,
                evaluated_at=evaluated_at,
                remediation=(
                    "Run the registered evaluation job for the current version."
                ),
            )
        )

        if not self.profile.require_monitoring:
            monitoring_status = ReadinessStatus.NOT_APPLICABLE
            monitoring_detail = "this profile does not require production monitoring"
            monitoring_severity = ReadinessSeverity.INFORMATIONAL
        else:
            monitoring_status = self._boolean_status(
                evidence.telemetry.monitoring_configured
            )
            monitoring_detail = (
                "production monitoring is configured"
                if monitoring_status is ReadinessStatus.PASS
                else "production monitoring configuration is missing or unknown"
            )
            monitoring_severity = ReadinessSeverity.BLOCKING
        add(
            self._result(
                rule_id="production_monitoring",
                description="Production monitoring or an approved equivalent exists.",
                status=monitoring_status,
                evidence=monitoring_detail,
                evaluated_at=evaluated_at,
                remediation="Configure the approved production monitoring profile.",
                severity=monitoring_severity,
            )
        )

        if evidence.telemetry.service_levels_passed is True:
            slo_status = ReadinessStatus.PASS
            slo_detail = "error and latency service levels pass"
        elif evidence.telemetry.approved_new_application_exception:
            slo_status = ReadinessStatus.PASS
            slo_detail = "an approved new-application exception covers unknown SLOs"
        else:
            slo_status = self._boolean_status(evidence.telemetry.service_levels_passed)
            slo_detail = (
                "error or latency service levels failed"
                if slo_status is ReadinessStatus.FAIL
                else "error and latency service-level evidence is unknown"
            )
        add(
            self._result(
                rule_id="service_levels",
                description="Error and latency service levels pass.",
                status=slo_status,
                evidence=slo_detail,
                evaluated_at=evaluated_at,
                remediation="Resolve the service-level failure or obtain an exception.",
            )
        )

        if not self.profile.require_ai_gateway:
            gateway_status = ReadinessStatus.NOT_APPLICABLE
            gateway_detail = "this profile does not require Unity AI Gateway"
            gateway_severity = ReadinessSeverity.INFORMATIONAL
        else:
            gateway_status = self._boolean_status(
                evidence.authentication.approved_ai_gateway
            )
            gateway_detail = (
                "traffic uses an approved Unity AI Gateway service"
                if gateway_status is ReadinessStatus.PASS
                else "approved Unity AI Gateway use is missing or unknown"
            )
            gateway_severity = ReadinessSeverity.BLOCKING
        add(
            self._result(
                rule_id="approved_ai_gateway",
                description="AI traffic uses an approved Unity AI Gateway service.",
                status=gateway_status,
                evidence=gateway_detail,
                evaluated_at=evaluated_at,
                remediation="Route AI traffic through an approved gateway service.",
                severity=gateway_severity,
            )
        )

        if not self.profile.require_request_tags:
            tag_status = ReadinessStatus.NOT_APPLICABLE
            tag_detail = "this profile does not require observed request tags"
            tag_severity = ReadinessSeverity.INFORMATIONAL
        else:
            tag_status = self._boolean_status(
                evidence.telemetry.required_request_tags_observed
            )
            tag_detail = (
                "required endpoint and request tags were observed"
                if tag_status is ReadinessStatus.PASS
                else "required endpoint or request tags are missing or unknown"
            )
            tag_severity = ReadinessSeverity.BLOCKING
        add(
            self._result(
                rule_id="observed_request_tags",
                description="Required endpoint and request tags are observed.",
                status=tag_status,
                evidence=tag_detail,
                evaluated_at=evaluated_at,
                remediation="Emit only the approved low-cardinality request tags.",
                severity=tag_severity,
            )
        )

        if not self.profile.require_rate_limit:
            rate_status = ReadinessStatus.NOT_APPLICABLE
            rate_detail = "this profile does not require rate limiting"
            rate_severity = ReadinessSeverity.INFORMATIONAL
        else:
            rate_status = self._boolean_status(
                evidence.authentication.rate_limit_configured
            )
            rate_detail = (
                "required rate limiting is configured"
                if rate_status is ReadinessStatus.PASS
                else "required rate limiting is missing or unknown"
            )
            rate_severity = ReadinessSeverity.BLOCKING
        add(
            self._result(
                rule_id="rate_limiting",
                description="Rate limiting is configured where required.",
                status=rate_status,
                evidence=rate_detail,
                evaluated_at=evaluated_at,
                remediation="Configure the rate limit required by this risk profile.",
                severity=rate_severity,
            )
        )

        if not evidence.telemetry.load_test_required:
            load_status = ReadinessStatus.NOT_APPLICABLE
            load_detail = "load-test evidence is not required for this release"
            load_severity = ReadinessSeverity.INFORMATIONAL
        else:
            load_status = self._boolean_status(
                evidence.telemetry.current_load_test_evidence
            )
            load_detail = (
                "current load-test evidence exists"
                if load_status is ReadinessStatus.PASS
                else "current load-test evidence is missing or unknown"
            )
            load_severity = ReadinessSeverity.BLOCKING
        add(
            self._result(
                rule_id="load_test_evidence",
                description="Current load-test evidence exists where required.",
                status=load_status,
                evidence=load_detail,
                evaluated_at=evaluated_at,
                remediation="Run and register the required load test.",
                severity=load_severity,
            )
        )

        jobs_ok = (
            evidence.jobs.evaluation_job_configured
            and evidence.jobs.promotion_job_configured
        )
        add(
            self._result(
                rule_id="workflow_jobs",
                description="Evaluation and promotion jobs are configured.",
                status=self._boolean_status(jobs_ok),
                evidence=(
                    "evaluation and promotion jobs are configured"
                    if jobs_ok
                    else "evaluation or promotion job is not configured"
                ),
                evaluated_at=evaluated_at,
                remediation="Register both approved workflow job IDs.",
            )
        )

        blockers = evidence.administrator.unresolved_blocking_actions
        if blockers is None:
            blocker_status = ReadinessStatus.UNKNOWN
            blocker_evidence = "blocking administrator actions are unknown"
        elif blockers == 0:
            blocker_status = ReadinessStatus.PASS
            blocker_evidence = "no blocking administrator actions are open"
        else:
            blocker_status = ReadinessStatus.FAIL
            blocker_evidence = f"{blockers} blocking administrator action(s) are open"
        add(
            self._result(
                rule_id="administrator_blockers",
                description="No unresolved blocking administrator action exists.",
                status=blocker_status,
                evidence=blocker_evidence,
                evaluated_at=evaluated_at,
                remediation="Resolve the blocking administrator actions.",
            )
        )

        return result
