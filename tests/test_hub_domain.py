"""Pure health and readiness behavior for the AI Platform Hub."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from aai_console.hub.models import (
    DeploymentStatus,
    HealthEvidence,
    HealthProfile,
    HealthStatus,
    ReadinessSeverity,
    ReadinessStatus,
    Role,
)
from aai_console.hub.readiness import (
    AdministratorEvidence,
    AuthenticationEvidence,
    CostEvidence,
    EvaluationEvidence,
    JobEvidence,
    ManifestEvidence,
    OwnershipEvidence,
    ReadinessEvaluator,
    ReadinessEvidence,
    ReadinessProfile,
    RiskEvidence,
    TagEvidence,
    TelemetryEvidence,
    TracingEvidence,
    calculate_health,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def _health_profile() -> HealthProfile:
    return HealthProfile(
        profile_id="standard",
        version="1",
        maximum_error_rate=0.02,
        critical_error_rate=0.10,
        maximum_p95_latency_ms=8_000.0,
        critical_p95_latency_ms=15_000.0,
        maximum_telemetry_age_seconds=3_600,
    )


def _readiness_evidence(**changes) -> ReadinessEvidence:
    values = {
        "application_id": "analyst",
        "environment": "prod",
        "application_version_id": "version-1",
        "manifest": ManifestEvidence(
            schema_valid=True,
            git_sha_matches_registered_version=True,
            resources_resolve=True,
        ),
        "ownership": OwnershipEvidence(
            owner_registered=True,
            support_group_registered=True,
        ),
        "tags": TagEvidence(
            observed_tag_keys=(
                "team",
                "domain",
                "cost_center",
                "environment",
                "application_id",
            ),
            data_classification_declared=True,
        ),
        "risk": RiskEvidence(risk_tier="medium"),
        "authentication": AuthenticationEvidence(
            mode_declared=True,
            approved_ai_gateway=True,
            rate_limit_configured=True,
        ),
        "cost": CostEvidence(
            budget_policy_configured=True,
            attribution_verified=True,
        ),
        "tracing": TracingEvidence(
            configured=True,
            recent_trace_metadata_complete=True,
        ),
        "jobs": JobEvidence(
            evaluation_job_configured=True,
            promotion_job_configured=True,
        ),
        "evaluation": EvaluationEvidence(
            dataset_exists=True,
            dataset_case_count=40,
            targets_current_application_version=True,
            evaluated_at=NOW - timedelta(hours=2),
            blocking_thresholds_passed=True,
        ),
        "telemetry": TelemetryEvidence(
            monitoring_configured=True,
            service_levels_passed=True,
            required_request_tags_observed=True,
        ),
        "administrator": AdministratorEvidence(unresolved_blocking_actions=0),
    }
    values.update(changes)
    return ReadinessEvidence(**values)


def test_domain_models_are_strict_frozen_and_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        HealthProfile(
            profile_id="standard",
            version="1",
            maximum_error_rate="0.02",
            critical_error_rate=0.10,
            maximum_p95_latency_ms=8_000.0,
            critical_p95_latency_ms=15_000.0,
        )
    with pytest.raises(ValidationError):
        HealthProfile(
            profile_id="standard",
            version="1",
            maximum_error_rate=0.02,
            critical_error_rate=0.10,
            maximum_p95_latency_ms=8_000.0,
            critical_p95_latency_ms=15_000.0,
            surprise=True,
        )

    profile = _health_profile()
    with pytest.raises(ValidationError):
        profile.version = "2"


def test_role_vocabulary_is_small_and_platform_owned():
    assert {role.value for role in Role} == {
        "viewer",
        "contributor",
        "owner",
        "platform_viewer",
        "platform_administrator",
        "auditor",
    }


def test_zero_traffic_is_healthy_when_the_evidence_window_is_fresh():
    snapshot = calculate_health(
        HealthEvidence(
            application_id="analyst",
            environment="prod",
            deployment_status=DeploymentStatus.RUNNING,
            telemetry_observed_at=NOW,
            request_count=0,
            error_count=0,
        ),
        _health_profile(),
        evaluated_at=NOW,
    )
    assert snapshot.status is HealthStatus.HEALTHY
    assert any("no requests" in reason for reason in snapshot.reasons)


def test_failed_deployment_is_critical_even_when_telemetry_is_stale():
    snapshot = calculate_health(
        HealthEvidence(
            application_id="analyst",
            environment="prod",
            deployment_status=DeploymentStatus.FAILED,
            telemetry_observed_at=NOW - timedelta(days=1),
            request_count=100,
            error_count=0,
            p95_latency_ms=100.0,
        ),
        _health_profile(),
        evaluated_at=NOW,
    )
    assert snapshot.status is HealthStatus.CRITICAL
    assert snapshot.reasons[0] == "active deployment is failed"


def test_stale_evidence_is_unknown_and_slo_breaches_are_deterministic():
    stale = calculate_health(
        HealthEvidence(
            application_id="analyst",
            environment="prod",
            deployment_status=DeploymentStatus.RUNNING,
            telemetry_observed_at=NOW - timedelta(hours=2),
            request_count=100,
            error_count=0,
            p95_latency_ms=100.0,
        ),
        _health_profile(),
        evaluated_at=NOW,
    )
    assert stale.status is HealthStatus.UNKNOWN

    degraded = calculate_health(
        HealthEvidence(
            application_id="analyst",
            environment="prod",
            deployment_status=DeploymentStatus.RUNNING,
            telemetry_observed_at=NOW,
            request_count=100,
            error_count=5,
            p95_latency_ms=10_000.0,
        ),
        _health_profile(),
        evaluated_at=NOW,
    )
    assert degraded.status is HealthStatus.DEGRADED

    critical = calculate_health(
        HealthEvidence(
            application_id="analyst",
            environment="prod",
            deployment_status=DeploymentStatus.RUNNING,
            telemetry_observed_at=NOW,
            request_count=100,
            error_count=11,
            p95_latency_ms=100.0,
        ),
        _health_profile(),
        evaluated_at=NOW,
    )
    assert critical.status is HealthStatus.CRITICAL


def test_health_rejects_inconsistent_counts_and_unordered_profiles():
    with pytest.raises(ValidationError):
        HealthEvidence(
            application_id="analyst",
            environment="prod",
            deployment_status=DeploymentStatus.RUNNING,
            request_count=2,
            error_count=3,
        )
    with pytest.raises(ValidationError):
        HealthProfile(
            profile_id="bad",
            version="1",
            maximum_error_rate=0.2,
            critical_error_rate=0.1,
            maximum_p95_latency_ms=100.0,
            critical_p95_latency_ms=90.0,
        )


def test_readiness_passes_complete_evidence_and_is_immutable():
    snapshot = ReadinessEvaluator(
        ReadinessProfile(profile_id="medium-risk-production", version="1")
    ).evaluate(_readiness_evidence(), evaluated_at=NOW)

    assert snapshot.ready is True
    assert all(
        result.status in {ReadinessStatus.PASS, ReadinessStatus.NOT_APPLICABLE}
        for result in snapshot.results
    )
    severities = [result.severity for result in snapshot.results]
    first_information = severities.index(ReadinessSeverity.INFORMATIONAL)
    assert all(
        severity is ReadinessSeverity.BLOCKING
        for severity in severities[:first_information]
    )
    with pytest.raises(ValidationError):
        snapshot.ready = False


def test_readiness_reports_blocking_failures_before_non_applicable_rules():
    evidence = _readiness_evidence(
        tags=TagEvidence(
            observed_tag_keys=("team", "environment"),
            data_classification_declared=False,
        ),
        administrator=AdministratorEvidence(unresolved_blocking_actions=2),
    )
    snapshot = ReadinessEvaluator(
        ReadinessProfile(profile_id="medium-risk-production", version="7")
    ).evaluate(evidence, evaluated_at=NOW)

    assert snapshot.ready is False
    failed = [
        result for result in snapshot.results if result.status is ReadinessStatus.FAIL
    ]
    assert {result.rule_id for result in failed} >= {
        "required_tags",
        "risk_and_classification",
        "administrator_blockers",
    }
    assert snapshot.results[0].severity is ReadinessSeverity.BLOCKING
    assert snapshot.results[0].status is ReadinessStatus.FAIL
    assert all(result.rule_version == "7" for result in snapshot.results)


def test_unknown_blocking_evidence_prevents_readiness():
    evidence = _readiness_evidence(
        manifest=ManifestEvidence(
            schema_valid=True,
            git_sha_matches_registered_version=True,
            resources_resolve=None,
        )
    )
    snapshot = ReadinessEvaluator(
        ReadinessProfile(profile_id="medium-risk-production", version="1")
    ).evaluate(evidence, evaluated_at=NOW)
    resource = next(
        result for result in snapshot.results if result.rule_id == "resource_resolution"
    )
    assert resource.status is ReadinessStatus.UNKNOWN
    assert snapshot.ready is False


def test_production_budget_and_cost_attribution_fail_closed():
    evidence = _readiness_evidence(
        cost=CostEvidence(
            budget_policy_configured=None,
            attribution_verified=False,
        )
    )
    snapshot = ReadinessEvaluator(
        ReadinessProfile(
            profile_id="medium-risk-production",
            version="1",
            require_budget_policy=True,
        )
    ).evaluate(evidence, evaluated_at=NOW)

    results = {result.rule_id: result for result in snapshot.results}
    assert results["budget_policy"].status is ReadinessStatus.UNKNOWN
    assert results["cost_attribution"].status is ReadinessStatus.FAIL
    assert snapshot.ready is False


def test_new_application_exception_covers_only_unknown_service_levels():
    evidence = _readiness_evidence(
        telemetry=TelemetryEvidence(
            monitoring_configured=True,
            service_levels_passed=None,
            approved_new_application_exception=True,
            required_request_tags_observed=True,
        )
    )
    snapshot = ReadinessEvaluator(
        ReadinessProfile(profile_id="medium-risk-production", version="1")
    ).evaluate(evidence, evaluated_at=NOW)
    service_levels = next(
        result for result in snapshot.results if result.rule_id == "service_levels"
    )
    assert service_levels.status is ReadinessStatus.PASS
    assert snapshot.ready is True
