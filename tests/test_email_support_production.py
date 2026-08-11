"""Credential-free tests for the email-support production evidence policy."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "email-support-agent" / "src"
_ADDED_SOURCE = str(SOURCE) not in sys.path
if _ADDED_SOURCE:
    sys.path.insert(0, str(SOURCE))

from email_support_agent.production import (  # noqa: E402
    CanaryEvidence,
    CostEvidence,
    DependencyObservation,
    EvidenceOrigin,
    EvidenceProvenance,
    IngressVerificationEvidence,
    LoadEvidence,
    OperationalCapability,
    OperationalEvidence,
    ProductionEvidencePack,
    ProductionReadinessPolicy,
    ReadinessStatus,
    ReleaseComponents,
    ReleaseLineageEvidence,
    ReliabilityEvidence,
    RestoreEvidence,
    SecurityEvidence,
    SloObservationEvidence,
    evaluate_production_readiness,
    illustrative_policy,
    release_digest,
    seal_evidence,
)

if _ADDED_SOURCE:
    sys.path.remove(str(SOURCE))


NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def _digest(value: str) -> str:
    return "sha256:" + sha256(value.encode()).hexdigest()


def _provenance(
    origin: EvidenceOrigin,
    release: str,
    name: str,
    *,
    observed_at: datetime | None = None,
    valid_until: datetime | None = None,
) -> EvidenceProvenance:
    observed = observed_at or NOW - timedelta(hours=1)
    valid = valid_until or NOW + timedelta(hours=1)
    scheme = "synthetic" if origin is EvidenceOrigin.SYNTHETIC else "secure"
    return EvidenceProvenance(
        origin=origin,
        application_id="email-support-agent",
        environment="production",
        evidence_ref=f"{scheme}://evidence/test-{name}",
        attestation_digest=_digest(f"attestation:{name}"),
        target_release_digest=release,
        observed_at=observed,
        valid_until=valid,
    )


def _components() -> ReleaseComponents:
    return ReleaseComponents(
        application_release="email-support-2026.08.10",
        git_commit="a" * 40,
        artifact_digest=_digest("artifact"),
        prompt_release="support-prompts-v4",
        model_release="gateway:gpt-support-v3",
        tool_release="support-tools-v2",
        knowledge_release="kb-2026-08-01",
        embedding_release="embedding-v2",
        chunking_release="chunking-v3",
        policy_release="support-policy-v5",
        schema_release="email-contract-v3",
    )


def _approved_policy() -> ProductionReadinessPolicy:
    return ProductionReadinessPolicy(
        application_id="email-support-agent",
        environment="production",
        profile_id="owner-approved-email-support-v1",
        version="1",
        owner_group="group:support-platform-owners",
        approved_for_production=True,
        approval_ref="secure://policy-approval/test-owner-approval",
    )


def _complete_pack() -> ProductionEvidencePack:
    components = _components()
    target = release_digest(components)
    dependencies = tuple(
        DependencyObservation(
            capability=capability,
            configured=True,
            health_check_passed=True,
            authorization_check_passed=True,
            least_privilege_reviewed=True,
            failure_policy_configured=True,
        )
        for capability in OperationalCapability
    )
    pack = ProductionEvidencePack(
        application_id="email-support-agent",
        environment="production",
        target_release_digest=target,
        assembled_at=NOW - timedelta(minutes=1),
        ingress=IngressVerificationEvidence(
            provenance=_provenance(EvidenceOrigin.PLATFORM_CONTROL, target, "ingress"),
            messages_observed=1_000,
            authenticated_messages=1_000,
            malware_scanned_messages=1_000,
            dlp_processed_messages=1_000,
            access_context_verified_messages=1_000,
            raw_payload_checkpoint_count=0,
        ),
        operations=OperationalEvidence(
            provenance=_provenance(
                EvidenceOrigin.PLATFORM_CONTROL, target, "operations"
            ),
            dependencies=dependencies,
        ),
        lineage=ReleaseLineageEvidence(
            provenance=_provenance(EvidenceOrigin.PLATFORM_CONTROL, target, "lineage"),
            components=components,
            declared_release_digest=target,
            evaluated_release_digest=target,
            deployed_release_digest=target,
            canary_release_digest=target,
        ),
        slo=SloObservationEvidence(
            provenance=_provenance(EvidenceOrigin.PLATFORM_CONTROL, target, "slo"),
            window_seconds=3_600,
            request_count=1_000,
            error_count=5,
            available_request_count=1_000,
            p95_latency_ms=1_500.0,
            p95_review_wait_seconds=120.0,
        ),
        reliability=ReliabilityEvidence(
            provenance=_provenance(
                EvidenceOrigin.PLATFORM_CONTROL, target, "reliability"
            ),
            scenarios_executed=5,
            scenarios_passed=5,
            delivery_attempts=1_000,
            successful_deliveries=1_000,
            idempotency_replay_cases=100,
            idempotency_replay_passes=100,
            duplicate_send_count=0,
            checkpoint_resume_passed=True,
            outbox_recovery_passed=True,
            retry_budget_enforced=True,
            dead_letter_recovery_passed=True,
        ),
        security=SecurityEvidence(
            provenance=_provenance(EvidenceOrigin.SECURITY_CONTROL, target, "security"),
            security_test_cases=100,
            cross_tenant_test_cases=25,
            unauthorized_send_test_cases=25,
            high_risk_routing_cases=25,
            cross_tenant_leakage_count=0,
            unauthorized_send_count=0,
            high_risk_false_auto_send_count=0,
            dlp_admission_passed=True,
            trusted_access_context_passed=True,
            reviewer_authorization_passed=True,
            trace_policy_passed=True,
            prompt_injection_precheck_passed=True,
            least_privilege_review_passed=True,
        ),
        load=LoadEvidence(
            provenance=_provenance(EvidenceOrigin.PERFORMANCE_HARNESS, target, "load"),
            duration_seconds=900,
            request_count=10_000,
            error_count=50,
            target_rps=10.0,
            achieved_rps=12.0,
            p95_latency_ms=2_000.0,
            peak_resource_saturation=0.50,
        ),
        restore=RestoreEvidence(
            provenance=_provenance(EvidenceOrigin.RESTORE_HARNESS, target, "restore"),
            restored_release_digest=target,
            restore_succeeded=True,
            observed_rto_seconds=600.0,
            observed_rpo_seconds=60.0,
            checkpoint_integrity_passed=True,
            outbox_reconciled=True,
            duplicate_send_count=0,
        ),
        canary=CanaryEvidence(
            provenance=_provenance(EvidenceOrigin.CANARY_TELEMETRY, target, "canary"),
            duration_seconds=1_800,
            request_count=1_000,
            error_count=5,
            traffic_fraction=0.05,
            p95_latency_ms=2_000.0,
            traced_request_count=1_000,
            rollback_ready=True,
            cross_tenant_leakage_count=0,
            unauthorized_send_count=0,
            duplicate_send_count=0,
            high_risk_false_auto_send_count=0,
        ),
        cost=CostEvidence(
            provenance=_provenance(EvidenceOrigin.FINOPS_LEDGER, target, "cost"),
            requests_observed=1_000,
            requests_costed=1_000,
            resolved_cases=800,
            inference_cost_usd=30.0,
            operational_cost_usd=20.0,
            p95_model_calls=2.0,
        ),
    )
    return seal_evidence(pack)


def _check(scorecard, check_id: str):
    return next(item for item in scorecard.checks if item.check_id == check_id)


def _score(pack: ProductionEvidencePack, policy=None):
    return evaluate_production_readiness(
        pack,
        policy or _approved_policy(),
        evaluated_at=NOW,
    )


def _replace_section(pack: ProductionEvidencePack, name: str, **changes):
    section = getattr(pack, name)
    assert section is not None
    changed = section.model_copy(update=changes)
    return seal_evidence(pack.model_copy(update={name: changed}))


def test_contracts_are_strict_immutable_and_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        ProductionReadinessPolicy(
            application_id="email-support-agent",
            environment="production",
            profile_id="strict-policy",
            version="1",
            owner_group="group:owners",
            approved_for_production=False,
            maximum_evidence_age_hours="24",
        )
    with pytest.raises(ValidationError):
        ProductionReadinessPolicy(
            application_id="email-support-agent",
            environment="production",
            profile_id="strict-policy",
            version="1",
            owner_group="group:owners",
            approved_for_production=False,
            surprise=True,
        )

    pack = _complete_pack()
    with pytest.raises(ValidationError):
        pack.environment = "test"


def test_complete_attested_shape_can_pass_but_does_not_validate_external_systems():
    """This unit fixture exercises policy logic; it is not external evidence."""

    scorecard = _score(_complete_pack())
    assert scorecard.ready is True
    assert scorecard.score == 1.0
    assert scorecard.passed_checks == scorecard.required_checks
    assert all(item.status is ReadinessStatus.PASS for item in scorecard.checks)


def test_illustrative_policy_and_synthetic_evidence_cannot_claim_readiness():
    pack = _complete_pack()
    illustrative = _score(pack, illustrative_policy())
    assert illustrative.ready is False
    assert _check(illustrative, "policy_approval").status is ReadinessStatus.UNVERIFIED

    assert pack.load is not None
    synthetic_provenance = _provenance(
        EvidenceOrigin.SYNTHETIC,
        pack.target_release_digest,
        "synthetic-load",
    )
    synthetic_pack = _replace_section(
        pack,
        "load",
        provenance=synthetic_provenance,
    )
    synthetic = _score(synthetic_pack)
    assert synthetic.ready is False
    assert _check(synthetic, "load_test").status is ReadinessStatus.UNVERIFIED


def test_missing_required_evidence_fails_closed():
    pack = seal_evidence(_complete_pack().model_copy(update={"cost": None}))
    scorecard = _score(pack)
    assert scorecard.ready is False
    assert _check(scorecard, "unit_economics").status is ReadinessStatus.MISSING


def test_stale_evidence_fails_closed_even_when_metrics_pass():
    pack = _complete_pack()
    assert pack.ingress is not None
    stale_provenance = _provenance(
        EvidenceOrigin.PLATFORM_CONTROL,
        pack.target_release_digest,
        "stale-ingress",
        observed_at=NOW - timedelta(days=3),
        valid_until=NOW - timedelta(days=2),
    )
    stale_pack = _replace_section(pack, "ingress", provenance=stale_provenance)
    scorecard = _score(stale_pack)
    assert scorecard.ready is False
    assert _check(scorecard, "ingress_verification").status is ReadinessStatus.STALE


def test_release_mismatches_fail_at_provenance_and_lineage_boundaries():
    pack = _complete_pack()
    other = _digest("other-release")
    mismatched_provenance = _provenance(
        EvidenceOrigin.CANARY_TELEMETRY,
        other,
        "wrong-release-canary",
    )
    canary_pack = _replace_section(
        pack,
        "canary",
        provenance=mismatched_provenance,
    )
    canary_score = _score(canary_pack)
    assert _check(canary_score, "canary").status is ReadinessStatus.MISMATCH

    lineage_pack = _replace_section(
        pack,
        "lineage",
        deployed_release_digest=other,
    )
    lineage_score = _score(lineage_pack)
    assert _check(lineage_score, "release_lineage").status is ReadinessStatus.MISMATCH


def test_pack_digest_detects_mutation_after_sealing():
    pack = _complete_pack()
    assert pack.cost is not None
    changed_cost = pack.cost.model_copy(update={"inference_cost_usd": 999.0})
    tampered = pack.model_copy(update={"cost": changed_cost})

    scorecard = _score(tampered)
    assert scorecard.ready is False
    assert _check(scorecard, "evidence_integrity").status is ReadinessStatus.TAMPERED


@pytest.mark.parametrize(
    ("section", "field", "check_id"),
    (
        ("security", "cross_tenant_leakage_count", "zero_cross_tenant_leakage"),
        ("security", "unauthorized_send_count", "zero_unauthorized_sends"),
        ("reliability", "duplicate_send_count", "zero_duplicate_sends"),
        (
            "security",
            "high_risk_false_auto_send_count",
            "zero_high_risk_false_auto_sends",
        ),
    ),
)
def test_zero_tolerance_invariants_cannot_be_relaxed(section, field, check_id):
    pack = _replace_section(_complete_pack(), section, **{field: 1})
    scorecard = _score(pack)
    assert scorecard.ready is False
    assert _check(scorecard, check_id).status is ReadinessStatus.FAIL


@pytest.mark.parametrize(
    ("section", "changes", "check_id"),
    (
        ("slo", {"error_count": 11}, "service_levels"),
        ("load", {"achieved_rps": 9.0}, "load_test"),
        ("restore", {"observed_rto_seconds": 1_801.0}, "restore_test"),
        ("cost", {"requests_costed": 900}, "unit_economics"),
    ),
)
def test_owner_threshold_breaches_block_readiness(section, changes, check_id):
    pack = _replace_section(_complete_pack(), section, **changes)
    scorecard = _score(pack)
    assert scorecard.ready is False
    assert _check(scorecard, check_id).status is ReadinessStatus.FAIL
