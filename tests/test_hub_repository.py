"""Atomic persistence semantics shared by durable Hub repository adapters."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from aai_console.hub.models import (
    ActionEntityType,
    ApplicationPrincipalRecord,
    ApplicationRecord,
    ApplicationVersionRecord,
    AuthorizationContext,
    EvaluationRunRecord,
    EvaluationStatus,
    PrincipalType,
    PromotionRequestRecord,
    PromotionStatus,
    ReadinessSnapshot,
    Role,
)
from aai_console.hub.repository import (
    DuplicateActiveEvaluationError,
    DuplicateActivePromotionError,
    FourEyesViolationError,
    HubAuthorizationError,
    HubConflictError,
    HubRepositoryUnavailableError,
    ImmutableApplicationIdConflictError,
    InMemoryHubRepository,
    OptimisticConcurrencyError,
    UnavailableHubRepository,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def _application(application_id="analyst", *, updated_at=NOW):
    return ApplicationRecord(
        application_id=application_id,
        name="Analyst",
        description="A governed assistant",
        owner_principal="owner@example.com",
        support_group="analyst-support",
        business_domain="investments",
        cost_center="technology",
        risk_tier="medium",
        lifecycle_state="development",
        created_at=NOW,
        updated_at=updated_at,
    )


def _version(
    version_id="version-1",
    *,
    application_id="analyst",
    environment="dev",
    git_repository="https://github.com/aai-test/analyst",
    git_commit_sha="a" * 40,
    manifest_hash="a" * 64,
    registered_at=NOW,
    deployment_target=None,
):
    return ApplicationVersionRecord(
        version_id=version_id,
        application_id=application_id,
        environment=environment,
        git_repository=git_repository,
        git_commit_sha=git_commit_sha,
        manifest_version="ai-platform/v1",
        manifest_hash=manifest_hash,
        manifest_json="{}",
        registered_by="ci@example.com",
        registered_at=registered_at,
        deployment_target=deployment_target or environment,
    )


def _ready_snapshot(
    *,
    application_id="analyst",
    environment="prod",
    application_version_id="version-1",
):
    return ReadinessSnapshot(
        application_id=application_id,
        environment=environment,
        application_version_id=application_version_id,
        profile_id="medium-risk-production",
        profile_version="1",
        evaluated_at=NOW,
        ready=True,
        results=(),
    )


def _evaluation(run_id="eval-1"):
    return EvaluationRunRecord(
        evaluation_run_id=run_id,
        application_id="analyst",
        environment="dev",
        application_version_id="version-1",
        evaluation_profile="grounded-agent-v1",
        dataset_name="catalog.evaluations.analyst",
        dataset_version="dataset-1",
        job_id="123",
        requested_by="owner@example.com",
        status=EvaluationStatus.REQUESTED,
        requested_at=NOW,
    )


def _promotion(request_id="promotion-1"):
    return PromotionRequestRecord(
        promotion_request_id=request_id,
        application_id="analyst",
        source_environment="dev",
        target_environment="prod",
        application_version_id="version-1",
        requested_by="owner@example.com",
        requested_at=NOW,
        status=PromotionStatus.PENDING_REVIEW,
        readiness_snapshot=_ready_snapshot(),
        promotion_job_id="456",
    )


def _registered_repository():
    repository = InMemoryHubRepository()
    repository.register_application(_application(), _version())
    return repository


def test_registration_is_idempotent_and_moves_environment_pointer_atomically():
    repository = InMemoryHubRepository()
    first = repository.register_application(_application(), _version())
    replay = repository.register_application(_application(), _version())
    assert first.created is True
    assert replay.created is False
    assert replay.version.version_id == first.version.version_id
    assert len(repository.list_versions("analyst")) == 1

    second = repository.register_application(
        _application(updated_at=NOW + timedelta(minutes=1)),
        _version(
            "version-2",
            git_commit_sha="b" * 40,
            manifest_hash="b" * 64,
            registered_at=NOW + timedelta(minutes=1),
        ),
    )
    versions = repository.list_versions("analyst")
    assert second.created is True
    assert repository.get_current_version("analyst", "dev").version_id == "version-2"
    assert [version.is_current for version in versions] == [False, True]
    assert repository.get_application("analyst").row_version == 2


def test_idempotent_registration_cannot_retarget_the_same_version():
    repository = InMemoryHubRepository()
    repository.register_application(_application(), _version())

    with pytest.raises(HubConflictError, match="deployment_target"):
        repository.register_application(
            _application(),
            _version(deployment_target="another-target"),
        )


def test_registration_rejects_an_immutable_id_bound_to_another_repository():
    repository = _registered_repository()
    with pytest.raises(ImmutableApplicationIdConflictError):
        repository.register_application(
            _application(updated_at=NOW + timedelta(minutes=1)),
            _version(
                "version-2",
                git_repository="https://github.com/aai-test/other-analyst",
                git_commit_sha="b" * 40,
                manifest_hash="b" * 64,
                registered_at=NOW + timedelta(minutes=1),
            ),
        )


def test_concurrent_registration_creates_exactly_one_version():
    repository = InMemoryHubRepository()

    def register():
        return repository.register_application(_application(), _version()).created

    with ThreadPoolExecutor(max_workers=8) as pool:
        created = list(pool.map(lambda _: register(), range(20)))
    assert created.count(True) == 1
    assert created.count(False) == 19
    assert len(repository.list_versions("analyst")) == 1


def test_visibility_uses_explicit_principals_ownership_groups_and_fleet_roles():
    repository = _registered_repository()
    repository.upsert_application_principal(
        ApplicationPrincipalRecord(
            application_id="analyst",
            principal_type=PrincipalType.GROUP,
            principal_name="analyst-contributors",
            application_role=Role.CONTRIBUTOR,
        )
    )
    repository.upsert_application_principal(
        ApplicationPrincipalRecord(
            application_id="analyst",
            principal_type=PrincipalType.GROUP,
            principal_name="analyst-support",
            application_role=Role.CONTRIBUTOR,
        )
    )
    unrelated = AuthorizationContext(principal="other@example.com")
    assert repository.list_visible_applications(unrelated) == ()

    contributor = AuthorizationContext(
        principal="contributor@example.com",
        groups=("analyst-contributors",),
    )
    assert [
        item.application_id
        for item in repository.list_visible_applications(contributor)
    ] == ["analyst"]

    support = AuthorizationContext(
        principal="supporter@example.com",
        groups=("analyst-support",),
    )
    assert repository.list_visible_applications(support)

    fleet = AuthorizationContext(
        principal="auditor@example.com",
        platform_roles=(Role.AUDITOR,),
    )
    assert repository.list_visible_applications(fleet)


def test_evaluation_deduplicates_active_runs_and_enforces_row_versions():
    repository = _registered_repository()
    requested = repository.create_evaluation(_evaluation())
    assert requested.status is EvaluationStatus.REQUESTED

    with pytest.raises(DuplicateActiveEvaluationError):
        repository.create_evaluation(_evaluation("eval-2"))

    running = requested.model_copy(
        update={
            "status": EvaluationStatus.RUNNING,
            "started_at": NOW + timedelta(seconds=5),
            "job_run_id": "run-123",
            "row_version": 2,
        }
    )
    running = EvaluationRunRecord.model_validate(running.model_dump(mode="python"))
    repository.update_evaluation(running, expected_row_version=1)

    with pytest.raises(OptimisticConcurrencyError):
        repository.update_evaluation(running, expected_row_version=1)

    succeeded = running.model_copy(
        update={
            "status": EvaluationStatus.SUCCEEDED,
            "completed_at": NOW + timedelta(minutes=5),
            "mlflow_run_id": "mlflow-1",
            "summary_json": (
                '{"dataset_exists":true,"dataset_case_count":30,'
                '"blocking_thresholds_passed":true,'
                '"metrics":{"groundedness":0.91}}'
            ),
            "row_version": 3,
        }
    )
    succeeded = EvaluationRunRecord.model_validate(succeeded.model_dump(mode="python"))
    repository.update_evaluation(succeeded, expected_row_version=2)
    stored = repository.get_evaluation("eval-1")
    assert stored.status is EvaluationStatus.SUCCEEDED
    assert stored.summary_json == (
        '{"blocking_thresholds_passed":true,"dataset_case_count":30,'
        '"dataset_exists":true,"metrics":{"groundedness":0.91}}'
    )

    # A terminal predecessor no longer blocks a deliberate new run.
    assert repository.create_evaluation(_evaluation("eval-2"))


def test_promotion_deduplicates_active_requests():
    repository = _registered_repository()
    repository.create_promotion_request(_promotion())
    with pytest.raises(DuplicateActivePromotionError):
        repository.create_promotion_request(_promotion("promotion-2"))


def test_promotion_approval_enforces_admin_four_eyes_and_concurrency():
    repository = _registered_repository()
    repository.create_promotion_request(_promotion())

    non_admin = AuthorizationContext(principal="reviewer@example.com")
    with pytest.raises(HubAuthorizationError):
        repository.approve_promotion(
            "promotion-1",
            actor=non_admin,
            expected_row_version=1,
            reviewed_at=NOW + timedelta(minutes=1),
            actor_request_id="request-1",
        )

    requester_admin = AuthorizationContext(
        principal="owner@example.com",
        platform_roles=(Role.PLATFORM_ADMINISTRATOR,),
    )
    with pytest.raises(FourEyesViolationError):
        repository.approve_promotion(
            "promotion-1",
            actor=requester_admin,
            expected_row_version=1,
            reviewed_at=NOW + timedelta(minutes=1),
            actor_request_id="request-2",
        )

    reviewer = AuthorizationContext(
        principal="reviewer@example.com",
        platform_roles=(Role.PLATFORM_ADMINISTRATOR,),
    )
    approved = repository.approve_promotion(
        "promotion-1",
        actor=reviewer,
        expected_row_version=1,
        reviewed_at=NOW + timedelta(minutes=1),
        actor_request_id="request-3",
        comment="Ready for controlled execution.",
    )
    assert approved.status is PromotionStatus.APPROVED
    assert approved.reviewed_by == "reviewer@example.com"
    assert approved.row_version == 2

    with pytest.raises(OptimisticConcurrencyError):
        repository.approve_promotion(
            "promotion-1",
            actor=reviewer,
            expected_row_version=1,
            reviewed_at=NOW + timedelta(minutes=2),
            actor_request_id="request-4",
        )

    events = repository.list_action_events(
        entity_type=ActionEntityType.PROMOTION,
        entity_id="promotion-1",
    )
    assert [event.new_state for event in events] == [
        PromotionStatus.PENDING_REVIEW.value,
        PromotionStatus.APPROVED.value,
    ]


def test_rejection_requires_comment_and_events_are_append_only():
    repository = _registered_repository()
    repository.create_promotion_request(_promotion())
    reviewer = AuthorizationContext(
        principal="reviewer@example.com",
        platform_roles=(Role.PLATFORM_ADMINISTRATOR,),
    )
    with pytest.raises(HubConflictError):
        repository.reject_promotion(
            "promotion-1",
            actor=reviewer,
            expected_row_version=1,
            reviewed_at=NOW + timedelta(minutes=1),
            actor_request_id="request-1",
            comment="  ",
        )
    rejected = repository.reject_promotion(
        "promotion-1",
        actor=reviewer,
        expected_row_version=1,
        reviewed_at=NOW + timedelta(minutes=1),
        actor_request_id="request-2",
        comment="Missing external approval.",
    )
    assert rejected.status is PromotionStatus.REJECTED
    events = repository.list_action_events(entity_id="promotion-1")
    with pytest.raises(HubConflictError):
        repository.append_action_event(events[0])


def test_unavailable_repository_fails_closed_for_reads_and_writes():
    repository = UnavailableHubRepository("Lakebase is not configured")
    assert repository.available is False
    with pytest.raises(HubRepositoryUnavailableError, match="not configured"):
        repository.list_visible_applications(
            AuthorizationContext(principal="viewer@example.com")
        )
    with pytest.raises(HubRepositoryUnavailableError, match="not configured"):
        repository.register_application(_application(), _version())
