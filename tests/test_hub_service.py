"""Focused application-service tests for governed Hub workflows."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from aai_console.hub.jobs import RecordingJobRunner
from aai_console.hub.models import (
    ApplicationPrincipalRecord,
    AuthorizationContext,
    EvaluationRunRecord,
    EvaluationStatus,
    PrincipalType,
    PromotionStatus,
    ReadinessRuleResult,
    ReadinessSeverity,
    ReadinessSnapshot,
    ReadinessStatus,
    Role,
)
from aai_console.hub.repository import (
    DuplicateActiveEvaluationError,
    HubConflictError,
    HubNotFoundError,
    ImmutableApplicationIdConflictError,
    InMemoryHubRepository,
)
from aai_console.hub.service import (
    EvaluationRequest,
    HubCapabilityUnavailableError,
    HubPermissionDeniedError,
    HubQueryValidationError,
    HubReadinessBlockedError,
    HubService,
    PortfolioQuery,
    PromotionRequest,
    PromotionReviewRequest,
    RegistrationRequest,
    parse_tag_filters,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
CI_PRINCIPAL = "github-actions@example.com"
CI_ACTOR = AuthorizationContext(principal=CI_PRINCIPAL)
FLEET_VIEWER = AuthorizationContext(
    principal="platform-viewer@example.com",
    platform_roles=(Role.PLATFORM_VIEWER,),
)


def _manifest(
    application_id: str = "analyst",
    *,
    name: str | None = None,
    repository_url: str | None = None,
    owner: str | None = None,
    support_group: str | None = None,
    team: str = "investment_ai",
    domain: str = "investments",
    cost_center: str = "CC-100",
    evaluation_job_id: str | None = "123",
    evaluation_job_key: str | None = None,
    promotion_job_id: str | None = None,
    environments: tuple[str, ...] = ("dev",),
) -> dict:
    if evaluation_job_id is None and evaluation_job_key is None:
        evaluation_job_key = "release_gate"
    resources: dict[str, object] = {}
    if evaluation_job_id is not None:
        resources["evaluationJobId"] = evaluation_job_id
    if evaluation_job_key is not None:
        resources["evaluationJobKey"] = evaluation_job_key
    if promotion_job_id is not None:
        resources["promotionJobId"] = promotion_job_id

    environment_specs = {
        environment: {
            "workspaceId": "123",
            "databricksAppName": (f"{application_id.replace('_', '-')}-{environment}"),
            "mlflowExperimentId": "456",
            "aiGatewayService": "ai_platform.models.enterprise_chat",
            "tags": {"environment": environment},
        }
        for environment in environments
    }
    return {
        "apiVersion": "ai-platform/v1",
        "kind": "AIApplication",
        "metadata": {
            "id": application_id,
            "name": name or application_id.replace("_", " ").title(),
            "description": f"Governed {application_id} application.",
            "owner": owner or f"owner-{application_id}@example.com",
            "supportGroup": support_group or f"group:{application_id}-support",
            "businessDomain": domain,
            "costCenter": cost_center,
            "riskTier": "medium",
            "tags": {
                "application_id": application_id,
                "team": team,
                "domain": domain,
                "cost_center": cost_center,
                "data_classification": "internal",
                "lifecycle": "experimental",
            },
        },
        "spec": {
            "repository": {
                "url": repository_url or f"https://github.com/aai-test/{application_id}"
            },
            "authorization": {"mode": "application"},
            "environments": environment_specs,
            "resources": resources,
            "evaluation": {
                "profile": "grounded_agent_v1",
                "dataset": f"ai_platform.evaluations.{application_id}",
                "minimumCases": 30,
                "maximumAgeHours": 168,
                "thresholds": {
                    "groundedness": 0.85,
                    "safety_pass_rate": 1.0,
                },
            },
            "readiness": {"profile": "medium_risk_production_v1"},
            "costControls": {"budgetPolicy": "platform_standard_v1"},
            "serviceLevels": {
                "maximumErrorRate": 0.02,
                "p95LatencyMs": 8000,
            },
        },
    }


def _service(
    *,
    job_runner: RecordingJobRunner | None = None,
    registration_principals: frozenset[str] = frozenset({CI_PRINCIPAL}),
) -> tuple[InMemoryHubRepository, HubService]:
    repository = InMemoryHubRepository()
    service = HubService(
        repository,
        registration_principals=registration_principals,
        job_runner=job_runner,
        clock=lambda: NOW,
    )
    return repository, service


def _register(
    service: HubService,
    manifest: dict,
    *,
    actor: AuthorizationContext = CI_ACTOR,
    environment: str = "dev",
    git_sha: str = "a" * 40,
    authorize: bool = True,
):
    result = service.register(
        RegistrationRequest(
            manifest=manifest,
            environment=environment,
            gitCommitSha=git_sha,
            deploymentTarget=f"{environment}-bundle",
        ),
        actor=actor,
    )
    if authorize:
        for principal in (
            ApplicationPrincipalRecord(
                application_id=result.application.application_id,
                principal_type=PrincipalType.USER,
                principal_name=result.application.owner_principal,
                application_role=Role.OWNER,
            ),
            ApplicationPrincipalRecord(
                application_id=result.application.application_id,
                principal_type=PrincipalType.GROUP,
                principal_name=result.application.support_group,
                application_role=Role.CONTRIBUTOR,
            ),
        ):
            service.repository.upsert_application_principal(principal)
    return result


def _ready_snapshot_for(version, *, evidence: str = "all checks passed"):
    result = ReadinessRuleResult(
        rule_id="promotion_gate",
        rule_version="1",
        description="Promotion evidence remains valid.",
        severity=ReadinessSeverity.BLOCKING,
        status=ReadinessStatus.PASS,
        evidence=(evidence,),
        evaluated_at=NOW,
    )
    return ReadinessSnapshot(
        application_id=version.application_id,
        environment=version.environment,
        application_version_id=version.version_id,
        profile_id="medium_risk_production_v1",
        profile_version="1",
        evaluated_at=NOW,
        ready=True,
        results=(result,),
    )


def test_registration_is_idempotent_for_the_same_manifest_and_git_sha():
    repository, service = _service()
    manifest = _manifest()

    first = _register(service, manifest)
    replay = _register(service, manifest)

    assert first.created is True
    assert replay.created is False
    assert replay.application == first.application
    assert replay.version == first.version
    assert len(repository.list_versions("analyst")) == 1
    assert len(repository.list_action_events()) == 1


def test_registration_rejects_an_application_id_bound_to_another_repository():
    _, service = _service()
    _register(
        service,
        _manifest(repository_url="https://github.com/aai-test/analyst"),
    )

    with pytest.raises(
        ImmutableApplicationIdConflictError,
        match="already bound to another Git repository",
    ):
        _register(
            service,
            _manifest(repository_url="https://github.com/aai-test/other-analyst"),
            git_sha="b" * 40,
        )


def test_registration_requires_the_case_insensitive_ci_allowlist():
    repository, service = _service(
        registration_principals=frozenset({CI_PRINCIPAL.upper()})
    )
    assert service.registration_enabled is True

    result = _register(service, _manifest())
    assert result.created is True

    denied = AuthorizationContext(principal="unapproved-ci@example.com")
    with pytest.raises(HubPermissionDeniedError, match="approved for registration"):
        _register(service, _manifest("unapproved"), actor=denied)
    with pytest.raises(HubNotFoundError):
        repository.get_application("unapproved")

    _, disabled = _service(registration_principals=frozenset())
    assert disabled.registration_enabled is False
    with pytest.raises(HubPermissionDeniedError):
        _register(disabled, _manifest("disabled"))


def test_visibility_is_fail_closed_and_direct_reads_do_not_create_an_idor():
    _, service = _service()
    _register(
        service,
        _manifest(
            "private_app",
            owner="private-owner@example.com",
            support_group="group:private-support",
        ),
    )
    stranger = AuthorizationContext(principal="stranger@example.com")

    assert service.list_portfolio(stranger, PortfolioQuery()).items == ()
    for application_id in ("private_app", "does_not_exist"):
        with pytest.raises(
            HubNotFoundError,
            match=rf"application {application_id!r} was not found",
        ):
            service.get_application(application_id, actor=stranger)
    with pytest.raises(HubNotFoundError):
        service.get_application_version("private_app", actor=stranger)
    with pytest.raises(HubNotFoundError):
        service.list_versions("private_app", actor=stranger)
    with pytest.raises(HubNotFoundError):
        service.list_evaluations("private_app", actor=stranger)

    owner = AuthorizationContext(principal="private-owner@example.com")
    support = AuthorizationContext(
        principal="supporter@example.com",
        groups=("group:private-support",),
    )
    assert service.get_application("private_app", actor=owner).application_id == (
        "private_app"
    )
    assert service.get_application("private_app", actor=support).application_id == (
        "private_app"
    )
    fleet_application = service.get_application("private_app", actor=FLEET_VIEWER)
    assert fleet_application.application_id == "private_app"


def test_manifest_principal_metadata_does_not_self_grant_access():
    repository, service = _service()
    _register(
        service,
        _manifest(
            "descriptive_only",
            owner="claimed-owner@example.com",
            support_group="group:claimed-support",
        ),
        authorize=False,
    )

    assert repository.list_application_principals("descriptive_only") == ()
    for actor in (
        AuthorizationContext(principal="claimed-owner@example.com"),
        AuthorizationContext(
            principal="supporter@example.com",
            groups=("group:claimed-support",),
        ),
    ):
        assert service.list_portfolio(actor, PortfolioQuery()).items == ()
        with pytest.raises(HubNotFoundError):
            service.get_application("descriptive_only", actor=actor)


def test_tag_filters_apply_or_within_a_key_and_and_across_keys():
    _, service = _service()
    _register(service, _manifest("alpha", team="blue", domain="investments"))
    _register(service, _manifest("beta", team="green", domain="investments"))
    _register(service, _manifest("gamma", team="blue", domain="research"))

    parsed = parse_tag_filters("team=blue|green,domain=investments")
    assert parsed == {
        "team": frozenset({"blue", "green"}),
        "domain": frozenset({"investments"}),
    }

    page = service.list_portfolio(
        FLEET_VIEWER,
        PortfolioQuery(tag_filters=parsed),
    )
    assert [item.application.application_id for item in page.items] == [
        "alpha",
        "beta",
    ]


@pytest.mark.parametrize(
    "expression",
    [
        "team",
        "Team=blue",
        "team-name=blue",
        "team=",
        "team=blue||green",
        "team=blue,team=green",
        "team=blue,,domain=investments",
        "team=blue\nred",
        f"team={'x' * 2_001}",
    ],
)
def test_tag_filter_parser_rejects_ambiguous_or_unsafe_grammar(expression):
    with pytest.raises(HubQueryValidationError):
        parse_tag_filters(expression)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sort", "name; DROP TABLE applications"),
        ("sort", "__class__"),
        ("sort", "application,-owner"),
        ("lifecycle", "all"),
        ("health", "HEALTHY OR TRUE"),
        ("ownership", "everyone"),
    ],
)
def test_portfolio_query_rejects_non_allowlisted_filters_and_sorts(field, value):
    with pytest.raises(ValidationError):
        PortfolioQuery(**{field: value})


def test_portfolio_query_accepts_candidate_lifecycle():
    assert PortfolioQuery(lifecycle="candidate").lifecycle == "candidate"


def test_portfolio_pagination_is_stable_and_reports_total_pages():
    _, service = _service()
    for application_id, name in (
        ("echo", "Echo"),
        ("alpha", "Alpha"),
        ("delta", "Delta"),
        ("bravo", "Bravo"),
        ("charlie", "Charlie"),
    ):
        _register(service, _manifest(application_id, name=name))

    page = service.list_portfolio(
        FLEET_VIEWER,
        PortfolioQuery(page=2, page_size=2, sort="application"),
    )

    assert page.total == 5
    assert page.pages == 3
    assert page.page == 2
    assert page.page_size == 2
    assert [item.application.name for item in page.items] == ["Charlie", "Delta"]

    beyond_end = service.list_portfolio(
        FLEET_VIEWER,
        PortfolioQuery(page=4, page_size=2),
    )
    assert beyond_end.total == 5
    assert beyond_end.pages == 3
    assert beyond_end.items == ()


def test_readiness_is_unknown_and_fail_closed_without_serving_evidence():
    _, service = _service()
    registered = _register(
        service,
        _manifest(
            promotion_job_id="456",
            environments=("dev", "prod"),
        ),
    )

    snapshot = service.readiness_for_version(registered.version)
    results = {result.rule_id: result for result in snapshot.results}

    assert snapshot.ready is False
    assert snapshot.evaluated_at == NOW
    assert snapshot.profile_version == "2"
    for rule_id in (
        "production_monitoring",
        "service_levels",
        "approved_ai_gateway",
        "observed_request_tags",
        "rate_limiting",
        "budget_policy",
        "cost_attribution",
    ):
        assert results[rule_id].status is ReadinessStatus.UNKNOWN
        assert results[rule_id].severity is ReadinessSeverity.BLOCKING
    assert results["manifest_schema"].status is ReadinessStatus.PASS
    assert results["workflow_jobs"].status is ReadinessStatus.PASS


def test_legacy_manifest_without_budget_policy_fails_production_readiness():
    _, service = _service()
    manifest = _manifest(
        promotion_job_id="456",
        environments=("dev", "prod"),
    )
    del manifest["spec"]["costControls"]
    registered = _register(service, manifest)

    snapshot = service.readiness_for_version(registered.version)
    budget = next(
        result for result in snapshot.results if result.rule_id == "budget_policy"
    )

    assert budget.status is ReadinessStatus.FAIL
    assert snapshot.ready is False


def test_evaluation_launch_records_a_governed_job_and_rejects_an_active_duplicate():
    runner = RecordingJobRunner()
    repository, service = _service(job_runner=runner)
    _register(service, _manifest(evaluation_job_id="123"))
    owner = AuthorizationContext(principal="owner-analyst@example.com")
    request = EvaluationRequest(environment="dev", datasetVersion="dataset-v1")

    launched = service.start_evaluation("analyst", request, actor=owner)

    assert launched.status is EvaluationStatus.QUEUED
    assert launched.job_run_id == "preview-run-000001"
    assert launched.row_version == 2
    assert len(runner.requests) == 1
    job_request = runner.requests[0]
    assert job_request.job_id == "123"
    assert job_request.idempotency_token.startswith("evaluation:")
    assert dict(job_request.parameters) == {
        "application_id": "analyst",
        "application_version_id": launched.application_version_id,
        "dataset": "ai_platform.evaluations.analyst",
        "dataset_version": "dataset-v1",
        "environment": "dev",
        "evaluation_profile": "grounded_agent_v1",
        "evaluation_run_id": launched.evaluation_run_id,
        "git_sha": "a" * 40,
    }

    with pytest.raises(DuplicateActiveEvaluationError, match="already active"):
        service.start_evaluation("analyst", request, actor=owner)
    assert len(runner.requests) == 1
    assert repository.list_evaluations("analyst") == (launched,)
    readiness = service.readiness_for_version(
        repository.get_current_version("analyst", "dev")
    )
    dataset_rule = next(
        result for result in readiness.results if result.rule_id == "evaluation_dataset"
    )
    assert dataset_rule.status is ReadinessStatus.UNKNOWN


@pytest.mark.parametrize(
    ("summary_json", "error_match"),
    [
        pytest.param(
            '{"dataset_exists":',
            "valid JSON",
            id="malformed-json",
        ),
        pytest.param(
            '{"dataset_case_count":"thirty"}',
            "dataset_case_count",
            id="typed-wrong",
        ),
        pytest.param(
            '{"metrics":{"groundedness":0.9},"padding":"' + ("x" * 16_384) + '"}',
            "16 KiB",
            id="oversized",
        ),
        pytest.param(
            '{"metrics":{"groundedness":NaN}}',
            "finite",
            id="non-finite",
        ),
    ],
)
def test_invalid_evaluation_summaries_are_rejected_without_breaking_readiness(
    summary_json,
    error_match,
):
    repository, service = _service(job_runner=RecordingJobRunner())
    _register(service, _manifest(evaluation_job_id="123"))
    owner = AuthorizationContext(principal="owner-analyst@example.com")
    launched = service.start_evaluation(
        "analyst",
        EvaluationRequest(environment="dev", datasetVersion="dataset-v1"),
        actor=owner,
    )
    invalid_completion = launched.model_copy(
        update={
            "status": EvaluationStatus.SUCCEEDED,
            "completed_at": NOW,
            "summary_json": summary_json,
            "row_version": launched.row_version + 1,
        }
    )

    with pytest.raises(ValidationError, match=error_match):
        EvaluationRunRecord.model_validate(invalid_completion.model_dump(mode="python"))

    stored = repository.get_evaluation(launched.evaluation_run_id)
    assert stored.summary_json is None
    portfolio = service.list_portfolio(owner, PortfolioQuery())
    assert len(portfolio.items) == 1
    dataset_rule = next(
        result
        for result in portfolio.items[0].readiness.results
        if result.rule_id == "evaluation_dataset"
    )
    assert dataset_rule.status is ReadinessStatus.UNKNOWN


def test_evaluation_does_not_launch_an_unresolved_logical_job_key():
    runner = RecordingJobRunner()
    repository, service = _service(job_runner=runner)
    _register(
        service,
        _manifest(evaluation_job_id=None, evaluation_job_key="release_gate"),
    )
    owner = AuthorizationContext(principal="owner-analyst@example.com")

    with pytest.raises(
        HubCapabilityUnavailableError,
        match="not been resolved to an approved job ID",
    ):
        service.start_evaluation(
            "analyst",
            EvaluationRequest(environment="dev", datasetVersion="dataset-v1"),
            actor=owner,
        )

    assert runner.requests == ()
    assert repository.list_evaluations("analyst") == ()


def test_production_promotion_is_blocked_by_unknown_readiness_evidence():
    _, service = _service(job_runner=RecordingJobRunner())
    _register(
        service,
        _manifest(
            evaluation_job_id="123",
            promotion_job_id="456",
            environments=("dev", "prod"),
        ),
    )
    owner = AuthorizationContext(principal="owner-analyst@example.com")

    with pytest.raises(HubReadinessBlockedError) as exc_info:
        service.request_promotion(
            "analyst",
            PromotionRequest(
                sourceEnvironment="dev",
                targetEnvironment="prod",
            ),
            actor=owner,
        )

    snapshot = exc_info.value.snapshot
    statuses = {result.rule_id: result.status for result in snapshot.results}
    assert snapshot.ready is False
    assert statuses["production_monitoring"] is ReadinessStatus.UNKNOWN
    assert statuses["service_levels"] is ReadinessStatus.UNKNOWN
    assert service.list_promotions("analyst", actor=owner) == ()


def test_promotion_approval_requires_a_new_request_when_source_version_changes(
    monkeypatch,
):
    runner = RecordingJobRunner()
    repository, service = _service(job_runner=runner)
    manifest = _manifest(
        evaluation_job_id="123",
        promotion_job_id="456",
        environments=("dev", "prod"),
    )
    registered = _register(service, manifest)
    monkeypatch.setattr(
        service,
        "readiness_for_version",
        lambda version: _ready_snapshot_for(version),
    )
    owner = AuthorizationContext(principal="owner-analyst@example.com")
    promotion = service.request_promotion(
        "analyst",
        PromotionRequest(sourceEnvironment="dev", targetEnvironment="prod"),
        actor=owner,
        actor_request_id="request-promotion",
    )
    newer = _register(service, manifest, git_sha="b" * 40)
    assert newer.version.version_id != registered.version.version_id
    administrator = AuthorizationContext(
        principal="platform-admin@example.com",
        platform_roles=(Role.PLATFORM_ADMINISTRATOR,),
    )

    with pytest.raises(HubConflictError, match="new promotion request"):
        service.approve_promotion(
            promotion.promotion_request_id,
            PromotionReviewRequest(rowVersion=promotion.row_version),
            actor=administrator,
            actor_request_id="approve-stale-version",
        )

    unchanged = repository.get_promotion_request(promotion.promotion_request_id)
    assert unchanged.status is PromotionStatus.PENDING_REVIEW
    assert unchanged.row_version == 1
    assert unchanged.approval_readiness_snapshot is None
    replacement = service.request_promotion(
        "analyst",
        PromotionRequest(sourceEnvironment="dev", targetEnvironment="prod"),
        actor=owner,
        actor_request_id="replacement-promotion",
    )
    assert replacement.promotion_request_id != promotion.promotion_request_id
    assert replacement.application_version_id == newer.version.version_id
    assert runner.requests == ()


def test_promotion_approval_requires_a_new_request_when_ready_evidence_changes(
    monkeypatch,
):
    runner = RecordingJobRunner()
    repository, service = _service(job_runner=runner)
    registered = _register(
        service,
        _manifest(
            evaluation_job_id="123",
            promotion_job_id="456",
            environments=("dev", "prod"),
        ),
    )
    request_snapshot = _ready_snapshot_for(
        registered.version,
        evidence="evaluation run eval-1 passed",
    )
    approval_snapshot = _ready_snapshot_for(
        registered.version,
        evidence="evaluation run eval-2 passed",
    )
    assert request_snapshot.ready is approval_snapshot.ready is True
    assert (
        request_snapshot.decision_signature() != approval_snapshot.decision_signature()
    )
    snapshots = iter((request_snapshot, approval_snapshot))
    monkeypatch.setattr(
        service,
        "readiness_for_version",
        lambda _version: next(snapshots),
    )
    owner = AuthorizationContext(principal="owner-analyst@example.com")
    promotion = service.request_promotion(
        "analyst",
        PromotionRequest(sourceEnvironment="dev", targetEnvironment="prod"),
        actor=owner,
        actor_request_id="request-promotion",
    )
    administrator = AuthorizationContext(
        principal="platform-admin@example.com",
        platform_roles=(Role.PLATFORM_ADMINISTRATOR,),
    )

    with pytest.raises(HubConflictError, match="new promotion request"):
        service.approve_promotion(
            promotion.promotion_request_id,
            PromotionReviewRequest(rowVersion=promotion.row_version),
            actor=administrator,
            actor_request_id="approve-changed-evidence",
        )

    unchanged = repository.get_promotion_request(promotion.promotion_request_id)
    assert unchanged.status is PromotionStatus.PENDING_REVIEW
    assert unchanged.row_version == 1
    assert unchanged.approval_readiness_snapshot is None
    assert runner.requests == ()
