"""Application-service boundary for AI Platform Hub workflows.

The service coordinates validated manifests, authorization, readiness evidence, and
the persistence contract.  It intentionally has no FastAPI or Databricks SDK imports:
HTTP delivery, durable storage, and Jobs execution remain replaceable adapters.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, ValidationError, field_validator, model_validator

from aai_core.manifest import (
    AIApplicationManifest,
    build_manifest_envelope,
    load_manifest,
)

from .jobs import (
    JobExecutionError,
    JobLaunchRequest,
    JobRunner,
    UnavailableJobRunner,
    contains_credential_material,
)
from .models import (
    ApplicationRecord,
    ApplicationVersionRecord,
    AuthorizationContext,
    EvaluationRunRecord,
    EvaluationStatus,
    EvaluationSummary,
    HubModel,
    NonEmptyStr,
    PrincipalType,
    PromotionRequestRecord,
    PromotionStatus,
    ReadinessSnapshot,
    RegistrationResult,
    Role,
    Tag,
)
from .readiness import (
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
)
from .repository import HubConflictError, HubNotFoundError, HubRepository

_TAG_KEY = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_ENVIRONMENT = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_SAFE_FILTER_VALUE = re.compile(r"^[^,\r\n|]{1,256}$")
_LIFECYCLES = frozenset({"experimental", "candidate", "production", "retired"})
_HEALTH_STATES = frozenset({"HEALTHY", "DEGRADED", "CRITICAL", "UNKNOWN"})
_OWNERSHIP = frozenset({"visible", "owned", "teams"})
_SORTS = frozenset(
    {
        "application",
        "-application",
        "updated_at",
        "-updated_at",
        "owner",
        "-owner",
    }
)


class HubServiceError(RuntimeError):
    """Base class for expected service-layer failures."""


class HubCapabilityUnavailableError(HubServiceError):
    """A required production capability has not been configured."""


class HubPermissionDeniedError(HubServiceError):
    """The trusted actor is authenticated but not authorized."""


class HubQueryValidationError(HubServiceError):
    """A list filter or sort expression is outside the published grammar."""


class HubReadinessBlockedError(HubServiceError):
    """Production promotion is blocked by the immutable readiness evidence."""

    def __init__(self, snapshot: ReadinessSnapshot) -> None:
        self.snapshot = snapshot
        super().__init__("blocking production-readiness checks have not passed")


class HubExternalServiceError(HubServiceError):
    """An approved provider operation failed without exposing its payload."""


class RegistrationRequest(HubModel):
    """Typed registration input; actor identity is deliberately absent."""

    manifest: Mapping[str, Any]
    environment: NonEmptyStr
    git_commit_sha: NonEmptyStr = Field(alias="gitCommitSha")
    deployment_target: NonEmptyStr = Field(alias="deploymentTarget")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not _ENVIRONMENT.fullmatch(normalized):
            raise ValueError("environment must be a lowercase identifier")
        return normalized

    @field_validator("git_commit_sha")
    @classmethod
    def validate_git_sha(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{7,64}", normalized):
            raise ValueError("gitCommitSha must be a 7-64 character hexadecimal SHA")
        return normalized


class EvaluationRequest(HubModel):
    """Input needed to start the registered evaluation job."""

    environment: NonEmptyStr
    dataset_version: NonEmptyStr = Field(alias="datasetVersion")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        if not _ENVIRONMENT.fullmatch(value):
            raise ValueError("environment must be a lowercase identifier")
        return value

    @field_validator("dataset_version")
    @classmethod
    def validate_dataset_version(cls, value: str) -> str:
        """Apply the exact credential-free Jobs parameter contract at ingress."""

        try:
            validated = JobLaunchRequest(
                job_id="1",
                idempotency_token="evaluation-input",
                parameters={"dataset_version": value},
            )
        except ValidationError:
            raise ValueError(
                "datasetVersion must be a credential-free printable ASCII value "
                "of at most 2048 characters"
            ) from None
        return validated.parameters["dataset_version"]


class PromotionRequest(HubModel):
    """Input needed to create a reviewed production-promotion request."""

    source_environment: NonEmptyStr = Field(alias="sourceEnvironment")
    target_environment: NonEmptyStr = Field(alias="targetEnvironment")

    @field_validator("source_environment", "target_environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        if not _ENVIRONMENT.fullmatch(value):
            raise ValueError("environment must be a lowercase identifier")
        return value

    @model_validator(mode="after")
    def environments_must_differ(self) -> PromotionRequest:
        if self.source_environment == self.target_environment:
            raise ValueError("sourceEnvironment and targetEnvironment must differ")
        return self


class PromotionReviewRequest(HubModel):
    """Optimistic-concurrency input for an administrator review."""

    row_version: int = Field(alias="rowVersion", ge=1)
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator("comment")
    @classmethod
    def reject_credential_material(cls, value: str | None) -> str | None:
        if value is not None and contains_credential_material(value):
            raise ValueError("review comments must not contain credential material")
        return value


class PortfolioQuery(HubModel):
    """Allowlisted, server-side portfolio query."""

    search: str = Field(default="", max_length=200)
    lifecycle: str | None = None
    health: str | None = None
    ownership: Literal["visible", "owned", "teams"] = "visible"
    tag_filters: Mapping[str, frozenset[str]] = Field(default_factory=dict)
    sort: str = "application"
    page: int = Field(default=1, ge=1, le=1_000_000)
    page_size: int = Field(default=25, ge=1, le=100)

    @field_validator("lifecycle")
    @classmethod
    def validate_lifecycle(cls, value: str | None) -> str | None:
        if value is not None and value not in _LIFECYCLES:
            raise ValueError("lifecycle filter is not supported")
        return value

    @field_validator("health")
    @classmethod
    def validate_health(cls, value: str | None) -> str | None:
        if value is not None and value not in _HEALTH_STATES:
            raise ValueError("health filter is not supported")
        return value

    @field_validator("ownership")
    @classmethod
    def validate_ownership(cls, value: str) -> str:
        if value not in _OWNERSHIP:
            raise ValueError("ownership filter is not supported")
        return value

    @field_validator("sort")
    @classmethod
    def validate_sort(cls, value: str) -> str:
        if value not in _SORTS:
            raise ValueError("sort field is not supported")
        return value


class PortfolioItem(HubModel):
    application: ApplicationRecord
    current_version: ApplicationVersionRecord
    deployments: tuple[ApplicationVersionRecord, ...]
    readiness: ReadinessSnapshot


class PortfolioPage(HubModel):
    items: tuple[PortfolioItem, ...]
    page: int
    page_size: int
    total: int
    pages: int


def parse_tag_filters(expression: str) -> Mapping[str, frozenset[str]]:
    """Parse ``key=v1|v2,key2=v`` with AND across keys and OR within values."""

    if not expression.strip():
        return {}
    if len(expression) > 2_000:
        raise HubQueryValidationError("tag filter is too long")
    parsed: dict[str, frozenset[str]] = {}
    for clause in expression.split(","):
        key, separator, raw_values = clause.partition("=")
        key = key.strip()
        if separator != "=" or not _TAG_KEY.fullmatch(key):
            raise HubQueryValidationError("tag filters must use key=value syntax")
        if key in parsed:
            raise HubQueryValidationError("each tag key may appear only once")
        values = tuple(value.strip() for value in raw_values.split("|"))
        if not values or any(
            not value or not _SAFE_FILTER_VALUE.fullmatch(value) for value in values
        ):
            raise HubQueryValidationError("tag filter values are invalid")
        parsed[key] = frozenset(values)
    return parsed


class HubService:
    """Coordinates Hub use cases over an injected repository."""

    def __init__(
        self,
        repository: HubRepository,
        *,
        registration_principals: frozenset[str] = frozenset(),
        job_runner: JobRunner | None = None,
        clock: Callable[[], datetime] | None = None,
        readiness_profile: ReadinessProfile | None = None,
    ) -> None:
        self.repository = repository
        self._registration_principals = frozenset(
            principal.casefold() for principal in registration_principals
        )
        self.job_runner = job_runner or UnavailableJobRunner()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._readiness_profiles = {
            "development_v1": ReadinessProfile(
                profile_id="development_v1",
                # Evaluator v2 records the new cost-control rules as explicit
                # non-applicable evidence for development snapshots.
                version="2",
                minimum_evaluation_cases=1,
                require_ai_gateway=False,
                require_request_tags=False,
                require_monitoring=False,
            ),
            "medium_risk_production_v1": ReadinessProfile(
                profile_id="medium_risk_production_v1",
                # Evaluator v2 adds mandatory rate-limit, budget-policy, and
                # cost-attribution evidence without changing the manifest's
                # long-lived profile identifier.
                version="2",
                require_rate_limit=True,
                require_budget_policy=True,
            ),
        }
        if readiness_profile is not None:
            if readiness_profile.profile_id not in self._readiness_profiles:
                raise ValueError("readiness_profile must replace a supported profile")
            self._readiness_profiles[readiness_profile.profile_id] = readiness_profile

    @property
    def available(self) -> bool:
        return self.repository.available

    @property
    def registration_enabled(self) -> bool:
        return self.available and bool(self._registration_principals)

    @property
    def workflow_preview_enabled(self) -> bool:
        capability = self.job_runner.capability
        return capability.enabled and not capability.remote_execution

    def register(
        self,
        request: RegistrationRequest,
        *,
        actor: AuthorizationContext,
        actor_request_id: str | None = None,
    ) -> RegistrationResult:
        self._require_available()
        if actor.principal.casefold() not in self._registration_principals:
            raise HubPermissionDeniedError(
                "the authenticated principal is not approved for registration"
            )

        envelope = build_manifest_envelope(request.manifest)
        manifest = envelope.manifest
        original_manifest_json = json.dumps(
            dict(request.manifest),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if request.environment not in manifest.spec.environments:
            raise HubConflictError(
                "the requested environment is not declared in the manifest"
            )

        now = self._clock()
        application_tags = dict(manifest.metadata.tags)
        application = ApplicationRecord(
            application_id=manifest.metadata.id,
            name=manifest.metadata.name,
            description=manifest.metadata.description,
            owner_principal=manifest.metadata.owner,
            support_group=manifest.metadata.support_group,
            business_domain=manifest.metadata.business_domain,
            cost_center=manifest.metadata.cost_center,
            risk_tier=manifest.metadata.risk_tier,
            lifecycle_state=application_tags.get("lifecycle", "experimental"),
            tags=tuple(
                Tag(key=key, value=value)
                for key, value in sorted(application_tags.items())
            ),
            created_at=now,
            updated_at=now,
        )
        version_id = self._version_id(
            application_id=manifest.metadata.id,
            environment=request.environment,
            git_commit_sha=request.git_commit_sha,
            manifest_hash=envelope.manifest_hash,
        )
        version = ApplicationVersionRecord(
            version_id=version_id,
            application_id=manifest.metadata.id,
            environment=request.environment,
            git_repository=manifest.spec.repository.url,
            git_commit_sha=request.git_commit_sha,
            manifest_version=manifest.api_version,
            manifest_hash=envelope.manifest_hash,
            manifest_json=envelope.canonical_json,
            original_manifest_json=original_manifest_json,
            registered_by=actor.principal,
            registered_at=now,
            deployment_target=request.deployment_target,
        )
        return self.repository.register_application(
            application,
            version,
            actor_request_id=actor_request_id,
        )

    def list_portfolio(
        self,
        actor: AuthorizationContext,
        query: PortfolioQuery,
    ) -> PortfolioPage:
        self._require_available()
        if query.health and query.health != "UNKNOWN":
            # No health serving view is configured in the operational repository.
            applications: tuple[ApplicationRecord, ...] = ()
            total = 0
        else:
            applications, total = self.repository.query_visible_applications(
                actor,
                search=query.search,
                lifecycle=query.lifecycle,
                ownership=query.ownership,
                tag_filters=query.tag_filters,
                sort=query.sort,
                page=query.page,
                page_size=query.page_size,
            )
        items: list[PortfolioItem] = []
        for application in applications:
            deployments = self._current_versions(application.application_id)
            current = max(
                deployments,
                key=lambda version: (version.registered_at, version.version_id),
            )
            items.append(
                PortfolioItem(
                    application=application,
                    current_version=current,
                    deployments=deployments,
                    readiness=self.readiness_for_version(current),
                )
            )
        pages = max(1, (total + query.page_size - 1) // query.page_size)
        return PortfolioPage(
            items=tuple(items),
            page=query.page,
            page_size=query.page_size,
            total=total,
            pages=pages,
        )

    def get_application(
        self,
        application_id: str,
        *,
        actor: AuthorizationContext,
    ) -> ApplicationRecord:
        self._require_available()
        return self.repository.get_visible_application(application_id, actor)

    def get_application_version(
        self,
        application_id: str,
        *,
        actor: AuthorizationContext,
        environment: str | None = None,
    ) -> ApplicationVersionRecord:
        self.get_application(application_id, actor=actor)
        return self._select_current_version(application_id, environment=environment)

    def list_versions(
        self,
        application_id: str,
        *,
        actor: AuthorizationContext,
    ) -> tuple[ApplicationVersionRecord, ...]:
        self.get_application(application_id, actor=actor)
        return self.repository.list_versions(application_id)

    def readiness_for_version(
        self, version: ApplicationVersionRecord
    ) -> ReadinessSnapshot:
        manifest = load_manifest(json.loads(version.manifest_json))
        effective_tags = self._effective_tags(manifest, version.environment)

        evaluations = self._list_evaluations(version.application_id)
        latest = next(
            (
                item
                for item in reversed(evaluations)
                if item.application_version_id == version.version_id
            ),
            None,
        )
        summary = self._evaluation_summary(latest)
        evidence = ReadinessEvidence(
            application_id=version.application_id,
            environment=version.environment,
            application_version_id=version.version_id,
            manifest=ManifestEvidence(
                schema_valid=True,
                # Registration proves what CI declared, not what is currently
                # deployed. Resource discovery must supply this comparison.
                git_sha_matches_registered_version=None,
                resources_resolve=None,
            ),
            ownership=OwnershipEvidence(
                owner_registered=bool(manifest.metadata.owner),
                support_group_registered=bool(manifest.metadata.support_group),
            ),
            tags=TagEvidence(
                observed_tag_keys=tuple(sorted(effective_tags)),
                data_classification_declared="data_classification" in effective_tags,
            ),
            risk=RiskEvidence(risk_tier=manifest.metadata.risk_tier),
            authentication=AuthenticationEvidence(
                mode_declared=bool(manifest.spec.authorization.mode),
                approved_ai_gateway=None,
                rate_limit_configured=None,
            ),
            cost=CostEvidence(
                # A manifest reference declares intent. Only platform discovery
                # can prove that the external budget and billing controls exist.
                budget_policy_configured=(
                    False if manifest.spec.cost_controls is None else None
                ),
                attribution_verified=None,
            ),
            tracing=TracingEvidence(
                # A declared experiment ID is not proof that tracing is configured
                # or that the runtime can write to it.
                configured=None,
                recent_trace_metadata_complete=None,
            ),
            jobs=JobEvidence(
                evaluation_job_configured=bool(
                    manifest.spec.resources.evaluation_job_id
                ),
                promotion_job_configured=bool(manifest.spec.resources.promotion_job_id),
            ),
            evaluation=EvaluationEvidence(
                dataset_exists=(None if latest is None else summary.dataset_exists),
                dataset_case_count=summary.dataset_case_count,
                targets_current_application_version=(
                    None
                    if latest is None
                    else latest.application_version_id == version.version_id
                ),
                evaluated_at=None if latest is None else latest.completed_at,
                blocking_thresholds_passed=summary.blocking_thresholds_passed,
            ),
            telemetry=TelemetryEvidence(
                monitoring_configured=None,
                service_levels_passed=None,
                required_request_tags_observed=None,
            ),
            administrator=AdministratorEvidence(
                unresolved_blocking_actions=None,
            ),
        )
        profile = self._readiness_profiles[manifest.spec.readiness.profile].model_copy(
            update={
                "minimum_evaluation_cases": manifest.spec.evaluation.minimum_cases,
                "maximum_evaluation_age_hours": (
                    manifest.spec.evaluation.maximum_age_hours
                ),
            }
        )
        return ReadinessEvaluator(profile).evaluate(
            evidence,
            evaluated_at=self._clock(),
        )

    def can_contribute(
        self, application: ApplicationRecord, actor: AuthorizationContext
    ) -> bool:
        if actor.has_platform_role(Role.PLATFORM_ADMINISTRATOR):
            return True
        return bool(
            self._application_roles_for_actor(application.application_id, actor)
            & {Role.CONTRIBUTOR, Role.OWNER}
        )

    def list_evaluations(
        self,
        application_id: str,
        *,
        actor: AuthorizationContext,
    ) -> tuple[EvaluationRunRecord, ...]:
        self.get_application(application_id, actor=actor)
        return self._list_evaluations(application_id)

    def get_evaluation(
        self,
        evaluation_run_id: str,
        *,
        actor: AuthorizationContext,
    ) -> EvaluationRunRecord:
        self._require_available()
        evaluation = self.repository.get_evaluation(evaluation_run_id)
        self.get_application(evaluation.application_id, actor=actor)
        return evaluation

    def start_evaluation(
        self,
        application_id: str,
        request: EvaluationRequest,
        *,
        actor: AuthorizationContext,
        actor_request_id: str | None = None,
    ) -> EvaluationRunRecord:
        application = self.get_application(application_id, actor=actor)
        if not self.can_contribute(application, actor):
            raise HubPermissionDeniedError(
                "contributor or owner authorization is required"
            )
        version = self.get_application_version(
            application_id,
            actor=actor,
            environment=request.environment,
        )
        manifest = self.manifest_for_version(version)
        job_id = manifest.spec.resources.evaluation_job_id
        if job_id is None:
            raise HubCapabilityUnavailableError(
                "the evaluation job has not been resolved to an approved job ID"
            )
        self._require_executable_job_runner()

        evaluation_run_id = f"eval_{uuid4().hex}"
        parameters = {
            "application_id": application_id,
            "application_version_id": version.version_id,
            "dataset": manifest.spec.evaluation.dataset,
            "dataset_version": request.dataset_version,
            "environment": request.environment,
            "evaluation_profile": manifest.spec.evaluation.profile,
            "evaluation_run_id": evaluation_run_id,
            "git_sha": version.git_commit_sha,
        }
        try:
            launch_request = JobLaunchRequest(
                job_id=job_id,
                idempotency_token=self._job_token("evaluation", evaluation_run_id),
                parameters=parameters,
            )
        except ValidationError:
            raise HubQueryValidationError(
                "the registered evaluation job parameters are invalid"
            ) from None

        now = self._clock()
        evaluation = EvaluationRunRecord(
            evaluation_run_id=evaluation_run_id,
            application_id=application_id,
            environment=request.environment,
            application_version_id=version.version_id,
            evaluation_profile=manifest.spec.evaluation.profile,
            dataset_name=manifest.spec.evaluation.dataset,
            dataset_version=request.dataset_version,
            job_id=job_id,
            requested_by=actor.principal,
            status=EvaluationStatus.REQUESTED,
            requested_at=now,
        )
        stored = self.repository.create_evaluation(
            evaluation,
            actor_request_id=actor_request_id,
        )
        try:
            launched = self.job_runner.launch(launch_request)
        except JobExecutionError:
            failed = EvaluationRunRecord.model_validate(
                stored.model_copy(
                    update={
                        "status": EvaluationStatus.FAILED,
                        "failure_message": (
                            "The approved evaluation job could not be launched."
                        ),
                        "completed_at": self._clock(),
                        "row_version": stored.row_version + 1,
                    }
                ).model_dump(mode="python")
            )
            self.repository.update_evaluation(
                failed,
                expected_row_version=stored.row_version,
                actor_request_id=actor_request_id,
            )
            raise HubExternalServiceError(
                "the approved evaluation job could not be launched"
            ) from None

        queued = EvaluationRunRecord.model_validate(
            stored.model_copy(
                update={
                    "status": EvaluationStatus.QUEUED,
                    "job_run_id": launched.run_id,
                    "row_version": stored.row_version + 1,
                }
            ).model_dump(mode="python")
        )
        return self.repository.update_evaluation(
            queued,
            expected_row_version=stored.row_version,
            actor_request_id=actor_request_id,
        )

    def list_promotions(
        self,
        application_id: str,
        *,
        actor: AuthorizationContext,
    ) -> tuple[PromotionRequestRecord, ...]:
        self.get_application(application_id, actor=actor)
        return self.repository.list_promotion_requests(application_id)

    def request_promotion(
        self,
        application_id: str,
        request: PromotionRequest,
        *,
        actor: AuthorizationContext,
        actor_request_id: str | None = None,
    ) -> PromotionRequestRecord:
        application = self.get_application(application_id, actor=actor)
        if not self.can_contribute(application, actor):
            raise HubPermissionDeniedError(
                "contributor or owner authorization is required"
            )
        version = self.get_application_version(
            application_id,
            actor=actor,
            environment=request.source_environment,
        )
        manifest = self.manifest_for_version(version)
        if request.target_environment not in manifest.spec.environments:
            raise HubConflictError(
                "the target environment is not declared in the manifest"
            )
        job_id = manifest.spec.resources.promotion_job_id
        if job_id is None:
            raise HubCapabilityUnavailableError(
                "the promotion job has not been resolved to an approved job ID"
            )
        self._require_executable_job_runner()
        readiness = self.readiness_for_version(version)
        if not readiness.ready:
            raise HubReadinessBlockedError(readiness)
        now = self._clock()
        promotion = PromotionRequestRecord(
            promotion_request_id=f"promo_{uuid4().hex}",
            application_id=application_id,
            source_environment=request.source_environment,
            target_environment=request.target_environment,
            application_version_id=version.version_id,
            requested_by=actor.principal,
            requested_at=now,
            status=PromotionStatus.PENDING_REVIEW,
            readiness_snapshot=readiness,
            promotion_job_id=job_id,
        )
        return self.repository.create_promotion_request(
            promotion,
            actor_request_id=actor_request_id,
        )

    def list_admin_actions(
        self, actor: AuthorizationContext
    ) -> tuple[PromotionRequestRecord, ...]:
        self._require_administrator(actor)
        pending: list[PromotionRequestRecord] = []
        for application in self.repository.list_visible_applications(actor):
            pending.extend(
                request
                for request in self.list_promotions(
                    application.application_id,
                    actor=actor,
                )
                if request.status is PromotionStatus.PENDING_REVIEW
            )
        return tuple(
            sorted(
                pending,
                key=lambda request: (
                    request.requested_at,
                    request.promotion_request_id,
                ),
                reverse=True,
            )
        )

    def approve_promotion(
        self,
        promotion_request_id: str,
        request: PromotionReviewRequest,
        *,
        actor: AuthorizationContext,
        actor_request_id: str,
    ) -> PromotionRequestRecord:
        self._require_administrator(actor)
        current = self.repository.get_promotion_request(promotion_request_id)
        current_source = self.repository.get_current_version(
            current.application_id,
            current.source_environment,
        )
        if current_source.version_id != current.application_version_id:
            raise HubConflictError(
                "the source environment has a newer current version; "
                "submit a new promotion request"
            )
        version = next(
            (
                version
                for version in self.repository.list_versions(current.application_id)
                if version.version_id == current.application_version_id
            ),
            None,
        )
        if version is None:
            raise HubNotFoundError("the promotion application version was not found")
        readiness = self.readiness_for_version(version)
        if not readiness.ready:
            raise HubReadinessBlockedError(readiness)
        if (
            readiness.decision_signature()
            != current.readiness_snapshot.decision_signature()
        ):
            raise HubConflictError(
                "readiness evidence changed; submit a new promotion request"
            )
        self._require_executable_job_runner()
        approved = self.repository.approve_promotion(
            promotion_request_id,
            actor=actor,
            expected_row_version=request.row_version,
            reviewed_at=self._clock(),
            actor_request_id=actor_request_id,
            readiness_snapshot=readiness,
            comment=request.comment,
        )
        parameters = {
            "application_id": current.application_id,
            "application_version_id": current.application_version_id,
            "git_sha": version.git_commit_sha,
            "promotion_request_id": current.promotion_request_id,
            "source_environment": current.source_environment,
            "target_environment": current.target_environment,
        }
        try:
            launched = self.job_runner.launch(
                JobLaunchRequest(
                    job_id=current.promotion_job_id,
                    idempotency_token=self._job_token(
                        "promotion", current.promotion_request_id
                    ),
                    parameters=parameters,
                )
            )
        except JobExecutionError:
            failed = PromotionRequestRecord.model_validate(
                approved.model_copy(
                    update={
                        "status": PromotionStatus.FAILED,
                        "row_version": approved.row_version + 1,
                    }
                ).model_dump(mode="python")
            )
            self.repository.update_promotion(
                failed,
                expected_row_version=approved.row_version,
                actor_principal=actor.principal,
                actor_request_id=actor_request_id,
                event_time=self._clock(),
                comment="Approved promotion job launch failed.",
            )
            raise HubExternalServiceError(
                "the approved promotion job could not be launched"
            ) from None

        executing = PromotionRequestRecord.model_validate(
            approved.model_copy(
                update={
                    "status": PromotionStatus.EXECUTING,
                    "promotion_job_run_id": launched.run_id,
                    "row_version": approved.row_version + 1,
                }
            ).model_dump(mode="python")
        )
        return self.repository.update_promotion(
            executing,
            expected_row_version=approved.row_version,
            actor_principal=actor.principal,
            actor_request_id=actor_request_id,
            event_time=self._clock(),
        )

    def reject_promotion(
        self,
        promotion_request_id: str,
        request: PromotionReviewRequest,
        *,
        actor: AuthorizationContext,
        actor_request_id: str,
    ) -> PromotionRequestRecord:
        self._require_administrator(actor)
        if request.comment is None or not request.comment.strip():
            raise HubConflictError("a review comment is required")
        return self.repository.reject_promotion(
            promotion_request_id,
            actor=actor,
            expected_row_version=request.row_version,
            reviewed_at=self._clock(),
            actor_request_id=actor_request_id,
            comment=request.comment,
        )

    def request_promotion_changes(
        self,
        promotion_request_id: str,
        request: PromotionReviewRequest,
        *,
        actor: AuthorizationContext,
        actor_request_id: str,
    ) -> PromotionRequestRecord:
        self._require_administrator(actor)
        if request.comment is None or not request.comment.strip():
            raise HubConflictError("a review comment is required")
        return self.repository.request_promotion_changes(
            promotion_request_id,
            actor=actor,
            expected_row_version=request.row_version,
            reviewed_at=self._clock(),
            actor_request_id=actor_request_id,
            comment=request.comment,
        )

    @staticmethod
    def manifest_for_version(
        version: ApplicationVersionRecord,
    ) -> AIApplicationManifest:
        return load_manifest(json.loads(version.manifest_json))

    @staticmethod
    def _version_id(
        *,
        application_id: str,
        environment: str,
        git_commit_sha: str,
        manifest_hash: str,
    ) -> str:
        identity = "\x00".join(
            (application_id, environment, git_commit_sha, manifest_hash)
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()
        return f"ver_{digest[:24]}"

    @staticmethod
    def _effective_tags(
        manifest: AIApplicationManifest, environment: str
    ) -> dict[str, str]:
        return {
            **dict(manifest.metadata.tags),
            **dict(manifest.spec.environments[environment].tags),
        }

    def _select_current_version(
        self, application_id: str, *, environment: str | None = None
    ) -> ApplicationVersionRecord:
        versions = tuple(
            version
            for version in self._current_versions(application_id)
            if environment is None or version.environment == environment
        )
        if not versions:
            raise HubNotFoundError(
                f"no current application version exists for {application_id!r}"
            )
        return max(
            versions,
            key=lambda version: (version.registered_at, version.version_id),
        )

    def _current_versions(
        self,
        application_id: str,
    ) -> tuple[ApplicationVersionRecord, ...]:
        versions = tuple(
            sorted(
                (
                    version
                    for version in self.repository.list_versions(application_id)
                    if version.is_current
                ),
                key=lambda version: (version.environment, version.version_id),
            )
        )
        if not versions:
            raise HubNotFoundError(
                f"no current application version exists for {application_id!r}"
            )
        return versions

    def _matches_ownership(
        self,
        application: ApplicationRecord,
        actor: AuthorizationContext,
        ownership: str,
    ) -> bool:
        if ownership == "visible":
            return True
        if ownership == "owned":
            return Role.OWNER in self._application_roles_for_actor(
                application.application_id,
                actor,
            )
        groups = {group.casefold() for group in actor.groups}
        return any(
            principal.principal_type is PrincipalType.GROUP
            and principal.principal_name.casefold() in groups
            for principal in self.repository.list_application_principals(
                application.application_id
            )
        )

    def _application_roles_for_actor(
        self,
        application_id: str,
        actor: AuthorizationContext,
    ) -> set[Role]:
        groups = {group.casefold() for group in actor.groups}
        principal_name = actor.principal.casefold()
        return {
            principal.application_role
            for principal in self.repository.list_application_principals(application_id)
            if (
                principal.principal_type is PrincipalType.USER
                and principal.principal_name.casefold() == principal_name
            )
            or (
                principal.principal_type is PrincipalType.GROUP
                and principal.principal_name.casefold() in groups
            )
        }

    def _list_evaluations(self, application_id: str) -> tuple[EvaluationRunRecord, ...]:
        return self.repository.list_evaluations(application_id)

    @staticmethod
    def _evaluation_summary(
        evaluation: EvaluationRunRecord | None,
    ) -> EvaluationSummary:
        if evaluation is None or evaluation.summary_json is None:
            return EvaluationSummary()
        return EvaluationSummary.model_validate_json(evaluation.summary_json)

    def _require_available(self) -> None:
        if not self.available:
            raise HubCapabilityUnavailableError(
                "a durable Hub operational store has not been configured"
            )

    def _require_executable_job_runner(self) -> None:
        capability = self.job_runner.capability
        if not capability.enabled:
            raise HubCapabilityUnavailableError(capability.detail)
        if capability.remote_execution:
            raise HubCapabilityUnavailableError(
                "remote workflow launch remains gated until durable status "
                "reconciliation and sanitized result ingestion are implemented"
            )

    @staticmethod
    def _job_token(workflow: str, entity_id: str) -> str:
        digest = hashlib.sha256(f"{workflow}\x00{entity_id}".encode()).hexdigest()
        return f"{workflow}:{digest[:48]}"

    @staticmethod
    def _require_administrator(actor: AuthorizationContext) -> None:
        if not actor.has_platform_role(Role.PLATFORM_ADMINISTRATOR):
            raise HubPermissionDeniedError(
                "platform administrator authorization is required"
            )


__all__ = [
    "EvaluationRequest",
    "HubCapabilityUnavailableError",
    "HubExternalServiceError",
    "HubPermissionDeniedError",
    "HubQueryValidationError",
    "HubReadinessBlockedError",
    "HubService",
    "HubServiceError",
    "PortfolioItem",
    "PortfolioPage",
    "PortfolioQuery",
    "PromotionRequest",
    "PromotionReviewRequest",
    "RegistrationRequest",
    "parse_tag_filters",
]
