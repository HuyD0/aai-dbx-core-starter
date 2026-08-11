"""Persistence boundary and a deterministic in-memory Hub repository.

The in-memory implementation is a production-shaped test double, not a durable
backend.  It keeps the same atomicity, idempotency, visibility, state-transition,
and optimistic-concurrency rules expected from the Lakebase or SQL adapter.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from threading import RLock
from typing import Protocol, runtime_checkable
from uuid import uuid4

from .models import (
    ActionEntityType,
    ActionEvent,
    ActionEventType,
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
    RegistrationResult,
    ResourceBindingRecord,
    Role,
)


class HubRepositoryError(RuntimeError):
    """Base class for repository failures callers may safely classify."""


class HubRepositoryUnavailableError(HubRepositoryError):
    """The configured operational store cannot safely serve the request."""


# Short alias for adapters that prefer the conventional error name.
RepositoryUnavailableError = HubRepositoryUnavailableError


class HubNotFoundError(HubRepositoryError):
    """The requested record does not exist or is not visible to the caller."""


class HubConflictError(HubRepositoryError):
    """The request conflicts with immutable or active state."""


class ImmutableApplicationIdConflictError(HubConflictError):
    """An application ID is already bound to a different source repository."""


class DuplicateActiveEvaluationError(HubConflictError):
    """An equivalent evaluation is already active."""


class DuplicateActivePromotionError(HubConflictError):
    """An equivalent promotion request is already active."""


class InvalidStateTransitionError(HubConflictError):
    """A workflow state transition is not allowed."""


class OptimisticConcurrencyError(HubConflictError):
    """The caller's row version is stale."""

    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"row version mismatch: expected {expected}, current version is {actual}"
        )


class HubAuthorizationError(HubRepositoryError):
    """The trusted actor context lacks the required role."""


class FourEyesViolationError(HubAuthorizationError):
    """The requester attempted to approve their own environment promotion."""


def _merge_registered_application(
    application: ApplicationRecord,
    existing: ApplicationRecord,
    version: ApplicationVersionRecord,
) -> ApplicationRecord:
    """Merge mutable registry metadata without allowing time to move backwards."""

    replacement = application.model_copy(
        update={
            "created_at": existing.created_at,
            "updated_at": max(existing.updated_at, version.registered_at),
            "row_version": existing.row_version + 1,
        }
    )
    # model_copy deliberately skips validation. Reconstruct before any mutation/write
    # so delayed registrations cannot produce an impossible timestamp ordering.
    return ApplicationRecord.model_validate(replacement.model_dump(mode="python"))


@runtime_checkable
class HubRepository(Protocol):
    @property
    def available(self) -> bool: ...

    def register_application(
        self,
        application: ApplicationRecord,
        version: ApplicationVersionRecord,
        *,
        actor_request_id: str | None = None,
    ) -> RegistrationResult: ...

    def get_application(self, application_id: str) -> ApplicationRecord: ...

    def get_visible_application(
        self,
        application_id: str,
        actor: AuthorizationContext,
    ) -> ApplicationRecord: ...

    def get_current_version(
        self, application_id: str, environment: str
    ) -> ApplicationVersionRecord: ...

    def list_versions(
        self, application_id: str
    ) -> tuple[ApplicationVersionRecord, ...]: ...

    def upsert_application_principal(
        self, principal: ApplicationPrincipalRecord
    ) -> ApplicationPrincipalRecord: ...

    def replace_application_principals(
        self,
        application_id: str,
        principals: Iterable[ApplicationPrincipalRecord],
    ) -> tuple[ApplicationPrincipalRecord, ...]: ...

    def list_application_principals(
        self,
        application_id: str,
    ) -> tuple[ApplicationPrincipalRecord, ...]: ...

    def list_visible_applications(
        self, actor: AuthorizationContext
    ) -> tuple[ApplicationRecord, ...]: ...

    def query_visible_applications(
        self,
        actor: AuthorizationContext,
        *,
        search: str,
        lifecycle: str | None,
        ownership: str,
        tag_filters: Mapping[str, frozenset[str]],
        sort: str,
        page: int,
        page_size: int,
    ) -> tuple[tuple[ApplicationRecord, ...], int]: ...

    def upsert_resource_binding(
        self,
        binding: ResourceBindingRecord,
    ) -> ResourceBindingRecord: ...

    def list_resource_bindings(
        self,
        application_id: str,
        *,
        environment: str | None = None,
    ) -> tuple[ResourceBindingRecord, ...]: ...

    def create_evaluation(
        self,
        evaluation: EvaluationRunRecord,
        *,
        actor_request_id: str | None = None,
    ) -> EvaluationRunRecord: ...

    def update_evaluation(
        self,
        evaluation: EvaluationRunRecord,
        *,
        expected_row_version: int,
        actor_request_id: str | None = None,
    ) -> EvaluationRunRecord: ...

    def get_evaluation(self, evaluation_run_id: str) -> EvaluationRunRecord: ...

    def list_evaluations(
        self,
        application_id: str,
    ) -> tuple[EvaluationRunRecord, ...]: ...

    def create_promotion_request(
        self,
        request: PromotionRequestRecord,
        *,
        actor_request_id: str | None = None,
    ) -> PromotionRequestRecord: ...

    def get_promotion_request(
        self,
        promotion_request_id: str,
    ) -> PromotionRequestRecord: ...

    def list_promotion_requests(
        self,
        application_id: str,
    ) -> tuple[PromotionRequestRecord, ...]: ...

    def approve_promotion(
        self,
        promotion_request_id: str,
        *,
        actor: AuthorizationContext,
        expected_row_version: int,
        reviewed_at: datetime,
        actor_request_id: str,
        readiness_snapshot: ReadinessSnapshot | None = None,
        comment: str | None = None,
    ) -> PromotionRequestRecord: ...

    def reject_promotion(
        self,
        promotion_request_id: str,
        *,
        actor: AuthorizationContext,
        expected_row_version: int,
        reviewed_at: datetime,
        actor_request_id: str,
        comment: str,
    ) -> PromotionRequestRecord: ...

    def request_promotion_changes(
        self,
        promotion_request_id: str,
        *,
        actor: AuthorizationContext,
        expected_row_version: int,
        reviewed_at: datetime,
        actor_request_id: str,
        comment: str,
    ) -> PromotionRequestRecord: ...

    def update_promotion(
        self,
        request: PromotionRequestRecord,
        *,
        expected_row_version: int,
        actor_principal: str,
        actor_request_id: str,
        event_time: datetime,
        comment: str | None = None,
    ) -> PromotionRequestRecord: ...

    def append_action_event(self, event: ActionEvent) -> ActionEvent: ...

    def list_action_events(
        self,
        *,
        entity_type: ActionEntityType | None = None,
        entity_id: str | None = None,
    ) -> tuple[ActionEvent, ...]: ...

    def list_application_action_events(
        self,
        application_id: str,
    ) -> tuple[ActionEvent, ...]:
        """Return application, evaluation, and promotion events for one app."""

        ...


class InMemoryHubRepository:
    """Thread-safe reference implementation of Hub persistence semantics."""

    _EVALUATION_TRANSITIONS = {
        EvaluationStatus.REQUESTED: {
            EvaluationStatus.QUEUED,
            EvaluationStatus.RUNNING,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        },
        EvaluationStatus.QUEUED: {
            EvaluationStatus.RUNNING,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        },
        EvaluationStatus.RUNNING: {
            EvaluationStatus.SUCCEEDED,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        },
        EvaluationStatus.SUCCEEDED: set(),
        EvaluationStatus.FAILED: set(),
        EvaluationStatus.CANCELLED: set(),
    }
    _PROMOTION_TRANSITIONS = {
        PromotionStatus.PENDING_REVIEW: {
            PromotionStatus.CHANGES_REQUESTED,
            PromotionStatus.REJECTED,
            PromotionStatus.APPROVED,
            PromotionStatus.CANCELLED,
        },
        PromotionStatus.CHANGES_REQUESTED: {
            PromotionStatus.PENDING_REVIEW,
            PromotionStatus.CANCELLED,
        },
        PromotionStatus.APPROVED: {
            PromotionStatus.EXECUTING,
            PromotionStatus.FAILED,
            PromotionStatus.CANCELLED,
        },
        PromotionStatus.EXECUTING: {
            PromotionStatus.SUCCEEDED,
            PromotionStatus.FAILED,
        },
        PromotionStatus.REJECTED: set(),
        PromotionStatus.SUCCEEDED: set(),
        PromotionStatus.FAILED: set(),
        PromotionStatus.CANCELLED: set(),
    }
    _FLEET_ROLES = {
        Role.PLATFORM_VIEWER,
        Role.PLATFORM_ADMINISTRATOR,
        Role.AUDITOR,
    }

    def __init__(self) -> None:
        self._lock = RLock()
        self._applications: dict[str, ApplicationRecord] = {}
        self._versions: dict[str, ApplicationVersionRecord] = {}
        self._version_keys: dict[tuple[str, str, str, str], str] = {}
        self._current_versions: dict[tuple[str, str], str] = {}
        self._principals: dict[
            tuple[str, PrincipalType, str], ApplicationPrincipalRecord
        ] = {}
        self._resources: dict[str, ResourceBindingRecord] = {}
        self._evaluations: dict[str, EvaluationRunRecord] = {}
        self._promotions: dict[str, PromotionRequestRecord] = {}
        self._events: dict[str, ActionEvent] = {}

    @property
    def available(self) -> bool:
        return True

    def register_application(
        self,
        application: ApplicationRecord,
        version: ApplicationVersionRecord,
        *,
        actor_request_id: str | None = None,
    ) -> RegistrationResult:
        if version.application_id != application.application_id:
            raise HubConflictError("application and version IDs do not match")
        if application.row_version != 1:
            raise HubConflictError("new registration input must have row_version 1")

        key = (
            version.application_id,
            version.environment,
            version.git_commit_sha.lower(),
            version.manifest_hash,
        )
        with self._lock:
            existing_version_id = self._version_keys.get(key)
            if existing_version_id is not None:
                existing_version = self._versions[existing_version_id]
                if existing_version.deployment_target != version.deployment_target:
                    raise HubConflictError(
                        "an idempotent registration cannot change deployment_target"
                    )
                return RegistrationResult(
                    application=self._applications[application.application_id],
                    version=existing_version,
                    created=False,
                )

            conflicting_version = self._versions.get(version.version_id)
            if conflicting_version is not None:
                raise HubConflictError(
                    f"version ID {version.version_id!r} already describes "
                    "another version"
                )

            existing_application = self._applications.get(application.application_id)
            if existing_application is None:
                stored_application = application
            else:
                repositories = {
                    item.git_repository
                    for item in self._versions.values()
                    if item.application_id == application.application_id
                }
                if repositories and version.git_repository not in repositories:
                    raise ImmutableApplicationIdConflictError(
                        f"application ID {application.application_id!r} is "
                        "already bound to another Git repository"
                    )
                stored_application = _merge_registered_application(
                    application,
                    existing_application,
                    version,
                )

            pointer = (version.application_id, version.environment)
            previous_current_id = self._current_versions.get(pointer)
            if previous_current_id is not None:
                previous = self._versions[previous_current_id]
                self._versions[previous_current_id] = previous.model_copy(
                    update={"is_current": False}
                )

            stored_version = version.model_copy(update={"is_current": True})
            self._applications[application.application_id] = stored_application
            self._versions[stored_version.version_id] = stored_version
            self._version_keys[key] = stored_version.version_id
            self._current_versions[pointer] = stored_version.version_id
            self._append_event_locked(
                ActionEvent(
                    event_id=f"registration:{stored_version.version_id}",
                    entity_type=ActionEntityType.APPLICATION,
                    entity_id=application.application_id,
                    event_type=ActionEventType.APPLICATION_REGISTERED,
                    actor_principal=stored_version.registered_by,
                    actor_request_id=(
                        actor_request_id or f"registration:{stored_version.version_id}"
                    ),
                    event_time=stored_version.registered_at,
                    previous_state=previous_current_id,
                    new_state=stored_version.version_id,
                    details_json=json.dumps(
                        {
                            "environment": stored_version.environment,
                            "manifest_hash": stored_version.manifest_hash,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
            return RegistrationResult(
                application=stored_application,
                version=stored_version,
                created=True,
            )

    def get_application(self, application_id: str) -> ApplicationRecord:
        with self._lock:
            try:
                return self._applications[application_id]
            except KeyError as error:
                raise HubNotFoundError(
                    f"application {application_id!r} was not found"
                ) from error

    def get_visible_application(
        self, application_id: str, actor: AuthorizationContext
    ) -> ApplicationRecord:
        visible = {
            application.application_id: application
            for application in self.list_visible_applications(actor)
        }
        try:
            return visible[application_id]
        except KeyError as error:
            # Deliberately indistinguishable from a missing ID.
            raise HubNotFoundError(
                f"application {application_id!r} was not found"
            ) from error

    def get_current_version(
        self, application_id: str, environment: str
    ) -> ApplicationVersionRecord:
        with self._lock:
            version_id = self._current_versions.get((application_id, environment))
            if version_id is None:
                raise HubNotFoundError(
                    f"no current {environment!r} version exists for "
                    f"{application_id!r}"
                )
            return self._versions[version_id]

    def list_versions(
        self, application_id: str
    ) -> tuple[ApplicationVersionRecord, ...]:
        with self._lock:
            if application_id not in self._applications:
                raise HubNotFoundError(f"application {application_id!r} was not found")
            return tuple(
                sorted(
                    (
                        version
                        for version in self._versions.values()
                        if version.application_id == application_id
                    ),
                    key=lambda version: (version.registered_at, version.version_id),
                )
            )

    def upsert_application_principal(
        self, principal: ApplicationPrincipalRecord
    ) -> ApplicationPrincipalRecord:
        with self._lock:
            if principal.application_id not in self._applications:
                raise HubNotFoundError(
                    f"application {principal.application_id!r} was not found"
                )
            key = (
                principal.application_id,
                principal.principal_type,
                principal.principal_name.casefold(),
            )
            self._principals[key] = principal
            return principal

    def replace_application_principals(
        self,
        application_id: str,
        principals: Iterable[ApplicationPrincipalRecord],
    ) -> tuple[ApplicationPrincipalRecord, ...]:
        replacement = tuple(principals)
        keys: set[tuple[PrincipalType, str]] = set()
        for principal in replacement:
            if principal.application_id != application_id:
                raise HubConflictError("principal belongs to another application")
            key = (principal.principal_type, principal.principal_name.casefold())
            if key in keys:
                raise HubConflictError("application principal appears more than once")
            keys.add(key)
        with self._lock:
            if application_id not in self._applications:
                raise HubNotFoundError(f"application {application_id!r} was not found")
            self._principals = {
                key: value
                for key, value in self._principals.items()
                if key[0] != application_id
            }
            for principal in replacement:
                key = (
                    application_id,
                    principal.principal_type,
                    principal.principal_name.casefold(),
                )
                self._principals[key] = principal
            return replacement

    def list_application_principals(
        self, application_id: str
    ) -> tuple[ApplicationPrincipalRecord, ...]:
        with self._lock:
            if application_id not in self._applications:
                raise HubNotFoundError(f"application {application_id!r} was not found")
            return tuple(
                sorted(
                    (
                        principal
                        for principal in self._principals.values()
                        if principal.application_id == application_id
                    ),
                    key=lambda principal: (
                        principal.principal_type.value,
                        principal.principal_name.casefold(),
                    ),
                )
            )

    def list_visible_applications(
        self, actor: AuthorizationContext
    ) -> tuple[ApplicationRecord, ...]:
        with self._lock:
            if set(actor.platform_roles).intersection(self._FLEET_ROLES):
                visible = self._applications.values()
            else:
                principal_name = actor.principal.casefold()
                groups = {group.casefold() for group in actor.groups}
                explicit: set[str] = set()
                for principal in self._principals.values():
                    name = principal.principal_name.casefold()
                    if (
                        principal.principal_type is PrincipalType.USER
                        and name == principal_name
                    ) or (
                        principal.principal_type is PrincipalType.GROUP
                        and name in groups
                    ):
                        explicit.add(principal.application_id)
                visible = (
                    application
                    for application in self._applications.values()
                    if application.application_id in explicit
                )
            return tuple(
                sorted(visible, key=lambda application: application.application_id)
            )

    def query_visible_applications(
        self,
        actor: AuthorizationContext,
        *,
        search: str,
        lifecycle: str | None,
        ownership: str,
        tag_filters: Mapping[str, frozenset[str]],
        sort: str,
        page: int,
        page_size: int,
    ) -> tuple[tuple[ApplicationRecord, ...], int]:
        """Apply the production query boundary before per-row evidence work."""

        visible = list(self.list_visible_applications(actor))
        needle = search.strip().casefold()
        filtered: list[ApplicationRecord] = []
        actor_groups = {group.casefold() for group in actor.groups}
        actor_name = actor.principal.casefold()
        with self._lock:
            for application in visible:
                if needle and needle not in (
                    f"{application.application_id} {application.name}".casefold()
                ):
                    continue
                if lifecycle and application.lifecycle_state != lifecycle:
                    continue
                principals = tuple(
                    principal
                    for principal in self._principals.values()
                    if principal.application_id == application.application_id
                )
                if ownership == "owned" and not any(
                    principal.application_role is Role.OWNER
                    and (
                        (
                            principal.principal_type is PrincipalType.USER
                            and principal.principal_name.casefold() == actor_name
                        )
                        or (
                            principal.principal_type is PrincipalType.GROUP
                            and principal.principal_name.casefold() in actor_groups
                        )
                    )
                    for principal in principals
                ):
                    continue
                if ownership == "teams" and not any(
                    principal.principal_type is PrincipalType.GROUP
                    and principal.principal_name.casefold() in actor_groups
                    for principal in principals
                ):
                    continue
                tags = {tag.key: tag.value for tag in application.tags}
                if any(
                    tags.get(key) not in accepted
                    for key, accepted in tag_filters.items()
                ):
                    continue
                filtered.append(application)

        reverse = sort.startswith("-")
        field = sort.removeprefix("-")
        key = {
            "application": lambda item: (
                item.name.casefold(),
                item.application_id,
            ),
            "updated_at": lambda item: (
                item.updated_at,
                item.application_id,
            ),
            "owner": lambda item: (
                item.owner_principal.casefold(),
                item.application_id,
            ),
        }[field]
        filtered.sort(key=key, reverse=reverse)
        total = len(filtered)
        start = (page - 1) * page_size
        return tuple(filtered[start : start + page_size]), total

    def upsert_resource_binding(
        self, binding: ResourceBindingRecord
    ) -> ResourceBindingRecord:
        with self._lock:
            if binding.application_id not in self._applications:
                raise HubNotFoundError(
                    f"application {binding.application_id!r} was not found"
                )
            self._resources[binding.binding_id] = binding
            return binding

    def list_resource_bindings(
        self,
        application_id: str,
        *,
        environment: str | None = None,
    ) -> tuple[ResourceBindingRecord, ...]:
        with self._lock:
            if application_id not in self._applications:
                raise HubNotFoundError(f"application {application_id!r} was not found")
            return tuple(
                sorted(
                    (
                        binding
                        for binding in self._resources.values()
                        if binding.application_id == application_id
                        and (environment is None or binding.environment == environment)
                    ),
                    key=lambda binding: binding.binding_id,
                )
            )

    def create_evaluation(
        self,
        evaluation: EvaluationRunRecord,
        *,
        actor_request_id: str | None = None,
    ) -> EvaluationRunRecord:
        if evaluation.status is not EvaluationStatus.REQUESTED:
            raise HubConflictError("new evaluations must start in REQUESTED")
        if evaluation.row_version != 1:
            raise HubConflictError("new evaluations must have row_version 1")
        with self._lock:
            if evaluation.evaluation_run_id in self._evaluations:
                raise HubConflictError(
                    f"evaluation ID {evaluation.evaluation_run_id!r} already exists"
                )
            self._validate_application_version_locked(
                evaluation.application_id,
                evaluation.application_version_id,
                environment=evaluation.environment,
            )
            duplicate = next(
                (
                    current
                    for current in self._evaluations.values()
                    if current.status.active
                    and current.application_id == evaluation.application_id
                    and current.environment == evaluation.environment
                    and current.application_version_id
                    == evaluation.application_version_id
                ),
                None,
            )
            if duplicate is not None:
                raise DuplicateActiveEvaluationError(
                    f"evaluation {duplicate.evaluation_run_id!r} is already active"
                )
            self._evaluations[evaluation.evaluation_run_id] = evaluation
            self._append_event_locked(
                ActionEvent(
                    event_id=f"evaluation-requested:{evaluation.evaluation_run_id}",
                    entity_type=ActionEntityType.EVALUATION,
                    entity_id=evaluation.evaluation_run_id,
                    event_type=ActionEventType.EVALUATION_REQUESTED,
                    actor_principal=evaluation.requested_by,
                    actor_request_id=(
                        actor_request_id or f"evaluation:{evaluation.evaluation_run_id}"
                    ),
                    event_time=evaluation.requested_at,
                    previous_state=None,
                    new_state=evaluation.status.value,
                )
            )
            return evaluation

    def get_evaluation(self, evaluation_run_id: str) -> EvaluationRunRecord:
        with self._lock:
            try:
                return self._evaluations[evaluation_run_id]
            except KeyError as error:
                raise HubNotFoundError(
                    f"evaluation {evaluation_run_id!r} was not found"
                ) from error

    def list_evaluations(self, application_id: str) -> tuple[EvaluationRunRecord, ...]:
        with self._lock:
            if application_id not in self._applications:
                raise HubNotFoundError(f"application {application_id!r} was not found")
            return tuple(
                sorted(
                    (
                        evaluation
                        for evaluation in self._evaluations.values()
                        if evaluation.application_id == application_id
                    ),
                    key=lambda evaluation: (
                        evaluation.requested_at,
                        evaluation.evaluation_run_id,
                    ),
                )
            )

    def update_evaluation(
        self,
        evaluation: EvaluationRunRecord,
        *,
        expected_row_version: int,
        actor_request_id: str | None = None,
    ) -> EvaluationRunRecord:
        with self._lock:
            current = self.get_evaluation(evaluation.evaluation_run_id)
            self._assert_row_version(current.row_version, expected_row_version)
            self._assert_evaluation_identity(current, evaluation)
            if evaluation.row_version != current.row_version + 1:
                raise HubConflictError(
                    "updated evaluation must increment row_version exactly once"
                )
            if (
                evaluation.status is not current.status
                and evaluation.status
                not in self._EVALUATION_TRANSITIONS[current.status]
            ):
                raise InvalidStateTransitionError(
                    f"cannot transition evaluation from {current.status.value} "
                    f"to {evaluation.status.value}"
                )
            self._evaluations[evaluation.evaluation_run_id] = evaluation
            self._append_event_locked(
                ActionEvent(
                    event_id=f"evaluation-update:{uuid4()}",
                    entity_type=ActionEntityType.EVALUATION,
                    entity_id=evaluation.evaluation_run_id,
                    event_type=ActionEventType.EVALUATION_STATUS_CHANGED,
                    actor_principal=evaluation.requested_by,
                    actor_request_id=(
                        actor_request_id or f"evaluation:{evaluation.evaluation_run_id}"
                    ),
                    event_time=(
                        evaluation.completed_at
                        or evaluation.started_at
                        or evaluation.requested_at
                    ),
                    previous_state=current.status.value,
                    new_state=evaluation.status.value,
                    details_json=json.dumps(
                        (
                            {}
                            if evaluation.job_run_id is None
                            else {"job_run_id": evaluation.job_run_id}
                        ),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
            return evaluation

    def create_promotion_request(
        self,
        request: PromotionRequestRecord,
        *,
        actor_request_id: str | None = None,
    ) -> PromotionRequestRecord:
        if request.status is not PromotionStatus.PENDING_REVIEW:
            raise HubConflictError("new promotions must start in PENDING_REVIEW")
        if request.row_version != 1:
            raise HubConflictError("new promotions must have row_version 1")
        if not request.readiness_snapshot.ready:
            raise HubConflictError(
                "a blocked version cannot request environment promotion"
            )
        with self._lock:
            if request.promotion_request_id in self._promotions:
                raise HubConflictError(
                    f"promotion ID {request.promotion_request_id!r} already exists"
                )
            self._validate_application_version_locked(
                request.application_id,
                request.application_version_id,
                environment=request.source_environment,
            )
            duplicate = next(
                (
                    current
                    for current in self._promotions.values()
                    if current.status.active
                    and current.application_id == request.application_id
                    and current.application_version_id == request.application_version_id
                    and current.target_environment == request.target_environment
                ),
                None,
            )
            if duplicate is not None:
                raise DuplicateActivePromotionError(
                    f"promotion {duplicate.promotion_request_id!r} is already active"
                )
            self._promotions[request.promotion_request_id] = request
            self._append_event_locked(
                ActionEvent(
                    event_id=f"promotion-requested:{request.promotion_request_id}",
                    entity_type=ActionEntityType.PROMOTION,
                    entity_id=request.promotion_request_id,
                    event_type=ActionEventType.PROMOTION_REQUESTED,
                    actor_principal=request.requested_by,
                    actor_request_id=(
                        actor_request_id or f"promotion:{request.promotion_request_id}"
                    ),
                    event_time=request.requested_at,
                    previous_state=None,
                    new_state=request.status.value,
                )
            )
            return request

    def get_promotion_request(
        self, promotion_request_id: str
    ) -> PromotionRequestRecord:
        with self._lock:
            try:
                return self._promotions[promotion_request_id]
            except KeyError as error:
                raise HubNotFoundError(
                    f"promotion {promotion_request_id!r} was not found"
                ) from error

    def list_promotion_requests(
        self, application_id: str
    ) -> tuple[PromotionRequestRecord, ...]:
        with self._lock:
            if application_id not in self._applications:
                raise HubNotFoundError(f"application {application_id!r} was not found")
            return tuple(
                sorted(
                    (
                        request
                        for request in self._promotions.values()
                        if request.application_id == application_id
                    ),
                    key=lambda request: (
                        request.requested_at,
                        request.promotion_request_id,
                    ),
                )
            )

    def approve_promotion(
        self,
        promotion_request_id: str,
        *,
        actor: AuthorizationContext,
        expected_row_version: int,
        reviewed_at: datetime,
        actor_request_id: str,
        readiness_snapshot: ReadinessSnapshot | None = None,
        comment: str | None = None,
    ) -> PromotionRequestRecord:
        self._require_administrator(actor)
        with self._lock:
            current = self.get_promotion_request(promotion_request_id)
            self._assert_row_version(current.row_version, expected_row_version)
            if current.status is not PromotionStatus.PENDING_REVIEW:
                raise InvalidStateTransitionError(
                    f"cannot approve a promotion in {current.status.value}"
                )
            if current.requested_by.casefold() == actor.principal.casefold():
                raise FourEyesViolationError(
                    "a requester cannot approve their own environment promotion"
                )
            current_readiness = readiness_snapshot or current.readiness_snapshot
            self._validate_readiness_for_promotion(current, current_readiness)
            updated = current.model_copy(
                update={
                    "status": PromotionStatus.APPROVED,
                    "approval_readiness_snapshot": current_readiness,
                    "reviewed_by": actor.principal,
                    "reviewed_at": reviewed_at,
                    "review_comment": comment,
                    "row_version": current.row_version + 1,
                }
            )
            # model_copy does not revalidate; reconstruct to preserve contract checks.
            updated = PromotionRequestRecord.model_validate(
                updated.model_dump(mode="python")
            )
            event = self._promotion_event(
                current,
                updated,
                event_type=ActionEventType.PROMOTION_APPROVED,
                actor_principal=actor.principal,
                actor_request_id=actor_request_id,
                event_time=reviewed_at,
                comment=comment,
            )
            self._append_event_locked(event)
            self._promotions[promotion_request_id] = updated
            return updated

    def reject_promotion(
        self,
        promotion_request_id: str,
        *,
        actor: AuthorizationContext,
        expected_row_version: int,
        reviewed_at: datetime,
        actor_request_id: str,
        comment: str,
    ) -> PromotionRequestRecord:
        return self._review_promotion(
            promotion_request_id,
            actor=actor,
            expected_row_version=expected_row_version,
            reviewed_at=reviewed_at,
            actor_request_id=actor_request_id,
            comment=comment,
            status=PromotionStatus.REJECTED,
            event_type=ActionEventType.PROMOTION_REJECTED,
        )

    def request_promotion_changes(
        self,
        promotion_request_id: str,
        *,
        actor: AuthorizationContext,
        expected_row_version: int,
        reviewed_at: datetime,
        actor_request_id: str,
        comment: str,
    ) -> PromotionRequestRecord:
        return self._review_promotion(
            promotion_request_id,
            actor=actor,
            expected_row_version=expected_row_version,
            reviewed_at=reviewed_at,
            actor_request_id=actor_request_id,
            comment=comment,
            status=PromotionStatus.CHANGES_REQUESTED,
            event_type=ActionEventType.PROMOTION_CHANGES_REQUESTED,
        )

    def update_promotion(
        self,
        request: PromotionRequestRecord,
        *,
        expected_row_version: int,
        actor_principal: str,
        actor_request_id: str,
        event_time: datetime,
        comment: str | None = None,
    ) -> PromotionRequestRecord:
        """Advance execution state after approval.

        Review transitions are intentionally excluded so this method cannot bypass
        administrator authorization or four-eyes approval.
        """

        with self._lock:
            current = self.get_promotion_request(request.promotion_request_id)
            self._assert_row_version(current.row_version, expected_row_version)
            self._assert_promotion_identity(current, request)
            if request.row_version != current.row_version + 1:
                raise HubConflictError(
                    "updated promotion must increment row_version exactly once"
                )
            if request.status in {
                PromotionStatus.APPROVED,
                PromotionStatus.REJECTED,
                PromotionStatus.CHANGES_REQUESTED,
            }:
                raise HubAuthorizationError(
                    "review transitions require the dedicated administrator methods"
                )
            if request.status not in self._PROMOTION_TRANSITIONS[current.status]:
                raise InvalidStateTransitionError(
                    f"cannot transition promotion from {current.status.value} "
                    f"to {request.status.value}"
                )
            event_type = {
                PromotionStatus.EXECUTING: ActionEventType.PROMOTION_EXECUTION_STARTED,
                PromotionStatus.SUCCEEDED: ActionEventType.PROMOTION_SUCCEEDED,
                PromotionStatus.FAILED: ActionEventType.PROMOTION_FAILED,
                PromotionStatus.CANCELLED: ActionEventType.PROMOTION_CANCELLED,
                PromotionStatus.PENDING_REVIEW: ActionEventType.PROMOTION_REQUESTED,
            }[request.status]
            event = self._promotion_event(
                current,
                request,
                event_type=event_type,
                actor_principal=actor_principal,
                actor_request_id=actor_request_id,
                event_time=event_time,
                comment=comment,
            )
            self._append_event_locked(event)
            self._promotions[request.promotion_request_id] = request
            return request

    def append_action_event(self, event: ActionEvent) -> ActionEvent:
        with self._lock:
            self._append_event_locked(event)
            return event

    def list_action_events(
        self,
        *,
        entity_type: ActionEntityType | None = None,
        entity_id: str | None = None,
    ) -> tuple[ActionEvent, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        event
                        for event in self._events.values()
                        if (entity_type is None or event.entity_type is entity_type)
                        and (entity_id is None or event.entity_id == entity_id)
                    ),
                    key=lambda event: (event.event_time, event.event_id),
                )
            )

    def list_application_action_events(
        self,
        application_id: str,
    ) -> tuple[ActionEvent, ...]:
        with self._lock:
            if application_id not in self._applications:
                raise HubNotFoundError(f"application {application_id!r} was not found")
            evaluation_ids = {
                evaluation.evaluation_run_id
                for evaluation in self._evaluations.values()
                if evaluation.application_id == application_id
            }
            promotion_ids = {
                promotion.promotion_request_id
                for promotion in self._promotions.values()
                if promotion.application_id == application_id
            }
            allowed_entities = {
                (ActionEntityType.APPLICATION, application_id),
                *((ActionEntityType.EVALUATION, item) for item in evaluation_ids),
                *((ActionEntityType.PROMOTION, item) for item in promotion_ids),
            }
            return tuple(
                sorted(
                    (
                        event
                        for event in self._events.values()
                        if (event.entity_type, event.entity_id) in allowed_entities
                    ),
                    key=lambda event: (event.event_time, event.event_id),
                )
            )

    def _review_promotion(
        self,
        promotion_request_id: str,
        *,
        actor: AuthorizationContext,
        expected_row_version: int,
        reviewed_at: datetime,
        actor_request_id: str,
        comment: str,
        status: PromotionStatus,
        event_type: ActionEventType,
    ) -> PromotionRequestRecord:
        self._require_administrator(actor)
        if not comment.strip():
            raise HubConflictError("a review comment is required")
        with self._lock:
            current = self.get_promotion_request(promotion_request_id)
            self._assert_row_version(current.row_version, expected_row_version)
            if current.status is not PromotionStatus.PENDING_REVIEW:
                raise InvalidStateTransitionError(
                    f"cannot review a promotion in {current.status.value}"
                )
            updated = current.model_copy(
                update={
                    "status": status,
                    "reviewed_by": actor.principal,
                    "reviewed_at": reviewed_at,
                    "review_comment": comment,
                    "row_version": current.row_version + 1,
                }
            )
            updated = PromotionRequestRecord.model_validate(
                updated.model_dump(mode="python")
            )
            event = self._promotion_event(
                current,
                updated,
                event_type=event_type,
                actor_principal=actor.principal,
                actor_request_id=actor_request_id,
                event_time=reviewed_at,
                comment=comment,
            )
            self._append_event_locked(event)
            self._promotions[promotion_request_id] = updated
            return updated

    def _validate_application_version_locked(
        self,
        application_id: str,
        application_version_id: str,
        *,
        environment: str,
    ) -> None:
        if application_id not in self._applications:
            raise HubNotFoundError(f"application {application_id!r} was not found")
        version = self._versions.get(application_version_id)
        if version is None:
            raise HubNotFoundError(
                f"application version {application_version_id!r} was not found"
            )
        if (
            version.application_id != application_id
            or version.environment != environment
        ):
            raise HubConflictError(
                "application version does not match the requested "
                "application/environment"
            )

    @staticmethod
    def _assert_row_version(actual: int, expected: int) -> None:
        if actual != expected:
            raise OptimisticConcurrencyError(expected=expected, actual=actual)

    @staticmethod
    def _assert_evaluation_identity(
        current: EvaluationRunRecord, updated: EvaluationRunRecord
    ) -> None:
        immutable = (
            "application_id",
            "environment",
            "application_version_id",
            "evaluation_profile",
            "dataset_name",
            "dataset_version",
            "job_id",
            "requested_by",
            "requested_at",
        )
        changed = any(
            getattr(current, field) != getattr(updated, field) for field in immutable
        )
        if changed:
            raise HubConflictError("immutable evaluation fields cannot change")

    @staticmethod
    def _assert_promotion_identity(
        current: PromotionRequestRecord, updated: PromotionRequestRecord
    ) -> None:
        immutable = (
            "application_id",
            "source_environment",
            "target_environment",
            "application_version_id",
            "requested_by",
            "requested_at",
            "promotion_job_id",
        )
        changed = any(
            getattr(current, field) != getattr(updated, field) for field in immutable
        )
        if changed:
            raise HubConflictError("immutable promotion fields cannot change")
        if current.readiness_snapshot != updated.readiness_snapshot:
            raise HubConflictError(
                "execution transitions cannot replace request-time readiness evidence"
            )
        if current.approval_readiness_snapshot != updated.approval_readiness_snapshot:
            raise HubConflictError(
                "execution transitions cannot replace approval readiness evidence"
            )

    @staticmethod
    def _require_administrator(actor: AuthorizationContext) -> None:
        if not actor.has_platform_role(Role.PLATFORM_ADMINISTRATOR):
            raise HubAuthorizationError("platform administrator role is required")

    @staticmethod
    def _validate_readiness_for_promotion(
        request: PromotionRequestRecord, snapshot: ReadinessSnapshot
    ) -> None:
        if not snapshot.ready:
            raise HubConflictError("blocking readiness checks prevent approval")
        if snapshot.application_id != request.application_id:
            raise HubConflictError("readiness snapshot belongs to another application")
        if snapshot.application_version_id != request.application_version_id:
            raise HubConflictError("readiness snapshot belongs to another version")

    def _append_event_locked(self, event: ActionEvent) -> None:
        if event.event_id in self._events:
            raise HubConflictError(f"action event {event.event_id!r} already exists")
        self._events[event.event_id] = event

    @staticmethod
    def _promotion_event(
        previous: PromotionRequestRecord,
        updated: PromotionRequestRecord,
        *,
        event_type: ActionEventType,
        actor_principal: str,
        actor_request_id: str,
        event_time: datetime,
        comment: str | None,
    ) -> ActionEvent:
        details = {"row_version": updated.row_version}
        if updated.promotion_job_run_id is not None:
            details["promotion_job_run_id"] = updated.promotion_job_run_id
        if event_type is ActionEventType.PROMOTION_APPROVED:
            approval_snapshot = updated.approval_readiness_snapshot
            details.update(
                {
                    "request_readiness_evaluated_at": (
                        updated.readiness_snapshot.evaluated_at.isoformat()
                    ),
                    "approval_readiness_evaluated_at": (
                        None
                        if approval_snapshot is None
                        else approval_snapshot.evaluated_at.isoformat()
                    ),
                    "readiness_evidence_changed": (
                        approval_snapshot is not None
                        and approval_snapshot.decision_signature()
                        != updated.readiness_snapshot.decision_signature()
                    ),
                }
            )
        return ActionEvent(
            event_id=f"promotion-update:{uuid4()}",
            entity_type=ActionEntityType.PROMOTION,
            entity_id=updated.promotion_request_id,
            event_type=event_type,
            actor_principal=actor_principal,
            actor_request_id=actor_request_id,
            event_time=event_time,
            previous_state=previous.status.value,
            new_state=updated.status.value,
            comment=comment,
            details_json=json.dumps(details, sort_keys=True, separators=(",", ":")),
        )


class UnavailableHubRepository:
    """Fail-closed repository used when no durable store is configured."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    @property
    def available(self) -> bool:
        return False

    def _raise(self, *args, **kwargs):
        raise HubRepositoryUnavailableError(self.reason)

    register_application = _raise
    get_application = _raise
    get_visible_application = _raise
    get_current_version = _raise
    list_versions = _raise
    upsert_application_principal = _raise
    replace_application_principals = _raise
    list_application_principals = _raise
    list_visible_applications = _raise
    query_visible_applications = _raise
    upsert_resource_binding = _raise
    list_resource_bindings = _raise
    create_evaluation = _raise
    get_evaluation = _raise
    list_evaluations = _raise
    update_evaluation = _raise
    create_promotion_request = _raise
    get_promotion_request = _raise
    list_promotion_requests = _raise
    approve_promotion = _raise
    reject_promotion = _raise
    request_promotion_changes = _raise
    update_promotion = _raise
    append_action_event = _raise
    list_action_events = _raise
    list_application_action_events = _raise
