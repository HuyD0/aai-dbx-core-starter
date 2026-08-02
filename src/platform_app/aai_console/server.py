"""FastAPI application for the AAI platform console.

Serves a small server-rendered UI: full pages for navigation, HTML fragments for
in-place swaps, and a narrow JSON surface. There is no bundler and no third-party
client library — `scripts/cloud-verify.sh` performs an offline `uv sync --locked`, so
an npm lockfile ecosystem would be a change of security posture, not a dependency.

Responses are assembled field by field. Never serialise an SDK object wholesale:
`dataclasses.asdict()` recurses into `PlatformSettings.raw`, and `repr=False` does not
prevent it.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote, urlencode
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from . import __version__
from .checks import (
    PLATFORM_STATE_HEADING,
    PlatformCheck,
    assert_platform_state,
    run_checks,
)
from .config import ConfigError, ConsoleConfig, HubJobMode, load_config
from .content import Track, inline_code, resolve_placeholders
from .estimator import (
    EstimateError,
    EstimateRequest,
    estimate,
    estimate_csv,
    estimator_page_context,
)
from .generate import GenerateError, GenerateRequest, bundle_init
from .hub.api_models import (
    AdminActionListResponse,
    ApplicationDetailResponse,
    ApplicationListResponse,
    EvaluationListResponse,
    EvaluationResponse,
    PromotionListResponse,
    RegistrationResponse,
    VersionsResponse,
)
from .hub.identity import (
    FailClosedRoleResolver,
    HubAuthenticationError,
    HubRoleResolutionError,
    RoleAssignment,
    RoleResolver,
    StaticRoleResolver,
    actor_view,
    authorization_context_from_headers,
)
from .hub.jobs import (
    DatabricksJobRunner,
    JobRunner,
    RecordingJobRunner,
    UnavailableJobRunner,
)
from .hub.manifest import load_manifest, manifest_json_schema
from .hub.models import (
    AuthorizationContext,
    EvaluationRunRecord,
    EvaluationSummary,
    PromotionRequestRecord,
    PromotionStatus,
    ReadinessStatus,
    Role,
)
from .hub.repository import (
    FourEyesViolationError,
    HubAuthorizationError,
    HubConflictError,
    HubNotFoundError,
    HubRepository,
    HubRepositoryError,
    HubRepositoryUnavailableError,
    InMemoryHubRepository,
    OptimisticConcurrencyError,
    UnavailableHubRepository,
)
from .hub.service import (
    EvaluationRequest,
    HubCapabilityUnavailableError,
    HubExternalServiceError,
    HubPermissionDeniedError,
    HubQueryValidationError,
    HubReadinessBlockedError,
    HubService,
    HubServiceError,
    PortfolioQuery,
    PromotionRequest,
    PromotionReviewRequest,
    RegistrationRequest,
    parse_tag_filters,
)
from .pricing import load_snapshot, usd
from .registry import TrackRegistry

PACKAGE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger("aai_console")

_CREDENTIAL_SHAPED = re.compile(
    r"(?i)(?:"
    r"(?<![A-Za-z0-9])(?:dapi|github_pat_|gh[oprsu]_|sk-)[A-Za-z0-9_-]{8,}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r"|(?:[?&]sig=|accountkey=|sharedaccesssignature=)"
    r")"
)


def _route_template(scope: dict) -> str:
    """Return framework-owned route text, never attacker-controlled path segments."""

    route = scope.get("route")
    path = getattr(route, "path", None)
    if (
        isinstance(path, str)
        and path.startswith("/")
        and len(path) <= 300
        and not any(character in path for character in "\r\n\x00")
    ):
        return path
    return "<unmatched>"


def _request_id_from_scope(scope: dict) -> str:
    for raw_name, raw_value in scope.get("headers", ()):
        if raw_name.lower() != b"x-request-id":
            continue
        supplied = raw_value.decode("ascii", "ignore").strip()
        if _REQUEST_ID.fullmatch(supplied) and not _CREDENTIAL_SHAPED.search(supplied):
            return supplied
        break
    return str(uuid4())


class ContainExceptions:
    """Outermost ASGI wrapper: stops an exception message reaching the server's log.

    An `@app.exception_handler(Exception)` is not enough. Starlette's
    ServerErrorMiddleware sends the handler's response and then deliberately re-raises
    ("This allows servers to log the error"), so uvicorn logs the traceback *and the
    exception message*. That message can carry a provider payload or a credential; an
    app's log is readable by anyone with CAN MANAGE, and this process's environment
    holds a live OAuth client secret.

    ServerErrorMiddleware is the outermost layer Starlette builds, so `add_middleware`
    cannot get outside it — the app object has to be wrapped instead. Attribute access
    proxies through, so this stays a drop-in for the FastAPI instance.
    """

    def __init__(self, app) -> None:
        self._app = app

    def __getattr__(self, name):
        return getattr(self._app, name)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = _request_id_from_scope(scope)
        scope["aai.request_id"] = request_id
        started = False

        async def _send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self._app(scope, receive, _send)
        except Exception as exc:
            # Type only — never str(exc), which is the leak this class exists to stop.
            logger.error(
                json.dumps(
                    {
                        "event": "hub_unhandled_error",
                        "request_id": request_id,
                        "route": _route_template(scope),
                        "error_type": type(exc).__name__,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            if not started:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [
                            (b"content-type", b"application/problem+json"),
                            (b"x-request-id", request_id.encode("ascii")),
                        ],
                    }
                )
                body = json.dumps(
                    {
                        "type": "about:blank",
                        "title": "Internal server error",
                        "status": 500,
                        "detail": "The request could not be completed.",
                        "instance": _route_template(scope),
                        "request_id": request_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                await send({"type": "http.response.body", "body": body})


_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_APPLICATION_ID = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_METRIC_KEY = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


class ProblemError(RuntimeError):
    """Safe RFC 7807 problem raised by an HTTP delivery adapter."""

    def __init__(
        self,
        status: int,
        title: str,
        detail: str,
        *,
        problem_type: str = "about:blank",
        errors: list[dict] | None = None,
    ) -> None:
        self.status = status
        self.title = title
        self.detail = detail
        self.problem_type = problem_type
        self.errors = errors
        super().__init__(title)


def _problem_response(
    request: Request,
    *,
    status: int,
    title: str,
    detail: str,
    problem_type: str = "about:blank",
    errors: list[dict] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unavailable")
    body: dict = {
        "type": problem_type,
        "title": title,
        "status": status,
        "detail": detail,
        "instance": _route_template(request.scope),
        "request_id": request_id,
    }
    if errors:
        body["errors"] = errors
    return JSONResponse(
        body,
        status_code=status,
        media_type="application/problem+json",
        headers={"X-Request-Id": request_id},
    )


def _validation_errors(error: ValidationError | RequestValidationError) -> list[dict]:
    """Return paths and rule messages without reflecting submitted values."""

    safe = []
    issues = (
        error.errors()
        if isinstance(error, RequestValidationError)
        else error.errors(include_url=False, include_input=False)
    )
    for issue in issues:
        location = [
            str(component)
            for component in issue.get("loc", ())
            if component not in {"body"}
        ]
        safe.append(
            {
                "path": "$"
                + "".join(
                    f"[{part}]" if part.isdigit() else f".{part}" for part in location
                ),
                "message": str(issue.get("msg", "invalid value"))[:300],
                "code": str(issue.get("type", "validation_error"))[:100],
            }
        )
    return safe


def _repository_for(config: ConsoleConfig) -> HubRepository:
    if config.hub_state_mode.value == "memory":
        return InMemoryHubRepository()
    return UnavailableHubRepository(
        "No approved durable Hub store is bound. Hosted registry and workflow "
        "writes remain disabled."
    )


def _job_runner_for(config: ConsoleConfig) -> JobRunner:
    if config.hub_job_mode is HubJobMode.DATABRICKS:
        return DatabricksJobRunner()
    if config.hub_job_mode is HubJobMode.PREVIEW:
        return RecordingJobRunner()
    return UnavailableJobRunner()


def _hub_capabilities(config: ConsoleConfig, service: HubService) -> dict[str, str]:
    if service.available and config.hosted:
        state_store = "ready"
        state_detail = "The configured durable operational store is available."
    elif service.available:
        state_store = "local_preview"
        state_detail = (
            "Using an explicit in-memory local preview. Data is not durable and no "
            "cloud resource is created."
        )
    else:
        state_store = "unavailable"
        state_detail = (
            "Bind an approved Lakebase Autoscaling schema or reviewed SQL fallback "
            "before enabling hosted registry and workflow writes."
        )
    jobs = service.job_runner.capability
    if not service.available:
        workflow_status = "gated"
        workflow_detail = "Workflow actions also require the durable operational store."
    elif jobs.enabled and not jobs.remote_execution:
        workflow_status = "local_preview"
        workflow_detail = (
            "Preview runs are process-local and never launch a Databricks job."
        )
    elif jobs.remote_execution:
        workflow_status = "gated"
        workflow_detail = (
            "The remote launch adapter is configured, but durable status "
            "reconciliation and sanitized result ingestion are not implemented."
        )
    else:
        workflow_status = "gated"
        workflow_detail = jobs.detail
    return {
        "state_store": state_store,
        "state_store_detail": state_detail,
        "registration": "ready" if service.registration_enabled else "gated",
        "registration_detail": (
            "Approved CI callers and operational storage are configured."
            if service.registration_enabled
            else "Registration requires a durable store and an explicit CI allowlist."
        ),
        "evaluation": workflow_status,
        "evaluation_detail": workflow_detail,
        "promotion": workflow_status,
        "promotion_detail": workflow_detail,
        "costs": "unavailable",
        "costs_detail": (
            "Reviewed Unity Catalog serving views and freshness metadata are not "
            "configured. The Hub will not query account system tables per request."
        ),
        "traces": "unavailable",
        "traces_detail": (
            "A bounded, sanitized MLflow trace-summary adapter is not configured."
        ),
        "optimization": "unavailable",
        "optimization_detail": (
            "Versioned detector output and visibility-enforcing serving views are "
            "not configured."
        ),
        "admin": "gated",
        "admin_detail": (
            "Account-group resolution is fail-closed until the approved role source "
            "and group names are configured."
        ),
    }


def _actor_for(request: Request) -> AuthorizationContext:
    config: ConsoleConfig = request.app.state.config
    try:
        actor = authorization_context_from_headers(
            request.headers,
            hosted=config.hosted,
            local_actor=config.hub_local_actor,
            role_resolver=request.app.state.role_resolver,
        )
    except HubAuthenticationError as error:
        request.state.authentication_outcome = "denied"
        request.state.authorization_decision = "not_evaluated"
        raise ProblemError(
            401,
            "Authentication required",
            "A trusted Databricks Apps identity assertion is required.",
        ) from error
    except HubRoleResolutionError as error:
        request.state.authentication_outcome = "authenticated"
        request.state.authorization_decision = "unavailable"
        raise ProblemError(
            503,
            "Authorization unavailable",
            "The trusted role source could not resolve this request.",
        ) from error
    request.state.actor_principal = actor.principal
    request.state.authentication_outcome = "authenticated"
    request.state.authorization_decision = "evaluated"
    return actor


def _validate_query_keys(
    request: Request,
    allowed: frozenset[str],
) -> None:
    observed = [name for name, _ in request.query_params.multi_items()]
    unknown = sorted(set(observed).difference(allowed))
    duplicates = sorted(name for name in set(observed) if observed.count(name) > 1)
    if unknown:
        raise ProblemError(
            400,
            "Unsupported filter",
            "One or more query filter keys are not supported.",
            errors=[
                {
                    "path": f"$.query.{name}",
                    "message": "unsupported query key",
                    "code": "unsupported_filter",
                }
                for name in unknown
            ],
        )
    if duplicates:
        raise ProblemError(
            400,
            "Ambiguous filter",
            "A query filter key may be supplied only once.",
            errors=[
                {
                    "path": f"$.query.{name}",
                    "message": "duplicate query key",
                    "code": "duplicate_filter",
                }
                for name in duplicates
            ],
        )


def _application_id(value: str) -> str:
    if not _APPLICATION_ID.fullmatch(value):
        raise ProblemError(
            404,
            "Application not found",
            "The requested application does not exist or is not visible.",
        )
    return value


def _model_json(model) -> dict:
    return model.model_dump(mode="json", by_alias=True)


def _version_projection(version) -> dict:
    return {
        "versionId": version.version_id,
        "applicationId": version.application_id,
        "environment": version.environment,
        "gitRepository": version.git_repository,
        "gitCommitSha": version.git_commit_sha,
        "manifestVersion": version.manifest_version,
        "manifestHash": version.manifest_hash,
        "registeredBy": version.registered_by,
        "registeredAt": version.registered_at.isoformat(),
        "deploymentTarget": version.deployment_target,
        "current": version.is_current,
    }


def _evaluation_projection(evaluation: EvaluationRunRecord) -> dict:
    """Expose only bounded scalar metrics, never raw reconciler/provider payloads."""

    metrics: dict[str, bool | int | float] = {}
    if evaluation.summary_json:
        summary = EvaluationSummary.model_validate_json(evaluation.summary_json)
        metrics = dict(summary.metrics)
    return {
        "evaluationRunId": evaluation.evaluation_run_id,
        "applicationId": evaluation.application_id,
        "environment": evaluation.environment,
        "applicationVersionId": evaluation.application_version_id,
        "evaluationProfile": evaluation.evaluation_profile,
        "datasetName": evaluation.dataset_name,
        "datasetVersion": evaluation.dataset_version,
        "jobId": evaluation.job_id,
        "jobRunId": evaluation.job_run_id,
        "mlflowRunId": evaluation.mlflow_run_id,
        "requestedBy": evaluation.requested_by,
        "status": evaluation.status.value,
        "requestedAt": evaluation.requested_at.isoformat(),
        "startedAt": (
            None if evaluation.started_at is None else evaluation.started_at.isoformat()
        ),
        "completedAt": (
            None
            if evaluation.completed_at is None
            else evaluation.completed_at.isoformat()
        ),
        "metrics": metrics,
        "failureCategory": (
            "evaluation_failed" if evaluation.status.value == "FAILED" else None
        ),
    }


def _readiness_projection(snapshot) -> dict:
    return {
        "applicationId": snapshot.application_id,
        "environment": snapshot.environment,
        "applicationVersionId": snapshot.application_version_id,
        "profileId": snapshot.profile_id,
        "profileVersion": snapshot.profile_version,
        "evaluatedAt": snapshot.evaluated_at.isoformat(),
        "ready": snapshot.ready,
        "results": [
            {
                "ruleId": result.rule_id,
                "ruleVersion": result.rule_version,
                "description": result.description,
                "severity": result.severity.value,
                "status": result.status.value,
                "evidence": list(result.evidence),
                "evaluatedAt": result.evaluated_at.isoformat(),
                "remediation": result.remediation,
            }
            for result in snapshot.results
        ],
    }


def _application_projection(item) -> dict:
    application = item.application
    version = item.current_version
    blocking = sum(
        result.status in {ReadinessStatus.FAIL, ReadinessStatus.UNKNOWN}
        and result.severity.value == "BLOCKING"
        for result in item.readiness.results
    )
    return {
        "applicationId": application.application_id,
        "name": application.name,
        "description": application.description,
        "owner": application.owner_principal,
        "supportGroup": application.support_group,
        "businessDomain": application.business_domain,
        "costCenter": application.cost_center,
        "riskTier": application.risk_tier,
        "lifecycle": application.lifecycle_state,
        "tags": {tag.key: tag.value for tag in application.tags},
        "environment": version.environment,
        "deployments": [
            _version_projection(deployment) for deployment in item.deployments
        ],
        "health": {
            "status": "UNKNOWN",
            "reason": "Governed health aggregates are not configured.",
            "evidenceAt": None,
        },
        "readiness": {
            "ready": item.readiness.ready,
            "blockingIssues": blocking,
            "profileId": item.readiness.profile_id,
            "evaluatedAt": item.readiness.evaluated_at.isoformat(),
        },
        "currentVersion": _version_projection(version),
        "lastSuccessfulEvaluation": None,
        "applicationCost": None,
        "directUserCost": None,
        "requestVolume": None,
        "p95LatencyMs": None,
        "errorRate": None,
        "outstandingActions": None,
    }


def _portfolio_template_item(item) -> dict:
    application = item.application
    version = item.current_version
    blocking = sum(
        result.status in {ReadinessStatus.FAIL, ReadinessStatus.UNKNOWN}
        and result.severity.value == "BLOCKING"
        for result in item.readiness.results
    )
    return {
        "application_id": application.application_id,
        "name": application.name,
        "owner": application.owner_principal,
        "support_group": application.support_group,
        "lifecycle": application.lifecycle_state,
        "environment": version.environment,
        "health": "UNKNOWN",
        "health_reason": "Health aggregates unavailable",
        "readiness": "READY" if item.readiness.ready else "BLOCKED",
        "readiness_tone": "healthy" if item.readiness.ready else "critical",
        "blocking_issues": blocking,
        "git_sha": version.git_commit_sha,
        "registered_at": version.registered_at.strftime("%Y-%m-%d %H:%M UTC"),
        "deployments": tuple(
            {
                "environment": deployment.environment,
                "git_sha": deployment.git_commit_sha,
                "registered_at": deployment.registered_at.strftime(
                    "%Y-%m-%d %H:%M UTC"
                ),
            }
            for deployment in item.deployments
        ),
        "tags": tuple((tag.key, tag.value) for tag in application.tags),
        "application_cost": None,
        "direct_cost": None,
        "last_evaluation": None,
        "request_volume": None,
        "p95_latency": None,
        "error_rate": None,
        "outstanding_actions": None,
    }


def _readiness_template(snapshot) -> list[dict]:
    tone = {
        ReadinessStatus.PASS: "healthy",
        ReadinessStatus.FAIL: "critical",
        ReadinessStatus.UNKNOWN: "warning",
        ReadinessStatus.NOT_APPLICABLE: "neutral",
    }
    return [
        {
            "rule_id": result.rule_id,
            "description": result.description,
            "status": result.status.value,
            "tone": tone[result.status],
            "evidence": "; ".join(result.evidence),
            "severity": result.severity.value,
            "version": result.rule_version,
            "evaluated_at": result.evaluated_at.strftime("%Y-%m-%d %H:%M UTC"),
            "remediation": result.remediation,
        }
        for result in snapshot.results
    ]


def _resource_template_items(manifest, environment: str) -> list[dict]:
    resources = manifest.spec.resources
    items: list[dict] = []
    for resource_type, resource_id in (
        ("evaluation job", resources.evaluation_job_id),
        ("evaluation job key", resources.evaluation_job_key),
        ("promotion job", resources.promotion_job_id),
        ("promotion job key", resources.promotion_job_key),
        ("SQL warehouse", resources.sql_warehouse_id),
    ):
        if resource_id is not None:
            items.append(
                {"type": resource_type, "name": resource_id, "id": resource_id}
            )
    for resource_type, values in (
        ("AI search index", resources.ai_search_indexes),
        ("Unity Catalog function", resources.unity_catalog_functions),
        ("MCP service", resources.mcp_services),
    ):
        items.extend(
            {"type": resource_type, "name": value, "id": value} for value in values
        )
    configured = manifest.spec.environments[environment]
    for resource_type, value in (
        ("Databricks App", configured.databricks_app_name),
        ("MLflow experiment", configured.mlflow_experiment_id),
        ("AI Gateway service", configured.ai_gateway_service),
    ):
        if value is not None:
            items.append({"type": resource_type, "name": value, "id": value})
    return items


def _evaluation_template_items(
    evaluations,
    versions,
    config: ConsoleConfig,
) -> list[dict]:
    items = []
    host = config.identifiers.get("databricks_host", "").rstrip("/")
    version_details = {}
    for version in versions:
        manifest = load_manifest(json.loads(version.manifest_json))
        environment = manifest.spec.environments[version.environment]
        version_details[version.version_id] = {
            "git_sha": version.git_commit_sha,
            "mlflow_experiment_id": environment.mlflow_experiment_id,
        }
    for evaluation in reversed(evaluations):
        summary = "No summarized metrics"
        metrics = _evaluation_projection(evaluation)["metrics"]
        if metrics:
            summary = (
                ", ".join(f"{key}: {value}" for key, value in sorted(metrics.items()))[
                    :300
                ]
                or summary
            )
        run_url = None
        if host and evaluation.job_run_id:
            run_url = (
                f"{host}/jobs/{quote(evaluation.job_id, safe='')}/runs/"
                f"{quote(evaluation.job_run_id, safe='')}"
            )
        version_detail = version_details.get(evaluation.application_version_id, {})
        mlflow_url = None
        experiment_id = version_detail.get("mlflow_experiment_id")
        if host and experiment_id and evaluation.mlflow_run_id:
            mlflow_url = (
                f"{host}/ml/experiments/{quote(experiment_id, safe='')}/runs/"
                f"{quote(evaluation.mlflow_run_id, safe='')}"
            )
        items.append(
            {
                "status": evaluation.status.value,
                "tone": (
                    "healthy"
                    if evaluation.status.value == "SUCCEEDED"
                    else (
                        "critical" if evaluation.status.value == "FAILED" else "warning"
                    )
                ),
                "git_sha": version_detail.get("git_sha", "Unavailable"),
                "dataset": evaluation.dataset_name,
                "dataset_version": evaluation.dataset_version,
                "summary": summary,
                "requested_by": evaluation.requested_by,
                "requested_at": evaluation.requested_at.strftime("%Y-%m-%d %H:%M UTC"),
                "run_url": run_url,
                "mlflow_url": mlflow_url,
            }
        )
    return items


def _activity_template_items(repository, application_id: str) -> list[dict]:
    try:
        events = repository.list_application_action_events(application_id)
    except HubRepositoryError:
        return []
    return [
        {
            "event_time": event.event_time.strftime("%Y-%m-%d %H:%M UTC"),
            "event_type": event.event_type.value.replace("_", " ").title(),
            "summary": (
                f"{event.previous_state or 'new'} → {event.new_state or 'recorded'}"
            ),
            "actor": event.actor_principal,
        }
        for event in reversed(events)
    ]


def _pagination_url(request: Request, page: int) -> str:
    values = dict(request.query_params)
    values["page"] = str(page)
    return f"{request.url.path}?{urlencode(values)}"


def _install_openapi_contract(app: FastAPI) -> None:
    """Describe the trusted ingress identity and shared RFC 7807 failures."""

    def custom_openapi() -> dict:
        if app.openapi_schema is not None:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            routes=app.routes,
        )
        components = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {})["DatabricksAppIdentity"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-Forwarded-User",
            "description": (
                "Authenticated identity asserted by the Databricks Apps reverse "
                "proxy. Callers authenticate to the App with OAuth; this header is "
                "not a caller-selectable role or permission claim."
            ),
        }
        components.setdefault("schemas", {})["Problem"] = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "type",
                "title",
                "status",
                "detail",
                "instance",
                "request_id",
            ],
            "properties": {
                "type": {"type": "string"},
                "title": {"type": "string"},
                "status": {"type": "integer"},
                "detail": {"type": "string"},
                "instance": {"type": "string"},
                "request_id": {"type": "string"},
                "errors": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
        }
        problem_response = {
            "content": {
                "application/problem+json": {
                    "schema": {"$ref": "#/components/schemas/Problem"}
                }
            }
        }
        descriptions = {
            "401": "Trusted Databricks Apps identity is unavailable.",
            "403": "The authenticated actor is not authorized.",
            "404": "The resource is absent or not visible.",
            "409": "The requested workflow state conflicts with current state.",
            "422": "One or more request fields are invalid.",
            "502": "An approved provider operation failed safely.",
            "503": "A required Hub capability is unavailable.",
            "500": "The request failed without exposing provider details.",
        }
        public_paths = {
            "/api/v1/capabilities",
            "/api/v1/manifest-schemas/ai-platform-v1",
        }
        for path, path_item in schema.get("paths", {}).items():
            if not path.startswith("/api/v1/") or path in public_paths:
                continue
            for method in {"get", "post", "put", "patch", "delete"}:
                operation = path_item.get(method)
                if operation is None:
                    continue
                operation["security"] = [{"DatabricksAppIdentity": []}]
                responses = operation.setdefault("responses", {})
                for status, description in descriptions.items():
                    declared = {
                        "description": description,
                        **problem_response,
                    }
                    if status == "422":
                        # FastAPI's generated HTTPValidationError contract is replaced
                        # by the runtime RFC 7807 handler above.
                        responses[status] = declared
                    else:
                        responses.setdefault(status, declared)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi


def _render_tracks(
    tracks: tuple[Track, ...], config: ConsoleConfig
) -> tuple[Track, ...]:
    """Substitute identifier placeholders in every code block."""
    rendered = []
    for track in tracks:
        steps = []
        for step in track.steps:
            blocks = tuple(
                type(block)(
                    lang=block.lang,
                    code=resolve_placeholders(block.code, config),
                )
                for block in step.blocks
            )
            steps.append(type(step)(**{**step.__dict__, "blocks": blocks}))
        rendered.append(type(track)(**{**track.__dict__, "steps": tuple(steps)}))
    return tuple(rendered)


def create_app(
    config: ConsoleConfig | None = None,
    *,
    probe=None,
    registry: TrackRegistry | None = None,
    hub_repository: HubRepository | None = None,
    role_resolver: RoleResolver | None = None,
    job_runner: JobRunner | None = None,
) -> FastAPI:
    app = FastAPI(
        title="AAI AI Platform Hub",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    app.state.config = config if config is not None else load_config()
    app.state.probe = probe
    app.state.registry = registry if registry is not None else TrackRegistry.default()
    if role_resolver is not None:
        app.state.role_resolver = role_resolver
    elif app.state.config.hosted:
        app.state.role_resolver = FailClosedRoleResolver()
    else:
        # Local preview has no trusted group source. Its single configured actor gets
        # an explicit preview-only fleet/admin assignment so generated examples remain
        # usable without turning manifest ownership metadata into permissions.
        app.state.role_resolver = StaticRoleResolver(
            {
                app.state.config.hub_local_actor: RoleAssignment(
                    platform_roles=(Role.PLATFORM_ADMINISTRATOR,)
                )
            }
        )
    app.state.hub_repository = hub_repository or _repository_for(app.state.config)
    app.state.hub_service = HubService(
        app.state.hub_repository,
        registration_principals=(app.state.config.hub_registration_principals),
        job_runner=job_runner or _job_runner_for(app.state.config),
    )

    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    # Starlette's select_autoescape keys off the template extension, and every
    # console template ends in .j2 — so without this, user-shaped values (search
    # filters, estimator labels) would render unescaped. Force it on: templates
    # here are HTML only, and inline_code already returns Markup.
    templates.env.autoescape = True
    # The only markup content may use. Escapes first, so it cannot inject an element.
    templates.env.filters["icode"] = inline_code
    templates.env.filters["usd"] = usd
    app.mount(
        "/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static"
    )

    # Content is static for the life of the process, so parse and render it once.
    # Re-reading the YAML per request would put blocking disk I/O on the event loop of a
    # single-worker server for no benefit.
    app.state.tracks = _render_tracks(app.state.registry.tracks(), app.state.config)

    # Same rationale as tracks: the bundled list-price snapshot is static for the
    # life of the process, and a malformed snapshot should fail at import.
    app.state.pricing = load_snapshot()

    def tracks_for(request: Request) -> tuple[Track, ...]:
        return request.app.state.tracks

    def service_for(request: Request) -> HubService:
        return request.app.state.hub_service

    def common_context(
        request: Request,
        *,
        actor: AuthorizationContext | None,
        active_section: str,
    ) -> dict:
        service = service_for(request)
        return {
            "tracks": tracks_for(request),
            "session": request.app.state.config,
            "actor": actor_view(actor) if actor is not None else None,
            "active_section": active_section,
            "hub_capabilities": _hub_capabilities(
                request.app.state.config,
                service,
            ),
        }

    @app.middleware("http")
    async def _request_context(request: Request, call_next):
        request_id = request.scope.get("aai.request_id") or str(uuid4())
        request.state.request_id = request_id
        started = time.monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-Id"] = request_id
            return response
        finally:
            application_id = getattr(request.state, "application_id", None)
            if application_id is None:
                application_id = request.path_params.get("application_id")
            if not isinstance(application_id, str) or not _APPLICATION_ID.fullmatch(
                application_id
            ):
                application_id = None
            authentication_outcome = getattr(
                request.state,
                "authentication_outcome",
                "not_required",
            )
            authorization_decision = getattr(
                request.state,
                "authorization_decision",
                "not_evaluated",
            )
            if authentication_outcome == "authenticated":
                if status == 403:
                    authorization_decision = "denied"
                elif status < 400:
                    authorization_decision = "allowed"
            record = {
                "event": "hub_http_request",
                "request_id": request_id,
                "method": request.method,
                "route": _route_template(request.scope),
                "action": getattr(request.scope.get("route"), "name", "unmatched"),
                "status": status,
                "result": (
                    "success"
                    if status < 400
                    else ("client_error" if status < 500 else "server_error")
                ),
                "latency_ms": round((time.monotonic() - started) * 1000, 2),
                "actor": getattr(request.state, "actor_principal", None),
                "authentication_outcome": authentication_outcome,
                "authorization_decision": authorization_decision,
                "application_id": application_id,
                "target_environment": getattr(
                    request.state,
                    "target_environment",
                    None,
                ),
            }
            # JSON is intentional: no query, body, header collection, or exception
            # message can accidentally enter the structured application log.
            logger.info(json.dumps(record, sort_keys=True, separators=(",", ":")))

    @app.exception_handler(ProblemError)
    async def _problem(request: Request, exc: ProblemError) -> JSONResponse:
        return _problem_response(
            request,
            status=exc.status,
            title=exc.title,
            detail=exc.detail,
            problem_type=exc.problem_type,
            errors=exc.errors,
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _problem_response(
            request,
            status=422,
            title="Request validation failed",
            detail="One or more request fields are invalid.",
            problem_type="urn:aai:problem:request-validation",
            errors=_validation_errors(exc),
        )

    @app.exception_handler(HubReadinessBlockedError)
    async def _readiness_blocked(
        request: Request, exc: HubReadinessBlockedError
    ) -> JSONResponse:
        failed = [
            {
                "path": f"$.readiness.{result.rule_id}",
                "message": result.description,
                "code": result.status.value,
                "remediation": result.remediation,
            }
            for result in exc.snapshot.results
            if result.status in {ReadinessStatus.FAIL, ReadinessStatus.UNKNOWN}
            and result.severity.value == "BLOCKING"
        ]
        return _problem_response(
            request,
            status=409,
            title="Production readiness is blocked",
            detail="Blocking production-readiness evidence has not passed.",
            problem_type="urn:aai:problem:readiness-blocked",
            errors=failed,
        )

    @app.exception_handler(HubServiceError)
    async def _service_error(request: Request, exc: HubServiceError) -> JSONResponse:
        if isinstance(exc, HubCapabilityUnavailableError):
            status, title = 503, "Capability unavailable"
        elif isinstance(exc, HubPermissionDeniedError):
            status, title = 403, "Forbidden"
        elif isinstance(exc, HubQueryValidationError):
            status, title = 400, "Invalid query"
        elif isinstance(exc, HubExternalServiceError):
            status, title = 502, "Provider operation failed"
        else:
            status, title = 409, "Hub request could not be completed"
        return _problem_response(
            request,
            status=status,
            title=title,
            detail=str(exc),
            problem_type=f"urn:aai:problem:{title.lower().replace(' ', '-')}",
        )

    @app.exception_handler(HubRepositoryError)
    async def _repository_error(
        request: Request, exc: HubRepositoryError
    ) -> JSONResponse:
        if isinstance(exc, HubRepositoryUnavailableError):
            status, title = 503, "Operational store unavailable"
            detail = "The Hub operational store is not available."
        elif isinstance(exc, HubNotFoundError):
            status, title = 404, "Resource not found"
            detail = "The requested resource does not exist or is not visible."
        elif isinstance(exc, (HubAuthorizationError, FourEyesViolationError)):
            status, title = 403, "Forbidden"
            detail = str(exc)
        elif isinstance(exc, OptimisticConcurrencyError):
            status, title = 409, "Stale workflow state"
            detail = "The record changed; refresh it before trying again."
        elif isinstance(exc, HubConflictError):
            status, title = 409, "State conflict"
            detail = str(exc)
        else:
            status, title = 409, "Repository operation failed"
            detail = "The requested state transition could not be completed."
        return _problem_response(
            request,
            status=status,
            title=title,
            detail=detail,
            problem_type=f"urn:aai:problem:{title.lower().replace(' ', '-')}",
        )

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        title = {
            400: "Bad request",
            401: "Authentication required",
            403: "Forbidden",
            404: "Resource not found",
            503: "Capability unavailable",
        }.get(exc.status_code, "Request failed")
        detail = exc.detail if isinstance(exc.detail, str) else title
        return _problem_response(
            request,
            status=exc.status_code,
            title=title,
            detail=detail[:300],
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Shapes the client's response only. ContainExceptions does the logging, so this
        # deliberately logs nothing — the two together would emit the same line twice.
        del exc
        return _problem_response(
            request,
            status=500,
            title="Internal server error",
            detail="The request could not be completed.",
        )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/session")
    async def session(request: Request) -> dict:
        config: ConsoleConfig = request.app.state.config
        return {
            "hosted": config.hosted,
            "app_name": config.app_name,
            "version": __version__,
            "capability": "ai-platform-hub",
        }

    @app.get("/api/content")
    async def content(request: Request) -> dict:
        return {
            "tracks": [
                {
                    "id": track.id,
                    "title": track.title,
                    "subtitle": track.subtitle,
                    "glyph": track.glyph,
                    "steps": [
                        {"id": step.id, "title": step.title} for step in track.steps
                    ],
                }
                for track in tracks_for(request)
            ]
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        search: str = Query(default="", max_length=200),
        lifecycle: str | None = None,
        health: str | None = None,
        ownership: str = "visible",
        tags: str = Query(default="", max_length=2000),
        start_date: date | None = None,
        end_date: date | None = None,
        sort: str = "application",
        page: int = Query(default=1, ge=1, le=1_000_000),
        page_size: int = Query(default=25, ge=1, le=100),
    ) -> HTMLResponse:
        _validate_query_keys(
            request,
            frozenset(
                {
                    "search",
                    "lifecycle",
                    "health",
                    "ownership",
                    "tags",
                    "start_date",
                    "end_date",
                    "sort",
                    "page",
                    "page_size",
                }
            ),
        )
        actor = _actor_for(request)
        service = service_for(request)
        complete_day = datetime.now(UTC).date() - timedelta(days=1)
        selected_end = end_date or complete_day
        selected_start = start_date or (selected_end - timedelta(days=29))
        if selected_end > complete_day:
            raise ProblemError(
                400,
                "Invalid date range",
                "The end date must not be later than the last complete UTC day.",
            )
        if selected_start > selected_end:
            raise ProblemError(
                400,
                "Invalid date range",
                "The start date must not be after the end date.",
            )
        if (selected_end - selected_start).days > 366:
            raise ProblemError(
                400,
                "Invalid date range",
                "Portfolio date ranges may not exceed 367 days.",
            )
        applications: list[dict] = []
        total = 0
        pages = 1
        if service.available:
            try:
                query = PortfolioQuery(
                    search=search,
                    lifecycle=lifecycle,
                    health=health,
                    ownership=ownership,
                    tag_filters=parse_tag_filters(tags),
                    sort=sort,
                    page=page,
                    page_size=page_size,
                )
            except (ValidationError, HubQueryValidationError) as error:
                raise ProblemError(
                    400,
                    "Invalid portfolio filter",
                    "One or more portfolio filters are invalid.",
                ) from error
            portfolio = service.list_portfolio(actor, query)
            applications = [_portfolio_template_item(item) for item in portfolio.items]
            total = portfolio.total
            pages = portfolio.pages

        params = dict(request.query_params)
        current_sort = params.get("sort", "application")
        params["sort"] = (
            "-application" if current_sort == "application" else "application"
        )
        params.pop("page", None)
        sort_link = f"/?{urlencode(params)}" if params else "/?sort=-application"
        context = common_context(
            request,
            actor=actor,
            active_section="portfolio",
        )
        context.update(
            {
                "applications": applications,
                "summary": {
                    "application_count": total,
                    "blocking_count": sum(
                        item["blocking_issues"] for item in applications
                    ),
                    "application_cost": None,
                    "direct_cost": None,
                    "cost_freshness": "Unavailable",
                    "registry_freshness": (
                        "process-local preview"
                        if service.available and not request.app.state.config.hosted
                        else "unavailable"
                    ),
                },
                "filters": {
                    "search": search,
                    "lifecycle": lifecycle or "",
                    "health": health or "",
                    "ownership": ownership,
                    "tags": tags,
                    "start_date": selected_start.isoformat(),
                    "end_date": selected_end.isoformat(),
                },
                "sort_links": {"application": sort_link},
                "page": {
                    "number": page,
                    "pages": pages,
                    "total": total,
                    "previous": (
                        _pagination_url(request, page - 1) if page > 1 else None
                    ),
                    "next": (
                        _pagination_url(request, page + 1) if page < pages else None
                    ),
                },
            }
        )
        return templates.TemplateResponse(
            request,
            "portfolio.html.j2",
            context,
        )

    @app.get("/track/{track_id}", response_class=HTMLResponse)
    async def track_page(request: Request, track_id: str) -> HTMLResponse:
        tracks = tracks_for(request)
        match = next((t for t in tracks if t.id == track_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail="unknown track")
        context = common_context(
            request,
            actor=_actor_for(request),
            active_section="get_started",
        )
        context.update(
            {
                "active": match,
                "platform_state_heading": PLATFORM_STATE_HEADING,
            }
        )
        return templates.TemplateResponse(
            request,
            "index.html.j2",
            context,
        )

    @app.get("/optimization", response_class=HTMLResponse)
    async def optimization_page(request: Request) -> HTMLResponse:
        actor = _actor_for(request)
        context = common_context(
            request,
            actor=actor,
            active_section="optimization",
        )
        context["optimization_requirements"] = [
            {
                "title": "Governed telemetry facts",
                "status": "GATED",
                "tone": "warning",
                "detail": (
                    "Hourly and daily application-scoped usage facts with freshness "
                    "metadata are required."
                ),
            },
            {
                "title": "Versioned detectors",
                "status": "GATED",
                "tone": "warning",
                "detail": (
                    "Recommendations require reproducible evidence windows, "
                    "confidence, and qualified savings."
                ),
            },
            {
                "title": "Visibility-enforcing serving views",
                "status": "GATED",
                "tone": "warning",
                "detail": (
                    "List, detail, aggregate, and export paths must enforce the same "
                    "row-level visibility."
                ),
            },
        ]
        return templates.TemplateResponse(
            request,
            "optimization.html.j2",
            context,
        )

    @app.get("/estimator", response_class=HTMLResponse)
    async def estimator_page(request: Request) -> HTMLResponse:
        actor = _actor_for(request)
        context = common_context(
            request,
            actor=actor,
            active_section="estimator",
        )
        context.update(
            estimator_page_context(
                request.app.state.pricing,
                today=datetime.now(UTC).date(),
            )
        )
        return templates.TemplateResponse(request, "estimator.html.j2", context)

    def _priced_estimate(request: Request, payload: EstimateRequest):
        try:
            return estimate(payload, request.app.state.pricing)
        except EstimateError as error:
            # error.message names request paths and snapshot vocabulary only,
            # never user-entered text, so it is safe for a problem body.
            raise ProblemError(
                422,
                "Estimate cannot be priced",
                "The request references pricing the snapshot does not carry.",
                problem_type="urn:aai:problem:estimate-unpriceable",
                errors=[error.as_problem_item()],
            ) from error

    @app.post("/api/estimator/render", response_class=HTMLResponse)
    async def estimator_render(
        request: Request,
        payload: EstimateRequest,
    ) -> HTMLResponse:
        _actor_for(request)
        result = _priced_estimate(request, payload)
        return templates.TemplateResponse(
            request,
            "fragments/estimate.html.j2",
            {"estimate": result},
        )

    @app.post("/api/estimator/export.csv")
    async def estimator_export(
        request: Request,
        payload: EstimateRequest,
    ) -> PlainTextResponse:
        _actor_for(request)
        result = _priced_estimate(request, payload)
        return PlainTextResponse(
            estimate_csv(result),
            media_type="text/csv; charset=utf-8",
        )

    @app.get("/admin/actions", response_class=HTMLResponse)
    async def admin_actions_page(request: Request) -> HTMLResponse:
        actor = _actor_for(request)
        context = common_context(
            request,
            actor=actor,
            active_section="admin",
        )
        actions: list[dict] = []
        status_code = 200
        if not actor.has_platform_role(Role.PLATFORM_ADMINISTRATOR):
            status_code = 403
        elif service_for(request).available:
            for promotion in service_for(request).list_admin_actions(actor):
                application = service_for(request).get_application(
                    promotion.application_id,
                    actor=actor,
                )
                version = next(
                    item
                    for item in service_for(request).list_versions(
                        promotion.application_id,
                        actor=actor,
                    )
                    if item.version_id == promotion.application_version_id
                )
                age = datetime.now(UTC) - promotion.requested_at
                actions.append(
                    {
                        "application_id": application.application_id,
                        "application_name": application.name,
                        "promotion_request_id": promotion.promotion_request_id,
                        "request_type": "Promotion",
                        "source_environment": promotion.source_environment,
                        "target_environment": promotion.target_environment,
                        "git_sha": version.git_commit_sha,
                        "requested_by": promotion.requested_by,
                        "status": promotion.status.value,
                        "tone": (
                            "warning"
                            if promotion.status is PromotionStatus.PENDING_REVIEW
                            else "neutral"
                        ),
                        "age": f"{max(0, math.floor(age.total_seconds() / 3600))}h",
                        "row_version": promotion.row_version,
                    }
                )
        context["actions"] = actions
        return templates.TemplateResponse(
            request,
            "admin_actions.html.j2",
            context,
            status_code=status_code,
        )

    @app.get("/applications/{application_id}", response_class=HTMLResponse)
    async def application_detail_page(
        request: Request,
        application_id: str,
        environment: str | None = None,
        tab: str | None = None,
    ) -> HTMLResponse:
        _validate_query_keys(request, frozenset({"environment", "tab"}))
        del tab  # The browser activates the requested allowlisted tab.
        actor = _actor_for(request)
        service = service_for(request)
        application_id = _application_id(application_id)
        application = service.get_application(application_id, actor=actor)
        version = service.get_application_version(
            application_id,
            actor=actor,
            environment=environment,
        )
        versions = service.list_versions(application_id, actor=actor)
        deployments = tuple(
            version_item for version_item in versions if version_item.is_current
        )
        manifest = service.manifest_for_version(version)
        readiness = service.readiness_for_version(version)
        evaluations = service.list_evaluations(application_id, actor=actor)
        can_contribute = service.can_contribute(application, actor)
        evaluation_job_ready = (
            manifest.spec.resources.evaluation_job_id is not None
            and service.workflow_preview_enabled
        )
        promotion_job_ready = (
            manifest.spec.resources.promotion_job_id is not None
            and service.workflow_preview_enabled
        )
        promotion_target = next(
            (
                candidate
                for candidate in ("prod", "production")
                if candidate in manifest.spec.environments
                and candidate != version.environment
            ),
            None,
        )
        app_view = {
            "application_id": application.application_id,
            "name": application.name,
            "description": application.description,
            "owner": application.owner_principal,
            "support_group": application.support_group,
            "business_domain": application.business_domain,
            "cost_center": application.cost_center,
            "risk_tier": application.risk_tier,
            "lifecycle": application.lifecycle_state,
            "tags": tuple((tag.key, tag.value) for tag in application.tags),
            "environment": version.environment,
            "git_sha": version.git_commit_sha,
            "registered_at": version.registered_at.strftime("%Y-%m-%d %H:%M UTC"),
            "deployments": tuple(
                {
                    "environment": deployment.environment,
                    "git_sha": deployment.git_commit_sha,
                    "deployment_target": deployment.deployment_target,
                    "registered_at": deployment.registered_at.strftime(
                        "%Y-%m-%d %H:%M UTC"
                    ),
                    "selected": deployment.version_id == version.version_id,
                }
                for deployment in deployments
            ),
            "health": "UNKNOWN",
            "health_reason": "Governed health aggregates unavailable",
            "readiness": "READY" if readiness.ready else "BLOCKED",
            "readiness_profile": readiness.profile_id,
            "readiness_evaluated_at": readiness.evaluated_at.strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
            "resources": _resource_template_items(
                manifest,
                version.environment,
            ),
            "can_run_evaluation": can_contribute and evaluation_job_ready,
            "evaluation_disabled_reason": (
                "A governed dataset version, contributor role, resolved job ID, and "
                "approved Jobs runner are required."
                if can_contribute and evaluation_job_ready
                else (
                    "Contributor access and an approved resolved evaluation job "
                    "are required."
                )
            ),
            "can_request_promotion": (
                can_contribute
                and promotion_job_ready
                and readiness.ready
                and promotion_target is not None
            ),
            "promotion_target": promotion_target,
            "promotion_disabled_reason": (
                "Blocking production-readiness checks must pass."
                if not readiness.ready
                else (
                    "A distinct prod or production target must be declared."
                    if promotion_target is None
                    else (
                        "Contributor access and an approved promotion job are required."
                    )
                )
            ),
        }
        context = common_context(
            request,
            actor=actor,
            active_section="portfolio",
        )
        context.update(
            {
                "application": app_view,
                "readiness": _readiness_template(readiness),
                "evaluations": _evaluation_template_items(
                    evaluations,
                    versions,
                    request.app.state.config,
                ),
                "activity": _activity_template_items(
                    request.app.state.hub_repository,
                    application_id,
                ),
            }
        )
        return templates.TemplateResponse(
            request,
            "application_detail.html.j2",
            context,
        )

    # Deliberately `def`, not `async def`. run_checks makes blocking Databricks SDK
    # network calls and app.yaml starts a single-worker uvicorn, so on an async route
    # an unreachable workspace would stall the event loop for the whole SDK timeout
    # and take health, navigation and generation down with it. A sync route runs in
    # Starlette's worker threadpool instead.
    @app.post("/api/checks/run", response_class=HTMLResponse)
    def checks(request: Request) -> HTMLResponse:
        _actor_for(request)
        results: list[PlatformCheck] = run_checks(
            request.app.state.config, request.app.state.probe
        )
        # Raises if anyone ever tries to present these as the viewer's own access.
        assert_platform_state(results, PLATFORM_STATE_HEADING)
        return templates.TemplateResponse(
            request,
            "fragments/checks.html.j2",
            {"checks": results, "heading": PLATFORM_STATE_HEADING},
        )

    @app.post("/api/generate", response_class=HTMLResponse)
    async def generate(request: Request) -> HTMLResponse:
        payload = await request.json()
        try:
            blocks = bundle_init(
                GenerateRequest(
                    template=str(payload.get("template", "")),
                    project_name=str(payload.get("project_name") or "my-project"),
                ),
                request.app.state.config,
            )
        except GenerateError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except ConfigError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return templates.TemplateResponse(
            request, "fragments/blocks.html.j2", {"blocks": blocks}
        )

    @app.get("/api/palette")
    async def palette(request: Request, q: str = "") -> dict:
        needle = q.strip().lower()
        hits = []
        for track in tracks_for(request):
            for step in track.steps:
                haystack = f"{track.title} {step.title} {step.body}".lower()
                if not needle or needle in haystack:
                    hits.append(
                        {
                            "track": track.id,
                            "track_title": track.title,
                            "step": step.id,
                            "title": step.title,
                        }
                    )
        return {"results": hits[:20]}

    @app.get("/api/v1/capabilities")
    def hub_capabilities(request: Request) -> dict:
        return _hub_capabilities(
            request.app.state.config,
            service_for(request),
        )

    @app.get("/api/v1/manifest-schemas/ai-platform-v1")
    def manifest_schema() -> dict:
        return manifest_json_schema()

    @app.post(
        "/api/v1/registrations",
        response_model=RegistrationResponse,
        responses={201: {"model": RegistrationResponse}},
    )
    def register_application(
        request: Request,
        payload: RegistrationRequest,
    ) -> JSONResponse:
        actor = _actor_for(request)
        request.state.target_environment = payload.environment
        try:
            result = service_for(request).register(
                payload,
                actor=actor,
                actor_request_id=request.state.request_id,
            )
        except ValidationError as error:
            raise ProblemError(
                422,
                "Manifest validation failed",
                "The manifest does not conform to ai-platform/v1.",
                problem_type="urn:aai:problem:manifest-validation",
                errors=_validation_errors(error),
            ) from error
        except (TypeError, ValueError) as error:
            raise ProblemError(
                422,
                "Manifest validation failed",
                "The manifest cannot be normalized safely.",
                problem_type="urn:aai:problem:manifest-validation",
            ) from error
        body = {
            "created": result.created,
            "application": {
                "applicationId": result.application.application_id,
                "name": result.application.name,
                "owner": result.application.owner_principal,
                "supportGroup": result.application.support_group,
                "businessDomain": result.application.business_domain,
                "costCenter": result.application.cost_center,
                "riskTier": result.application.risk_tier,
                "lifecycle": result.application.lifecycle_state,
                "tags": {tag.key: tag.value for tag in result.application.tags},
            },
            "version": _version_projection(result.version),
        }
        request.state.application_id = result.application.application_id
        return JSONResponse(body, status_code=201 if result.created else 200)

    @app.get(
        "/api/v1/applications",
        response_model=ApplicationListResponse,
    )
    def list_applications(
        request: Request,
        search: str = Query(default="", max_length=200),
        lifecycle: str | None = None,
        health: str | None = None,
        ownership: str = "visible",
        tags: str = Query(default="", max_length=2000),
        start_date: date | None = None,
        end_date: date | None = None,
        sort: str = "application",
        page: int = Query(default=1, ge=1, le=1_000_000),
        page_size: int = Query(default=25, ge=1, le=100),
    ) -> dict:
        _validate_query_keys(
            request,
            frozenset(
                {
                    "search",
                    "lifecycle",
                    "health",
                    "ownership",
                    "tags",
                    "start_date",
                    "end_date",
                    "sort",
                    "page",
                    "page_size",
                }
            ),
        )
        actor = _actor_for(request)
        complete_day = datetime.now(UTC).date() - timedelta(days=1)
        selected_end = end_date or complete_day
        selected_start = start_date or (selected_end - timedelta(days=29))
        if (
            selected_end > complete_day
            or selected_start > selected_end
            or (selected_end - selected_start).days > 366
        ):
            raise ProblemError(
                400,
                "Invalid date range",
                "The date range must end by the last complete UTC day and "
                "must not exceed 367 days.",
            )
        try:
            query = PortfolioQuery(
                search=search,
                lifecycle=lifecycle,
                health=health,
                ownership=ownership,
                tag_filters=parse_tag_filters(tags),
                sort=sort,
                page=page,
                page_size=page_size,
            )
        except ValidationError as error:
            raise ProblemError(
                400,
                "Invalid portfolio filter",
                "One or more portfolio filters are invalid.",
                errors=_validation_errors(error),
            ) from error
        portfolio = service_for(request).list_portfolio(actor, query)
        return {
            "items": [_application_projection(item) for item in portfolio.items],
            "page": portfolio.page,
            "pageSize": portfolio.page_size,
            "total": portfolio.total,
            "pages": portfolio.pages,
            "period": {
                "startDate": selected_start.isoformat(),
                "endDate": selected_end.isoformat(),
                "completeDays": True,
            },
        }

    @app.get(
        "/api/v1/applications/{application_id}",
        response_model=ApplicationDetailResponse,
    )
    def get_application(
        request: Request,
        application_id: str,
        environment: str | None = None,
    ) -> dict:
        _validate_query_keys(request, frozenset({"environment"}))
        actor = _actor_for(request)
        application_id = _application_id(application_id)
        application = service_for(request).get_application(
            application_id,
            actor=actor,
        )
        version = service_for(request).get_application_version(
            application_id,
            actor=actor,
            environment=environment,
        )
        readiness = service_for(request).readiness_for_version(version)
        deployments = tuple(
            item
            for item in service_for(request).list_versions(
                application_id,
                actor=actor,
            )
            if item.is_current
        )
        return {
            "application": {
                "applicationId": application.application_id,
                "name": application.name,
                "description": application.description,
                "owner": application.owner_principal,
                "supportGroup": application.support_group,
                "businessDomain": application.business_domain,
                "costCenter": application.cost_center,
                "riskTier": application.risk_tier,
                "lifecycle": application.lifecycle_state,
                "tags": {tag.key: tag.value for tag in application.tags},
            },
            "currentVersion": _version_projection(version),
            "deployments": [
                _version_projection(deployment) for deployment in deployments
            ],
            "health": {
                "status": "UNKNOWN",
                "reason": "Governed health aggregates are not configured.",
                "evidenceAt": None,
            },
            "readiness": _readiness_projection(readiness),
            "costs": {
                "application": None,
                "directUser": None,
                "allocated": None,
                "unattributed": None,
                "freshness": None,
            },
        }

    @app.get(
        "/api/v1/applications/{application_id}/versions",
        response_model=VersionsResponse,
    )
    def list_application_versions(
        request: Request,
        application_id: str,
    ) -> dict:
        _validate_query_keys(request, frozenset())
        actor = _actor_for(request)
        versions = service_for(request).list_versions(
            _application_id(application_id),
            actor=actor,
        )
        return {"items": [_version_projection(version) for version in versions]}

    @app.post(
        "/api/v1/applications/{application_id}/evaluations",
        response_model=EvaluationResponse,
        status_code=202,
    )
    def run_evaluation(
        request: Request,
        application_id: str,
        payload: EvaluationRequest,
    ) -> dict:
        actor = _actor_for(request)
        request.state.target_environment = payload.environment
        evaluation = service_for(request).start_evaluation(
            _application_id(application_id),
            payload,
            actor=actor,
            actor_request_id=request.state.request_id,
        )
        return _evaluation_projection(evaluation)

    @app.get(
        "/api/v1/applications/{application_id}/evaluations",
        response_model=EvaluationListResponse,
    )
    def list_evaluations(
        request: Request,
        application_id: str,
        limit: int = Query(default=50, ge=1, le=100),
    ) -> dict:
        _validate_query_keys(request, frozenset({"limit"}))
        actor = _actor_for(request)
        evaluations = service_for(request).list_evaluations(
            _application_id(application_id),
            actor=actor,
        )
        return {
            "items": [
                _evaluation_projection(evaluation)
                for evaluation in tuple(reversed(evaluations))[:limit]
            ],
            "limit": limit,
        }

    @app.get(
        "/api/v1/evaluations/{evaluation_run_id}",
        response_model=EvaluationResponse,
    )
    def get_evaluation(
        request: Request,
        evaluation_run_id: str,
    ) -> dict:
        _validate_query_keys(request, frozenset())
        actor = _actor_for(request)
        evaluation = service_for(request).get_evaluation(
            evaluation_run_id,
            actor=actor,
        )
        return _evaluation_projection(evaluation)

    @app.post(
        "/api/v1/applications/{application_id}/promotion-requests",
        response_model=PromotionRequestRecord,
        status_code=201,
    )
    def request_promotion(
        request: Request,
        application_id: str,
        payload: PromotionRequest,
    ) -> PromotionRequestRecord:
        actor = _actor_for(request)
        request.state.target_environment = payload.target_environment
        promotion = service_for(request).request_promotion(
            _application_id(application_id),
            payload,
            actor=actor,
            actor_request_id=request.state.request_id,
        )
        return promotion

    @app.get(
        "/api/v1/applications/{application_id}/promotion-requests",
        response_model=PromotionListResponse,
    )
    def list_promotion_requests(
        request: Request,
        application_id: str,
        limit: int = Query(default=50, ge=1, le=100),
    ) -> dict:
        _validate_query_keys(request, frozenset({"limit"}))
        actor = _actor_for(request)
        promotions = service_for(request).list_promotions(
            _application_id(application_id),
            actor=actor,
        )
        return {
            "items": list(tuple(reversed(promotions))[:limit]),
            "limit": limit,
        }

    @app.get(
        "/api/v1/admin/actions",
        response_model=AdminActionListResponse,
    )
    def list_admin_actions(
        request: Request,
        status: str | None = None,
        request_type: str | None = None,
        application: str | None = None,
        business_domain: str | None = None,
        risk_tier: str | None = None,
        owner: str | None = None,
        age: int | None = Query(default=None, ge=0, le=87600),
        tags: str = Query(default="", max_length=2000),
    ) -> dict:
        _validate_query_keys(
            request,
            frozenset(
                {
                    "status",
                    "request_type",
                    "application",
                    "business_domain",
                    "risk_tier",
                    "owner",
                    "age",
                    "tags",
                }
            ),
        )
        actor = _actor_for(request)
        if request_type not in {None, "promotion"}:
            raise ProblemError(
                400,
                "Invalid action filter",
                "Only promotion requests are currently represented in the queue.",
            )
        accepted_tags = parse_tag_filters(tags)
        actions = []
        for promotion in service_for(request).list_admin_actions(actor):
            app_record = service_for(request).get_application(
                promotion.application_id,
                actor=actor,
            )
            app_tags = {tag.key: tag.value for tag in app_record.tags}
            elapsed_hours = max(
                0,
                math.floor(
                    (datetime.now(UTC) - promotion.requested_at).total_seconds() / 3600
                ),
            )
            if status is not None and promotion.status.value != status:
                continue
            if application is not None and promotion.application_id != application:
                continue
            if (
                business_domain is not None
                and app_record.business_domain != business_domain
            ):
                continue
            if risk_tier is not None and app_record.risk_tier != risk_tier:
                continue
            if owner is not None and app_record.owner_principal != owner:
                continue
            if age is not None and elapsed_hours < age:
                continue
            if any(
                app_tags.get(key) not in values for key, values in accepted_tags.items()
            ):
                continue
            actions.append(
                {
                    "request": promotion,
                    "application": {
                        "applicationId": app_record.application_id,
                        "name": app_record.name,
                        "owner": app_record.owner_principal,
                        "businessDomain": app_record.business_domain,
                        "riskTier": app_record.risk_tier,
                    },
                    "ageHours": elapsed_hours,
                }
            )
        return {"items": actions, "total": len(actions)}

    @app.post(
        "/api/v1/admin/promotion-requests/{promotion_request_id}/approve",
        response_model=PromotionRequestRecord,
    )
    def approve_promotion(
        request: Request,
        promotion_request_id: str,
        payload: PromotionReviewRequest,
    ) -> PromotionRequestRecord:
        actor = _actor_for(request)
        result = service_for(request).approve_promotion(
            promotion_request_id,
            payload,
            actor=actor,
            actor_request_id=request.state.request_id,
        )
        request.state.target_environment = result.target_environment
        return result

    @app.post(
        "/api/v1/admin/promotion-requests/{promotion_request_id}/reject",
        response_model=PromotionRequestRecord,
    )
    def reject_promotion(
        request: Request,
        promotion_request_id: str,
        payload: PromotionReviewRequest,
    ) -> PromotionRequestRecord:
        actor = _actor_for(request)
        result = service_for(request).reject_promotion(
            promotion_request_id,
            payload,
            actor=actor,
            actor_request_id=request.state.request_id,
        )
        request.state.target_environment = result.target_environment
        return result

    @app.post(
        "/api/v1/admin/promotion-requests/{promotion_request_id}/request-changes",
        response_model=PromotionRequestRecord,
    )
    def request_promotion_changes(
        request: Request,
        promotion_request_id: str,
        payload: PromotionReviewRequest,
    ) -> PromotionRequestRecord:
        actor = _actor_for(request)
        result = service_for(request).request_promotion_changes(
            promotion_request_id,
            payload,
            actor=actor,
            actor_request_id=request.state.request_id,
        )
        request.state.target_environment = result.target_environment
        return result

    def unavailable_observability(
        request: Request,
        *,
        capability: str,
    ) -> None:
        detail = _hub_capabilities(
            request.app.state.config,
            service_for(request),
        )[f"{capability}_detail"]
        raise ProblemError(
            503,
            f"{capability.title()} unavailable",
            detail,
            problem_type=f"urn:aai:problem:{capability}-unavailable",
        )

    @app.get(
        "/api/v1/applications/{application_id}/traces",
        status_code=503,
        responses={503: {"description": "Sanitized trace adapter is unavailable."}},
    )
    def list_trace_summaries(
        request: Request,
        application_id: str,
    ) -> None:
        service_for(request).get_application(
            _application_id(application_id),
            actor=_actor_for(request),
        )
        unavailable_observability(request, capability="traces")

    @app.get(
        "/api/v1/applications/{application_id}/costs",
        status_code=503,
        responses={503: {"description": "Governed cost views are unavailable."}},
    )
    def get_application_costs(
        request: Request,
        application_id: str,
    ) -> None:
        service_for(request).get_application(
            _application_id(application_id),
            actor=_actor_for(request),
        )
        unavailable_observability(request, capability="costs")

    @app.get(
        "/api/v1/optimization/recommendations",
        status_code=503,
        responses={503: {"description": "Optimization evidence is unavailable."}},
    )
    @app.get(
        "/api/v1/optimization/summary",
        status_code=503,
        responses={503: {"description": "Optimization evidence is unavailable."}},
    )
    def optimization_unavailable(request: Request) -> None:
        _actor_for(request)
        unavailable_observability(request, capability="optimization")

    @app.get(
        "/api/v1/optimization/recommendations/{recommendation_id}",
        status_code=503,
        responses={503: {"description": "Optimization evidence is unavailable."}},
    )
    def optimization_detail_unavailable(
        request: Request,
        recommendation_id: str,
    ) -> None:
        del recommendation_id
        _actor_for(request)
        unavailable_observability(request, capability="optimization")

    @app.post(
        "/api/v1/admin/optimization/recommendations/{recommendation_id}/assign",
        status_code=503,
        responses={403: {"description": "Platform administrator role is required."}},
    )
    @app.post(
        "/api/v1/admin/optimization/recommendations/{recommendation_id}/acknowledge",
        status_code=503,
        responses={403: {"description": "Platform administrator role is required."}},
    )
    @app.post(
        "/api/v1/admin/optimization/recommendations/{recommendation_id}/snooze",
        status_code=503,
        responses={403: {"description": "Platform administrator role is required."}},
    )
    @app.post(
        "/api/v1/admin/optimization/recommendations/{recommendation_id}/accept",
        status_code=503,
        responses={403: {"description": "Platform administrator role is required."}},
    )
    @app.post(
        "/api/v1/admin/optimization/recommendations/{recommendation_id}/dismiss",
        status_code=503,
        responses={403: {"description": "Platform administrator role is required."}},
    )
    @app.post(
        "/api/v1/admin/optimization/recommendations/{recommendation_id}/resolve",
        status_code=503,
        responses={403: {"description": "Platform administrator role is required."}},
    )
    def optimization_mutation_unavailable(
        request: Request,
        recommendation_id: str,
    ) -> None:
        del recommendation_id
        actor = _actor_for(request)
        if not actor.has_platform_role(Role.PLATFORM_ADMINISTRATOR):
            raise ProblemError(
                403,
                "Forbidden",
                "Platform administrator authorization is required.",
            )
        unavailable_observability(request, capability="optimization")

    _install_openapi_contract(app)

    # Wrapped, not returned bare: the containment layer must sit outside
    # Starlette's ServerErrorMiddleware, which always re-raises.
    return ContainExceptions(app)


# The Apps runtime starts `uvicorn aai_console.server:app`, which needs an instance
# rather than a factory. Building it at import keeps a config failure loud and early.
app = create_app()
