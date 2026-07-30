"""Explicit, secret-minimizing response contracts for the Hub HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import NonEmptyStr, PromotionRequestRecord


class APIResponseModel(BaseModel):
    """Strict response defaults so OpenAPI and runtime projections stay aligned."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )


class VersionResponse(APIResponseModel):
    version_id: NonEmptyStr = Field(alias="versionId")
    application_id: NonEmptyStr = Field(alias="applicationId")
    environment: NonEmptyStr
    git_repository: NonEmptyStr = Field(alias="gitRepository")
    git_commit_sha: NonEmptyStr = Field(alias="gitCommitSha")
    manifest_version: NonEmptyStr = Field(alias="manifestVersion")
    manifest_hash: NonEmptyStr = Field(alias="manifestHash")
    registered_by: NonEmptyStr = Field(alias="registeredBy")
    registered_at: NonEmptyStr = Field(alias="registeredAt")
    deployment_target: NonEmptyStr = Field(alias="deploymentTarget")
    current: bool


class ApplicationMetadataResponse(APIResponseModel):
    application_id: NonEmptyStr = Field(alias="applicationId")
    name: NonEmptyStr
    description: str = ""
    owner: NonEmptyStr
    support_group: NonEmptyStr = Field(alias="supportGroup")
    business_domain: NonEmptyStr = Field(alias="businessDomain")
    cost_center: NonEmptyStr = Field(alias="costCenter")
    risk_tier: NonEmptyStr = Field(alias="riskTier")
    lifecycle: NonEmptyStr
    tags: dict[str, str]


class RegistrationResponse(APIResponseModel):
    created: bool
    application: ApplicationMetadataResponse
    version: VersionResponse


class HealthResponse(APIResponseModel):
    status: NonEmptyStr
    reason: NonEmptyStr
    evidence_at: str | None = Field(alias="evidenceAt")


class ReadinessSummaryResponse(APIResponseModel):
    ready: bool
    blocking_issues: int = Field(alias="blockingIssues", ge=0)
    profile_id: NonEmptyStr = Field(alias="profileId")
    evaluated_at: NonEmptyStr = Field(alias="evaluatedAt")


class ReadinessRuleResponse(APIResponseModel):
    rule_id: NonEmptyStr = Field(alias="ruleId")
    rule_version: NonEmptyStr = Field(alias="ruleVersion")
    description: NonEmptyStr
    severity: NonEmptyStr
    status: NonEmptyStr
    evidence: list[str]
    evaluated_at: NonEmptyStr = Field(alias="evaluatedAt")
    remediation: str | None


class ReadinessResponse(APIResponseModel):
    application_id: NonEmptyStr = Field(alias="applicationId")
    environment: NonEmptyStr
    application_version_id: NonEmptyStr = Field(alias="applicationVersionId")
    profile_id: NonEmptyStr = Field(alias="profileId")
    profile_version: NonEmptyStr = Field(alias="profileVersion")
    evaluated_at: NonEmptyStr = Field(alias="evaluatedAt")
    ready: bool
    results: list[ReadinessRuleResponse]


class ApplicationListItemResponse(ApplicationMetadataResponse):
    environment: NonEmptyStr
    deployments: list[VersionResponse]
    health: HealthResponse
    readiness: ReadinessSummaryResponse
    current_version: VersionResponse = Field(alias="currentVersion")
    last_successful_evaluation: dict[str, Any] | None = Field(
        alias="lastSuccessfulEvaluation"
    )
    application_cost: dict[str, Any] | None = Field(alias="applicationCost")
    direct_user_cost: dict[str, Any] | None = Field(alias="directUserCost")
    request_volume: int | None = Field(alias="requestVolume", ge=0)
    p95_latency_ms: float | None = Field(alias="p95LatencyMs", ge=0)
    error_rate: float | None = Field(alias="errorRate", ge=0, le=1)
    outstanding_actions: int | None = Field(alias="outstandingActions", ge=0)


class PeriodResponse(APIResponseModel):
    start_date: NonEmptyStr = Field(alias="startDate")
    end_date: NonEmptyStr = Field(alias="endDate")
    complete_days: bool = Field(alias="completeDays")


class ApplicationListResponse(APIResponseModel):
    items: list[ApplicationListItemResponse]
    page: int = Field(ge=1)
    page_size: int = Field(alias="pageSize", ge=1, le=100)
    total: int = Field(ge=0)
    pages: int = Field(ge=1)
    period: PeriodResponse


class CostViewsResponse(APIResponseModel):
    application: None = None
    direct_user: None = Field(default=None, alias="directUser")
    allocated: None = None
    unattributed: None = None
    freshness: None = None


class ApplicationDetailResponse(APIResponseModel):
    application: ApplicationMetadataResponse
    current_version: VersionResponse = Field(alias="currentVersion")
    deployments: list[VersionResponse]
    health: HealthResponse
    readiness: ReadinessResponse
    costs: CostViewsResponse


class VersionsResponse(APIResponseModel):
    items: list[VersionResponse]


class EvaluationResponse(APIResponseModel):
    """Sanitized evaluation projection; raw provider payloads never cross the API."""

    evaluation_run_id: NonEmptyStr = Field(alias="evaluationRunId")
    application_id: NonEmptyStr = Field(alias="applicationId")
    environment: NonEmptyStr
    application_version_id: NonEmptyStr = Field(alias="applicationVersionId")
    evaluation_profile: NonEmptyStr = Field(alias="evaluationProfile")
    dataset_name: NonEmptyStr = Field(alias="datasetName")
    dataset_version: NonEmptyStr = Field(alias="datasetVersion")
    job_id: NonEmptyStr = Field(alias="jobId")
    job_run_id: NonEmptyStr | None = Field(default=None, alias="jobRunId")
    mlflow_run_id: NonEmptyStr | None = Field(default=None, alias="mlflowRunId")
    requested_by: NonEmptyStr = Field(alias="requestedBy")
    status: NonEmptyStr
    requested_at: NonEmptyStr = Field(alias="requestedAt")
    started_at: NonEmptyStr | None = Field(default=None, alias="startedAt")
    completed_at: NonEmptyStr | None = Field(default=None, alias="completedAt")
    metrics: dict[str, bool | int | float]
    failure_category: NonEmptyStr | None = Field(default=None, alias="failureCategory")


class EvaluationListResponse(APIResponseModel):
    items: list[EvaluationResponse]
    limit: int = Field(ge=1, le=100)


class PromotionListResponse(APIResponseModel):
    items: list[PromotionRequestRecord]
    limit: int = Field(ge=1, le=100)


class AdminApplicationResponse(APIResponseModel):
    application_id: NonEmptyStr = Field(alias="applicationId")
    name: NonEmptyStr
    owner: NonEmptyStr
    business_domain: NonEmptyStr = Field(alias="businessDomain")
    risk_tier: NonEmptyStr = Field(alias="riskTier")


class AdminActionResponse(APIResponseModel):
    request: PromotionRequestRecord
    application: AdminApplicationResponse
    age_hours: int = Field(alias="ageHours", ge=0)


class AdminActionListResponse(APIResponseModel):
    items: list[AdminActionResponse]
    total: int = Field(ge=0)


__all__ = [
    "AdminActionListResponse",
    "ApplicationDetailResponse",
    "ApplicationListResponse",
    "EvaluationResponse",
    "EvaluationListResponse",
    "PromotionListResponse",
    "RegistrationResponse",
    "VersionResponse",
    "VersionsResponse",
]
