"""Versioned golden-path application manifest contracts.

``ai-app.yaml`` is registration input, not a replacement for the SDK's runtime
``aai-platform.yml``.  It describes application ownership, deployed environments,
resource bindings, evaluation policy, readiness policy, and service levels in a form
that the Hub can validate and hash deterministically.

Models in this module are strict and recursively immutable.  Normalization happens only
for platform-owned identifiers and tag keys; provider resource names and IDs are
validated but never silently rewritten.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, Self, cast
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from aai_core._sensitive import is_sensitive_name
from aai_core.tags import DataClassification

MANIFEST_API_VERSION = "ai-platform/v1"
MANIFEST_KIND = "AIApplication"
SUPPORTED_READINESS_PROFILES = (
    "development_v1",
    "medium_risk_production_v1",
)

_CONTROLLED_TAGS = frozenset(
    {"team", "domain", "cost_center", "environment", "application_id"}
)
_SNAKE_COMPONENT = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_RESOURCE_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,254}$")
_APP_NAME = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
_WORKSPACE_ID = re.compile(r"^[0-9]{1,32}$")
_DATASET_NAME = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_-]*"
    r"\.[A-Za-z_][A-Za-z0-9_-]*"
    r"\.[A-Za-z_][A-Za-z0-9_-]*$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
_CONTROL_CHAR = re.compile(r"[\x00-\x1f\x7f]")
_TAG_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,127}$")
_PERSONAL_TAG_KEY_PARTS = frozenset(
    {"email", "identity", "member", "person", "principal", "user"}
)
_SENSITIVE_VALUE = re.compile(
    r"(?i)(?:"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(?:password|passwd|client[_-]?secret|access[_-]?token|refresh[_-]?token"
    r"|api[_-]?key|authorization)\s*[:=]"
    r"|(?:bearer|basic)\s+[A-Za-z0-9+/_.=-]+"
    r"|(?<![A-Za-z0-9])(?:dapi|github_pat_|gh[oprsu]_|sk-)[A-Za-z0-9_-]{8,}"
    r"|(?:[?&]sig=|accountkey=|sharedaccesssignature=)"
    r")"
)
_JWT_VALUE = re.compile(r"^eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}$")


class _ManifestModel(BaseModel):
    """Strict, immutable default for every persisted manifest boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )

    @model_validator(mode="before")
    @classmethod
    def reject_secret_material(cls, value: Any) -> Any:
        _reject_secret_like_keys(value)
        _reject_secret_like_values(value)
        return value


def _key_words(key: str) -> tuple[str, ...]:
    separated = _CAMEL_BOUNDARY.sub("_", key)
    return tuple(part.lower() for part in _NON_ALNUM.split(separated) if part.strip())


def _is_secret_like_key(key: str) -> bool:
    """Identify credential-bearing field names without rejecting usage metrics.

    A metric such as ``token_count`` is not a credential, while ``access_token`` and
    ``vendorApiKey`` are.  ``authorization`` is also a legitimate manifest section;
    only an authorization *header* or credential-bearing child is forbidden.
    """

    words = _key_words(key)
    if not words:
        return False
    # ``authorization`` is a legitimate manifest policy section; credential-
    # bearing children such as authorizationHeader remain forbidden.
    if words == ("authorization",):
        return False
    return is_sensitive_name(key)


def _reject_secret_like_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            child_path = f"{path}.{key}"
            if _is_secret_like_key(key):
                raise ValueError(
                    f"{child_path} is secret-like; manifests contain identifiers "
                    "and policy only, never credentials or secret values"
                )
            _reject_secret_like_keys(item, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _reject_secret_like_keys(item, f"{path}[{index}]")


def _reject_secret_like_values(value: Any, path: str = "$") -> None:
    """Reject strong credential shapes without ever echoing the submitted value."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if isinstance(key, str) else path
            _reject_secret_like_values(item, child)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _reject_secret_like_values(item, f"{path}[{index}]")
    elif isinstance(value, str) and (
        _SENSITIVE_VALUE.search(value) or _JWT_VALUE.fullmatch(value.strip())
    ):
        raise ValueError(
            f"{path} contains credential material; manifests contain identifiers "
            "and policy only"
        )


def _clean_text(value: Any, *, field: str, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must not be empty")
    if len(cleaned) > maximum:
        raise ValueError(f"{field} must be at most {maximum} characters")
    if _CONTROL_CHAR.search(cleaned):
        raise ValueError(f"{field} must not contain control characters")
    return cleaned


def _snake_identifier(value: Any, *, field: str, maximum: int = 128) -> str:
    text = _clean_text(value, field=field, maximum=maximum * 2)
    text = _CAMEL_BOUNDARY.sub("_", text)
    normalized = _NON_ALNUM.sub("_", text).strip("_").lower()
    if len(normalized) > maximum or not _SNAKE_COMPONENT.fullmatch(normalized):
        raise ValueError(
            f"{field} must normalize to a lowercase snake-case identifier "
            f"starting with a letter and no longer than {maximum} characters"
        )
    return normalized


def _normalize_tags(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    if len(value) > 64:
        raise ValueError(f"{field} must contain at most 64 tags")
    normalized: dict[str, str] = {}
    original_keys: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _snake_identifier(raw_key, field=f"{field} key")
        if key == "owner" or set(_key_words(key)).intersection(_PERSONAL_TAG_KEY_PARTS):
            raise ValueError(f"{field}.{key} is a personal-identity tag")
        if key in normalized:
            raise ValueError(
                f"{field} keys {original_keys[key]!r} and {raw_key!r} both "
                f"normalize to {key!r}"
            )
        tag_value = _clean_text(raw_value, field=f"{field}.{key}", maximum=128)
        if key in {"application_id", "environment", "domain"}:
            tag_value = _snake_identifier(
                tag_value, field=f"{field}.{key}", maximum=128
            )
        if not _TAG_VALUE.fullmatch(tag_value):
            raise ValueError(
                f"{field}.{key} must be a non-personal low-cardinality tag value"
            )
        normalized[key] = tag_value
        original_keys[key] = str(raw_key)
    return normalized


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _validate_resource_reference(value: Any, *, field: str) -> str:
    reference = _clean_text(value, field=field, maximum=255)
    if not _RESOURCE_REFERENCE.fullmatch(reference):
        raise ValueError(
            f"{field} must contain only letters, numbers, '.', '_', ':', '/', or '-'"
        )
    return reference


def _as_tuple(value: Any, *, field: str) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise ValueError(f"{field} must be an array")


class ManifestMetadata(_ManifestModel):
    """Normalized application identity, ownership, support, and risk metadata."""

    id: str
    name: str
    description: str
    owner: str
    support_group: str = Field(alias="supportGroup")
    business_domain: str = Field(alias="businessDomain")
    cost_center: str = Field(alias="costCenter")
    risk_tier: Literal["low", "medium", "high", "critical"] = Field(alias="riskTier")
    tags: Mapping[str, str]

    @field_validator("id", mode="before")
    @classmethod
    def normalize_application_id(cls, value: Any) -> str:
        return _snake_identifier(value, field="metadata.id")

    @field_validator("business_domain", mode="before")
    @classmethod
    def normalize_business_domain(cls, value: Any) -> str:
        return _snake_identifier(value, field="metadata.businessDomain")

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value: Any) -> str:
        return _clean_text(value, field="metadata.name", maximum=200)

    @field_validator("description", mode="before")
    @classmethod
    def validate_description(cls, value: Any) -> str:
        return _clean_text(value, field="metadata.description", maximum=2000)

    @field_validator("owner", "support_group", "cost_center", mode="before")
    @classmethod
    def validate_principal_or_cost_field(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> str:
        return _clean_text(
            value,
            field=f"metadata.{info.field_name}",
            maximum=256,
        )

    @field_validator("risk_tier", mode="before")
    @classmethod
    def normalize_risk_tier(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return value.strip().lower()

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_metadata_tags(cls, value: Any) -> dict[str, str]:
        return _normalize_tags(value, field="metadata.tags")

    @field_validator("tags")
    @classmethod
    def freeze_tags(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return _freeze_mapping(value)

    @field_serializer("tags")
    def serialize_tags(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)


class RepositorySpec(_ManifestModel):
    """Credential-free HTTPS source repository reference."""

    url: str

    @field_validator("url", mode="before")
    @classmethod
    def validate_repository_url(cls, value: Any) -> str:
        url = _clean_text(value, field="spec.repository.url", maximum=2048)
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("spec.repository.url must be an absolute HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("spec.repository.url must not contain user information")
        if parsed.query or parsed.fragment:
            raise ValueError(
                "spec.repository.url must not contain a query string or fragment"
            )
        hostname = parsed.hostname.casefold()
        if (
            hostname == "example.com"
            or hostname.endswith(".example.com")
            or hostname.endswith(".example.invalid")
            or hostname.endswith(".example.test")
            or "replace-with" in parsed.path.casefold()
        ):
            raise ValueError(
                "spec.repository.url must identify the real application repository"
            )
        return url


class AuthorizationSpec(_ManifestModel):
    """Declared end-user, application, or hybrid authorization mode."""

    mode: Literal["user", "application", "hybrid"]

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return value.strip().lower()


class EnvironmentSpec(_ManifestModel):
    """One deployment environment and its externally provisioned resources."""

    workspace_id: str | None = Field(default=None, alias="workspaceId")
    databricks_app_name: str | None = Field(default=None, alias="databricksAppName")
    mlflow_experiment_id: str | None = Field(default=None, alias="mlflowExperimentId")
    ai_gateway_service: str | None = Field(default=None, alias="aiGatewayService")
    tags: Mapping[str, str]

    @field_validator(
        "workspace_id",
        "mlflow_experiment_id",
        "ai_gateway_service",
        mode="before",
    )
    @classmethod
    def clean_optional_resource_fields(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
        if value is None:
            return None
        return _clean_text(value, field=f"environment.{info.field_name}", maximum=255)

    @field_validator("workspace_id")
    @classmethod
    def validate_workspace_id(cls, value: str | None) -> str | None:
        if value is not None and not _WORKSPACE_ID.fullmatch(value):
            raise ValueError("workspaceId must be a numeric Databricks workspace ID")
        return value

    @field_validator("mlflow_experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_resource_reference(value, field="mlflowExperimentId")
        return value

    @field_validator("ai_gateway_service")
    @classmethod
    def validate_gateway_service(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_resource_reference(value, field="aiGatewayService")
        return value

    @field_validator("databricks_app_name", mode="before")
    @classmethod
    def validate_databricks_app_name(cls, value: Any) -> Any:
        if value is None:
            return None
        name = _clean_text(value, field="databricksAppName", maximum=63)
        if not _APP_NAME.fullmatch(name):
            raise ValueError(
                "databricksAppName must be lowercase alphanumeric/hyphenated "
                "and start with a letter"
            )
        return name

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_environment_tags(cls, value: Any) -> dict[str, str]:
        return _normalize_tags(value, field="environment.tags")

    @field_validator("tags")
    @classmethod
    def freeze_tags(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return _freeze_mapping(value)

    @field_serializer("tags")
    def serialize_tags(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)


class ResourceBindings(_ManifestModel):
    """Logical bindings to evaluation, promotion, SQL, search, and tool resources."""

    evaluation_job_id: str | None = Field(default=None, alias="evaluationJobId")
    evaluation_job_key: str | None = Field(default=None, alias="evaluationJobKey")
    promotion_job_id: str | None = Field(default=None, alias="promotionJobId")
    promotion_job_key: str | None = Field(default=None, alias="promotionJobKey")
    sql_warehouse_id: str | None = Field(default=None, alias="sqlWarehouseId")
    ai_search_indexes: tuple[str, ...] = Field(default=(), alias="aiSearchIndexes")
    unity_catalog_functions: tuple[str, ...] = Field(
        default=(), alias="unityCatalogFunctions"
    )
    mcp_services: tuple[str, ...] = Field(default=(), alias="mcpServices")

    @field_validator(
        "evaluation_job_id",
        "promotion_job_id",
        "sql_warehouse_id",
        mode="before",
    )
    @classmethod
    def validate_optional_resource_ids(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
        if value is None:
            return None
        return _validate_resource_reference(
            value,
            field=info.field_name or "resource",
        )

    @field_validator("evaluation_job_id", "promotion_job_id")
    @classmethod
    def validate_databricks_job_ids(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        if value is None:
            return None
        if (
            not value.isascii()
            or not value.isdigit()
            or value.startswith("0")
            or int(value) > 9_223_372_036_854_775_807
        ):
            raise ValueError(
                f"{info.field_name} must be a positive numeric Databricks job ID"
            )
        return value

    @field_validator(
        "evaluation_job_key",
        "promotion_job_key",
        mode="before",
    )
    @classmethod
    def normalize_optional_job_keys(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
        if value is None:
            return None
        return _snake_identifier(value, field=info.field_name or "job_key")

    @field_validator(
        "ai_search_indexes",
        "unity_catalog_functions",
        "mcp_services",
        mode="before",
    )
    @classmethod
    def normalize_reference_arrays(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> tuple[str, ...]:
        field_name = info.field_name or "resource_references"
        values = _as_tuple(value, field=field_name)
        normalized = tuple(
            _validate_resource_reference(item, field=field_name) for item in values
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{info.field_name} must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_job_bindings(self) -> Self:
        evaluation_bindings = (
            self.evaluation_job_id,
            self.evaluation_job_key,
        )
        if sum(value is not None for value in evaluation_bindings) != 1:
            raise ValueError(
                "resources requires exactly one of evaluationJobId or "
                "evaluationJobKey"
            )
        if self.promotion_job_id is not None and self.promotion_job_key is not None:
            raise ValueError(
                "resources may define promotionJobId or promotionJobKey, not both"
            )
        return self


class EvaluationSpec(_ManifestModel):
    """Evaluation dataset, freshness, case-count, and threshold policy."""

    profile: str
    dataset: str
    minimum_cases: int = Field(alias="minimumCases", ge=1, le=1_000_000)
    maximum_age_hours: int = Field(alias="maximumAgeHours", ge=1, le=8_760)
    thresholds: Mapping[str, float]

    @field_validator("profile", mode="before")
    @classmethod
    def normalize_profile(cls, value: Any) -> str:
        return _snake_identifier(value, field="evaluation.profile")

    @field_validator("dataset", mode="before")
    @classmethod
    def validate_dataset(cls, value: Any) -> str:
        dataset = _clean_text(value, field="evaluation.dataset", maximum=384)
        if not _DATASET_NAME.fullmatch(dataset):
            raise ValueError(
                "evaluation.dataset must be a three-part Unity Catalog name"
            )
        return dataset

    @field_validator("thresholds", mode="before")
    @classmethod
    def normalize_thresholds(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise ValueError("evaluation.thresholds must be an object")
        if not value:
            raise ValueError("evaluation.thresholds must contain at least one rule")
        normalized: dict[str, float] = {}
        originals: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = _snake_identifier(raw_key, field="evaluation threshold")
            if key in normalized:
                raise ValueError(
                    f"evaluation threshold keys {originals[key]!r} and "
                    f"{raw_key!r} both normalize to {key!r}"
                )
            if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
                raise ValueError(f"evaluation.thresholds.{key} must be numeric")
            number = float(raw_value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(
                    f"evaluation.thresholds.{key} must be finite and non-negative"
                )
            if (key.endswith("_rate") or key in {"groundedness", "relevance"}) and (
                number > 1
            ):
                raise ValueError(f"evaluation.thresholds.{key} must be between 0 and 1")
            if key.endswith("_latency_ms") and number <= 0:
                raise ValueError(
                    f"evaluation.thresholds.{key} must be greater than zero"
                )
            normalized[key] = number
            originals[key] = str(raw_key)
        return normalized

    @field_validator("thresholds")
    @classmethod
    def freeze_thresholds(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return _freeze_mapping(value)

    @field_serializer("thresholds")
    def serialize_thresholds(self, value: Mapping[str, float]) -> dict[str, float]:
        return dict(value)


class ReadinessSpec(_ManifestModel):
    """Named platform readiness profile applied to the application."""

    profile: Literal["development_v1", "medium_risk_production_v1"]

    @field_validator("profile", mode="before")
    @classmethod
    def normalize_profile(cls, value: Any) -> str:
        return _snake_identifier(value, field="readiness.profile")


class CostControlsSpec(_ManifestModel):
    """Reference to the externally enforced platform budget policy."""

    budget_policy: str = Field(alias="budgetPolicy")

    @field_validator("budget_policy", mode="before")
    @classmethod
    def normalize_budget_policy(cls, value: Any) -> str:
        return _snake_identifier(value, field="costControls.budgetPolicy")


class ServiceLevels(_ManifestModel):
    """Declared error-rate and latency objectives for the application."""

    maximum_error_rate: float = Field(alias="maximumErrorRate", ge=0.0, le=1.0)
    p95_latency_ms: int = Field(alias="p95LatencyMs", gt=0, le=86_400_000)


class ApplicationSpec(_ManifestModel):
    """Complete repository, resource, evaluation, readiness, and SLO contract."""

    repository: RepositorySpec
    authorization: AuthorizationSpec
    environments: Mapping[str, EnvironmentSpec]
    resources: ResourceBindings
    evaluation: EvaluationSpec
    readiness: ReadinessSpec
    # Optional in ai-platform/v1 for canonical/hash compatibility with manifests
    # registered before budget-policy references were introduced. New generated
    # projects require it through their local cross-file validator.
    cost_controls: CostControlsSpec | None = Field(default=None, alias="costControls")
    service_levels: ServiceLevels = Field(alias="serviceLevels")

    @field_validator("environments", mode="before")
    @classmethod
    def normalize_environments(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("spec.environments must be an object")
        if not value:
            raise ValueError("spec.environments must contain at least one environment")
        if len(value) > 32:
            raise ValueError("spec.environments may contain at most 32 environments")
        normalized: dict[str, Any] = {}
        originals: dict[str, str] = {}
        for raw_name, environment in value.items():
            name = _snake_identifier(raw_name, field="environment name", maximum=63)
            if name in normalized:
                raise ValueError(
                    f"environment names {originals[name]!r} and {raw_name!r} both "
                    f"normalize to {name!r}"
                )
            normalized[name] = environment
            originals[name] = str(raw_name)
        return normalized

    @field_validator("environments")
    @classmethod
    def freeze_environments(
        cls, value: Mapping[str, EnvironmentSpec]
    ) -> Mapping[str, EnvironmentSpec]:
        return _freeze_mapping(value)

    @field_serializer("environments")
    def serialize_environments(
        self, value: Mapping[str, EnvironmentSpec]
    ) -> dict[str, EnvironmentSpec]:
        return dict(value)


class AIApplicationManifest(_ManifestModel):
    """Normalized ``ai-platform/v1`` registration contract."""

    api_version: Literal["ai-platform/v1"] = Field(alias="apiVersion")
    kind: Literal["AIApplication"]
    metadata: ManifestMetadata
    spec: ApplicationSpec

    @model_validator(mode="after")
    def validate_controlled_tags(self) -> Self:
        base_tags = dict(self.metadata.tags)
        uses_historical_candidate = False
        for environment_name, environment in self.spec.environments.items():
            effective = {**base_tags, **dict(environment.tags)}
            missing = sorted(_CONTROLLED_TAGS.difference(effective))
            if missing:
                raise ValueError(
                    f"environment {environment_name!r} is missing controlled tags: "
                    + ", ".join(missing)
                )
            if effective["application_id"] != self.metadata.id:
                raise ValueError(
                    f"environment {environment_name!r} application_id tag must "
                    f"equal normalized metadata.id {self.metadata.id!r}"
                )
            if effective["environment"] != environment_name:
                raise ValueError(
                    f"environment {environment_name!r} environment tag must equal "
                    "its normalized environment name"
                )
            if effective["domain"] != self.metadata.business_domain:
                raise ValueError(
                    f"environment {environment_name!r} domain tag must equal "
                    "metadata.businessDomain"
                )
            if effective["cost_center"] != self.metadata.cost_center:
                raise ValueError(
                    f"environment {environment_name!r} cost_center tag must equal "
                    "metadata.costCenter"
                )
            classification = effective.get("data_classification")
            if classification is not None and classification not in {
                item.value for item in DataClassification
            }:
                raise ValueError(
                    f"environment {environment_name!r} data_classification tag "
                    "must be public, internal, confidential, or restricted"
                )
            lifecycle = effective.get("lifecycle", "experimental")
            if lifecycle not in {
                "experimental",
                "candidate",
                "validation",
                "production",
                "retired",
            }:
                raise ValueError(
                    f"environment {environment_name!r} lifecycle tag must be "
                    "experimental, candidate (historical), validation, "
                    "production, or retired"
                )
            uses_historical_candidate = (
                uses_historical_candidate or lifecycle == "candidate"
            )
        if uses_historical_candidate:
            warnings.warn(
                "the candidate lifecycle tag is accepted only for historical "
                "ai-platform/v1 manifests; write validation for new evidence",
                DeprecationWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def prevent_readiness_profile_downgrades(self) -> Self:
        if self.spec.readiness.profile != "development_v1":
            return self
        production_declared = bool(
            {"prod", "production"}.intersection(self.spec.environments)
        )
        if self.metadata.risk_tier in {"high", "critical"} or production_declared:
            raise ValueError(
                "development_v1 is not permitted for high-risk or "
                "production-declared applications"
            )
        return self


def load_manifest(
    document: Mapping[str, Any] | AIApplicationManifest,
) -> AIApplicationManifest:
    """Validate and normalize a manifest document."""

    if isinstance(document, AIApplicationManifest):
        return document
    if not isinstance(document, Mapping):
        raise TypeError("manifest document must be an object")
    return AIApplicationManifest.model_validate(document)


def canonical_manifest_json(
    document: Mapping[str, Any] | AIApplicationManifest,
) -> str:
    """Return stable normalized JSON for hashing and immutable registration."""

    manifest = load_manifest(document)
    normalized = manifest.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class ManifestEnvelope(_ManifestModel):
    """A validated manifest plus its canonical representation and SHA-256 hash."""

    manifest: AIApplicationManifest
    canonical_json: str = Field(alias="canonicalJson")
    manifest_hash: str = Field(alias="manifestHash")
    hash_algorithm: Literal["sha256"] = Field(default="sha256", alias="hashAlgorithm")

    @field_validator("manifest_hash")
    @classmethod
    def validate_hash_shape(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("manifestHash must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def verify_envelope(self) -> Self:
        canonical = canonical_manifest_json(self.manifest)
        if self.canonical_json != canonical:
            raise ValueError(
                "canonicalJson does not match the normalized manifest document"
            )
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if self.manifest_hash != expected:
            raise ValueError("manifestHash does not match canonicalJson")
        return self


def build_manifest_envelope(
    document: Mapping[str, Any] | AIApplicationManifest,
) -> ManifestEnvelope:
    """Build canonical manifest registration evidence with its SHA-256 digest."""

    manifest = load_manifest(document)
    canonical = canonical_manifest_json(manifest)
    return ManifestEnvelope(
        manifest=manifest,
        canonicalJson=canonical,
        manifestHash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def manifest_json_schema() -> dict[str, Any]:
    """Publishable JSON Schema for the ``ai-platform/v1`` manifest."""

    schema = AIApplicationManifest.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:aai:schema:ai-platform:v1"
    return cast(dict[str, Any], json.loads(json.dumps(schema, allow_nan=False)))


__all__ = [
    "AIApplicationManifest",
    "ApplicationSpec",
    "AuthorizationSpec",
    "CostControlsSpec",
    "EnvironmentSpec",
    "EvaluationSpec",
    "MANIFEST_API_VERSION",
    "MANIFEST_KIND",
    "ManifestEnvelope",
    "ManifestMetadata",
    "ReadinessSpec",
    "RepositorySpec",
    "ResourceBindings",
    "ServiceLevels",
    "SUPPORTED_READINESS_PROFILES",
    "build_manifest_envelope",
    "canonical_manifest_json",
    "load_manifest",
    "manifest_json_schema",
]
