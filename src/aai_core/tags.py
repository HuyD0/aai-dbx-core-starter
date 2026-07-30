"""Canonical ownership, lifecycle, and cost-attribution metadata."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, ValidationInfo, field_validator

from aai_core.contracts import ContractModel

_AZURE_VALUE = re.compile(r"[^A-Za-z0-9 +\-=._:/@]")
_PLACEHOLDERS = {"", "unset", "unknown", "todo", "changeme"}
_AI_GATEWAY_TAG_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,127}$")
_AI_GATEWAY_APPLICATION_ID = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_APPLICATION_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
_EMAIL_SHAPED = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_CREDENTIAL_PREFIX = re.compile(
    r"^(?:"
    r"dapi[a-z0-9]{16,}|"
    r"github_pat_[a-z0-9_]{16,}|"
    r"gh[oprsu]_[a-z0-9_]{16,}|"
    r"sk-[a-z0-9_-]{16,}|"
    r"eyj[a-z0-9_-]+\.[a-z0-9_-]+\.[a-z0-9_-]+"
    r")$",
    re.IGNORECASE,
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?:^|[^a-z0-9])"
    r"(?:api[_-]?key|authorization|bearer|client[_-]?secret|password|secret|token)"
    r"\s*[:=]",
    re.IGNORECASE,
)
_BEARER_CREDENTIAL = re.compile(r"^bearer\s+\S+", re.IGNORECASE)

DATABRICKS_AI_GATEWAY_REQUEST_TAGS_HEADER = "Databricks-Ai-Gateway-Request-Tags"


class LifecycleStage(StrEnum):
    """Supported maturity states for an AI application resource."""

    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    RETIRED = "retired"


class ResourceContext(ContractModel):
    """Validated ownership, lifecycle, and cost-attribution contract."""

    application: str = Field(min_length=1)
    project: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    team: str = Field(min_length=1)
    owner_group: str = Field(min_length=1)
    cost_center: str = Field(min_length=1)
    data_classification: str = Field(min_length=1)
    lifecycle: LifecycleStage
    repository: str = Field(min_length=1)
    release: str = Field(min_length=1)
    tag_schema_version: Literal["1"] = "1"

    @field_validator("lifecycle", mode="before")
    @classmethod
    def parse_lifecycle(cls, value: object) -> LifecycleStage:
        if isinstance(value, LifecycleStage):
            return value
        if not isinstance(value, str):
            raise ValueError("lifecycle must be a string")
        try:
            return LifecycleStage(value.strip().lower())
        except ValueError as error:
            choices = ", ".join(stage.value for stage in LifecycleStage)
            raise ValueError(f"lifecycle must be one of: {choices}") from error

    def validate(self, *, strict: bool = False) -> None:
        values = self.model_dump(mode="python")
        empty = [name for name, value in values.items() if not str(value).strip()]
        if empty:
            fields = ", ".join(empty)
            raise ValueError(f"Resource context contains empty fields: {fields}")

        if strict:
            placeholders = [
                name
                for name, value in values.items()
                if str(value).strip().lower() in _PLACEHOLDERS
            ]
            if placeholders:
                raise ValueError(
                    "Production resource context contains placeholders: "
                    + ", ".join(placeholders)
                )
            if "@" in self.owner_group:
                raise ValueError(
                    "owner_group must be a non-personal group identifier, not an email"
                )

    def as_dict(self) -> dict[str, str]:
        return {
            key: str(value) for key, value in self.model_dump(mode="python").items()
        }

    def for_mlflow(self) -> dict[str, str]:
        return {f"aai.{key}": value for key, value in self.as_dict().items()}

    def for_trace(self) -> dict[str, str]:
        return self.for_mlflow()

    def for_databricks(self) -> dict[str, str]:
        return self.as_dict()

    def for_azure(self) -> dict[str, str]:
        return {
            key[:512]: _AZURE_VALUE.sub("_", value)[:256]
            for key, value in self.as_dict().items()
        }

    def merged(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        tags = self.as_dict()
        if extra:
            controlled = set(tags)
            conflicts = controlled.intersection(extra)
            if conflicts:
                raise ValueError(
                    "Application code cannot override controlled tags: "
                    + ", ".join(sorted(conflicts))
                )
            tags.update({str(key): str(value) for key, value in extra.items()})
        return tags


class DatabricksAIRequestTags(ContractModel):
    """Approved, non-personal attribution sent to Unity AI Gateway.

    This deliberately excludes an end-user identifier. Shared application
    traffic remains application-attributed unless a separate, reviewed opaque
    identity contract is introduced.
    """

    # Invalid values may themselves be credentials or personal identifiers.
    # Never echo the rejected input through a Pydantic validation message.
    model_config = ConfigDict(hide_input_in_errors=True)

    application_id: str
    environment: str
    team: str
    cost_center: str
    application_version: str

    @field_validator("application_id", mode="before")
    @classmethod
    def normalize_application_id(cls, value: object) -> object:
        """Map a safe display name to the manifest's canonical ID vocabulary.

        ``ResourceContext.application`` predates request tags and intentionally
        permits human-readable names. The billing join key must nevertheless be
        identical for ``Claims Agent``, ``claims-agent``, and ``claims_agent``.
        Check sensitive patterns before normalization so punctuation cannot disguise
        a credential.
        """

        if not isinstance(value, str):
            return value
        lowered = value.lower()
        if (
            "@" in value
            or "://" in value
            or "-----begin" in lowered
            or _CREDENTIAL_PREFIX.fullmatch(value)
            or _CREDENTIAL_ASSIGNMENT.search(value)
            or _BEARER_CREDENTIAL.match(value)
        ):
            raise ValueError(
                "application_id must not contain a personal identifier or "
                "credential material"
            )
        separated = _CAMEL_BOUNDARY.sub("_", value.strip())
        normalized = _APPLICATION_NON_ALNUM.sub("_", separated).strip("_").lower()
        if not _AI_GATEWAY_APPLICATION_ID.fullmatch(normalized):
            raise ValueError(
                "application_id must normalize to a lowercase platform identifier"
            )
        return normalized

    @field_validator("*")
    @classmethod
    def validate_tag_value(cls, value: str, info: ValidationInfo) -> str:
        """Reject values that are unsafe for persisted billing telemetry."""

        if value != value.strip():
            raise ValueError(
                f"{info.field_name} must not contain surrounding whitespace"
            )
        lowered = value.lower()
        if lowered in _PLACEHOLDERS:
            raise ValueError(f"{info.field_name} must not be a placeholder")
        if "@" in value or _EMAIL_SHAPED.fullmatch(value):
            raise ValueError(
                f"{info.field_name} must not contain an email or personal identifier"
            )
        if (
            "://" in value
            or "-----begin" in lowered
            or _CREDENTIAL_PREFIX.fullmatch(value)
            or _CREDENTIAL_ASSIGNMENT.search(value)
            or _BEARER_CREDENTIAL.match(value)
        ):
            raise ValueError(f"{info.field_name} must not contain credential material")
        if not _AI_GATEWAY_TAG_VALUE.fullmatch(value):
            raise ValueError(
                f"{info.field_name} must be 1-128 ASCII letters, numbers, or "
                "'.', '_', ':', '+', '/', '-' characters and start with a "
                "letter or number"
            )
        return value

    @classmethod
    def from_resource_context(cls, context: ResourceContext) -> DatabricksAIRequestTags:
        """Project only the approved platform-owned attribution fields."""

        return cls(
            application_id=context.application,
            environment=context.environment,
            team=context.team,
            cost_center=context.cost_center,
            application_version=context.release,
        )

    def header_value(self) -> str:
        """Return canonical compact JSON suitable for the gateway header."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


def databricks_ai_gateway_request_headers(
    context: ResourceContext,
) -> dict[str, str]:
    """Build the immutable client-level Unity AI Gateway header."""

    tags = DatabricksAIRequestTags.from_resource_context(context)
    return {DATABRICKS_AI_GATEWAY_REQUEST_TAGS_HEADER: tags.header_value()}
