"""Canonical ownership, lifecycle, and cost-attribution metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from aai_core.contracts import ContractModel

_AZURE_VALUE = re.compile(r"[^A-Za-z0-9 +\-=._:/@]")
_PLACEHOLDERS = {"", "unset", "unknown", "todo", "changeme"}


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
