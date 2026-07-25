"""Canonical ownership, lifecycle, and cost-attribution metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

_AZURE_VALUE = re.compile(r"[^A-Za-z0-9 +\-=._:/@]")
_PLACEHOLDERS = {"", "unset", "unknown", "todo", "changeme"}


@dataclass(frozen=True)
class ResourceContext:
    application: str
    project: str
    environment: str
    team: str
    owner_group: str
    cost_center: str
    data_classification: str
    lifecycle: str
    repository: str
    release: str
    tag_schema_version: str = "1"

    def validate(self, *, strict: bool = False) -> None:
        values = asdict(self)
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
        return {key: str(value) for key, value in asdict(self).items()}

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
