"""Typed, non-secret platform configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aai_core.tags import ResourceContext

_RESOURCE_ENV = {
    "application": "AAI_APPLICATION",
    "project": "AAI_PROJECT",
    "environment": "AAI_ENVIRONMENT",
    "team": "AAI_TEAM",
    "owner_group": "AAI_OWNER_GROUP",
    "cost_center": "AAI_COST_CENTER",
    "data_classification": "AAI_DATA_CLASSIFICATION",
    "lifecycle": "AAI_LIFECYCLE",
    "repository": "AAI_REPOSITORY",
    "release": "AAI_RELEASE",
}

_PLATFORM_ENV = {
    "catalog": "AAI_CATALOG",
    "schema": "AAI_SCHEMA",
    "experiment_name": "AAI_EXPERIMENT_NAME",
    "azure_identity": "AAI_AZURE_IDENTITY",
}


@dataclass(frozen=True)
class PlatformSettings:
    """Configuration shared by SDK services.

    The settings object contains references and identifiers only. Secret values
    are deliberately resolved later by :mod:`aai_core.secrets`.
    """

    resource: ResourceContext
    catalog: str = "unset"
    schema: str = "unset"
    experiment_name: str = "unset"
    azure_identity: str = "auto"
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    embeddings: dict[str, dict[str, Any]] = field(default_factory=dict)
    retrievers: dict[str, dict[str, Any]] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def strict(self) -> bool:
        strict_environments = {"test", "uat", "prod", "production"}
        return self.resource.environment.lower() in strict_environments

    def validate(self) -> None:
        self.resource.validate(strict=self.strict)
        if self.strict:
            missing = [
                name
                for name in ("catalog", "schema", "experiment_name")
                if getattr(self, name) in {"", "unset"}
            ]
            if missing:
                joined = ", ".join(missing)
                raise ValueError(f"Production platform settings are missing: {joined}")
            if self.azure_identity == "auto":
                raise ValueError(
                    "Production must select workload_identity or managed_identity; "
                    "azure_identity=auto is allowed only for local development."
                )

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        environ: dict[str, str] | None = None,
        **overrides: Any,
    ) -> PlatformSettings:
        """Load YAML, then apply ``AAI_*`` environment and explicit overrides."""

        environment = environ if environ is not None else dict(os.environ)
        config_path = Path(path or "aai-platform.yml")
        document: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open(encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{config_path} must contain a YAML mapping")
            document = loaded

        platform = dict(document.get("platform", {}))
        providers = dict(document.get("providers", {}))
        secret_refs = dict(document.get("secrets", {}))

        resource_values: dict[str, Any] = {}
        for field_name, env_name in _RESOURCE_ENV.items():
            resource_values[field_name] = environment.get(
                env_name, platform.get(field_name, _resource_default(field_name))
            )
        resource_values["tag_schema_version"] = platform.get("tag_schema_version", "1")

        platform_values: dict[str, Any] = {}
        for field_name, env_name in _PLATFORM_ENV.items():
            default = "auto" if field_name == "azure_identity" else "unset"
            platform_values[field_name] = environment.get(
                env_name, platform.get(field_name, default)
            )

        for key, value in overrides.items():
            if key in resource_values:
                resource_values[key] = value
            elif key in platform_values:
                platform_values[key] = value
            else:
                raise TypeError(f"Unknown platform setting override: {key}")

        settings = cls(
            resource=ResourceContext(**resource_values),
            models=dict(providers.get("models", {})),
            embeddings=dict(providers.get("embeddings", {})),
            retrievers=dict(providers.get("retrievers", {})),
            secrets=secret_refs,
            raw=document,
            **platform_values,
        )
        settings.validate()
        return settings


def _resource_default(field_name: str) -> str:
    defaults = {
        "environment": "dev",
        "data_classification": "internal",
        "lifecycle": "experimental",
        "release": "dev",
    }
    return defaults.get(field_name, "unset")
