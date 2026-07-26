"""Typed, non-secret platform configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_serializer, field_validator

from aai_core.contracts import ContractModel, freeze_value, thaw_value
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


class PlatformSettings(ContractModel):
    """Configuration shared by SDK services.

    The settings object contains references and identifiers only. Secret values
    are deliberately resolved later by :mod:`aai_core.secrets`.
    """

    resource: ResourceContext
    catalog: str = "unset"
    schema_name: str = Field(default="unset", alias="schema")
    experiment_name: str = "unset"
    azure_identity: str = "auto"
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)
    embeddings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    retrievers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict, repr=False, exclude=True)

    @field_validator(
        "models",
        "embeddings",
        "retrievers",
        "secrets",
        "raw",
        mode="after",
    )
    @classmethod
    def freeze_configuration(cls, value: Mapping[str, Any]):
        return freeze_value(value)

    @field_serializer("models", "embeddings", "retrievers", "secrets")
    def serialize_configuration(self, value: Mapping[str, Any]):
        return thaw_value(value)

    @property
    def strict(self) -> bool:
        strict_environments = {"test", "staging", "uat", "prod", "production"}
        return self.resource.environment.lower() in strict_environments

    @property
    def schema(self) -> str:
        """Unity Catalog schema name.

        The internal field name avoids shadowing Pydantic's deprecated
        ``BaseModel.schema`` compatibility method; the constructor and public
        SDK continue to use ``schema=...``.
        """

        return self.schema_name

    @property
    def effective_experiment_name(self) -> str:
        """The experiment every SDK surface uses.

        Explicit configuration wins; otherwise the platform naming convention
        applies: ``/Shared/<team>-<project>-<application>``. The experiment is
        the durable comparison space for one AI application; ``environment``
        remains a required run tag instead of fragmenting the evidence across
        experiments. Strict environments still require an explicit name
        (validated in :meth:`validate`).
        """

        if self.experiment_name not in {"", "unset"}:
            return self.experiment_name
        resource = self.resource
        return f"/Shared/{resource.team}-{resource.project}-{resource.application}"

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
        """Load YAML, then apply ``AAI_*`` environment and explicit overrides.

        When ``path`` is omitted, the configuration file is discovered
        portably (see :func:`find_platform_config`) so notebooks and scripts
        work unchanged regardless of the working directory they run from.
        """

        environment = environ if environ is not None else dict(os.environ)
        config_path = (
            Path(path)
            if path is not None
            else find_platform_config(environ=environment)
        )
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


def find_platform_config(
    start: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Locate ``aai-platform.yml`` without hardcoding paths.

    Resolution order: the ``AAI_PLATFORM_CONFIG`` environment variable, then
    an upward search from ``start`` (default: the working directory) through
    its parents. Notebooks and jobs therefore never embed repo-relative
    paths — ``bootstrap()`` with no arguments finds the project
    configuration wherever the process starts. Returns the conventional
    ``aai-platform.yml`` in the start directory when nothing is found, so
    missing-file errors stay clear.
    """

    env = environ if environ is not None else os.environ
    override = env.get("AAI_PLATFORM_CONFIG")
    if override:
        return Path(override)
    base = Path(start) if start is not None else Path.cwd()
    for directory in (base, *base.parents):
        config_path = directory / "aai-platform.yml"
        if config_path.is_file():
            return config_path
    return base / "aai-platform.yml"


def _resource_default(field_name: str) -> str:
    defaults = {
        "environment": "dev",
        "data_classification": "internal",
        "lifecycle": "experimental",
        "release": "dev",
    }
    return defaults.get(field_name, "unset")
