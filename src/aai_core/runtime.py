"""Typed, non-secret platform configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
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
        strict_environments = {"test", "staging", "uat", "prod", "production"}
        return self.resource.environment.lower() in strict_environments

    @property
    def effective_experiment_name(self) -> str:
        """The experiment every SDK surface uses.

        Explicit configuration wins; otherwise the platform naming convention
        applies: ``/Shared/<team>-<application>-<environment>`` — derived from
        the governed tags so experiments are attributable and consistent
        across teams without per-project invention. Strict environments still
        require an explicit name (validated in :meth:`validate`).
        """

        if self.experiment_name not in {"", "unset"}:
            return self.experiment_name
        resource = self.resource
        return (
            f"/Shared/{resource.team}-{resource.application}-" f"{resource.environment}"
        )

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
        candidate = directory / "aai-platform.yml"
        if candidate.is_file():
            return candidate
    return base / "aai-platform.yml"


def _resource_default(field_name: str) -> str:
    defaults = {
        "environment": "dev",
        "data_classification": "internal",
        "lifecycle": "experimental",
        "release": "dev",
    }
    return defaults.get(field_name, "unset")
