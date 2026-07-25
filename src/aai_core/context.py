"""Composition root for the SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aai_core.identity import databricks_workspace_client
from aai_core.logging import Redactor
from aai_core.runtime import PlatformSettings
from aai_core.secrets import SecretResolver, default_secret_resolver

if TYPE_CHECKING:
    from aai_core.experiments import ExperimentManager
    from aai_core.prompts import PromptManager
    from aai_core.providers import ProviderResolver


@dataclass
class PlatformContext:
    settings: PlatformSettings
    redactor: Redactor = field(default_factory=Redactor)
    _secrets: SecretResolver | None = field(default=None, init=False, repr=False)
    _workspace: object | None = field(default=None, init=False, repr=False)
    _providers: ProviderResolver | None = field(default=None, init=False, repr=False)
    _experiments: ExperimentManager | None = field(default=None, init=False, repr=False)
    _prompts: PromptManager | None = field(default=None, init=False, repr=False)

    @property
    def tags(self):
        return self.settings.resource

    @property
    def secrets(self) -> SecretResolver:
        if self._secrets is None:
            self._secrets = default_secret_resolver(
                redactor=self.redactor,
                allow_environment=not self.settings.strict,
            )
        return self._secrets

    @property
    def workspace(self) -> object:
        if self._workspace is None:
            self._workspace = databricks_workspace_client()
        return self._workspace

    @property
    def providers(self) -> ProviderResolver:
        if self._providers is None:
            from aai_core.providers import ProviderResolver

            self._providers = ProviderResolver(self)
        return self._providers

    @property
    def experiments(self) -> ExperimentManager:
        if self._experiments is None:
            from aai_core.experiments import ExperimentManager

            self._experiments = ExperimentManager(
                experiment_name=self.settings.experiment_name,
                context=self.tags,
            )
        return self._experiments

    @property
    def prompts(self) -> PromptManager:
        if self._prompts is None:
            from aai_core.prompts import PromptManager

            self._prompts = PromptManager(
                context=self.tags,
                catalog=self.settings.catalog,
                schema=self.settings.schema,
            )
        return self._prompts

    def configure_tracing(self, **kwargs: Any) -> None:
        from aai_core.tracing import configure_tracing

        configure_tracing(
            self.tags,
            experiment_name=self.settings.experiment_name,
            **kwargs,
        )

    def configure_logging(self, **kwargs: Any) -> None:
        from aai_core.logging import configure_logging

        configure_logging(self.tags, redactor=self.redactor, **kwargs)


def bootstrap(
    config_path: str | Path | None = None,
    **overrides: Any,
) -> PlatformContext:
    """Load settings and return the SDK composition root."""

    return PlatformContext(PlatformSettings.load(config_path, **overrides))
