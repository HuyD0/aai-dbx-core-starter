"""Composition root for the SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from aai_core.identity import databricks_workspace_client
from aai_core.logging import Redactor
from aai_core.runtime import PlatformSettings
from aai_core.secrets import SecretResolver, default_secret_resolver
from aai_core.tags import ResourceContext

if TYPE_CHECKING:
    from aai_core.experiments import ExperimentManager
    from aai_core.prompts import PromptManager
    from aai_core.providers import ProviderResolver
    from aai_core.tracing import TraceState

__all__ = ["PlatformContext", "bootstrap"]


@dataclass
class PlatformContext:
    """Thread-safe composition root that owns SDK-created synchronous clients."""

    settings: PlatformSettings
    redactor: Redactor = field(default_factory=Redactor)
    _secrets: SecretResolver | None = field(default=None, init=False, repr=False)
    _workspace: object | None = field(default=None, init=False, repr=False)
    _providers: ProviderResolver | None = field(default=None, init=False, repr=False)
    _experiments: ExperimentManager | None = field(default=None, init=False, repr=False)
    _prompts: PromptManager | None = field(default=None, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def tags(self) -> ResourceContext:
        """Return the governed resource context shared by SDK integrations."""

        return self.settings.resource

    @property
    def secrets(self) -> SecretResolver:
        """Return the one lazily constructed secret resolver."""

        with self._lock:
            self._ensure_open()
            if self._secrets is None:
                self._secrets = default_secret_resolver(
                    redactor=self.redactor,
                    allow_environment=not self.settings.strict,
                    azure_identity=self.settings.azure_identity,
                )
            return self._secrets

    @property
    def workspace(self) -> object:
        """Return the one SDK-owned Databricks workspace client."""

        with self._lock:
            self._ensure_open()
            if self._workspace is None:
                self._workspace = databricks_workspace_client()
            return self._workspace

    @property
    def providers(self) -> ProviderResolver:
        """Return the one provider resolver for this context."""

        with self._lock:
            self._ensure_open()
            if self._providers is None:
                from aai_core.providers import ProviderResolver

                self._providers = ProviderResolver(self)
            return self._providers

    @property
    def experiments(self) -> ExperimentManager:
        """Return the one experiment manager for this context."""

        with self._lock:
            self._ensure_open()
            if self._experiments is None:
                from aai_core.experiments import ExperimentManager

                self._experiments = ExperimentManager(
                    experiment_name=self.settings.effective_experiment_name,
                    context=self.tags,
                )
            return self._experiments

    @property
    def prompts(self) -> PromptManager:
        """Return the one prompt manager for this context."""

        with self._lock:
            self._ensure_open()
            if self._prompts is None:
                from aai_core.prompts import PromptManager

                self._prompts = PromptManager(
                    context=self.tags,
                    catalog=self.settings.catalog,
                    schema=self.settings.schema,
                )
            return self._prompts

    def configure_tracing(self, **kwargs: Any) -> TraceState:
        """Configure governed tracing for this context."""

        with self._lock:
            self._ensure_open()
        from aai_core.tracing import configure_tracing

        return configure_tracing(
            self.tags,
            experiment_name=self.settings.effective_experiment_name,
            **kwargs,
        )

    def configure_logging(self, **kwargs: Any) -> None:
        """Configure structured logging without replacing host handlers."""

        with self._lock:
            self._ensure_open()
        from aai_core.logging import configure_logging

        configure_logging(self.tags, redactor=self.redactor, **kwargs)

    def close(self) -> None:
        """Close SDK-owned synchronous resources exactly once.

        Registered provider clients remain caller-owned and are never closed.
        """

        with self._lock:
            if self._closed:
                return
            self._closed = True
            resources = [self._workspace, self._secrets, self._providers]
            self._workspace = None
            self._secrets = None
            self._providers = None
            self._experiments = None
            self._prompts = None
        _close_resources(resources)

    def __enter__(self) -> PlatformContext:
        """Return this open context for synchronous ``with`` usage."""

        with self._lock:
            self._ensure_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close SDK-owned resources when leaving a synchronous context."""

        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PlatformContext is closed")


def bootstrap(
    config_path: str | Path | None = None,
    **overrides: Any,
) -> PlatformContext:
    """Load settings and return the SDK composition root."""

    return PlatformContext(PlatformSettings.load(config_path, **overrides))


def _close_resources(resources: list[object | None]) -> None:
    failures = 0
    for resource in reversed(resources):
        if resource is None:
            continue
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:
            failures += 1
    if failures:
        raise RuntimeError(
            f"Failed to close {failures} SDK-owned resource(s)"
        ) from None
