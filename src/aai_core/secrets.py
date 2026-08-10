"""Secret references and resolvers.

Secret values are intentionally represented by :class:`SecretValue`, whose
string and repr forms are always redacted. Call ``reveal()`` only at the point
where a native client requires the raw value.
"""

from __future__ import annotations

import os
import time
from base64 import b64decode
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from threading import RLock
from typing import Any, NoReturn, Protocol, SupportsIndex, cast
from urllib.parse import urlparse

from aai_core.logging import Redactor

__all__ = [
    "AzureKeyVaultSecretProvider",
    "DatabricksSecretProvider",
    "EnvironmentSecretProvider",
    "SecretProvider",
    "SecretRef",
    "SecretResolver",
    "SecretValue",
    "default_secret_resolver",
]


@dataclass(frozen=True)
class SecretRef:
    """Parsed provider, authority, and name for a non-secret reference URI."""

    scheme: str
    authority: str
    name: str

    @classmethod
    def parse(cls, value: str | SecretRef) -> SecretRef:
        if isinstance(value, SecretRef):
            return value
        parsed = urlparse(value)
        if not parsed.scheme or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(
                "Secret references must use scheme://authority/name syntax"
            )
        return cls(
            scheme=parsed.scheme,
            authority=parsed.netloc,
            name=parsed.path.strip("/"),
        )

    def __str__(self) -> str:
        return f"{self.scheme}://{self.authority}/{self.name}"


class SecretValue:
    """Opaque secret value whose string, repr, and serialization stay redacted."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretValue('[REDACTED]')"

    def __str__(self) -> str:
        return "[REDACTED]"

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("SecretValue cannot be serialized")

    def __getstate__(self) -> NoReturn:
        raise TypeError("SecretValue cannot be serialized")


class SecretProvider(Protocol):
    """Protocol implemented by native secret-reference resolvers."""

    def resolve(self, reference: SecretRef) -> str: ...


class _CachingProvider:
    def __init__(self, *, ttl_seconds: float = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, str]] = {}
        self._inflight: dict[str, Future[str]] = {}
        self._lock = RLock()
        self._closed = False

    def _cached(self, reference: SecretRef, loader: Callable[[], str]) -> str:
        key = str(reference)
        with self._lock:
            self._ensure_open()
            cached = self._cache.get(key)
            if cached and cached[0] > time.monotonic():
                return cached[1]
            future = self._inflight.get(key)
            leader = future is None
            if future is None:
                future = Future()
                self._inflight[key] = future

        if not leader:
            value = future.result()
            with self._lock:
                self._ensure_open()
                return value

        try:
            value = loader()
        except BaseException as error:
            failure, closed = self._finish_failed_load(key, future, error)
            if closed:
                raise failure from None
            raise
        return self._finish_successful_load(key, future, value)

    def _finish_successful_load(self, key: str, future: Future[str], value: str) -> str:
        with self._lock:
            if self._closed:
                failure = self._closed_error()
                if self._inflight.get(key) is future:
                    self._inflight.pop(key, None)
                if not future.done():
                    future.set_exception(failure)
                raise failure
            self._cache[key] = (time.monotonic() + self.ttl_seconds, value)
            if self._inflight.get(key) is future:
                self._inflight.pop(key, None)
            if not future.done():
                future.set_result(value)
            return value

    def _finish_failed_load(
        self,
        key: str,
        future: Future[str],
        error: BaseException,
    ) -> tuple[BaseException, bool]:
        with self._lock:
            closed = self._closed
            failure = self._closed_error() if closed else error
            if self._inflight.get(key) is future:
                self._inflight.pop(key, None)
            if not future.done():
                future.set_exception(failure)
            return failure, closed

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def close(self) -> None:
        """Clear cached values; subclasses also close their native clients."""

        with self._lock:
            self._mark_closed_locked()

    def _ensure_open(self) -> None:
        if self._closed:
            raise self._closed_error()

    def _closed_error(self) -> RuntimeError:
        return RuntimeError(f"{type(self).__name__} is closed")

    def _mark_closed_locked(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        self._cache.clear()
        inflight = tuple(self._inflight.values())
        self._inflight.clear()
        for future in inflight:
            if not future.done():
                future.set_exception(self._closed_error())
        return True


class AzureKeyVaultSecretProvider(_CachingProvider):
    """Cached keyless Azure Key Vault secret provider."""

    def __init__(
        self,
        *,
        credential: object | None = None,
        client_factory: Callable[[str, object], object] | None = None,
        ttl_seconds: float = 300,
        azure_identity: str = "auto",
    ) -> None:
        super().__init__(ttl_seconds=ttl_seconds)
        self._credential = credential
        self._owns_credential = credential is None
        self._client_factory = client_factory
        self._azure_identity = azure_identity
        self._clients: dict[str, object] = {}

    def resolve(self, reference: SecretRef) -> str:
        def load() -> str:
            client = self._client(reference.authority)
            secret = client.get_secret(reference.name)  # type: ignore[attr-defined]
            return str(secret.value)

        return self._cached(reference, load)

    def _client(self, vault: str) -> object:
        with self._lock:
            if self._closed:
                raise RuntimeError("AzureKeyVaultSecretProvider is closed")
            if vault in self._clients:
                return self._clients[vault]
            if self._credential is None:
                from aai_core.identity import azure_credential

                # Honor the configured identity mode; falling back to the
                # DefaultAzureCredential chain is for local development only.
                self._credential = azure_credential(self._azure_identity)
            if self._client_factory is None:
                from azure.keyvault.secrets import SecretClient

                self._client_factory = lambda vault_url, credential: SecretClient(
                    vault_url=vault_url,
                    credential=cast(Any, credential),
                )
            factory = self._client_factory
            client = factory(f"https://{vault}.vault.azure.net", self._credential)
            self._clients[vault] = client
            return client

    def close(self) -> None:
        """Close provider-created clients and credentials exactly once."""

        with self._lock:
            if not self._mark_closed_locked():
                return
            clients = list(self._clients.values())
            self._clients.clear()
            credential = self._credential if self._owns_credential else None
            self._credential = None
        # _close_resources closes in reverse order: clients before credential.
        _close_resources([credential, *clients])


class DatabricksSecretProvider(_CachingProvider):
    """Cached Databricks secret-scope provider."""

    def __init__(
        self,
        *,
        getter: Callable[[str, str], str] | None = None,
        ttl_seconds: float = 300,
    ) -> None:
        super().__init__(ttl_seconds=ttl_seconds)
        self._getter = getter
        self._owned_getter_resource: object | None = None

    def resolve(self, reference: SecretRef) -> str:
        def load() -> str:
            getter = self._resolved_getter()
            return str(getter(reference.authority, reference.name))

        return self._cached(reference, load)

    def _resolved_getter(self) -> Callable[[str, str], str]:
        with self._lock:
            self._ensure_open()
            if self._getter is None:
                self._getter, self._owned_getter_resource = _databricks_secret_getter()
            return self._getter

    def close(self) -> None:
        """Close only the lazily created fallback client, exactly once."""

        with self._lock:
            if not self._mark_closed_locked():
                return
            resource = self._owned_getter_resource
            self._owned_getter_resource = None
            if resource is not None:
                # The fallback callable closes over the workspace client. Drop
                # that reference after capturing the owned resource for close.
                self._getter = None
        _close_resources([resource])


class EnvironmentSecretProvider(_CachingProvider):
    """Explicit local-only provider; never register it in strict environments."""

    def __init__(
        self, *, environ: dict[str, str] | None = None, ttl_seconds: float = 30
    ) -> None:
        super().__init__(ttl_seconds=ttl_seconds)
        self._environ = environ if environ is not None else os.environ

    def resolve(self, reference: SecretRef) -> str:
        def load() -> str:
            if reference.name not in self._environ:
                raise KeyError(f"Environment variable {reference.name!r} is not set")
            return self._environ[reference.name]

        return self._cached(reference, load)


class SecretResolver:
    """Route secret references and register resolved values for redaction."""

    def __init__(self, *, redactor: Redactor | None = None) -> None:
        self.redactor = redactor or Redactor()
        self._providers: dict[str, SecretProvider] = {}
        self._owned_schemes: set[str] = set()
        self._lock = RLock()
        self._closed = False

    def register(
        self,
        scheme: str,
        provider: SecretProvider,
        *,
        owned: bool = False,
    ) -> None:
        """Register a provider, optionally transferring close ownership."""

        with self._lock:
            self._ensure_open()
            if scheme in self._providers:
                raise ValueError(f"Secret provider {scheme!r} is already registered")
            self._providers[scheme] = provider
            if owned:
                self._owned_schemes.add(scheme)
            else:
                self._owned_schemes.discard(scheme)

    def resolve(self, reference: str | SecretRef) -> SecretValue:
        parsed = SecretRef.parse(reference)
        with self._lock:
            self._ensure_open()
            try:
                provider = self._providers[parsed.scheme]
            except KeyError as error:
                raise ValueError(
                    f"No secret provider registered for {parsed.scheme!r}"
                ) from error
        value = provider.resolve(parsed)
        with self._lock:
            self._ensure_open()
            self.redactor.register(value)
            return SecretValue(value)

    def close(self) -> None:
        """Close only providers whose ownership was transferred to the resolver."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            providers = [
                provider
                for scheme, provider in self._providers.items()
                if scheme in self._owned_schemes
            ]
            self._providers.clear()
            self._owned_schemes.clear()
        _close_resources(providers)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SecretResolver is closed")


def default_secret_resolver(
    *,
    redactor: Redactor | None = None,
    allow_environment: bool = False,
    azure_identity: str = "auto",
) -> SecretResolver:
    """Build the standard keyless resolver for the selected identity mode."""

    resolver = SecretResolver(redactor=redactor)
    resolver.register(
        "keyvault",
        AzureKeyVaultSecretProvider(azure_identity=azure_identity),
        owned=True,
    )
    resolver.register("databricks-secret", DatabricksSecretProvider(), owned=True)
    if allow_environment:
        resolver.register("env", EnvironmentSecretProvider(), owned=True)
    return resolver


def _close_resources(resources: Sequence[object | None]) -> None:
    failures = 0
    seen: set[int] = set()
    for resource in reversed(resources):
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:
            failures += 1
    if failures:
        raise RuntimeError(
            f"Failed to close {failures} SDK-owned secret resource(s)"
        ) from None


def _databricks_secret_getter() -> tuple[Callable[[str, str], str], object | None]:
    try:
        import builtins

        dbutils = cast(Any, builtins).dbutils
        return cast(Callable[[str, str], str], dbutils.secrets.get), None
    except (AttributeError, ImportError):
        try:
            from databricks.sdk import WorkspaceClient
        except ImportError as error:
            raise RuntimeError(
                "Databricks secret resolution requires notebook dbutils, "
                "`aai-core[databricks]`, or an explicitly injected getter"
            ) from error

        workspace = WorkspaceClient()

        def get(scope: str, key: str) -> str:
            encoded = workspace.secrets.get_secret(scope=scope, key=key).value
            if encoded is None:
                raise KeyError(f"Databricks secret {scope}/{key} has no value")
            return b64decode(encoded).decode("utf-8")

        return get, _databricks_workspace_close_resource(workspace)


def _databricks_workspace_close_resource(workspace: object) -> object:
    """Return the narrowest closeable resource owned by a WorkspaceClient.

    The certified Databricks SDK has no public ``WorkspaceClient.close()``.
    Prefer one when it becomes available; until then close its requests session
    so the provider does not retain connection pools after shutdown.
    """

    if callable(getattr(workspace, "close", None)):
        return workspace
    api_client = getattr(workspace, "api_client", None)
    transport = getattr(api_client, "_api_client", None)
    session = getattr(transport, "_session", None)
    return session if callable(getattr(session, "close", None)) else workspace
