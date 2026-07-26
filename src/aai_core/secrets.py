"""Secret references and resolvers.

Secret values are intentionally represented by :class:`SecretValue`, whose
string and repr forms are always redacted. Call ``reveal()`` only at the point
where a native client requires the raw value.
"""

from __future__ import annotations

import os
import time
from base64 import b64decode
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Protocol
from urllib.parse import urlparse

from aai_core.logging import Redactor


@dataclass(frozen=True)
class SecretRef:
    scheme: str
    authority: str
    name: str

    @classmethod
    def parse(cls, value: str | SecretRef) -> SecretRef:
        if isinstance(value, cls):
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


class SecretProvider(Protocol):
    def resolve(self, reference: SecretRef) -> str: ...


class _CachingProvider:
    def __init__(self, *, ttl_seconds: float = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, str]] = {}
        self._lock = RLock()

    def _cached(self, reference: SecretRef, loader: Callable[[], str]) -> str:
        key = str(reference)
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[0] > now:
                return cached[1]
        value = loader()
        with self._lock:
            self._cache[key] = (now + self.ttl_seconds, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class AzureKeyVaultSecretProvider(_CachingProvider):
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
        if vault in self._clients:
            return self._clients[vault]
        if self._credential is None:
            from aai_core.identity import azure_credential

            # Honor the configured identity mode; falling back to the
            # DefaultAzureCredential chain is for local development only.
            self._credential = azure_credential(self._azure_identity)
        if self._client_factory is None:
            from azure.keyvault.secrets import SecretClient

            self._client_factory = SecretClient
        client = self._client_factory(
            f"https://{vault}.vault.azure.net", self._credential
        )
        self._clients[vault] = client
        return client


class DatabricksSecretProvider(_CachingProvider):
    def __init__(
        self,
        *,
        getter: Callable[[str, str], str] | None = None,
        ttl_seconds: float = 300,
    ) -> None:
        super().__init__(ttl_seconds=ttl_seconds)
        self._getter = getter

    def resolve(self, reference: SecretRef) -> str:
        def load() -> str:
            getter = self._getter or _databricks_secret_getter()
            return str(getter(reference.authority, reference.name))

        return self._cached(reference, load)


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
    def __init__(self, *, redactor: Redactor | None = None) -> None:
        self.redactor = redactor or Redactor()
        self._providers: dict[str, SecretProvider] = {}

    def register(self, scheme: str, provider: SecretProvider) -> None:
        self._providers[scheme] = provider

    def resolve(self, reference: str | SecretRef) -> SecretValue:
        parsed = SecretRef.parse(reference)
        try:
            provider = self._providers[parsed.scheme]
        except KeyError as error:
            raise ValueError(
                f"No secret provider registered for {parsed.scheme!r}"
            ) from error
        value = provider.resolve(parsed)
        self.redactor.register(value)
        return SecretValue(value)


def default_secret_resolver(
    *,
    redactor: Redactor | None = None,
    allow_environment: bool = False,
    azure_identity: str = "auto",
) -> SecretResolver:
    resolver = SecretResolver(redactor=redactor)
    resolver.register(
        "keyvault", AzureKeyVaultSecretProvider(azure_identity=azure_identity)
    )
    resolver.register("databricks-secret", DatabricksSecretProvider())
    if allow_environment:
        resolver.register("env", EnvironmentSecretProvider())
    return resolver


def _databricks_secret_getter() -> Callable[[str, str], str]:
    try:
        import builtins

        dbutils = builtins.dbutils
        return dbutils.secrets.get
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

        return get
