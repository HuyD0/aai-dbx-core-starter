"""Production persistence for the durable LangGraph recipe, backed by Lakebase.

Lakebase is managed PostgreSQL, so the native LangGraph Postgres saver and
store are the persistence implementation; the only Databricks-specific part
is credential minting. This module wires the two together without ever
placing an OAuth token in a connection string, environment variable, log
field, or exception. Every pooled connection pins its search path to the
application-owned schema, so durable state never lands in a shared
``public`` schema.

LangGraph owns graph state and checkpoint APIs. `aai-core` supplies resource
context, tracing policy, provider clients, evaluation and release evidence;
it does not wrap the graph or its persistence.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_TLS_SSLMODES = frozenset({"require", "verify-ca", "verify-full"})
_POSTGRES_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_HOSTNAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")
_ENDPOINT_PATH = re.compile(
    r"^projects/[a-z][a-z0-9-]{0,62}/branches/"
    r"[a-z][a-z0-9-]{0,62}/endpoints/[a-z][a-z0-9-]{0,62}$"
)
_SAFE_TEXT = re.compile(r"^[^\x00\r\n]+$")


class LakebasePersistenceError(RuntimeError):
    """The runtime contract for durable persistence is unsafe or incomplete."""


class LakebaseSettings(BaseModel):
    """Non-secret connection coordinates supplied by the App resource binding.

    A Databricks Apps ``postgres`` resource binding supplies ``PGHOST``,
    ``PGPORT``, ``PGDATABASE``, ``PGUSER``, ``PGSSLMODE`` and the
    ``LAKEBASE_ENDPOINT`` resource path at runtime; ``LAKEBASE_SCHEMA`` names
    the schema the application role owns. The Lakebase instance, role, and
    grants are provisioned through the external platform process; this recipe
    only ever connects to them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    host: str
    port: int = Field(default=5432, ge=1, le=65_535)
    database: str
    user: str
    endpoint: str
    schema_name: str
    sslmode: str = "require"
    application_name: str = "aai-agent-app"
    connect_timeout_seconds: int = Field(default=10, ge=1, le=60)
    pool_min_size: int = Field(default=1, ge=0, le=16)
    pool_max_size: int = Field(default=5, ge=1, le=32)

    @field_validator("host")
    @classmethod
    def _valid_hostname(cls, value: str) -> str:
        if not _HOSTNAME.fullmatch(value):
            raise ValueError("host is not a valid hostname")
        return value

    @field_validator("endpoint")
    @classmethod
    def _valid_endpoint(cls, value: str) -> str:
        if not _ENDPOINT_PATH.fullmatch(value):
            raise ValueError(
                "endpoint must be a full Autoscaling endpoint resource path"
            )
        return value

    @field_validator("schema_name")
    @classmethod
    def _valid_schema_name(cls, value: str) -> str:
        if not _POSTGRES_IDENTIFIER.fullmatch(value):
            raise ValueError("schema_name must be a lowercase PostgreSQL identifier")
        return value

    @field_validator("database", "user", "application_name")
    @classmethod
    def _safe_text(cls, value: str) -> str:
        if not _SAFE_TEXT.fullmatch(value):
            raise ValueError("value contains invalid characters")
        return value

    @model_validator(mode="after")
    def _tls_required_off_box(self) -> LakebaseSettings:
        # A Lakebase endpoint is never loopback, so production bindings always
        # use TLS — require or the stronger certificate-verifying modes.
        # Plaintext exists solely for a test server on this host.
        if self.sslmode not in _TLS_SSLMODES and self.host not in _LOOPBACK_HOSTS:
            raise ValueError(
                "sslmode must be require, verify-ca, or verify-full for "
                "non-loopback hosts"
            )
        return self

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> LakebaseSettings:
        required = {
            "PGHOST": environ.get("PGHOST", "").strip(),
            "PGDATABASE": environ.get("PGDATABASE", "").strip(),
            "PGUSER": environ.get("PGUSER", "").strip(),
            "LAKEBASE_ENDPOINT": environ.get("LAKEBASE_ENDPOINT", "").strip(),
            "LAKEBASE_SCHEMA": environ.get("LAKEBASE_SCHEMA", "").strip(),
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise LakebasePersistenceError(
                "Lakebase resource binding is missing required variables: "
                + ", ".join(missing)
            )
        return cls(
            host=required["PGHOST"],
            port=int(environ.get("PGPORT", "").strip() or "5432"),
            database=required["PGDATABASE"],
            user=required["PGUSER"],
            endpoint=required["LAKEBASE_ENDPOINT"],
            schema_name=required["LAKEBASE_SCHEMA"],
            sslmode=(environ.get("PGSSLMODE", "require").strip() or "require"),
        )


@dataclass(frozen=True, repr=False)
class _Token:
    value: str
    expires_at: datetime

    def __repr__(self) -> str:
        return "_Token(value=<redacted>, expires_at=<redacted>)"


class LakebaseCredentialProvider:
    """Cache an OAuth database credential and refresh it before it expires.

    ``generate`` is the injected minting call — in production a closure over
    ``WorkspaceClient().postgres.generate_database_credential(endpoint=...)``.
    It runs in a worker thread because the Databricks SDK call is blocking.
    The provider fails closed: a credential without a token, or one that
    expires inside the refresh skew, is an error rather than a retry storm.
    """

    def __init__(
        self,
        generate: Callable[[], Any],
        *,
        clock: Callable[[], datetime] | None = None,
        refresh_skew: timedelta = timedelta(minutes=2),
    ) -> None:
        self._generate = generate
        self._clock = clock or (lambda: datetime.now(UTC))
        self._refresh_skew = refresh_skew
        self._lock = asyncio.Lock()
        self._current: _Token | None = None

    def __repr__(self) -> str:
        return "LakebaseCredentialProvider(<credential redacted>)"

    async def password(self) -> str:
        now = self._aware(self._clock())
        current = self._current
        if current is not None and current.expires_at > now + self._refresh_skew:
            return current.value
        async with self._lock:
            now = self._aware(self._clock())
            current = self._current
            if current is not None and current.expires_at > now + self._refresh_skew:
                return current.value
            credential = await asyncio.to_thread(self._generate)
            token = getattr(credential, "token", None)
            expires = getattr(credential, "expire_time", None)
            if not isinstance(token, str) or not token:
                raise LakebasePersistenceError(
                    "Lakebase credential generation returned no token."
                )
            expires_at = self._timestamp(expires)
            if expires_at <= now + self._refresh_skew:
                raise LakebasePersistenceError(
                    "Lakebase credential expires too soon for safe pooling."
                )
            self._current = _Token(value=token, expires_at=expires_at)
            return token

    @staticmethod
    def _timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return LakebaseCredentialProvider._aware(value)
        converter = getattr(value, "ToDatetime", None)
        if callable(converter):
            try:
                converted = converter(tzinfo=UTC)
            except TypeError:
                converted = converter().replace(tzinfo=UTC)
            return LakebaseCredentialProvider._aware(converted)
        raise LakebasePersistenceError(
            "Lakebase credential returned no usable expiration time."
        )

    @staticmethod
    def _aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class _FreshTokenPool(AsyncConnectionPool):
    """Connection pool whose every new physical connection mints a live token.

    A pooled connection can be created long after the token that opened the
    pool expired, so the password is refreshed inside the connect path — the
    token exists only in the connection keyword arguments for the duration
    of the call and never in the ``conninfo`` string.
    """

    def __init__(
        self,
        *args: Any,
        credential_provider: LakebaseCredentialProvider,
        **kwargs: Any,
    ) -> None:
        parent_connect = getattr(AsyncConnectionPool, "_connect", None)
        if not asyncio.iscoroutinefunction(parent_connect):
            raise LakebasePersistenceError(
                "psycopg_pool.AsyncConnectionPool no longer exposes an async "
                "_connect hook; re-certify the recipe against this version."
            )
        self._credential_provider = credential_provider
        self._mint_lock = asyncio.Lock()
        super().__init__(*args, **kwargs)

    async def _connect(self, timeout: float | None = None) -> Any:
        # The lock serializes connection establishment so the token can be
        # scrubbed from the shared kwargs mapping immediately after each
        # connect instead of persisting for the pool's lifetime. New physical
        # connections are rare; serializing them costs little.
        async with self._mint_lock:
            self.kwargs["password"] = await self._credential_provider.password()
            try:
                return await super()._connect(timeout=timeout)
            finally:
                self.kwargs.pop("password", None)


def _search_path_configure(schema_name: str) -> Callable[[Any], Awaitable[None]]:
    """Build the pool ``configure`` hook that pins ``search_path``.

    The LangGraph saver and store issue every statement — DDL included —
    with unqualified table names, so the connection's search path is the
    only thing deciding which schema they live in. The hook runs for each
    new physical connection as a session-level ``SET`` rather than the
    libpq ``options`` startup parameter, which pooled Lakebase endpoints
    can strip. The pool discards any connection its configure hook leaves
    inside a transaction; the connect kwargs' ``autocommit=True`` is what
    keeps this ``SET`` transaction-free. The search path is the schema
    alone — no ``public`` fallback — so state can never be read from or
    written to a shared schema.
    """

    statement = sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name))

    async def configure(connection: Any) -> None:
        await connection.execute(statement)

    return configure


_SCHEMA_OWNER_PROBE = (
    "SELECT pg_get_userbyid(nspowner) AS schema_owner, "
    "current_user AS connected_role "
    "FROM pg_namespace WHERE nspname = %s"
)


async def _ensure_owned_schema(pool: AsyncConnectionPool, schema_name: str) -> None:
    """Create the schema when absent and require the connected role to own it.

    Probing before creating matters: ``CREATE SCHEMA IF NOT EXISTS`` checks
    the database-level CREATE privilege even when the schema already exists,
    which would reject a deployment whose schema an administrator pre-created
    ``WITH AUTHORIZATION`` for a role that cannot create database objects.
    First-run creation is what the App binding's ``CAN_CONNECT_AND_CREATE``
    permission covers. Ownership is required either way — the one-time DDL
    must never land in a schema another principal controls.
    """

    async with pool.connection() as connection:
        result = await connection.execute(_SCHEMA_OWNER_PROBE, (schema_name,))
        row = await result.fetchone()
        if row is None:
            await connection.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(schema_name)
                )
            )
            result = await connection.execute(_SCHEMA_OWNER_PROBE, (schema_name,))
            row = await result.fetchone()
        if row is None or row["schema_owner"] != row["connected_role"]:
            raise LakebasePersistenceError(
                "The Lakebase persistence schema must be created and owned by "
                "the connected application role."
            )


@asynccontextmanager
async def build_lakebase_persistence(
    settings: LakebaseSettings,
    generate_credential: Callable[[], Any],
    *,
    run_setup: bool = False,
    clock: Callable[[], datetime] | None = None,
) -> AsyncIterator[tuple[AsyncPostgresSaver, AsyncPostgresStore]]:
    """Yield a durable ``(checkpointer, store)`` pair satisfying the recipe.

    The pair plugs directly into the sibling recipe's ``build_graph()`` and
    passes its async-checkpointer construction check. Every pooled connection
    pins ``search_path`` to the validated ``settings.schema_name``, so the
    LangGraph tables and lookups resolve only inside the application's own
    schema. ``run_setup=True`` executes the savers' one-time DDL — the
    application owns that decision; the schema is created when absent and
    must be owned by the connected role. The recipe never provisions
    Lakebase objects themselves.
    """

    provider = LakebaseCredentialProvider(generate_credential, clock=clock)
    pool = _FreshTokenPool(
        conninfo="",
        credential_provider=provider,
        configure=_search_path_configure(settings.schema_name),
        kwargs={
            "host": settings.host,
            "port": settings.port,
            "dbname": settings.database,
            "user": settings.user,
            "sslmode": settings.sslmode,
            "connect_timeout": settings.connect_timeout_seconds,
            "application_name": settings.application_name,
            # The LangGraph Postgres saver and store require autocommit
            # connections returning dict rows. The search_path configure
            # hook also depends on autocommit: the pool discards any
            # connection whose configure hook leaves a transaction open.
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        min_size=settings.pool_min_size,
        max_size=settings.pool_max_size,
        open=False,
    )
    await pool.open(wait=False)
    try:
        checkpointer = AsyncPostgresSaver(pool)
        store = AsyncPostgresStore(pool)
        if run_setup:
            await _ensure_owned_schema(pool, settings.schema_name)
            await checkpointer.setup()
            await store.setup()
        yield checkpointer, store
    finally:
        await pool.close()
