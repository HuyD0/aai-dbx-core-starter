"""Lakebase Autoscaling persistence for the AI Platform Hub.

The adapter binds to an existing Databricks App ``postgres`` resource.  It never
creates a Lakebase project, branch, database, endpoint, role, or cloud identity.  The
app service principal creates and owns only its configured PostgreSQL schema.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any
from uuid import uuid4

from .lakebase_migrations import MIGRATIONS
from .models import (
    ActionEntityType,
    ActionEvent,
    ActionEventType,
    ApplicationPrincipalRecord,
    ApplicationRecord,
    ApplicationVersionRecord,
    AuthorizationContext,
    EvaluationRunRecord,
    EvaluationStatus,
    PrincipalType,
    PromotionRequestRecord,
    PromotionStatus,
    ReadinessSnapshot,
    RegistrationResult,
    ResourceBindingRecord,
    Role,
)
from .repository import (
    DuplicateActiveEvaluationError,
    DuplicateActivePromotionError,
    FourEyesViolationError,
    HubAuthorizationError,
    HubConflictError,
    HubNotFoundError,
    HubRepositoryError,
    HubRepositoryUnavailableError,
    ImmutableApplicationIdConflictError,
    InvalidStateTransitionError,
    OptimisticConcurrencyError,
    _merge_registered_application,
)

_POSTGRES_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_ENDPOINT_PATH = re.compile(
    r"^projects/[a-z][a-z0-9-]{0,62}/branches/"
    r"[a-z][a-z0-9-]{0,62}/endpoints/[a-z][a-z0-9-]{0,62}$"
)
_HOSTNAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")
_SAFE_TEXT = re.compile(r"^[^\x00\r\n]+$")
_CONNECTION_RETRY_DELAYS = (0.25, 1.0)


class LakebaseConfigurationError(RuntimeError):
    """The bound resource does not expose a safe, complete runtime contract."""


@dataclass(frozen=True)
class LakebaseRuntimeSettings:
    """Non-secret connection coordinates supplied by the App resource binding."""

    host: str
    port: int
    database: str
    user: str
    endpoint: str
    schema: str
    sslmode: str = "require"
    pool_size: int = 5
    max_overflow: int = 5
    pool_timeout_seconds: int = 10
    pool_recycle_seconds: int = 3000
    connect_timeout_seconds: int = 10
    statement_timeout_ms: int = 30_000
    lock_timeout_ms: int = 5_000

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str],
        *,
        schema: str,
    ) -> LakebaseRuntimeSettings:
        required = {
            "PGHOST": environ.get("PGHOST", "").strip(),
            "PGDATABASE": environ.get("PGDATABASE", "").strip(),
            "PGUSER": environ.get("PGUSER", "").strip(),
            "LAKEBASE_ENDPOINT": environ.get("LAKEBASE_ENDPOINT", "").strip(),
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise LakebaseConfigurationError(
                "Lakebase resource binding is missing required variables: "
                + ", ".join(missing)
            )

        host = required["PGHOST"]
        endpoint = required["LAKEBASE_ENDPOINT"]
        database = required["PGDATABASE"]
        user = required["PGUSER"]
        if not _HOSTNAME.fullmatch(host):
            raise LakebaseConfigurationError("PGHOST is not a valid hostname")
        if not _ENDPOINT_PATH.fullmatch(endpoint):
            raise LakebaseConfigurationError(
                "LAKEBASE_ENDPOINT must be a full Autoscaling endpoint resource path"
            )
        if not _POSTGRES_IDENTIFIER.fullmatch(schema):
            raise LakebaseConfigurationError(
                "AAI_HUB_LAKEBASE_SCHEMA must be a lowercase PostgreSQL identifier"
            )
        if not _SAFE_TEXT.fullmatch(database) or not _SAFE_TEXT.fullmatch(user):
            raise LakebaseConfigurationError(
                "Lakebase database and user values contain invalid characters"
            )

        sslmode = (environ.get("PGSSLMODE", "require").strip() or "require").lower()
        if sslmode != "require":
            raise LakebaseConfigurationError("PGSSLMODE must be require")

        def integer(name: str, default: int, *, minimum: int, maximum: int) -> int:
            raw = environ.get(name, "").strip()
            try:
                value = default if not raw else int(raw)
            except ValueError as error:
                raise LakebaseConfigurationError(
                    f"{name} must be an integer"
                ) from error
            if not minimum <= value <= maximum:
                raise LakebaseConfigurationError(
                    f"{name} must be between {minimum} and {maximum}"
                )
            return value

        statement_timeout_ms = integer(
            "AAI_HUB_LAKEBASE_STATEMENT_TIMEOUT_MS",
            30_000,
            minimum=1_000,
            maximum=300_000,
        )
        lock_timeout_ms = integer(
            "AAI_HUB_LAKEBASE_LOCK_TIMEOUT_MS",
            5_000,
            minimum=100,
            maximum=60_000,
        )
        if lock_timeout_ms > statement_timeout_ms:
            raise LakebaseConfigurationError(
                "AAI_HUB_LAKEBASE_LOCK_TIMEOUT_MS must not exceed "
                "AAI_HUB_LAKEBASE_STATEMENT_TIMEOUT_MS"
            )

        return cls(
            host=host,
            port=integer("PGPORT", 5432, minimum=1, maximum=65535),
            database=database,
            user=user,
            endpoint=endpoint,
            schema=schema,
            sslmode=sslmode,
            pool_size=integer("AAI_HUB_LAKEBASE_POOL_SIZE", 5, minimum=1, maximum=20),
            max_overflow=integer(
                "AAI_HUB_LAKEBASE_MAX_OVERFLOW", 5, minimum=0, maximum=40
            ),
            pool_timeout_seconds=integer(
                "AAI_HUB_LAKEBASE_POOL_TIMEOUT_SECONDS",
                10,
                minimum=1,
                maximum=60,
            ),
            pool_recycle_seconds=integer(
                "AAI_HUB_LAKEBASE_POOL_RECYCLE_SECONDS",
                3000,
                minimum=300,
                maximum=3300,
            ),
            connect_timeout_seconds=integer(
                "AAI_HUB_LAKEBASE_CONNECT_TIMEOUT_SECONDS",
                10,
                minimum=1,
                maximum=60,
            ),
            statement_timeout_ms=statement_timeout_ms,
            lock_timeout_ms=lock_timeout_ms,
        )

    @property
    def connection_options(self) -> str:
        """Validated per-session PostgreSQL execution bounds."""

        return (
            f"-c statement_timeout={self.statement_timeout_ms} "
            f"-c lock_timeout={self.lock_timeout_ms}"
        )


@dataclass(frozen=True, repr=False)
class _OAuthToken:
    value: str
    expires_at: datetime

    def __repr__(self) -> str:
        return "_OAuthToken(value=<redacted>, expires_at=<redacted>)"


class LakebaseOAuthTokenProvider:
    """Cache an OAuth database credential and refresh it before connection churn."""

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
        self._lock = Lock()
        self._current: _OAuthToken | None = None

    def __repr__(self) -> str:
        return "LakebaseOAuthTokenProvider(<credential redacted>)"

    def password(self) -> str:
        now = self._aware(self._clock())
        current = self._current
        if current is not None and current.expires_at > now + self._refresh_skew:
            return current.value
        with self._lock:
            now = self._aware(self._clock())
            current = self._current
            if current is not None and current.expires_at > now + self._refresh_skew:
                return current.value
            credential = self._generate()
            token = getattr(credential, "token", None)
            expires = getattr(credential, "expire_time", None)
            if not isinstance(token, str) or not token:
                raise HubRepositoryUnavailableError(
                    "Lakebase OAuth credential generation returned no token."
                )
            expires_at = self._timestamp(expires)
            if expires_at <= now + self._refresh_skew:
                raise HubRepositoryUnavailableError(
                    "Lakebase OAuth credential expires too soon for safe pooling."
                )
            self._current = _OAuthToken(value=token, expires_at=expires_at)
            return token

    @staticmethod
    def _timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return LakebaseOAuthTokenProvider._aware(value)
        converter = getattr(value, "ToDatetime", None)
        if callable(converter):
            try:
                converted = converter(tzinfo=UTC)
            except TypeError:
                converted = converter().replace(tzinfo=UTC)
            return LakebaseOAuthTokenProvider._aware(converted)
        raise HubRepositoryUnavailableError(
            "Lakebase OAuth credential returned no usable expiration time."
        )

    @staticmethod
    def _aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def _is_transient_connection_error(
    error: Exception,
    *,
    operational_error_type: type[Exception],
) -> bool:
    """Classify failures that can recover after scale-to-zero or a brief outage.

    PostgreSQL class 08 is the connection-exception family. The 57P0x states cover
    server shutdown/recovery windows. Driver-level network and DNS failures commonly
    have no SQLSTATE, so they receive the same small bounded retry. Authentication and
    database/configuration SQLSTATEs deliberately fail on the first attempt.
    """

    if not isinstance(error, operational_error_type):
        return False
    sqlstate = getattr(error, "sqlstate", None)
    return (
        sqlstate is None
        or str(sqlstate).startswith("08")
        or sqlstate
        in {
            "57P01",  # admin_shutdown
            "57P02",  # crash_shutdown
            "57P03",  # cannot_connect_now
        }
    )


def _connect_with_retry(
    connect: Callable[[], Any],
    *,
    is_transient: Callable[[Exception], bool],
    sleep: Callable[[float], None] = time.sleep,
    delays: tuple[float, ...] = _CONNECTION_RETRY_DELAYS,
) -> Any:
    """Retry only transient connection establishment failures, then preserve cause."""

    for delay in (*delays, None):
        try:
            return connect()
        except Exception as error:
            if delay is None or not is_transient(error):
                raise
            sleep(delay)
    raise AssertionError(
        "connection retry loop did not return or raise"
    )  # pragma: no cover


def create_lakebase_engine(
    settings: LakebaseRuntimeSettings,
    *,
    workspace_client: Any | None = None,
) -> Any:
    """Build a bounded SQLAlchemy pool whose new connections use fresh OAuth."""

    try:
        import psycopg
        from sqlalchemy import create_engine

        if workspace_client is None:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()
    except Exception:
        raise HubRepositoryUnavailableError(
            "Lakebase runtime dependencies or app authorization are unavailable."
        ) from None

    provider = LakebaseOAuthTokenProvider(
        lambda: workspace_client.postgres.generate_database_credential(
            endpoint=settings.endpoint
        )
    )

    def connect_once():
        # OAuth values exist only in this call. They are never placed in a URL,
        # exception, engine repr, log field, or environment variable.
        return psycopg.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.database,
            user=settings.user,
            password=provider.password(),
            sslmode=settings.sslmode,
            connect_timeout=settings.connect_timeout_seconds,
            application_name="aai-platform-hub",
            options=settings.connection_options,
        )

    def connect():
        return _connect_with_retry(
            connect_once,
            is_transient=lambda error: _is_transient_connection_error(
                error,
                operational_error_type=psycopg.OperationalError,
            ),
        )

    return create_engine(
        "postgresql+psycopg://",
        creator=connect,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout_seconds,
        pool_recycle=settings.pool_recycle_seconds,
        pool_pre_ping=True,
        future=True,
    )


class LakebaseHubRepository:
    """Transactional Lakebase implementation of :class:`HubRepository`."""

    _FLEET_ROLES = {
        Role.PLATFORM_VIEWER,
        Role.PLATFORM_ADMINISTRATOR,
        Role.AUDITOR,
    }
    _EVALUATION_TRANSITIONS = {
        EvaluationStatus.REQUESTED: {
            EvaluationStatus.QUEUED,
            EvaluationStatus.RUNNING,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        },
        EvaluationStatus.QUEUED: {
            EvaluationStatus.RUNNING,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        },
        EvaluationStatus.RUNNING: {
            EvaluationStatus.SUCCEEDED,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        },
        EvaluationStatus.SUCCEEDED: set(),
        EvaluationStatus.FAILED: set(),
        EvaluationStatus.CANCELLED: set(),
    }
    _PROMOTION_TRANSITIONS = {
        PromotionStatus.PENDING_REVIEW: {
            PromotionStatus.CHANGES_REQUESTED,
            PromotionStatus.REJECTED,
            PromotionStatus.APPROVED,
            PromotionStatus.CANCELLED,
        },
        PromotionStatus.CHANGES_REQUESTED: {
            PromotionStatus.PENDING_REVIEW,
            PromotionStatus.CANCELLED,
        },
        PromotionStatus.APPROVED: {
            PromotionStatus.EXECUTING,
            PromotionStatus.FAILED,
            PromotionStatus.CANCELLED,
        },
        PromotionStatus.EXECUTING: {
            PromotionStatus.SUCCEEDED,
            PromotionStatus.FAILED,
        },
        PromotionStatus.REJECTED: set(),
        PromotionStatus.SUCCEEDED: set(),
        PromotionStatus.FAILED: set(),
        PromotionStatus.CANCELLED: set(),
    }

    def __init__(
        self,
        engine: Any,
        *,
        schema: str,
        migrate: bool = True,
    ) -> None:
        if not _POSTGRES_IDENTIFIER.fullmatch(schema):
            raise LakebaseConfigurationError("invalid Lakebase schema name")
        self._engine = engine
        self._schema_name = schema
        self._schema = f'"{schema}"'
        if migrate:
            self.migrate()

    @classmethod
    def from_runtime(
        cls,
        settings: LakebaseRuntimeSettings,
        *,
        workspace_client: Any | None = None,
    ) -> LakebaseHubRepository:
        return cls(
            create_lakebase_engine(settings, workspace_client=workspace_client),
            schema=settings.schema,
        )

    @property
    def available(self) -> bool:
        return True

    def close(self) -> None:
        self._engine.dispose()

    def migrate(self) -> None:
        migration_table = self._table("hub_schema_migrations")
        with self._transaction() as connection:
            self._locks(connection, f"schema-migration:{self._schema_name}")
            self._execute(
                connection,
                f"CREATE SCHEMA IF NOT EXISTS {self._schema}",
            )
            owner = (
                self._execute(
                    connection,
                    """
                SELECT
                    pg_get_userbyid(nspowner) AS schema_owner,
                    current_user AS current_role
                FROM pg_namespace
                WHERE nspname = :schema
                """,
                    {"schema": self._schema_name},
                )
                .mappings()
                .one_or_none()
            )
            if owner is None or owner["schema_owner"] != owner["current_role"]:
                raise HubRepositoryUnavailableError(
                    "The Hub schema must be created and owned by the app service "
                    "principal."
                )
            self._execute(
                connection,
                f"""
                CREATE TABLE IF NOT EXISTS {migration_table} (
                    version INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    checksum CHAR(64) NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """,
            )
            applied = {
                int(row["version"]): str(row["checksum"])
                for row in self._execute(
                    connection,
                    f"SELECT version, checksum FROM {migration_table} ORDER BY version",
                ).mappings()
            }
            known = {migration.version for migration in MIGRATIONS}
            if set(applied).difference(known):
                raise HubRepositoryUnavailableError(
                    "Lakebase Hub schema is newer than this application release."
                )
            for migration in MIGRATIONS:
                checksum = applied.get(migration.version)
                if checksum is not None:
                    if checksum != migration.checksum:
                        raise HubRepositoryUnavailableError(
                            "Lakebase Hub migration history failed checksum validation."
                        )
                    continue
                expected_previous = set(range(1, migration.version))
                if not expected_previous.issubset(applied):
                    raise HubRepositoryUnavailableError(
                        "Lakebase Hub migration history contains a gap."
                    )
                for statement in migration.render(self._schema):
                    self._execute(connection, statement)
                self._execute(
                    connection,
                    f"""
                    INSERT INTO {migration_table}
                        (version, description, checksum)
                    VALUES (:version, :description, :checksum)
                    """,
                    {
                        "version": migration.version,
                        "description": migration.description,
                        "checksum": migration.checksum,
                    },
                )
                applied[migration.version] = migration.checksum

    def register_application(
        self,
        application: ApplicationRecord,
        version: ApplicationVersionRecord,
        *,
        actor_request_id: str | None = None,
    ) -> RegistrationResult:
        if version.application_id != application.application_id:
            raise HubConflictError("application and version IDs do not match")
        if application.row_version != 1:
            raise HubConflictError("new registration input must have row_version 1")

        versions = self._table("application_versions")
        applications = self._table("applications")
        with self._transaction() as connection:
            self._locks(
                connection,
                f"application:{application.application_id}",
                f"application-version:{version.version_id}",
            )
            replay = (
                self._execute(
                    connection,
                    f"""
                SELECT version_id, application_id, environment, git_repository,
                       git_commit_sha_normalized, manifest_hash, deployment_target,
                       registered_at, record, is_current
                FROM {versions}
                WHERE application_id = :application_id
                  AND environment = :environment
                  AND git_commit_sha_normalized = :git_sha
                  AND manifest_hash = :manifest_hash
                """,
                    {
                        "application_id": version.application_id,
                        "environment": version.environment,
                        "git_sha": version.git_commit_sha.lower(),
                        "manifest_hash": version.manifest_hash,
                    },
                )
                .mappings()
                .one_or_none()
            )
            if replay is not None:
                stored_version = self._version(replay)
                if stored_version.deployment_target != version.deployment_target:
                    raise HubConflictError(
                        "an idempotent registration cannot change deployment_target"
                    )
                stored_application = self._get_application(
                    connection, application.application_id
                )
                return RegistrationResult(
                    application=stored_application,
                    version=stored_version,
                    created=False,
                )

            if (
                self._execute(
                    connection,
                    f"SELECT 1 FROM {versions} WHERE version_id = :version_id",
                    {"version_id": version.version_id},
                ).first()
                is not None
            ):
                raise HubConflictError(
                    f"version ID {version.version_id!r} already describes "
                    "another version"
                )

            current_application_row = (
                self._execute(
                    connection,
                    f"""
                SELECT application_id, record, row_version, updated_at
                FROM {applications}
                WHERE application_id = :application_id
                FOR UPDATE
                """,
                    {"application_id": application.application_id},
                )
                .mappings()
                .one_or_none()
            )
            if current_application_row is None:
                stored_application = application
                self._execute(
                    connection,
                    f"""
                    INSERT INTO {applications}
                        (application_id, row_version, updated_at, record)
                    VALUES
                        (:application_id, :row_version, :updated_at,
                         CAST(:record AS JSONB))
                    """,
                    self._application_params(stored_application),
                )
            else:
                existing_application = self._application(current_application_row)
                repositories = {
                    row["git_repository"]
                    for row in self._execute(
                        connection,
                        f"""
                        SELECT DISTINCT git_repository
                        FROM {versions}
                        WHERE application_id = :application_id
                        """,
                        {"application_id": application.application_id},
                    ).mappings()
                }
                if repositories and version.git_repository not in repositories:
                    raise ImmutableApplicationIdConflictError(
                        f"application ID {application.application_id!r} is "
                        "already bound to another Git repository"
                    )
                stored_application = _merge_registered_application(
                    application,
                    existing_application,
                    version,
                )
                self._execute(
                    connection,
                    f"""
                    UPDATE {applications}
                    SET row_version = :row_version,
                        updated_at = :updated_at,
                        record = CAST(:record AS JSONB)
                    WHERE application_id = :application_id
                    """,
                    self._application_params(stored_application),
                )

            previous = (
                self._execute(
                    connection,
                    f"""
                SELECT version_id
                FROM {versions}
                WHERE application_id = :application_id
                  AND environment = :environment
                  AND is_current
                FOR UPDATE
                """,
                    {
                        "application_id": version.application_id,
                        "environment": version.environment,
                    },
                )
                .mappings()
                .one_or_none()
            )
            previous_id = None if previous is None else str(previous["version_id"])
            if previous_id is not None:
                self._execute(
                    connection,
                    f"UPDATE {versions} SET is_current = FALSE WHERE version_id = :id",
                    {"id": previous_id},
                )

            stored_version = version.model_copy(update={"is_current": True})
            self._execute(
                connection,
                f"""
                INSERT INTO {versions} (
                    version_id,
                    application_id,
                    environment,
                    git_repository,
                    git_commit_sha_normalized,
                    manifest_hash,
                    deployment_target,
                    registered_at,
                    is_current,
                    record
                ) VALUES (
                    :version_id,
                    :application_id,
                    :environment,
                    :git_repository,
                    :git_sha,
                    :manifest_hash,
                    :deployment_target,
                    :registered_at,
                    TRUE,
                    CAST(:record AS JSONB)
                )
                """,
                {
                    "version_id": stored_version.version_id,
                    "application_id": stored_version.application_id,
                    "environment": stored_version.environment,
                    "git_repository": stored_version.git_repository,
                    "git_sha": stored_version.git_commit_sha.lower(),
                    "manifest_hash": stored_version.manifest_hash,
                    "deployment_target": stored_version.deployment_target,
                    "registered_at": stored_version.registered_at,
                    "record": self._json(stored_version),
                },
            )
            self._insert_event(
                connection,
                ActionEvent(
                    event_id=f"registration:{stored_version.version_id}",
                    entity_type=ActionEntityType.APPLICATION,
                    entity_id=application.application_id,
                    event_type=ActionEventType.APPLICATION_REGISTERED,
                    actor_principal=stored_version.registered_by,
                    actor_request_id=(
                        actor_request_id or f"registration:{stored_version.version_id}"
                    ),
                    event_time=stored_version.registered_at,
                    previous_state=previous_id,
                    new_state=stored_version.version_id,
                    details_json=json.dumps(
                        {
                            "environment": stored_version.environment,
                            "manifest_hash": stored_version.manifest_hash,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            return RegistrationResult(
                application=stored_application,
                version=stored_version,
                created=True,
            )

    def get_application(self, application_id: str) -> ApplicationRecord:
        with self._transaction() as connection:
            return self._get_application(connection, application_id)

    def get_visible_application(
        self,
        application_id: str,
        actor: AuthorizationContext,
    ) -> ApplicationRecord:
        with self._transaction() as connection:
            visible = {
                item.application_id: item
                for item in self._visible_applications(connection, actor)
            }
            try:
                return visible[application_id]
            except KeyError as error:
                raise HubNotFoundError(
                    f"application {application_id!r} was not found"
                ) from error

    def get_current_version(
        self,
        application_id: str,
        environment: str,
    ) -> ApplicationVersionRecord:
        with self._transaction() as connection:
            row = (
                self._execute(
                    connection,
                    f"""
                SELECT version_id, application_id, environment, git_repository,
                       git_commit_sha_normalized, manifest_hash, deployment_target,
                       registered_at, record, is_current
                FROM {self._table('application_versions')}
                WHERE application_id = :application_id
                  AND environment = :environment
                  AND is_current
                """,
                    {"application_id": application_id, "environment": environment},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise HubNotFoundError(
                    f"no current {environment!r} version exists for {application_id!r}"
                )
            return self._version(row)

    def list_versions(
        self,
        application_id: str,
    ) -> tuple[ApplicationVersionRecord, ...]:
        with self._transaction() as connection:
            self._require_application(connection, application_id)
            rows = self._execute(
                connection,
                f"""
                SELECT version_id, application_id, environment, git_repository,
                       git_commit_sha_normalized, manifest_hash, deployment_target,
                       registered_at, record, is_current
                FROM {self._table('application_versions')}
                WHERE application_id = :application_id
                ORDER BY registered_at, version_id
                """,
                {"application_id": application_id},
            ).mappings()
            return tuple(self._version(row) for row in rows)

    def upsert_application_principal(
        self,
        principal: ApplicationPrincipalRecord,
    ) -> ApplicationPrincipalRecord:
        with self._transaction() as connection:
            self._locks(
                connection,
                f"application-principals:{principal.application_id}",
            )
            self._require_application(connection, principal.application_id)
            self._execute(
                connection,
                f"""
                INSERT INTO {self._table('application_principals')} (
                    application_id,
                    principal_type,
                    principal_name_normalized,
                    record
                ) VALUES (
                    :application_id,
                    :principal_type,
                    :principal_name,
                    CAST(:record AS JSONB)
                )
                ON CONFLICT (
                    application_id,
                    principal_type,
                    principal_name_normalized
                ) DO UPDATE SET record = EXCLUDED.record
                """,
                {
                    "application_id": principal.application_id,
                    "principal_type": principal.principal_type.value,
                    "principal_name": principal.principal_name.casefold(),
                    "record": self._json(principal),
                },
            )
            return principal

    def replace_application_principals(
        self,
        application_id: str,
        principals: Iterable[ApplicationPrincipalRecord],
    ) -> tuple[ApplicationPrincipalRecord, ...]:
        replacement = tuple(principals)
        keys: set[tuple[PrincipalType, str]] = set()
        for principal in replacement:
            if principal.application_id != application_id:
                raise HubConflictError("principal belongs to another application")
            key = (principal.principal_type, principal.principal_name.casefold())
            if key in keys:
                raise HubConflictError("application principal appears more than once")
            keys.add(key)
        with self._transaction() as connection:
            self._locks(connection, f"application-principals:{application_id}")
            self._require_application(connection, application_id)
            self._execute(
                connection,
                f"""
                DELETE FROM {self._table('application_principals')}
                WHERE application_id = :application_id
                """,
                {"application_id": application_id},
            )
            for principal in replacement:
                self._execute(
                    connection,
                    f"""
                    INSERT INTO {self._table('application_principals')} (
                        application_id,
                        principal_type,
                        principal_name_normalized,
                        record
                    ) VALUES (
                        :application_id,
                        :principal_type,
                        :principal_name,
                        CAST(:record AS JSONB)
                    )
                    """,
                    {
                        "application_id": principal.application_id,
                        "principal_type": principal.principal_type.value,
                        "principal_name": principal.principal_name.casefold(),
                        "record": self._json(principal),
                    },
                )
            return replacement

    def list_application_principals(
        self,
        application_id: str,
    ) -> tuple[ApplicationPrincipalRecord, ...]:
        with self._transaction() as connection:
            self._require_application(connection, application_id)
            return self._principals(connection, application_id)

    def list_visible_applications(
        self,
        actor: AuthorizationContext,
    ) -> tuple[ApplicationRecord, ...]:
        with self._transaction() as connection:
            return self._visible_applications(connection, actor)

    def query_visible_applications(
        self,
        actor: AuthorizationContext,
        *,
        search: str,
        lifecycle: str | None,
        ownership: str,
        tag_filters: Mapping[str, frozenset[str]],
        sort: str,
        page: int,
        page_size: int,
    ) -> tuple[tuple[ApplicationRecord, ...], int]:
        with self._transaction() as connection:
            visible = self._visible_applications(connection, actor)
            principals = self._all_principals(connection)

        needle = search.strip().casefold()
        actor_groups = {group.casefold() for group in actor.groups}
        actor_name = actor.principal.casefold()
        filtered: list[ApplicationRecord] = []
        for application in visible:
            if needle and needle not in (
                f"{application.application_id} {application.name}".casefold()
            ):
                continue
            if lifecycle and application.lifecycle_state != lifecycle:
                continue
            app_principals = tuple(
                principal
                for principal in principals
                if principal.application_id == application.application_id
            )
            if ownership == "owned" and not any(
                principal.application_role is Role.OWNER
                and self._principal_matches(principal, actor_name, actor_groups)
                for principal in app_principals
            ):
                continue
            if ownership == "teams" and not any(
                principal.principal_type is PrincipalType.GROUP
                and principal.principal_name.casefold() in actor_groups
                for principal in app_principals
            ):
                continue
            tags = {tag.key: tag.value for tag in application.tags}
            if any(
                tags.get(key) not in accepted for key, accepted in tag_filters.items()
            ):
                continue
            filtered.append(application)

        reverse = sort.startswith("-")
        field = sort.removeprefix("-")
        sort_key = {
            "application": lambda item: (item.name.casefold(), item.application_id),
            "updated_at": lambda item: (item.updated_at, item.application_id),
            "owner": lambda item: (
                item.owner_principal.casefold(),
                item.application_id,
            ),
        }[field]
        filtered.sort(key=sort_key, reverse=reverse)
        total = len(filtered)
        start = (page - 1) * page_size
        return tuple(filtered[start : start + page_size]), total

    def upsert_resource_binding(
        self,
        binding: ResourceBindingRecord,
    ) -> ResourceBindingRecord:
        with self._transaction() as connection:
            self._require_application(connection, binding.application_id)
            self._execute(
                connection,
                f"""
                INSERT INTO {self._table('resource_bindings')} (
                    binding_id, application_id, environment, record
                ) VALUES (
                    :binding_id, :application_id, :environment, CAST(:record AS JSONB)
                )
                ON CONFLICT (binding_id) DO UPDATE SET
                    application_id = EXCLUDED.application_id,
                    environment = EXCLUDED.environment,
                    record = EXCLUDED.record
                """,
                {
                    "binding_id": binding.binding_id,
                    "application_id": binding.application_id,
                    "environment": binding.environment,
                    "record": self._json(binding),
                },
            )
            return binding

    def list_resource_bindings(
        self,
        application_id: str,
        *,
        environment: str | None = None,
    ) -> tuple[ResourceBindingRecord, ...]:
        with self._transaction() as connection:
            self._require_application(connection, application_id)
            sql = f"""
                SELECT binding_id, application_id, environment, record
                FROM {self._table('resource_bindings')}
                WHERE application_id = :application_id
            """
            params: dict[str, Any] = {"application_id": application_id}
            if environment is not None:
                sql += " AND environment = :environment"
                params["environment"] = environment
            sql += " ORDER BY binding_id"
            return tuple(
                self._binding(row)
                for row in self._execute(connection, sql, params).mappings()
            )

    def create_evaluation(
        self,
        evaluation: EvaluationRunRecord,
        *,
        actor_request_id: str | None = None,
    ) -> EvaluationRunRecord:
        if evaluation.status is not EvaluationStatus.REQUESTED:
            raise HubConflictError("new evaluations must start in REQUESTED")
        if evaluation.row_version != 1:
            raise HubConflictError("new evaluations must have row_version 1")
        table = self._table("evaluation_runs")
        with self._transaction() as connection:
            self._locks(
                connection,
                f"evaluation:{evaluation.evaluation_run_id}",
                "evaluation-active:"
                f"{evaluation.application_id}:{evaluation.environment}:"
                f"{evaluation.application_version_id}",
            )
            if (
                self._execute(
                    connection,
                    f"SELECT 1 FROM {table} WHERE evaluation_run_id = :id",
                    {"id": evaluation.evaluation_run_id},
                ).first()
                is not None
            ):
                raise HubConflictError(
                    f"evaluation ID {evaluation.evaluation_run_id!r} already exists"
                )
            self._validate_application_version(
                connection,
                evaluation.application_id,
                evaluation.application_version_id,
                environment=evaluation.environment,
            )
            duplicate = (
                self._execute(
                    connection,
                    f"""
                SELECT evaluation_run_id
                FROM {table}
                WHERE application_id = :application_id
                  AND environment = :environment
                  AND application_version_id = :version_id
                  AND status IN ('REQUESTED', 'QUEUED', 'RUNNING')
                """,
                    {
                        "application_id": evaluation.application_id,
                        "environment": evaluation.environment,
                        "version_id": evaluation.application_version_id,
                    },
                )
                .mappings()
                .one_or_none()
            )
            if duplicate is not None:
                raise DuplicateActiveEvaluationError(
                    f"evaluation {duplicate['evaluation_run_id']!r} is already active"
                )
            self._execute(
                connection,
                f"""
                INSERT INTO {table} (
                    evaluation_run_id,
                    application_id,
                    environment,
                    application_version_id,
                    status,
                    requested_at,
                    row_version,
                    record
                ) VALUES (
                    :id, :application_id, :environment, :version_id, :status,
                    :requested_at, :row_version, CAST(:record AS JSONB)
                )
                """,
                self._evaluation_params(evaluation),
            )
            self._insert_event(
                connection,
                ActionEvent(
                    event_id=f"evaluation-requested:{evaluation.evaluation_run_id}",
                    entity_type=ActionEntityType.EVALUATION,
                    entity_id=evaluation.evaluation_run_id,
                    event_type=ActionEventType.EVALUATION_REQUESTED,
                    actor_principal=evaluation.requested_by,
                    actor_request_id=(
                        actor_request_id or f"evaluation:{evaluation.evaluation_run_id}"
                    ),
                    event_time=evaluation.requested_at,
                    previous_state=None,
                    new_state=evaluation.status.value,
                ),
            )
            return evaluation

    def update_evaluation(
        self,
        evaluation: EvaluationRunRecord,
        *,
        expected_row_version: int,
        actor_request_id: str | None = None,
    ) -> EvaluationRunRecord:
        table = self._table("evaluation_runs")
        with self._transaction() as connection:
            current = self._get_evaluation(
                connection,
                evaluation.evaluation_run_id,
                for_update=True,
            )
            self._assert_row_version(current.row_version, expected_row_version)
            self._assert_evaluation_identity(current, evaluation)
            if evaluation.row_version != current.row_version + 1:
                raise HubConflictError(
                    "updated evaluation must increment row_version exactly once"
                )
            if (
                evaluation.status is not current.status
                and evaluation.status
                not in self._EVALUATION_TRANSITIONS[current.status]
            ):
                raise InvalidStateTransitionError(
                    f"cannot transition evaluation from {current.status.value} "
                    f"to {evaluation.status.value}"
                )
            result = self._execute(
                connection,
                f"""
                UPDATE {table}
                SET status = :status,
                    row_version = :row_version,
                    record = CAST(:record AS JSONB)
                WHERE evaluation_run_id = :id
                  AND row_version = :expected_row_version
                """,
                {
                    **self._evaluation_params(evaluation),
                    "expected_row_version": expected_row_version,
                },
            )
            if result.rowcount != 1:
                actual = self._get_evaluation(
                    connection, evaluation.evaluation_run_id
                ).row_version
                raise OptimisticConcurrencyError(
                    expected=expected_row_version,
                    actual=actual,
                )
            self._insert_event(
                connection,
                ActionEvent(
                    event_id=f"evaluation-update:{uuid4()}",
                    entity_type=ActionEntityType.EVALUATION,
                    entity_id=evaluation.evaluation_run_id,
                    event_type=ActionEventType.EVALUATION_STATUS_CHANGED,
                    actor_principal=evaluation.requested_by,
                    actor_request_id=(
                        actor_request_id or f"evaluation:{evaluation.evaluation_run_id}"
                    ),
                    event_time=(
                        evaluation.completed_at
                        or evaluation.started_at
                        or evaluation.requested_at
                    ),
                    previous_state=current.status.value,
                    new_state=evaluation.status.value,
                    details_json=json.dumps(
                        (
                            {}
                            if evaluation.job_run_id is None
                            else {"job_run_id": evaluation.job_run_id}
                        ),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            return evaluation

    def get_evaluation(self, evaluation_run_id: str) -> EvaluationRunRecord:
        with self._transaction() as connection:
            return self._get_evaluation(connection, evaluation_run_id)

    def list_evaluations(
        self,
        application_id: str,
    ) -> tuple[EvaluationRunRecord, ...]:
        with self._transaction() as connection:
            self._require_application(connection, application_id)
            return tuple(
                self._evaluation(row)
                for row in self._execute(
                    connection,
                    f"""
                    SELECT evaluation_run_id, application_id, environment,
                           application_version_id, status, requested_at,
                           row_version, record
                    FROM {self._table('evaluation_runs')}
                    WHERE application_id = :application_id
                    ORDER BY requested_at, evaluation_run_id
                    """,
                    {"application_id": application_id},
                ).mappings()
            )

    def create_promotion_request(
        self,
        request: PromotionRequestRecord,
        *,
        actor_request_id: str | None = None,
    ) -> PromotionRequestRecord:
        if request.status is not PromotionStatus.PENDING_REVIEW:
            raise HubConflictError("new promotions must start in PENDING_REVIEW")
        if request.row_version != 1:
            raise HubConflictError("new promotions must have row_version 1")
        if not request.readiness_snapshot.ready:
            raise HubConflictError(
                "a blocked version cannot request environment promotion"
            )
        table = self._table("promotion_requests")
        with self._transaction() as connection:
            self._locks(
                connection,
                f"promotion:{request.promotion_request_id}",
                "promotion-active:"
                f"{request.application_id}:{request.application_version_id}:"
                f"{request.target_environment}",
            )
            if (
                self._execute(
                    connection,
                    f"SELECT 1 FROM {table} WHERE promotion_request_id = :id",
                    {"id": request.promotion_request_id},
                ).first()
                is not None
            ):
                raise HubConflictError(
                    f"promotion ID {request.promotion_request_id!r} already exists"
                )
            self._validate_application_version(
                connection,
                request.application_id,
                request.application_version_id,
                environment=request.source_environment,
            )
            duplicate = (
                self._execute(
                    connection,
                    f"""
                SELECT promotion_request_id
                FROM {table}
                WHERE application_id = :application_id
                  AND application_version_id = :version_id
                  AND target_environment = :target_environment
                  AND status IN (
                      'PENDING_REVIEW',
                      'CHANGES_REQUESTED',
                      'APPROVED',
                      'EXECUTING'
                  )
                """,
                    {
                        "application_id": request.application_id,
                        "version_id": request.application_version_id,
                        "target_environment": request.target_environment,
                    },
                )
                .mappings()
                .one_or_none()
            )
            if duplicate is not None:
                raise DuplicateActivePromotionError(
                    f"promotion {duplicate['promotion_request_id']!r} is already active"
                )
            self._execute(
                connection,
                f"""
                INSERT INTO {table} (
                    promotion_request_id,
                    application_id,
                    application_version_id,
                    target_environment,
                    status,
                    requested_at,
                    row_version,
                    record
                ) VALUES (
                    :id, :application_id, :version_id, :target_environment,
                    :status, :requested_at, :row_version, CAST(:record AS JSONB)
                )
                """,
                self._promotion_params(request),
            )
            self._insert_event(
                connection,
                ActionEvent(
                    event_id=f"promotion-requested:{request.promotion_request_id}",
                    entity_type=ActionEntityType.PROMOTION,
                    entity_id=request.promotion_request_id,
                    event_type=ActionEventType.PROMOTION_REQUESTED,
                    actor_principal=request.requested_by,
                    actor_request_id=(
                        actor_request_id or f"promotion:{request.promotion_request_id}"
                    ),
                    event_time=request.requested_at,
                    previous_state=None,
                    new_state=request.status.value,
                ),
            )
            return request

    def get_promotion_request(
        self,
        promotion_request_id: str,
    ) -> PromotionRequestRecord:
        with self._transaction() as connection:
            return self._get_promotion(connection, promotion_request_id)

    def list_promotion_requests(
        self,
        application_id: str,
    ) -> tuple[PromotionRequestRecord, ...]:
        with self._transaction() as connection:
            self._require_application(connection, application_id)
            return tuple(
                self._promotion(row)
                for row in self._execute(
                    connection,
                    f"""
                    SELECT promotion_request_id, application_id,
                           application_version_id, target_environment, status,
                           requested_at, row_version, record
                    FROM {self._table('promotion_requests')}
                    WHERE application_id = :application_id
                    ORDER BY requested_at, promotion_request_id
                    """,
                    {"application_id": application_id},
                ).mappings()
            )

    def approve_promotion(
        self,
        promotion_request_id: str,
        *,
        actor: AuthorizationContext,
        expected_row_version: int,
        reviewed_at: datetime,
        actor_request_id: str,
        readiness_snapshot: ReadinessSnapshot | None = None,
        comment: str | None = None,
    ) -> PromotionRequestRecord:
        self._require_administrator(actor)
        with self._transaction() as connection:
            current = self._get_promotion(
                connection, promotion_request_id, for_update=True
            )
            self._assert_row_version(current.row_version, expected_row_version)
            if current.status is not PromotionStatus.PENDING_REVIEW:
                raise InvalidStateTransitionError(
                    f"cannot approve a promotion in {current.status.value}"
                )
            if current.requested_by.casefold() == actor.principal.casefold():
                raise FourEyesViolationError(
                    "a requester cannot approve their own environment promotion"
                )
            current_readiness = readiness_snapshot or current.readiness_snapshot
            self._validate_readiness_for_promotion(current, current_readiness)
            updated = current.model_copy(
                update={
                    "status": PromotionStatus.APPROVED,
                    "approval_readiness_snapshot": current_readiness,
                    "reviewed_by": actor.principal,
                    "reviewed_at": reviewed_at,
                    "review_comment": comment,
                    "row_version": current.row_version + 1,
                }
            )
            updated = PromotionRequestRecord.model_validate(
                updated.model_dump(mode="python")
            )
            self._store_promotion(
                connection,
                updated,
                expected_row_version=expected_row_version,
            )
            self._insert_event(
                connection,
                self._promotion_event(
                    current,
                    updated,
                    event_type=ActionEventType.PROMOTION_APPROVED,
                    actor_principal=actor.principal,
                    actor_request_id=actor_request_id,
                    event_time=reviewed_at,
                    comment=comment,
                ),
            )
            return updated

    def reject_promotion(
        self,
        promotion_request_id: str,
        *,
        actor: AuthorizationContext,
        expected_row_version: int,
        reviewed_at: datetime,
        actor_request_id: str,
        comment: str,
    ) -> PromotionRequestRecord:
        return self._review_promotion(
            promotion_request_id,
            actor=actor,
            expected_row_version=expected_row_version,
            reviewed_at=reviewed_at,
            actor_request_id=actor_request_id,
            comment=comment,
            status=PromotionStatus.REJECTED,
            event_type=ActionEventType.PROMOTION_REJECTED,
        )

    def request_promotion_changes(
        self,
        promotion_request_id: str,
        *,
        actor: AuthorizationContext,
        expected_row_version: int,
        reviewed_at: datetime,
        actor_request_id: str,
        comment: str,
    ) -> PromotionRequestRecord:
        return self._review_promotion(
            promotion_request_id,
            actor=actor,
            expected_row_version=expected_row_version,
            reviewed_at=reviewed_at,
            actor_request_id=actor_request_id,
            comment=comment,
            status=PromotionStatus.CHANGES_REQUESTED,
            event_type=ActionEventType.PROMOTION_CHANGES_REQUESTED,
        )

    def update_promotion(
        self,
        request: PromotionRequestRecord,
        *,
        expected_row_version: int,
        actor_principal: str,
        actor_request_id: str,
        event_time: datetime,
        comment: str | None = None,
    ) -> PromotionRequestRecord:
        with self._transaction() as connection:
            current = self._get_promotion(
                connection, request.promotion_request_id, for_update=True
            )
            self._assert_row_version(current.row_version, expected_row_version)
            self._assert_promotion_identity(current, request)
            if request.row_version != current.row_version + 1:
                raise HubConflictError(
                    "updated promotion must increment row_version exactly once"
                )
            if request.status in {
                PromotionStatus.APPROVED,
                PromotionStatus.REJECTED,
                PromotionStatus.CHANGES_REQUESTED,
            }:
                raise HubAuthorizationError(
                    "review transitions require the dedicated administrator methods"
                )
            if request.status not in self._PROMOTION_TRANSITIONS[current.status]:
                raise InvalidStateTransitionError(
                    f"cannot transition promotion from {current.status.value} "
                    f"to {request.status.value}"
                )
            event_type = {
                PromotionStatus.EXECUTING: ActionEventType.PROMOTION_EXECUTION_STARTED,
                PromotionStatus.SUCCEEDED: ActionEventType.PROMOTION_SUCCEEDED,
                PromotionStatus.FAILED: ActionEventType.PROMOTION_FAILED,
                PromotionStatus.CANCELLED: ActionEventType.PROMOTION_CANCELLED,
                PromotionStatus.PENDING_REVIEW: ActionEventType.PROMOTION_REQUESTED,
            }[request.status]
            self._store_promotion(
                connection,
                request,
                expected_row_version=expected_row_version,
            )
            self._insert_event(
                connection,
                self._promotion_event(
                    current,
                    request,
                    event_type=event_type,
                    actor_principal=actor_principal,
                    actor_request_id=actor_request_id,
                    event_time=event_time,
                    comment=comment,
                ),
            )
            return request

    def append_action_event(self, event: ActionEvent) -> ActionEvent:
        with self._transaction() as connection:
            self._locks(connection, f"action-event:{event.event_id}")
            if (
                self._execute(
                    connection,
                    f"SELECT 1 FROM {self._table('action_events')} "
                    "WHERE event_id = :id",
                    {"id": event.event_id},
                ).first()
                is not None
            ):
                raise HubConflictError(
                    f"action event {event.event_id!r} already exists"
                )
            self._insert_event(connection, event)
            return event

    def list_action_events(
        self,
        *,
        entity_type: ActionEntityType | None = None,
        entity_id: str | None = None,
    ) -> tuple[ActionEvent, ...]:
        clauses = []
        params: dict[str, Any] = {}
        if entity_type is not None:
            clauses.append("entity_type = :entity_type")
            params["entity_type"] = entity_type.value
        if entity_id is not None:
            clauses.append("entity_id = :entity_id")
            params["entity_id"] = entity_id
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        with self._transaction() as connection:
            return tuple(
                self._event(row)
                for row in self._execute(
                    connection,
                    f"""
                    SELECT event_id, entity_type, entity_id, event_time, record
                    FROM {self._table('action_events')}
                    {where}
                    ORDER BY event_time, event_id
                    """,
                    params,
                ).mappings()
            )

    def list_application_action_events(
        self,
        application_id: str,
    ) -> tuple[ActionEvent, ...]:
        with self._transaction() as connection:
            self._require_application(connection, application_id)
            return tuple(
                self._event(row)
                for row in self._execute(
                    connection,
                    f"""
                    SELECT event.event_id, event.entity_type, event.entity_id,
                           event.event_time, event.record
                    FROM {self._table('action_events')} AS event
                    WHERE (
                        event.entity_type = 'APPLICATION'
                        AND event.entity_id = :application_id
                    ) OR (
                        event.entity_type = 'EVALUATION'
                        AND event.entity_id IN (
                            SELECT evaluation_run_id
                            FROM {self._table('evaluation_runs')}
                            WHERE application_id = :application_id
                        )
                    ) OR (
                        event.entity_type = 'PROMOTION'
                        AND event.entity_id IN (
                            SELECT promotion_request_id
                            FROM {self._table('promotion_requests')}
                            WHERE application_id = :application_id
                        )
                    )
                    ORDER BY event.event_time, event.event_id
                    """,
                    {"application_id": application_id},
                ).mappings()
            )

    def _review_promotion(
        self,
        promotion_request_id: str,
        *,
        actor: AuthorizationContext,
        expected_row_version: int,
        reviewed_at: datetime,
        actor_request_id: str,
        comment: str,
        status: PromotionStatus,
        event_type: ActionEventType,
    ) -> PromotionRequestRecord:
        self._require_administrator(actor)
        if not comment.strip():
            raise HubConflictError("a review comment is required")
        with self._transaction() as connection:
            current = self._get_promotion(
                connection, promotion_request_id, for_update=True
            )
            self._assert_row_version(current.row_version, expected_row_version)
            if current.status is not PromotionStatus.PENDING_REVIEW:
                raise InvalidStateTransitionError(
                    f"cannot review a promotion in {current.status.value}"
                )
            updated = current.model_copy(
                update={
                    "status": status,
                    "reviewed_by": actor.principal,
                    "reviewed_at": reviewed_at,
                    "review_comment": comment,
                    "row_version": current.row_version + 1,
                }
            )
            updated = PromotionRequestRecord.model_validate(
                updated.model_dump(mode="python")
            )
            self._store_promotion(
                connection,
                updated,
                expected_row_version=expected_row_version,
            )
            self._insert_event(
                connection,
                self._promotion_event(
                    current,
                    updated,
                    event_type=event_type,
                    actor_principal=actor.principal,
                    actor_request_id=actor_request_id,
                    event_time=reviewed_at,
                    comment=comment,
                ),
            )
            return updated

    def _store_promotion(
        self,
        connection: Any,
        request: PromotionRequestRecord,
        *,
        expected_row_version: int,
    ) -> None:
        result = self._execute(
            connection,
            f"""
            UPDATE {self._table('promotion_requests')}
            SET status = :status,
                row_version = :row_version,
                record = CAST(:record AS JSONB)
            WHERE promotion_request_id = :id
              AND row_version = :expected_row_version
            """,
            {
                **self._promotion_params(request),
                "expected_row_version": expected_row_version,
            },
        )
        if result.rowcount != 1:
            actual = self._get_promotion(
                connection, request.promotion_request_id
            ).row_version
            raise OptimisticConcurrencyError(
                expected=expected_row_version,
                actual=actual,
            )

    def _get_application(
        self,
        connection: Any,
        application_id: str,
    ) -> ApplicationRecord:
        row = (
            self._execute(
                connection,
                f"""
            SELECT application_id, record, row_version, updated_at
            FROM {self._table('applications')}
            WHERE application_id = :application_id
            """,
                {"application_id": application_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise HubNotFoundError(f"application {application_id!r} was not found")
        return self._application(row)

    def _require_application(self, connection: Any, application_id: str) -> None:
        self._get_application(connection, application_id)

    def _validate_application_version(
        self,
        connection: Any,
        application_id: str,
        application_version_id: str,
        *,
        environment: str,
    ) -> None:
        self._require_application(connection, application_id)
        row = (
            self._execute(
                connection,
                f"""
            SELECT application_id, environment
            FROM {self._table('application_versions')}
            WHERE version_id = :version_id
            """,
                {"version_id": application_version_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise HubNotFoundError(
                f"application version {application_version_id!r} was not found"
            )
        if row["application_id"] != application_id or row["environment"] != environment:
            raise HubConflictError(
                "application version does not match the requested "
                "application/environment"
            )

    def _get_evaluation(
        self,
        connection: Any,
        evaluation_run_id: str,
        *,
        for_update: bool = False,
    ) -> EvaluationRunRecord:
        lock = " FOR UPDATE" if for_update else ""
        row = (
            self._execute(
                connection,
                f"""
            SELECT evaluation_run_id, application_id, environment,
                   application_version_id, status, requested_at,
                   row_version, record
            FROM {self._table('evaluation_runs')}
            WHERE evaluation_run_id = :id{lock}
            """,
                {"id": evaluation_run_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise HubNotFoundError(f"evaluation {evaluation_run_id!r} was not found")
        return self._evaluation(row)

    def _get_promotion(
        self,
        connection: Any,
        promotion_request_id: str,
        *,
        for_update: bool = False,
    ) -> PromotionRequestRecord:
        lock = " FOR UPDATE" if for_update else ""
        row = (
            self._execute(
                connection,
                f"""
            SELECT promotion_request_id, application_id, application_version_id,
                   target_environment, status, requested_at, row_version, record
            FROM {self._table('promotion_requests')}
            WHERE promotion_request_id = :id{lock}
            """,
                {"id": promotion_request_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise HubNotFoundError(f"promotion {promotion_request_id!r} was not found")
        return self._promotion(row)

    def _visible_applications(
        self,
        connection: Any,
        actor: AuthorizationContext,
    ) -> tuple[ApplicationRecord, ...]:
        applications = tuple(
            self._application(row)
            for row in self._execute(
                connection,
                f"""
                SELECT application_id, row_version, updated_at, record
                FROM {self._table('applications')}
                ORDER BY application_id
                """,
            ).mappings()
        )
        if set(actor.platform_roles).intersection(self._FLEET_ROLES):
            return applications
        actor_name = actor.principal.casefold()
        actor_groups = {group.casefold() for group in actor.groups}
        explicit = {
            principal.application_id
            for principal in self._all_principals(connection)
            if self._principal_matches(principal, actor_name, actor_groups)
        }
        return tuple(
            application
            for application in applications
            if application.application_id in explicit
        )

    def _all_principals(
        self,
        connection: Any,
    ) -> tuple[ApplicationPrincipalRecord, ...]:
        return tuple(
            self._principal(row)
            for row in self._execute(
                connection,
                f"""
                SELECT application_id, principal_type,
                       principal_name_normalized, record
                FROM {self._table('application_principals')}
                """,
            ).mappings()
        )

    def _principals(
        self,
        connection: Any,
        application_id: str,
    ) -> tuple[ApplicationPrincipalRecord, ...]:
        return tuple(
            self._principal(row)
            for row in self._execute(
                connection,
                f"""
                SELECT application_id, principal_type,
                       principal_name_normalized, record
                FROM {self._table('application_principals')}
                WHERE application_id = :application_id
                ORDER BY principal_type, principal_name_normalized
                """,
                {"application_id": application_id},
            ).mappings()
        )

    @staticmethod
    def _principal_matches(
        principal: ApplicationPrincipalRecord,
        actor_name: str,
        actor_groups: set[str],
    ) -> bool:
        name = principal.principal_name.casefold()
        return (
            principal.principal_type is PrincipalType.USER and name == actor_name
        ) or (principal.principal_type is PrincipalType.GROUP and name in actor_groups)

    def _insert_event(self, connection: Any, event: ActionEvent) -> None:
        self._execute(
            connection,
            f"""
            INSERT INTO {self._table('action_events')} (
                event_id, entity_type, entity_id, event_time, record
            ) VALUES (
                :event_id, :entity_type, :entity_id, :event_time,
                CAST(:record AS JSONB)
            )
            """,
            {
                "event_id": event.event_id,
                "entity_type": event.entity_type.value,
                "entity_id": event.entity_id,
                "event_time": event.event_time,
                "record": self._json(event),
            },
        )

    def _locks(self, connection: Any, *keys: str) -> None:
        for key in sorted(set(keys)):
            self._execute(
                connection,
                "SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))",
                {"key": f"aai-platform-hub:{self._schema_name}:{key}"},
            )

    @contextmanager
    def _transaction(self):
        try:
            with self._engine.begin() as connection:
                yield connection
        except HubRepositoryError:
            raise
        except Exception:
            # Database and driver messages can include connection coordinates or
            # provider payloads. Delivery adapters get only a stable safe category.
            raise HubRepositoryUnavailableError(
                "The configured Lakebase Hub store is unavailable."
            ) from None

    @staticmethod
    def _execute(
        connection: Any,
        sql: str,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        from sqlalchemy import text

        return connection.execute(text(sql), dict(params or {}))

    def _table(self, name: str) -> str:
        if not _POSTGRES_IDENTIFIER.fullmatch(name):
            raise LakebaseConfigurationError("invalid Hub table name")
        return f'{self._schema}."{name}"'

    @staticmethod
    def _json(model: Any) -> str:
        return model.model_dump_json(by_alias=False)

    @staticmethod
    def _model(model_type: Any, value: Any) -> Any:
        try:
            if isinstance(value, str):
                return model_type.model_validate_json(value)
            # psycopg decodes JSONB into ordinary Python dictionaries/lists. Strict
            # persisted models intentionally reject list -> tuple coercion in Python
            # mode, while JSON mode correctly restores JSON arrays to tuple fields.
            return model_type.model_validate_json(
                json.dumps(value, sort_keys=True, separators=(",", ":"))
            )
        except Exception:
            raise HubRepositoryUnavailableError(
                "Lakebase Hub state failed persisted-record validation."
            ) from None

    def _version(self, row: Mapping[str, Any]) -> ApplicationVersionRecord:
        version = self._model(ApplicationVersionRecord, row["record"])
        self._require_persisted_consistency(
            version.version_id == row["version_id"]
            and version.application_id == row["application_id"]
            and version.environment == row["environment"]
            and version.git_repository == row["git_repository"]
            and version.git_commit_sha.lower() == row["git_commit_sha_normalized"]
            and version.manifest_hash == row["manifest_hash"]
            and version.deployment_target == row["deployment_target"]
            and version.registered_at == row["registered_at"]
        )
        return version.model_copy(update={"is_current": bool(row["is_current"])})

    def _application(self, row: Mapping[str, Any]) -> ApplicationRecord:
        application = self._model(ApplicationRecord, row["record"])
        self._validate_row_version(application.row_version, row["row_version"])
        self._require_persisted_consistency(
            application.application_id == row["application_id"]
            and application.updated_at == row["updated_at"]
        )
        return application

    def _principal(self, row: Mapping[str, Any]) -> ApplicationPrincipalRecord:
        principal = self._model(ApplicationPrincipalRecord, row["record"])
        self._require_persisted_consistency(
            principal.application_id == row["application_id"]
            and principal.principal_type.value == row["principal_type"]
            and principal.principal_name.casefold() == row["principal_name_normalized"]
        )
        return principal

    def _binding(self, row: Mapping[str, Any]) -> ResourceBindingRecord:
        binding = self._model(ResourceBindingRecord, row["record"])
        self._require_persisted_consistency(
            binding.binding_id == row["binding_id"]
            and binding.application_id == row["application_id"]
            and binding.environment == row["environment"]
        )
        return binding

    def _evaluation(self, row: Mapping[str, Any]) -> EvaluationRunRecord:
        evaluation = self._model(EvaluationRunRecord, row["record"])
        self._validate_row_version(evaluation.row_version, row["row_version"])
        self._require_persisted_consistency(
            evaluation.evaluation_run_id == row["evaluation_run_id"]
            and evaluation.application_id == row["application_id"]
            and evaluation.environment == row["environment"]
            and evaluation.application_version_id == row["application_version_id"]
            and evaluation.status.value == row["status"]
            and evaluation.requested_at == row["requested_at"]
        )
        return evaluation

    def _promotion(self, row: Mapping[str, Any]) -> PromotionRequestRecord:
        promotion = self._model(PromotionRequestRecord, row["record"])
        self._validate_row_version(promotion.row_version, row["row_version"])
        self._require_persisted_consistency(
            promotion.promotion_request_id == row["promotion_request_id"]
            and promotion.application_id == row["application_id"]
            and promotion.application_version_id == row["application_version_id"]
            and promotion.target_environment == row["target_environment"]
            and promotion.status.value == row["status"]
            and promotion.requested_at == row["requested_at"]
        )
        return promotion

    def _event(self, row: Mapping[str, Any]) -> ActionEvent:
        event = self._model(ActionEvent, row["record"])
        self._require_persisted_consistency(
            event.event_id == row["event_id"]
            and event.entity_type.value == row["entity_type"]
            and event.entity_id == row["entity_id"]
            and event.event_time == row["event_time"]
        )
        return event

    @staticmethod
    def _require_persisted_consistency(condition: bool) -> None:
        if not condition:
            raise HubRepositoryUnavailableError(
                "Lakebase Hub relational and JSON state is internally inconsistent."
            )

    def _application_params(self, application: ApplicationRecord) -> dict[str, Any]:
        return {
            "application_id": application.application_id,
            "row_version": application.row_version,
            "updated_at": application.updated_at,
            "record": self._json(application),
        }

    def _evaluation_params(self, evaluation: EvaluationRunRecord) -> dict[str, Any]:
        return {
            "id": evaluation.evaluation_run_id,
            "application_id": evaluation.application_id,
            "environment": evaluation.environment,
            "version_id": evaluation.application_version_id,
            "status": evaluation.status.value,
            "requested_at": evaluation.requested_at,
            "row_version": evaluation.row_version,
            "record": self._json(evaluation),
        }

    def _promotion_params(self, request: PromotionRequestRecord) -> dict[str, Any]:
        return {
            "id": request.promotion_request_id,
            "application_id": request.application_id,
            "version_id": request.application_version_id,
            "target_environment": request.target_environment,
            "status": request.status.value,
            "requested_at": request.requested_at,
            "row_version": request.row_version,
            "record": self._json(request),
        }

    @staticmethod
    def _assert_row_version(actual: int, expected: int) -> None:
        if actual != expected:
            raise OptimisticConcurrencyError(expected=expected, actual=actual)

    @staticmethod
    def _validate_row_version(model_value: int, column_value: Any) -> None:
        if model_value != int(column_value):
            raise HubRepositoryUnavailableError(
                "Lakebase Hub row-version state is internally inconsistent."
            )

    @staticmethod
    def _assert_evaluation_identity(
        current: EvaluationRunRecord,
        updated: EvaluationRunRecord,
    ) -> None:
        immutable = (
            "application_id",
            "environment",
            "application_version_id",
            "evaluation_profile",
            "dataset_name",
            "dataset_version",
            "job_id",
            "requested_by",
            "requested_at",
        )
        if any(
            getattr(current, field) != getattr(updated, field) for field in immutable
        ):
            raise HubConflictError("immutable evaluation fields cannot change")

    @staticmethod
    def _assert_promotion_identity(
        current: PromotionRequestRecord,
        updated: PromotionRequestRecord,
    ) -> None:
        immutable = (
            "application_id",
            "source_environment",
            "target_environment",
            "application_version_id",
            "requested_by",
            "requested_at",
            "promotion_job_id",
        )
        if any(
            getattr(current, field) != getattr(updated, field) for field in immutable
        ):
            raise HubConflictError("immutable promotion fields cannot change")
        if current.readiness_snapshot != updated.readiness_snapshot:
            raise HubConflictError(
                "execution transitions cannot replace request-time readiness evidence"
            )
        if current.approval_readiness_snapshot != updated.approval_readiness_snapshot:
            raise HubConflictError(
                "execution transitions cannot replace approval readiness evidence"
            )

    @staticmethod
    def _require_administrator(actor: AuthorizationContext) -> None:
        if not actor.has_platform_role(Role.PLATFORM_ADMINISTRATOR):
            raise HubAuthorizationError("platform administrator role is required")

    @staticmethod
    def _validate_readiness_for_promotion(
        request: PromotionRequestRecord,
        snapshot: ReadinessSnapshot,
    ) -> None:
        if not snapshot.ready:
            raise HubConflictError("blocking readiness checks prevent approval")
        if snapshot.application_id != request.application_id:
            raise HubConflictError("readiness snapshot belongs to another application")
        if snapshot.application_version_id != request.application_version_id:
            raise HubConflictError("readiness snapshot belongs to another version")

    @staticmethod
    def _promotion_event(
        previous: PromotionRequestRecord,
        updated: PromotionRequestRecord,
        *,
        event_type: ActionEventType,
        actor_principal: str,
        actor_request_id: str,
        event_time: datetime,
        comment: str | None,
    ) -> ActionEvent:
        details: dict[str, Any] = {"row_version": updated.row_version}
        if updated.promotion_job_run_id is not None:
            details["promotion_job_run_id"] = updated.promotion_job_run_id
        if event_type is ActionEventType.PROMOTION_APPROVED:
            approval_snapshot = updated.approval_readiness_snapshot
            details.update(
                {
                    "request_readiness_evaluated_at": (
                        updated.readiness_snapshot.evaluated_at.isoformat()
                    ),
                    "approval_readiness_evaluated_at": (
                        None
                        if approval_snapshot is None
                        else approval_snapshot.evaluated_at.isoformat()
                    ),
                    "readiness_evidence_changed": (
                        approval_snapshot is not None
                        and approval_snapshot.decision_signature()
                        != updated.readiness_snapshot.decision_signature()
                    ),
                }
            )
        return ActionEvent(
            event_id=f"promotion-update:{uuid4()}",
            entity_type=ActionEntityType.PROMOTION,
            entity_id=updated.promotion_request_id,
            event_type=event_type,
            actor_principal=actor_principal,
            actor_request_id=actor_request_id,
            event_time=event_time,
            previous_state=previous.status.value,
            new_state=updated.status.value,
            comment=comment,
            details_json=json.dumps(details, sort_keys=True, separators=(",", ":")),
        )
