"""Credential-free contracts for the Hub's Lakebase Autoscaling adapter."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from aai_console.config import ConfigError, HubStateMode, load_config
from aai_console.hub.lakebase import (
    LakebaseConfigurationError,
    LakebaseHubRepository,
    LakebaseOAuthTokenProvider,
    LakebaseRuntimeSettings,
    _connect_with_retry,
    _is_transient_connection_error,
    create_lakebase_engine,
)
from aai_console.hub.lakebase_migrations import MIGRATIONS
from aai_console.hub.models import (
    ApplicationRecord,
    ApplicationVersionRecord,
    AuthorizationContext,
    EvaluationRunRecord,
    EvaluationStatus,
    Role,
    Tag,
)
from aai_console.hub.repository import (
    DuplicateActiveEvaluationError,
    HubRepository,
    HubRepositoryUnavailableError,
    OptimisticConcurrencyError,
)


def _environment(**updates: str) -> dict[str, str]:
    values = {
        "DATABRICKS_APP_NAME": "aai-platform-console-uat",
        "AAI_HUB_STATE_MODE": "lakebase",
        "AAI_HUB_LAKEBASE_SCHEMA": "aai_platform_hub",
        "PGHOST": "ep-example.database.cloud.databricks.com",
        "PGPORT": "5432",
        "PGDATABASE": "hub-uat",
        "PGUSER": "00000000-0000-0000-0000-000000000000",
        "PGSSLMODE": "require",
        "LAKEBASE_ENDPOINT": ("projects/aai-platform/branches/uat/endpoints/primary"),
    }
    values.update(updates)
    return values


def test_lakebase_mode_requires_the_complete_bound_resource_contract() -> None:
    loaded = load_config(_environment())
    assert loaded.hub_state_mode is HubStateMode.LAKEBASE
    assert loaded.hub_lakebase is not None
    assert loaded.hub_lakebase.endpoint.endswith("/endpoints/primary")
    assert loaded.hub_lakebase.schema == "aai_platform_hub"
    assert loaded.hub_lakebase.sslmode == "require"

    for missing in ("PGHOST", "PGDATABASE", "PGUSER", "LAKEBASE_ENDPOINT"):
        environment = _environment()
        environment.pop(missing)
        with pytest.raises(ConfigError, match=missing):
            load_config(environment)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("PGSSLMODE", "disable", "PGSSLMODE must be require"),
        ("PGPORT", "0", "PGPORT must be between"),
        (
            "LAKEBASE_ENDPOINT",
            "projects/aai-platform/branches/uat/databases/hub",
            "endpoint resource path",
        ),
        ("AAI_HUB_LAKEBASE_SCHEMA", "Public", "lowercase PostgreSQL"),
        (
            "AAI_HUB_LAKEBASE_STATEMENT_TIMEOUT_MS",
            "999",
            "STATEMENT_TIMEOUT_MS must be between",
        ),
        (
            "AAI_HUB_LAKEBASE_LOCK_TIMEOUT_MS",
            "60001",
            "LOCK_TIMEOUT_MS must be between",
        ),
    ],
)
def test_lakebase_configuration_fails_closed(
    name: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(_environment(**{name: value}))


def test_pool_recycles_before_the_one_hour_oauth_credential_expires() -> None:
    now = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    clock = [now]
    calls: list[int] = []

    class Credential:
        def __init__(self, token: str, expires_at: datetime) -> None:
            self.token = token
            self.expire_time = expires_at

    def generate() -> Credential:
        calls.append(len(calls) + 1)
        return Credential(f"token-{calls[-1]}", clock[0] + timedelta(hours=1))

    provider = LakebaseOAuthTokenProvider(generate, clock=lambda: clock[0])
    assert provider.password() == "token-1"
    clock[0] = now + timedelta(minutes=50)
    assert provider.password() == "token-1"
    clock[0] = now + timedelta(minutes=58, seconds=1)
    assert provider.password() == "token-2"
    assert calls == [1, 2]
    assert "token-" not in repr(provider)


def test_token_provider_rejects_incomplete_credentials_without_echoing_them() -> None:
    credential = type(
        "Credential",
        (),
        {"token": "sensitive-token", "expire_time": None},
    )()
    provider = LakebaseOAuthTokenProvider(lambda: credential)
    with pytest.raises(HubRepositoryUnavailableError) as captured:
        provider.password()
    assert "sensitive-token" not in str(captured.value)


def test_migrations_are_forward_only_checksum_protected_and_schema_scoped() -> None:
    assert [migration.version for migration in MIGRATIONS] == list(
        range(1, len(MIGRATIONS) + 1)
    )
    assert len({migration.checksum for migration in MIGRATIONS}) == len(MIGRATIONS)
    sql = "\n".join(
        statement for migration in MIGRATIONS for statement in migration.statements
    )
    assert "__HUB_SCHEMA__.applications" in sql
    assert "JSONB NOT NULL" in sql
    assert "uq_hub_active_evaluation" in sql
    assert "uq_hub_active_promotion" in sql


class _NoConnectionEngine:
    def __init__(self) -> None:
        self.disposed = False

    @contextmanager
    def begin(self):
        raise RuntimeError("password=sensitive-token host=private.example")
        yield  # pragma: no cover

    def dispose(self) -> None:
        self.disposed = True


def test_adapter_implements_the_repository_contract_and_scrubs_driver_errors() -> None:
    engine = _NoConnectionEngine()
    repository = LakebaseHubRepository(
        engine,
        schema="aai_platform_hub",
        migrate=False,
    )
    assert isinstance(repository, HubRepository)
    with pytest.raises(HubRepositoryUnavailableError) as captured:
        repository.list_visible_applications(
            AuthorizationContext(principal="viewer@example.com")
        )
    assert "sensitive-token" not in str(captured.value)
    assert "private.example" not in str(captured.value)
    repository.close()
    assert engine.disposed


def test_jsonb_payloads_restore_strict_tuple_models() -> None:
    application = ApplicationRecord(
        application_id="analyst",
        name="Analyst",
        owner_principal="owner@example.com",
        support_group="analyst-support",
        business_domain="investments",
        cost_center="technology",
        risk_tier="medium",
        lifecycle_state="validation",
        tags=(Tag(key="team", value="platform"),),
        created_at=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
        updated_at=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
    )
    jsonb_value = application.model_dump(mode="json")
    restored = LakebaseHubRepository._model(ApplicationRecord, jsonb_value)
    assert restored == application
    assert isinstance(restored.tags, tuple)


def test_schema_identifier_is_validated_before_any_database_call() -> None:
    with pytest.raises(LakebaseConfigurationError, match="schema"):
        LakebaseHubRepository(
            _NoConnectionEngine(),
            schema='hub"; DROP SCHEMA public; --',
            migrate=False,
        )


def test_pool_configuration_stays_below_oauth_expiry() -> None:
    settings = LakebaseRuntimeSettings.from_environment(
        _environment(),
        schema="aai_platform_hub",
    )
    assert settings.pool_recycle_seconds <= 3300
    with pytest.raises(LakebaseConfigurationError, match="POOL_RECYCLE"):
        LakebaseRuntimeSettings.from_environment(
            _environment(AAI_HUB_LAKEBASE_POOL_RECYCLE_SECONDS="3599"),
            schema="aai_platform_hub",
        )


def test_postgres_execution_timeouts_are_finite_and_strict() -> None:
    settings = LakebaseRuntimeSettings.from_environment(
        _environment(),
        schema="aai_platform_hub",
    )
    assert settings.connection_options == (
        "-c statement_timeout=30000 -c lock_timeout=5000"
    )

    with pytest.raises(LakebaseConfigurationError, match="must not exceed"):
        LakebaseRuntimeSettings.from_environment(
            _environment(
                AAI_HUB_LAKEBASE_STATEMENT_TIMEOUT_MS="1000",
                AAI_HUB_LAKEBASE_LOCK_TIMEOUT_MS="1001",
            ),
            schema="aai_platform_hub",
        )


def test_engine_applies_execution_timeouts_to_each_new_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg
    import sqlalchemy

    settings = LakebaseRuntimeSettings.from_environment(
        _environment(),
        schema="aai_platform_hub",
    )
    captured: dict[str, object] = {}
    connection = object()

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return connection

    def fake_create_engine(url: str, **kwargs):
        captured["url"] = url
        return kwargs["creator"]

    credential = type(
        "Credential",
        (),
        {
            "token": "sensitive-token",
            "expire_time": datetime.now(UTC) + timedelta(hours=1),
        },
    )()
    postgres = type(
        "Postgres",
        (),
        {"generate_database_credential": lambda self, endpoint: credential},
    )()
    workspace_client = type("Workspace", (), {"postgres": postgres})()
    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(sqlalchemy, "create_engine", fake_create_engine)

    creator = create_lakebase_engine(settings, workspace_client=workspace_client)
    assert creator() is connection
    assert captured["url"] == "postgresql+psycopg://"
    assert captured["options"] == settings.connection_options
    assert "sensitive-token" not in str(captured["url"])


def test_relational_json_mismatch_fails_closed() -> None:
    timestamp = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    application = ApplicationRecord(
        application_id="analyst",
        name="Analyst",
        owner_principal="owner@example.com",
        support_group="analyst-support",
        business_domain="investments",
        cost_center="technology",
        risk_tier="medium",
        lifecycle_state="validation",
        created_at=timestamp,
        updated_at=timestamp,
    )
    repository = LakebaseHubRepository(
        _NoConnectionEngine(),
        schema="aai_platform_hub",
        migrate=False,
    )
    with pytest.raises(HubRepositoryUnavailableError, match="internally inconsistent"):
        repository._application(
            {
                "application_id": "another-application",
                "row_version": 1,
                "updated_at": timestamp,
                "record": application.model_dump(mode="json"),
            }
        )


def test_connection_retry_is_small_bounded_and_deterministic() -> None:
    class TransientConnectionError(Exception):
        pass

    attempts = 0
    sleeps: list[float] = []

    def connect() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientConnectionError("host=sensitive.internal")
        return "connected"

    assert (
        _connect_with_retry(
            connect,
            is_transient=lambda error: isinstance(error, TransientConnectionError),
            sleep=sleeps.append,
            delays=(0.25, 1.0),
        )
        == "connected"
    )
    assert attempts == 3
    assert sleeps == [0.25, 1.0]


def test_connection_retry_does_not_retry_authentication_or_config_errors() -> None:
    class OperationalError(Exception):
        def __init__(self, sqlstate: str | None) -> None:
            self.sqlstate = sqlstate

    attempts = 0
    sleeps: list[float] = []

    def connect() -> None:
        nonlocal attempts
        attempts += 1
        raise OperationalError("28P01")

    with pytest.raises(OperationalError):
        _connect_with_retry(
            connect,
            is_transient=lambda error: _is_transient_connection_error(
                error,
                operational_error_type=OperationalError,
            ),
            sleep=sleeps.append,
        )
    assert attempts == 1
    assert sleeps == []


@pytest.mark.parametrize("sqlstate", [None, "08006", "57P03"])
def test_transient_connection_sqlstates_are_retryable(sqlstate: str | None) -> None:
    class OperationalError(Exception):
        def __init__(self, state: str | None) -> None:
            self.sqlstate = state

    assert _is_transient_connection_error(
        OperationalError(sqlstate),
        operational_error_type=OperationalError,
    )


def test_postgres_repository_contract_when_explicit_dsn_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real PostgreSQL semantics without provisioning or container startup."""

    dsn = os.environ.get("AAI_TEST_POSTGRES_DSN", "").strip()
    if not dsn:
        pytest.skip("AAI_TEST_POSTGRES_DSN is not configured")

    from sqlalchemy import create_engine, text

    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    elif dsn.startswith("postgres://"):
        dsn = dsn.replace("postgres://", "postgresql+psycopg://", 1)

    schema = f"aai_hub_test_{uuid4().hex[:16]}"
    engine = create_engine(dsn, pool_pre_ping=True, future=True)
    repository: LakebaseHubRepository | None = None
    now = datetime.now(UTC)

    application = ApplicationRecord(
        application_id="contract_app",
        name="Contract app",
        owner_principal="owner@example.com",
        support_group="contract-support",
        business_domain="platform",
        cost_center="technology",
        risk_tier="medium",
        lifecycle_state="validation",
        tags=(Tag(key="team", value="platform"),),
        created_at=now,
        updated_at=now,
    )
    version_one = ApplicationVersionRecord(
        version_id="contract-version-1",
        application_id=application.application_id,
        environment="uat",
        git_repository="https://github.com/example/contract-app",
        git_commit_sha="a" * 40,
        manifest_version="ai-platform/v1",
        manifest_hash="a" * 64,
        manifest_json='{"metadata":{"id":"contract_app"}}',
        registered_by="ci@example.com",
        registered_at=now,
        deployment_target="uat",
    )
    evaluation = EvaluationRunRecord(
        evaluation_run_id="contract-evaluation-1",
        application_id=application.application_id,
        environment="uat",
        application_version_id=version_one.version_id,
        evaluation_profile="contract-v1",
        dataset_name="catalog.schema.contract_cases",
        dataset_version="dataset-1",
        job_id="123",
        requested_by="owner@example.com",
        status=EvaluationStatus.REQUESTED,
        requested_at=now,
    )

    try:
        repository = LakebaseHubRepository(engine, schema=schema)
        with engine.connect() as connection:
            applied = (
                connection.execute(
                    text(
                        "SELECT version, checksum "
                        f'FROM "{schema}".hub_schema_migrations ORDER BY version'
                    )
                )
                .mappings()
                .all()
            )
        assert [(row["version"], row["checksum"]) for row in applied] == [
            (migration.version, migration.checksum) for migration in MIGRATIONS
        ]

        first = repository.register_application(application, version_one)
        replay = repository.register_application(application, version_one)
        assert first.created is True
        assert replay.created is False
        assert replay.application == first.application

        later = now + timedelta(minutes=1)
        version_two = version_one.model_copy(
            update={
                "version_id": "contract-version-2",
                "git_commit_sha": "b" * 40,
                "manifest_hash": "b" * 64,
                "registered_at": later,
            }
        )
        version_two = ApplicationVersionRecord.model_validate(
            version_two.model_dump(mode="python")
        )
        updated_application = application.model_copy(update={"updated_at": later})
        updated_application = ApplicationRecord.model_validate(
            updated_application.model_dump(mode="python")
        )
        repository.register_application(updated_application, version_two)

        versions = repository.list_versions(application.application_id)
        assert [version.is_current for version in versions] == [False, True]
        assert (
            repository.get_current_version(application.application_id, "uat").version_id
            == version_two.version_id
        )
        visible = repository.list_visible_applications(
            AuthorizationContext(
                principal="auditor@example.com",
                platform_roles=(Role.AUDITOR,),
            )
        )
        assert visible[0].tags == application.tags

        repository.create_evaluation(evaluation)
        with pytest.raises(DuplicateActiveEvaluationError):
            repository.create_evaluation(
                evaluation.model_copy(
                    update={"evaluation_run_id": "contract-evaluation-2"}
                )
            )

        running = evaluation.model_copy(
            update={
                "status": EvaluationStatus.RUNNING,
                "started_at": later,
                "row_version": 2,
            }
        )
        running = EvaluationRunRecord.model_validate(running.model_dump(mode="python"))

        original_insert_event = repository._insert_event

        def fail_after_update(*args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError("forced post-update failure")

        monkeypatch.setattr(repository, "_insert_event", fail_after_update)
        with pytest.raises(HubRepositoryUnavailableError):
            repository.update_evaluation(running, expected_row_version=1)
        monkeypatch.setattr(repository, "_insert_event", original_insert_event)
        rolled_back = repository.get_evaluation(evaluation.evaluation_run_id)
        assert rolled_back.status is EvaluationStatus.REQUESTED
        assert rolled_back.row_version == 1

        repository.update_evaluation(running, expected_row_version=1)
        with pytest.raises(OptimisticConcurrencyError):
            repository.update_evaluation(running, expected_row_version=1)
        listed = repository.list_evaluations(application.application_id)
        assert listed == (running,)
    finally:
        with engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        engine.dispose()
