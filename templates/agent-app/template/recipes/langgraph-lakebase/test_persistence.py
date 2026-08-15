"""Contract tests for the Lakebase persistence recipe.

The default tier is credential-free: it exercises the settings boundary, the
token lifecycle, and the fresh-token connect path against fakes. The
integration tier runs the sibling graph recipe against a real PostgreSQL
server and is skipped unless ``AAI_LAKEBASE_TEST_DSN`` is set.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
persistence = _load("aai_lakebase_persistence", HERE / "persistence.py")
memory_tools = _load("aai_lakebase_memory_tools", HERE / "memory_tools.py")
graph_recipe = _load(
    "aai_langgraph_recipe_for_lakebase", HERE.parent / "langgraph" / "graph.py"
)

TEST_DSN = os.environ.get("AAI_LAKEBASE_TEST_DSN", "")
requires_postgres = pytest.mark.skipif(
    not TEST_DSN, reason="AAI_LAKEBASE_TEST_DSN is not set"
)

APPROVE = {"approved": True, "reason_code": "approved"}
REJECT_MODEL_ERROR = {
    "approved": False,
    "reason_code": "model_error",
    "note": "The proposed case references the wrong account.",
}
LOCAL_ENDPOINT = "projects/local/branches/test/endpoints/test"


class FakeCredential:
    def __init__(self, token: str, expires_in: timedelta, now: datetime) -> None:
        self.token = token
        self.expire_time = now + expires_in


class SteppingClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def _settings(**overrides):
    values = {
        "host": "db.example.internal",
        "database": "agents",
        "user": "agent_app",
        "endpoint": LOCAL_ENDPOINT,
    }
    values.update(overrides)
    return persistence.LakebaseSettings(**values)


def test_settings_reject_unsafe_runtime_contracts():
    with pytest.raises(ValidationError):
        _settings(host="bad host!")
    with pytest.raises(ValidationError):
        _settings(endpoint="not-an-endpoint-path")
    with pytest.raises(ValidationError):
        _settings(sslmode="disable")
    with pytest.raises(ValidationError):
        _settings(user="agent\napp")


def test_settings_allow_plaintext_only_for_loopback():
    local = _settings(host="localhost", sslmode="disable")
    assert local.sslmode == "disable"


def test_settings_accept_stronger_tls_modes_off_box():
    # verify-ca / verify-full are stricter than require; hardening the
    # binding must not brick startup.
    for sslmode in ("verify-ca", "verify-full"):
        assert _settings(sslmode=sslmode).sslmode == sslmode


def test_settings_from_environment_requires_the_full_binding():
    with pytest.raises(persistence.LakebasePersistenceError) as excinfo:
        persistence.LakebaseSettings.from_environment(
            {"PGHOST": "db.example.internal", "PGUSER": "agent_app"}
        )
    assert "LAKEBASE_ENDPOINT" in str(excinfo.value)
    assert "PGDATABASE" in str(excinfo.value)

    settings = persistence.LakebaseSettings.from_environment(
        {
            "PGHOST": "db.example.internal",
            "PGPORT": "5432",
            "PGDATABASE": "agents",
            "PGUSER": "agent_app",
            "PGSSLMODE": "require",
            "LAKEBASE_ENDPOINT": LOCAL_ENDPOINT,
        }
    )
    assert settings.port == 5432
    assert settings.sslmode == "require"


def test_credential_provider_caches_and_refreshes_by_expiry():
    async def scenario():
        clock = SteppingClock(datetime(2026, 1, 1, tzinfo=UTC))
        minted: list[str] = []

        def generate():
            token = f"token-{len(minted) + 1}"
            minted.append(token)
            return FakeCredential(token, timedelta(minutes=15), clock.now)

        provider = persistence.LakebaseCredentialProvider(generate, clock=clock)
        assert await provider.password() == "token-1"
        # Within the validity window the cached token is reused.
        clock.now += timedelta(minutes=5)
        assert await provider.password() == "token-1"
        # Inside the refresh skew a fresh token is minted before expiry.
        clock.now += timedelta(minutes=8, seconds=30)
        assert await provider.password() == "token-2"
        assert minted == ["token-1", "token-2"]

    asyncio.run(scenario())


def test_credential_provider_fails_closed():
    async def scenario():
        clock = SteppingClock(datetime(2026, 1, 1, tzinfo=UTC))

        provider = persistence.LakebaseCredentialProvider(
            lambda: FakeCredential("", timedelta(minutes=15), clock.now),
            clock=clock,
        )
        with pytest.raises(persistence.LakebasePersistenceError, match="no token"):
            await provider.password()

        short = persistence.LakebaseCredentialProvider(
            lambda: FakeCredential("token", timedelta(seconds=30), clock.now),
            clock=clock,
        )
        with pytest.raises(
            persistence.LakebasePersistenceError, match="expires too soon"
        ):
            await short.password()

    asyncio.run(scenario())


def test_credential_provider_never_reveals_the_token():
    async def scenario():
        clock = SteppingClock(datetime(2026, 1, 1, tzinfo=UTC))
        provider = persistence.LakebaseCredentialProvider(
            lambda: FakeCredential("s3cret-token", timedelta(minutes=15), clock.now),
            clock=clock,
        )
        await provider.password()
        for rendered in (repr(provider), str(provider), repr(provider._current)):
            assert "s3cret-token" not in rendered

    asyncio.run(scenario())


def test_pool_mints_a_fresh_token_for_every_new_connection(monkeypatch):
    async def scenario():
        clock = SteppingClock(datetime(2026, 1, 1, tzinfo=UTC))
        minted: list[str] = []

        def generate():
            token = f"token-{len(minted) + 1}"
            minted.append(token)
            return FakeCredential(token, timedelta(minutes=15), clock.now)

        provider = persistence.LakebaseCredentialProvider(generate, clock=clock)
        connect_kwargs: list[dict] = []

        async def fake_parent_connect(self, timeout=None):
            connect_kwargs.append(dict(self.kwargs))
            return object()

        monkeypatch.setattr(AsyncConnectionPool, "_connect", fake_parent_connect)
        pool = persistence._FreshTokenPool(
            conninfo="",
            credential_provider=provider,
            kwargs={"host": "db.example.internal", "user": "agent_app"},
            min_size=0,
            max_size=2,
            open=False,
        )

        await pool._connect()
        clock.now += timedelta(minutes=14)
        await pool._connect()

        assert [kwargs["password"] for kwargs in connect_kwargs] == [
            "token-1",
            "token-2",
        ]
        # The token travels only through connection kwargs during the
        # connect call — never conninfo, and never the pool's persistent
        # kwargs mapping afterwards.
        assert pool.conninfo == ""
        assert "password" not in pool.kwargs

    asyncio.run(scenario())


def test_async_postgres_saver_satisfies_the_recipe_construction_contract():
    # The sibling recipe refuses sync-only checkpointers at build time; the
    # native Postgres saver must keep satisfying that async surface. This is
    # a credential-free upstream-drift canary: no connection is opened.
    async def scenario():
        graph_recipe._require_async_checkpointer(AsyncPostgresSaver(None))

    asyncio.run(scenario())


def test_build_lakebase_persistence_wires_saver_and_store_without_connecting():
    async def scenario():
        clock = SteppingClock(datetime(2026, 1, 1, tzinfo=UTC))
        settings = _settings(pool_min_size=0)
        generate = lambda: FakeCredential(  # noqa: E731
            "token", timedelta(minutes=15), clock.now
        )
        async with persistence.build_lakebase_persistence(
            settings, generate, clock=clock
        ) as (checkpointer, store):
            assert isinstance(checkpointer, AsyncPostgresSaver)
            assert isinstance(store, AsyncPostgresStore)
            graph_recipe._require_async_checkpointer(checkpointer)

    asyncio.run(scenario())


class RecordingDependencies:
    def __init__(self) -> None:
        self.execute_attempts = 0
        self.side_effects = 0
        self.results: dict[str, dict] = {}

    async def propose(self, request, *, feedback=None):
        await asyncio.sleep(0)
        return {"action": "open_case", "question": request.question}

    async def execute_once(self, *, idempotency_key, action):
        await asyncio.sleep(0)
        self.execute_attempts += 1
        if idempotency_key not in self.results:
            self.side_effects += 1
            self.results[idempotency_key] = {
                "status": "completed",
                "idempotency_key": idempotency_key,
                **action,
            }
        return self.results[idempotency_key]


def _integration_settings_and_generate():
    parsed = urlsplit(TEST_DSN)
    assert parsed.scheme.startswith("postgres")
    now = datetime.now(UTC)
    settings = persistence.LakebaseSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=(parsed.path or "/postgres").lstrip("/"),
        user=unquote(parsed.username or "postgres"),
        endpoint=LOCAL_ENDPOINT,
        sslmode="disable",
        pool_min_size=0,
    )
    password = unquote(parsed.password or "")

    def generate():
        return FakeCredential(password or "postgres", timedelta(hours=1), now)

    return settings, generate


@requires_postgres
def test_interrupt_resume_is_durable_across_fresh_saver_instances():
    async def scenario():
        settings, generate = _integration_settings_and_generate()
        dependencies = RecordingDependencies()
        run = uuid.uuid4().hex[:12]

        request = graph_recipe.SupportRequest(
            conversation_id=f"conversation-{run}",
            request_id=f"request-{run}",
            question="Open a support case",
        )

        for index, thread_id in enumerate((f"thread-{run}-1", f"thread-{run}-2")):
            config = {"configurable": {"thread_id": thread_id}}

            async with persistence.build_lakebase_persistence(
                settings, generate, run_setup=True
            ) as (checkpointer, store):
                graph = graph_recipe.build_graph(
                    dependencies, checkpointer=checkpointer, store=store
                )
                interrupted = await graph.ainvoke(
                    graph_recipe.initial_state(request), config
                )
                assert interrupted["__interrupt__"]

            # A fresh saver instance proves durability beyond process memory:
            # the thread resumes from PostgreSQL state alone.
            async with persistence.build_lakebase_persistence(settings, generate) as (
                checkpointer,
                store,
            ):
                graph = graph_recipe.build_graph(
                    dependencies, checkpointer=checkpointer, store=store
                )
                completed = await graph.ainvoke(Command(resume=APPROVE), config)
                assert completed["result"]["status"] == "completed"
                assert completed["decision"]["reason_code"] == "approved"

            # Duplicate delivery of the same request on a second thread makes
            # the execute attempt but the idempotency key prevents a second
            # side effect.
            assert dependencies.execute_attempts == index + 1
            assert dependencies.side_effects == 1

    asyncio.run(scenario())


@requires_postgres
def test_rejection_reason_survives_a_real_checkpoint_round_trip():
    async def scenario():
        settings, generate = _integration_settings_and_generate()
        dependencies = RecordingDependencies()
        run = uuid.uuid4().hex[:12]
        config = {"configurable": {"thread_id": f"thread-{run}-rejected"}}
        request = graph_recipe.SupportRequest(
            conversation_id=f"conversation-{run}",
            request_id=f"request-{run}",
            question="Open a support case",
        )

        async with persistence.build_lakebase_persistence(
            settings, generate, run_setup=True
        ) as (checkpointer, store):
            graph = graph_recipe.build_graph(
                dependencies, checkpointer=checkpointer, store=store
            )
            interrupted = await graph.ainvoke(
                graph_recipe.initial_state(request), config
            )
            assert interrupted["__interrupt__"]

        async with persistence.build_lakebase_persistence(settings, generate) as (
            checkpointer,
            store,
        ):
            graph = graph_recipe.build_graph(
                dependencies, checkpointer=checkpointer, store=store
            )
            rejected = await graph.ainvoke(Command(resume=REJECT_MODEL_ERROR), config)
            assert rejected["result"]["status"] == "rejected"
            assert rejected["result"]["reason_code"] == "model_error"
            assert rejected["result"]["note"] == REJECT_MODEL_ERROR["note"]
            assert dependencies.execute_attempts == 0

    asyncio.run(scenario())


@requires_postgres
def test_memory_tools_are_durable_across_store_instances():
    async def scenario():
        settings, generate = _integration_settings_and_generate()
        run = uuid.uuid4().hex[:12]
        user_id = f"user-{run}"

        async with persistence.build_lakebase_persistence(
            settings, generate, run_setup=True
        ) as (_, store):
            _, save, _ = memory_tools.build_user_memory_tools(store, user_id=user_id)
            saved = await save.handler(
                key="region-decision",
                content="Rejected opening the case in the EU region.",
                kind=memory_tools.MemoryKind.DECISION,
                reason_code="policy_boundary",
                request_id=f"request-{run}",
            )
            assert saved == {
                "saved": True,
                "key": "region-decision",
                "kind": "decision",
            }

        async with persistence.build_lakebase_persistence(settings, generate) as (
            _,
            store,
        ):
            get, _, _ = memory_tools.build_user_memory_tools(store, user_id=user_id)
            found = await get.handler(key="region-decision")
            assert found["found"] is True
            assert found["memory"]["reason_code"] == "policy_boundary"
            assert found["memory"]["request_id"] == f"request-{run}"

    asyncio.run(scenario())
