"""Behavioral contract for the user-scoped memory tools (credential-free)."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from langgraph.store.memory import InMemoryStore
from pydantic import ValidationError


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


memory_tools = _load(
    "aai_lakebase_memory_tools_contract",
    Path(__file__).with_name("memory_tools.py"),
)


def _tools(store, user_id="user-a"):
    return memory_tools.build_user_memory_tools(store, user_id=user_id)


def test_save_get_delete_round_trip_with_structured_not_found():
    async def scenario():
        store = InMemoryStore()
        get, save, delete = _tools(store)

        missing = await get.handler(key="preferred-region")
        assert missing == {"found": False, "key": "preferred-region"}

        saved = await save.handler(
            key="preferred-region", content="Deploy support cases in westeurope."
        )
        assert saved == {"saved": True, "key": "preferred-region", "kind": "preference"}

        found = await get.handler(key="preferred-region")
        assert found["found"] is True
        assert found["memory"] == {
            "content": "Deploy support cases in westeurope.",
            "kind": "preference",
        }

        deleted = await delete.handler(key="preferred-region")
        assert deleted == {"deleted": True, "key": "preferred-region"}
        assert (await get.handler(key="preferred-region"))["found"] is False

        # Deletion is idempotent: a repeated delete is a success, not a crash.
        assert await delete.handler(key="preferred-region") == {
            "deleted": True,
            "key": "preferred-region",
        }

    asyncio.run(scenario())


def test_decision_memories_carry_their_lineage():
    async def scenario():
        store = InMemoryStore()
        get, save, _ = _tools(store)

        await save.handler(
            key="region-decision",
            content="Rejected opening the case in the EU region.",
            kind=memory_tools.MemoryKind.DECISION,
            reason_code="policy_boundary",
            request_id="request-9",
        )
        found = await get.handler(key="region-decision")
        assert found["memory"] == {
            "content": "Rejected opening the case in the EU region.",
            "kind": "decision",
            "reason_code": "policy_boundary",
            "request_id": "request-9",
        }

    asyncio.run(scenario())


def test_email_shaped_user_ids_work_and_stay_isolated():
    async def scenario():
        # Store namespace labels reject periods on writes only; an unencoded
        # email-shaped user id would read "not found" forever and crash on
        # the first save. The builder encodes labels injectively instead.
        store = InMemoryStore()
        get_a, save_a, delete_a = _tools(store, user_id="jane.doe@example.com")
        await save_a.handler(key="preferred-region", content="westeurope")
        assert (await get_a.handler(key="preferred-region"))["found"] is True

        # Injective encoding: a crafted id that matches the encoded form of
        # another id still lands in its own namespace.
        get_b, _, _ = _tools(store, user_id="jane%2Edoe@example.com")
        assert (await get_b.handler(key="preferred-region"))["found"] is False

        assert (await delete_a.handler(key="preferred-region"))["deleted"] is True

    asyncio.run(scenario())


def test_handlers_enforce_the_strict_models_on_the_direct_binding_path():
    async def scenario():
        store = InMemoryStore()
        get, save, _ = _tools(store)

        # Wire-string kind (JSON graph state, model tool calls) keeps its
        # lineage instead of silently storing a decision without it.
        await save.handler(
            key="region-decision",
            content="Rejected the EU region.",
            kind="decision",
            reason_code="policy_boundary",
            request_id="request-3",
        )
        found = await get.handler(key="region-decision")
        assert found["memory"]["reason_code"] == "policy_boundary"
        assert found["memory"]["request_id"] == "request-3"

        # A decision without lineage is rejected even when the handler is
        # bound directly, and unknown kinds never persist silently.
        with pytest.raises(ValidationError):
            await save.handler(key="k", content="x", kind="decision")
        with pytest.raises(ValidationError):
            await save.handler(key="k", content="x", kind="not-a-kind")

    asyncio.run(scenario())


def test_builder_rejects_namespace_prefixes_the_store_would_refuse():
    with pytest.raises(ValueError, match="namespace_prefix"):
        memory_tools.build_user_memory_tools(
            InMemoryStore(), user_id="user-a", namespace_prefix=("mem.ories",)
        )
    with pytest.raises(ValueError, match="reserved label"):
        memory_tools.build_user_memory_tools(
            InMemoryStore(), user_id="user-a", namespace_prefix=("langgraph",)
        )
    with pytest.raises(ValueError, match="namespace_prefix"):
        memory_tools.build_user_memory_tools(
            InMemoryStore(), user_id="user-a", namespace_prefix=()
        )


def test_namespaces_isolate_users():
    async def scenario():
        store = InMemoryStore()
        get_a, save_a, _ = _tools(store, user_id="user-a")
        get_b, _, delete_b = _tools(store, user_id="user-b")

        await save_a.handler(key="preferred-region", content="westeurope")
        assert (await get_b.handler(key="preferred-region"))["found"] is False

        # Another user's delete cannot remove this user's memory.
        await delete_b.handler(key="preferred-region")
        assert (await get_a.handler(key="preferred-region"))["found"] is True

    asyncio.run(scenario())


def test_input_models_reject_coercion_unknown_fields_and_broken_lineage():
    save_input = memory_tools.SaveMemoryInput

    with pytest.raises(ValidationError):
        save_input.model_validate({"key": "k", "content": 5})
    with pytest.raises(ValidationError):
        save_input.model_validate({"key": "k", "content": "x", "unexpected": True})
    # A decision without lineage loses the review signal.
    with pytest.raises(ValidationError):
        save_input.model_validate({"key": "k", "content": "x", "kind": "decision"})
    # Lineage fields on a preference would be silent misclassification.
    with pytest.raises(ValidationError):
        save_input.model_validate(
            {"key": "k", "content": "x", "reason_code": "model_error"}
        )
    with pytest.raises(ValidationError):
        memory_tools.GetMemoryInput.model_validate({"key": "k", "extra": 1})


def test_handlers_validate_keys_and_builder_validates_identity():
    async def scenario():
        store = InMemoryStore()
        get, _, _ = _tools(store)
        with pytest.raises(ValueError, match="memory key"):
            await get.handler(key="Not A Key!")

    asyncio.run(scenario())

    with pytest.raises(ValueError, match="user_id"):
        memory_tools.build_user_memory_tools(InMemoryStore(), user_id="")
    with pytest.raises(ValueError, match="namespace_prefix"):
        memory_tools.build_user_memory_tools(
            InMemoryStore(), user_id="user-a", namespace_prefix=("",)
        )


def test_tool_outputs_never_leak_the_user_identity():
    async def scenario():
        store = InMemoryStore()
        user_id = "confidential-user-17"
        get, save, delete = memory_tools.build_user_memory_tools(store, user_id=user_id)

        results = [
            await save.handler(key="preferred-region", content="westeurope"),
            await get.handler(key="preferred-region"),
            await delete.handler(key="preferred-region"),
            await get.handler(key="preferred-region"),
        ]
        for result in results:
            assert user_id not in repr(result)

    asyncio.run(scenario())
