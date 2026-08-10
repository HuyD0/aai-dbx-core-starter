"""Tool unit tests — pure functions, zero cloud."""

import asyncio
import json

import pytest
from pydantic import BaseModel, ConfigDict

from app.tools import (
    AsyncToolRegistry,
    ToolExecutionError,
    ToolSpec,
    build_agent_registry,
    lookup_order_status,
)


def test_lookup_returns_status_or_explicit_not_found():
    assert asyncio.run(lookup_order_status("A-1001"))["status"] == "shipped"
    assert asyncio.run(lookup_order_status("missing"))["error"] == "order not found"


def test_registry_exposes_openai_tool_metadata_and_executes():
    registry = build_agent_registry(timeout_seconds=3.5)

    tools = registry.openai_tools()
    assert tools[0]["function"]["name"] == "lookup_order_status"
    assert "order_id" in tools[0]["function"]["parameters"]["properties"]
    assert registry._specs["lookup_order_status"].timeout_seconds == 3.5

    result = json.loads(
        asyncio.run(registry.execute("lookup_order_status", {"order_id": "A-1002"}))
    )
    assert result["status"] == "processing"

    with pytest.raises(ToolExecutionError, match="schema validation"):
        asyncio.run(registry.execute("lookup_order_status", {"order_id": 1002}))
    with pytest.raises(ToolExecutionError, match="schema validation"):
        asyncio.run(
            registry.execute(
                "lookup_order_status",
                {"order_id": "A-1002", "unapproved": True},
            )
        )


def test_tool_timeout_is_normalized_and_cancellation_propagates():
    class EmptyInput(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

    async def blocked():
        await asyncio.Future()

    registry = AsyncToolRegistry(
        (
            ToolSpec(
                name="blocked",
                description="Wait indefinitely.",
                input_model=EmptyInput,
                handler=blocked,
                timeout_seconds=0.01,
            ),
        )
    )
    with pytest.raises(ToolExecutionError, match="timeout"):
        asyncio.run(registry.execute("blocked", {}))

    async def cancel():
        task = asyncio.create_task(registry.execute("blocked", {}))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            _ = await task

    asyncio.run(cancel())


def test_tool_output_is_bounded_before_it_reenters_the_prompt():
    class EmptyInput(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

    async def oversized():
        return "12345"

    registry = AsyncToolRegistry(
        (
            ToolSpec(
                name="oversized",
                description="Return too much content.",
                input_model=EmptyInput,
                handler=oversized,
                max_output_chars=4,
            ),
        )
    )

    with pytest.raises(ToolExecutionError, match="output exceeded"):
        asyncio.run(registry.execute("oversized", {}))
