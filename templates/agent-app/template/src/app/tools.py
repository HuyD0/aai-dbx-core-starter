"""Application-owned async tool contracts and execution.

The platform SDK deliberately does not own an agent loop or tool registry.
This module is ordinary application code and can be replaced by LangGraph,
LangChain tools, or another native framework without changing aai-core.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aai_core.exceptions import AaiCoreError
from aai_core.tracing import provider_span

_ORDERS = {
    "A-1001": {"status": "shipped", "eta_days": 2},
    "A-1002": {"status": "processing", "eta_days": 5},
}


class ToolExecutionError(AaiCoreError):
    code = "app.tool_execution"


async def lookup_order_status(order_id: str) -> dict[str, Any]:
    """Code tool: look up an order in the fulfillment system."""

    order = _ORDERS.get(order_id)
    if order is None:
        return {"order_id": order_id, "error": "order not found"}
    return {"order_id": order_id, **order}


class OrderStatusInput(BaseModel):
    """Strict tool boundary; unexpected or coerced arguments are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    order_id: str = Field(description="Order identifier, e.g. A-1001")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[..., Awaitable[Any]]
    timeout_seconds: float = 10.0

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


class AsyncToolRegistry:
    def __init__(self, specs: tuple[ToolSpec, ...]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Tool names must be unique")
        for spec in specs:
            if spec.input_model.model_config.get("extra") != "forbid":
                raise ValueError(
                    f"Tool input model {spec.input_model.__name__!r} must forbid extras"
                )
            if not inspect.iscoroutinefunction(spec.handler):
                raise TypeError(f"Tool handler {spec.name!r} must be async")

    def openai_tools(self) -> list[dict[str, Any]]:
        return [spec.as_openai_tool() for spec in self._specs.values()]

    async def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        try:
            spec = self._specs[name]
        except KeyError as error:
            raise ToolExecutionError(
                f"Model requested unknown tool {name!r}"
            ) from error
        try:
            inputs = spec.input_model.model_validate(dict(arguments), strict=True)
        except ValidationError as error:
            raise ToolExecutionError(
                f"Arguments for tool {name!r} failed schema validation"
            ) from error
        validated = inputs.model_dump(mode="python")
        with provider_span(
            name,
            span_type="TOOL",
            attributes={"gen_ai.tool.name": name},
        ) as span:
            if span is not None:
                span.set_inputs(validated)
            try:
                result = await asyncio.wait_for(
                    spec.handler(**validated),
                    timeout=spec.timeout_seconds,
                )
            except TimeoutError as error:
                raise ToolExecutionError(
                    f"Tool {name!r} exceeded its {spec.timeout_seconds:g}s timeout"
                ) from error
            except Exception as error:
                raise ToolExecutionError(f"Tool {name!r} failed") from error
            serialized = result if isinstance(result, str) else json.dumps(result)
            if span is not None:
                span.set_outputs(serialized)
            return serialized


def build_registry() -> AsyncToolRegistry:
    return AsyncToolRegistry(
        (
            ToolSpec(
                name="lookup_order_status",
                description="Look up an order's fulfillment status by order id.",
                input_model=OrderStatusInput,
                handler=lookup_order_status,
            ),
        )
    )


# Unity Catalog functions are externally provisioned resources. A generated
# application may replace this code-tool registry with the provider-native
# UCFunctionToolkit and declare the same functions as serving resources.
UC_FUNCTION_TOOLS: tuple[str, ...] = ()
