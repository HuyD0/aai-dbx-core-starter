"""Durable, dependency-injected LangGraph reference implementation."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict


class SupportRequest(BaseModel):
    """Untrusted application input is validated before entering graph state."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    conversation_id: str
    request_id: str
    question: str


class GraphState(TypedDict, total=False):
    # Persist only ordinary JSON-compatible data. Pydantic validates the
    # untrusted boundary, but custom model instances do not enter checkpoints.
    request: dict[str, Any]
    proposed_action: dict[str, Any]
    approved: bool
    result: dict[str, Any]


class Dependencies(Protocol):
    async def propose(self, request: SupportRequest) -> dict[str, Any]: ...

    async def execute_once(
        self,
        *,
        idempotency_key: str,
        action: dict[str, Any],
    ) -> dict[str, Any]: ...


def initial_state(request: SupportRequest | Mapping[str, Any]) -> GraphState:
    """Validate an external request and return checkpoint-safe graph state."""

    validated = SupportRequest.model_validate(request, strict=True)
    return {"request": validated.model_dump(mode="json")}


def build_graph(
    dependencies: Dependencies,
    *,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
):
    """Compile with async durable state supplied by the deployment environment."""

    _require_async_checkpointer(checkpointer)

    async def propose(state: GraphState) -> GraphState:
        request = SupportRequest.model_validate(state["request"], strict=True)
        return {
            "request": request.model_dump(mode="json"),
            "proposed_action": await dependencies.propose(request),
        }

    async def approve(state: GraphState) -> GraphState:
        approved = interrupt(
            {
                "question": "Approve the proposed action?",
                "action": state["proposed_action"],
            }
        )
        return {"approved": approved is True}

    def route(state: GraphState) -> str:
        return "execute" if state.get("approved") else "rejected"

    async def execute(state: GraphState) -> GraphState:
        request = SupportRequest.model_validate(state["request"], strict=True)
        result = await dependencies.execute_once(
            idempotency_key=request.request_id,
            action=state["proposed_action"],
        )
        return {"result": result}

    async def rejected(state: GraphState) -> GraphState:
        return {"result": {"status": "rejected"}}

    builder = StateGraph(GraphState)
    builder.add_node("propose", propose)
    builder.add_node("approve", approve)
    builder.add_node("execute", execute)
    builder.add_node("rejected", rejected)
    builder.add_edge(START, "propose")
    builder.add_edge("propose", "approve")
    builder.add_conditional_edges(
        "approve",
        route,
        {"execute": "execute", "rejected": "rejected"},
    )
    builder.add_edge("execute", END)
    builder.add_edge("rejected", END)
    return builder.compile(checkpointer=checkpointer, store=store)


def _require_async_checkpointer(checkpointer: BaseCheckpointSaver) -> None:
    """Fail at construction instead of blocking async execution later."""

    missing = []
    for name in ("aget_tuple", "aput", "aput_writes", "alist"):
        method = getattr(checkpointer, name, None)
        is_async = inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(
            method
        )
        if not is_async:
            missing.append(name)
    if missing:
        raise TypeError(
            "Async LangGraph execution requires an async-compatible checkpointer; "
            "missing async methods: " + ", ".join(missing)
        )
