"""Native LangGraph adapter around the accelerator's tested workflow core."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from typing import Any, TypedDict

from email_support_agent.contracts import (
    ExecutionResult,
    PreparedCase,
    RedactedEmail,
    ReviewDecision,
)
from email_support_agent.workflow import (
    EmailSupportWorkflow,
    checkpoint_state,
    proposal_digest,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import ValidationError


class GraphState(TypedDict, total=False):
    request: dict[str, Any]
    prepared: dict[str, Any]
    result: dict[str, Any]


def initial_state(email: RedactedEmail | Mapping[str, Any]) -> GraphState:
    """Validate before persistence and keep framework state JSON-compatible."""

    request = RedactedEmail.model_validate(email, strict=True)
    return {"request": request.model_dump(mode="json")}


def build_graph(
    workflow: EmailSupportWorkflow,
    *,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
):
    """Compile the graph with deployment-owned durable state dependencies."""

    _require_async_checkpointer(checkpointer)

    async def prepare(state: GraphState) -> GraphState:
        request = RedactedEmail.model_validate(state["request"], strict=True)
        prepared = await workflow.prepare(request)
        return {"prepared": checkpoint_state(prepared)}

    def route_after_prepare(state: GraphState) -> str:
        prepared = _restore(PreparedCase, state["prepared"])
        return "review" if prepared.requires_review else "commit"

    async def review(state: GraphState) -> GraphState:
        prepared = _restore(PreparedCase, state["prepared"])
        resumed = interrupt(
            {
                "case_id": prepared.email.case_id,
                "proposal_digest": proposal_digest(prepared),
                "application_release": prepared.application_release,
                "route": prepared.route.value,
                "route_reasons": list(prepared.route_reasons),
                "draft": prepared.draft.model_dump(mode="json"),
                "gates": [item.model_dump(mode="json") for item in prepared.gates],
                "planned_actions": [
                    {
                        "kind": item.kind.value,
                        "idempotency_key": item.idempotency_key,
                    }
                    for item in prepared.planned_actions
                ],
            }
        )
        try:
            decision = _restore(ReviewDecision, resumed)
        except (TypeError, ValueError, ValidationError):
            raise ValueError("review decision failed strict admission policy") from None
        # Validate identity, edit policy, and commit inside the resumed node.
        # Invalid free text or authorization never becomes graph state.
        result = await workflow.commit(prepared, review=decision)
        return {"result": result.model_dump(mode="json")}

    async def commit(state: GraphState) -> GraphState:
        prepared = _restore(PreparedCase, state["prepared"])
        result = await workflow.commit(prepared)
        return {"result": result.model_dump(mode="json")}

    builder = StateGraph(GraphState)
    builder.add_node("prepare", prepare)
    builder.add_node("review", review)
    builder.add_node("commit", commit)
    builder.add_edge(START, "prepare")
    builder.add_conditional_edges(
        "prepare",
        route_after_prepare,
        {"review": "review", "commit": "commit"},
    )
    builder.add_edge("review", END)
    builder.add_edge("commit", END)
    return builder.compile(checkpointer=checkpointer, store=store)


def final_result(state: Mapping[str, Any]) -> ExecutionResult:
    return _restore(ExecutionResult, state["result"])


def _restore(model: Any, value: Any) -> Any:
    if isinstance(value, model):
        return value
    return model.model_validate_json(json.dumps(value), strict=True)


def _require_async_checkpointer(checkpointer: BaseCheckpointSaver) -> None:
    missing = []
    for name in ("aget_tuple", "aput", "aput_writes", "alist"):
        method = getattr(checkpointer, name, None)
        if not (
            inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method)
        ):
            missing.append(name)
    if missing:
        raise TypeError(
            "async LangGraph execution requires an async-compatible durable "
            "checkpointer; missing: " + ", ".join(missing)
        )
