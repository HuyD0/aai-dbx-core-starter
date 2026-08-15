"""Durable, dependency-injected LangGraph reference implementation."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

DEFAULT_MAX_PROPOSAL_ATTEMPTS = 2


class SupportRequest(BaseModel):
    """Untrusted application input is validated before entering graph state."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    conversation_id: str
    request_id: str
    question: str


class DecisionReason(StrEnum):
    """Why the reviewer approved or rejected the proposed action.

    A rejection without a reason fixes one case and loses the signal. The
    reason decides what the intervention becomes: a regression case, a
    replan, a guardrail, or a context refresh.
    """

    APPROVED = "approved"
    MODEL_ERROR = "model_error"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    POLICY_BOUNDARY = "policy_boundary"
    STALE_CONTEXT = "stale_context"


class ApprovalDecision(BaseModel):
    """Untrusted resume payload validated at the interrupt boundary.

    The value delivered through ``Command(resume=...)`` crosses a trust
    boundary exactly like the original request, so it is validated with the
    same strictness before it can steer execution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    approved: bool
    reason_code: DecisionReason
    note: str = Field(default="", max_length=500)

    @field_validator("reason_code", mode="before")
    @classmethod
    def _reason_from_wire(cls, value: Any) -> Any:
        # Resume payloads arrive as checkpoint-safe JSON; accept the wire
        # string for the enum while strict mode still rejects coerced bools
        # and non-string notes.
        if isinstance(value, str) and not isinstance(value, DecisionReason):
            return DecisionReason(value)
        return value

    @model_validator(mode="after")
    def _reason_matches_outcome(self) -> ApprovalDecision:
        if self.approved and self.reason_code is not DecisionReason.APPROVED:
            raise ValueError("an approval must use reason_code 'approved'")
        if not self.approved and self.reason_code is DecisionReason.APPROVED:
            raise ValueError("a rejection needs a reason other than 'approved'")
        return self


class GraphState(TypedDict, total=False):
    # Persist only ordinary JSON-compatible data. Pydantic validates the
    # untrusted boundary, but custom model instances do not enter checkpoints.
    request: dict[str, Any]
    proposed_action: dict[str, Any]
    attempts: int
    decision: dict[str, Any]
    result: dict[str, Any]


class Dependencies(Protocol):
    async def propose(
        self,
        request: SupportRequest,
        *,
        feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

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
    max_proposal_attempts: int = DEFAULT_MAX_PROPOSAL_ATTEMPTS,
):
    """Compile with async durable state supplied by the deployment environment."""

    _require_async_checkpointer(checkpointer)
    if max_proposal_attempts < 1:
        raise ValueError("max_proposal_attempts must be at least 1")

    async def propose(state: GraphState) -> GraphState:
        request = SupportRequest.model_validate(state["request"], strict=True)
        # On a replan the prior decision carries the reviewer's reason and
        # note so the new proposal can interpret the correction instead of
        # repeating the rejected one.
        feedback = state.get("decision")
        return {
            "request": request.model_dump(mode="json"),
            "attempts": state.get("attempts", 0) + 1,
            "proposed_action": await dependencies.propose(request, feedback=feedback),
        }

    async def approve(state: GraphState) -> GraphState:
        payload = {
            "question": "Approve the proposed action?",
            "action": state["proposed_action"],
            "attempt": state.get("attempts", 1),
        }
        while True:
            raw = interrupt(payload)
            try:
                decision = ApprovalDecision.model_validate(raw)
            except ValidationError as error:
                # A submitted resume value is durable: raising here would
                # replay the malformed value on every later resume and poison
                # the thread. Re-interrupt with the validation problem instead
                # so the reviewer can submit a well-formed decision.
                payload = {
                    **payload,
                    "error": "invalid decision payload: "
                    + "; ".join(
                        issue["msg"] for issue in error.errors(include_url=False)
                    ),
                }
                continue
            return {"decision": decision.model_dump(mode="json")}

    def route(state: GraphState) -> str:
        decision = state.get("decision") or {}
        if decision.get("approved"):
            return "execute"
        replannable = decision.get("reason_code") == DecisionReason.AMBIGUOUS_INTENT
        if replannable and state.get("attempts", 1) < max_proposal_attempts:
            return "propose"
        return "rejected"

    async def execute(state: GraphState) -> GraphState:
        request = SupportRequest.model_validate(state["request"], strict=True)
        result = await dependencies.execute_once(
            idempotency_key=request.request_id,
            action=state["proposed_action"],
        )
        return {"result": result}

    async def rejected(state: GraphState) -> GraphState:
        decision = state.get("decision") or {}
        return {
            "result": {
                "status": "rejected",
                "reason_code": decision.get("reason_code"),
                "note": decision.get("note", ""),
                "attempts": state.get("attempts", 1),
            }
        }

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
        {"execute": "execute", "propose": "propose", "rejected": "rejected"},
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
