"""MLflow AgentServer endpoint for the primary Databricks Apps deployment."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from functools import lru_cache
from typing import Any

import mlflow
from mlflow.genai.agent_server import invoke, stream
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from aai_core import bootstrap
from aai_core.agents import AgentRequest
from aai_core.tracing import (
    TraceIntegration,
    TracePolicy,
    set_trace_resource_context,
    set_trace_session,
    trace_context,
)
from app.agent import ToolAgent
from app.messages import response_message_text


@lru_cache(maxsize=1)
def _application() -> ToolAgent:
    """Initialize native clients once after the App identity is available."""

    context = bootstrap()
    context.configure_tracing(
        integration=TraceIntegration.MLFLOW_AGENT_SERVER,
        policy=TracePolicy(),
    )
    return ToolAgent(context, prompt_version=required_prompt_version())


async def initialize_application() -> None:
    """Select instrumentation and validate resources during worker startup."""

    _application()


def required_prompt_version() -> int:
    """Return the immutable Prompt Registry version configured for this App."""

    raw_version = os.environ.get("AAI_PROMPT_VERSION")
    if (
        raw_version is None
        or not raw_version.isascii()
        or not raw_version.isdigit()
        or raw_version.startswith("0")
    ):
        raise RuntimeError(
            "AAI_PROMPT_VERSION must be an explicit positive Prompt Registry "
            "version. Set bundle variable prompt_version to the exact version "
            "that passed release_gate."
        )
    return int(raw_version)


@invoke()
async def invoke_agent(
    request: ResponsesAgentRequest,
) -> ResponsesAgentResponse:
    """Serve one Responses API request through the framework-neutral agent."""

    # AgentServer opens the trace before invoking this function. Replace its
    # broad default input immediately with the bounded, text-only contract;
    # user identifiers and arbitrary custom inputs are intentionally omitted.
    _set_bounded_root_inputs([], None)
    messages, conversation_id = _bounded_request(request)
    _set_bounded_root_inputs(messages, conversation_id)

    application = _application()
    with trace_context(session_id=conversation_id):
        set_trace_resource_context(application.context.tags)
        if conversation_id:
            set_trace_session(conversation_id)
        response = await application.ainvoke(
            AgentRequest(messages=messages, session_id=conversation_id),
        )
    return ResponsesAgentResponse(
        output=[
            ResponsesAgent.create_text_output_item(
                text=response.content,
                id="agent-answer",
            )
        ],
        custom_outputs=dict(response.metadata),
    )


@stream()
async def stream_agent(
    request: ResponsesAgentRequest,
) -> AsyncIterator[ResponsesAgentStreamEvent]:
    """Stream provider-native text deltas through the Responses API contract."""

    _set_bounded_root_inputs([], None)
    messages, conversation_id = _bounded_request(request)
    _set_bounded_root_inputs(messages, conversation_id)
    application = _application()
    with trace_context(session_id=conversation_id):
        set_trace_resource_context(application.context.tags)
        if conversation_id:
            set_trace_session(conversation_id)

        text: list[str] = []
        async with aclosing(
            application.astream_text(
                AgentRequest(messages=messages, session_id=conversation_id)
            )
        ) as deltas:
            async for delta in deltas:
                text.append(delta)
                yield ResponsesAgentStreamEvent(
                    **ResponsesAgent.create_text_delta(
                        delta=delta,
                        item_id="agent-answer",
                    )
                )
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=ResponsesAgent.create_text_output_item(
                text="".join(text),
                id="agent-answer",
            ),
        )


async def close_application() -> None:
    """Release the worker-local async provider client during server shutdown."""

    if _application.cache_info().currsize:
        await _application().aclose()
        _application.cache_clear()


def _bounded_request(
    request: ResponsesAgentRequest,
) -> tuple[list[dict[str, str]], str | None]:
    messages = [
        {"role": item.role, "content": response_message_text(item)}
        for item in request.input
        if getattr(item, "role", None) in {"user", "assistant"}
    ]
    return messages, _context_value(request.context, "conversation_id")


def _context_value(context: Any, field: str) -> str | None:
    if context is None:
        return None
    value = (
        context.get(field)
        if isinstance(context, Mapping)
        else getattr(context, field, None)
    )
    return value if isinstance(value, str) and value else None


def _set_bounded_root_inputs(
    messages: list[dict[str, str]],
    conversation_id: str | None,
) -> None:
    """Replace Agent Server's broad request capture with the public contract."""

    span = mlflow.get_current_active_span()
    if span is None:
        return
    safe_inputs: dict[str, Any] = {"input": messages}
    if conversation_id:
        safe_inputs["context"] = {"conversation_id": conversation_id}
    span.set_inputs(safe_inputs)
