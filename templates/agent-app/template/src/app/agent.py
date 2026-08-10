"""Application-owned async tool loop over the provider-native OpenAI client."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aai_core import PlatformContext, bootstrap
from aai_core.agents import (
    AgentDecision,
    AgentDecisionType,
    AgentRequest,
    AgentResponse,
)
from aai_core.providers import ChatModel
from aai_core.tracing import provider_span, record_agent_decision, set_trace_session
from app.config import PROMPT_NAME
from app.controls import DEFAULT_AGENT_LIMITS, AgentLimits
from app.schemas import FinalAnswer
from app.tools import build_agent_registry


@dataclass(frozen=True)
class PreparedConversation:
    messages: tuple[Mapping[str, Any], ...]
    tools_used: tuple[str, ...]


class ToolAgent:
    def __init__(
        self,
        context: PlatformContext | None = None,
        *,
        prompt_version: int | None = None,
        async_client: Any | None = None,
        limits: AgentLimits = DEFAULT_AGENT_LIMITS,
    ) -> None:
        self.context = context or bootstrap()
        self.model = self.context.providers.model("general-chat")
        self._native_model_name = _native_model_name(self.model)
        self.limits = limits
        self.registry = build_agent_registry(
            timeout_seconds=limits.tool_timeout_seconds,
            max_output_chars=limits.max_tool_output_chars,
        )
        self._async_client = async_client
        self._owns_async_client = async_client is None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        if prompt_version is not None:
            resolved = self.context.prompts.load(PROMPT_NAME, version=prompt_version)
        else:
            alias = (
                "production"
                if self.context.settings.resource.environment in {"prod", "production"}
                else "development"
            )
            resolved = self.context.prompts.load(PROMPT_NAME, alias=alias)
        self.prompt_version = _positive_prompt_version(resolved)

    async def ainvoke(self, request: AgentRequest) -> AgentResponse:
        deadline = asyncio.timeout(self.limits.request_deadline_seconds)
        try:
            async with deadline:
                return await self._ainvoke(request)
        except TimeoutError as error:
            if not deadline.expired():
                raise
            raise RuntimeError(
                "Agent request exceeded its centrally configured deadline of "
                f"{self.limits.request_deadline_seconds:g} seconds"
            ) from error

    async def _ainvoke(self, request: AgentRequest) -> AgentResponse:
        _validate_request_bounds(request, self.limits)
        if request.session_id:
            set_trace_session(request.session_id)
        prompt = await asyncio.to_thread(self._request_prompt)
        prepared = await self._prepare(request, prompt)
        client = self._client()
        schema = FinalAnswer.model_json_schema()
        response = await self._complete(
            client,
            [
                *prepared.messages,
                {
                    "role": "user",
                    "content": "Return the final answer as the requested JSON object.",
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": FinalAnswer.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        content = response.choices[0].message.content or ""
        structured = FinalAnswer.model_validate_json(content)
        return AgentResponse(
            content=structured.answer,
            metadata={
                "confidence": structured.confidence,
                "tools_used": list(prepared.tools_used),
                "model": str(getattr(response, "model", self._native_model_name)),
                "usage": _usage_mapping(getattr(response, "usage", None)),
            },
        )

    async def astream_text(self, request: AgentRequest) -> AsyncGenerator[str, None]:
        """Yield native provider text deltas; no SDK stream type is introduced."""

        loop = asyncio.get_running_loop()
        expires_at = loop.time() + self.limits.request_deadline_seconds
        stream = self._astream_text(request)
        try:
            while True:
                if loop.time() >= expires_at:
                    raise _request_deadline_error(self.limits) from None
                try:
                    # The timeout covers provider/application work for this
                    # pull, never the period after a delta is yielded to a slow
                    # consumer. Every pull shares the same absolute deadline.
                    async with asyncio.timeout_at(expires_at):
                        delta = await anext(stream)
                except StopAsyncIteration:
                    return
                except TimeoutError as error:
                    raise _request_deadline_error(self.limits) from error
                yield delta
        finally:
            await stream.aclose()

    async def _astream_text(self, request: AgentRequest) -> AsyncGenerator[str, None]:
        _validate_request_bounds(request, self.limits)
        if request.session_id:
            set_trace_session(request.session_id)
        prompt = await asyncio.to_thread(self._request_prompt)
        prepared = await self._prepare(request, prompt)
        client = self._client()
        stream = None
        with provider_span(
            "model.stream",
            span_type="LLM",
            attributes=_model_span_attributes(self.model, self._native_model_name),
        ) as span:
            if span is not None:
                span.set_inputs({"messages": list(prepared.messages)})
            try:
                stream = await client.chat.completions.create(
                    model=self._native_model_name,
                    messages=[
                        *prepared.messages,
                        {
                            "role": "user",
                            "content": "Provide the final answer in plain text.",
                        },
                    ],
                    temperature=0.0,
                    max_tokens=self.limits.max_output_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                chunks: list[str] = []
                output_chars = 0
                async for event in stream:
                    usage = _usage_mapping(getattr(event, "usage", None))
                    if span is not None and usage:
                        span.set_attribute("mlflow.chat.tokenUsage", usage)
                    choices = getattr(event, "choices", None) or ()
                    if not choices:
                        continue
                    text = getattr(choices[0].delta, "content", None)
                    if isinstance(text, str) and text:
                        output_chars += len(text)
                        if output_chars > self.limits.max_stream_output_chars:
                            raise RuntimeError(
                                "Agent stream exceeded the centrally configured "
                                f"{self.limits.max_stream_output_chars}-character "
                                "output bound"
                            )
                        chunks.append(text)
                        yield text
                if span is not None:
                    span.set_outputs({"content": "".join(chunks)})
            finally:
                await _close_async_resource(stream)

    async def aclose(self) -> None:
        if self._owns_async_client:
            await _close_async_resource(self._async_client)
        self._async_client = None
        self._client_loop = None

    async def _prepare(
        self, request: AgentRequest, prompt: Any
    ) -> PreparedConversation:
        transcript = _conversation_messages(prompt, request)
        tools_used: list[str] = []
        client = self._client()
        for _ in range(self.limits.max_tool_turns):
            response = await self._complete(
                client,
                transcript,
                tools=self.registry.openai_tools(),
            )
            message = response.choices[0].message
            tool_calls = tuple(getattr(message, "tool_calls", None) or ())
            if not tool_calls:
                _record_convergence_decision(tool_evidence_exists=bool(tools_used))
                return PreparedConversation(
                    messages=tuple(transcript),
                    tools_used=tuple(tools_used),
                )
            if len(tool_calls) > self.limits.max_tool_calls_per_turn:
                raise RuntimeError(
                    "Model requested more than the centrally configured "
                    f"{self.limits.max_tool_calls_per_turn} tool calls in one turn"
                )
            if len(tools_used) + len(tool_calls) > self.limits.max_total_tool_calls:
                raise RuntimeError(
                    "Agent exceeded the centrally configured "
                    f"{self.limits.max_total_tool_calls} total tool-call bound"
                )
            transcript.append(_assistant_tool_message(message))
            for tool_call in tool_calls:
                name = tool_call.function.name
                _record_tool_selection(name)
                arguments = json.loads(tool_call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise TypeError(f"Tool arguments for {name!r} must be an object")
                output = await self.registry.execute(name, arguments)
                tools_used.append(name)
                transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    }
                )
        raise RuntimeError(
            "Tool loop did not converge within "
            f"{self.limits.max_tool_turns} turns; tighten the prompt or tool "
            "results before increasing the cost bound."
        )

    async def _complete(
        self,
        client: Any,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> Any:
        options.setdefault("max_tokens", self.limits.max_output_tokens)
        with provider_span(
            "model.generate",
            span_type="LLM",
            attributes=_model_span_attributes(self.model, self._native_model_name),
        ) as span:
            if span is not None:
                inputs: dict[str, Any] = {"messages": list(messages)}
                if options.get("tools"):
                    inputs["tools"] = options["tools"]
                span.set_inputs(inputs)
            response = await client.chat.completions.create(
                model=self._native_model_name,
                messages=list(messages),
                temperature=0.0,
                **options,
            )
            if span is not None:
                content = response.choices[0].message.content or ""
                span.set_outputs({"content": content})
                if usage := _usage_mapping(getattr(response, "usage", None)):
                    span.set_attribute("mlflow.chat.tokenUsage", usage)
            return response

    def _client(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._client_loop is not None and self._client_loop is not loop:
            raise RuntimeError(
                "ToolAgent async clients cannot be shared across event loops"
            )
        if self._async_client is None:
            self._async_client = self.model.create_native_async_client()
        self._client_loop = loop
        return self._async_client

    def _request_prompt(self) -> Any:
        # Load the immutable version while the request trace is active so
        # MLflow records prompt lineage. MLflow caches this small registry
        # metadata lookup for five minutes; the version itself never moves.
        return self.context.prompts.load(
            PROMPT_NAME,
            version=self.prompt_version,
            cache_ttl_seconds=300.0,
        )


def _positive_prompt_version(prompt: Any) -> int:
    try:
        version = int(prompt.version)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "The resolved Prompt Registry object requires an immutable version"
        ) from error
    if version < 1:
        raise RuntimeError("Prompt Registry versions must be positive integers")
    return version


def _request_deadline_error(limits: AgentLimits) -> RuntimeError:
    return RuntimeError(
        "Agent request exceeded its centrally configured deadline of "
        f"{limits.request_deadline_seconds:g} seconds"
    )


def _record_tool_selection(name: str) -> None:
    """Record the provider's explicit action without predicting its outcome."""

    try:
        decision = AgentDecision(
            decision_type=AgentDecisionType.TOOL_SELECTION,
            goal="Progress the request with an available action.",
            selected_action=name,
            reason="The provider response explicitly requested this tool.",
            evidence_refs=("provider_tool_calls",),
        )
    except ValueError:
        # A provider-supplied malformed name must still reach the registry's
        # existing validation path; observability cannot replace that error.
        return
    record_agent_decision(decision)


def _record_convergence_decision(*, tool_evidence_exists: bool) -> None:
    """Record the observable stop condition, not hidden model reasoning."""

    if tool_evidence_exists:
        decision = AgentDecision(
            decision_type=AgentDecisionType.EVIDENCE_SUFFICIENCY,
            goal="Determine whether more tool evidence is needed.",
            selected_action="answer",
            reason=(
                "The provider requested no additional tool calls after observed "
                "tool results."
            ),
            evidence_refs=("provider_tool_calls", "observed_tool_results"),
        )
    else:
        decision = AgentDecision(
            decision_type=AgentDecisionType.ANSWER_READINESS,
            goal="Determine whether the response is ready to produce.",
            selected_action="answer",
            reason="The provider requested no tool call for this request.",
            evidence_refs=("user_request", "provider_tool_calls"),
        )
    record_agent_decision(decision)


def _validate_request_bounds(request: AgentRequest, limits: AgentLimits) -> None:
    messages = [
        message
        for message in request.messages
        if message.get("role") in {"user", "assistant"}
    ]
    if len(messages) > limits.max_input_messages:
        raise ValueError(
            f"AgentRequest exceeds the {limits.max_input_messages}-message bound"
        )
    total = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("AgentRequest user/assistant messages must contain text")
        if len(content) > limits.max_message_chars:
            raise ValueError(
                "AgentRequest contains a message longer than the centrally "
                f"configured {limits.max_message_chars}-character bound"
            )
        total += len(content)
    if total > limits.max_total_input_chars:
        raise ValueError(
            "AgentRequest exceeds the centrally configured "
            f"{limits.max_total_input_chars}-character total input bound"
        )


def _conversation_messages(
    prompt: Any, request: AgentRequest
) -> list[Mapping[str, Any]]:
    history: list[Mapping[str, Any]] = []
    for message in request.messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("AgentRequest user/assistant messages must contain text")
        history.append({"role": role, "content": content})
    if (
        not history
        or history[-1]["role"] != "user"
        or not history[-1]["content"].strip()
    ):
        raise ValueError("AgentRequest must end with a non-empty user message")
    formatted = [
        dict(message) for message in prompt.format(question=history[-1]["content"])
    ]
    if not formatted or formatted[-1].get("role") != "user":
        raise ValueError("The registered prompt must end with a user message")
    governed = formatted[:-1]
    if not any(message.get("role") == "system" for message in governed):
        raise ValueError("The registered prompt requires a system message")
    return [*governed, *history[:-1], formatted[-1]]


def _assistant_tool_message(message: Any) -> dict[str, Any]:
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        rendered = dump(exclude_none=True)
        if not isinstance(rendered, Mapping):
            raise TypeError("provider messages must serialize to an object")
        return dict(rendered)
    return {
        "role": "assistant",
        "content": getattr(message, "content", None) or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ],
    }


def _usage_mapping(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    raw = value.model_dump() if callable(getattr(value, "model_dump", None)) else value
    if not isinstance(raw, Mapping):
        raw = {
            name: getattr(value, name, None)
            for name in (
                "input_tokens",
                "prompt_tokens",
                "output_tokens",
                "completion_tokens",
                "total_tokens",
            )
        }
    input_tokens = raw.get("input_tokens", raw.get("prompt_tokens"))
    output_tokens = raw.get("output_tokens", raw.get("completion_tokens"))
    total_tokens = raw.get("total_tokens")
    result: dict[str, int] = {}
    if isinstance(input_tokens, int):
        result["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        result["output_tokens"] = output_tokens
    if isinstance(total_tokens, int):
        result["total_tokens"] = total_tokens
    elif isinstance(input_tokens, int) and isinstance(output_tokens, int):
        result["total_tokens"] = input_tokens + output_tokens
    return result


def _native_model_name(model: ChatModel) -> str:
    value = getattr(model, "model", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError("the native chat client requires a configured model name")
    return value


def _model_span_attributes(model: ChatModel, native_model_name: str) -> dict[str, str]:
    return {
        "aai.provider": model.provider,
        "aai.logical_name": model.logical_name,
        "aai.model": native_model_name,
        "mlflow.llm.provider": model.provider,
        "mlflow.llm.model": native_model_name,
        "mlflow.message.format": "openai",
    }


async def _close_async_resource(resource: Any | None) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None) or getattr(resource, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        _ = await result
