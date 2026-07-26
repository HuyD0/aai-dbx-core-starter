"""Application-owned async tool loop over the provider-native OpenAI client."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from aai_core import PlatformContext, bootstrap
from aai_core.agents import AgentRequest, AgentResponse
from aai_core.tracing import provider_span, set_trace_session
from app.config import PROMPT_NAME
from app.schemas import FinalAnswer
from app.tools import build_registry


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
    ) -> None:
        self.context = context or bootstrap()
        self.model = self.context.providers.model("general-chat")
        self.registry = build_registry()
        self._async_client = async_client
        self._owns_async_client = async_client is None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        if prompt_version is not None:
            self.prompt = self.context.prompts.load(PROMPT_NAME, version=prompt_version)
        else:
            alias = (
                "production"
                if self.context.settings.resource.environment in {"prod", "production"}
                else "development"
            )
            self.prompt = self.context.prompts.load(PROMPT_NAME, alias=alias)

    async def ainvoke(self, request: AgentRequest) -> AgentResponse:
        if request.session_id:
            set_trace_session(request.session_id)
        prepared = await self._prepare(request)
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
                "model": str(getattr(response, "model", self.model.model)),
                "usage": _usage_mapping(getattr(response, "usage", None)),
            },
        )

    async def astream_text(self, request: AgentRequest) -> AsyncIterator[str]:
        """Yield native provider text deltas; no SDK stream type is introduced."""

        if request.session_id:
            set_trace_session(request.session_id)
        prepared = await self._prepare(request)
        client = self._client()
        stream = None
        with provider_span(
            "model.stream",
            span_type="LLM",
            attributes=_model_span_attributes(self.model),
        ) as span:
            if span is not None:
                span.set_inputs({"messages": list(prepared.messages)})
            try:
                stream = await client.chat.completions.create(
                    model=self.model.model,
                    messages=[
                        *prepared.messages,
                        {
                            "role": "user",
                            "content": "Provide the final answer in plain text.",
                        },
                    ],
                    temperature=0.0,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                chunks: list[str] = []
                async for event in stream:
                    usage = _usage_mapping(getattr(event, "usage", None))
                    if span is not None and usage:
                        span.set_attribute("mlflow.chat.tokenUsage", usage)
                    choices = getattr(event, "choices", None) or ()
                    if not choices:
                        continue
                    text = getattr(choices[0].delta, "content", None)
                    if isinstance(text, str) and text:
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

    async def _prepare(self, request: AgentRequest) -> PreparedConversation:
        transcript: list[Mapping[str, Any]] = _conversation_messages(
            self.prompt, request
        )
        tools_used: list[str] = []
        client = self._client()
        for _ in range(6):
            response = await self._complete(
                client,
                transcript,
                tools=self.registry.openai_tools(),
            )
            message = response.choices[0].message
            tool_calls = tuple(getattr(message, "tool_calls", None) or ())
            if not tool_calls:
                return PreparedConversation(
                    messages=tuple(transcript),
                    tools_used=tuple(tools_used),
                )
            transcript.append(_assistant_tool_message(message))
            for tool_call in tool_calls:
                name = tool_call.function.name
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
            "Tool loop did not converge within six turns; tighten the prompt or "
            "tool results before increasing the cost bound."
        )

    async def _complete(
        self,
        client: Any,
        messages: list[Mapping[str, Any]],
        **options: Any,
    ) -> Any:
        with provider_span(
            "model.generate",
            span_type="LLM",
            attributes=_model_span_attributes(self.model),
        ) as span:
            if span is not None:
                inputs: dict[str, Any] = {"messages": list(messages)}
                if options.get("tools"):
                    inputs["tools"] = options["tools"]
                span.set_inputs(inputs)
            response = await client.chat.completions.create(
                model=self.model.model,
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


def _conversation_messages(prompt: Any, request: AgentRequest) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
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


def _assistant_tool_message(message: Any) -> Mapping[str, Any]:
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
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


def _model_span_attributes(model: Any) -> dict[str, str]:
    return {
        "aai.provider": model.provider,
        "aai.logical_name": model.logical_name,
        "aai.model": model.model,
        "mlflow.llm.provider": model.provider,
        "mlflow.llm.model": model.model,
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
        await result
