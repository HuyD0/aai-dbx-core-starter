"""Framework-neutral agent application contract and tool execution."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from aai_core.exceptions import AaiCoreError

if TYPE_CHECKING:  # avoids a runtime import cycle with providers
    from aai_core.providers.types import ChatModel, ModelResponse


@dataclass(frozen=True)
class AgentRequest:
    messages: Sequence[Mapping[str, Any]]
    session_id: str | None = None
    user_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResponse:
    content: str
    trace_id: str | None = None
    citations: Sequence[Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class AgentApplication(Protocol):
    def invoke(self, request: AgentRequest) -> AgentResponse: ...


class ToolLoopError(AaiCoreError):
    code = "aai_core.agents.tool_loop"


@dataclass(frozen=True)
class ToolSpec:
    """A callable tool: OpenAI function-calling metadata plus its handler."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[..., Any]


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    arguments: Mapping[str, Any]
    result: str


@dataclass(frozen=True)
class ToolLoopResult:
    response: ModelResponse
    transcript: tuple[Mapping[str, Any], ...]
    tool_invocations: tuple[ToolInvocation, ...]

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(invocation.name for invocation in self.tool_invocations)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ToolLoopError(f"Tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": dict(spec.parameters),
                },
            }
            for spec in self._tools.values()
        ]

    def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        try:
            spec = self._tools[name]
        except KeyError as error:
            raise ToolLoopError(
                f"Model requested unknown tool {name!r}",
                remediation="Register the tool in the ToolRegistry or tighten "
                "the tool descriptions so the model stops inventing names.",
            ) from error
        result = spec.handler(**dict(arguments))
        return result if isinstance(result, str) else json.dumps(result)


def run_tool_loop(
    model: ChatModel,
    messages: Sequence[Mapping[str, Any]],
    registry: ToolRegistry,
    *,
    max_turns: int = 6,
    **options: Any,
) -> ToolLoopResult:
    """Drive the model↔tool conversation until a final answer.

    Each turn offers the registry's tools; requested calls are executed and
    appended as ``tool`` messages. Returns when the model answers without
    tool calls; raises :class:`ToolLoopError` if ``max_turns`` is exhausted
    (runaway loops are a cost bug, not a retry case).
    """

    transcript: list[Mapping[str, Any]] = list(messages)
    invocations: list[ToolInvocation] = []
    for _ in range(max_turns):
        response = model.generate(transcript, tools=registry.openai_tools(), **options)
        if not response.tool_calls:
            return ToolLoopResult(
                response=response,
                transcript=tuple(transcript),
                tool_invocations=tuple(invocations),
            )
        transcript.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [_serialize_tool_call(c) for c in response.tool_calls],
            }
        )
        for tool_call in response.tool_calls:
            function = tool_call.function
            arguments = json.loads(function.arguments or "{}")
            result = registry.execute(function.name, arguments)
            invocations.append(
                ToolInvocation(name=function.name, arguments=arguments, result=result)
            )
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(tool_call, "id", ""),
                    "content": result,
                }
            )
    raise ToolLoopError(
        f"Tool loop did not converge within {max_turns} turns",
        remediation="Raise max_turns deliberately if the workflow genuinely "
        "needs more steps, or tighten the system prompt / tool results so "
        "the model can finish.",
    )


def _serialize_tool_call(tool_call: Any) -> dict[str, Any]:
    function = tool_call.function
    return {
        "id": getattr(tool_call, "id", ""),
        "type": "function",
        "function": {"name": function.name, "arguments": function.arguments},
    }
