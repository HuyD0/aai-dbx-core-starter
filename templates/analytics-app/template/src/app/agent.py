"""The analytics runbook agent: an application-owned async tool loop.

Context assembly is deliberate: the system prompt carries the runbook, the
compact metric catalog, and the knowledge index summary — never the full
semantic YAML or whole reference docs (those load on demand through tools).
The provenance footer is rendered by code from the ProvenanceLog, so the
model can neither fabricate nor omit evidence. Token usage is accounted per
pass (main loop vs adversarial review) and returned with every answer so
evaluations gate on cost coverage without re-instrumenting.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from aai_core import PlatformContext, bootstrap
from aai_core.tracing import provider_span
from app.config import MAX_AGENT_TURNS, adversarial_review_enabled
from app.knowledge import KnowledgeRouter
from app.provenance import ProvenanceRecord, render_footer
from app.reviewer import (
    ReviewVerdict,
    apply_verdict,
    load_reviewer_messages,
    render_reviewer_messages,
    review_response_format,
)
from app.semantics.executor import WarehouseExecutor
from app.semantics.models import SemanticModel, load_semantic_model
from app.tools import ProvenanceLog, build_registry


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str
    caveats: tuple[str, ...] = ()


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    review_tokens: int = 0

    @property
    def captured(self) -> bool:
        return self.total_tokens > 0


class AnalyticsAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str
    prose: str
    records: tuple[ProvenanceRecord, ...]
    usage: TokenUsage
    tools_used: tuple[str, ...]
    reviewed: bool


class AnalyticsAgent:
    def __init__(
        self,
        context: PlatformContext | None = None,
        *,
        semantic_model: SemanticModel,
        knowledge: KnowledgeRouter,
        executor: WarehouseExecutor,
        system_messages: tuple[dict[str, Any], ...],
        reviewer_messages: tuple[dict[str, Any], ...] = (),
        enable_review: bool | None = None,
        async_client: Any | None = None,
    ) -> None:
        self.context = context or bootstrap()
        self.model = self.context.providers.model("general-chat")
        self.semantic_model = semantic_model
        self.knowledge = knowledge
        self.executor = executor
        self._system_messages = system_messages
        self._reviewer_messages = reviewer_messages
        if enable_review is None:
            enable_review = adversarial_review_enabled()
        self.enable_review = enable_review and bool(reviewer_messages)
        self._async_client = async_client
        self._owns_async_client = async_client is None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def from_project(
        cls,
        root: str | Path,
        context: PlatformContext | None = None,
        *,
        executor: WarehouseExecutor,
        enable_review: bool | None = None,
        async_client: Any | None = None,
    ) -> AnalyticsAgent:
        """Wire the agent from a generated project checkout/bundle sync."""

        root = Path(root)
        return cls(
            context,
            semantic_model=load_semantic_model(
                root / "semantics" / "semantic_model.yml"
            ),
            knowledge=KnowledgeRouter(root / "knowledge"),
            executor=executor,
            system_messages=_load_messages(root / "prompts" / "system_prompt.json"),
            reviewer_messages=load_reviewer_messages(
                root / "prompts" / "reviewer_prompt.json"
            ),
            enable_review=enable_review,
            async_client=async_client,
        )

    async def aanswer(self, question: str) -> AnalyticsAnswer:
        if not question.strip():
            raise ValueError("question must not be empty")
        usage = _UsageAccumulator()
        log = ProvenanceLog()
        registry = build_registry(
            self.semantic_model, self.knowledge, self.executor, log
        )
        client = self._client()
        transcript = self._render_system(question)
        tools_used: list[str] = []

        final_message = None
        for _ in range(MAX_AGENT_TURNS):
            response = await self._complete(
                client, transcript, usage, "main", tools=registry.openai_tools()
            )
            message = response.choices[0].message
            tool_calls = tuple(getattr(message, "tool_calls", None) or ())
            if not tool_calls:
                final_message = message
                break
            transcript.append(_assistant_tool_message(message))
            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise TypeError(f"Tool arguments for {name!r} must be an object")
                output = await registry.execute(name, arguments)
                tools_used.append(name)
                transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    }
                )
        if final_message is None:
            raise RuntimeError(
                f"Tool loop did not converge within {MAX_AGENT_TURNS} turns; "
                "tighten the runbook or tool results before raising the bound."
            )

        response = await self._complete(
            client,
            [
                *transcript,
                {
                    "role": "user",
                    "content": "Return the final answer as the requested JSON "
                    "object.",
                },
            ],
            usage,
            "main",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": FinalAnswer.__name__,
                    "schema": FinalAnswer.model_json_schema(),
                    "strict": True,
                },
            },
        )
        structured = FinalAnswer.model_validate_json(
            response.choices[0].message.content or ""
        )
        prose = structured.answer.strip()
        if structured.caveats:
            caveats = " ".join(structured.caveats)
            prose = f"{prose}\n\nCaveats: {caveats}"

        records = log.finalize()
        footer = render_footer(records)
        reviewed = False
        if self.enable_review:
            prose = await self._review(client, question, prose, footer, usage)
            reviewed = True

        answer = f"{prose}\n\n{footer}" if footer else prose
        return AnalyticsAnswer(
            answer=answer,
            prose=prose,
            records=records,
            usage=usage.snapshot(),
            tools_used=tuple(tools_used),
            reviewed=reviewed,
        )

    async def aclose(self) -> None:
        if self._owns_async_client:
            await _close_async_resource(self._async_client)
        self._async_client = None
        self._client_loop = None

    async def _review(
        self,
        client: Any,
        question: str,
        prose: str,
        evidence: str,
        usage: _UsageAccumulator,
    ) -> str:
        messages = render_reviewer_messages(
            self._reviewer_messages,
            question=question,
            answer=prose,
            evidence=evidence or "no evidence recorded",
        )
        response = await self._complete(
            client,
            messages,
            usage,
            "review",
            response_format=review_response_format(),
        )
        verdict = ReviewVerdict.model_validate_json(
            response.choices[0].message.content or ""
        )
        return apply_verdict(prose, verdict)

    def _render_system(self, question: str) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for message in self._system_messages:
            content = str(message.get("content", ""))
            content = content.replace(
                "{{metric_catalog}}", self.semantic_model.metric_catalog()
            )
            content = content.replace(
                "{{knowledge_index}}", self.knowledge.index_summary()
            )
            content = content.replace("{{question}}", question)
            rendered.append({"role": message["role"], "content": content})
        if not rendered or rendered[0].get("role") != "system":
            raise ValueError("the system prompt file must start with a system role")
        if rendered[-1].get("role") != "user":
            raise ValueError("the system prompt file must end with a user message")
        return rendered

    async def _complete(
        self,
        client: Any,
        messages: list[Mapping[str, Any]],
        usage: _UsageAccumulator,
        usage_pass: str,
        **options: Any,
    ) -> Any:
        with provider_span(
            "model.generate",
            span_type="LLM",
            attributes={
                "aai.provider": self.model.provider,
                "aai.logical_name": self.model.logical_name,
                "aai.model": self.model.model,
                "mlflow.llm.provider": self.model.provider,
                "mlflow.llm.model": self.model.model,
                "mlflow.message.format": "openai",
            },
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
            mapped = _usage_mapping(getattr(response, "usage", None))
            usage.add(mapped, usage_pass)
            if span is not None:
                content = response.choices[0].message.content or ""
                span.set_outputs({"content": content})
                if mapped:
                    span.set_attribute("mlflow.chat.tokenUsage", mapped)
            return response

    def _client(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._client_loop is not None and self._client_loop is not loop:
            raise RuntimeError(
                "AnalyticsAgent async clients cannot be shared across event loops"
            )
        if self._async_client is None:
            self._async_client = self.model.create_native_async_client()
        self._client_loop = loop
        return self._async_client


class _UsageAccumulator:
    def __init__(self) -> None:
        self._input = 0
        self._output = 0
        self._total = 0
        self._review = 0
        self._captured_calls = 0
        self._calls = 0

    def add(self, mapped: Mapping[str, int], usage_pass: str) -> None:
        self._calls += 1
        if not mapped:
            return
        self._captured_calls += 1
        self._input += mapped.get("input_tokens", 0)
        self._output += mapped.get("output_tokens", 0)
        total = mapped.get("total_tokens", 0)
        self._total += total
        if usage_pass == "review":
            self._review += total

    def snapshot(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self._input,
            output_tokens=self._output,
            total_tokens=self._total,
            review_tokens=self._review,
        )


def _load_messages(path: str | Path) -> tuple[dict[str, Any], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(dict(message) for message in payload["messages"])


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


async def _close_async_resource(resource: Any | None) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None) or getattr(resource, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result
