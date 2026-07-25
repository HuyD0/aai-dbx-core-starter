"""OpenAI-compatible model and embedding adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import monotonic
from typing import Any

from aai_core.providers.types import (
    ModelCapabilities,
    ModelResponse,
    UnsupportedCapabilityError,
)
from aai_core.tracing import provider_span


class OpenAICompatibleChatModel:
    def __init__(
        self,
        *,
        logical_name: str,
        provider: str,
        model: str,
        client: Any,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.logical_name = logical_name
        self.provider = provider
        self.model = model
        self.native_client = client
        self.capabilities = capabilities or ModelCapabilities()

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Mapping[str, Any] | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        if tools and not self.capabilities.tool_calling:
            raise UnsupportedCapabilityError(
                f"{self.logical_name} does not support tool calling"
            )
        if response_format and not self.capabilities.structured_output:
            raise UnsupportedCapabilityError(
                f"{self.logical_name} does not support structured output"
            )

        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
        }
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if tools:
            request["tools"] = list(tools)
        if response_format:
            request["response_format"] = dict(response_format)
        if provider_options:
            request.update(provider_options)

        started = monotonic()
        with provider_span(
            "model.generate",
            span_type="LLM",
            attributes={
                "aai.provider": self.provider,
                "aai.logical_name": self.logical_name,
                "aai.model": self.model,
            },
        ):
            response = self.native_client.chat.completions.create(**request)
        elapsed = (monotonic() - started) * 1000

        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        if not isinstance(content, str):
            content = str(content)
        usage = _as_mapping(getattr(response, "usage", None))
        normalized_usage = {
            str(key): int(value)
            for key, value in usage.items()
            if isinstance(value, (int, float))
        }
        return ModelResponse(
            content=content,
            provider=self.provider,
            logical_name=self.logical_name,
            model=str(getattr(response, "model", self.model)),
            latency_ms=elapsed,
            usage=normalized_usage,
            tool_calls=tuple(getattr(message, "tool_calls", None) or ()),
            raw=response,
        )


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        logical_name: str,
        provider: str,
        model: str,
        client: Any,
        dimensions: int | None = None,
    ) -> None:
        self.logical_name = logical_name
        self.provider = provider
        self.model = model
        self.native_client = client
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        with provider_span(
            "embedding.generate",
            span_type="EMBEDDING",
            attributes={
                "aai.provider": self.provider,
                "aai.logical_name": self.logical_name,
                "aai.model": self.model,
            },
        ):
            response = self.native_client.embeddings.create(
                model=self.model,
                input=list(texts),
                **({"dimensions": self.dimensions} if self.dimensions else {}),
            )
        return [list(item.embedding) for item in response.data]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {
        key: getattr(value, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(value, key)
    }
