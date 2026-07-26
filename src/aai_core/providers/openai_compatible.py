"""OpenAI-compatible model and embedding adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from time import monotonic
from typing import Any

from aai_core.providers.types import (
    ModelCapabilities,
    ModelResponse,
    ProviderConfigurationError,
    ProviderRequestError,
    UnsupportedCapabilityError,
)
from aai_core.tracing import provider_span

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
_CONTROLLED_CHAT_OPTIONS = {
    "max_tokens",
    "messages",
    "model",
    "response_format",
    "stream",
    "temperature",
    "timeout",
    "tools",
}
_FORBIDDEN_PER_CALL_OPTIONS = {
    "default_headers",
    "extra_body",
    "extra_headers",
    "headers",
}

_REMEDIATIONS = {
    401: "The request was not authenticated. Verify your keyless login "
    "(az login / DATABRICKS_AUTH_TYPE) and, behind a gateway, the token "
    "scope and subscription key reference in aai-platform.yml.",
    403: "The identity is authenticated but not authorized. Ask for "
    "CAN_QUERY on the serving endpoint (Databricks) or the required "
    "role/product subscription (Foundry/APIM).",
    404: "The configured deployment/endpoint does not exist. Check the "
    "`deployment` (and `base_url`/`endpoint`) for this logical name in "
    "aai-platform.yml.",
    429: "The provider or gateway rate limit was exhausted after retries. "
    "Reduce request volume or ask the platform team about the rate limit "
    "for your team.",
}


def _status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def _call_provider(
    operation: Callable[[], Any],
    *,
    description: str,
    provider: str,
    logical_name: str,
) -> Any:
    """Translate a final native-client failure without duplicating retries."""

    try:
        return operation()
    except Exception as error:
        status = _status_code(error)
        raise ProviderRequestError(
            f"{description} failed for {logical_name!r} via {provider}: {error}",
            remediation=_REMEDIATIONS.get(status) if status else None,
        ) from error


def _reject_running_event_loop(logical_name: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise UnsupportedCapabilityError(
        f"{logical_name!r} generate() is synchronous and cannot run on an "
        "active event loop",
        remediation="Create a provider-native async client with "
        "model.create_native_async_client() and await the native SDK call.",
    )


class OpenAICompatibleChatModel:
    def __init__(
        self,
        *,
        logical_name: str,
        provider: str,
        model: str,
        client: Any,
        async_client_factory: Callable[[], Any] | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.logical_name = logical_name
        self.provider = provider
        self.model = model
        self.native_client = client
        self._async_client_factory = async_client_factory
        self.capabilities = capabilities or ModelCapabilities()

    def create_native_async_client(self) -> Any:
        """Return a new native async client; the caller must close it."""

        if self._async_client_factory is None:
            raise UnsupportedCapabilityError(
                f"{self.logical_name!r} has no configured native async client",
                remediation="Register a model that supplies async_client_factory "
                "or use generate() outside an event loop.",
            )
        return self._async_client_factory()

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
        _reject_running_event_loop(self.logical_name)
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
            collisions = set(provider_options) & _CONTROLLED_CHAT_OPTIONS
            if collisions:
                raise ProviderConfigurationError(
                    "provider_options cannot override controlled chat fields: "
                    f"{', '.join(sorted(collisions))}",
                    remediation="Use the corresponding generate() argument; "
                    "streaming remains available only through native_client.",
                )
            forbidden = set(provider_options) & _FORBIDDEN_PER_CALL_OPTIONS
            if forbidden:
                raise ProviderConfigurationError(
                    "Per-call provider headers and raw request bodies are not allowed",
                    remediation="Configure keyless authentication or a governed "
                    "secret reference on the provider client. Use explicit "
                    "generate() arguments for request fields; never pass "
                    "credentials or extra_body through provider_options.",
                )
            request.update(provider_options)
        started = monotonic()
        with provider_span(
            "model.generate",
            span_type="LLM",
            attributes={
                "aai.provider": self.provider,
                "aai.logical_name": self.logical_name,
                "aai.model": self.model,
                "mlflow.llm.provider": self.provider,
                "mlflow.llm.model": self.model,
                "mlflow.message.format": "openai",
            },
        ) as span:
            if span is not None:
                trace_inputs: dict[str, Any] = {"messages": list(messages)}
                if tools:
                    # MLflow's agent scorers discover the available tool
                    # definitions from the LLM span's standard ``tools``
                    # input. This bounded manual span excludes additive
                    # provider-only options; framework autologging is a
                    # separate, explicit data-policy decision.
                    trace_inputs["tools"] = list(tools)
                span.set_inputs(trace_inputs)
            response = _call_provider(
                lambda: self.native_client.chat.completions.create(**request),
                description="chat completion",
                provider=self.provider,
                logical_name=self.logical_name,
            )

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
            if span is not None:
                span.set_outputs({"content": content})
                if canonical_usage := _canonical_token_usage(normalized_usage):
                    span.set_attribute("mlflow.chat.tokenUsage", canonical_usage)
        elapsed = (monotonic() - started) * 1000

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
        request: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.dimensions:
            request["dimensions"] = self.dimensions
        with provider_span(
            "embedding.generate",
            span_type="EMBEDDING",
            attributes={
                "aai.provider": self.provider,
                "aai.logical_name": self.logical_name,
                "aai.model": self.model,
                "mlflow.llm.provider": self.provider,
                "mlflow.llm.model": self.model,
            },
        ):
            response = _call_provider(
                lambda: self.native_client.embeddings.create(**request),
                description="embedding",
                provider=self.provider,
                logical_name=self.logical_name,
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


def _canonical_token_usage(usage: Mapping[str, int]) -> dict[str, int]:
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")
    canonical = {}
    if isinstance(input_tokens, int):
        canonical["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        canonical["output_tokens"] = output_tokens
    if isinstance(total_tokens, int):
        canonical["total_tokens"] = total_tokens
    elif isinstance(input_tokens, int) and isinstance(output_tokens, int):
        canonical["total_tokens"] = input_tokens + output_tokens
    return canonical
