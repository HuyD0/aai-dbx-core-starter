"""OpenAI-compatible model and embedding adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from time import monotonic
from typing import Any, cast

from aai_core.providers.types import (
    ModelCapabilities,
    ModelResponse,
    ProviderConfigurationError,
    ProviderRequestError,
    UnsupportedCapabilityError,
)
from aai_core.tracing import provider_span

__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT_SECONDS",
    "OpenAICompatibleChatModel",
    "OpenAICompatibleEmbeddingProvider",
]

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
    "role/product subscription (APIM).",
    404: "The configured deployment/endpoint does not exist. Check the "
    "`deployment` (and `base_url`/`endpoint`) for this logical name in "
    "aai-platform.yml.",
    429: "The provider or gateway rate limit was exhausted after retries. "
    "Reduce request volume or ask the platform team about the rate limit "
    "for your team.",
}


def _status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


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
    except Exception as native_error:
        status = _status_code(native_error)
        safe_provider = _safe_diagnostic_identifier(provider)
        safe_logical_name = _safe_diagnostic_identifier(logical_name)
        safe_operation = _safe_diagnostic_identifier(description.replace(" ", "_"))
        status_detail = f", status={status}" if status is not None else ""
        failure = ProviderRequestError(
            "Provider request failed "
            f"(provider={safe_provider}, operation={safe_operation}, "
            f"resource={safe_logical_name}{status_detail})",
            provider=safe_provider,
            operation=safe_operation,
            logical_name=safe_logical_name,
            status_code=status,
            remediation=_REMEDIATIONS.get(status) if status else None,
        )
    # Raise outside the native exception handler so no credential-bearing
    # native exception remains reachable through __context__.
    raise failure from None


def _safe_diagnostic_identifier(value: str) -> str:
    if 0 < len(value) <= 128 and all(
        character.isascii()
        and (character.isalnum() or character in {"-", "_", ".", ":", "/"})
        for character in value
    ):
        return value
    return "[invalid]"


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
    """Stable synchronous chat adapter over a caller-inspectable native client."""

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
        request = self._build_request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            response_format=response_format,
            provider_options=provider_options,
        )
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
            content, normalized_usage, message = _normalize_chat_response(response)
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

    def _build_request(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None,
        max_tokens: int | None,
        tools: Sequence[Mapping[str, Any]] | None,
        response_format: Mapping[str, Any] | None,
        provider_options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
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
        optional_fields = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": list(tools) if tools else None,
            "response_format": dict(response_format) if response_format else None,
        }
        request.update(
            {key: value for key, value in optional_fields.items() if value is not None}
        )
        self._add_provider_options(request, provider_options)
        return request

    @staticmethod
    def _add_provider_options(
        request: dict[str, Any],
        provider_options: Mapping[str, Any] | None,
    ) -> None:
        if not provider_options:
            return
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


class OpenAICompatibleEmbeddingProvider:
    """Stable embedding adapter over an OpenAI-compatible native client."""

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
        ) as span:
            response = _call_provider(
                lambda: self.native_client.embeddings.create(**request),
                description="embedding",
                provider=self.provider,
                logical_name=self.logical_name,
            )
            if span is not None:
                input_tokens = _embedding_input_tokens(response)
                if input_tokens is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        return [list(item.embedding) for item in response.data]


def _embedding_input_tokens(response: Any) -> int | None:
    """Billed input tokens for an embedding request, when the provider reports them.

    Deliberately *not* ``mlflow.chat.tokenUsage``. MLflow aggregates that key
    across every span type into the authoritative trace-level total, and
    ``aai_core.agentkit.economics`` prices that total at the project's
    configured rate pair for the *agent's* chat model. Embedding tokens are
    billed at a different rate, so folding them into the chat aggregate would
    over-state the cost of every retrieval query. The OpenTelemetry GenAI
    attribute keeps the evidence on the span, where the trace and per-model
    pricing can read it, without entering that aggregate.

    Embeddings have no generated tokens, so there is no output side to record.
    """

    usage = _as_mapping(getattr(response, "usage", None))
    tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
        return tokens
    return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return cast(Mapping[str, Any], value.model_dump())
    return {
        key: getattr(value, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(value, key)
    }


def _normalize_chat_response(response: Any) -> tuple[str, dict[str, int], Any]:
    message = response.choices[0].message
    content = message.content or ""
    if not isinstance(content, str):
        content = str(content)
    usage = _as_mapping(getattr(response, "usage", None))
    normalized_usage = {
        str(key): int(value)
        for key, value in usage.items()
        if isinstance(value, (int, float))
    }
    return content, normalized_usage, message


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
