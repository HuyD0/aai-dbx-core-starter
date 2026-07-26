"""Public in-memory fakes for testing applications built on aai-core.

Consuming teams test agents, evaluation wiring, and configuration with zero
cloud access by registering these fakes on a real ``PlatformContext`` — the
same pattern the SDK's own test suite uses:

    from aai_core.testing import FakeChatModel, FakeRetriever, dev_context

    context = dev_context()
    context.providers.register_model("general-chat", FakeChatModel())
    context.providers.register_retriever("product-knowledge", FakeRetriever())

The fakes honor the provider contracts (capability checks, retrieval-mode
validation, normalized result shapes), so a test that passes against them
exercises the same behavior the real adapters enforce.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aai_core.context import PlatformContext
from aai_core.providers.types import (
    ModelCapabilities,
    ModelResponse,
    SearchResult,
    UnsupportedCapabilityError,
)
from aai_core.runtime import PlatformSettings
from aai_core.tags import ResourceContext


def dev_settings(**overrides: Any) -> PlatformSettings:
    """A valid, non-strict settings object for tests (no YAML, no cloud)."""

    resource_fields = {
        "application": "test-app",
        "project": "test-project",
        "environment": "dev",
        "team": "test-team",
        "owner_group": "group:test-owners",
        "cost_center": "CC-0000",
        "data_classification": "internal",
        "lifecycle": "experimental",
        "repository": "test/repo",
        "release": "dev",
    }
    setting_fields: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in resource_fields:
            resource_fields[key] = value
        else:
            setting_fields[key] = value
    return PlatformSettings(
        resource=ResourceContext(**resource_fields), **setting_fields
    )


def dev_context(**overrides: Any) -> PlatformContext:
    """A real PlatformContext over :func:`dev_settings`; register fakes on
    ``context.providers`` and pass it wherever a context is expected."""

    return PlatformContext(dev_settings(**overrides))


class FakeChatModel:
    """ChatModel fake that records requests and returns a canned reply."""

    def __init__(
        self,
        *,
        logical_name: str = "general-chat",
        reply: str = "fake reply",
        capabilities: ModelCapabilities | None = None,
        usage: Mapping[str, int] | None = None,
    ) -> None:
        self.logical_name = logical_name
        self.provider = "fake"
        self.model = "fake-model"
        self.reply = reply
        self.capabilities = capabilities or ModelCapabilities(
            tool_calling=True, structured_output=True
        )
        self.usage = dict(usage or {"prompt_tokens": 1, "completion_tokens": 1})
        self.native_client = None
        self.requests: list[dict[str, Any]] = []

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> ModelResponse:
        if options.get("tools") and not self.capabilities.tool_calling:
            raise UnsupportedCapabilityError(
                f"{self.logical_name} does not support tool calling"
            )
        if options.get("response_format") and not self.capabilities.structured_output:
            raise UnsupportedCapabilityError(
                f"{self.logical_name} does not support structured output"
            )
        self.requests.append({"messages": list(messages), **options})
        return ModelResponse(
            content=self.reply,
            provider=self.provider,
            logical_name=self.logical_name,
            model=self.model,
            latency_ms=1.0,
            usage=dict(self.usage),
        )


class FakeEmbeddingProvider:
    """EmbeddingProvider fake returning a fixed vector."""

    def __init__(
        self,
        *,
        logical_name: str = "knowledge-embedding",
        vector: Sequence[float] = (0.1, 0.2, 0.3),
    ) -> None:
        self.logical_name = logical_name
        self.provider = "fake"
        self.dimensions = len(vector)
        self.native_client = None
        self.vector = list(vector)
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return list(self.vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class FakeRetriever:
    """Retriever fake with the same mode contract as the real adapters."""

    def __init__(
        self,
        *,
        logical_name: str = "product-knowledge",
        results: Sequence[SearchResult] | None = None,
    ) -> None:
        self.logical_name = logical_name
        self.provider = "fake"
        self.native_client = None
        self.results = list(
            results
            if results is not None
            else [
                SearchResult(
                    document_id="doc-1",
                    content="grounding evidence",
                    score=1.0,
                    source_uri="https://example/doc",
                    chunk_id="chunk-1",
                    provider="fake",
                )
            ]
        )
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
        query_vector: Sequence[float] | None = None,
        mode: str = "hybrid",
        provider_options: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        if mode.lower() not in {"text", "vector", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        self.queries.append(query)
        return list(self.results)[:top_k]


class StaticSecretProvider:
    """SecretProvider fake returning one fixed value for every reference."""

    def __init__(self, value: str = "fake-secret-value") -> None:
        self.value = value

    def resolve(self, reference: Any) -> str:
        return self.value
