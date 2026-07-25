"""Stable provider contracts and normalized results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Base error for a provider operation."""


class ProviderConfigurationError(ProviderError):
    """The logical resource cannot be resolved or is misconfigured."""


class UnsupportedCapabilityError(ProviderError):
    """The selected provider does not support a requested capability."""


@dataclass(frozen=True)
class ModelCapabilities:
    streaming: bool = False
    tool_calling: bool = True
    structured_output: bool = False
    embeddings: bool = False
    responses_api: bool = False


@dataclass(frozen=True)
class ModelResponse:
    content: str
    provider: str
    logical_name: str
    model: str
    latency_ms: float
    usage: Mapping[str, int] = field(default_factory=dict)
    tool_calls: tuple[Any, ...] = ()
    raw: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    content: str
    score: float | None
    source_uri: str | None = None
    chunk_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provider: str = "unknown"
    raw: Any = field(default=None, repr=False, compare=False)

    def as_mlflow_document(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        if self.source_uri:
            metadata["doc_uri"] = self.source_uri
        if self.chunk_id:
            metadata["chunk_id"] = self.chunk_id
        return {
            "id": self.document_id,
            "page_content": self.content,
            "metadata": metadata,
        }


@runtime_checkable
class ChatModel(Protocol):
    logical_name: str
    provider: str
    capabilities: ModelCapabilities
    native_client: Any

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Mapping[str, Any] | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> ModelResponse: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    logical_name: str
    provider: str
    dimensions: int | None
    native_client: Any

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class Retriever(Protocol):
    logical_name: str
    provider: str
    native_client: Any

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
        query_vector: Sequence[float] | None = None,
        mode: str = "hybrid",
        provider_options: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]: ...
