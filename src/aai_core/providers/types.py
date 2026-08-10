"""Stable provider contracts and normalized results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from aai_core.exceptions import AaiCoreError

__all__ = [
    "ChatModel",
    "EmbeddingProvider",
    "ModelCapabilities",
    "ModelResponse",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderRequestError",
    "Retriever",
    "SearchResult",
    "UnsupportedCapabilityError",
]


class ProviderError(AaiCoreError):
    """Base error for a provider operation."""

    code = "aai_core.provider.error"


class ProviderConfigurationError(ProviderError):
    """The logical resource cannot be resolved or is misconfigured."""

    code = "aai_core.provider.configuration"


class UnsupportedCapabilityError(ProviderError):
    """The selected provider does not support a requested capability."""

    code = "aai_core.provider.unsupported_capability"


class ProviderRequestError(ProviderError):
    """A sanitized provider failure with stable, non-secret diagnostics."""

    code = "aai_core.provider.request_failed"

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        operation: str | None = None,
        logical_name: str | None = None,
        status_code: int | None = None,
        remediation: str | None = None,
    ) -> None:
        super().__init__(message, remediation=remediation)
        self.provider = provider
        self.operation = operation
        self.logical_name = logical_name
        self.status_code = status_code


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities the stable synchronous adapter can honor."""

    tool_calling: bool = True
    structured_output: bool = False
    embeddings: bool = False


@dataclass(frozen=True)
class ModelResponse:
    """Normalized synchronous chat response with the native result retained."""

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
    """Normalized retrieval result suitable for MLflow document evidence."""

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
    """Capability-bearing synchronous chat model contract."""

    logical_name: str
    provider: str
    capabilities: ModelCapabilities
    native_client: Any

    def create_native_async_client(self) -> Any:
        """Create a provider-native async client owned by the caller."""
        ...

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
    """Synchronous query and document embedding contract."""

    logical_name: str
    provider: str
    dimensions: int | None
    native_client: Any

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class Retriever(Protocol):
    """Provider-neutral retrieval contract with a native-client escape hatch."""

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
