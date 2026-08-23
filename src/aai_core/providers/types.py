"""Stable provider contracts and normalized results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import Field, field_validator

from aai_core.contracts import ContractModel
from aai_core.exceptions import AaiCoreError

__all__ = [
    "ChatModel",
    "EmbeddingProvider",
    "ModelCapabilities",
    "ModelResponse",
    "AzureSemanticRankOptions",
    "DatabricksRerankOptions",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderRequestError",
    "Retriever",
    "RetrievalMode",
    "SearchRankingOptions",
    "SearchResult",
    "UnsupportedCapabilityError",
]


class RetrievalMode(StrEnum):
    """Portable retrieval algorithms exposed by both search adapters."""

    TEXT = "text"
    VECTOR = "vector"
    HYBRID = "hybrid"


class AzureSemanticRankOptions(ContractModel):
    """Typed query-time Azure AI Search semantic-ranking controls.

    The semantic configuration is provisioned with the index outside the SDK;
    this object only selects it for one governed query.  Captions and extractive
    answers deliberately remain available through ``native_client`` because the
    stable retriever contract returns documents, not an alternate answer shape.
    """

    provider: Literal["azure_ai_search"] = "azure_ai_search"
    semantic_configuration_name: str = Field(min_length=1, max_length=128)
    semantic_query: str | None = Field(default=None, min_length=1)
    error_mode: Literal["fail", "partial"] = "fail"
    max_wait_milliseconds: int | None = Field(default=None, gt=0, le=120_000)

    @field_validator("semantic_configuration_name", "semantic_query")
    @classmethod
    def reject_surrounding_whitespace(cls, value: str | None) -> str | None:
        if value is not None and value != value.strip():
            raise ValueError("semantic ranking names and queries must be trimmed")
        return value


class DatabricksRerankOptions(ContractModel):
    """Typed query-time controls for the Databricks cross-encoder reranker."""

    provider: Literal["databricks_ai_search"] = "databricks_ai_search"
    columns_to_rerank: tuple[str, ...] = Field(min_length=1)

    @field_validator("columns_to_rerank", mode="before")
    @classmethod
    def coerce_columns(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("columns_to_rerank")
    @classmethod
    def validate_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not column.strip() or column != column.strip() for column in value):
            raise ValueError("reranker columns must be non-empty trimmed names")
        if len(set(value)) != len(value):
            raise ValueError("reranker columns must be unique")
        return value


SearchRankingOptions: TypeAlias = AzureSemanticRankOptions | DatabricksRerankOptions


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
        raise NotImplementedError

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
        raise NotImplementedError


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Synchronous query and document embedding contract."""

    logical_name: str
    provider: str
    dimensions: int | None
    native_client: Any

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


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
        mode: str | RetrievalMode = RetrievalMode.HYBRID,
        ranking: SearchRankingOptions | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        raise NotImplementedError
