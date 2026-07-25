"""RAG-specific release metadata and MLflow document normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from aai_core.providers import SearchResult


@dataclass(frozen=True)
class EmbeddingProfile:
    logical_name: str
    provider: str
    model: str
    dimensions: int
    normalized: bool
    version: str

    def assert_compatible(self, other: EmbeddingProfile) -> None:
        fields = ("model", "dimensions", "normalized")
        mismatches = [
            field_name
            for field_name in fields
            if getattr(self, field_name) != getattr(other, field_name)
        ]
        if mismatches:
            raise ValueError(
                "Embedding profiles are incompatible: " + ", ".join(mismatches)
            )


@dataclass(frozen=True)
class ChunkingProfile:
    name: str
    version: str
    chunk_size: int
    chunk_overlap: int
    parser: str

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size")


@dataclass(frozen=True)
class RAGDocument:
    document_id: str
    page_content: str
    doc_uri: str | None = None
    chunk_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_mlflow_document(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        if self.doc_uri:
            metadata["doc_uri"] = self.doc_uri
        if self.chunk_id:
            metadata["chunk_id"] = self.chunk_id
        return {
            "id": self.document_id,
            "page_content": self.page_content,
            "metadata": metadata,
        }

    @classmethod
    def from_search_result(cls, result: SearchResult) -> RAGDocument:
        return cls(
            document_id=result.document_id,
            page_content=result.content,
            doc_uri=result.source_uri,
            chunk_id=result.chunk_id,
            metadata=result.metadata,
        )


def mlflow_documents(results: Sequence[SearchResult]) -> list[dict[str, Any]]:
    return [result.as_mlflow_document() for result in results]
