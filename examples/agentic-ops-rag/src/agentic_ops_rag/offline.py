"""Small transparent retrieval stand-ins for credential-free learning.

The arithmetic is intentionally inspectable. It demonstrates ranking behavior;
it is not a performance or quality claim about Azure AI Search.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from aai_core.providers import SearchResult
from agentic_ops_rag.contracts import OperationDocument, RetrievalMode

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP_WORDS = {
    "a",
    "after",
    "and",
    "are",
    "before",
    "but",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "not",
    "of",
    "should",
    "the",
    "this",
    "to",
    "what",
    "when",
    "which",
    "with",
}
_SYNONYMS = {
    "down": ("outage", "unavailable", "503"),
    "outage": ("down", "unavailable", "503"),
    "slow": ("latency", "timeout", "degraded"),
    "login": ("authentication", "identity", "mfa"),
    "signin": ("authentication", "identity", "mfa"),
    "rollback": ("revert", "restore", "deployment"),
    "restart": ("recycle", "recover", "service"),
}


def load_documents(path: str | Path) -> tuple[OperationDocument, ...]:
    documents = tuple(
        OperationDocument.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    identifiers = [document.document_id for document in documents]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("document_id values must be unique")
    return documents


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        token for token in _TOKEN.findall(text.lower()) if token not in _STOP_WORDS
    )


def expanded_query_tokens(text: str) -> tuple[str, ...]:
    tokens = list(tokenize(text))
    for token in tuple(tokens):
        tokens.extend(_SYNONYMS.get(token, ()))
    return tuple(dict.fromkeys(tokens))


def deterministic_embedding(text: str, *, dimensions: int = 48) -> list[float]:
    """Return a stable feature-hashed bag-of-words vector."""

    vector = [0.0] * dimensions
    for token in expanded_query_tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def lexical_score(query: str, document: OperationDocument) -> float:
    query_tokens = set(tokenize(query))
    document_tokens = set(tokenize(f"{document.title} {document.content}"))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & document_tokens) / len(query_tokens)
    identifiers = {token for token in query_tokens if "-" in token}
    identifier_bonus = 1.0 if identifiers & document_tokens else 0.0
    return overlap + identifier_bonus


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], *, rank_constant: int = 60
) -> dict[str, float]:
    """Fuse rankings without comparing incompatible provider score ranges."""

    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, document_id in enumerate(ranked, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (
                rank_constant + rank
            )
    maximum = len(ranked_lists) / (rank_constant + 1) if ranked_lists else 1.0
    return {key: value / maximum for key, value in scores.items()}


class OfflineEmbeddingProvider:
    logical_name = "operations-embedding-offline"
    provider = "offline_fixture"
    dimensions = 48
    native_client = None

    def embed_query(self, text: str) -> list[float]:
        return deterministic_embedding(text, dimensions=self.dimensions)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class OfflineOperationsRetriever:
    """Provider-contract-compatible retriever over the synthetic corpus."""

    logical_name = "operations-knowledge"
    provider = "offline_fixture"
    native_client = None

    def __init__(self, documents: Sequence[OperationDocument]) -> None:
        self.documents = tuple(documents)
        self.embedding_provider = OfflineEmbeddingProvider()
        self._vectors = {
            document.document_id: deterministic_embedding(
                f"{document.title} {document.content}"
            )
            for document in self.documents
        }

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
        selected_mode = RetrievalMode(mode.lower())
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        options = dict(provider_options or {})
        allowed_groups = tuple(str(item) for item in options.get("allowed_groups", ()))
        eligible = [
            document
            for document in self.documents
            if self._eligible(document, filters or {}, allowed_groups)
        ]
        lexical = {
            document.document_id: lexical_score(query, document)
            for document in eligible
        }
        vector = list(query_vector or deterministic_embedding(query))
        semantic = {
            document.document_id: cosine(vector, self._vectors[document.document_id])
            for document in eligible
        }
        lexical_rank = _rank(lexical)
        semantic_rank = _rank(semantic)
        if selected_mode is RetrievalMode.TEXT:
            scores = _rank_scores(lexical_rank)
        elif selected_mode is RetrievalMode.VECTOR:
            scores = _rank_scores(semantic_rank)
        else:
            scores = reciprocal_rank_fusion((lexical_rank, semantic_rank))

        if options.get("semantic_rerank"):
            scores = self._semantic_rerank(query, eligible, scores)

        by_id = {document.document_id: document for document in eligible}
        ordered = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
        return [
            self._as_result(
                by_id[document_id],
                score=scores[document_id],
                lexical=lexical[document_id],
                semantic=semantic[document_id],
            )
            for document_id in ordered
        ]

    @staticmethod
    def _eligible(
        document: OperationDocument,
        filters: Mapping[str, Any],
        allowed_groups: tuple[str, ...],
    ) -> bool:
        if not document.active:
            return False
        for field_name in ("tenant_id", "region", "service"):
            expected = filters.get(field_name)
            if expected is not None and getattr(document, field_name) != expected:
                return False
        return bool(set(document.allowed_groups).intersection(allowed_groups))

    @staticmethod
    def _semantic_rerank(
        query: str,
        documents: Iterable[OperationDocument],
        scores: Mapping[str, float],
    ) -> dict[str, float]:
        query_tokens = set(expanded_query_tokens(query))
        reranked: dict[str, float] = {}
        for document in documents:
            title_tokens = set(tokenize(document.title))
            title_coverage = len(query_tokens & title_tokens) / max(
                len(query_tokens), 1
            )
            exact_code = document.runbook_code.lower() in query.lower()
            reranked[document.document_id] = (
                scores[document.document_id]
                + (0.35 * title_coverage)
                + (0.75 if exact_code else 0.0)
            )
        return reranked

    def _as_result(
        self,
        document: OperationDocument,
        *,
        score: float,
        lexical: float,
        semantic: float,
    ) -> SearchResult:
        return SearchResult(
            document_id=document.document_id,
            content=document.content,
            score=score,
            source_uri=document.source_uri,
            chunk_id=document.chunk_id,
            metadata={
                "title": document.title,
                "tenant_id": document.tenant_id,
                "region": document.region,
                "service": document.service,
                "runbook_code": document.runbook_code,
                "effective_at": document.effective_at,
                "lexical_score": lexical,
                "semantic_score": semantic,
            },
            provider=self.provider,
            raw=None,
        )


def _rank(scores: Mapping[str, float]) -> list[str]:
    return sorted(scores, key=lambda key: (-scores[key], key))


def _rank_scores(ranked: Sequence[str]) -> dict[str, float]:
    return {document_id: 1.0 / rank for rank, document_id in enumerate(ranked, 1)}
