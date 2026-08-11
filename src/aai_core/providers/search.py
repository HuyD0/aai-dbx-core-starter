"""Azure AI Search and Databricks AI Search retriever adapters.

Both adapters share one contract: ``mode`` must be ``text``, ``vector``, or
``hybrid``; a mode that needs vectors either receives ``query_vector``, uses
the retriever's configured embedding provider to embed the query, or fails
with a configuration error that says how to fix it. No silent fallbacks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from aai_core._sensitive import is_sensitive_name, normalize_sensitive_name
from aai_core.providers.types import (
    ProviderConfigurationError,
    SearchResult,
    UnsupportedCapabilityError,
)
from aai_core.tracing import provider_span

_MODES = {"text", "vector", "hybrid"}
_AZURE_CONTROLLED_SEARCH_OPTIONS = frozenset(
    {"filter", "search_text", "select", "top", "vector_queries"}
)
_DATABRICKS_CONTROLLED_SEARCH_OPTIONS = frozenset(
    {"columns", "filters", "num_results", "query_text", "query_type", "query_vector"}
)
_FORBIDDEN_PER_CALL_SEARCH_OPTIONS = frozenset(
    {
        "allow_redirects",
        "body",
        "cert",
        "cookies",
        "connection",
        "connection_timeout",
        "default_headers",
        "extra_body",
        "extra_headers",
        "headers",
        "hooks",
        "http_client",
        "params",
        "pipeline",
        "proxies",
        "proxy",
        "read_timeout",
        "request_options",
        "retry",
        "retry_policy",
        "session",
        "stream",
        "timeout",
        "transport",
        "verify",
    }
)

__all__ = ["AzureAISearchRetriever", "DatabricksAISearchRetriever"]


def _validated_mode(mode: str) -> str:
    normalized = mode.lower()
    if normalized not in _MODES:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    return normalized


def _validated_provider_options(
    provider_options: Mapping[str, Any] | None,
    *,
    controlled_fields: frozenset[str],
) -> dict[str, Any]:
    """Return additive native options that cannot bypass the stable contract."""

    if not provider_options:
        return {}
    if any(not isinstance(key, str) for key in provider_options):
        raise ProviderConfigurationError(
            "provider_options keys must be strings",
            remediation="Pass named additive provider options only; use "
            "native_client for a different query contract.",
        )
    collisions = set(provider_options).intersection(controlled_fields)
    if collisions:
        raise ProviderConfigurationError(
            "provider_options cannot override controlled search fields: "
            f"{', '.join(sorted(collisions))}",
            remediation="Use the corresponding search() argument or retriever "
            "configuration; use native_client for a different query contract.",
        )
    if _contains_credential_option(provider_options):
        raise ProviderConfigurationError(
            "Per-call credentials, headers, raw requests, and transport controls "
            "are not allowed",
            remediation="Configure keyless authentication on the provider client; "
            "never pass credential material through provider_options.",
        )
    return dict(provider_options)


def _contains_credential_option(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and (
                is_sensitive_name(key)
                or normalize_sensitive_name(key) in _FORBIDDEN_PER_CALL_SEARCH_OPTIONS
            ):
                return True
            if _contains_credential_option(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_contains_credential_option(item) for item in value)
    return False


def _record_retriever_span(span: Any, query: str, results: list[SearchResult]) -> None:
    """Record query and normalized documents on the RETRIEVER span.

    Groundedness judges (and trace-based evaluation generally) read documents
    from RETRIEVER span outputs — without this, retrieval context is invisible
    to scorers and groundedness gates fail on missing evidence.
    """

    if span is None:
        return
    span.set_inputs({"query": query})
    span.set_outputs([result.as_mlflow_document() for result in results])


def _resolve_query_vector(
    *,
    retriever: Any,
    query: str,
    query_vector: Sequence[float] | None,
    mode: str,
    required: bool,
) -> Sequence[float] | None:
    """Return the vector for a vector-consuming mode, embedding on demand."""

    if query_vector is not None or mode == "text":
        return query_vector
    embedding = getattr(retriever, "embedding_provider", None)
    if embedding is not None:
        return cast(Sequence[float], embedding.embed_query(query))
    if required:
        raise ProviderConfigurationError(
            f"{mode} retrieval for {retriever.logical_name!r} needs a query " "vector",
            remediation="Pass query_vector, or set `embedding: "
            "<logical-embedding-name>` on this retriever in aai-platform.yml "
            "so the SDK embeds the query for you.",
        )
    return None


class AzureAISearchRetriever:
    """Normalize Azure AI Search text, vector, and hybrid retrieval."""

    provider = "azure_ai_search"

    def __init__(
        self,
        *,
        logical_name: str,
        client: Any,
        content_field: str,
        id_field: str,
        source_uri_field: str | None = None,
        chunk_id_field: str | None = None,
        vector_fields: Sequence[str] = (),
        embedding_provider: Any | None = None,
    ) -> None:
        self.logical_name = logical_name
        self.native_client = client
        self.content_field = content_field
        self.id_field = id_field
        self.source_uri_field = source_uri_field
        self.chunk_id_field = chunk_id_field
        self.vector_fields = tuple(vector_fields)
        self.embedding_provider = embedding_provider

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
        mode = _validated_mode(mode)
        additional_options = _validated_provider_options(
            provider_options,
            controlled_fields=_AZURE_CONTROLLED_SEARCH_OPTIONS,
        )
        # Azure AI Search always needs a client-side vector for vector/hybrid.
        query_vector = _resolve_query_vector(
            retriever=self,
            query=query,
            query_vector=query_vector,
            mode=mode,
            required=True,
        )
        if query_vector is not None and not self.vector_fields:
            raise UnsupportedCapabilityError(
                f"{self.logical_name} has no configured vector fields"
            )

        options: dict[str, Any] = {
            "search_text": query if mode in {"text", "hybrid"} else None,
            "top": top_k,
            "filter": _odata_filter(filters),
        }
        if query_vector is not None:
            from azure.search.documents.models import VectorizedQuery

            options["vector_queries"] = [
                VectorizedQuery(
                    vector=list(query_vector),
                    k_nearest_neighbors=top_k,
                    fields=",".join(self.vector_fields),
                )
            ]
        options.update(additional_options)
        options = {key: value for key, value in options.items() if value is not None}

        with provider_span(
            "retriever.search",
            span_type="RETRIEVER",
            attributes={
                "aai.provider": self.provider,
                "aai.logical_name": self.logical_name,
                "aai.retrieval_mode": mode,
            },
        ) as span:
            response = self.native_client.search(**options)
            results = [self._normalize(item) for item in response]
            _record_retriever_span(span, query, results)
            return results

    def _normalize(self, item: Mapping[str, Any]) -> SearchResult:
        reserved = {
            self.content_field,
            self.id_field,
            self.source_uri_field,
            self.chunk_id_field,
            "@search.score",
            "@search.reranker_score",
        }
        metadata = {key: value for key, value in item.items() if key not in reserved}
        score = item.get("@search.reranker_score", item.get("@search.score"))
        return SearchResult(
            document_id=str(item[self.id_field]),
            content=str(item[self.content_field]),
            score=float(score) if score is not None else None,
            source_uri=_optional_string(item, self.source_uri_field),
            chunk_id=_optional_string(item, self.chunk_id_field),
            metadata=metadata,
            provider=self.provider,
            raw=item,
        )


class DatabricksAISearchRetriever:
    """Normalize Databricks AI Search text, vector, and hybrid retrieval."""

    provider = "databricks_ai_search"

    def __init__(
        self,
        *,
        logical_name: str,
        index: Any,
        columns: Sequence[str],
        content_field: str,
        id_field: str,
        source_uri_field: str | None = None,
        chunk_id_field: str | None = None,
        embedding_provider: Any | None = None,
    ) -> None:
        self.logical_name = logical_name
        self.native_client = index
        self.columns = tuple(columns)
        self.content_field = content_field
        self.id_field = id_field
        self.source_uri_field = source_uri_field
        self.chunk_id_field = chunk_id_field
        self.embedding_provider = embedding_provider

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
        mode = _validated_mode(mode)
        additional_options = _validated_provider_options(
            provider_options,
            controlled_fields=_DATABRICKS_CONTROLLED_SEARCH_OPTIONS,
        )
        # Databricks can embed hybrid/text queries server-side, so a vector is
        # only mandatory for pure vector mode (never a silent fallback).
        query_vector = _resolve_query_vector(
            retriever=self,
            query=query,
            query_vector=query_vector,
            mode=mode,
            required=(mode == "vector"),
        )
        options: dict[str, Any] = {
            "columns": list(self.columns),
            "num_results": top_k,
            "filters": dict(filters or {}),
            "query_type": "HYBRID" if mode == "hybrid" else "ANN",
        }
        if mode != "vector":
            options["query_text"] = query
        if query_vector is not None:
            options["query_vector"] = list(query_vector)
        options.update(additional_options)

        with provider_span(
            "retriever.search",
            span_type="RETRIEVER",
            attributes={
                "aai.provider": self.provider,
                "aai.logical_name": self.logical_name,
                "aai.retrieval_mode": mode,
            },
        ) as span:
            response = self.native_client.similarity_search(**options)
            results = self._normalize_response(response)
            _record_retriever_span(span, query, results)
            return results

    def _normalize_response(self, response: Mapping[str, Any]) -> list[SearchResult]:
        manifest_columns = response.get("manifest", {}).get("columns", [])
        names = [column["name"] for column in manifest_columns]
        rows: Iterable[Sequence[Any]] = response.get("result", {}).get("data_array", [])
        results = []
        for row in rows:
            item = dict(zip(names, row, strict=False))
            score = item.pop("score", None)
            results.append(
                SearchResult(
                    document_id=str(item.pop(self.id_field)),
                    content=str(item.pop(self.content_field)),
                    score=float(score) if score is not None else None,
                    source_uri=_pop_optional(item, self.source_uri_field),
                    chunk_id=_pop_optional(item, self.chunk_id_field),
                    metadata=item,
                    provider=self.provider,
                    raw=row,
                )
            )
        return results


def _odata_filter(filters: Mapping[str, Any] | None) -> str | None:
    if not filters:
        return None
    clauses = []
    for key, value in sorted(filters.items()):
        if not key.replace("_", "").isalnum():
            raise ValueError(f"Unsafe Azure AI Search filter field: {key!r}")
        if isinstance(value, bool):
            literal = str(value).lower()
        elif isinstance(value, (int, float)):
            literal = str(value)
        else:
            literal = "'" + str(value).replace("'", "''") + "'"
        clauses.append(f"{key} eq {literal}")
    return " and ".join(clauses)


def _optional_string(item: Mapping[str, Any], field_name: str | None) -> str | None:
    if not field_name or item.get(field_name) is None:
        return None
    return str(item[field_name])


def _pop_optional(item: dict[str, Any], field_name: str | None) -> str | None:
    if not field_name:
        return None
    value = item.pop(field_name, None)
    return str(value) if value is not None else None
