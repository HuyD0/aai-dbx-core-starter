"""Application-owned routing, retrieval, reranking, and action boundary."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence

from aai_core.providers import SearchResult, UnsupportedCapabilityError
from agentic_ops_rag.contracts import (
    MeasurementSource,
    PipelineResult,
    QueryKind,
    RetrievalMode,
)

_IDENTIFIER = re.compile(r"\b(?:ERR|INC|OPS|RUN)-[A-Z0-9-]{2,}\b", re.IGNORECASE)
_ACTION = re.compile(
    r"\b(?:restart|recycle|fail[ -]?over|delete|disable|rotate|roll[ -]?back)\b",
    re.IGNORECASE,
)
_SENSITIVE_REQUEST = re.compile(
    r"\b(?:api key|client secret|password|private key|root credential|token)\b",
    re.IGNORECASE,
)


def authorized_search(
    retriever,
    query: str,
    *,
    tenant_id: str,
    region: str,
    allowed_groups: Sequence[str],
    mode: RetrievalMode | str,
    top_k: int,
    semantic_rerank: bool = False,
    provider_options: Mapping[str, object] | None = None,
) -> list[SearchResult]:
    """Search with a provider-native, fail-closed authorization pre-filter."""

    groups = tuple(
        dict.fromkeys(
            str(group).strip() for group in allowed_groups if str(group).strip()
        )
    )
    if not groups:
        return []

    selected_mode = RetrievalMode(mode).value
    provider = str(getattr(retriever, "provider", ""))
    filters: dict[str, object] = {"tenant_id": tenant_id, "region": region}
    native_options = dict(provider_options or {})
    controlled = {"filter", "filters"}.intersection(native_options)
    if controlled:
        raise ValueError(
            "provider_options cannot override authorization filters: "
            + ", ".join(sorted(controlled))
        )

    if provider == "offline_fixture":
        native_options.update(
            {
                "allowed_groups": groups,
                "semantic_rerank": semantic_rerank,
            }
        )
        return retriever.search(
            query,
            top_k=top_k,
            filters=filters,
            mode=selected_mode,
            provider_options=native_options,
        )

    if semantic_rerank:
        raise UnsupportedCapabilityError(
            "semantic_rerank is an offline teaching option; configure a "
            "provider-native reranker explicitly for connected retrieval"
        )

    if provider == "azure_ai_search":
        native_options["filter"] = _azure_access_filter(tenant_id, region, groups)
        return retriever.search(
            query,
            top_k=top_k,
            filters=None,
            mode=selected_mode,
            provider_options=native_options,
        )

    if provider == "databricks_ai_search":
        # ARRAY filters are supported by standard AI Search endpoints. A
        # storage-optimized endpoint rejects this expression, which is a safe
        # failure until the platform supplies a compatible scalar ACL field.
        filters["allowed_groups"] = list(groups)
        return retriever.search(
            query,
            top_k=top_k,
            filters=filters,
            mode=selected_mode,
            provider_options=native_options or None,
        )

    raise UnsupportedCapabilityError(
        f"Retriever provider {provider or '<unknown>'!r} has no approved access "
        "filter mapping; refusing connected retrieval"
    )


def _azure_access_filter(
    tenant_id: str, region: str, allowed_groups: Sequence[str]
) -> str:
    delimiter = "|"
    if any(delimiter in group for group in allowed_groups):
        raise ValueError(f"allowed group identifiers must not contain {delimiter!r}")
    group_values = delimiter.join(allowed_groups)
    return (
        f"tenant_id eq {_odata_literal(tenant_id)} and "
        f"region eq {_odata_literal(region)} and "
        "allowed_groups/any(group:search.in("
        f"group, {_odata_literal(group_values)}, {_odata_literal(delimiter)}))"
    )


def _odata_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def route_query(query: str) -> QueryKind:
    if _SENSITIVE_REQUEST.search(query):
        return QueryKind.SENSITIVE_REQUEST
    if _ACTION.search(query):
        return QueryKind.PROPOSE_ACTION
    if _IDENTIFIER.search(query):
        return QueryKind.EXACT_IDENTIFIER
    return QueryKind.KNOWLEDGE


class OperationsRAGPipeline:
    """A deterministic shell around any SDK-compatible retriever.

    The connected application swaps the retriever and answer generator. Access
    scope, evidence shape, action approval, and release metrics remain stable.
    """

    def __init__(
        self,
        retriever,
        *,
        answer_generator: Callable[[str, Sequence[SearchResult]], str] | None = None,
    ) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator

    def invoke(
        self,
        query: str,
        *,
        tenant_id: str,
        region: str,
        allowed_groups: Sequence[str],
        mode: RetrievalMode | str | None = None,
        candidate_k: int = 8,
        final_k: int = 3,
        semantic_rerank: bool = False,
    ) -> PipelineResult:
        started_at = time.perf_counter()
        offline_fixture = getattr(self.retriever, "provider", "") == "offline_fixture"
        if not query.strip():
            raise ValueError("query must not be blank")
        if not 0 < final_k <= candidate_k:
            raise ValueError("final_k must be positive and no larger than candidate_k")
        query_kind = route_query(query)
        selected_mode = RetrievalMode(mode or self._mode_for(query_kind))
        if query_kind is QueryKind.SENSITIVE_REQUEST:
            return PipelineResult(
                query=query,
                query_kind=query_kind,
                retrieval_mode=selected_mode,
                answer=(
                    "I cannot retrieve or disclose credentials. Follow the approved "
                    "secret-recovery and incident-escalation process."
                ),
                abstained=True,
                latency_ms=(
                    4.0
                    if offline_fixture
                    else (time.perf_counter() - started_at) * 1000.0
                ),
                measurement_source=(
                    MeasurementSource.SIMULATED_OFFLINE_FIXTURE
                    if offline_fixture
                    else MeasurementSource.CONNECTED_WALL_CLOCK
                ),
            )
        results = authorized_search(
            self.retriever,
            query,
            tenant_id=tenant_id,
            region=region,
            allowed_groups=allowed_groups,
            top_k=candidate_k,
            mode=selected_mode.value,
            semantic_rerank=semantic_rerank,
        )
        ranked = self._application_rerank(query, results)[:final_k]
        answerable = self._has_support(ranked)
        proposed_action = None
        requires_approval = False
        if not answerable:
            answer = (
                "I could not find authorized, current runbook evidence for this "
                "request. Escalate to the service owner."
            )
            citations: tuple[str, ...] = ()
        else:
            citations = tuple(result.document_id for result in ranked)
            if self.answer_generator is None:
                answer = " ".join(result.content for result in ranked)
            else:
                answer = self.answer_generator(query, tuple(ranked)).strip()
                if not answer:
                    raise ValueError("answer_generator returned a blank response")
            answer += " Sources: " + ", ".join(
                f"[{document_id}]" for document_id in citations
            )
            if query_kind is QueryKind.PROPOSE_ACTION:
                proposed_action = self._proposed_action(query)
                requires_approval = True
                answer += (
                    " No operational change was executed; the proposed action "
                    "requires an approved human checkpoint."
                )
        if offline_fixture:
            latency_ms = self._simulated_latency(
                selected_mode,
                candidate_count=len(results),
                semantic_rerank=semantic_rerank,
            )
            measurement_source = MeasurementSource.SIMULATED_OFFLINE_FIXTURE
        else:
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            measurement_source = MeasurementSource.CONNECTED_WALL_CLOCK
        return PipelineResult(
            query=query,
            query_kind=query_kind,
            retrieval_mode=selected_mode,
            answer=answer,
            citations=citations,
            retrieved_document_ids=tuple(result.document_id for result in ranked),
            retrieved_tenants=tuple(
                str(result.metadata.get("tenant_id", "")) for result in ranked
            ),
            abstained=not answerable,
            proposed_action=proposed_action,
            requires_approval=requires_approval,
            latency_ms=latency_ms,
            measurement_source=measurement_source,
        )

    @staticmethod
    def _mode_for(query_kind: QueryKind) -> RetrievalMode:
        if query_kind is QueryKind.EXACT_IDENTIFIER:
            return RetrievalMode.TEXT
        return RetrievalMode.HYBRID

    @staticmethod
    def _application_rerank(
        query: str, results: Sequence[SearchResult]
    ) -> list[SearchResult]:
        normalized = query.lower()

        def score(result: SearchResult) -> tuple[float, str]:
            provider_score = float(result.score or 0.0)
            code = str(result.metadata.get("runbook_code", "")).lower()
            freshness = str(result.metadata.get("effective_at", ""))
            exact_bonus = 2.0 if code and code in normalized else 0.0
            return provider_score + exact_bonus, freshness

        return sorted(
            results,
            key=lambda result: (score(result)[0], score(result)[1], result.document_id),
            reverse=True,
        )

    @staticmethod
    def _has_support(results: Sequence[SearchResult]) -> bool:
        if not results:
            return False
        top = results[0]
        if "lexical_score" in top.metadata or "semantic_score" in top.metadata:
            lexical = float(top.metadata.get("lexical_score", 0.0))
            semantic = float(top.metadata.get("semantic_score", 0.0))
            return lexical >= 0.2 or semantic >= 0.18

        # Connected provider scores are ranking signals with provider-specific
        # scales, so zero is treated only as missing support—not as a portable
        # quality threshold. Requiring normalized evidence fields keeps the
        # result usable by MLflow retriever spans and downstream judges.
        return (
            top.score is not None
            and float(top.score) > 0.0
            and bool(top.content.strip())
            and bool(top.source_uri)
            and bool(top.chunk_id)
        )

    @staticmethod
    def _proposed_action(query: str) -> str:
        match = _ACTION.search(query)
        return match.group(0).lower() if match else "review runbook action"

    @staticmethod
    def _simulated_latency(
        mode: RetrievalMode, *, candidate_count: int, semantic_rerank: bool
    ) -> float:
        base = {
            RetrievalMode.TEXT: 18.0,
            RetrievalMode.VECTOR: 27.0,
            RetrievalMode.HYBRID: 43.0,
        }[mode]
        return base + (candidate_count * 0.75) + (12.0 if semantic_rerank else 0.0)
