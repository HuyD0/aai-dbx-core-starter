"""Application-owned routing, retrieval, reranking, and action boundary."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence

from aai_core.providers import SearchResult, UnsupportedCapabilityError
from aai_core.tracing import provider_span
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
_SUPPORT_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SUPPORT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "explain",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "please",
    "the",
    "this",
    "to",
    "what",
    "when",
    "which",
    "with",
}
_CONNECTED_EVIDENCE_FIELDS = (
    "tenant_id",
    "region",
    "allowed_groups",
    "active",
    "runbook_code",
    "effective_at",
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
    controlled = {"columns", "filter", "filters", "select"}.intersection(native_options)
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
        native_options["select"] = _azure_selected_fields(retriever)
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
        configured_columns = set(getattr(retriever, "columns", ()))
        missing_columns = set(_CONNECTED_EVIDENCE_FIELDS).difference(configured_columns)
        if missing_columns:
            raise UnsupportedCapabilityError(
                "Databricks AI Search must return governed evidence columns: "
                + ", ".join(sorted(missing_columns))
            )
        filters.update({"active": True, "allowed_groups": list(groups)})
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
        "active eq true and "
        "allowed_groups/any(group:search.in("
        f"group, {_odata_literal(group_values)}, {_odata_literal(delimiter)}))"
    )


def _azure_selected_fields(retriever) -> list[str]:
    identity_fields = (
        getattr(retriever, "id_field", "id"),
        getattr(retriever, "content_field", "content"),
        getattr(retriever, "source_uri_field", "source_uri"),
        getattr(retriever, "chunk_id_field", "chunk_id"),
    )
    return list(
        dict.fromkeys(
            str(field)
            for field in (*identity_fields, *_CONNECTED_EVIDENCE_FIELDS)
            if field
        )
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
        with provider_span(
            "retriever.final_context",
            span_type="RETRIEVER",
            attributes={"aai.evidence_role": "model_context"},
        ) as final_context_span:
            # MLflow retrieval judges consume top-level RETRIEVER outputs. Keep
            # the provider adapter's raw candidate span nested here, then expose
            # only the exact supported documents supplied to the model on this
            # scorer-visible span.
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
            ranked = self._application_rerank(query, results)
            supported = tuple(
                result
                for result in ranked
                if self._result_has_support(
                    query,
                    result,
                    offline_fixture=offline_fixture,
                )
            )[:final_k]
            if final_context_span is not None:
                final_context_span.set_inputs(
                    {
                        "query": query,
                        "candidate_document_ids": [
                            result.document_id for result in results
                        ],
                    }
                )
                final_context_span.set_outputs(
                    [result.as_mlflow_document() for result in supported]
                )
                final_context_span.set_attribute("aai.candidate_count", len(results))
                final_context_span.set_attribute(
                    "aai.final_context_count", len(supported)
                )
        answerable = bool(supported)
        proposed_action = None
        requires_approval = False
        if not answerable:
            answer = (
                "I could not find authorized, current runbook evidence for this "
                "request. Escalate to the service owner."
            )
            citations: tuple[str, ...] = ()
        else:
            citations = tuple(result.document_id for result in supported)
            if self.answer_generator is None:
                answer = " ".join(result.content for result in supported)
            else:
                answer = self.answer_generator(query, supported).strip()
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
            retrieved_document_ids=tuple(result.document_id for result in supported),
            retrieved_tenants=tuple(
                str(result.metadata.get("tenant_id", "")) for result in supported
            ),
            retrieved_regions=tuple(
                str(result.metadata.get("region", "")) for result in supported
            ),
            retrieved_allowed_groups=tuple(
                _normalized_groups(result.metadata.get("allowed_groups"))
                for result in supported
            ),
            retrieved_active=tuple(
                result.metadata.get("active") is True for result in supported
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

        # Access filters request active evidence before ranking. Retain this
        # fail-closed application boundary as well: a provider/index that omits
        # or ignores the field cannot send retired material to generation.
        current = [
            result for result in results if result.metadata.get("active") is True
        ]

        # A provider can return several active revisions for one runbook code.
        # Keep the newest ISO-dated revision before comparing provider ranking
        # signals so a highly scored stale revision cannot displace it.
        latest_by_runbook: dict[str, SearchResult] = {}
        for result in current:
            code = str(result.metadata.get("runbook_code", "")).strip().lower()
            key = code or f"document:{result.document_id}"
            existing = latest_by_runbook.get(key)
            if existing is None or str(result.metadata.get("effective_at", "")) > str(
                existing.metadata.get("effective_at", "")
            ):
                latest_by_runbook[key] = result

        def score(result: SearchResult) -> tuple[bool, float, str]:
            provider_score = float(result.score or 0.0)
            code = str(result.metadata.get("runbook_code", "")).lower()
            freshness = str(result.metadata.get("effective_at", ""))
            exact_identifier = bool(code and code in normalized)
            return exact_identifier, provider_score, freshness

        return sorted(
            latest_by_runbook.values(),
            key=lambda result: (*score(result), result.document_id),
            reverse=True,
        )

    @staticmethod
    def _result_has_support(
        query: str,
        result: SearchResult,
        *,
        offline_fixture: bool,
    ) -> bool:
        if offline_fixture:
            lexical = float(result.metadata.get("lexical_score", 0.0))
            semantic = float(result.metadata.get("semantic_score", 0.0))
            return lexical >= 0.2 or semantic >= 0.18

        # Connected scores are ranking signals with provider-specific scales;
        # no positive value proves semantic support. Require complete evidence
        # plus deterministic identifier or lexical overlap. Ambiguous semantic-
        # only retrieval abstains until an evaluated application-owned support
        # policy (or governed judge) supplies evidence.
        if not (
            result.content.strip()
            and result.source_uri
            and result.chunk_id
            and result.metadata.get("active") is True
        ):
            return False
        evidence = " ".join(
            (
                result.content,
                str(result.metadata.get("title", "")),
                str(result.metadata.get("service", "")),
                str(result.metadata.get("runbook_code", "")),
            )
        )
        identifiers = {match.group(0).lower() for match in _IDENTIFIER.finditer(query)}
        if identifiers:
            normalized_evidence = evidence.lower()
            return identifiers.issubset(
                {
                    match.group(0).lower()
                    for match in _IDENTIFIER.finditer(normalized_evidence)
                }
            )
        query_terms = _support_terms(query)
        if not query_terms:
            return False
        overlap = query_terms.intersection(_support_terms(evidence))
        required_overlap = 1 if len(query_terms) == 1 else 2
        return len(overlap) >= required_overlap

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


def _support_terms(text: str) -> set[str]:
    return {
        token
        for token in _SUPPORT_TOKEN.findall(text.lower())
        if token not in _SUPPORT_STOP_WORDS
    }


def _normalized_groups(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(
        dict.fromkeys(str(group).strip() for group in value if str(group).strip())
    )
