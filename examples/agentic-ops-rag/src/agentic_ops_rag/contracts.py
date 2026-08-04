"""Strict boundaries used by the workshop's deterministic application."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator

from aai_core.contracts import ContractModel


class RetrievalMode(StrEnum):
    TEXT = "text"
    VECTOR = "vector"
    HYBRID = "hybrid"


class QueryKind(StrEnum):
    EXACT_IDENTIFIER = "exact_identifier"
    KNOWLEDGE = "knowledge"
    PROPOSE_ACTION = "propose_action"
    SENSITIVE_REQUEST = "sensitive_request"


class OperationDocument(ContractModel):
    """One already-chunked, access-scoped runbook document."""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_uri: str = Field(pattern=r"^synthetic://")
    chunk_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    service: str = Field(min_length=1)
    runbook_code: str = Field(min_length=1)
    effective_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    allowed_groups: tuple[str, ...] = Field(min_length=1)
    active: bool = True

    @field_validator("allowed_groups")
    @classmethod
    def require_unique_groups(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("allowed_groups must not contain duplicates")
        return value


class EvaluationCase(ContractModel):
    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    allowed_groups: tuple[str, ...] = Field(min_length=1)
    expected_document_ids: tuple[str, ...] = ()
    answerable: bool
    expects_action_proposal: bool = False


class PipelineResult(ContractModel):
    query: str = Field(min_length=1)
    query_kind: QueryKind
    retrieval_mode: RetrievalMode
    answer: str = Field(min_length=1)
    citations: tuple[str, ...] = ()
    retrieved_document_ids: tuple[str, ...] = ()
    retrieved_tenants: tuple[str, ...] = ()
    abstained: bool
    proposed_action: str | None = None
    requires_approval: bool = False
    latency_ms: float = Field(ge=0.0)
    measurement_source: str = "simulated_offline_fixture"

    @field_validator("measurement_source")
    @classmethod
    def require_explicit_fixture_label(cls, value: str) -> str:
        if value != "simulated_offline_fixture":
            raise ValueError("offline measurements must remain explicitly labelled")
        return value
