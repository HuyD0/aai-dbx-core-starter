"""Framework-neutral agent request, response, and decision contracts.

Agent loops, tool execution, streaming, and deployment belong to generated
applications or the selected native framework. Keeping them out of aai-core
prevents the SDK from competing with MLflow Agent Server and LangGraph.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from math import isfinite
from typing import Any

from pydantic import Field, field_serializer, field_validator

from aai_core.contracts import ContractModel, freeze_value, thaw_value

__all__ = [
    "AgentDecision",
    "AgentDecisionType",
    "AgentRequest",
    "AgentResponse",
]

_DECISION_ACTION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")
_DECISION_EVIDENCE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/#-]{0,127}$")


class AgentDecisionType(StrEnum):
    """Small vocabulary for meaningful, application-observable decisions."""

    TOOL_SELECTION = "tool_selection"
    EVIDENCE_SUFFICIENCY = "evidence_sufficiency"
    FALLBACK = "fallback"
    ANSWER_READINESS = "answer_readiness"
    HUMAN_APPROVAL = "human_approval"


class AgentDecision(ContractModel):
    """Concise provider-neutral evidence of a meaningful agent decision.

    This record captures an application's observable decision and short
    operational justification. It must not contain retrieved documents,
    prompts, provider-native reasoning, or hidden model chain-of-thought.
    Execution spans remain authoritative for what subsequently happened.
    """

    decision_type: AgentDecisionType
    goal: str = Field(min_length=1, max_length=256)
    selected_action: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=512)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=16)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    alternatives_considered: tuple[str, ...] = Field(default=(), max_length=8)
    expected_result: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("decision_type", mode="before")
    @classmethod
    def parse_decision_type(cls, value: Any) -> AgentDecisionType:
        if isinstance(value, AgentDecisionType):
            return value
        if not isinstance(value, str):
            raise ValueError("decision_type must be a string or AgentDecisionType")
        return AgentDecisionType(value.strip().lower())

    @field_validator("evidence_refs", "alternatives_considered", mode="before")
    @classmethod
    def normalize_string_sequences(cls, value: Any) -> tuple[Any, ...]:
        if isinstance(value, tuple):
            return value
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return tuple(value)
        raise ValueError("decision references and alternatives must be sequences")

    @field_validator("goal", "reason", "expected_result")
    @classmethod
    def require_concise_single_line(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value != value.strip():
            raise ValueError("decision text must not have surrounding whitespace")
        if "\n" in value or "\r" in value:
            raise ValueError("decision text must be a single line")
        return value

    @field_validator("selected_action")
    @classmethod
    def require_action_identifier(cls, value: str) -> str:
        if not _DECISION_ACTION.fullmatch(value):
            raise ValueError("selected_action must be a safe action identifier")
        return value

    @field_validator("evidence_refs")
    @classmethod
    def require_evidence_references(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _DECISION_EVIDENCE_REF.fullmatch(item) for item in value):
            raise ValueError("evidence_refs must contain safe reference identifiers")
        return value

    @field_validator("alternatives_considered")
    @classmethod
    def require_alternative_identifiers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _DECISION_ACTION.fullmatch(item) for item in value):
            raise ValueError(
                "alternatives_considered must contain safe action identifiers"
            )
        return value

    @field_validator("confidence")
    @classmethod
    def require_finite_confidence(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("confidence must be finite")
        return value


class AgentRequest(ContractModel):
    """Serializable input shared by generated application boundaries."""

    messages: tuple[Mapping[str, Any], ...]
    session_id: str | None = None
    user_id: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("messages", mode="before")
    @classmethod
    def normalize_messages(
        cls, value: Sequence[Mapping[str, Any]]
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(value)

    @field_validator("messages", "metadata", mode="after")
    @classmethod
    def freeze_boundaries(cls, value: Any) -> Any:
        return freeze_value(value)

    @field_serializer("messages", "metadata")
    def serialize_boundaries(self, value: Any) -> Any:
        return thaw_value(value)


class AgentResponse(ContractModel):
    """Serializable output shared by generated application boundaries."""

    content: str
    trace_id: str | None = None
    citations: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("citations", mode="before")
    @classmethod
    def normalize_citations(
        cls, value: Sequence[Mapping[str, Any]]
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(value)

    @field_validator("citations", "metadata", mode="after")
    @classmethod
    def freeze_boundaries(cls, value: Any) -> Any:
        return freeze_value(value)

    @field_serializer("citations", "metadata")
    def serialize_boundaries(self, value: Any) -> Any:
        return thaw_value(value)
