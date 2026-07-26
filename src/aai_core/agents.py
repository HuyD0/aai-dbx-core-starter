"""Framework-neutral agent request and response contracts.

Agent loops, tool execution, streaming, and deployment belong to generated
applications or the selected native framework. Keeping them out of aai-core
prevents the SDK from competing with MLflow Agent Server and LangGraph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field, field_serializer, field_validator

from aai_core.contracts import ContractModel, freeze_value, thaw_value


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
