"""Framework-neutral agent application contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class AgentRequest:
    messages: Sequence[Mapping[str, Any]]
    session_id: str | None = None
    user_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResponse:
    content: str
    trace_id: str | None = None
    citations: Sequence[Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class AgentApplication(Protocol):
    def invoke(self, request: AgentRequest) -> AgentResponse: ...
