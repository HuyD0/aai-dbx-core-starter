"""Inference-wide source and runtime identity boundary for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from ..training import (
    ExecutionContract,
    ExecutionSnapshot,
    capture_execution_snapshot,
    recheck_execution_snapshot,
)


@dataclass(frozen=True, slots=True)
class EvaluationSession:
    """One pre-inference execution snapshot retained through persistence."""

    _snapshot: ExecutionSnapshot

    @property
    def execution_contract(self) -> ExecutionContract:
        return self._snapshot.execution_contract

    @property
    def execution_contract_sha256(self) -> str:
        return self._snapshot.execution_contract_sha256


def start_evaluation_session() -> EvaluationSession:
    """Start the evidence boundary immediately before prediction generation."""

    return EvaluationSession(_snapshot=capture_execution_snapshot())


def recheck_evaluation_session(session: EvaluationSession) -> EvaluationSession:
    """Require the original source and package identities to remain unchanged."""

    recheck_execution_snapshot(session._snapshot)
    return session
