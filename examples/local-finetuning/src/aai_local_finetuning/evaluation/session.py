"""Inference-wide source and runtime identity boundary for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..settings import ProjectSettings
from ..training import (
    BaseModelExecutionContract,
    BaseModelSnapshot,
    ExecutionContract,
    ExecutionSnapshot,
    capture_base_model_snapshot,
    capture_execution_snapshot,
    recheck_base_model_snapshot,
    recheck_execution_snapshot,
)
from .models import GenerationConfig, LocalMLXInferenceConfig


@dataclass(frozen=True, slots=True)
class EvaluationSession:
    """Pre-inference source/package state and optional base-model identity."""

    _snapshot: ExecutionSnapshot = field(repr=False)
    _base_model_snapshot: BaseModelSnapshot | None = field(repr=False)

    @property
    def execution_contract(self) -> ExecutionContract:
        return self._snapshot.execution_contract

    @property
    def execution_contract_sha256(self) -> str:
        return self._snapshot.execution_contract_sha256

    @property
    def base_model_execution_contract(self) -> BaseModelExecutionContract | None:
        if self._base_model_snapshot is None:
            return None
        return self._base_model_snapshot.execution_contract

    @property
    def base_model_execution_contract_sha256(self) -> str | None:
        if self._base_model_snapshot is None:
            return None
        return self._base_model_snapshot.execution_contract_sha256


def start_evaluation_session(
    settings: ProjectSettings | None = None,
) -> EvaluationSession:
    """Start execution evidence, adding verified model evidence when configured."""

    execution_snapshot = capture_execution_snapshot()
    base_model_snapshot = (
        capture_base_model_snapshot(settings) if settings is not None else None
    )
    recheck_execution_snapshot(execution_snapshot)
    return EvaluationSession(
        _snapshot=execution_snapshot,
        _base_model_snapshot=base_model_snapshot,
    )


def recheck_evaluation_session(session: EvaluationSession) -> EvaluationSession:
    """Require every identity captured by the session to remain current."""

    recheck_execution_snapshot(session._snapshot)
    if session._base_model_snapshot is not None:
        recheck_base_model_snapshot(session._base_model_snapshot)
    recheck_execution_snapshot(session._snapshot)
    return session


def build_local_mlx_inference_config(
    evaluation_session: EvaluationSession,
    *,
    method: str,
    prompt_recipe: str,
    max_tokens: int,
    few_shot_examples: int = 0,
    adapter_manifest_sha256: str | None = None,
) -> LocalMLXInferenceConfig:
    """Build model evidence from the exact snapshot that brackets inference."""

    base_model = evaluation_session.base_model_execution_contract
    if base_model is None:
        raise ValueError(
            "local MLX inference requires start_evaluation_session(settings)"
        )
    return LocalMLXInferenceConfig(
        method=method,
        prompt_recipe=prompt_recipe,
        few_shot_examples=few_shot_examples,
        generation=GenerationConfig(max_tokens=max_tokens),
        base_model=base_model,
        adapter_manifest_sha256=adapter_manifest_sha256,
    )
