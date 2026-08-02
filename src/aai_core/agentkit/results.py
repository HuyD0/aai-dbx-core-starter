"""The results record — one scoring run's evidence, written to disk.

Every scoring command writes one of these under ``.aai/agentkit/results/``.
``gate`` and ``evidence`` read the newest record rather than re-running an
evaluation, so a CI job can score once and gate/report from the same
evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_serializer, field_validator

from aai_core.agentkit.baseline import BaselineDataset, BaselineScope, BaselineVersions
from aai_core.agentkit.errors import ConfigError
from aai_core.contracts import ContractModel, freeze_value, thaw_value

RESULTS_GLOB = "*.json"


class ResultsRecord(ContractModel):
    """What one ``compare``/``smoke``/``eval`` run produced."""

    schema_version: Literal[1] = 1
    command: str = Field(min_length=1)
    recorded_at: str = Field(min_length=1)
    run_id: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    agent: str = Field(min_length=1)
    dataset: BaselineDataset
    scope: BaselineScope
    mode: str = Field(min_length=1)
    metrics: Mapping[str, float] = Field(default_factory=dict)
    versions: BaselineVersions
    baseline_run_id: str | None = None
    baseline_metrics: Mapping[str, float] = Field(default_factory=dict)
    established_baseline: bool = False
    decision: str = Field(min_length=1)
    change_id: str = Field(min_length=1)
    gate_passed: bool
    gate_failures: tuple[Mapping[str, str], ...] = ()
    warnings: tuple[str, ...] = ()
    judges_enabled: bool = False

    @field_validator("gate_failures", "warnings", mode="before")
    @classmethod
    def coerce_sequences(cls, value: Any) -> Any:
        # Round-tripping through JSON turns tuples into lists; strict mode
        # would otherwise refuse to reload a record it just wrote.
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("metrics", "baseline_metrics", mode="before")
    @classmethod
    def coerce_metrics(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: float(item) if isinstance(item, int) else item
                for key, item in value.items()
                if not isinstance(item, bool)
            }
        return value

    @field_validator("metrics", "baseline_metrics", mode="after")
    @classmethod
    def freeze_metrics(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return freeze_value(value)

    @field_validator("gate_failures", mode="after")
    @classmethod
    def freeze_failures(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        return freeze_value(value)

    @field_serializer("metrics", "baseline_metrics")
    def serialize_metrics(self, value: Mapping[str, float]) -> dict[str, float]:
        return thaw_value(value)

    @field_serializer("gate_failures")
    def serialize_failures(self, value: tuple[Any, ...]) -> list[dict[str, str]]:
        return thaw_value(value)

    @property
    def is_comparison(self) -> bool:
        """True when this run named what it was scored against.

        A run that established the baseline is itself the named reference;
        anything else must link a baseline run or carry baseline metrics.
        """

        return bool(
            self.established_baseline or self.baseline_run_id or self.baseline_metrics
        )


def write_results(directory: Path, record: ResultsRecord) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = record.recorded_at.replace(":", "").replace("-", "")
    path = directory / f"{stamp}-{record.command}.json"
    path.write_text(
        json.dumps(record.model_dump(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def read_results(path: Path) -> ResultsRecord:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path} is not valid JSON: {error}") from error
    try:
        return ResultsRecord(**document)
    except ValidationError as error:
        raise ConfigError(f"{path} is not a valid results record: {error}") from error


def load_latest_results(directory: Path) -> tuple[ResultsRecord, Path] | None:
    """Newest results record in the directory, or None when none exist."""

    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob(RESULTS_GLOB))
    if not candidates:
        return None
    newest = max(candidates, key=lambda item: (item.stat().st_mtime, item.name))
    return read_results(newest), newest
