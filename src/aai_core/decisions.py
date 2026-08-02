"""Persistable lifecycle decisions binding baseline, gate, and release evidence.

The lifecycle vocabulary ends every comparison in an explicit ``adopt``,
``reject``, or ``inconclusive`` decision. This module gives that decision a
strict, persisted contract and records it as a governed MLflow run so the
decision is searchable next to the evidence it was made from. It composes
:class:`~aai_core.experiments.ExperimentManager`; it owns no MLflow surface
of its own.
"""

from __future__ import annotations

import json
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from aai_core.contracts import ContractModel
from aai_core.evaluation import GateResult
from aai_core.experiments import (
    ExperimentManager,
    ExperimentRunMetadata,
    RunPurpose,
)


class Decision(StrEnum):
    """Closed vocabulary for the outcome of a baseline/change comparison."""

    ADOPT = "adopt"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


class DecisionRecord(ContractModel):
    """Immutable decision evidence for one deliberate change."""

    decision: Decision
    change_id: str = Field(min_length=1)
    change_summary: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    baseline_run_id: str | None = Field(default=None, min_length=1)
    change_run_id: str | None = Field(default=None, min_length=1)
    gate: GateResult | None = None
    prompt_digest: str | None = Field(default=None, min_length=1)
    release_digest: str | None = Field(default=None, min_length=1)
    decided_by: str | None = Field(default=None, min_length=1)
    schema_version: Literal["1"] = "1"

    @field_validator("decision", mode="before")
    @classmethod
    def parse_decision(cls, value: Any) -> Decision:
        if isinstance(value, Decision):
            return value
        if not isinstance(value, str):
            raise TypeError("decision must be a string or Decision")
        return Decision(value.strip().lower())

    @field_validator("decided_by")
    @classmethod
    def refuse_personal_identity(cls, value: str | None) -> str | None:
        if value is not None and "@" in value:
            raise ValueError(
                "decided_by must be a non-personal identity such as a group "
                "name, never an email address"
            )
        return value

    @model_validator(mode="after")
    def adopt_requires_passing_gate(self) -> Self:
        if self.decision is Decision.ADOPT:
            if self.gate is None:
                raise ValueError(
                    "An adopt decision requires gate evidence; attach the "
                    "passing GateResult it was decided from, or record "
                    "inconclusive"
                )
            if not self.gate.passed:
                raise ValueError(
                    "An adopt decision cannot cite a failing gate; record "
                    "reject or inconclusive, or attach the passing gate "
                    "evidence"
                )
            if not self.gate.metrics:
                raise ValueError(
                    "An adopt decision requires gate evidence with recorded "
                    "metrics; an empty gate result proves no evaluation "
                    "rule was applied"
                )
        return self

    def as_tags(self) -> dict[str, str]:
        """Searchable tag values; governed runs prefix them with ``aai.``."""

        values = {"decision": self.decision.value}
        if self.change_run_id:
            values["change_run_id"] = self.change_run_id
        if self.gate is not None:
            values["gate_passed"] = str(self.gate.passed).lower()
        if self.prompt_digest:
            values["prompt_digest"] = self.prompt_digest
        if self.release_digest:
            values["release_digest"] = self.release_digest
        return values


def record_decision(
    record: DecisionRecord,
    *,
    experiments: ExperimentManager,
    run_name: str | None = None,
) -> str:
    """Persist a decision as a governed run and return its run id.

    The run carries ``aai.run_purpose="decision"``, the searchable
    ``aai.decision`` tags, the gate metrics when present, and the complete
    record as a ``decision.json`` artifact.
    """

    resolved_name = run_name or f"decision-{record.change_id}"
    metadata = ExperimentRunMetadata(
        purpose=RunPurpose.DECISION,
        change_id=record.change_id,
        change_summary=record.change_summary,
        baseline_run_id=record.baseline_run_id,
    )
    mlflow = experiments.native_client
    with experiments.run(
        run_name=resolved_name,
        description=record.rationale,
        tags=record.as_tags(),
        metadata=metadata,
    ) as active_run:
        if record.gate is not None:
            mlflow.log_metrics(dict(record.gate.metrics))
        payload = json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True)
        with tempfile.TemporaryDirectory() as scratch:
            decision_file = Path(scratch) / "decision.json"
            decision_file.write_text(payload + "\n", encoding="utf-8")
            mlflow.log_artifact(str(decision_file), artifact_path="decision")
        return str(active_run.info.run_id)
