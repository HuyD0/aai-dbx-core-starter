"""Persistable lifecycle decisions binding baseline, gate, and release evidence.

The lifecycle vocabulary ends every comparison in an explicit ``adopt``,
``reject``, or ``inconclusive`` decision. This module gives that decision a
strict, persisted contract and records it as a governed MLflow run so the
decision is searchable next to the evidence it was made from. It composes
:class:`~aai_core.experiments.ExperimentManager`; it owns no MLflow surface
of its own.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from re import fullmatch
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from aai_core.contracts import ContractModel
from aai_core.evaluation import (
    GateResult,
    _is_missing_registry_error,
    _is_placeholder,
)
from aai_core.exceptions import AaiCoreError
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


class DecisionEvidenceError(AaiCoreError):
    """A persisted decision run is missing or contradicts its artifact."""

    code = "aai_core.decisions.evidence_invalid"


_RUN_ID_PATTERN = r"[A-Za-z0-9_-]{1,64}"


class DecisionRecord(ContractModel):
    """Immutable decision evidence for one deliberate change."""

    decision: Decision
    # change_id and change_summary become searchable run tags, so the id is
    # an identifier and the summary is bounded prose: prompts, user
    # content, and secrets belong in artifacts, never tags. The free-form
    # rationale is persisted only inside the decision.json artifact.
    change_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    change_summary: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1)
    # Bounded opaque identifiers (MLflow run ids are 32-hex; fixtures and
    # backends vary) so free text and secrets cannot enter governed tags.
    baseline_run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    change_run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    gate: GateResult | None = None
    # The digest binds content; the qualified name and immutable version
    # bind registry identity, so evidence for one prompt can never promote
    # another prompt that happens to share a template. The constrained
    # catalog.schema.name shape keeps typos, prompt text, and secrets out
    # of the governed tag.
    prompt_name: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$",
    )
    prompt_version: int | None = Field(default=None, ge=1)
    # Exactly sha256 hexdigests (prompt_digest() and
    # ApplicationRelease.digest): raw prompt text, user content, or secrets
    # can never enter the persisted tags through these fields.
    prompt_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    release_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
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

    @field_validator("change_summary", "rationale", "decided_by")
    @classmethod
    def require_substantive_text(cls, value: str | None) -> str | None:
        # min_length alone accepts "   ", which would tag a blank summary
        # and persist a decision.json with no stated reasoning. Trim so
        # the stored evidence is exactly what a reader sees, and refuse a
        # value that says nothing.
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            raise ValueError(
                "decision evidence must be substantive; a whitespace-only "
                "value records nothing"
            )
        return trimmed

    @field_validator("prompt_name")
    @classmethod
    def refuse_placeholder_components(cls, value: str | None) -> str | None:
        if value is not None and any(
            _is_placeholder(part) for part in value.split(".")
        ):
            raise ValueError(
                "prompt_name must not contain placeholder components; record "
                "the real qualified prompt name the evidence was made for"
            )
        return value

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
            policy = self.gate.policy
            if policy is None:
                raise ValueError(
                    "An adopt decision requires gate evidence that records "
                    "the applied release policy; produce the gate with "
                    "apply_gate() so the policy travels with the result"
                )
            # A rule only constrains the release if it was actually
            # applied: an absolute threshold always is, a positive
            # cost-coverage minimum always is, but a regression-only rule
            # is skipped by _evaluate_policy when the baseline lacks its
            # metric and the policy waives missing baselines — evidence
            # from such a gate proves nothing was checked.
            baseline = self.gate.baseline_metrics or {}
            enforced = bool(policy.minimum_cost_coverage) or any(
                rule.required is not None
                or (rule.max_regression is not None and rule.metric in baseline)
                for rule in policy.rules
            )
            if not enforced:
                raise ValueError(
                    "An adopt decision requires gate evidence whose applied "
                    "policy enforced at least one substantive release rule; "
                    "a rule-free policy, a zero cost-coverage threshold "
                    "(which rejects no coverage value), and regression-only "
                    "rules evaluated without their baseline values all gate "
                    "nothing"
                )
        return self

    def as_tags(self) -> dict[str, str]:
        """Searchable tag values; governed runs prefix them with ``aai.``."""

        values = {"decision": self.decision.value}
        if self.change_run_id:
            values["change_run_id"] = self.change_run_id
        if self.gate is not None:
            values["gate_passed"] = str(self.gate.passed).lower()
        if self.prompt_name:
            values["prompt_name"] = self.prompt_name
        if self.prompt_version is not None:
            values["prompt_version"] = str(self.prompt_version)
        if self.prompt_digest:
            values["prompt_digest"] = self.prompt_digest
        if self.release_digest:
            values["release_digest"] = self.release_digest
        return values


def decision_digest(record: DecisionRecord) -> str:
    """Return the canonical digest used to bind a run to ``decision.json``."""

    canonical = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_decision(
    decision_run_id: str,
    *,
    mlflow_module: Any | None = None,
) -> DecisionRecord:
    """Load and verify a decision from its finished governed MLflow run.

    The strict ``decision.json`` artifact is the source of truth. Its
    canonical digest, searchable lifecycle tags, gate metrics, run purpose,
    run identity, and terminal status must all agree before the record can be
    used for promotion. Provider authentication and transport failures are
    deliberately allowed to propagate; only contradictory evidence is
    converted to :class:`DecisionEvidenceError`.
    """

    run_id = _validated_run_id(decision_run_id)
    mlflow = _decision_client(mlflow_module)
    client = mlflow.MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception as error:
        if not _is_missing_decision_resource(error):
            raise
        raise DecisionEvidenceError("The cited decision run does not exist.") from error
    info = getattr(run, "info", None)
    if str(getattr(info, "run_id", "")) != run_id:
        raise DecisionEvidenceError(
            "The decision lookup returned a different run identity."
        )
    if str(getattr(info, "status", "")).upper() != "FINISHED":
        raise DecisionEvidenceError(
            "The decision run is not finished; only completed evidence may "
            "authorize promotion."
        )

    try:
        artifact_path = client.download_artifacts(run_id, "decision/decision.json")
    except Exception as error:
        # A finished run without the decision artifact is invalid evidence,
        # not a provider outage. Structured auth and transient codes override
        # not-found wording inside the shared classifier and still propagate.
        if not _is_missing_decision_resource(error):
            raise
        raise DecisionEvidenceError(
            "The decision run does not contain a valid decision/decision.json "
            "artifact."
        ) from error
    try:
        record = DecisionRecord.model_validate_json(
            Path(artifact_path).read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError) as error:
        raise DecisionEvidenceError(
            "The decision run does not contain a valid decision/decision.json "
            "artifact."
        ) from error

    data = getattr(run, "data", None)
    tags = getattr(data, "tags", None)
    metrics = getattr(data, "metrics", None)
    if not isinstance(tags, Mapping) or not isinstance(metrics, Mapping):
        raise DecisionEvidenceError(
            "The decision run does not expose verifiable tags and metrics."
        )

    expected_tags = _persisted_tags(record)
    for name, expected in expected_tags.items():
        if tags.get(name) != expected:
            raise DecisionEvidenceError(
                f"The decision run tag {name!r} contradicts decision.json."
            )

    expected_metrics = dict(record.gate.metrics) if record.gate is not None else {}
    for name, expected in expected_metrics.items():
        observed = metrics.get(name)
        if not isinstance(observed, (int, float)) or float(observed) != float(expected):
            raise DecisionEvidenceError(
                f"The decision run metric {name!r} contradicts decision.json."
            )
    return record


def _persisted_tags(record: DecisionRecord) -> dict[str, str]:
    tags = {
        "aai.run_purpose": RunPurpose.DECISION.value,
        "aai.change_id": record.change_id,
        "aai.change_summary": record.change_summary,
        "aai.decision_digest": decision_digest(record),
        **{f"aai.{name}": value for name, value in record.as_tags().items()},
    }
    if record.baseline_run_id:
        tags["aai.baseline_run_id"] = record.baseline_run_id
    return tags


def _is_missing_decision_resource(error: Exception) -> bool:
    """Recognize only authoritative run/artifact absence."""

    return isinstance(error, FileNotFoundError) or _is_missing_registry_error(error)


def _validated_run_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"decision_run_id must be a string; got {type(value).__name__}")
    if not fullmatch(_RUN_ID_PATTERN, value):
        raise ValueError(
            "decision_run_id must be a bounded opaque run identifier containing "
            "only letters, digits, underscores, and hyphens"
        )
    return value


def _decision_client(mlflow_module: Any | None) -> Any:
    if mlflow_module is not None:
        return mlflow_module
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "Decision evidence requires the `genai` extra. Install "
            "`aai-core[genai]` in the consuming environment."
        ) from error
    return mlflow


def record_decision(
    record: DecisionRecord,
    *,
    experiments: ExperimentManager,
) -> str:
    """Persist a decision as a governed run and return its run id.

    The run carries ``aai.run_purpose="decision"``, the searchable
    ``aai.decision`` tags, the gate metrics when present, and the complete
    record as a ``decision.json`` artifact. The run name derives
    exclusively from the bounded ``change_id`` — MLflow persists run names
    as the ``mlflow.runName`` tag, so a free-form override would let
    prompts, user content, or secrets bypass the record's bounded fields.
    """

    # Reconstruct through validation before anything is written. The
    # contract's guarantees — an adopt cites a passing, substantively
    # gated result; identifiers stay bounded — hold at construction, but
    # model_copy(update=...) skips validators, so a derived record could
    # otherwise persist tags, metrics, and a decision.json that
    # contradict each other.
    record = DecisionRecord.model_validate(record.model_dump())
    resolved_name = f"decision-{record.change_id}"
    metadata = ExperimentRunMetadata(
        purpose=RunPurpose.DECISION,
        change_id=record.change_id,
        change_summary=record.change_summary,
        baseline_run_id=record.baseline_run_id,
    )
    mlflow = experiments.native_client
    # No description: MLflow persists run descriptions as the
    # mlflow.note.content tag, and the free-form rationale belongs only in
    # the decision.json artifact.
    with experiments.run(
        run_name=resolved_name,
        tags={**record.as_tags(), "decision_digest": decision_digest(record)},
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
