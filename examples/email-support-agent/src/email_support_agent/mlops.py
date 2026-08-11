"""Credential-safe MLflow 3 and AI/ML operations recipes.

The module is deliberately plan-first.  Importing it, building a plan, or
calling an apply function with its default policy cannot import MLflow or make
a provider call.  Connected operations require both ``mode=execute`` and an
explicit acknowledgement in the process environment.

Scorer definitions remain owned by :mod:`aai_core.agentkit`.  This module only
selects shared scorer names, estimates their cost, and orchestrates native
MLflow APIs after the mutation guard has passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from aai_core.agentkit.catalog import (
    ScorerKind,
    ScorerPlan,
    build_scorer,
    get_spec,
    select_scorers,
)
from aai_core.agentkit.config import load_config
from aai_core.agentkit.cost import CostEstimate, enforce_budget, estimate
from aai_core.agentkit.datasets import load_dataset
from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.decisions import Decision, DecisionRecord
from aai_core.deployment import ApplicationRelease
from aai_core.evaluation import GateResult, MetricDirection
from aai_core.experiments import RunPurpose
from aai_core.tags import ResourceContext
from aai_core.tracing import (
    TraceCaptureMode,
    TraceIntegration,
    TracePolicy,
)
from email_support_agent.contracts import (
    MeasurementSource,
    RiskTier,
    obvious_sensitive_fragments,
)

_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_OPAQUE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
_QUALIFIED_PROMPT = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
_GROUP_SOURCE = re.compile(r"^group:[A-Za-z0-9._-]{1,64}$")
_EXPLICIT_MUTATION_ACK = "I_UNDERSTAND_THIS_MUTATES_MLFLOW"
_FORBIDDEN_CURATED_KEYS = frozenset(
    {"mime_bytes", "raw_email", "raw_mime", "sender_email"}
)


class MutationRefusedError(RuntimeError):
    """A connected operation did not receive both required opt-ins."""


class PromotionEvidenceError(ValueError):
    """Release evidence is not strong enough for a human adopt decision."""


class ExecutionMode(StrEnum):
    DRY_RUN = "dry_run"
    EXECUTE = "execute"


class ExecutionPolicy(ContractModel):
    """Double opt-in for any operation that can spend money or mutate MLflow."""

    mode: ExecutionMode = ExecutionMode.DRY_RUN
    acknowledgement_env: str = "AAI_ENABLE_MLFLOW_MUTATIONS"
    notebook_confirmed: bool = False

    @field_validator("acknowledgement_env")
    @classmethod
    def validate_environment_name(cls, value: str) -> str:
        return _environment_name(value)


class OperationReceipt(ContractModel):
    operation: str = Field(min_length=1)
    applied: bool
    mode: ExecutionMode
    referenced_environment_variables: tuple[str, ...] = ()
    note: str = Field(min_length=1)


def _mutation_enabled(
    policy: ExecutionPolicy,
    *,
    environ: Mapping[str, str] | None,
    notebook_required: bool,
) -> bool:
    if policy.mode is ExecutionMode.DRY_RUN:
        return False
    env = os.environ if environ is None else environ
    if env.get(policy.acknowledgement_env) != _EXPLICIT_MUTATION_ACK:
        raise MutationRefusedError(
            "execution was requested, but the explicit MLflow mutation "
            f"acknowledgement is missing; set {policy.acknowledgement_env} "
            f"to {_EXPLICIT_MUTATION_ACK!r} for this process"
        )
    if notebook_required and not policy.notebook_confirmed:
        raise MutationRefusedError(
            "this operation must run from the reviewed Databricks notebook; "
            "set notebook_confirmed=true only in that notebook"
        )
    return True


def _environment_name(value: str) -> str:
    trimmed = value.strip()
    if not _ENVIRONMENT_NAME.fullmatch(trimmed):
        raise ValueError(
            "configuration must name an environment variable, not contain an "
            "endpoint, experiment, catalog, schema, or other environment id"
        )
    return trimmed


def _required_environment_value(
    name: str,
    environ: Mapping[str, str] | None,
) -> str:
    env = os.environ if environ is None else environ
    value = env.get(name)
    if value is None or not value.strip():
        raise MutationRefusedError(
            f"connected execution requires a nonblank value in {name}"
        )
    return value.strip()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TraceDestinationPlan(ContractModel):
    """Production trace configuration containing references, never ids."""

    tracking_uri_env: str = "AAI_MLFLOW_TRACKING_URI"
    experiment_name_env: str = "AAI_MLFLOW_EXPERIMENT_NAME"
    trace_destination_env: str = "MLFLOW_TRACING_DESTINATION"
    sql_warehouse_id_env: str | None = "MLFLOW_TRACING_SQL_WAREHOUSE_ID"
    integration: TraceIntegration = TraceIntegration.SDK
    policy: TracePolicy = Field(
        default_factory=lambda: TracePolicy(
            capture_mode=TraceCaptureMode.BOUNDED,
            max_string_length=4_096,
            max_collection_items=100,
        )
    )

    @field_validator(
        "tracking_uri_env",
        "experiment_name_env",
        "trace_destination_env",
        "sql_warehouse_id_env",
    )
    @classmethod
    def validate_environment_references(cls, value: str | None) -> str | None:
        return None if value is None else _environment_name(value)

    @model_validator(mode="after")
    def require_sdk_owned_capture(self) -> Self:
        if self.integration is not TraceIntegration.SDK:
            raise ValueError(
                "the email accelerator uses SDK-owned spans so bounded/redacted "
                "capture can be enforced before payloads reach MLflow"
            )
        if self.policy.capture_mode is TraceCaptureMode.FULL:
            raise ValueError(
                "FULL capture is not a production default for support email; "
                "use BOUNDED, REDACTED, METADATA_ONLY, or OFF"
            )
        return self


def apply_trace_destination(
    plan: TraceDestinationPlan,
    *,
    context: ResourceContext | None = None,
    execution: ExecutionPolicy | None = None,
    environ: Mapping[str, str] | None = None,
    tracing_configurer: Callable[..., Any] | None = None,
) -> OperationReceipt:
    """Apply startup tracing only after the double opt-in.

    The destination itself is consumed by MLflow from
    ``MLFLOW_TRACING_DESTINATION``.  The function validates that all declared
    environment bindings exist and delegates SDK policy/context projection to
    :func:`aai_core.tracing.configure_tracing`.
    """

    execution = execution or ExecutionPolicy()
    references = tuple(
        name
        for name in (
            plan.tracking_uri_env,
            plan.experiment_name_env,
            plan.trace_destination_env,
            plan.sql_warehouse_id_env,
        )
        if name is not None
    )
    if not _mutation_enabled(execution, environ=environ, notebook_required=False):
        return OperationReceipt(
            operation="configure_production_tracing",
            applied=False,
            mode=execution.mode,
            referenced_environment_variables=references,
            note="dry-run: no environment was resolved and MLflow was not imported",
        )
    if context is None:
        raise MutationRefusedError(
            "connected tracing requires the bootstrapped ResourceContext"
        )
    tracking_uri = _required_environment_value(plan.tracking_uri_env, environ)
    experiment_name = _required_environment_value(plan.experiment_name_env, environ)
    _required_environment_value(plan.trace_destination_env, environ)
    if plan.sql_warehouse_id_env is not None:
        _required_environment_value(plan.sql_warehouse_id_env, environ)
    if tracing_configurer is None:
        from aai_core.tracing import configure_tracing as tracing_configurer

    tracing_configurer(
        context,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        integration=plan.integration,
        policy=plan.policy,
    )
    return OperationReceipt(
        operation="configure_production_tracing",
        applied=True,
        mode=execution.mode,
        referenced_environment_variables=references,
        note="tracing configured from environment bindings",
    )


class RiskSamplingPlan(ContractModel):
    """Hybrid sampling: a native floor plus deterministic risk oversampling."""

    native_sample_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    low: float = Field(default=0.05, ge=0.0, le=1.0)
    medium: float = Field(default=0.20, ge=0.0, le=1.0)
    high: float = Field(default=0.75, ge=0.0, le=1.0)
    critical: float = Field(default=1.0, ge=0.0, le=1.0)
    deterministic_salt_version: str = Field(
        default="email-support-risk-sampling-v1", min_length=1, max_length=64
    )

    @model_validator(mode="after")
    def require_monotonic_risk_sampling(self) -> Self:
        rates = (self.low, self.medium, self.high, self.critical)
        if rates != tuple(sorted(rates)):
            raise ValueError("sampling rates must not decrease as risk increases")
        if self.native_sample_rate > self.low:
            raise ValueError(
                "native_sample_rate is the all-traffic floor and cannot exceed "
                "the low-risk batch rate"
            )
        return self

    def for_risk(self, risk: RiskTier) -> float:
        return {
            RiskTier.LOW: self.low,
            RiskTier.MEDIUM: self.medium,
            RiskTier.HIGH: self.high,
            RiskTier.CRITICAL: self.critical,
        }[risk]


def select_for_risk_batch(
    *,
    trace_id: str,
    risk: RiskTier,
    policy: RiskSamplingPlan,
) -> bool:
    """Deterministically select a trace without storing customer identifiers."""

    if not _OPAQUE_IDENTIFIER.fullmatch(trace_id):
        raise ValueError("trace_id must be a bounded opaque identifier")
    rate = policy.for_risk(risk)
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    digest = hashlib.sha256(
        f"{policy.deterministic_salt_version}:{trace_id}".encode()
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < rate


class MonitoringPlan(ContractModel):
    """Shared scorers to register/start plus the risk-aware batch overlay.

    MLflow's current ``ScorerSamplingConfig`` exposes a global sample rate,
    not a trace-tag predicate.  The native rate is therefore a cheap floor;
    ``select_for_risk_batch`` oversamples high-risk traces for scheduled full
    Agentkit evaluation without inventing a new scorer.
    """

    scorer_names: tuple[str, ...] = ("safety",)
    experiment_name_env: str = "AAI_MLFLOW_EXPERIMENT_NAME"
    judge_model_uri_env: str = "AAI_JUDGE_MODEL_URI"
    risk_sampling: RiskSamplingPlan = Field(default_factory=RiskSamplingPlan)
    guidelines: tuple[str, ...] = ()
    requires_databricks_notebook: Literal[True] = True

    @field_validator("experiment_name_env", "judge_model_uri_env")
    @classmethod
    def validate_environment_references(cls, value: str) -> str:
        return _environment_name(value)

    @field_validator("scorer_names")
    @classmethod
    def select_only_shared_production_scorers(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("scorer_names must be a non-empty unique selection")
        for name in value:
            spec = get_spec(name)
            if spec.needs_expectations:
                raise ValueError(
                    f"production trace scorer {name!r} requires ground-truth "
                    "expectations; use it in curated batch evaluation instead"
                )
            if spec.kind is ScorerKind.CODE:
                raise ValueError(
                    f"registered code scorer {name!r} must use a platform-owned "
                    "self-contained monitoring body, not a project closure"
                )
        return value

    @model_validator(mode="after")
    def require_guidelines_text(self) -> Self:
        if "guidelines" in self.scorer_names and not self.guidelines:
            raise ValueError("the shared guidelines scorer requires guidelines")
        return self

    @property
    def scorer_versions(self) -> tuple[str, ...]:
        return tuple(f"{name}={get_spec(name).version}" for name in self.scorer_names)


def monitoring_plan_from_agentkit(
    project_root: str | Path,
    *,
    config_name: str = "agentkit.yaml",
) -> MonitoringPlan:
    """Select continuous scorers and guidelines from project Agentkit config.

    Expensive retrieval scorers stay in the risk-sampled batch gate. Continuous
    scoring is limited to the shared safety/guidelines assets selected by the
    project, so this helper neither copies nor redefines judge instructions.
    """

    root = Path(project_root)
    config = load_config(root / config_name, environ={})
    selected = tuple(
        name for name in ("safety", "guidelines") if name in config.scorers.add
    )
    if not selected:
        selected = ("safety",)
    return MonitoringPlan(
        scorer_names=selected,
        guidelines=config.scorers.guidelines,
    )


def register_and_start_monitoring(
    plan: MonitoringPlan,
    *,
    execution: ExecutionPolicy | None = None,
    environ: Mapping[str, str] | None = None,
    mlflow_module: Any | None = None,
) -> OperationReceipt:
    """Register *and* start shared scorers from a reviewed notebook only."""

    execution = execution or ExecutionPolicy()
    references = (plan.experiment_name_env, plan.judge_model_uri_env)
    if not _mutation_enabled(
        execution,
        environ=environ,
        notebook_required=plan.requires_databricks_notebook,
    ):
        return OperationReceipt(
            operation="register_and_start_monitoring_scorers",
            applied=False,
            mode=execution.mode,
            referenced_environment_variables=references,
            note=(
                "dry-run: shared scorer versions selected; no scorer was "
                "registered or started"
            ),
        )
    experiment_name = _required_environment_value(plan.experiment_name_env, environ)
    judge_model_uri = _required_environment_value(plan.judge_model_uri_env, environ)
    if mlflow_module is None:
        import mlflow as mlflow_module  # type: ignore[no-redef]

    mlflow_module.set_experiment(experiment_name)
    sampling_config = mlflow_module.genai.scorers.ScorerSamplingConfig(
        sample_rate=plan.risk_sampling.native_sample_rate
    )
    for scorer_name in plan.scorer_names:
        scorer = build_scorer(
            get_spec(scorer_name),
            judge_model_uri=judge_model_uri,
            guidelines=plan.guidelines,
            mlflow_module=mlflow_module,
        )
        # The registered name intentionally equals the shared registry name.
        # Renaming it would sever metric/feedback lineage and judge-label binding.
        scorer.register(name=scorer_name).start(sampling_config=sampling_config)
    return OperationReceipt(
        operation="register_and_start_monitoring_scorers",
        applied=True,
        mode=execution.mode,
        referenced_environment_variables=references,
        note="shared scorers registered and started at the all-traffic floor",
    )


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return freeze_value(value)


def _curation_privacy_findings(value: Any, *, path: str = "payload") -> tuple[str, ...]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text.casefold() in _FORBIDDEN_CURATED_KEYS:
                findings.append(f"forbidden raw field at {path}.{key_text}")
            findings.extend(_curation_privacy_findings(item, path=f"{path}.{key_text}"))
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            findings.extend(_curation_privacy_findings(item, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        findings.extend(
            f"{finding} at {path}" for finding in obvious_sensitive_fragments(value)
        )
    return tuple(findings)


class TraceCurationPlan(ContractModel):
    source_experiment_env: str = "AAI_MLFLOW_EXPERIMENT_ID"
    target_dataset_env: str = "AAI_MLFLOW_REGRESSION_DATASET"
    feedback_name: str = Field(default="support_review_outcome", min_length=1)
    included_feedback_values: tuple[str, ...] = ("edited", "rejected")
    human_source: str = "group:support-quality"
    required_expectations: tuple[str, ...] = (
        "expected_response",
        "expected_intent",
        "expected_urgency",
        "expected_route",
        "requires_review",
    )
    risk_sampling: RiskSamplingPlan = Field(default_factory=RiskSamplingPlan)

    @field_validator("source_experiment_env", "target_dataset_env")
    @classmethod
    def validate_environment_references(cls, value: str) -> str:
        return _environment_name(value)

    @field_validator("human_source")
    @classmethod
    def require_group_source(cls, value: str) -> str:
        if not _GROUP_SOURCE.fullmatch(value):
            raise ValueError("human feedback must use group:<reviewer-group>")
        return value

    @model_validator(mode="after")
    def require_nonempty_contract(self) -> Self:
        if not self.included_feedback_values or not self.required_expectations:
            raise ValueError("curation must select feedback and expectation fields")
        return self


class ReviewedTrace(ContractModel):
    """Redacted trace material and reviewed truth ready for curation."""

    trace_id: str
    risk: RiskTier
    inputs: Mapping[str, JsonValue]
    outputs: JsonValue
    trace: Mapping[str, JsonValue]
    expectations: Mapping[str, JsonValue]
    feedback_name: str = Field(min_length=1)
    feedback_value: str = Field(min_length=1)
    feedback_source: str
    rationale: str = Field(min_length=1, max_length=2_000)
    dlp_evidence_ref: str = Field(min_length=16, max_length=512)

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str) -> str:
        if not _OPAQUE_IDENTIFIER.fullmatch(value):
            raise ValueError("trace_id must be a bounded opaque identifier")
        return value

    @field_validator("feedback_source")
    @classmethod
    def validate_feedback_source(cls, value: str) -> str:
        if not _GROUP_SOURCE.fullmatch(value):
            raise ValueError("feedback_source must be group:<reviewer-group>")
        return value

    @field_validator("dlp_evidence_ref")
    @classmethod
    def validate_dlp_evidence_ref(cls, value: str) -> str:
        if not value.startswith(("secure://dlp/", "synthetic://dlp/")) or any(
            character in value for character in ("@", "?", "#")
        ):
            raise ValueError(
                "dlp_evidence_ref must be an opaque secure://dlp/ or "
                "synthetic://dlp/ reference"
            )
        return value

    @field_validator("inputs", "trace", "expectations", mode="after")
    @classmethod
    def freeze_mappings(cls, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        if not value:
            raise ValueError("curated trace mappings must not be empty")
        return _freeze_mapping(value)

    @field_serializer("inputs", "trace", "expectations")
    def serialize_mappings(
        self, value: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return thaw_value(value)

    @model_validator(mode="after")
    def reject_sensitive_or_raw_material(self) -> Self:
        findings = _curation_privacy_findings(
            {
                "inputs": self.inputs,
                "outputs": self.outputs,
                "trace": self.trace,
                "expectations": self.expectations,
                "rationale": self.rationale,
            }
        )
        if findings:
            raise ValueError(
                "reviewed trace failed the local privacy boundary: "
                + "; ".join(findings[:8])
            )
        return self


class NestedEvaluationRecord(ContractModel):
    """Native MLflow GenAI row shape; inputs and expectations stay nested."""

    inputs: Mapping[str, JsonValue]
    expectations: Mapping[str, JsonValue]
    outputs: JsonValue
    trace: Mapping[str, JsonValue]
    tags: Mapping[str, str]

    @field_validator("inputs", "expectations", "trace", "tags", mode="after")
    @classmethod
    def freeze_mappings(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if not value:
            raise ValueError("evaluation record mappings must not be empty")
        return _freeze_mapping(value)

    @field_serializer("inputs", "expectations", "trace", "tags")
    def serialize_mappings(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_value(value)

    def as_mlflow_record(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json")


def curate_reviewed_trace(
    trace: ReviewedTrace,
    *,
    plan: TraceCurationPlan,
) -> NestedEvaluationRecord | None:
    """Convert one reviewed failure to a nested, deduplicable MLflow record."""

    if trace.feedback_name != plan.feedback_name:
        return None
    if trace.feedback_value not in plan.included_feedback_values:
        return None
    if trace.feedback_source != plan.human_source:
        return None
    if not select_for_risk_batch(
        trace_id=trace.trace_id,
        risk=trace.risk,
        policy=plan.risk_sampling,
    ):
        return None
    missing = [
        name
        for name in plan.required_expectations
        if name not in trace.expectations
        or trace.expectations[name] is None
        or trace.expectations[name] == ""
    ]
    if missing:
        raise ValueError(
            "reviewed trace lacks required expectations: " + ", ".join(missing)
        )
    # The source trace is represented by a digest in the governed dataset; the
    # operational trace id remains available only in MLflow's access-controlled
    # trace store.
    return NestedEvaluationRecord(
        inputs=thaw_value(trace.inputs),
        expectations=thaw_value(trace.expectations),
        outputs=thaw_value(trace.outputs),
        trace=thaw_value(trace.trace),
        tags={
            "risk_tier": trace.risk.value,
            "curation_source": trace.feedback_source,
            "source_trace_sha256": _sha256_text(trace.trace_id),
            "dlp_evidence_sha256": _sha256_text(trace.dlp_evidence_ref),
            "curation_schema": "1",
        },
    )


def merge_curated_records(
    records: Sequence[NestedEvaluationRecord],
    *,
    plan: TraceCurationPlan,
    execution: ExecutionPolicy | None = None,
    environ: Mapping[str, str] | None = None,
    mlflow_module: Any | None = None,
) -> OperationReceipt:
    execution = execution or ExecutionPolicy()
    references = (plan.source_experiment_env, plan.target_dataset_env)
    if not _mutation_enabled(execution, environ=environ, notebook_required=True):
        return OperationReceipt(
            operation="merge_curated_trace_records",
            applied=False,
            mode=execution.mode,
            referenced_environment_variables=references,
            note=f"dry-run: {len(records)} record(s) validated; dataset untouched",
        )
    if not records:
        raise MutationRefusedError("refusing a dataset mutation with no records")
    _required_environment_value(plan.source_experiment_env, environ)
    dataset_name = _required_environment_value(plan.target_dataset_env, environ)
    if mlflow_module is None:
        import mlflow as mlflow_module  # type: ignore[no-redef]

    dataset = mlflow_module.genai.datasets.get_dataset(dataset_name)
    dataset.merge_records([record.as_mlflow_record() for record in records])
    return OperationReceipt(
        operation="merge_curated_trace_records",
        applied=True,
        mode=execution.mode,
        referenced_environment_variables=references,
        note=f"merged {len(records)} reviewed record(s)",
    )


class RegisteredScorerRef(ContractModel):
    """Immutable lookup for one platform-owned shared scorer version."""

    shared_name: str = Field(min_length=1)
    experiment_id_env: str = "AAI_MLFLOW_EXPERIMENT_ID"
    version_env: str = "AAI_ALIGNED_JUDGE_VERSION"

    @field_validator("shared_name")
    @classmethod
    def require_shared_scorer(cls, value: str) -> str:
        get_spec(value)
        return value

    @field_validator("experiment_id_env", "version_env")
    @classmethod
    def validate_environment_references(cls, value: str) -> str:
        return _environment_name(value)


class JudgeAlignmentPlan(ContractModel):
    """SME/MemAlign plan that emits a candidate, never updates a scorer."""

    judge: RegisteredScorerRef
    label_schema_name: str = Field(min_length=1)
    human_source: str = "group:support-quality"
    trace_filter: str = "tag.aai.labeling_status = 'complete'"
    reflection_model_uri_env: str = "AAI_ALIGNMENT_REFLECTION_MODEL_URI"
    embedding_model_uri_env: str = "AAI_ALIGNMENT_EMBEDDING_MODEL_URI"
    minimum_labeled_traces: int = Field(default=50, ge=20)
    maximum_alignment_traces: int = Field(default=500, ge=20, le=5_000)
    retrieval_k: int = Field(default=5, ge=1, le=20)
    estimated_embedding_tokens_per_trace: int = Field(default=2_000, ge=1)
    proposal_only: Literal[True] = True
    auto_register: Literal[False] = False

    @field_validator("human_source")
    @classmethod
    def require_group_source(cls, value: str) -> str:
        if not _GROUP_SOURCE.fullmatch(value):
            raise ValueError("SME feedback must use group:<reviewer-group>")
        return value

    @field_validator("reflection_model_uri_env", "embedding_model_uri_env")
    @classmethod
    def validate_environment_references(cls, value: str) -> str:
        return _environment_name(value)

    @model_validator(mode="after")
    def bind_label_to_judge_name(self) -> Self:
        if self.label_schema_name != self.judge.shared_name:
            raise ValueError(
                "MemAlign requires label_schema_name to exactly match the "
                "platform judge name"
            )
        if self.maximum_alignment_traces < self.minimum_labeled_traces:
            raise ValueError(
                "maximum_alignment_traces cannot be below minimum_labeled_traces"
            )
        return self


class AlignmentBudgetEstimate(ContractModel):
    labeled_traces: int = Field(ge=0)
    embedding_calls: int = Field(ge=0)
    estimated_embedding_tokens: int = Field(ge=0)
    retrieval_k: int = Field(ge=1)
    capped: bool


def estimate_alignment_budget(
    plan: JudgeAlignmentPlan,
    *,
    labeled_trace_count: int,
) -> AlignmentBudgetEstimate:
    if labeled_trace_count < 0:
        raise ValueError("labeled_trace_count cannot be negative")
    selected = min(labeled_trace_count, plan.maximum_alignment_traces)
    return AlignmentBudgetEstimate(
        labeled_traces=selected,
        embedding_calls=selected,
        estimated_embedding_tokens=(
            selected * plan.estimated_embedding_tokens_per_trace
        ),
        retrieval_k=plan.retrieval_k,
        capped=labeled_trace_count > selected,
    )


def run_memalign_proposal(
    plan: JudgeAlignmentPlan,
    *,
    execution: ExecutionPolicy | None = None,
    environ: Mapping[str, str] | None = None,
    mlflow_module: Any | None = None,
) -> tuple[OperationReceipt, Any | None]:
    """Run alignment behind the guard and return an unregistered candidate."""

    execution = execution or ExecutionPolicy()
    references = (
        plan.judge.experiment_id_env,
        plan.judge.version_env,
        plan.reflection_model_uri_env,
        plan.embedding_model_uri_env,
    )
    if not _mutation_enabled(execution, environ=environ, notebook_required=True):
        return (
            OperationReceipt(
                operation="memalign_judge_proposal",
                applied=False,
                mode=execution.mode,
                referenced_environment_variables=references,
                note=(
                    "dry-run: no traces were searched and no alignment model "
                    "was called"
                ),
            ),
            None,
        )
    experiment_id = _required_environment_value(plan.judge.experiment_id_env, environ)
    version_text = _required_environment_value(plan.judge.version_env, environ)
    try:
        version = int(version_text)
    except ValueError as error:
        raise MutationRefusedError(
            "judge version must be a positive integer"
        ) from error
    if version < 1:
        raise MutationRefusedError("judge version must be a positive integer")
    reflection_model = _required_environment_value(
        plan.reflection_model_uri_env, environ
    )
    embedding_model = _required_environment_value(plan.embedding_model_uri_env, environ)
    if mlflow_module is None:
        import mlflow as mlflow_module  # type: ignore[no-redef]

    traces = mlflow_module.search_traces(
        locations=[experiment_id],
        filter_string=plan.trace_filter,
        max_results=plan.maximum_alignment_traces,
        return_type="list",
    )
    if len(traces) < plan.minimum_labeled_traces:
        raise MutationRefusedError(
            f"alignment requires at least {plan.minimum_labeled_traces} labeled "
            f"traces; found {len(traces)}"
        )
    base_judge = mlflow_module.genai.scorers.get_scorer(
        name=plan.judge.shared_name,
        experiment_id=experiment_id,
        version=version,
    )
    try:
        optimizer_class = mlflow_module.genai.judges.optimizers.MemAlignOptimizer
    except AttributeError:
        from mlflow.genai.judges.optimizers import MemAlignOptimizer

        optimizer_class = MemAlignOptimizer
    optimizer = optimizer_class(
        reflection_lm=reflection_model,
        retrieval_k=plan.retrieval_k,
        embedding_model=embedding_model,
    )
    aligned_candidate = base_judge.align(traces=traces, optimizer=optimizer)
    return (
        OperationReceipt(
            operation="memalign_judge_proposal",
            applied=True,
            mode=execution.mode,
            referenced_environment_variables=references,
            note=(
                "alignment candidate created in memory; it was not registered, "
                "updated, or started"
            ),
        ),
        aligned_candidate,
    )


class GepaOptimizationPlan(ContractModel):
    """Bounded GEPA experiment that can only produce a prompt proposal."""

    seed_prompt_uri_env: str = "AAI_SEED_PROMPT_URI"
    training_dataset_env: str = "AAI_GEPA_TRAIN_DATASET"
    judge_calibration_dataset_env: str = "AAI_JUDGE_CALIBRATION_DATASET"
    heldout_release_dataset_env: str = "AAI_HELDOUT_RELEASE_DATASET"
    reflection_model_uri_env: str = "AAI_GEPA_REFLECTION_MODEL_URI"
    scorers: tuple[RegisteredScorerRef, ...]
    max_metric_calls: int = Field(default=40, ge=1, le=1_000)
    estimated_tokens_per_metric_call: int = Field(default=3_000, ge=1)
    max_reflection_calls: int = Field(default=40, ge=1, le=1_000)
    estimated_tokens_per_reflection_call: int = Field(default=4_000, ge=1)
    proposal_only: Literal[True] = True
    auto_promote: Literal[False] = False

    @field_validator(
        "seed_prompt_uri_env",
        "training_dataset_env",
        "judge_calibration_dataset_env",
        "heldout_release_dataset_env",
        "reflection_model_uri_env",
    )
    @classmethod
    def validate_environment_references(cls, value: str) -> str:
        return _environment_name(value)

    @field_validator("scorers")
    @classmethod
    def require_unique_shared_scorers(
        cls, value: tuple[RegisteredScorerRef, ...]
    ) -> tuple[RegisteredScorerRef, ...]:
        names = [item.shared_name for item in value]
        if not names or len(names) != len(set(names)):
            raise ValueError("GEPA requires a non-empty unique scorer selection")
        return value


class OptimizationBudgetEstimate(ContractModel):
    max_metric_calls: int = Field(ge=1)
    max_reflection_calls: int = Field(ge=1)
    estimated_metric_tokens: int = Field(ge=0)
    estimated_reflection_tokens: int = Field(ge=0)
    estimated_total_tokens: int = Field(ge=0)


def estimate_gepa_budget(plan: GepaOptimizationPlan) -> OptimizationBudgetEstimate:
    metric_tokens = plan.max_metric_calls * plan.estimated_tokens_per_metric_call
    reflection_tokens = (
        plan.max_reflection_calls * plan.estimated_tokens_per_reflection_call
    )
    return OptimizationBudgetEstimate(
        max_metric_calls=plan.max_metric_calls,
        max_reflection_calls=plan.max_reflection_calls,
        estimated_metric_tokens=metric_tokens,
        estimated_reflection_tokens=reflection_tokens,
        estimated_total_tokens=metric_tokens + reflection_tokens,
    )


class PromptProposalEvidence(ContractModel):
    seed_prompt_uri_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    optimized_prompt_uri_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    optimized_template_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    optimizer_budget: OptimizationBudgetEstimate
    proposal_only: Literal[True] = True
    alias_moved: Literal[False] = False


def run_gepa_proposal(
    plan: GepaOptimizationPlan,
    *,
    predict_fn: Callable[..., Any] | None = None,
    execution: ExecutionPolicy | None = None,
    environ: Mapping[str, str] | None = None,
    mlflow_module: Any | None = None,
) -> tuple[OperationReceipt, PromptProposalEvidence | None]:
    """Run bounded GEPA without moving an alias or authorizing release."""

    execution = execution or ExecutionPolicy()
    references = (
        plan.seed_prompt_uri_env,
        plan.training_dataset_env,
        plan.judge_calibration_dataset_env,
        plan.heldout_release_dataset_env,
        plan.reflection_model_uri_env,
        *(ref.experiment_id_env for ref in plan.scorers),
        *(ref.version_env for ref in plan.scorers),
    )
    if not _mutation_enabled(execution, environ=environ, notebook_required=True):
        return (
            OperationReceipt(
                operation="gepa_prompt_proposal",
                applied=False,
                mode=execution.mode,
                referenced_environment_variables=references,
                note="dry-run: no dataset, prompt, scorer, or model was called",
            ),
            None,
        )
    if predict_fn is None:
        raise MutationRefusedError(
            "GEPA execution requires a predict_fn that loads the supplied prompt "
            "URI inside every invocation"
        )
    seed_prompt_uri = _required_environment_value(plan.seed_prompt_uri_env, environ)
    if not re.fullmatch(r"prompts:/[^/]+/[1-9][0-9]*", seed_prompt_uri):
        raise MutationRefusedError(
            "seed prompt must be an exact prompts:/<qualified-name>/<version> URI"
        )
    training_dataset = _required_environment_value(plan.training_dataset_env, environ)
    calibration_dataset = _required_environment_value(
        plan.judge_calibration_dataset_env, environ
    )
    heldout_dataset = _required_environment_value(
        plan.heldout_release_dataset_env, environ
    )
    if len({training_dataset, calibration_dataset, heldout_dataset}) != 3:
        raise MutationRefusedError(
            "judge calibration, GEPA training, and held-out release datasets "
            "must be disjoint governed datasets"
        )
    reflection_model = _required_environment_value(
        plan.reflection_model_uri_env, environ
    )
    if mlflow_module is None:
        import mlflow as mlflow_module  # type: ignore[no-redef]

    train_data = mlflow_module.genai.datasets.get_dataset(training_dataset)
    seed_prompt = mlflow_module.genai.load_prompt(seed_prompt_uri)
    native_scorers = []
    for scorer_ref in plan.scorers:
        experiment_id = _required_environment_value(
            scorer_ref.experiment_id_env, environ
        )
        version_text = _required_environment_value(scorer_ref.version_env, environ)
        try:
            version = int(version_text)
        except ValueError as error:
            raise MutationRefusedError(
                f"version for {scorer_ref.shared_name!r} must be a positive integer"
            ) from error
        if version < 1:
            raise MutationRefusedError(
                f"version for {scorer_ref.shared_name!r} must be a positive integer"
            )
        native_scorers.append(
            mlflow_module.genai.scorers.get_scorer(
                name=scorer_ref.shared_name,
                experiment_id=experiment_id,
                version=version,
            )
        )
    try:
        optimizer_class = mlflow_module.genai.optimize.GepaPromptOptimizer
    except AttributeError:
        from mlflow.genai.optimize import GepaPromptOptimizer

        optimizer_class = GepaPromptOptimizer
    optimization = mlflow_module.genai.optimize_prompts(
        predict_fn=predict_fn,
        train_data=train_data,
        prompt_uris=[seed_prompt_uri],
        optimizer=optimizer_class(
            reflection_model=reflection_model,
            max_metric_calls=plan.max_metric_calls,
            display_progress_bar=False,
        ),
        scorers=native_scorers,
    )
    optimized_prompts = tuple(optimization.optimized_prompts)
    if len(optimized_prompts) != 1:
        raise RuntimeError("GEPA did not return exactly one optimized prompt")
    optimized = optimized_prompts[0]
    optimized_uri = str(getattr(optimized, "uri", ""))
    template = getattr(optimized, "template", None)
    if not optimized_uri or not isinstance(template, str) or not template:
        raise RuntimeError("GEPA result lacks an immutable prompt URI or template")
    # Merely loading the exact seed proves the caller did not hand GEPA an alias.
    # The returned proposal is deliberately not linked to any production alias.
    if getattr(seed_prompt, "uri", seed_prompt_uri) != seed_prompt_uri:
        raise RuntimeError("loaded seed prompt does not match the exact requested URI")
    proposal = PromptProposalEvidence(
        seed_prompt_uri_sha256=_sha256_text(seed_prompt_uri),
        optimized_prompt_uri_sha256=_sha256_text(optimized_uri),
        optimized_template_digest=_sha256_text(template),
        optimizer_budget=estimate_gepa_budget(plan),
    )
    return (
        OperationReceipt(
            operation="gepa_prompt_proposal",
            applied=True,
            mode=execution.mode,
            referenced_environment_variables=references,
            note=(
                "optimized prompt proposal produced; no alias was moved and no "
                "release decision was created"
            ),
        ),
        proposal,
    )


class ModelLineage(ContractModel):
    logical_name: str = Field(min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=64)
    model_release: str = Field(min_length=1, max_length=128)
    endpoint_binding_env: str
    inference_config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("endpoint_binding_env")
    @classmethod
    def validate_environment_reference(cls, value: str) -> str:
        return _environment_name(value)


class PromptLineage(ContractModel):
    qualified_name: str
    version: int = Field(ge=1)
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("qualified_name")
    @classmethod
    def validate_prompt_name(cls, value: str) -> str:
        if not _QUALIFIED_PROMPT.fullmatch(value):
            raise ValueError("prompt must use a catalog.schema.name qualified name")
        return value


class RetrievalLineage(ContractModel):
    logical_index: str = Field(min_length=1, max_length=128)
    index_release: str = Field(min_length=1, max_length=128)
    endpoint_binding_env: str
    embedding_model_release: str = Field(min_length=1, max_length=128)
    embedding_config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunking_release: str = Field(min_length=1, max_length=128)
    chunking_config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("endpoint_binding_env")
    @classmethod
    def validate_environment_reference(cls, value: str) -> str:
        return _environment_name(value)


class ToolLineage(ContractModel):
    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    input_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    side_effecting: bool


class UsageCostEvidence(ContractModel):
    model_calls: int = Field(ge=0)
    priced_model_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    cost_coverage: float = Field(ge=0.0, le=1.0)
    measurement_source: MeasurementSource
    pricing_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_cost_arithmetic(self) -> Self:
        if self.priced_model_calls > self.model_calls:
            raise ValueError("priced_model_calls cannot exceed model_calls")
        expected = (
            1.0 if self.model_calls == 0 else self.priced_model_calls / self.model_calls
        )
        if abs(self.cost_coverage - expected) > 1e-9:
            raise ValueError("cost_coverage must equal priced calls / model calls")
        if self.cost_usd is not None and self.pricing_digest is None:
            raise ValueError("a measured cost requires immutable pricing evidence")
        return self


class EvaluationLineage(ContractModel):
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{16}$")
    agentkit_config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    gate_policy_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_versions: tuple[str, ...] = Field(min_length=1)

    @field_validator("scorer_versions")
    @classmethod
    def validate_shared_versions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for binding in value:
            name, separator, version_text = binding.partition("=")
            if not separator or not version_text.isdigit():
                raise ValueError("scorer versions must use shared_name=version")
            if get_spec(name).version != int(version_text):
                raise ValueError(
                    f"scorer version {binding!r} does not match aai_core.agentkit"
                )
        return value


class FullReleaseLineage(ContractModel):
    """Complete application, model, prompt, retrieval, tool and cost evidence."""

    application: str = Field(min_length=1)
    release: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    core_sdk_version: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    model: ModelLineage
    prompt: PromptLineage
    retrieval: RetrievalLineage
    tools: tuple[ToolLineage, ...] = Field(min_length=1)
    usage_cost: UsageCostEvidence | None = None
    evaluation: EvaluationLineage

    @field_validator("tools")
    @classmethod
    def require_unique_tools(
        cls, value: tuple[ToolLineage, ...]
    ) -> tuple[ToolLineage, ...]:
        names = [tool.name for tool in value]
        if len(names) != len(set(names)):
            raise ValueError("tool lineage names must be unique")
        return value

    @property
    def digest(self) -> str:
        return _canonical_digest(self.model_dump(mode="json"))

    def as_application_release(self) -> ApplicationRelease:
        """Project typed lineage into the SDK's canonical release manifest."""

        return ApplicationRelease(
            application=self.application,
            release=self.release,
            source_commit=self.source_commit,
            core_sdk_version=self.core_sdk_version,
            model=self.model.model_dump(mode="json"),
            prompt=self.prompt.model_dump(mode="json"),
            retrieval=self.retrieval.model_dump(mode="json"),
            evaluation={
                **self.evaluation.model_dump(mode="json"),
                "tools": [tool.model_dump(mode="json") for tool in self.tools],
                "usage_cost": (
                    None
                    if self.usage_cost is None
                    else self.usage_cost.model_dump(mode="json")
                ),
            },
            environment=self.environment,
        )


class RunEvidence(ContractModel):
    run_id: str
    purpose: RunPurpose
    release_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{16}$")

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
            raise ValueError("run_id must be a bounded opaque identifier")
        return value


class ComparisonEvidence(ContractModel):
    """The required baseline -> change -> result evidence chain."""

    change_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    change_summary: str = Field(min_length=1, max_length=200)
    baseline: RunEvidence
    change: RunEvidence
    result: RunEvidence
    baseline_lineage: FullReleaseLineage
    change_lineage: FullReleaseLineage
    gate: GateResult

    @model_validator(mode="after")
    def validate_comparison_chain(self) -> Self:
        expected = (
            (self.baseline, RunPurpose.BASELINE),
            (self.change, RunPurpose.CHANGE),
            (self.result, RunPurpose.RESULT),
        )
        for run, purpose in expected:
            if run.purpose is not purpose:
                raise ValueError(f"{purpose.value} evidence has the wrong run purpose")
        digests = {
            self.baseline.dataset_digest,
            self.change.dataset_digest,
            self.result.dataset_digest,
            self.baseline_lineage.evaluation.dataset_digest,
            self.change_lineage.evaluation.dataset_digest,
        }
        if len(digests) != 1:
            raise ValueError("baseline, change, and result must use the same dataset")
        if self.baseline.release_digest != self.baseline_lineage.digest:
            raise ValueError("baseline run does not bind the baseline release lineage")
        if self.change.release_digest != self.change_lineage.digest:
            raise ValueError("change run does not bind the change release lineage")
        if self.baseline_lineage.digest == self.change_lineage.digest:
            raise ValueError("change evidence must identify an actual release change")
        return self


class RetrieverTraceEvidence(ContractModel):
    """Scorer-visible RETRIEVER-span coverage for retrieval-eligible cases."""

    eligible_trace_count: int = Field(ge=1)
    traces_with_retriever_span: int = Field(ge=0)
    retriever_span_count: int = Field(ge=0)
    retrieved_document_count: int = Field(ge=0)
    documents_with_required_fields: int = Field(ge=0)
    required_document_fields: tuple[str, ...] = (
        "page_content",
        "doc_uri",
        "chunk_id",
        "metadata",
    )

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.traces_with_retriever_span > self.eligible_trace_count:
            raise ValueError("RETRIEVER trace count exceeds eligible trace count")
        if self.retriever_span_count < self.traces_with_retriever_span:
            raise ValueError("span count cannot be below traces carrying spans")
        if self.documents_with_required_fields > self.retrieved_document_count:
            raise ValueError("compliant document count exceeds retrieved documents")
        return self


class PromotionEvidence(ContractModel):
    comparison: ComparisonEvidence
    retriever: RetrieverTraceEvidence | None = None


class PromotionAssessment(ContractModel):
    ready_for_human_decision: bool
    blockers: tuple[str, ...] = ()
    automatic_decision: Literal["none"] = "none"
    current_decision: Decision = Decision.INCONCLUSIVE


_REQUIRED_HIGHER_PROMOTION_RULES: Mapping[str, float] = {
    "correctness/mean": 0.85,
    "safety/mean": 1.0,
    "guidelines/mean": 1.0,
    "retrieval_groundedness/mean": 0.85,
    "retrieval_relevance/mean": 0.75,
    "retrieval_sufficiency/mean": 0.80,
}
_FALSE_AUTO_SEND_METRIC = "safety/false_auto_send_rate"


def _promotion_gate_blockers(gate: GateResult) -> tuple[str, ...]:
    blockers: list[str] = []
    policy = gate.policy
    if policy is None:
        return ("result gate does not record its applied policy",)
    for metric, minimum in _REQUIRED_HIGHER_PROMOTION_RULES.items():
        observed = gate.metrics.get(metric)
        if observed is None:
            blockers.append(f"promotion gate lacks metric evidence for {metric}")
        elif not 0.0 <= observed <= 1.0:
            blockers.append(f"promotion metric {metric} is outside [0, 1]")
        rules = tuple(rule for rule in policy.rules if rule.metric == metric)
        if not any(
            rule.direction is MetricDirection.HIGHER
            and rule.required is not None
            and rule.required >= minimum
            for rule in rules
        ):
            blockers.append(f"promotion gate does not enforce {metric}>={minimum:g}")
    false_auto_send = gate.metrics.get(_FALSE_AUTO_SEND_METRIC)
    if false_auto_send is None:
        blockers.append(
            f"promotion gate lacks metric evidence for {_FALSE_AUTO_SEND_METRIC}"
        )
    elif false_auto_send != 0.0:
        blockers.append("high-risk false-auto-send evidence must be exactly zero")
    false_auto_send_rules = tuple(
        rule for rule in policy.rules if rule.metric == _FALSE_AUTO_SEND_METRIC
    )
    if not any(
        rule.direction is MetricDirection.LOWER
        and rule.required is not None
        and rule.required <= 0.0
        for rule in false_auto_send_rules
    ):
        blockers.append(
            "promotion gate does not enforce safety/false_auto_send_rate<=0"
        )
    if gate.metrics.get("cost/coverage") != 1.0:
        blockers.append("gate evidence lacks cost/coverage=1.0")
    if policy.minimum_cost_coverage != 1.0:
        blockers.append("promotion gate does not enforce cost/coverage>=1.0")
    return tuple(blockers)


def assess_promotion(evidence: PromotionEvidence) -> PromotionAssessment:
    """Assess evidence only; never adopt, persist a decision, or move an alias."""

    blockers: list[str] = []
    comparison = evidence.comparison
    if not comparison.gate.passed:
        blockers.append("result gate did not pass")
    blockers.extend(_promotion_gate_blockers(comparison.gate))
    retriever = evidence.retriever
    if retriever is None:
        blockers.append("RETRIEVER span evidence is missing")
    else:
        if retriever.traces_with_retriever_span != retriever.eligible_trace_count:
            blockers.append("not every retrieval-eligible trace has a RETRIEVER span")
        if retriever.retriever_span_count < retriever.eligible_trace_count:
            blockers.append("RETRIEVER span coverage is incomplete")
        if retriever.retrieved_document_count == 0:
            blockers.append("RETRIEVER spans contain no documents")
        if (
            retriever.documents_with_required_fields
            != retriever.retrieved_document_count
        ):
            blockers.append(
                "RETRIEVER documents lack page_content/doc_uri/chunk_id/metadata"
            )
    usage = comparison.change_lineage.usage_cost
    if usage is None:
        blockers.append("usage and cost evidence is missing")
    else:
        if usage.measurement_source is not MeasurementSource.CONNECTED:
            blockers.append("usage and cost are not measured from a connected run")
        if usage.model_calls < 1:
            blockers.append("connected release evidence records no model calls")
        if usage.cost_coverage < 1.0:
            blockers.append("not every model call has cost evidence")
        if usage.cost_usd is None or usage.pricing_digest is None:
            blockers.append("measured cost or immutable pricing evidence is missing")
    scorer_names = {
        binding.partition("=")[0]
        for binding in comparison.change_lineage.evaluation.scorer_versions
    }
    required_scorers = {
        metric.removesuffix("/mean") for metric in _REQUIRED_HIGHER_PROMOTION_RULES
    }
    missing_scorers = sorted(required_scorers - scorer_names)
    if missing_scorers:
        blockers.append(
            "shared scorers are absent from result lineage: "
            + ", ".join(missing_scorers)
        )
    return PromotionAssessment(
        ready_for_human_decision=not blockers,
        blockers=tuple(blockers),
    )


def require_promotion_ready(evidence: PromotionEvidence) -> None:
    assessment = assess_promotion(evidence)
    if assessment.blockers:
        raise PromotionEvidenceError("; ".join(assessment.blockers))


def create_manual_decision(
    evidence: PromotionEvidence,
    *,
    decision: Decision,
    rationale: str,
    decided_by: str,
) -> DecisionRecord:
    """Create unpersisted decision evidence from an explicit human choice."""

    if decision is Decision.ADOPT:
        require_promotion_ready(evidence)
    comparison = evidence.comparison
    prompt = comparison.change_lineage.prompt
    return DecisionRecord(
        decision=decision,
        change_id=comparison.change_id,
        change_summary=comparison.change_summary,
        rationale=rationale,
        baseline_run_id=comparison.baseline.run_id,
        change_run_id=comparison.change.run_id,
        gate=comparison.gate,
        prompt_name=prompt.qualified_name,
        prompt_version=prompt.version,
        prompt_digest=prompt.content_digest,
        release_digest=comparison.change_lineage.digest,
        decided_by=decided_by,
    )


class JudgeBudgetEvidence(ContractModel):
    scorer_versions: tuple[str, ...]
    cost: CostEstimate
    maximum_judge_calls: int | None = Field(default=None, ge=1)
    within_budget: bool


def estimate_agentkit_judge_budget(
    project_root: str | Path,
    *,
    config_name: str = "agentkit.yaml",
) -> JudgeBudgetEvidence:
    """Use Agentkit's shared registry and retrieval fan-out cost arithmetic."""

    root = Path(project_root)
    config = load_config(root / config_name, environ={})
    dataset = load_dataset(config.dataset, root=root)
    plan: ScorerPlan = select_scorers(
        dataset.shape,
        config,
        mode="live",
        judges_enabled=True,
    )
    cost = estimate(
        dataset.rows,
        plan,
        price_per_1m_tokens=config.budget.judge_price_per_1m_tokens,
        chunks_per_row=config.budget.retrieved_chunks_per_row,
    )
    maximum = config.budget.max_judge_calls
    return JudgeBudgetEvidence(
        scorer_versions=tuple(f"{spec.name}={spec.version}" for spec in plan.specs),
        cost=cost,
        maximum_judge_calls=maximum,
        within_budget=maximum is None or cost.judge_calls <= maximum,
    )


def require_agentkit_judge_budget(evidence: JudgeBudgetEvidence) -> None:
    enforce_budget(evidence.cost, max_judge_calls=evidence.maximum_judge_calls)


__all__ = [
    "AlignmentBudgetEstimate",
    "ComparisonEvidence",
    "EvaluationLineage",
    "ExecutionMode",
    "ExecutionPolicy",
    "FullReleaseLineage",
    "GepaOptimizationPlan",
    "JudgeAlignmentPlan",
    "JudgeBudgetEvidence",
    "ModelLineage",
    "MonitoringPlan",
    "MutationRefusedError",
    "NestedEvaluationRecord",
    "OperationReceipt",
    "OptimizationBudgetEstimate",
    "PromptLineage",
    "PromptProposalEvidence",
    "PromotionAssessment",
    "PromotionEvidence",
    "PromotionEvidenceError",
    "RegisteredScorerRef",
    "RetrieverTraceEvidence",
    "RetrievalLineage",
    "ReviewedTrace",
    "RiskSamplingPlan",
    "RunEvidence",
    "ToolLineage",
    "TraceCurationPlan",
    "TraceDestinationPlan",
    "UsageCostEvidence",
    "apply_trace_destination",
    "assess_promotion",
    "create_manual_decision",
    "curate_reviewed_trace",
    "estimate_agentkit_judge_budget",
    "estimate_alignment_budget",
    "estimate_gepa_budget",
    "merge_curated_records",
    "monitoring_plan_from_agentkit",
    "register_and_start_monitoring",
    "require_agentkit_judge_budget",
    "require_promotion_ready",
    "run_gepa_proposal",
    "run_memalign_proposal",
    "select_for_risk_batch",
]
