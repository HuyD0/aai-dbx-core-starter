"""Small release-gate contracts and thin helpers over native MLflow GenAI
evaluation results.

The contracts (:class:`GatePolicy`, :class:`GateResult`) apply deterministic
policy to a native result without wrapping or mutating it. The helpers stay
equally thin: they resolve the approved judge endpoint, compose the native
``mlflow.genai.evaluate()`` call with the gate, persist gate evidence on the
active run, and manage governed evaluation datasets. None of them owns an
MLflow run or mirrors native parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from math import isfinite
from numbers import Real
from typing import Any, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.exceptions import AaiCoreError
from aai_core.providers.types import ProviderConfigurationError


class MetricDirection(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


class MetricRule(ContractModel):
    """One absolute and/or regression rule for a native MLflow metric."""

    metric: str = Field(min_length=1)
    direction: MetricDirection
    required: float | None = None
    max_regression: float | None = Field(default=None, ge=0.0)

    @field_validator("direction", mode="before")
    @classmethod
    def parse_direction(cls, value: Any) -> MetricDirection:
        if isinstance(value, MetricDirection):
            return value
        if not isinstance(value, str):
            raise TypeError("direction must be a string or MetricDirection")
        return MetricDirection(value.strip().lower())

    @field_validator("required", "max_regression")
    @classmethod
    def require_finite_number(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("metric rule values must be finite")
        return value

    @model_validator(mode="after")
    def require_a_constraint(self) -> Self:
        if self.required is None and self.max_regression is None:
            raise ValueError("A metric rule requires required or max_regression")
        return self


class GatePolicy(ContractModel):
    """Persistable policy applied after ``mlflow.genai.evaluate()``."""

    rules: tuple[MetricRule, ...] = ()
    minimum_cost_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    cost_coverage_metric: str = Field(default="cost/coverage", min_length=1)
    fail_on_scorer_errors: bool = True
    scorer_error_metric_suffix: str = Field(default="/error_count", min_length=1)
    allow_missing_regression_baseline: bool = False


class GateFailure(ContractModel):
    metric: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class GateResult(ContractModel):
    """Immutable release-gate evidence; native evaluation results stay native."""

    metrics: Mapping[str, float]
    failures: tuple[GateFailure, ...] = ()

    @field_validator("metrics", mode="after")
    @classmethod
    def freeze_metrics(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return freeze_value(value)

    @field_serializer("metrics")
    def serialize_metrics(self, value: Mapping[str, float]) -> dict[str, float]:
        return thaw_value(value)

    @property
    def passed(self) -> bool:
        return not self.failures

    def require_passed(self) -> None:
        if not self.failures:
            return
        raise EvaluationGateError(
            "; ".join(
                f"{failure.metric}: {failure.reason}" for failure in self.failures
            )
        )


class EvaluationGateError(AaiCoreError):
    code = "aai_core.evaluation.gate_failed"


def apply_gate(
    evaluation_result: Any,
    *,
    policy: GatePolicy,
    baseline_metrics: Mapping[str, float] | None = None,
) -> GateResult:
    """Apply deterministic policy to a native MLflow result or metric mapping."""

    metrics = _extract_metrics(evaluation_result)
    baseline = dict(baseline_metrics or {})
    failures: list[GateFailure] = []

    if policy.fail_on_scorer_errors:
        for metric, value in metrics.items():
            if metric.endswith(policy.scorer_error_metric_suffix) and value > 0:
                failures.append(
                    GateFailure(
                        metric=metric,
                        reason=f"{value:g} scorer invocation(s) failed",
                    )
                )

    if policy.minimum_cost_coverage is not None:
        observed = metrics.get(policy.cost_coverage_metric)
        if observed is None:
            failures.append(
                GateFailure(
                    metric=policy.cost_coverage_metric,
                    reason="cost coverage is unknown",
                )
            )
        elif observed < policy.minimum_cost_coverage:
            failures.append(
                GateFailure(
                    metric=policy.cost_coverage_metric,
                    reason=(
                        f"{observed:g} is below required "
                        f"{policy.minimum_cost_coverage:g}"
                    ),
                )
            )

    for rule in policy.rules:
        observed = metrics.get(rule.metric)
        if observed is None:
            failures.append(GateFailure(metric=rule.metric, reason="metric is missing"))
            continue
        if rule.required is not None:
            below = (
                rule.direction is MetricDirection.HIGHER and observed < rule.required
            )
            above = rule.direction is MetricDirection.LOWER and observed > rule.required
            if below or above:
                comparison = "below" if below else "above"
                failures.append(
                    GateFailure(
                        metric=rule.metric,
                        reason=(
                            f"{observed:g} is {comparison} required "
                            f"{rule.required:g}"
                        ),
                    )
                )
        if rule.max_regression is None:
            continue
        reference = baseline.get(rule.metric)
        if reference is None:
            if not policy.allow_missing_regression_baseline:
                failures.append(
                    GateFailure(
                        metric=rule.metric,
                        reason="regression baseline is missing",
                    )
                )
            continue
        regression = (
            reference - observed
            if rule.direction is MetricDirection.HIGHER
            else observed - reference
        )
        if regression > rule.max_regression:
            failures.append(
                GateFailure(
                    metric=rule.metric,
                    reason=(
                        f"regressed by {regression:g} from baseline {reference:g}; "
                        f"maximum allowed is {rule.max_regression:g}"
                    ),
                )
            )

    return GateResult(metrics=metrics, failures=tuple(failures))


def _extract_metrics(result: Any) -> dict[str, float]:
    source = result if isinstance(result, Mapping) else getattr(result, "metrics", None)
    if not isinstance(source, Mapping):
        raise TypeError(
            "evaluation_result must be a metric mapping or expose a metrics mapping"
        )
    metrics: dict[str, float] = {}
    for name, value in source.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        numeric = float(value)
        if isfinite(numeric):
            metrics[str(name)] = numeric
    return metrics


def judge_model_uri(settings: Any, logical_name: str = "judge-model") -> str:
    """Resolve an approved logical judge into MLflow's model URI.

    Judges run through a Databricks serving endpoint so authentication,
    gateway policy, and cost controls stay platform-owned. A Foundry model
    must first be exposed through a governed Databricks external-model
    endpoint.
    """

    models = getattr(settings, "models", {})
    config = models.get(logical_name) if isinstance(models, Mapping) else None
    if not isinstance(config, Mapping):
        raise ProviderConfigurationError(
            f"aai-platform.yml has no {logical_name!r} model entry",
            remediation=f"Add providers.models.{logical_name} with the "
            "gateway-fronted Databricks serving endpoint approved for judges.",
        )
    if config.get("provider") != "databricks":
        raise ProviderConfigurationError(
            f"LLM judge {logical_name!r} must use provider 'databricks'",
            remediation="Route the judge through a Databricks serving "
            "endpoint; for Foundry models, use an external-model endpoint.",
        )
    deployment = config.get("deployment")
    if not isinstance(deployment, str) or not deployment.strip():
        raise ProviderConfigurationError(
            f"LLM judge {logical_name!r} has no deployment",
            remediation=f"Set providers.models.{logical_name}.deployment to "
            "the approved serving endpoint name.",
        )
    return f"endpoints:/{deployment.strip()}"


def log_gate_evidence(
    gate: GateResult,
    *,
    mlflow_module: Any | None = None,
) -> dict[str, str]:
    """Persist gate metrics and the ``aai.gate_passed`` tag on the active run.

    Call inside a governed run. Returns the tags it set.
    """

    mlflow = _mlflow(mlflow_module)
    if gate.metrics:
        mlflow.log_metrics(dict(gate.metrics))
    tags = {"aai.gate_passed": str(gate.passed).lower()}
    mlflow.set_tags(tags)
    return tags


def evaluate_with_gate(
    *,
    policy: GatePolicy,
    baseline_metrics: Mapping[str, float] | None = None,
    mlflow_module: Any | None = None,
    **evaluate_options: Any,
) -> tuple[Any, GateResult]:
    """Run native ``mlflow.genai.evaluate()`` and apply the gate policy.

    Every keyword in ``evaluate_options`` passes through to the native call
    untouched, and the native result is returned by identity, so new MLflow
    evaluation arguments never require an SDK change. Run governance stays
    with :class:`~aai_core.experiments.ExperimentManager`; persisting the
    evidence stays the explicit :func:`log_gate_evidence` call.
    """

    mlflow = _mlflow(mlflow_module)
    native_result = mlflow.genai.evaluate(**evaluate_options)
    gate = apply_gate(native_result, policy=policy, baseline_metrics=baseline_metrics)
    return native_result, gate


def get_or_create_evaluation_dataset(
    *,
    name: str,
    catalog: str,
    schema: str,
    experiment_id: str,
    records: Sequence[Mapping[str, Any]] | None = None,
    mlflow_module: Any | None = None,
) -> Any:
    """Return the governed evaluation dataset, creating and merging as needed.

    ``name`` is a logical dataset name; the catalog and schema qualify it.
    The native dataset object is returned unchanged.
    """

    logical_name = name.strip()
    if not logical_name or "." in logical_name:
        raise ValueError(
            "name must be a non-blank logical name without catalog or schema"
        )
    if not str(experiment_id).strip():
        raise ValueError("experiment_id must not be blank")

    mlflow = _mlflow(mlflow_module)
    qualified_name = f"{catalog}.{schema}.{logical_name}"
    try:
        dataset = mlflow.genai.datasets.get_dataset(name=qualified_name)
    except Exception as exc:
        if not _is_missing_dataset(exc):
            raise
        # Databricks-managed EvaluationDatasets reject MLflow dataset tags.
        # Governed context belongs on runs and UC securables instead.
        dataset = mlflow.genai.datasets.create_dataset(
            name=qualified_name,
            experiment_id=experiment_id,
        )

    experiment_ids = {
        str(associated) for associated in (dataset.experiment_ids or [])
    }
    if str(experiment_id) not in experiment_ids:
        raise RuntimeError(
            f"Unity Catalog dataset {qualified_name!r} is not associated with "
            f"MLflow experiment {experiment_id!r}. Databricks does not "
            "support adding experiment associations through this API; use a "
            "new approved dataset name or ask the platform owner to repair it."
        )
    if records:
        dataset.merge_records(list(records))
    return dataset


def _is_missing_dataset(error: Exception) -> bool:
    error_code = str(getattr(error, "error_code", "")).upper()
    message = str(error).upper()
    return error_code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"} or any(
        marker in message
        for marker in ("NOT_FOUND", "RESOURCE_DOES_NOT_EXIST", "DOES NOT EXIST")
    )


def _mlflow(module: Any | None) -> Any:
    if module is not None:
        return module
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "Evaluation support requires the `genai` extra. From an aai-core "
            "checkout run `make examples-install` and use `.venv/bin/python`; "
            "in a consuming environment install `aai-core[genai]`."
        ) from error
    return mlflow
