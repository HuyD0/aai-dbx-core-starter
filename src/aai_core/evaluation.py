"""Small release-gate contracts over native MLflow GenAI evaluation results."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from math import isfinite
from numbers import Real
from typing import Any, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.exceptions import AaiCoreError


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
