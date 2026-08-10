"""Small release-gate contracts over native MLflow GenAI evaluation results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from math import isfinite
from numbers import Real
from typing import Any, Self, cast

from pydantic import Field, field_serializer, field_validator, model_validator

from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.exceptions import AaiCoreError

__all__ = [
    "EvaluationGateError",
    "GateFailure",
    "GatePolicy",
    "GateResult",
    "MetricDirection",
    "MetricRule",
    "apply_gate",
]


class MetricDirection(StrEnum):
    """Whether a governed metric improves by increasing or decreasing."""

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

    @property
    def digest(self) -> str:
        """Canonical identifier for the exact deterministic gate policy."""

        return _canonical_digest(self.model_dump(mode="json"))


class GateFailure(ContractModel):
    """One deterministic reason an evaluation gate did not pass."""

    metric: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class GateResult(ContractModel):
    """Immutable release-gate evidence; native evaluation results stay native."""

    metrics: Mapping[str, float]
    failures: tuple[GateFailure, ...] = ()
    policy_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator("metrics", mode="after")
    @classmethod
    def freeze_metrics(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return cast(Mapping[str, float], freeze_value(value))

    @field_serializer("metrics")
    def serialize_metrics(self, value: Mapping[str, float]) -> dict[str, float]:
        return cast(dict[str, float], thaw_value(value))

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
    """Raised when callers require a passing evaluation gate."""

    code = "aai_core.evaluation.gate_failed"


def apply_gate(
    evaluation_result: Any,
    *,
    policy: GatePolicy,
    baseline_metrics: Mapping[str, float] | None = None,
) -> GateResult:
    """Apply deterministic policy to a native MLflow result or metric mapping."""

    metrics = _extract_metrics(evaluation_result)
    baseline = (
        _extract_metrics(baseline_metrics) if baseline_metrics is not None else {}
    )
    baseline_digest = (
        _canonical_digest(baseline) if baseline_metrics is not None else None
    )
    failures = _scorer_failures(metrics, policy)
    failures.extend(_cost_coverage_failures(metrics, policy))
    for rule in policy.rules:
        failures.extend(_metric_rule_failures(metrics, baseline, policy, rule))

    return GateResult(
        metrics=metrics,
        failures=tuple(failures),
        policy_digest=policy.digest,
        baseline_digest=baseline_digest,
    )


def _scorer_failures(
    metrics: Mapping[str, float],
    policy: GatePolicy,
) -> list[GateFailure]:
    if not policy.fail_on_scorer_errors:
        return []
    return [
        GateFailure(
            metric=metric,
            reason=f"{value:g} scorer invocation(s) failed",
        )
        for metric, value in metrics.items()
        if metric.endswith(policy.scorer_error_metric_suffix) and value > 0
    ]


def _cost_coverage_failures(
    metrics: Mapping[str, float],
    policy: GatePolicy,
) -> list[GateFailure]:
    required = policy.minimum_cost_coverage
    if required is None:
        return []
    observed = metrics.get(policy.cost_coverage_metric)
    if observed is None:
        reason = "cost coverage is unknown"
    elif observed < required:
        reason = f"{observed:g} is below required {required:g}"
    else:
        return []
    return [GateFailure(metric=policy.cost_coverage_metric, reason=reason)]


def _metric_rule_failures(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    policy: GatePolicy,
    rule: MetricRule,
) -> list[GateFailure]:
    observed = metrics.get(rule.metric)
    if observed is None:
        return [GateFailure(metric=rule.metric, reason="metric is missing")]
    failures = _required_metric_failures(observed, rule)
    failures.extend(_regression_failures(observed, baseline, policy, rule))
    return failures


def _required_metric_failures(
    observed: float,
    rule: MetricRule,
) -> list[GateFailure]:
    required = rule.required
    if required is None:
        return []
    below = rule.direction is MetricDirection.HIGHER and observed < required
    above = rule.direction is MetricDirection.LOWER and observed > required
    if not (below or above):
        return []
    comparison = "below" if below else "above"
    return [
        GateFailure(
            metric=rule.metric,
            reason=f"{observed:g} is {comparison} required {required:g}",
        )
    ]


def _regression_failures(
    observed: float,
    baseline: Mapping[str, float],
    policy: GatePolicy,
    rule: MetricRule,
) -> list[GateFailure]:
    if rule.max_regression is None:
        return []
    reference = baseline.get(rule.metric)
    if reference is None:
        if policy.allow_missing_regression_baseline:
            return []
        return [
            GateFailure(metric=rule.metric, reason="regression baseline is missing")
        ]
    regression = (
        reference - observed
        if rule.direction is MetricDirection.HIGHER
        else observed - reference
    )
    if regression <= rule.max_regression:
        return []
    return [
        GateFailure(
            metric=rule.metric,
            reason=(
                f"regressed by {regression:g} from baseline {reference:g}; "
                f"maximum allowed is {rule.max_regression:g}"
            ),
        )
    ]


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


def _canonical_digest(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
