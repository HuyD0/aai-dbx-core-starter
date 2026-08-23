"""Deterministic uncertainty evidence for agent evaluation metrics.

MLflow remains the scorer and aggregate-metric authority.  This module adds
the evidence a release decision needs around those aggregates: how many rows
contributed, an approximate confidence interval, and (when the same dataset was
scored against a recorded baseline) a paired improvement interval.  No prompt,
response, or expectation content is persisted here -- only nullable numeric
scores in dataset order.

Two interval methods are supported.  ``normal`` (the default) uses the normal
approximation to the sampling distribution of the mean.  ``bootstrap``
resamples the recorded scores with replacement and reads percentile bounds off
the resampled means, which keeps bounds inside the score's feasible range for
the bounded, skewed scales judge verdicts and pass rates actually follow.
Bootstrap draws come from a seeded generator, so the same scores and
configuration always reproduce the same interval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from random import Random
from statistics import NormalDist, fmean, stdev
from typing import Any, Literal, Self, cast

from pydantic import Field, field_serializer, field_validator, model_validator

from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.evaluation import MetricDirection, MetricRule

_STATISTICS_SEGMENT = "/statistics/"


class IntervalMethod(StrEnum):
    """Platform-owned vocabulary for how confidence bounds are computed."""

    NORMAL = "normal"
    BOOTSTRAP = "bootstrap"


MeanIntervalMethod = Literal["normal-mean-v1", "bootstrap-percentile-v1"]
PairedIntervalMethod = Literal[
    "paired-normal-mean-v1", "paired-bootstrap-percentile-v1"
]


class StatisticsConfig(ContractModel):
    """Project-owned policy for reporting or enforcing uncertainty.

    Reporting is on by default.  Enforcement is opt-in because existing small
    evaluation suites must first grow to the configured minimum sample size.
    ``minimum_effect`` names metrics whose *paired lower confidence bound* must
    show at least that much practical improvement over the baseline.

    ``method`` selects how the bounds are computed.  The normal approximation
    is cheap and adequate for large, roughly symmetric samples; ``bootstrap``
    fits the bounded, skewed distributions judge scores and pass/fail rates
    follow, at the cost of ``bootstrap_resamples`` extra mean computations per
    metric.  Bootstrap bounds are deterministic for a given ``bootstrap_seed``;
    re-running with another seed is a cheap check that a promotion decision
    does not hinge on resampling noise.
    """

    enabled: bool = True
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    minimum_cases: int = Field(default=30, ge=2, le=1_000_000)
    enforce_confidence: bool = False
    minimum_effect: Mapping[str, float] = Field(default_factory=dict)
    method: IntervalMethod = IntervalMethod.NORMAL
    bootstrap_resamples: int = Field(default=1000, ge=100, le=10_000)
    bootstrap_seed: int = Field(default=0, ge=0)

    @field_validator("method", mode="before")
    @classmethod
    def parse_method(cls, value: Any) -> Any:
        if isinstance(value, IntervalMethod):
            return value
        if not isinstance(value, str):
            raise TypeError("statistics.method must be a string or IntervalMethod")
        return IntervalMethod(value.strip().lower())

    @field_validator("minimum_effect", mode="before")
    @classmethod
    def normalize_minimum_effect(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized: dict[str, float] = {}
        for raw_name, raw_effect in value.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("statistics.minimum_effect keys must name metrics")
            if isinstance(raw_effect, bool) or not isinstance(raw_effect, int | float):
                raise ValueError(f"statistics.minimum_effect.{name} must be numeric")
            effect = float(raw_effect)
            if not math.isfinite(effect) or effect < 0:
                raise ValueError(
                    f"statistics.minimum_effect.{name} must be finite and non-negative"
                )
            normalized[name] = effect
        return normalized

    @field_validator("minimum_effect", mode="after")
    @classmethod
    def freeze_minimum_effect(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return cast(Mapping[str, float], freeze_value(value))

    @field_serializer("minimum_effect")
    def serialize_minimum_effect(self, value: Mapping[str, float]) -> dict[str, float]:
        return cast(dict[str, float], thaw_value(value))


class MeanConfidence(ContractModel):
    metric: str = Field(min_length=1)
    sample_size: int = Field(ge=1)
    mean: float
    lower: float
    upper: float
    confidence_level: float = Field(gt=0.5, lt=1.0)
    method: MeanIntervalMethod = "normal-mean-v1"

    @field_validator("mean", "lower", "upper")
    @classmethod
    def finite_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence evidence values must be finite")
        return value


class PairedConfidence(ContractModel):
    metric: str = Field(min_length=1)
    pair_count: int = Field(ge=1)
    mean_improvement: float
    lower_improvement: float
    upper_improvement: float
    standardized_effect: float | None = None
    confidence_level: float = Field(gt=0.5, lt=1.0)
    method: PairedIntervalMethod = "paired-normal-mean-v1"

    @field_validator(
        "mean_improvement",
        "lower_improvement",
        "upper_improvement",
        "standardized_effect",
    )
    @classmethod
    def finite_values(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("paired confidence evidence values must be finite")
        return value


class StatisticalEvidence(ContractModel):
    confidence_level: float = Field(gt=0.5, lt=1.0)
    minimum_cases: int = Field(ge=2)
    enforced: bool
    # Records written before the bootstrap option existed carry no method
    # and deserialize as the normal approximation they were computed with.
    method: IntervalMethod = IntervalMethod.NORMAL
    bootstrap_resamples: int | None = Field(default=None, ge=100, le=10_000)
    bootstrap_seed: int | None = Field(default=None, ge=0)
    estimates: tuple[MeanConfidence, ...] = ()
    paired: tuple[PairedConfidence, ...] = ()

    @field_validator("method", mode="before")
    @classmethod
    def parse_method(cls, value: Any) -> Any:
        if isinstance(value, IntervalMethod):
            return value
        if not isinstance(value, str):
            raise TypeError("method must be a string or IntervalMethod")
        return IntervalMethod(value.strip().lower())

    @field_validator("estimates", "paired", mode="before")
    @classmethod
    def coerce_sequences(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def bootstrap_parameters_match_method(self) -> Self:
        # The resample count and seed are what make a bootstrap interval
        # reproducible, so evidence claiming that method must carry them --
        # and evidence computed analytically must not imply otherwise.
        recorded = (
            self.bootstrap_resamples is not None or self.bootstrap_seed is not None
        )
        if self.method is IntervalMethod.BOOTSTRAP:
            if self.bootstrap_resamples is None or self.bootstrap_seed is None:
                raise ValueError(
                    "bootstrap evidence must record bootstrap_resamples and "
                    "bootstrap_seed"
                )
        elif recorded:
            raise ValueError(
                "bootstrap_resamples and bootstrap_seed require method 'bootstrap'"
            )
        return self

    @property
    def method_label(self) -> str:
        """Human label for reports: which interval family produced the bounds."""

        return (
            "bootstrap-percentile"
            if self.method is IntervalMethod.BOOTSTRAP
            else "normal-mean"
        )


def statistics_metric(metric: str, component: str) -> str:
    """Return the stable synthetic metric name used by the gate policy."""

    return f"{metric}{_STATISTICS_SEGMENT}{component}"


def is_statistics_metric(metric: str) -> bool:
    return _STATISTICS_SEGMENT in metric


def extend_rules_with_statistics(
    rules: Sequence[MetricRule],
    config: StatisticsConfig,
    *,
    allow_missing_regression_baseline: bool,
) -> tuple[MetricRule, ...]:
    """Add confidence-bound rules without changing the original thresholds."""

    base = tuple(rule for rule in rules if not is_statistics_metric(rule.metric))
    if not config.enabled:
        return base
    minimum_effect = dict(config.minimum_effect)
    augmented: dict[str, MetricRule] = {rule.metric: rule for rule in base}
    for rule in base:
        enforce_for_metric = config.enforce_confidence or rule.metric in minimum_effect
        if not enforce_for_metric:
            continue
        count_name = statistics_metric(rule.metric, "sample_count")
        augmented[count_name] = MetricRule(
            metric=count_name,
            direction=MetricDirection.HIGHER,
            required=float(config.minimum_cases),
        )
        if config.enforce_confidence and rule.required is not None:
            component = (
                "confidence_lower"
                if rule.direction is MetricDirection.HIGHER
                else "confidence_upper"
            )
            bound_name = statistics_metric(rule.metric, component)
            augmented[bound_name] = MetricRule(
                metric=bound_name,
                direction=rule.direction,
                required=rule.required,
            )
        if (
            config.enforce_confidence
            and rule.max_regression is not None
            and not allow_missing_regression_baseline
        ):
            regression_name = statistics_metric(rule.metric, "paired_improvement_lower")
            augmented[regression_name] = MetricRule(
                metric=regression_name,
                direction=MetricDirection.HIGHER,
                required=-rule.max_regression,
            )
        if rule.metric in minimum_effect and not allow_missing_regression_baseline:
            effect_name = statistics_metric(rule.metric, "paired_improvement_lower")
            augmented[effect_name] = MetricRule(
                metric=effect_name,
                direction=MetricDirection.HIGHER,
                required=minimum_effect[rule.metric],
            )
    return tuple(augmented[name] for name in sorted(augmented))


def build_statistical_evidence(
    samples: Mapping[str, Sequence[float | None]],
    baseline_samples: Mapping[str, Sequence[float | None]],
    rules: Sequence[MetricRule],
    config: StatisticsConfig,
) -> tuple[StatisticalEvidence | None, dict[str, float]]:
    """Build persisted evidence and the synthetic metrics consumed by gates."""

    if not config.enabled:
        return None, {}
    directions = {
        rule.metric: rule.direction
        for rule in rules
        if not is_statistics_metric(rule.metric)
    }
    bootstrap = config.method is IntervalMethod.BOOTSTRAP
    mean_method: MeanIntervalMethod = (
        "bootstrap-percentile-v1" if bootstrap else "normal-mean-v1"
    )
    paired_method: PairedIntervalMethod = (
        "paired-bootstrap-percentile-v1" if bootstrap else "paired-normal-mean-v1"
    )
    estimates: list[MeanConfidence] = []
    paired: list[PairedConfidence] = []
    metrics: dict[str, float] = {}
    for metric in sorted(samples):
        values = tuple(value for value in samples[metric] if value is not None)
        if not values:
            continue
        mean, lower, upper = _interval(values, config, purpose="mean", metric=metric)
        estimate = MeanConfidence(
            metric=metric,
            sample_size=len(values),
            mean=mean,
            lower=lower,
            upper=upper,
            confidence_level=config.confidence_level,
            method=mean_method,
        )
        estimates.append(estimate)
        metrics[statistics_metric(metric, "sample_count")] = float(len(values))
        metrics[statistics_metric(metric, "confidence_lower")] = lower
        metrics[statistics_metric(metric, "confidence_upper")] = upper

        direction = directions.get(metric)
        reference = baseline_samples.get(metric)
        if direction is None or reference is None:
            continue
        improvements = _paired_improvements(samples[metric], reference, direction)
        if not improvements:
            continue
        paired_mean, paired_lower, paired_upper = _interval(
            improvements, config, purpose="paired", metric=metric
        )
        dispersion = stdev(improvements) if len(improvements) > 1 else 0.0
        standardized = paired_mean / dispersion if dispersion > 0 else None
        comparison = PairedConfidence(
            metric=metric,
            pair_count=len(improvements),
            mean_improvement=paired_mean,
            lower_improvement=paired_lower,
            upper_improvement=paired_upper,
            standardized_effect=standardized,
            confidence_level=config.confidence_level,
            method=paired_method,
        )
        paired.append(comparison)
        metrics[statistics_metric(metric, "paired_count")] = float(len(improvements))
        metrics[statistics_metric(metric, "paired_improvement_mean")] = paired_mean
        metrics[statistics_metric(metric, "paired_improvement_lower")] = paired_lower
        metrics[statistics_metric(metric, "paired_improvement_upper")] = paired_upper
        if standardized is not None:
            metrics[statistics_metric(metric, "standardized_effect")] = standardized

    return (
        StatisticalEvidence(
            confidence_level=config.confidence_level,
            minimum_cases=config.minimum_cases,
            enforced=config.enforce_confidence or bool(config.minimum_effect),
            method=config.method,
            bootstrap_resamples=config.bootstrap_resamples if bootstrap else None,
            bootstrap_seed=config.bootstrap_seed if bootstrap else None,
            estimates=tuple(estimates),
            paired=tuple(paired),
        ),
        metrics,
    )


def _interval(
    values: Sequence[float],
    config: StatisticsConfig,
    *,
    purpose: str,
    metric: str,
) -> tuple[float, float, float]:
    if config.method is IntervalMethod.BOOTSTRAP:
        return _bootstrap_interval(
            values,
            config.confidence_level,
            resamples=config.bootstrap_resamples,
            rng=_bootstrap_rng(config.bootstrap_seed, purpose, metric),
        )
    return _mean_interval(values, config.confidence_level)


def _mean_interval(
    values: Sequence[float], confidence_level: float
) -> tuple[float, float, float]:
    mean = fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    standard_error = stdev(values) / math.sqrt(len(values))
    if standard_error == 0:
        return mean, mean, mean
    critical = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    margin = critical * standard_error
    return mean, mean - margin, mean + margin


def _bootstrap_rng(seed: int, purpose: str, metric: str) -> Random:
    # String seeding hashes all of its bits (`random.seed(..., version=2)`),
    # so every (seed, purpose, metric) tuple is an independent stream and
    # adding one metric never shifts another metric's draws.
    return Random(f"{seed}:{purpose}:{metric}")


def _bootstrap_interval(
    values: Sequence[float],
    confidence_level: float,
    *,
    resamples: int,
    rng: Random,
) -> tuple[float, float, float]:
    # The point estimate stays the observed mean: the interval qualifies the
    # aggregate MLflow reported, it never replaces it.
    mean = fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    population = list(values)
    size = len(population)
    resampled = sorted(fmean(rng.choices(population, k=size)) for _ in range(resamples))
    tail = (1.0 - confidence_level) / 2.0
    return mean, _quantile(resampled, tail), _quantile(resampled, 1.0 - tail)


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence."""

    position = fraction * (len(ordered) - 1)
    below = math.floor(position)
    above = math.ceil(position)
    if below == above:
        return ordered[below]
    weight = position - below
    return ordered[below] * (1.0 - weight) + ordered[above] * weight


def _paired_improvements(
    current: Sequence[float | None],
    baseline: Sequence[float | None],
    direction: MetricDirection,
) -> tuple[float, ...]:
    improvements: list[float] = []
    for observed, reference in zip(current, baseline, strict=False):
        if observed is None or reference is None:
            continue
        improvement = (
            observed - reference
            if direction is MetricDirection.HIGHER
            else reference - observed
        )
        if math.isfinite(improvement):
            improvements.append(improvement)
    return tuple(improvements)
