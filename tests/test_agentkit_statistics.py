import pytest
from pydantic import ValidationError

from aai_core.agentkit.statistics import (
    StatisticsConfig,
    build_statistical_evidence,
    extend_rules_with_statistics,
    statistics_metric,
)
from aai_core.evaluation import MetricDirection, MetricRule


def _rule(
    metric="quality/mean",
    direction=MetricDirection.HIGHER,
    *,
    required=0.8,
    max_regression=0.05,
):
    return MetricRule(
        metric=metric,
        direction=direction,
        required=required,
        max_regression=max_regression,
    )


def test_statistics_report_mean_and_paired_direction_without_content():
    evidence, metrics = build_statistical_evidence(
        {"quality/mean": (0.8, 0.9, None, 1.0)},
        {"quality/mean": (0.7, 0.85, 0.4, 0.9)},
        (_rule(),),
        StatisticsConfig(),
    )

    assert evidence is not None
    assert evidence.estimates[0].sample_size == 3
    assert evidence.estimates[0].mean == pytest.approx(0.9)
    assert evidence.paired[0].pair_count == 3
    assert evidence.paired[0].mean_improvement == pytest.approx(1 / 12)
    assert metrics[statistics_metric("quality/mean", "sample_count")] == 3
    assert (
        metrics[statistics_metric("quality/mean", "paired_improvement_lower")]
        <= evidence.paired[0].mean_improvement
        <= metrics[statistics_metric("quality/mean", "paired_improvement_upper")]
    )


def test_lower_is_better_metrics_normalize_improvement_to_positive():
    evidence, _ = build_statistical_evidence(
        {"latency/mean": (0.8, 0.9, 1.0)},
        {"latency/mean": (1.0, 1.0, 1.0)},
        (
            _rule(
                "latency/mean",
                MetricDirection.LOWER,
                required=1.2,
                max_regression=0.1,
            ),
        ),
        StatisticsConfig(),
    )

    assert evidence is not None
    assert evidence.paired[0].mean_improvement == pytest.approx(0.1)


def test_confidence_enforcement_uses_bounds_sample_size_and_paired_delta():
    config = StatisticsConfig(
        minimum_cases=30,
        enforce_confidence=True,
        minimum_effect={"quality/mean": 0.02},
    )
    rules = extend_rules_with_statistics(
        (_rule(),), config, allow_missing_regression_baseline=False
    )
    by_metric = {rule.metric: rule for rule in rules}

    assert by_metric[statistics_metric("quality/mean", "sample_count")].required == 30
    assert (
        by_metric[statistics_metric("quality/mean", "confidence_lower")].required == 0.8
    )
    # The explicit practical-effect requirement is stronger than merely
    # showing that regression is within the 0.05 budget.
    assert (
        by_metric[
            statistics_metric("quality/mean", "paired_improvement_lower")
        ].required
        == 0.02
    )


def test_baseline_establishment_does_not_require_a_nonexistent_pair():
    rules = extend_rules_with_statistics(
        (_rule(),),
        StatisticsConfig(enforce_confidence=True),
        allow_missing_regression_baseline=True,
    )
    names = {rule.metric for rule in rules}

    assert statistics_metric("quality/mean", "confidence_lower") in names
    assert statistics_metric("quality/mean", "paired_improvement_lower") not in names


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_minimum_effect_must_be_finite_and_non_negative(value):
    with pytest.raises(ValidationError):
        StatisticsConfig(minimum_effect={"quality/mean": value})
