import pytest
from pydantic import ValidationError

from aai_core.agentkit.statistics import (
    IntervalMethod,
    StatisticalEvidence,
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


def _routing_accuracy_samples():
    # A supervisor that routed 29 of 30 synthetic cases correctly: the
    # bounded, near-boundary success-rate shape bootstrap exists for.
    return {"subagent_routing_accuracy/mean": (1.0,) * 29 + (0.0,)}


def test_bootstrap_bounds_stay_inside_the_score_range_where_normal_escapes():
    samples = _routing_accuracy_samples()
    rule = _rule("subagent_routing_accuracy/mean", required=0.9, max_regression=0.05)

    normal_evidence, _ = build_statistical_evidence(
        samples, {}, (rule,), StatisticsConfig()
    )
    bootstrap_evidence, _ = build_statistical_evidence(
        samples, {}, (rule,), StatisticsConfig(method="bootstrap")
    )

    assert normal_evidence is not None and bootstrap_evidence is not None
    # The normal approximation promises accuracy above 100% here; the
    # percentile bootstrap cannot leave the observed 0..1 range.
    assert normal_evidence.estimates[0].upper > 1.0
    assert bootstrap_evidence.estimates[0].lower >= 0.0
    assert bootstrap_evidence.estimates[0].upper <= 1.0
    assert bootstrap_evidence.estimates[0].mean == pytest.approx(29 / 30)
    assert (
        bootstrap_evidence.estimates[0].lower
        <= bootstrap_evidence.estimates[0].mean
        <= bootstrap_evidence.estimates[0].upper
    )


def test_bootstrap_intervals_are_deterministic_and_seed_sensitive():
    samples = {"quality/mean": tuple(value / 10 for value in range(1, 11)) * 3}
    rule = _rule()

    first, _ = build_statistical_evidence(
        samples, {}, (rule,), StatisticsConfig(method="bootstrap")
    )
    second, _ = build_statistical_evidence(
        samples, {}, (rule,), StatisticsConfig(method="bootstrap")
    )
    reseeded, _ = build_statistical_evidence(
        samples, {}, (rule,), StatisticsConfig(method="bootstrap", bootstrap_seed=7)
    )

    assert first is not None and second is not None and reseeded is not None
    assert first.estimates[0] == second.estimates[0]
    assert (reseeded.estimates[0].lower, reseeded.estimates[0].upper) != (
        first.estimates[0].lower,
        first.estimates[0].upper,
    )


def test_bootstrap_streams_are_independent_per_metric():
    quality = {"quality/mean": (0.2, 0.4, 0.5, 0.7, 0.9, 1.0, 0.3, 0.6)}
    config = StatisticsConfig(method="bootstrap")

    alone, _ = build_statistical_evidence(quality, {}, (_rule(),), config)
    with_neighbor, _ = build_statistical_evidence(
        {**quality, "latency/mean": (1.0, 2.0, 3.0)},
        {},
        (_rule(), _rule("latency/mean", MetricDirection.LOWER, required=5.0)),
        config,
    )

    assert alone is not None and with_neighbor is not None
    by_metric = {estimate.metric: estimate for estimate in with_neighbor.estimates}
    assert by_metric["quality/mean"] == alone.estimates[0]


def test_bootstrap_paired_improvement_keeps_direction_normalization():
    evidence, metrics = build_statistical_evidence(
        {"latency/mean": (0.8, 0.9, 1.0, 0.7, 0.85, 0.95)},
        {"latency/mean": (1.0, 1.0, 1.1, 0.9, 1.0, 1.05)},
        (
            _rule(
                "latency/mean",
                MetricDirection.LOWER,
                required=1.2,
                max_regression=0.1,
            ),
        ),
        StatisticsConfig(method="bootstrap"),
    )

    assert evidence is not None
    assert evidence.paired[0].method == "paired-bootstrap-percentile-v1"
    assert evidence.paired[0].mean_improvement > 0
    assert (
        evidence.paired[0].lower_improvement
        <= evidence.paired[0].mean_improvement
        <= evidence.paired[0].upper_improvement
    )
    assert (
        metrics[statistics_metric("latency/mean", "paired_improvement_lower")]
        == evidence.paired[0].lower_improvement
    )


def test_interval_method_only_changes_bounds_never_the_gate_surface():
    samples = {"quality/mean": (0.8, 0.9, 1.0, 0.7)}
    baseline = {"quality/mean": (0.7, 0.85, 0.9, 0.6)}
    rule = _rule()

    _, normal_metrics = build_statistical_evidence(
        samples, baseline, (rule,), StatisticsConfig()
    )
    bootstrap_evidence, bootstrap_metrics = build_statistical_evidence(
        samples, baseline, (rule,), StatisticsConfig(method="bootstrap")
    )

    assert set(normal_metrics) == set(bootstrap_metrics)
    assert bootstrap_evidence is not None
    assert bootstrap_evidence.method is IntervalMethod.BOOTSTRAP
    assert bootstrap_evidence.bootstrap_resamples == 1000
    assert bootstrap_evidence.bootstrap_seed == 0
    assert bootstrap_evidence.estimates[0].method == "bootstrap-percentile-v1"


def test_records_from_before_the_bootstrap_option_deserialize_as_normal():
    evidence = StatisticalEvidence.model_validate(
        {"confidence_level": 0.95, "minimum_cases": 30, "enforced": False}
    )

    assert evidence.method is IntervalMethod.NORMAL
    assert evidence.bootstrap_resamples is None
    assert evidence.bootstrap_seed is None


def test_bootstrap_evidence_must_carry_its_reproduction_parameters():
    with pytest.raises(ValidationError):
        StatisticalEvidence(
            confidence_level=0.95,
            minimum_cases=30,
            enforced=False,
            method="bootstrap",
        )
    with pytest.raises(ValidationError):
        StatisticalEvidence(
            confidence_level=0.95,
            minimum_cases=30,
            enforced=False,
            bootstrap_seed=3,
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("method", "jackknife"),
        ("bootstrap_resamples", 50),
        ("bootstrap_resamples", 20_000),
        ("bootstrap_seed", -1),
    ],
)
def test_bootstrap_configuration_is_validated(field, value):
    with pytest.raises(ValidationError):
        StatisticsConfig(**{field: value})
