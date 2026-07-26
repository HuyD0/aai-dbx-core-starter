from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_core.evaluation import (
    EvaluationGateError,
    GatePolicy,
    MetricDirection,
    MetricRule,
    apply_gate,
)


def test_gate_accepts_native_mlflow_result_and_absolute_rules():
    native_result = SimpleNamespace(
        metrics={"groundedness/mean": 0.92, "latency_ms/mean": 240}
    )
    policy = GatePolicy(
        rules=(
            MetricRule(
                metric="groundedness/mean",
                direction=MetricDirection.HIGHER,
                required=0.8,
            ),
            MetricRule(
                metric="latency_ms/mean",
                direction=MetricDirection.LOWER,
                required=500,
            ),
        )
    )

    result = apply_gate(native_result, policy=policy)

    assert result.passed
    assert result.metrics["groundedness/mean"] == 0.92


def test_gate_reports_absolute_and_missing_metric_failures():
    policy = GatePolicy(
        rules=(
            MetricRule(
                metric="groundedness/mean",
                direction=MetricDirection.HIGHER,
                required=0.8,
            ),
            MetricRule(
                metric="safety/mean",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
        )
    )

    result = apply_gate({"groundedness/mean": 0.7}, policy=policy)

    assert [failure.metric for failure in result.failures] == [
        "groundedness/mean",
        "safety/mean",
    ]
    with pytest.raises(EvaluationGateError, match="groundedness/mean"):
        result.require_passed()


def test_regression_requires_baseline_unless_bootstrap_is_explicit():
    rule = MetricRule(
        metric="quality/mean",
        direction=MetricDirection.HIGHER,
        max_regression=0.02,
    )

    missing = apply_gate(
        {"quality/mean": 0.95},
        policy=GatePolicy(rules=(rule,)),
    )
    bootstrap = apply_gate(
        {"quality/mean": 0.95},
        policy=GatePolicy(
            rules=(rule,),
            allow_missing_regression_baseline=True,
        ),
    )

    assert not missing.passed
    assert missing.failures[0].reason == "regression baseline is missing"
    assert bootstrap.passed


@pytest.mark.parametrize(
    ("direction", "observed", "baseline"),
    [
        (MetricDirection.HIGHER, 0.94, 0.99),
        (MetricDirection.LOWER, 120.0, 80.0),
    ],
)
def test_gate_detects_regression_in_both_directions(direction, observed, baseline):
    policy = GatePolicy(
        rules=(
            MetricRule(
                metric="score",
                direction=direction,
                max_regression=0.02,
            ),
        )
    )

    result = apply_gate(
        {"score": observed},
        policy=policy,
        baseline_metrics={"score": baseline},
    )

    assert not result.passed
    assert "regressed" in result.failures[0].reason


def test_unknown_or_incomplete_cost_evidence_never_becomes_zero():
    policy = GatePolicy(minimum_cost_coverage=1.0)

    unknown = apply_gate({}, policy=policy)
    incomplete = apply_gate({"cost/coverage": 0.5}, policy=policy)
    complete_zero = apply_gate(
        {"cost/coverage": 1.0, "cost/total_usd": 0.0},
        policy=policy,
    )

    assert not unknown.passed
    assert unknown.failures[0].reason == "cost coverage is unknown"
    assert not incomplete.passed
    assert complete_zero.passed


def test_scorer_error_metrics_fail_without_exposing_error_text():
    result = apply_gate(
        {
            "correctness/mean": 1.0,
            "correctness/error_count": 2,
        },
        policy=GatePolicy(),
    )

    assert not result.passed
    assert result.failures[0].reason == "2 scorer invocation(s) failed"


def test_non_numeric_and_non_finite_native_metrics_are_not_gate_evidence():
    result = apply_gate(
        {
            "quality": float("nan"),
            "label": "good",
            "flag": True,
            "latency": 42,
        },
        policy=GatePolicy(
            rules=(
                MetricRule(
                    metric="quality",
                    direction=MetricDirection.HIGHER,
                    required=0.8,
                ),
            )
        ),
    )

    assert dict(result.metrics) == {"latency": 42.0}
    assert result.failures[0].reason == "metric is missing"


def test_gate_contracts_are_strict_frozen_and_serializable():
    with pytest.raises(ValidationError):
        GatePolicy(minimum_cost_coverage="1.0")
    with pytest.raises(ValidationError):
        GatePolicy(unrecognized=True)
    with pytest.raises(ValidationError):
        MetricRule(metric="quality", direction=MetricDirection.HIGHER)

    policy = GatePolicy(
        rules=(
            MetricRule(
                metric="quality",
                direction=MetricDirection.HIGHER,
                required=0.8,
            ),
        )
    )
    result = apply_gate({"quality": 0.9}, policy=policy)

    with pytest.raises(ValidationError):
        policy.minimum_cost_coverage = 0.5
    assert result.model_dump(mode="json") == {
        "metrics": {"quality": 0.9},
        "failures": [],
    }


def test_gate_rejects_values_without_a_native_metrics_mapping():
    with pytest.raises(TypeError, match="metrics mapping"):
        apply_gate(object(), policy=GatePolicy())
