"""Evaluation stays native to MLflow; aai-core only applies deterministic policy."""

from types import SimpleNamespace

from aai_core.evaluation import GatePolicy, MetricDirection, MetricRule, apply_gate


def test_native_evaluation_object_is_not_wrapped_or_mutated():
    native_result = SimpleNamespace(
        metrics={"correctness/mean": 0.9},
        result_df=object(),
        run_id="run-123",
    )

    gate = apply_gate(
        native_result,
        policy=GatePolicy(
            rules=(
                MetricRule(
                    metric="correctness/mean",
                    direction=MetricDirection.HIGHER,
                    required=0.8,
                ),
            )
        ),
    )

    assert gate.passed
    assert native_result.run_id == "run-123"
    assert native_result.result_df is not None


def test_gate_result_contains_only_immutable_release_evidence():
    native_result = SimpleNamespace(
        metrics={
            "quality/mean": 0.95,
            "debug/label": "not-a-metric",
        }
    )

    gate = apply_gate(native_result, policy=GatePolicy())

    assert gate.model_dump(mode="json") == {
        "metrics": {"quality/mean": 0.95},
        "failures": [],
    }
