"""Evaluation stays native to MLflow; aai-core only applies deterministic policy."""

from types import SimpleNamespace

from aai_core.evaluation import (
    GatePolicy,
    GateResult,
    MetricDirection,
    MetricRule,
    apply_gate,
    evaluate_with_gate,
    log_gate_evidence,
)


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


def test_evaluate_with_gate_returns_the_native_result_by_identity():
    native_result = SimpleNamespace(
        metrics={"correctness/mean": 0.9},
        result_df=object(),
        run_id="run-123",
    )
    recorded_options: dict = {}

    def fake_evaluate(**options):
        recorded_options.update(options)
        return native_result

    fake_mlflow = SimpleNamespace(genai=SimpleNamespace(evaluate=fake_evaluate))
    data = object()
    scorers = [object()]

    result, gate = evaluate_with_gate(
        policy=GatePolicy(
            rules=(
                MetricRule(
                    metric="correctness/mean",
                    direction=MetricDirection.HIGHER,
                    required=0.8,
                ),
            )
        ),
        mlflow_module=fake_mlflow,
        data=data,
        scorers=scorers,
        model_id="models:/app/1",
    )

    assert result is native_result
    assert recorded_options == {
        "data": data,
        "scorers": scorers,
        "model_id": "models:/app/1",
    }
    assert recorded_options["data"] is data
    assert gate.passed


def test_gate_evidence_persists_metrics_and_a_lowercase_tag():
    logged: dict = {"metrics": {}, "tags": {}}
    fake_mlflow = SimpleNamespace(
        log_metrics=lambda metrics: logged["metrics"].update(metrics),
        set_tags=lambda tags: logged["tags"].update(tags),
    )
    gate = GateResult(metrics={"quality/mean": 0.95})

    tags = log_gate_evidence(gate, mlflow_module=fake_mlflow)

    assert logged["metrics"] == {"quality/mean": 0.95}
    assert logged["tags"] == {"aai.gate_passed": "true"}
    assert tags == {"aai.gate_passed": "true"}
