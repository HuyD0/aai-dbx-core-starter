import pytest

from aai_core.evaluation import (
    EvaluationGateError,
    EvaluationSuite,
    QualityThreshold,
)


class FakeGenAI:
    def evaluate(self, **kwargs):
        assert kwargs["data"] == [{"input": "hello"}]
        return {"groundedness": 0.9, "latency_ms": 250}


class FakeMlflow:
    def __init__(self):
        self.genai = FakeGenAI()


def test_evaluation_gate_passes_required_and_baseline_checks():
    suite = EvaluationSuite(
        scorers=[],
        thresholds=[
            QualityThreshold(
                "groundedness",
                direction="higher",
                required=0.8,
                max_regression=0.05,
            ),
            QualityThreshold("latency_ms", direction="lower", required=500),
        ],
        mlflow_module=FakeMlflow(),
    )

    report = suite.run(
        data=[{"input": "hello"}],
        baseline_metrics={"groundedness": 0.92},
    )

    assert report.passed


def test_evaluation_gate_reports_regression():
    suite = EvaluationSuite(
        scorers=[],
        thresholds=[
            QualityThreshold(
                "groundedness",
                direction="higher",
                max_regression=0.01,
            )
        ],
        mlflow_module=FakeMlflow(),
    )

    report = suite.run(
        data=[{"input": "hello"}],
        baseline_metrics={"groundedness": 0.95},
    )

    assert not report.passed
    with pytest.raises(EvaluationGateError):
        report.require_passed()
