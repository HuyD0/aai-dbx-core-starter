from types import SimpleNamespace

import pytest

from aai_core.evaluation import (
    EvaluationGateError,
    EvaluationSuite,
    QualityThreshold,
    judge_model_uri,
)
from aai_core.providers.types import ProviderConfigurationError


class FakeGenAI:
    def evaluate(self, **kwargs):
        assert kwargs["data"] == [{"input": "hello"}]
        return {"groundedness": 0.9, "latency_ms": 250}


class FakeMlflow:
    def __init__(self):
        self.genai = FakeGenAI()
        self.feedback = None

    def log_feedback(self, **options):
        self.feedback = options


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


def test_gated_scorer_row_errors_fail_instead_of_false_green():
    class Frame:
        columns = [
            "quality/value",
            "quality/error_message",
            "report_only/error_message",
        ]

        def __len__(self):
            return 2

        def __getitem__(self, name):
            return {
                "quality/value": [1, None],
                "quality/error_message": [None, "private provider failure"],
                "report_only/error_message": [None, "optional judge failure"],
            }[name]

    raw = SimpleNamespace(
        metrics={"quality/mean": 1.0, "report_only/mean": 1.0},
        result_df=Frame(),
    )
    mlflow = FakeMlflow()
    mlflow.genai.evaluate = lambda **kwargs: raw
    suite = EvaluationSuite(
        scorers=[],
        thresholds=[QualityThreshold("quality/mean", direction="higher", required=0.8)],
        mlflow_module=mlflow,
    )

    report = suite.run(data=[{"input": "hello"}])

    assert not report.passed
    assert [failure.metric for failure in report.failures] == ["quality/mean"]
    assert "1 evaluation row(s) failed scoring" in report.failures[0].reason
    assert "private provider failure" not in report.failures[0].reason


def test_judge_model_uri_requires_an_approved_databricks_endpoint():
    settings = SimpleNamespace(
        models={
            "judge-model": {
                "provider": "databricks",
                "deployment": "approved-judge",
            }
        }
    )

    assert judge_model_uri(settings) == "endpoints:/approved-judge"

    with pytest.raises(ProviderConfigurationError, match="must use provider"):
        judge_model_uri(
            SimpleNamespace(
                models={
                    "judge-model": {
                        "provider": "foundry",
                        "deployment": "not-direct",
                    }
                }
            )
        )


def test_log_feedback_passes_only_supplied_native_options():
    mlflow = FakeMlflow()
    suite = EvaluationSuite(scorers=[], thresholds=[], mlflow_module=mlflow)

    suite.log_feedback(
        trace_id="trace-1",
        name="domain_quality",
        value=True,
    )

    assert mlflow.feedback == {
        "trace_id": "trace-1",
        "name": "domain_quality",
        "value": True,
    }


def test_log_feedback_passes_alignment_context():
    mlflow = FakeMlflow()
    suite = EvaluationSuite(scorers=[], thresholds=[], mlflow_module=mlflow)
    source = object()

    suite.log_feedback(
        trace_id="trace-1",
        name="domain_quality",
        value=False,
        rationale="Missing required evidence.",
        source=source,
        metadata={"review_queue": "domain-experts"},
        span_id="span-2",
    )

    assert mlflow.feedback == {
        "trace_id": "trace-1",
        "name": "domain_quality",
        "value": False,
        "rationale": "Missing required evidence.",
        "source": source,
        "metadata": {"review_queue": "domain-experts"},
        "span_id": "span-2",
    }
