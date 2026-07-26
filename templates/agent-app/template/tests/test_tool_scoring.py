"""Exact tool-trajectory scorer compatibility tests."""

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from app.tool_scoring import exact_tool_call_scorer


class FakeTrace:
    def __init__(self, calls):
        self.calls = calls

    def search_spans(self, *, span_type):
        assert span_type == "TOOL"
        return [
            SimpleNamespace(name=name, inputs=arguments)
            for name, arguments in self.calls
        ]


@pytest.fixture(autouse=True)
def fake_mlflow_scorers(monkeypatch):
    @dataclass
    class Feedback:
        name: str
        value: str | None = None
        rationale: str | None = None
        error: Exception | None = None
        source: object | None = None

    class AssessmentSource:
        def __init__(self, **values):
            self.values = values

    class AssessmentSourceType:
        CODE = "CODE"

    class ToolCallCorrectness:
        def __init__(self, **options):
            self.options = options

        def __call__(self, **kwargs):
            return Feedback(name="tool_call_correctness", value="yes")

    def scorer(*, name):
        def decorate(function):
            function.name = name
            return function

        return decorate

    entities = ModuleType("mlflow.entities")
    entities.AssessmentSource = AssessmentSource
    entities.AssessmentSourceType = AssessmentSourceType
    entities.Feedback = Feedback
    scorers = ModuleType("mlflow.genai.scorers")
    scorers.ToolCallCorrectness = ToolCallCorrectness
    scorers.scorer = scorer
    genai = ModuleType("mlflow.genai")
    genai.scorers = scorers
    mlflow = ModuleType("mlflow")
    mlflow.entities = entities
    mlflow.genai = genai
    for name, module in {
        "mlflow": mlflow,
        "mlflow.entities": entities,
        "mlflow.genai": genai,
        "mlflow.genai.scorers": scorers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_no_tool_expectation_passes_only_without_tool_spans():
    scorer = exact_tool_call_scorer()
    expectations = {"expected_tool_calls": []}

    passed = scorer(trace=FakeTrace([]), expectations=expectations)
    failed = scorer(
        trace=FakeTrace([("lookup_order_status", {"order_id": "A-1001"})]),
        expectations=expectations,
    )

    assert passed.value == "yes"
    assert failed.value == "no"


def test_exact_unordered_comparison_preserves_duplicate_multiplicity():
    scorer = exact_tool_call_scorer()
    a = {"order_id": "A-1001"}
    b = {"order_id": "A-1002"}
    expectations = {
        "expected_tool_calls": [
            {"name": "lookup_order_status", "arguments": a},
            {"name": "lookup_order_status", "arguments": a},
            {"name": "lookup_order_status", "arguments": b},
            {"name": "lookup_order_status", "arguments": b},
        ]
    }

    feedback = scorer(
        trace=FakeTrace(
            [
                ("lookup_order_status", a),
                ("lookup_order_status", a),
                ("lookup_order_status", a),
                ("lookup_order_status", b),
            ]
        ),
        expectations=expectations,
    )

    assert feedback.value == "no"
    assert "multiplicity differed" in feedback.rationale
