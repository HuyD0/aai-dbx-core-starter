"""Exact tool-trajectory scorer compatibility tests."""

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from app.tool_scoring import (
    decision_action_consistency_scorer,
    decision_tool_appropriateness_scorer,
    exact_tool_call_scorer,
    trace_execution_success_scorer,
)


class FakeTrace:
    def __init__(
        self,
        calls,
        *,
        decisions=(),
        errored_call_indexes=(),
        root_errored=False,
        root_span_type="AGENT",
        root_status=None,
        nested_agent_errored=False,
    ):
        self.calls = calls
        self.decisions = decisions
        self.errored_call_indexes = set(errored_call_indexes)
        self.root_errored = root_errored
        self.root_span_type = root_span_type
        self.nested_agent_errored = nested_agent_errored
        self.root_span = SimpleNamespace(
            name="agent.evaluate",
            span_type=root_span_type,
            parent_id=None,
            status=SimpleNamespace(
                status_code=(
                    root_status
                    if root_status is not None
                    else "ERROR" if root_errored else "OK"
                )
            ),
        )
        self.data = SimpleNamespace(spans=[self.root_span])

    def search_spans(self, *, span_type):
        if span_type == "TOOL":
            return [
                SimpleNamespace(
                    name=name,
                    inputs=arguments,
                    status=SimpleNamespace(
                        status_code=(
                            "ERROR" if index in self.errored_call_indexes else "OK"
                        )
                    ),
                )
                for index, (name, arguments) in enumerate(self.calls)
            ]
        if span_type == "AGENT":
            agent_spans = [
                SimpleNamespace(
                    name=f"decision.{decision_type}",
                    attributes={
                        "agent.decision.type": decision_type,
                        "agent.decision.selected_action": action,
                    },
                )
                for decision_type, action in self.decisions
            ]
            if self.root_span_type == "AGENT":
                agent_spans.append(self.root_span)
            if self.nested_agent_errored:
                agent_spans.append(
                    SimpleNamespace(
                        name="agent.nested",
                        parent_id="root-span",
                        status=SimpleNamespace(status_code="ERROR"),
                    )
                )
            return agent_spans
        raise AssertionError(f"unexpected span type {span_type!r}")


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


def test_exact_scorer_preserves_sdk_name_and_input_matching():
    feedback = exact_tool_call_scorer()(
        trace=FakeTrace(
            [("lookup_order_status", {"order_id": "A-1001"})],
        ),
        expectations={
            "expected_tool_calls": [
                {
                    "name": "lookup_order_status",
                    "arguments": {"order_id": "A-1001"},
                }
            ]
        },
    )

    assert feedback.value == "yes"


def test_exact_sdk_match_never_invokes_mlflow_native_or_default_judge(monkeypatch):
    class ForbiddenToolCallCorrectness:
        def __init__(self, **options):
            raise AssertionError(
                f"deterministic exact scoring delegated to MLflow with {options!r}"
            )

    monkeypatch.setattr(
        sys.modules["mlflow.genai.scorers"],
        "ToolCallCorrectness",
        ForbiddenToolCallCorrectness,
    )
    tool_span = SimpleNamespace(
        name="lookup_order_status",
        inputs={"order_id": "A-1001"},
    )

    class SdkTraceWithoutLlmToolSchemas:
        def search_spans(self, *, span_type):
            assert span_type == "TOOL"
            return [tool_span]

    feedback = exact_tool_call_scorer()(
        trace=SdkTraceWithoutLlmToolSchemas(),
        expectations={
            "expected_tool_calls": [
                {
                    "name": "lookup_order_status",
                    "arguments": {"order_id": "A-1001"},
                }
            ]
        },
    )

    assert feedback.value == "yes"
    assert "canonical TOOL spans" in feedback.rationale


@pytest.mark.parametrize(
    "tool_span",
    [
        SimpleNamespace(
            name="execute_tool",
            inputs={
                "call": {
                    "tool_name": "lookup_order_status",
                    "arguments": {"order_id": "A-1001"},
                }
            },
            attributes={},
        ),
        SimpleNamespace(
            name="execute_tool",
            inputs=None,
            attributes={
                "gen_ai.tool.name": "lookup_order_status",
                "gen_ai.tool.call.arguments": '{"order_id":"A-1001"}',
            },
        ),
        SimpleNamespace(
            name="execute_tool",
            inputs=None,
            attributes={
                "gen_ai.tool.name": "lookup_order_status",
                "gen_ai.tool.call.arguments": {"order_id": "A-1001"},
            },
        ),
    ],
)
def test_exact_scorer_canonicalizes_native_and_otel_tool_spans(tool_span):
    trace = SimpleNamespace(
        search_spans=lambda *, span_type: [tool_span] if span_type == "TOOL" else []
    )

    feedback = exact_tool_call_scorer()(
        trace=trace,
        expectations={
            "expected_tool_calls": [
                {
                    "name": "lookup_order_status",
                    "arguments": {"order_id": "A-1001"},
                }
            ]
        },
    )

    assert feedback.value == "yes"
    assert "canonical TOOL spans" in feedback.rationale


@pytest.mark.parametrize(
    "arguments",
    [None, "not-json", "[]"],
)
def test_exact_scorer_reports_missing_or_malformed_otel_arguments(arguments):
    attributes = {"gen_ai.tool.name": "lookup_order_status"}
    if arguments is not None:
        attributes["gen_ai.tool.call.arguments"] = arguments
    tool_span = SimpleNamespace(
        name="execute_tool",
        inputs=None,
        attributes=attributes,
    )
    trace = SimpleNamespace(
        search_spans=lambda *, span_type: [tool_span] if span_type == "TOOL" else []
    )

    feedback = exact_tool_call_scorer()(
        trace=trace,
        expectations={
            "expected_tool_calls": [
                {
                    "name": "lookup_order_status",
                    "arguments": {"order_id": "A-1001"},
                }
            ]
        },
    )

    assert feedback.value is None
    assert isinstance(feedback.error, TypeError | ValueError)


def test_decision_action_consistency_matches_ordered_tool_spans():
    scorer = decision_action_consistency_scorer()
    trace = FakeTrace(
        [
            ("lookup_order_status", {"order_id": "A-1001"}),
            ("lookup_order_status", {"order_id": "A-1002"}),
        ],
        decisions=[
            ("tool_selection", "lookup_order_status"),
            ("tool_selection", "lookup_order_status"),
            ("evidence_sufficiency", "answer"),
        ],
    )

    feedback = scorer(trace=trace)

    assert feedback.value == "yes"
    assert "authoritative TOOL spans" in feedback.rationale


def test_decision_consistency_uses_canonical_otel_tool_name():
    tool_span = SimpleNamespace(
        name="execute_tool",
        inputs=None,
        attributes={
            "gen_ai.tool.name": "lookup_order_status",
            "gen_ai.tool.call.arguments": '{"order_id":"A-1001"}',
        },
        status=SimpleNamespace(status_code="OK"),
    )
    decisions = [
        SimpleNamespace(
            name="decision.tool_selection",
            attributes={
                "agent.decision.type": "tool_selection",
                "agent.decision.selected_action": "lookup_order_status",
            },
        ),
        SimpleNamespace(
            name="decision.evidence_sufficiency",
            attributes={
                "agent.decision.type": "evidence_sufficiency",
                "agent.decision.selected_action": "answer",
            },
        ),
    ]

    class OtelTrace:
        def search_spans(self, *, span_type):
            return [tool_span] if span_type == "TOOL" else decisions

    feedback = decision_action_consistency_scorer()(trace=OtelTrace())

    assert feedback.value == "yes"


def test_decision_action_consistency_fails_for_mismatch_or_missing_decisions():
    scorer = decision_action_consistency_scorer()
    observed = [("lookup_order_status", {"order_id": "A-1001"})]

    mismatched = scorer(
        trace=FakeTrace(
            observed,
            decisions=[("tool_selection", "unobserved_tool")],
        )
    )
    missing = scorer(trace=FakeTrace(observed))

    assert mismatched.value == "no"
    assert "unobserved_tool" in mismatched.rationale
    assert missing.value == "no"
    assert "No decision span" in missing.rationale


def test_decision_action_consistency_requires_terminal_decision():
    scorer = decision_action_consistency_scorer()
    trace = FakeTrace(
        [("lookup_order_status", {"order_id": "A-1001"})],
        decisions=[("tool_selection", "lookup_order_status")],
    )

    feedback = scorer(trace=trace)

    assert feedback.value == "no"
    assert "terminal 'evidence_sufficiency'" in feedback.rationale


@pytest.mark.parametrize("root_status", ["OK", "UNSET"])
def test_failed_tool_requires_convergence_when_root_did_not_fail(root_status):
    scorer = decision_action_consistency_scorer()
    trace = FakeTrace(
        [("lookup_order_status", {"order_id": "A-1001"})],
        decisions=[("tool_selection", "lookup_order_status")],
        errored_call_indexes={0},
        root_status=root_status,
    )

    feedback = scorer(trace=trace)

    assert feedback.value == "no"
    assert "terminal 'evidence_sufficiency'" in feedback.rationale


@pytest.mark.parametrize("root_span_type", ["AGENT", "CHAIN", "CUSTOM"])
def test_failed_root_of_any_span_type_waives_missing_convergence(root_span_type):
    scorer = decision_action_consistency_scorer()
    trace = FakeTrace(
        [("lookup_order_status", {"order_id": "A-1001"})],
        decisions=[("tool_selection", "lookup_order_status")],
        root_errored=True,
        root_span_type=root_span_type,
    )

    feedback = scorer(trace=trace)

    assert feedback.value == "yes"


def test_failed_tool_with_successful_root_and_post_tool_convergence_is_consistent():
    root = SimpleNamespace(
        name="agent.evaluate",
        span_type="CHAIN",
        parent_id=None,
        status=SimpleNamespace(status_code="OK"),
    )
    tool = SimpleNamespace(
        name="lookup_order_status",
        start_time_ns=10,
        end_time_ns=20,
        parent_id="request-root",
        status=SimpleNamespace(status_code="ERROR"),
    )
    decisions = [
        SimpleNamespace(
            name="decision.tool_selection",
            start_time_ns=1,
            end_time_ns=2,
            parent_id="request-root",
            attributes={
                "agent.decision.type": "tool_selection",
                "agent.decision.selected_action": "lookup_order_status",
            },
        ),
        SimpleNamespace(
            name="decision.evidence_sufficiency",
            start_time_ns=21,
            end_time_ns=22,
            parent_id="request-root",
            attributes={
                "agent.decision.type": "evidence_sufficiency",
                "agent.decision.selected_action": "answer",
            },
        ),
    ]

    class RecoveredTrace:
        data = SimpleNamespace(spans=[root, tool, *decisions])

        def search_spans(self, *, span_type):
            return [tool] if span_type == "TOOL" else decisions

    feedback = decision_action_consistency_scorer()(trace=RecoveredTrace())

    assert feedback.value == "yes"


def test_nested_agent_error_does_not_waive_terminal_decision():
    scorer = decision_action_consistency_scorer()
    trace = FakeTrace(
        [("lookup_order_status", {"order_id": "A-1001"})],
        decisions=[("tool_selection", "lookup_order_status")],
        nested_agent_errored=True,
    )

    feedback = scorer(trace=trace)

    assert feedback.value == "no"
    assert "terminal 'evidence_sufficiency'" in feedback.rationale


def test_decision_consistency_uses_span_timestamps_when_available():
    class TimedTrace:
        def search_spans(self, *, span_type):
            if span_type == "TOOL":
                return [
                    SimpleNamespace(name="second_tool", start_time_ns=20),
                    SimpleNamespace(name="first_tool", start_time_ns=10),
                ]
            return [
                SimpleNamespace(
                    name="decision.evidence_sufficiency",
                    start_time_ns=30,
                    attributes={
                        "agent.decision.type": "evidence_sufficiency",
                        "agent.decision.selected_action": "answer",
                    },
                ),
                SimpleNamespace(
                    name="decision.tool_selection",
                    start_time_ns=20,
                    attributes={
                        "agent.decision.type": "tool_selection",
                        "agent.decision.selected_action": "second_tool",
                    },
                ),
                SimpleNamespace(
                    name="decision.tool_selection",
                    start_time_ns=10,
                    attributes={
                        "agent.decision.type": "tool_selection",
                        "agent.decision.selected_action": "first_tool",
                    },
                ),
            ]

    feedback = decision_action_consistency_scorer()(trace=TimedTrace())

    assert feedback.value == "yes"


@pytest.mark.parametrize(
    ("decision_start", "tool_start", "decision_parent", "tool_parent", "reason"),
    [
        (20, 10, "root", "root", "recorded after its TOOL span"),
        (10, 20, "root-a", "root-b", "different trace parents"),
    ],
)
def test_decision_consistency_rejects_invalid_temporal_or_parent_structure(
    decision_start,
    tool_start,
    decision_parent,
    tool_parent,
    reason,
):
    class InvalidStructureTrace:
        def search_spans(self, *, span_type):
            if span_type == "TOOL":
                return [
                    SimpleNamespace(
                        name="lookup_order_status",
                        start_time_ns=tool_start,
                        parent_id=tool_parent,
                    )
                ]
            return [
                SimpleNamespace(
                    name="decision.tool_selection",
                    start_time_ns=decision_start,
                    parent_id=decision_parent,
                    attributes={
                        "agent.decision.type": "tool_selection",
                        "agent.decision.selected_action": "lookup_order_status",
                    },
                ),
                SimpleNamespace(
                    name="decision.evidence_sufficiency",
                    start_time_ns=30,
                    parent_id=decision_parent,
                    attributes={
                        "agent.decision.type": "evidence_sufficiency",
                        "agent.decision.selected_action": "answer",
                    },
                ),
            ]

    feedback = decision_action_consistency_scorer()(trace=InvalidStructureTrace())

    assert feedback.value == "no"
    assert reason in feedback.rationale


def test_decision_consistency_rejects_span_overlapping_tool_execution():
    class OverlappingTrace:
        def search_spans(self, *, span_type):
            if span_type == "TOOL":
                return [
                    SimpleNamespace(
                        name="lookup_order_status",
                        start_time_ns=20,
                        parent_id="root",
                    )
                ]
            return [
                SimpleNamespace(
                    name="decision.tool_selection",
                    start_time_ns=10,
                    end_time_ns=25,
                    parent_id="root",
                    attributes={
                        "agent.decision.type": "tool_selection",
                        "agent.decision.selected_action": "lookup_order_status",
                    },
                ),
                SimpleNamespace(
                    name="decision.evidence_sufficiency",
                    start_time_ns=30,
                    end_time_ns=31,
                    parent_id="root",
                    attributes={
                        "agent.decision.type": "evidence_sufficiency",
                        "agent.decision.selected_action": "answer",
                    },
                ),
            ]

    feedback = decision_action_consistency_scorer()(trace=OverlappingTrace())

    assert feedback.value == "no"
    assert "ended after its TOOL span began" in feedback.rationale


@pytest.mark.parametrize(
    ("terminal_start", "terminal_parent", "tool_status", "reason"),
    [
        (15, "root", "OK", "began before the final TOOL span ended"),
        (25, "other-root", "OK", "different trace parents"),
        (15, "root", "ERROR", "began before the final TOOL span ended"),
        (25, "other-root", "ERROR", "different trace parents"),
    ],
)
def test_terminal_decision_must_follow_final_tool_and_share_its_parent(
    terminal_start,
    terminal_parent,
    tool_status,
    reason,
):
    class InvalidTerminalTrace:
        def search_spans(self, *, span_type):
            if span_type == "TOOL":
                return [
                    SimpleNamespace(
                        name="lookup_order_status",
                        start_time_ns=10,
                        end_time_ns=20,
                        parent_id="root",
                        status=SimpleNamespace(status_code=tool_status),
                    )
                ]
            return [
                SimpleNamespace(
                    name="decision.tool_selection",
                    start_time_ns=1,
                    end_time_ns=2,
                    parent_id="root",
                    attributes={
                        "agent.decision.type": "tool_selection",
                        "agent.decision.selected_action": "lookup_order_status",
                    },
                ),
                SimpleNamespace(
                    name="decision.evidence_sufficiency",
                    start_time_ns=terminal_start,
                    end_time_ns=terminal_start + 1,
                    parent_id=terminal_parent,
                    attributes={
                        "agent.decision.type": "evidence_sufficiency",
                        "agent.decision.selected_action": "answer",
                    },
                ),
            ]

    feedback = decision_action_consistency_scorer()(trace=InvalidTerminalTrace())

    assert feedback.value == "no"
    assert reason in feedback.rationale


def test_decision_tool_appropriateness_reuses_expected_names_and_multiplicity():
    scorer = decision_tool_appropriateness_scorer()
    trace = FakeTrace(
        [],
        decisions=[
            ("tool_selection", "lookup_order_status"),
            ("tool_selection", "lookup_order_status"),
        ],
    )
    expectations = {
        "expected_tool_calls": [
            {"name": "lookup_order_status", "arguments": {"order_id": "A-1001"}},
            {"name": "lookup_order_status", "arguments": {"order_id": "A-1002"}},
        ]
    }

    passed = scorer(trace=trace, expectations=expectations)
    failed = scorer(
        trace=FakeTrace(
            [],
            decisions=[("tool_selection", "lookup_order_status")],
        ),
        expectations=expectations,
    )

    assert passed.value == "yes"
    assert failed.value == "no"
    assert "Reviewed cases expected" in failed.rationale


def test_no_tool_appropriateness_requires_answer_readiness_decision():
    scorer = decision_tool_appropriateness_scorer()
    expectations = {"expected_tool_calls": []}

    passed = scorer(
        trace=FakeTrace([], decisions=[("answer_readiness", "answer")]),
        expectations=expectations,
    )
    wrong_action = scorer(
        trace=FakeTrace(
            [],
            decisions=[("tool_selection", "answer_from_memory")],
        ),
        expectations=expectations,
    )
    missing = scorer(trace=FakeTrace([]), expectations=expectations)

    assert passed.value == "yes"
    assert wrong_action.value == "no"
    assert missing.value == "no"


def test_trace_execution_success_is_independent_operational_feedback():
    scorer = trace_execution_success_scorer()

    no_tool = scorer(trace=FakeTrace([]))
    successful_tool = scorer(
        trace=FakeTrace(
            [("lookup_order_status", {"order_id": "A-1001"})],
        )
    )

    assert no_tool.value == "yes"
    assert successful_tool.value == "yes"
    assert "root or TOOL span" in successful_tool.rationale

    unset_root = SimpleNamespace(
        name="agent.evaluate",
        parent_id=None,
        status=SimpleNamespace(status_code="UNSET"),
    )
    unset_trace = SimpleNamespace(
        data=SimpleNamespace(spans=[unset_root]),
        search_spans=lambda *, span_type: [],
    )
    assert scorer(trace=unset_trace).value == "yes"


def test_trace_execution_success_fails_for_root_or_tool_errors():
    scorer = trace_execution_success_scorer()

    root_error = scorer(trace=FakeTrace([], root_errored=True))
    tool_error = scorer(
        trace=FakeTrace(
            [("lookup_order_status", {"order_id": "A-1001"})],
            errored_call_indexes={0},
        )
    )

    assert root_error.value == "no"
    assert "agent.evaluate" in root_error.rationale
    assert tool_error.value == "no"
    assert "lookup_order_status" in tool_error.rationale


@pytest.mark.parametrize(
    "root_span",
    [
        None,
        SimpleNamespace(name="agent.evaluate", parent_id=None, status=None),
        SimpleNamespace(
            name="agent.evaluate",
            parent_id=None,
            status=SimpleNamespace(status_code="UNKNOWN"),
        ),
    ],
)
def test_trace_execution_success_reports_missing_or_unknown_status(root_span):
    spans = [] if root_span is None else [root_span]
    trace = SimpleNamespace(
        data=SimpleNamespace(spans=spans),
        search_spans=lambda *, span_type: [],
    )

    feedback = trace_execution_success_scorer()(trace=trace)

    assert feedback.value is None
    assert isinstance(feedback.error, TypeError | ValueError)


def test_trace_execution_success_reports_missing_tool_status():
    trace = FakeTrace([])
    tool = SimpleNamespace(name="lookup_order_status", status=None)
    trace.search_spans = lambda *, span_type: (
        [tool] if span_type == "TOOL" else [trace.root_span]
    )

    feedback = trace_execution_success_scorer()(trace=trace)

    assert feedback.value is None
    assert isinstance(feedback.error, TypeError)
