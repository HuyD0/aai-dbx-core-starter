"""Unit tests for governed feedback and trace curation."""

from types import SimpleNamespace

import pytest

from aai_core.monitoring import (
    FeedbackSourceKind,
    log_feedback,
    traces_with_feedback,
)


class FakeAssessmentSource:
    def __init__(self, *, source_type, source_id=None):
        self.source_type = source_type
        self.source_id = source_id


def _fake_mlflow(captured):
    return SimpleNamespace(
        entities=SimpleNamespace(AssessmentSource=FakeAssessmentSource),
        log_feedback=lambda **kwargs: captured.update(kwargs),
    )


def test_log_feedback_forwards_native_kwargs_with_governed_source():
    captured: dict = {}

    log_feedback(
        trace_id="trace-1",
        name="correct",
        value=False,
        rationale="Cited the wrong quarter.",
        source_kind="human",
        source_id="group:domain-reviewers",
        mlflow_module=_fake_mlflow(captured),
    )

    assert captured["trace_id"] == "trace-1"
    assert captured["name"] == "correct"
    assert captured["value"] is False
    assert captured["rationale"] == "Cited the wrong quarter."
    assert captured["source"].source_type == "HUMAN"
    assert captured["source"].source_id == "group:domain-reviewers"
    assert "metadata" not in captured
    assert "span_id" not in captured


def test_log_feedback_parses_the_source_kind_vocabulary():
    captured: dict = {}

    log_feedback(
        trace_id="trace-1",
        name="groundedness",
        value=0.5,
        source_kind=FeedbackSourceKind.LLM_JUDGE,
        source_id="judge:groundedness-v1",
        mlflow_module=_fake_mlflow(captured),
    )

    assert captured["source"].source_type == "LLM_JUDGE"
    with pytest.raises(ValueError):
        log_feedback(
            trace_id="trace-1",
            name="groundedness",
            value=0.5,
            source_kind="vibes",
            source_id="judge:groundedness-v1",
            mlflow_module=_fake_mlflow({}),
        )


def test_log_feedback_refuses_personal_email_source():
    with pytest.raises(ValueError, match="non-personal"):
        log_feedback(
            trace_id="trace-1",
            name="correct",
            value=True,
            source_id="reviewer@example.com",
            mlflow_module=_fake_mlflow({}),
        )


def test_log_feedback_requires_kind_namespaced_provenance():
    # A bare username or employee id is a personal identity even without
    # an "@"; the kind namespace makes the non-personal claim structural.
    for personal in (
        "alice",
        "employee-1234",
        "group:",
        "group:with space",
        "group:" + "a" * 65,
        "GROUP:domain-reviewers",
    ):
        with pytest.raises(ValueError, match="non-personal"):
            log_feedback(
                trace_id="trace-1",
                name="correct",
                value=True,
                source_id=personal,
                mlflow_module=_fake_mlflow({}),
            )


def test_log_feedback_namespace_must_match_the_source_kind():
    # A judge cannot claim reviewer-group provenance and vice versa; each
    # kind owns exactly one namespace.
    with pytest.raises(ValueError, match="judge:"):
        log_feedback(
            trace_id="trace-1",
            name="groundedness",
            value=0.5,
            source_kind=FeedbackSourceKind.LLM_JUDGE,
            source_id="group:domain-reviewers",
            mlflow_module=_fake_mlflow({}),
        )
    with pytest.raises(ValueError, match="code:"):
        log_feedback(
            trace_id="trace-1",
            name="response_length_ok",
            value=1.0,
            source_kind="code",
            source_id="judge:groundedness-v1",
            mlflow_module=_fake_mlflow({}),
        )
    captured: dict = {}
    log_feedback(
        trace_id="trace-1",
        name="response_length_ok",
        value=1.0,
        source_kind="code",
        source_id="code:response_length_ok",
        mlflow_module=_fake_mlflow(captured),
    )
    assert captured["source"].source_id == "code:response_length_ok"


def test_log_feedback_requires_string_identifiers():
    # str() would turn None into the nonblank literal "None" and an int
    # into a plausible id, so MLflow would be called for the wrong trace
    # instead of the input failing at this untrusted-input boundary.
    for field, bad in (
        ("trace_id", None),
        ("trace_id", 123),
        ("name", None),
        ("span_id", 7),
    ):
        arguments = {
            "trace_id": "trace-1",
            "name": "correct",
            "value": True,
            "source_id": "group:domain-reviewers",
            "mlflow_module": _fake_mlflow({}),
        }
        arguments[field] = bad
        with pytest.raises(TypeError, match=field):
            log_feedback(**arguments)


def test_log_feedback_normalizes_forwarded_identifiers():
    # These reach MLflow: an untrimmed trace id addresses a different or
    # invalid trace, and an untrimmed name records feedback under a label
    # traces_with_feedback will never match.
    captured: dict = {}

    log_feedback(
        trace_id="  trace-1 ",
        name=" correct\n",
        value=False,
        source_id="group:domain-reviewers",
        span_id=" span-7 ",
        mlflow_module=_fake_mlflow(captured),
    )

    assert captured["trace_id"] == "trace-1"
    assert captured["name"] == "correct"
    assert captured["span_id"] == "span-7"
    with pytest.raises(ValueError, match="span_id"):
        log_feedback(
            trace_id="trace-1",
            name="correct",
            value=False,
            source_id="group:domain-reviewers",
            span_id="   ",
            mlflow_module=_fake_mlflow({}),
        )


def test_log_feedback_refuses_blank_identifiers():
    with pytest.raises(ValueError, match="trace_id"):
        log_feedback(
            trace_id=" ",
            name="correct",
            value=True,
            source_id="group:domain-reviewers",
            mlflow_module=_fake_mlflow({}),
        )


def test_log_feedback_requires_a_provenance_source_id():
    # Provenance is mandatory: omitting source_id is a signature error, and
    # a blank value cannot slip past as an empty identity.
    with pytest.raises(TypeError, match="source_id"):
        log_feedback(
            trace_id="trace-1",
            name="correct",
            value=True,
            mlflow_module=_fake_mlflow({}),
        )
    with pytest.raises(ValueError, match="source_id must not be blank"):
        log_feedback(
            trace_id="trace-1",
            name="correct",
            value=True,
            source_id="  ",
            mlflow_module=_fake_mlflow({}),
        )


def _trace(trace_id, assessments=None, shape="info"):
    if shape == "info":
        return SimpleNamespace(
            info=SimpleNamespace(assessments=assessments), trace_id=trace_id
        )
    if shape == "flat":
        return SimpleNamespace(assessments=assessments, trace_id=trace_id)
    return SimpleNamespace(trace_id=trace_id)


def test_traces_with_feedback_filters_by_name_and_value():
    wrong = SimpleNamespace(name="correct", value=False)
    right = SimpleNamespace(name="correct", value=True)
    other = SimpleNamespace(name="latency", value=False)
    traces = [
        _trace("keep-info", [wrong]),
        _trace("keep-flat", [wrong, other], shape="flat"),
        _trace("drop-value", [right]),
        _trace("drop-name", [other]),
        _trace("drop-shapeless", shape="bare"),
    ]

    selected = traces_with_feedback(traces, name="correct", value=False)

    assert [trace.trace_id for trace in selected] == ["keep-info", "keep-flat"]
    assert traces_with_feedback(traces, name="correct") == traces[:3]


def test_traces_with_feedback_reads_nested_feedback_values():
    nested = SimpleNamespace(
        name="correct",
        value=None,
        feedback=SimpleNamespace(value=False),
    )
    trace = _trace("nested", [nested])

    assert traces_with_feedback([trace], name="correct", value=False) == [trace]
    assert traces_with_feedback([trace], name="correct", value=True) == []


def test_traces_with_feedback_ignores_invalidated_assessments():
    invalidated = SimpleNamespace(name="correct", value=False, valid=False)
    corrected = SimpleNamespace(name="correct", value=True, valid=True)
    trace = _trace("overridden", [invalidated, corrected])

    assert traces_with_feedback([trace], name="correct", value=False) == []
    assert traces_with_feedback([trace], name="correct", value=True) == [trace]


def test_traces_with_feedback_excludes_expectation_assessments():
    expectation = SimpleNamespace(
        name="correct", value=False, expectation=SimpleNamespace(value=False)
    )
    feedback = SimpleNamespace(name="correct", value=False)
    only_expectation = _trace("ground-truth", [expectation])
    reviewed = _trace("reviewed", [expectation, feedback])

    assert traces_with_feedback([only_expectation], name="correct", value=False) == []
    assert traces_with_feedback([reviewed], name="correct", value=False) == [reviewed]


def test_traces_with_feedback_excludes_errored_scorer_feedback():
    # A failed scorer's Feedback stays valid and feedback-typed but carries
    # an error; a wildcard query must not curate it as reviewed feedback.
    top_level_error = SimpleNamespace(
        name="correct", value=None, error=SimpleNamespace(error_code="SCORER_ERROR")
    )
    nested_error = SimpleNamespace(
        name="correct",
        value=None,
        feedback=SimpleNamespace(value=None, error=SimpleNamespace(code="boom")),
    )
    reviewed = SimpleNamespace(name="correct", value=False)

    assert (
        traces_with_feedback([_trace("errored", [top_level_error])], name="correct")
        == []
    )
    assert (
        traces_with_feedback([_trace("nested", [nested_error])], name="correct") == []
    )
    trace = _trace("mixed", [top_level_error, reviewed])
    assert traces_with_feedback([trace], name="correct") == [trace]


def test_traces_with_feedback_excludes_issue_references():
    issue_link = SimpleNamespace(
        name="correct", value=None, expectation=None, issue=SimpleNamespace(id="i-1")
    )
    trace = _trace("issue-linked", [issue_link])

    assert traces_with_feedback([trace], name="correct") == []


def test_traces_with_feedback_requests_only_feedback_from_native_search():
    requested: dict = {}

    def search_assessments(**kwargs):
        requested.update(kwargs)
        return [SimpleNamespace(name="correct", value=False)]

    trace = SimpleNamespace(trace_id="typed", search_assessments=search_assessments)

    assert traces_with_feedback([trace], name="correct", value=False) == [trace]
    assert requested == {"type": "feedback"}


def test_traces_with_feedback_prefers_native_assessment_search():
    stale = SimpleNamespace(name="correct", value=False)
    corrected = SimpleNamespace(name="correct", value=True)
    trace = SimpleNamespace(
        trace_id="searched",
        info=SimpleNamespace(assessments=[stale, corrected]),
        search_assessments=lambda: [corrected],
    )

    # The native search already excludes overridden feedback, so the stale
    # raw-list entry must not select the trace.
    assert traces_with_feedback([trace], name="correct", value=False) == []
    assert traces_with_feedback([trace], name="correct", value=True) == [trace]
