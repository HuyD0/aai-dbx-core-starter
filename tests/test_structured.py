"""Typed structured-output boundary tests."""

from contextlib import contextmanager

import pytest
from pydantic import BaseModel, ConfigDict

from aai_core import tracing
from aai_core.structured import (
    StructuredOutputError,
    generate_structured,
    generate_typed,
)
from aai_core.testing import FakeChatModel


@contextmanager
def _sdk_trace_state():
    """Activate bounded SDK instrumentation without configuring real MLflow."""

    state = tracing.TraceState(
        metadata={},
        policy=tracing.TracePolicy(),
        integration=tracing.TraceIntegration.SDK,
    )
    token = tracing._TRACE_STATE.set(state)
    try:
        yield
    finally:
        tracing._TRACE_STATE.reset(token)


class _FakeSpan:
    def __init__(self):
        self.attributes = {}
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status


@contextmanager
def _recording_mlflow(monkeypatch):
    """Capture the spans the structured-output boundary opens."""

    from conftest import install_fake_module

    recorded = []

    @contextmanager
    def start_span(name, span_type):
        span = _FakeSpan()
        recorded.append({"name": name, "span_type": span_type, "span": span})
        yield span

    install_fake_module(monkeypatch, "mlflow", start_span=start_span)
    with _sdk_trace_state():
        yield recorded


class ShipmentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    answer: str
    confidence: float


def test_generate_typed_uses_model_schema_and_strict_validation():
    model = FakeChatModel(reply='{"answer":"shipped","confidence":0.9}')

    answer = generate_typed(
        model,
        [{"role": "user", "content": "Where is the order?"}],
        response_model=ShipmentAnswer,
    )

    assert answer == ShipmentAnswer(answer="shipped", confidence=0.9)
    response_format = model.requests[0]["response_format"]
    assert response_format["json_schema"]["name"] == "ShipmentAnswer"
    assert (
        response_format["json_schema"]["schema"]["properties"]["answer"]["type"]
        == "string"
    )
    assert response_format["json_schema"]["strict"] is True


def test_generate_typed_sanitizes_validation_failures():
    sensitive_content = '{"answer":"private customer content","confidence":"high"}'
    model = FakeChatModel(reply=sensitive_content)

    with pytest.raises(StructuredOutputError) as exc_info:
        generate_typed(
            model,
            [{"role": "user", "content": "question"}],
            response_model=ShipmentAnswer,
        )

    assert "invalid structured output" in str(exc_info.value)
    assert sensitive_content not in str(exc_info.value)
    assert "private customer content" not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_generate_typed_opens_a_parser_span(monkeypatch):
    model = FakeChatModel(reply='{"answer":"shipped","confidence":0.9}')

    with _recording_mlflow(monkeypatch) as recorded:
        generate_typed(
            model,
            [{"role": "user", "content": "Where is the order?"}],
            response_model=ShipmentAnswer,
        )

    parse = [entry for entry in recorded if entry["span_type"] == "PARSER"]
    assert len(parse) == 1
    assert parse[0]["name"] == "structured.parse"
    assert parse[0]["span"].attributes == {
        "aai.provider": "fake",
        "aai.logical_name": "general-chat",
        "gen_ai.output.type": "json",
    }
    assert parse[0]["span"].status is None


def test_parser_span_carries_the_failure_the_model_span_cannot(monkeypatch):
    """The provider call succeeds; only the parse span can show the failure."""

    sensitive_content = '{"answer":"private customer content","confidence":"high"}'
    model = FakeChatModel(reply=sensitive_content)

    with (
        _recording_mlflow(monkeypatch) as recorded,
        pytest.raises(StructuredOutputError),
    ):
        generate_typed(
            model,
            [{"role": "user", "content": "question"}],
            response_model=ShipmentAnswer,
        )

    parse = [entry for entry in recorded if entry["span_type"] == "PARSER"]
    assert len(parse) == 1
    assert parse[0]["span"].status == "ERROR"
    # The span identifies the call; it never restates the rejected content.
    assert sensitive_content not in str(parse[0]["span"].attributes)


def test_generate_structured_parser_span_reports_schema_violations(monkeypatch):
    model = FakeChatModel(reply='{"answer":"shipped"}')

    with (
        _recording_mlflow(monkeypatch) as recorded,
        pytest.raises(StructuredOutputError, match="missing required keys"),
    ):
        generate_structured(
            model,
            [{"role": "user", "content": "question"}],
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer", "confidence"],
            },
        )

    parse = [entry for entry in recorded if entry["span_type"] == "PARSER"]
    assert len(parse) == 1
    assert parse[0]["span"].status == "ERROR"
