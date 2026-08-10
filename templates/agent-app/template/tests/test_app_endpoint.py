"""App endpoint tests use native MLflow request/response contracts."""

import asyncio
import importlib
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from mlflow.types.responses import ResponsesAgentRequest

from aai_core import tracing
from aai_core.agents import AgentResponse
from aai_core.tracing import TraceCaptureMode, TracePolicy
from app import endpoint

start_server = importlib.import_module("start_server")
agent_server_app = start_server.app


@pytest.mark.parametrize("value", [None, "", "0", "01", "-1", "latest", " 7 "])
def test_app_requires_an_explicit_positive_prompt_version(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("AAI_PROMPT_VERSION", raising=False)
    else:
        monkeypatch.setenv("AAI_PROMPT_VERSION", value)

    with pytest.raises(RuntimeError, match="AAI_PROMPT_VERSION"):
        endpoint.required_prompt_version()


def test_app_loads_the_exact_configured_prompt_version(monkeypatch):
    captured = []
    sentinel = object()
    context = SimpleNamespace(
        configure_tracing=lambda **options: captured.append(("tracing", options))
    )

    monkeypatch.setenv("AAI_PROMPT_VERSION", "7")
    monkeypatch.setattr(endpoint, "bootstrap", lambda: context)
    monkeypatch.setattr(
        endpoint,
        "ToolAgent",
        lambda received_context, *, prompt_version: (
            captured.append(("agent", received_context, prompt_version)) or sentinel
        ),
    )
    endpoint._application.cache_clear()
    try:
        assert endpoint.required_prompt_version() == 7
        assert endpoint._application() is sentinel
        assert endpoint._application() is sentinel
        assert captured[0][0] == "tracing"
        assert captured[0][1]["integration"].value == "mlflow_agent_server"
        assert "policy" not in captured[0][1]
        assert captured[1] == ("agent", context, 7)
    finally:
        endpoint._application.cache_clear()


def test_agent_server_shutdown_closes_worker_client(monkeypatch):
    class FakeApplication:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    application = FakeApplication()
    context = SimpleNamespace(configure_tracing=lambda **options: None)
    monkeypatch.setenv("AAI_PROMPT_VERSION", "7")
    monkeypatch.setattr(endpoint, "bootstrap", lambda: context)
    monkeypatch.setattr(
        endpoint,
        "ToolAgent",
        lambda received_context, *, prompt_version: application,
    )
    endpoint._application.cache_clear()

    assert endpoint._application() is application
    asyncio.run(endpoint.close_application())

    assert application.closed
    assert endpoint._application.cache_info().currsize == 0


def test_server_lifecycle_initializes_and_closes_worker_resources():
    assert endpoint.initialize_application in agent_server_app.router.on_startup
    assert endpoint.close_application in agent_server_app.router.on_shutdown


def test_server_rejects_invalid_config_before_listening(monkeypatch):
    listened = []

    monkeypatch.delenv("AAI_PROMPT_VERSION", raising=False)
    monkeypatch.setattr(
        start_server.agent_server,
        "run",
        lambda **kwargs: listened.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="AAI_PROMPT_VERSION"):
        start_server.main()
    assert listened == []


def test_agent_server_endpoint_uses_bounded_conversation_context(monkeypatch):
    captured = []

    class FakeApplication:
        context = SimpleNamespace(
            tags=SimpleNamespace(for_trace=lambda: {"aai.application": "test"})
        )

        async def ainvoke(self, request):
            captured.append(request)
            return AgentResponse(
                content="ready",
                metadata={"quality": "checked"},
            )

    monkeypatch.setattr(endpoint, "_application", lambda: FakeApplication())
    monkeypatch.setattr(endpoint.mlflow, "get_current_active_span", lambda: None)
    monkeypatch.setattr(endpoint, "set_trace_resource_context", lambda context: None)
    monkeypatch.setattr(endpoint, "set_trace_session", lambda session_id: None)
    monkeypatch.setattr(
        endpoint,
        "current_trace_state",
        lambda: SimpleNamespace(policy=TracePolicy()),
    )
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "Hello"}],
        context={
            "conversation_id": "opaque-conversation-1",
            "user_id": "must-not-propagate",
        },
    )

    response = asyncio.run(endpoint.invoke_agent(request))

    assert captured[0].session_id == "opaque-conversation-1"
    assert captured[0].user_id is None
    assert response.custom_outputs == {"quality": "checked"}
    assert response.output[0].content[0]["text"] == "ready"


def test_agent_server_http_invocations_contract(monkeypatch):
    """Exercise MLflow's real request validator, route, and response serializer."""

    captured_requests = []

    class FakeApplication:
        context = SimpleNamespace(
            tags=SimpleNamespace(for_trace=lambda: {"aai.application": "test"})
        )

        async def ainvoke(self, request):
            captured_requests.append(request)
            return AgentResponse(
                content="ready over HTTP",
                metadata={"quality": "checked"},
            )

    class FakeSpan:
        trace_id = "tr-test"

        def __init__(self):
            self.inputs = None
            self.outputs = None
            self.attributes = {}

        def set_inputs(self, value):
            self.inputs = value

        def set_outputs(self, value):
            self.outputs = value

        def set_attribute(self, name, value):
            self.attributes[name] = value

    span = FakeSpan()

    @contextmanager
    def fake_start_span(*_args, **_kwargs):
        yield span

    monkeypatch.setattr(endpoint, "_application", lambda: FakeApplication())
    monkeypatch.setattr(endpoint.mlflow, "start_span", fake_start_span)
    monkeypatch.setattr(
        endpoint.mlflow,
        "get_current_active_span",
        lambda: span,
    )
    monkeypatch.setattr(
        endpoint,
        "set_trace_resource_context",
        lambda context: None,
    )
    monkeypatch.setattr(endpoint, "set_trace_session", lambda session_id: None)
    monkeypatch.setattr(
        endpoint,
        "current_trace_state",
        lambda: SimpleNamespace(policy=TracePolicy()),
    )

    async def invoke_over_http():
        transport = httpx.ASGITransport(app=agent_server_app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://agent.test",
        ) as client:
            return await client.post(
                "/invocations",
                json={
                    "input": [{"role": "user", "content": "Hello"}],
                    "context": {
                        "conversation_id": "opaque-conversation-1",
                        "user_id": "must-not-propagate",
                    },
                },
            )

    response = asyncio.run(invoke_over_http())
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["output"][0]["content"][0]["text"] == "ready over HTTP"
    assert payload["custom_outputs"] == {"quality": "checked"}
    assert captured_requests[0].session_id == "opaque-conversation-1"
    assert captured_requests[0].user_id is None
    assert span.inputs == {
        "input": [{"role": "user", "content": "Hello"}],
        "context": {"conversation_id": "opaque-conversation-1"},
    }


@pytest.mark.parametrize("mode", ["restricted", "off"])
def test_real_agent_server_http_trace_export_respects_privacy(mode, tmp_path):
    """Cover raw root outputs written by AgentServer after app code returns."""

    probe = Path(__file__).with_name("_endpoint_trace_probe.py")
    result = subprocess.run(
        [sys.executable, str(probe), mode, str(tmp_path / f"{mode}.db")],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_root_inputs_follow_internal_and_restricted_capture_policy(monkeypatch):
    class FakeSpan:
        def __init__(self):
            self.inputs = None

        def set_inputs(self, value):
            self.inputs = value

    span = FakeSpan()
    monkeypatch.setattr(endpoint.mlflow, "get_current_active_span", lambda: span)

    monkeypatch.setattr(
        endpoint,
        "current_trace_state",
        lambda: SimpleNamespace(
            policy=TracePolicy(capture_mode=TraceCaptureMode.BOUNDED)
        ),
    )
    endpoint._set_bounded_root_inputs(
        [{"role": "user", "content": "internal prompt"}],
        "conversation-1",
    )
    assert span.inputs["input"][0]["content"] == "internal prompt"

    monkeypatch.setattr(
        endpoint,
        "current_trace_state",
        lambda: SimpleNamespace(
            policy=TracePolicy(capture_mode=TraceCaptureMode.METADATA_ONLY)
        ),
    )
    endpoint._set_bounded_root_inputs(
        [{"role": "user", "content": "restricted prompt"}],
        "conversation-1",
    )
    tracing._native_span_policy_processor(
        TracePolicy(capture_mode=TraceCaptureMode.METADATA_ONLY)
    )(span)
    assert span.inputs == {
        "type": "mapping",
        "size": 2,
        "truncated": False,
    }
    assert "restricted prompt" not in str(span.inputs)

    span.inputs = {"input": "broad native request"}
    monkeypatch.setattr(
        endpoint,
        "current_trace_state",
        lambda: SimpleNamespace(policy=TracePolicy(capture_mode=TraceCaptureMode.OFF)),
    )
    endpoint._set_bounded_root_inputs(
        [{"role": "user", "content": "never capture"}],
        "conversation-1",
    )
    assert span.inputs is None


def test_agent_server_streaming_contract(monkeypatch):
    class FakeSpan:
        def __init__(self):
            self.inputs = None

        def set_inputs(self, value):
            self.inputs = value

    class FakeApplication:
        context = SimpleNamespace(
            tags=SimpleNamespace(for_trace=lambda: {"aai.application": "test"})
        )

        async def astream_text(self, request):
            assert request.session_id == "opaque-conversation-1"
            yield "rea"
            yield "dy"

    span = FakeSpan()
    monkeypatch.setattr(endpoint, "_application", lambda: FakeApplication())
    monkeypatch.setattr(endpoint.mlflow, "get_current_active_span", lambda: span)
    monkeypatch.setattr(endpoint, "set_trace_resource_context", lambda context: None)
    monkeypatch.setattr(endpoint, "set_trace_session", lambda session_id: None)
    monkeypatch.setattr(
        endpoint,
        "current_trace_state",
        lambda: SimpleNamespace(policy=TracePolicy()),
    )
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "Hello"}],
        context={"conversation_id": "opaque-conversation-1"},
    )

    async def collect():
        return [event async for event in endpoint.stream_agent(request)]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_item.done",
    ]
    assert events[0].model_extra["delta"] == "rea"
    assert events[-1].model_extra["item"]["content"][0]["text"] == "ready"
    assert span.inputs == {
        "input": [{"role": "user", "content": "Hello"}],
        "context": {"conversation_id": "opaque-conversation-1"},
    }


def test_agent_server_stream_cancellation_closes_application_iterator(monkeypatch):
    class FakeApplication:
        context = SimpleNamespace(
            tags=SimpleNamespace(for_trace=lambda: {"aai.application": "test"})
        )

        def __init__(self):
            self.closed = False

        async def astream_text(self, request):
            try:
                yield "partial"
                await asyncio.Future()
            finally:
                self.closed = True

    application = FakeApplication()
    monkeypatch.setattr(endpoint, "_application", lambda: application)
    monkeypatch.setattr(endpoint.mlflow, "get_current_active_span", lambda: None)
    monkeypatch.setattr(endpoint, "set_trace_resource_context", lambda context: None)
    monkeypatch.setattr(endpoint, "set_trace_session", lambda session_id: None)
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "Hello"}],
        context={"conversation_id": "opaque-conversation-1"},
    )

    async def cancel_after_first_event():
        first_event = asyncio.Event()

        async def consume():
            async for event in endpoint.stream_agent(request):
                assert event.type == "response.output_text.delta"
                first_event.set()

        pending = asyncio.create_task(consume())
        await first_event.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            _ = await pending

    asyncio.run(cancel_after_first_event())

    assert application.closed
