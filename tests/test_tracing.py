import asyncio
import inspect
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_core import tracing
from aai_core.providers import OpenAICompatibleChatModel
from aai_core.tags import ResourceContext


@pytest.fixture(autouse=True)
def _reset_trace_state(monkeypatch):
    default = tracing.TraceState(
        metadata={},
        policy=tracing.TracePolicy(
            capture_mode=tracing.TraceCaptureMode.OFF,
        ),
    )
    monkeypatch.setattr(tracing, "_DEFAULT_TRACE_STATE", default)
    monkeypatch.setattr(tracing, "_PROCESS_TRACE_CONFIGURATION", None)
    token = tracing._TRACE_STATE.set(None)
    yield
    tracing._TRACE_STATE.reset(token)


def _resource_context(
    *,
    application: str = "example-assistant",
    cost_center: str = "CC-1234",
) -> ResourceContext:
    return ResourceContext(
        application=application,
        project="example-ai",
        environment="dev",
        team="data-platform",
        owner_group="group:data-platform-owners",
        cost_center=cost_center,
        data_classification="internal",
        lifecycle="experimental",
        repository="example-org/example-repo",
        release="dev",
    )


class FakeMlflow:
    def __init__(self):
        self.active = False
        self.experiment = None
        self.trace_metadata = None
        self.trace_session = None
        self.trace_updates = []
        self.openai_autologged = False
        self.langchain_autologged = False
        self.openai = SimpleNamespace(autolog=self._autolog_openai)
        self.langchain = SimpleNamespace(autolog=self._autolog_langchain)

    def _autolog_openai(self, **_options):
        self.openai_autologged = True

    def _autolog_langchain(self, **_options):
        self.langchain_autologged = True

    def set_experiment(self, name):
        self.experiment = name

    def trace(self, **trace_options):
        def decorate(target):
            if inspect.iscoroutinefunction(target):

                @wraps(target)
                async def invoke_async(*args, **kwargs):
                    self.active = True
                    try:
                        return await target(*args, **kwargs)
                    finally:
                        self.active = False

                return invoke_async

            @wraps(target)
            def invoke(*args, **kwargs):
                self.active = True
                try:
                    return target(*args, **kwargs)
                finally:
                    self.active = False

            return invoke

        return decorate

    def update_current_trace(self, **options):
        assert self.active, "trace metadata must be applied inside an active trace"
        self.trace_updates.append(options)
        if "metadata" in options:
            self.trace_metadata = options["metadata"]
        if "session_id" in options:
            self.trace_session = options["session_id"]


def test_configured_metadata_is_applied_inside_traced_call(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    context = ResourceContext(
        application="example-assistant",
        project="example-ai",
        environment="dev",
        team="data-platform",
        owner_group="group:data-platform-owners",
        cost_center="CC-1234",
        data_classification="internal",
        lifecycle="experimental",
        repository="example-org/example-repo",
        release="dev",
    )

    tracing.configure_tracing(
        context,
        experiment_name="/Shared/example-ai",
        integration=tracing.TraceIntegration.SDK,
    )

    assert fake_mlflow.trace_metadata is None

    @tracing.traced(span_type="CHAIN")
    def answer() -> str:
        return "ready"

    assert answer() == "ready"
    assert fake_mlflow.experiment == "/Shared/example-ai"
    assert fake_mlflow.trace_metadata == context.for_trace()


def test_async_traced_call_stays_active_until_awaited_body_finishes(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    context = _resource_context()
    tracing.configure_tracing(
        context,
        experiment_name="/Shared/example-ai-async-trace",
    )

    @tracing.traced(span_type="CHAIN")
    async def answer() -> str:
        assert fake_mlflow.active
        await asyncio.sleep(0)
        assert fake_mlflow.active
        return "ready"

    assert inspect.iscoroutinefunction(answer)
    with tracing.trace_context(metadata={"release": "test"}):
        assert asyncio.run(answer()) == "ready"
    assert fake_mlflow.active is False
    assert fake_mlflow.trace_metadata == {
        **context.for_trace(),
        "release": "test",
    }


def test_langchain_autolog_is_opt_in(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    context = ResourceContext(
        application="example-assistant",
        project="example-ai",
        environment="dev",
        team="data-platform",
        owner_group="group:data-platform-owners",
        cost_center="CC-1234",
        data_classification="internal",
        lifecycle="experimental",
        repository="example-org/example-repo",
        release="dev",
    )

    tracing.configure_tracing(
        context,
        experiment_name="/Shared/example-ai",
        integration=tracing.TraceIntegration.MLFLOW_LANGCHAIN,
        policy=tracing.TracePolicy(
            capture_mode=tracing.TraceCaptureMode.FULL,
        ),
    )

    assert not fake_mlflow.openai_autologged
    assert fake_mlflow.langchain_autologged


def test_openai_autolog_is_opt_in(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    context = ResourceContext(
        application="example-assistant",
        project="example-ai",
        environment="dev",
        team="data-platform",
        owner_group="group:data-platform-owners",
        cost_center="CC-1234",
        data_classification="internal",
        lifecycle="experimental",
        repository="example-org/example-repo",
        release="dev",
    )

    tracing.configure_tracing(
        context,
        experiment_name="/Shared/example-ai",
        integration=tracing.TraceIntegration.MLFLOW_OPENAI,
        policy=tracing.TracePolicy(
            capture_mode=tracing.TraceCaptureMode.FULL,
        ),
    )

    assert fake_mlflow.openai_autologged
    assert not fake_mlflow.langchain_autologged


def test_native_autologging_requires_full_capture_policy(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    context = ResourceContext(
        application="example-assistant",
        project="example-ai",
        environment="dev",
        team="data-platform",
        owner_group="group:data-platform-owners",
        cost_center="CC-1234",
        data_classification="internal",
        lifecycle="experimental",
        repository="example-org/example-repo",
        release="dev",
    )

    with pytest.raises(ValueError, match="requires TraceCaptureMode.FULL"):
        tracing.configure_tracing(
            context,
            experiment_name="/Shared/example-ai",
            integration=tracing.TraceIntegration.MLFLOW_OPENAI,
            policy=tracing.TracePolicy(),
        )


def test_trace_session_uses_dedicated_mlflow_field(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    tracing.configure_tracing(
        _resource_context(),
        experiment_name="/Shared/example-ai-session-trace",
    )

    @tracing.traced(span_type="CHAIN")
    def answer() -> str:
        tracing.set_trace_session("conversation-123")
        return "ready"

    assert answer() == "ready"
    assert fake_mlflow.trace_session == "conversation-123"
    assert fake_mlflow.trace_updates[-1] == {"session_id": "conversation-123"}
    assert fake_mlflow.trace_updates[0]["metadata"] == _resource_context().for_trace()


def test_trace_context_is_execution_local_and_restored():
    original = dict(tracing.current_trace_state().metadata)

    async def observe(request_id):
        with tracing.trace_context(metadata={"request_id": request_id}):
            await asyncio.sleep(0)
            return dict(tracing.current_trace_state().metadata)

    async def collect():
        return await asyncio.gather(observe("first"), observe("second"))

    first, second = asyncio.run(collect())

    assert first["request_id"] == "first"
    assert second["request_id"] == "second"
    assert dict(tracing.current_trace_state().metadata) == original


def test_trace_context_cannot_change_process_capture_policy():
    with pytest.raises(TypeError):
        with tracing.trace_context(policy=tracing.TracePolicy()):  # type: ignore[call-arg]
            pass


def test_process_configuration_is_idempotent_and_rejects_conflicts(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    first = tracing.configure_tracing(
        _resource_context(application="first-application"),
        experiment_name="/Shared/first-application",
        policy=tracing.TracePolicy(max_string_length=10),
    )
    repeated = tracing.configure_tracing(
        _resource_context(application="first-application"),
        experiment_name="/Shared/first-application",
        policy=tracing.TracePolicy(max_string_length=10),
    )

    assert repeated is first
    with pytest.raises(RuntimeError, match="already configured"):
        tracing.configure_tracing(
            _resource_context(application="second-application"),
            experiment_name="/Shared/second-application",
            policy=tracing.TracePolicy(max_string_length=20),
        )


def test_request_metadata_cannot_replace_resource_fields_and_is_bounded(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    context = _resource_context()
    tracing.configure_tracing(
        context,
        experiment_name="/Shared/example-ai",
        policy=tracing.TracePolicy(max_string_length=4),
    )

    @tracing.traced(span_type="CHAIN")
    def answer() -> str:
        with pytest.raises(ValueError, match="controlled field"):
            tracing.set_trace_context({"aai.cost_center": "ATTACKER"})
        tracing.set_trace_context(
            {
                "request_id": "abcdef",
                "vendor_api_key": "never-log",
            }
        )
        return "ready"

    assert answer() == "ready"
    assert fake_mlflow.trace_updates[-1] == {
        "metadata": {
            "request_id": "abcd<truncated>",
            "vendor_api_key": "[REDACTED]",
        }
    }
    assert (
        tracing.current_trace_state().metadata["aai.cost_center"] == context.cost_center
    )


def test_native_server_can_bind_typed_resource_context_once(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    context = _resource_context()
    tracing.configure_tracing(
        context,
        experiment_name="/Shared/example-ai-agent-server",
        integration=tracing.TraceIntegration.MLFLOW_AGENT_SERVER,
    )

    @tracing.traced(span_type="AGENT")
    def answer() -> str:
        tracing.set_trace_resource_context(context)
        tracing.set_trace_context({"aai.cost_center": context.cost_center})
        with pytest.raises(ValueError, match="conflicts"):
            tracing.set_trace_resource_context(
                _resource_context(cost_center="CC-OTHER")
            )
        return "ready"

    assert answer() == "ready"
    assert fake_mlflow.trace_metadata == {
        "aai.cost_center": context.cost_center,
    }


def test_off_policy_skips_native_trace_creation(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    @tracing.traced(span_type="CHAIN")
    def answer() -> str:
        assert fake_mlflow.active is False
        return "ready"

    assert answer() == "ready"


def test_bounded_payload_redacts_and_limits_content():
    policy = tracing.TracePolicy(
        capture_mode=tracing.TraceCaptureMode.BOUNDED,
        max_payload_depth=3,
        max_string_length=4,
        max_collection_items=1,
    )

    redacted = tracing.sanitize_trace_payload(
        {"vendor_api_key": "never-log"},
        policy=policy,
    )
    bounded = tracing.sanitize_trace_payload(["abcdef", "second"], policy=policy)

    assert redacted["vendor_api_key"] == "[REDACTED]"
    assert bounded == ["abcd<truncated>", {"<truncated>": 1}]

    @dataclass
    class Payload:
        content: str
        api_key: str

    structured_policy = tracing.TracePolicy(
        capture_mode=tracing.TraceCaptureMode.BOUNDED,
        max_string_length=4,
        max_collection_items=2,
    )
    structured = tracing.sanitize_trace_payload(
        Payload(content="abcdef", api_key="never-log"),
        policy=structured_policy,
    )
    assert structured == {
        "content": "abcd<truncated>",
        "api_key": "[REDACTED]",
    }


def test_trace_policy_is_strict_frozen_and_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        tracing.TracePolicy(capture_mode="off")
    with pytest.raises(ValidationError):
        tracing.TracePolicy(unrecognized=True)

    policy = tracing.TracePolicy()
    with pytest.raises(ValidationError):
        policy.max_payload_depth = 2


def test_bounded_traced_function_never_hands_raw_payload_to_mlflow(monkeypatch):
    class FakeSpan:
        def __init__(self):
            self.inputs = None
            self.outputs = None
            self.attributes = {}

        def set_inputs(self, value):
            self.inputs = value

        def set_outputs(self, value):
            self.outputs = value

        def set_attribute(self, key, value):
            self.attributes[key] = value

    class SpanMlflow(FakeMlflow):
        def __init__(self):
            super().__init__()
            self.span = FakeSpan()

        @contextmanager
        def start_span(self, **options):
            self.active = True
            try:
                yield self.span
            finally:
                self.active = False

    fake_mlflow = SpanMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    policy = tracing.TracePolicy(
        capture_mode=tracing.TraceCaptureMode.BOUNDED,
        max_string_length=4,
    )
    tracing.configure_tracing(
        _resource_context(),
        experiment_name="/Shared/example-ai-bounded-payload",
        policy=policy,
    )

    @tracing.traced(span_type="CHAIN")
    def answer(api_key: str, question: str) -> str:
        return question

    assert answer("never-log", "abcdef") == "abcdef"

    assert fake_mlflow.span.inputs == {
        "api_key": "[REDACTED]",
        "question": "abcd<truncated>",
    }
    assert fake_mlflow.span.outputs == "abcd<truncated>"


def test_real_openai_autolog_nests_once_and_reports_usage_and_cost(tmp_path):
    """Canary the certified MLflow/OpenAI integration without a network call."""

    mlflow = pytest.importorskip("mlflow")
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    from mlflow import MlflowClient
    from mlflow.tracking import fluent

    original_tracking_uri = mlflow.get_tracking_uri()
    original_experiment_id = fluent._active_experiment_id
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "native-openai-autolog-compatibility"
    client = MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(
        experiment_name,
        artifact_location=(tmp_path / "artifacts").resolve().as_uri(),
    )

    def respond(_request):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-compatibility",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ready"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )

    model = OpenAICompatibleChatModel(
        logical_name="general-chat",
        provider="openai-compatible",
        model="gpt-4o-mini",
        client=openai.OpenAI(
            api_key="synthetic-test-value",
            base_url="https://example.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ),
    )
    try:
        mlflow.set_tracking_uri(tracking_uri)
        tracing.configure_tracing(
            _resource_context(),
            experiment_name=experiment_name,
            integration=tracing.TraceIntegration.MLFLOW_OPENAI,
            policy=tracing.TracePolicy(
                capture_mode=tracing.TraceCaptureMode.FULL,
            ),
        )

        @tracing.traced(name="application.invoke", span_type="CHAIN")
        def invoke(prompt: str) -> str:
            response = model.native_client.chat.completions.create(
                model=model.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

        assert invoke("synthetic prompt") == "ready"
        mlflow.flush_trace_async_logging()
        trace_id = mlflow.get_last_active_trace_id()
        assert trace_id is not None
        trace = mlflow.get_trace(trace_id)

        assert trace.info.token_usage == {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        }
        assert trace.info.cost is not None
        assert trace.info.cost["total_cost"] > 0
        assert trace.info.trace_metadata["aai.application"] == "example-assistant"
        span_names = [span.name for span in trace.data.spans]
        assert span_names.count("application.invoke") == 1
        assert span_names.count("Completions") == 1
    finally:
        mlflow.openai.autolog(disable=True)
        mlflow.set_tracking_uri(original_tracking_uri)
        fluent._active_experiment_id = original_experiment_id


def test_real_openai_async_and_streaming_autolog_are_complete_and_not_duplicated(
    tmp_path,
):
    """Behavioral canary for the native async and streaming escape hatch."""

    mlflow = pytest.importorskip("mlflow")
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    from mlflow import MlflowClient
    from mlflow.tracking import fluent

    original_tracking_uri = mlflow.get_tracking_uri()
    original_experiment_id = fluent._active_experiment_id
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "native-openai-async-stream-compatibility"
    tracking_client = MlflowClient(tracking_uri=tracking_uri)
    tracking_client.create_experiment(
        experiment_name,
        artifact_location=(tmp_path / "artifacts").resolve().as_uri(),
    )
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-async",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "ready",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )

    def stream_aware_respond(request):
        if b'"stream":true' not in request.content.replace(b" ", b""):
            return respond(request)
        requests.append(request)
        events = [
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "rea"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "dy"},
                        "finish_reason": "stop",
                    }
                ],
            },
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            },
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(body + "data: [DONE]\n\n").encode(),
        )

    model = OpenAICompatibleChatModel(
        logical_name="general-chat",
        provider="openai-compatible",
        model="gpt-4o-mini",
        client=object(),
        async_client_factory=lambda: openai.AsyncOpenAI(
            api_key="synthetic-test-value",
            base_url="https://example.invalid/v1",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(stream_aware_respond)
            ),
        ),
    )
    try:
        mlflow.set_tracking_uri(tracking_uri)
        tracing.configure_tracing(
            _resource_context(),
            experiment_name=experiment_name,
            integration=tracing.TraceIntegration.MLFLOW_OPENAI,
            policy=tracing.TracePolicy(
                capture_mode=tracing.TraceCaptureMode.FULL,
            ),
        )

        @tracing.traced(name="application.async_invoke", span_type="CHAIN")
        async def invoke_non_streaming(prompt: str) -> str:
            async with model.create_native_async_client() as native:
                response = await native.chat.completions.create(
                    model=model.model,
                    messages=[{"role": "user", "content": prompt}],
                )
            return response.choices[0].message.content or ""

        @tracing.traced(name="application.async_stream", span_type="CHAIN")
        async def invoke_streaming() -> str:
            async with model.create_native_async_client() as native:
                stream = await native.chat.completions.create(
                    model=model.model,
                    messages=[{"role": "user", "content": "synthetic prompt"}],
                    stream=True,
                    stream_options={"include_usage": True},
                )
                parts = []
                try:
                    async for event in stream:
                        choices = event.choices or ()
                        if choices and choices[0].delta.content:
                            parts.append(choices[0].delta.content)
                finally:
                    await stream.close()
            return "".join(parts)

        async def exercise():
            async def invoke_in_context(request_id):
                with tracing.trace_context(
                    metadata={"request_id": request_id},
                    session_id=f"session-{request_id}",
                ):
                    assert await invoke_non_streaming(request_id) == "ready"
                    return mlflow.get_last_active_trace_id()

            non_stream_trace_ids = await asyncio.gather(
                invoke_in_context("request-a"),
                invoke_in_context("request-b"),
            )
            assert await invoke_streaming() == "ready"
            stream_trace_id = mlflow.get_last_active_trace_id()
            return non_stream_trace_ids, stream_trace_id

        non_stream_trace_ids, stream_trace_id = asyncio.run(exercise())
        mlflow.flush_trace_async_logging()
        assert all(trace_id is not None for trace_id in non_stream_trace_ids)
        assert len(set(non_stream_trace_ids)) == 2
        assert stream_trace_id is not None
        assert stream_trace_id not in non_stream_trace_ids

        non_stream_traces = [
            mlflow.get_trace(trace_id) for trace_id in non_stream_trace_ids
        ]
        stream_trace = mlflow.get_trace(stream_trace_id)
        for trace in non_stream_traces:
            assert trace.info.token_usage == {
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            }
        assert stream_trace.info.token_usage == {
            "input_tokens": 4,
            "output_tokens": 2,
            "total_tokens": 6,
        }
        non_stream_names = [
            [span.name for span in trace.data.spans] for trace in non_stream_traces
        ]
        stream_names = [span.name for span in stream_trace.data.spans]
        assert all(
            names.count("application.async_invoke") == 1 for names in non_stream_names
        )
        assert stream_names.count("application.async_stream") == 1
        assert all(
            len([name for name in names if name != "application.async_invoke"]) == 1
            for names in non_stream_names
        )
        assert (
            len([name for name in stream_names if name != "application.async_stream"])
            == 1
        )
        sessions = {
            trace.info.trace_metadata["request_id"]: trace.info.trace_metadata[
                "mlflow.trace.session"
            ]
            for trace in non_stream_traces
        }
        assert sessions == {
            "request-a": "session-request-a",
            "request-b": "session-request-b",
        }
        assert len(requests) == 3
    finally:
        mlflow.openai.autolog(disable=True)
        mlflow.set_tracking_uri(original_tracking_uri)
        fluent._active_experiment_id = original_experiment_id


def test_agent_server_native_stream_cancellation_closes_without_invented_usage(
    tmp_path,
):
    mlflow = pytest.importorskip("mlflow")
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    from mlflow import MlflowClient
    from mlflow.tracking import fluent

    class BlockingSseBody(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = False

        async def __aiter__(self):
            event = {
                "id": "chatcmpl-cancel",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "partial"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(event)}\n\n".encode()
            await asyncio.Future()

        async def aclose(self):
            self.closed = True

    original_tracking_uri = mlflow.get_tracking_uri()
    original_experiment_id = fluent._active_experiment_id
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "native-openai-cancellation-compatibility"
    tracking_client = MlflowClient(tracking_uri=tracking_uri)
    tracking_client.create_experiment(
        experiment_name,
        artifact_location=(tmp_path / "artifacts").resolve().as_uri(),
    )
    body = BlockingSseBody()
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=body,
        )

    model = OpenAICompatibleChatModel(
        logical_name="general-chat",
        provider="openai-compatible",
        model="gpt-4o-mini",
        client=object(),
        async_client_factory=lambda: openai.AsyncOpenAI(
            api_key="synthetic-test-value",
            base_url="https://example.invalid/v1",
            max_retries=2,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ),
    )
    captured = {}
    try:
        mlflow.set_tracking_uri(tracking_uri)
        tracing.configure_tracing(
            _resource_context(),
            experiment_name=experiment_name,
            integration=tracing.TraceIntegration.MLFLOW_AGENT_SERVER,
            policy=tracing.TracePolicy(
                capture_mode=tracing.TraceCaptureMode.BOUNDED,
            ),
        )

        first_output = asyncio.Event()

        @tracing.traced(name="application.cancelled_stream", span_type="CHAIN")
        async def invoke_streaming() -> None:
            span = mlflow.get_current_active_span()
            captured["trace_id"] = span.trace_id
            with tracing.provider_span("model.stream", span_type="LLM") as provider:
                if provider is not None:
                    provider.set_inputs({"prompt": "synthetic prompt"})
                async with model.create_native_async_client() as native:
                    stream = await native.chat.completions.create(
                        model=model.model,
                        messages=[{"role": "user", "content": "synthetic prompt"}],
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    try:
                        async for event in stream:
                            if event.choices and event.choices[0].delta.content:
                                first_output.set()
                    finally:
                        await stream.close()

        async def cancel_after_output():
            task = asyncio.create_task(invoke_streaming())
            await asyncio.wait_for(first_output.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_after_output())
        mlflow.flush_trace_async_logging()

        assert body.closed
        assert len(requests) == 1
        trace = mlflow.get_trace(captured["trace_id"], flush=True)
        assert trace is not None
        assert not trace.info.token_usage
        assert not trace.info.cost
        names = [span.name for span in trace.data.spans]
        assert names.count("application.cancelled_stream") == 1
        assert names.count("model.stream") == 1
    finally:
        mlflow.openai.autolog(disable=True)
        mlflow.set_tracking_uri(original_tracking_uri)
        fluent._active_experiment_id = original_experiment_id
