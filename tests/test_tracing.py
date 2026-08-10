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
from aai_core.agents import AgentDecision, AgentDecisionType
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
    monkeypatch.setattr(tracing, "_PROCESS_TRACE_CONFIGURATION", {})
    token = tracing._TRACE_STATE.set(None)
    yield
    tracing._TRACE_STATE.reset(token)


def _resource_context(
    *,
    application: str = "example-assistant",
    cost_center: str = "CC-1234",
    data_classification: str = "internal",
) -> ResourceContext:
    return ResourceContext(
        application=application,
        project="example-ai",
        environment="dev",
        team="data-platform",
        owner_group="group:data-platform-owners",
        cost_center=cost_center,
        data_classification=data_classification,
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

    def get_current_active_span(self):
        return SimpleNamespace() if self.active else None


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


@pytest.mark.parametrize(
    ("classification", "capture_mode"),
    [
        ("public", tracing.TraceCaptureMode.BOUNDED),
        ("internal", tracing.TraceCaptureMode.BOUNDED),
        ("confidential", tracing.TraceCaptureMode.METADATA_ONLY),
        ("restricted", tracing.TraceCaptureMode.METADATA_ONLY),
    ],
)
def test_default_trace_policy_follows_data_classification(
    monkeypatch,
    classification,
    capture_mode,
):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    state = tracing.configure_tracing(
        _resource_context(data_classification=classification),
        experiment_name="/Shared/classification-policy",
    )

    assert state.policy.capture_mode is capture_mode


def test_explicit_trace_policy_can_strengthen_reviewed_capture(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    explicit = tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.OFF)

    state = tracing.configure_tracing(
        _resource_context(data_classification="restricted"),
        experiment_name="/Shared/explicit-policy",
        policy=explicit,
    )

    assert state.policy is explicit


def test_sensitive_classification_cannot_enable_payload_capture(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    with pytest.raises(ValueError, match="cannot weaken"):
        tracing.configure_tracing(
            _resource_context(data_classification="restricted"),
            experiment_name="/Shared/restricted-policy",
            policy=tracing.TracePolicy(
                capture_mode=tracing.TraceCaptureMode.BOUNDED,
            ),
        )


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


@pytest.mark.parametrize(
    ("integration", "span_type", "expected_manual_span"),
    [
        (tracing.TraceIntegration.MLFLOW_OPENAI, "LLM", False),
        (tracing.TraceIntegration.MLFLOW_OPENAI, "CHAT_MODEL", False),
        (tracing.TraceIntegration.MLFLOW_OPENAI, "EMBEDDING", False),
        (tracing.TraceIntegration.MLFLOW_OPENAI, "TOOL", True),
        (tracing.TraceIntegration.MLFLOW_OPENAI, "RETRIEVER", True),
        (tracing.TraceIntegration.MLFLOW_LANGCHAIN, "LLM", False),
        (tracing.TraceIntegration.MLFLOW_LANGCHAIN, "TOOL", False),
        (tracing.TraceIntegration.MLFLOW_LANGCHAIN, "RETRIEVER", False),
    ],
)
def test_native_autolog_provider_span_ownership(
    monkeypatch,
    integration,
    span_type,
    expected_manual_span,
):
    class OwnershipSpan:
        def __init__(self):
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

    class OwnershipMlflow(FakeMlflow):
        def __init__(self):
            super().__init__()
            self.started_spans = []

        @contextmanager
        def start_span(self, **options):
            span = OwnershipSpan()
            self.started_spans.append((options, span))
            self.active = True
            try:
                yield span
            finally:
                self.active = False

    fake_mlflow = OwnershipMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    tracing.configure_tracing(
        _resource_context(),
        experiment_name="/Shared/native-autolog-ownership",
        integration=integration,
        policy=tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.FULL),
    )

    with tracing.provider_span("operation", span_type=span_type) as span:
        assert (span is not None) is expected_manual_span

    assert len(fake_mlflow.started_spans) == int(expected_manual_span)


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


def test_metadata_only_trace_hashes_session_identifier_before_mlflow(monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    tracing.configure_tracing(
        _resource_context(data_classification="restricted"),
        experiment_name="/Shared/restricted-session",
    )

    @tracing.traced(span_type="CHAIN")
    def answer() -> str:
        tracing.set_trace_session("patient@example.com")
        return "ready"

    assert answer() == "ready"
    assert fake_mlflow.trace_session is not None
    assert fake_mlflow.trace_session.startswith("sha256:")
    assert "patient@example.com" not in fake_mlflow.trace_session


def test_trace_context_is_execution_local_and_restored():
    original = dict(tracing.current_trace_state().metadata)

    async def observe(request_id):
        with tracing.trace_context(metadata={"request_id": request_id}):
            await asyncio.sleep(0)
            return dict(tracing.current_trace_state().metadata)

    async def collect():
        return await asyncio.gather(observe("first"), observe("second"))

    first, second = asyncio.run(collect())

    assert first["request_id"] == tracing._opaque_identifier("first")
    assert second["request_id"] == tracing._opaque_identifier("second")
    assert first["request_id"] != second["request_id"]
    assert dict(tracing.current_trace_state().metadata) == original


def test_trace_context_cannot_change_process_capture_policy():
    with (
        pytest.raises(TypeError),
        tracing.trace_context(policy=tracing.TracePolicy()),  # type: ignore[call-arg]
    ):
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


def test_metadata_only_span_keeps_typed_operational_evidence_shape_only():
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

    native = FakeSpan()
    span = tracing.GovernedSpan(
        native,
        tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.METADATA_ONLY),
    )

    span.set_inputs({"patient-123-45-6789": "confidential prompt"})
    span.set_outputs({"answer": "confidential answer"})
    span.set_attribute("gen_ai.tool.name", "lookup_order")
    span.set_attribute("mlflow.llm.model", "gpt-4o-mini")
    span.set_attribute(
        "mlflow.chat.tokenUsage",
        {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    )
    span.set_attribute(
        "mlflow.llm.cost",
        {"input_cost": 0.001, "output_cost": 0.002, "total_cost": 0.003},
    )
    span.set_attribute("custom.prompt", "confidential prompt")
    span.set_attribute("aai.model", "github_pat_not-an-identifier")
    span.set_attribute("agent.decision.type", "tool_selection")
    span.set_attribute("agent.decision.selected_action", "lookup_order")
    span.set_attribute("agent.decision.confidence", 0.94)
    span.set_attribute("agent.decision.goal", "Use confidential order data")
    span.set_attribute(
        "agent.decision.reason",
        "The confidential request requires retrieval.",
    )
    span.set_attribute(
        "agent.decision.evidence_refs",
        ["confidential-request-reference"],
    )

    assert native.inputs == {"type": "mapping", "size": 1, "truncated": False}
    assert native.outputs == {
        "type": "mapping",
        "size": 1,
        "truncated": False,
    }
    assert native.attributes["gen_ai.tool.name"] == "lookup_order"
    assert native.attributes["mlflow.llm.model"] == "gpt-4o-mini"
    assert native.attributes["mlflow.chat.tokenUsage"] == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }
    assert native.attributes["mlflow.llm.cost"] == {
        "input_cost": 0.001,
        "output_cost": 0.002,
        "total_cost": 0.003,
    }
    assert "custom.prompt" not in native.attributes
    assert native.attributes["aai.model"] == {"type": "str", "length": 28}
    assert native.attributes["agent.decision.type"] == "tool_selection"
    assert native.attributes["agent.decision.selected_action"] == "lookup_order"
    assert native.attributes["agent.decision.confidence"] == 0.94
    assert "agent.decision.goal" not in native.attributes
    assert "agent.decision.reason" not in native.attributes
    assert "agent.decision.evidence_refs" not in native.attributes


@pytest.mark.parametrize("integration", tuple(tracing.TraceIntegration))
def test_record_agent_decision_creates_native_agent_span_for_every_owner(
    monkeypatch,
    integration,
):
    class DecisionSpan:
        def __init__(self):
            self.attributes = {}
            self.inputs = None
            self.outputs = None

        def set_attribute(self, key, value):
            self.attributes[key] = value

    class DecisionMlflow(FakeMlflow):
        def __init__(self):
            super().__init__()
            self.spans = []
            self._span_depth = 0

        @contextmanager
        def start_span(self, **options):
            span = DecisionSpan()
            self.spans.append((options, span))
            self._span_depth += 1
            self.active = True
            try:
                yield span
            finally:
                self._span_depth -= 1
                self.active = self._span_depth > 0

    fake_mlflow = DecisionMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    options = {
        "experiment_name": "/Shared/decision-record",
        "integration": integration,
    }
    if integration in {
        tracing.TraceIntegration.MLFLOW_OPENAI,
        tracing.TraceIntegration.MLFLOW_LANGCHAIN,
    }:
        options["policy"] = tracing.TracePolicy(
            capture_mode=tracing.TraceCaptureMode.FULL,
        )
    tracing.configure_tracing(_resource_context(), **options)
    decision = AgentDecision(
        decision_type=AgentDecisionType.TOOL_SELECTION,
        goal="Answer with authoritative order status",
        selected_action="lookup_order_status",
        reason="The request requires current order data.",
        evidence_refs=("user_request", "tool_schema:lookup_order_status"),
        confidence=0.94,
        alternatives_considered=("answer_without_tool",),
        expected_result="Return the current order status.",
    )

    with fake_mlflow.start_span(name="agent.root", span_type="AGENT"):
        tracing.record_agent_decision(decision)
        if integration in {
            tracing.TraceIntegration.MLFLOW_OPENAI,
            tracing.TraceIntegration.MLFLOW_LANGCHAIN,
        }:
            with tracing.provider_span("model.generate", span_type="LLM") as provider:
                assert provider is None

    assert [options["name"] for options, _span in fake_mlflow.spans] == [
        "agent.root",
        "decision.tool_selection",
    ]
    span_options, span = fake_mlflow.spans[-1]
    assert span_options == {
        "name": "decision.tool_selection",
        "span_type": "AGENT",
    }
    assert span.attributes == {
        "agent.decision.type": "tool_selection",
        "agent.decision.goal": "Answer with authoritative order status",
        "agent.decision.selected_action": "lookup_order_status",
        "agent.decision.reason": "The request requires current order data.",
        "agent.decision.evidence_refs": [
            "user_request",
            "tool_schema:lookup_order_status",
        ],
        "agent.decision.confidence": 0.94,
        "agent.decision.alternatives": ["answer_without_tool"],
        "agent.decision.expected_result": "Return the current order status.",
    }
    assert span.inputs is None
    assert span.outputs is None


def test_record_agent_decision_does_not_create_a_disconnected_trace(monkeypatch):
    class DisconnectedMlflow(FakeMlflow):
        def __init__(self):
            super().__init__()
            self.started_spans = []

        @contextmanager
        def start_span(self, **options):
            self.started_spans.append(options)
            yield SimpleNamespace()

    fake_mlflow = DisconnectedMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    tracing.configure_tracing(
        _resource_context(),
        experiment_name="/Shared/decision-record",
        integration=tracing.TraceIntegration.MLFLOW_OPENAI,
        policy=tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.FULL),
    )

    tracing.record_agent_decision(
        AgentDecision(
            decision_type=AgentDecisionType.ANSWER_READINESS,
            goal="Answer the request",
            selected_action="answer",
            reason="The available evidence is sufficient.",
            evidence_refs=("user_request",),
        )
    )

    assert fake_mlflow.started_spans == []


def test_record_agent_decision_rejects_non_mlflow_active_parent(monkeypatch):
    class ExternalParent:
        def get_attribute(self, key):
            assert key == "mlflow.traceRequestId"
            return None

    class ExternalParentMlflow(FakeMlflow):
        def __init__(self):
            super().__init__()
            self.started_spans = []

        def get_current_active_span(self):
            return ExternalParent()

        @contextmanager
        def start_span(self, **options):
            self.started_spans.append(options)
            yield SimpleNamespace()

    fake_mlflow = ExternalParentMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    tracing.configure_tracing(
        _resource_context(),
        experiment_name="/Shared/decision-record",
    )

    tracing.record_agent_decision(
        AgentDecision(
            decision_type=AgentDecisionType.ANSWER_READINESS,
            goal="Answer the request",
            selected_action="answer",
            reason="The available evidence is sufficient.",
            evidence_refs=("user_request",),
        )
    )

    assert fake_mlflow.started_spans == []


def test_record_agent_decision_is_disabled_by_off_policy(monkeypatch):
    class OffMlflow(FakeMlflow):
        def __init__(self):
            super().__init__()
            self.started_spans = []

        @contextmanager
        def start_span(self, **options):
            self.started_spans.append(options)
            yield SimpleNamespace()

    fake_mlflow = OffMlflow()
    fake_mlflow.active = True
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    tracing._TRACE_STATE.set(
        tracing.TraceState(
            metadata={},
            policy=tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.OFF),
        )
    )

    tracing.record_agent_decision(
        AgentDecision(
            decision_type=AgentDecisionType.ANSWER_READINESS,
            goal="Answer the request",
            selected_action="answer",
            reason="The available evidence is sufficient.",
            evidence_refs=("user_request",),
        )
    )

    assert fake_mlflow.started_spans == []


def test_record_agent_decision_is_a_noop_without_mlflow(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    tracing._TRACE_STATE.set(
        tracing.TraceState(
            metadata={},
            policy=tracing.TracePolicy(),
        )
    )

    tracing.record_agent_decision(
        AgentDecision(
            decision_type=AgentDecisionType.ANSWER_READINESS,
            goal="Answer the request",
            selected_action="answer",
            reason="The available evidence is sufficient.",
            evidence_refs=("user_request",),
        )
    )


def test_record_agent_decision_is_fail_open_for_tracing_errors(monkeypatch):
    @contextmanager
    def failing_span(*_args, **_kwargs):
        raise RuntimeError("tracing backend unavailable")
        yield

    monkeypatch.setattr(tracing, "_application_semantic_span", failing_span)

    tracing.record_agent_decision(
        AgentDecision(
            decision_type=AgentDecisionType.ANSWER_READINESS,
            goal="Answer the request",
            selected_action="answer",
            reason="The available evidence is sufficient.",
            evidence_refs=("tool_result:call-1",),
        )
    )


def test_record_agent_decision_omits_absent_optional_attributes(monkeypatch):
    captured = {}

    @contextmanager
    def capturing_span(name, *, span_type, attributes):
        captured.update(
            name=name,
            span_type=span_type,
            attributes=dict(attributes),
        )
        yield None

    monkeypatch.setattr(tracing, "_application_semantic_span", capturing_span)

    tracing.record_agent_decision(
        AgentDecision(
            decision_type=AgentDecisionType.ANSWER_READINESS,
            goal="Answer the request",
            selected_action="answer",
            reason="The available evidence is sufficient.",
            evidence_refs=("tool_result:call-1",),
        )
    )

    assert captured["name"] == "decision.answer_readiness"
    assert captured["span_type"] == "AGENT"
    assert "agent.decision.confidence" not in captured["attributes"]
    assert "agent.decision.alternatives" not in captured["attributes"]
    assert "agent.decision.expected_result" not in captured["attributes"]


def test_record_agent_decision_does_not_swallow_process_cancellation(monkeypatch):
    @contextmanager
    def interrupted_span(*_args, **_kwargs):
        raise KeyboardInterrupt
        yield

    monkeypatch.setattr(tracing, "_application_semantic_span", interrupted_span)
    decision = AgentDecision(
        decision_type=AgentDecisionType.ANSWER_READINESS,
        goal="Answer the request",
        selected_action="answer",
        reason="The available evidence is sufficient.",
        evidence_refs=("tool_result:call-1",),
    )

    with pytest.raises(KeyboardInterrupt):
        tracing.record_agent_decision(decision)


def test_off_governed_span_drops_attributes():
    class FakeSpan:
        def __init__(self):
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

    native = FakeSpan()
    span = tracing.GovernedSpan(
        native,
        tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.OFF),
    )

    span.set_attribute("mlflow.chat.tokenUsage", {"total_tokens": 12})

    assert native.attributes == {}


def test_native_span_processor_sanitizes_framework_owned_root():
    class FakeLiveSpan:
        def __init__(self):
            self.inputs = {"patient-123-45-6789": "confidential prompt"}
            self.outputs = {"answer": "confidential answer"}
            self.attributes = {
                "mlflow.traceRequestId": "tr-0123456789abcdef",
                "mlflow.spanInputs": self.inputs,
                "mlflow.spanOutputs": self.outputs,
                "mlflow.message.format": "openai",
                "mlflow.chat.tokenUsage": {"total_tokens": 12},
                "mlflow.gateway.linkedTraceId": "gateway-trace-123",
                "mlflow.linkedPrompts": (
                    '[{"name":"catalog.schema.agent-system","version":"7"}]'
                ),
                "session.id": "customer-session-123",
                "custom.payload": "confidential answer",
            }

        def set_inputs(self, value):
            self.inputs = value
            self.attributes["mlflow.spanInputs"] = value

        def set_outputs(self, value):
            self.outputs = value
            self.attributes["mlflow.spanOutputs"] = value

        def set_attribute(self, key, value):
            self.attributes[key] = value

    span = FakeLiveSpan()
    processor = tracing._native_span_policy_processor(
        tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.METADATA_ONLY)
    )

    processor(span)

    assert span.inputs == {"type": "mapping", "size": 1, "truncated": False}
    assert span.outputs == {"type": "mapping", "size": 1, "truncated": False}
    assert span.attributes["mlflow.traceRequestId"] == "tr-0123456789abcdef"
    assert span.attributes["mlflow.message.format"] == "openai"
    assert span.attributes["mlflow.chat.tokenUsage"] == {"total_tokens": 12}
    assert span.attributes["mlflow.gateway.linkedTraceId"] == "gateway-trace-123"
    assert span.attributes["mlflow.linkedPrompts"] == (
        '[{"name":"catalog.schema.agent-system","version":"7"}]'
    )
    assert span.attributes["session.id"].startswith("sha256:")
    assert "customer-session-123" not in span.attributes["session.id"]
    assert span.attributes["custom.payload"] == {"type": "str", "length": 19}


def test_agent_server_governed_span_is_sanitized_once_at_export():
    class FakeLiveSpan:
        def __init__(self):
            self.inputs = None
            self.outputs = None
            self.attributes = {}

        def set_inputs(self, value):
            self.inputs = value
            self.attributes["mlflow.spanInputs"] = value

        def set_outputs(self, value):
            self.outputs = value
            self.attributes["mlflow.spanOutputs"] = value

        def set_attribute(self, key, value):
            self.attributes[key] = value

    policy = tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.METADATA_ONLY)
    native = FakeLiveSpan()
    span = tracing.GovernedSpan(native, policy, sanitize_at_export=True)
    span.set_inputs({"a": "one", "b": "two"})
    span.set_outputs(["one", "two", "three"])

    tracing._native_span_policy_processor(policy)(native)

    assert native.inputs == {"type": "mapping", "size": 2, "truncated": False}
    assert native.outputs == {
        "type": "sequence",
        "size": 3,
        "item_types": ["str"],
        "truncated": False,
    }


def test_native_span_processor_clears_payload_before_sanitizer_failure(monkeypatch):
    class FakeLiveSpan:
        def __init__(self):
            self.inputs = {"secret": "raw-input"}
            self.outputs = {"secret": "raw-output"}
            self.attributes = {"custom.payload": "raw-attribute"}

        def set_inputs(self, value):
            self.inputs = value

        def set_outputs(self, value):
            self.outputs = value

        def set_attribute(self, key, value):
            self.attributes[key] = value

    native = FakeLiveSpan()
    monkeypatch.setattr(
        tracing,
        "sanitize_trace_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )

    tracing._native_span_policy_processor(tracing.TracePolicy())(native)

    assert native.inputs is None
    assert native.outputs is None
    assert native.attributes["custom.payload"] == {"type": "suppressed"}


def test_provider_span_does_not_record_sensitive_exception_text(monkeypatch):
    class FakeSpan:
        def __init__(self):
            self.statuses = []

        def set_status(self, value):
            self.statuses.append(value)

        def set_attribute(self, key, value):
            pass

    class SpanMlflow(FakeMlflow):
        def __init__(self):
            super().__init__()
            self.span = FakeSpan()
            self.exception_seen_by_context = None

        @contextmanager
        def start_span(self, **options):
            try:
                yield self.span
            except BaseException as error:
                self.exception_seen_by_context = error
                raise

    fake_mlflow = SpanMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    tracing.configure_tracing(
        _resource_context(data_classification="restricted"),
        experiment_name="/Shared/restricted-provider-error",
    )

    def invoke_failure():
        with tracing.provider_span("model", span_type="LLM"):
            raise ValueError("patient-secret-exception-9917")

    with pytest.raises(ValueError, match="patient-secret-exception-9917"):
        invoke_failure()

    assert fake_mlflow.exception_seen_by_context is None
    assert fake_mlflow.span.statuses == ["ERROR"]


def test_sdk_tracing_does_not_replace_native_span_processors(monkeypatch):
    class NativeTracing:
        def __init__(self):
            self.configure_calls = []

        def configure(self, **options):
            self.configure_calls.append(options)

    fake_mlflow = FakeMlflow()
    fake_mlflow.tracing = NativeTracing()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    tracing.configure_tracing(
        _resource_context(),
        experiment_name="/Shared/sdk-owned-spans",
        integration=tracing.TraceIntegration.SDK,
    )

    assert fake_mlflow.tracing.configure_calls == []


def test_agent_server_appends_privacy_processor_without_replacing_platform_ones():
    platform_calls = []

    def platform_processor(span):
        platform_calls.append(span)

    old_aai_processor = tracing._native_span_policy_processor(
        tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.BOUNDED)
    )

    class NativeTracing:
        def __init__(self):
            self.processors = [platform_processor, old_aai_processor]
            self.configure_calls = []
            self.config = SimpleNamespace(
                get_config=lambda: SimpleNamespace(
                    span_processors=list(self.processors)
                )
            )

        def configure(self, **options):
            self.configure_calls.append(options)
            self.processors = list(options["span_processors"])

        def enable(self):
            pass

    native = NativeTracing()
    fake_mlflow = SimpleNamespace(tracing=native)
    policy = tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.METADATA_ONLY)

    tracing._configure_native_span_policy(fake_mlflow, policy)

    assert native.processors[0] is platform_processor
    assert len(native.processors) == 2
    assert native.processors[-1].__name__ == "aai_core_trace_capture_policy"
    assert native.processors[-1] is not old_aai_processor
    sentinel = object()
    for processor in native.processors:
        if processor is platform_processor:
            processor(sentinel)
    assert platform_calls == [sentinel]


def test_agent_server_full_capture_leaves_existing_processors_unchanged():
    def platform_processor(_span):
        pass

    class NativeTracing:
        def __init__(self):
            self.configure_calls = []
            self.config = SimpleNamespace(
                get_config=lambda: SimpleNamespace(span_processors=[platform_processor])
            )

        def configure(self, **options):
            self.configure_calls.append(options)

        def enable(self):
            pass

    native = NativeTracing()

    tracing._configure_native_span_policy(
        SimpleNamespace(tracing=native),
        tracing.TracePolicy(capture_mode=tracing.TraceCaptureMode.FULL),
    )

    assert native.configure_calls == []


def test_real_traced_async_generator_keeps_children_in_one_trace(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow import MlflowClient
    from mlflow.tracking import fluent

    original_tracking_uri = mlflow.get_tracking_uri()
    original_experiment_id = fluent._active_experiment_id
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "async-generator-trace-composition"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment(
        experiment_name,
        artifact_location=(tmp_path / "artifacts").resolve().as_uri(),
    )
    try:
        mlflow.set_tracking_uri(tracking_uri)
        tracing.configure_tracing(
            _resource_context(data_classification="restricted"),
            experiment_name=experiment_name,
            integration=tracing.TraceIntegration.SDK,
        )

        @tracing.traced(name="stream.root", span_type="CHAIN")
        async def stream(private_input: str):
            with tracing.provider_span("stream.child", span_type="LLM") as span:
                assert span is not None
                span.set_inputs({"private": private_input})
                yield "one"
                yield "two"

        assert inspect.isasyncgenfunction(stream)

        async def collect():
            return [item async for item in stream("private-stream-input")]

        assert asyncio.run(collect()) == ["one", "two"]
        mlflow.flush_trace_async_logging()
        traces = client.search_traces(
            locations=[experiment_id], include_spans=True, flush=True
        )
        assert len(traces) == 1
        trace = traces[0]
        assert {span.name for span in trace.data.spans} == {
            "stream.root",
            "stream.child",
        }
        root = next(span for span in trace.data.spans if span.name == "stream.root")
        assert root.outputs == {
            "type": "mapping",
            "size": 1,
            "truncated": False,
        }
        assert "private-stream-input" not in trace.to_json()
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)
        fluent._active_experiment_id = original_experiment_id


def test_real_agent_decision_uses_certified_native_mlflow_span_api(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow import MlflowClient
    from mlflow.tracking import fluent
    from opentelemetry.sdk.trace import TracerProvider

    original_tracking_uri = mlflow.get_tracking_uri()
    original_experiment_id = fluent._active_experiment_id
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "native-agent-decision-span"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment(
        experiment_name,
        artifact_location=(tmp_path / "artifacts").resolve().as_uri(),
    )
    try:
        mlflow.set_tracking_uri(tracking_uri)
        tracing.configure_tracing(
            _resource_context(),
            experiment_name=experiment_name,
            integration=tracing.TraceIntegration.SDK,
        )

        external_tracer = TracerProvider().get_tracer("external-parent-canary")
        with external_tracer.start_as_current_span("external.root"):
            assert mlflow.get_current_active_span() is None
            tracing.record_agent_decision(
                AgentDecision(
                    decision_type=AgentDecisionType.ANSWER_READINESS,
                    goal="Answer the request",
                    selected_action="answer",
                    reason="The available evidence is sufficient.",
                    evidence_refs=("user_request",),
                )
            )

        with mlflow.start_span(name="agent.root", span_type="AGENT"):
            tracing.record_agent_decision(
                AgentDecision(
                    decision_type=AgentDecisionType.TOOL_SELECTION,
                    goal="Answer with authoritative order status",
                    selected_action="lookup_order_status",
                    reason="The request requires current order data.",
                    evidence_refs=("user_request",),
                    confidence=0.94,
                )
            )

        mlflow.flush_trace_async_logging()
        trace_id = mlflow.get_last_active_trace_id()
        assert trace_id is not None
        trace = mlflow.get_trace(trace_id, flush=True)
        assert trace is not None
        traces = client.search_traces(
            locations=[experiment_id], include_spans=True, flush=True
        )
        assert len(traces) == 1
        assert {span.name for span in trace.data.spans} == {
            "agent.root",
            "decision.tool_selection",
        }
        root_span = next(span for span in trace.data.spans if span.name == "agent.root")
        decision_span = next(
            span for span in trace.data.spans if span.name == "decision.tool_selection"
        )
        assert decision_span.parent_id == root_span.span_id
        assert decision_span.span_type == "AGENT"
        assert decision_span.get_attribute("agent.decision.type") == "tool_selection"
        assert (
            decision_span.get_attribute("agent.decision.selected_action")
            == "lookup_order_status"
        )
        assert decision_span.get_attribute("agent.decision.confidence") == 0.94
        assert decision_span.get_attribute("agent.decision.evidence_refs") == [
            "user_request"
        ]
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)
        fluent._active_experiment_id = original_experiment_id


def test_real_metadata_only_trace_keeps_usage_without_payloads(tmp_path):
    """Canary sensitive-data telemetry against the supported MLflow runtime."""

    mlflow = pytest.importorskip("mlflow")
    from mlflow import MlflowClient
    from mlflow.tracking import fluent

    original_tracking_uri = mlflow.get_tracking_uri()
    original_experiment_id = fluent._active_experiment_id
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "metadata-only-operational-evidence"
    client = MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(
        experiment_name,
        artifact_location=(tmp_path / "artifacts").resolve().as_uri(),
    )
    try:
        mlflow.set_tracking_uri(tracking_uri)
        tracing.configure_tracing(
            _resource_context(data_classification="restricted"),
            experiment_name=experiment_name,
            integration=tracing.TraceIntegration.SDK,
        )

        @tracing.traced(name="restricted.invoke", span_type="AGENT")
        def invoke(prompt: str) -> str:
            tracing.set_trace_context(
                {
                    "customer_name": "Alice Example",
                    "request_id": "patient@example.com",
                }
            )
            with tracing.provider_span(
                "restricted.model",
                span_type="LLM",
                attributes={"mlflow.llm.model": "gpt-4o-mini"},
            ) as span:
                assert span is not None
                span.set_inputs({"patient-123-45-6789": prompt})
                span.set_outputs({"answer": "confidential answer"})
                span.set_attribute(
                    "mlflow.chat.tokenUsage",
                    {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                )
                span.set_attribute(
                    "mlflow.llm.cost",
                    {
                        "input_cost": 0.0001,
                        "output_cost": 0.0002,
                        "total_cost": 0.0003,
                    },
                )
            tracing.record_agent_decision(
                AgentDecision(
                    decision_type=AgentDecisionType.ANSWER_READINESS,
                    goal="Answer Alice Example's restricted request",
                    selected_action="answer",
                    reason="The confidential evidence is sufficient.",
                    evidence_refs=("tool_result:restricted-model",),
                    confidence=0.8,
                )
            )
            return "confidential answer"

        assert invoke("confidential prompt") == "confidential answer"
        mlflow.flush_trace_async_logging()
        trace_id = mlflow.get_last_active_trace_id()
        assert trace_id is not None
        trace = mlflow.get_trace(trace_id)
        serialized = json.dumps(trace.to_dict())
        assert "patient-123-45-6789" not in serialized
        assert "Alice Example" not in serialized
        assert "patient@example.com" not in serialized
        assert "confidential prompt" not in serialized
        assert "confidential answer" not in serialized
        assert "customer_name" not in trace.info.trace_metadata
        assert trace.info.trace_metadata["request_id"].startswith("sha256:")

        model_span = next(
            span for span in trace.data.spans if span.name == "restricted.model"
        )
        assert model_span.get_attribute("mlflow.llm.model") == "gpt-4o-mini"
        assert model_span.get_attribute("mlflow.chat.tokenUsage") == {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        }
        assert model_span.get_attribute("mlflow.llm.cost") == {
            "input_cost": 0.0001,
            "output_cost": 0.0002,
            "total_cost": 0.0003,
        }
        decision_span = next(
            span
            for span in trace.data.spans
            if span.name == "decision.answer_readiness"
        )
        assert decision_span.get_attribute("agent.decision.type") == "answer_readiness"
        assert decision_span.get_attribute("agent.decision.selected_action") == "answer"
        assert decision_span.get_attribute("agent.decision.confidence") == 0.8
        assert decision_span.get_attribute("agent.decision.goal") is None
        assert decision_span.get_attribute("agent.decision.reason") is None
        assert decision_span.get_attribute("agent.decision.evidence_refs") is None
        assert trace.info.token_usage == {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        }
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)
        fluent._active_experiment_id = original_experiment_id


def test_real_metadata_only_provider_error_does_not_persist_exception_text(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow import MlflowClient
    from mlflow.tracking import fluent

    original_tracking_uri = mlflow.get_tracking_uri()
    original_experiment_id = fluent._active_experiment_id
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "metadata-only-provider-error"
    client = MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(
        experiment_name,
        artifact_location=(tmp_path / "artifacts").resolve().as_uri(),
    )
    secret = "provider rejected patient-secret-exception-9917"
    try:
        mlflow.set_tracking_uri(tracking_uri)
        tracing.configure_tracing(
            _resource_context(data_classification="restricted"),
            experiment_name=experiment_name,
            integration=tracing.TraceIntegration.SDK,
        )

        def invoke_failure():
            with tracing.provider_span("restricted.model", span_type="LLM"):
                raise ValueError(secret)

        with pytest.raises(ValueError, match="patient-secret-exception-9917"):
            invoke_failure()

        mlflow.flush_trace_async_logging()
        trace_id = mlflow.get_last_active_trace_id()
        assert trace_id is not None
        trace = mlflow.get_trace(trace_id, flush=True)
        assert trace is not None
        serialized = json.dumps(trace.to_dict())
        assert secret not in serialized
        span = next(
            item for item in trace.data.spans if item.name == "restricted.model"
        )
        assert span.status.status_code.value == "ERROR"
        assert not span.status.description
        assert not span.events
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)
        fluent._active_experiment_id = original_experiment_id


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
            tracing.record_agent_decision(
                AgentDecision(
                    decision_type=AgentDecisionType.ANSWER_READINESS,
                    goal="Answer the request",
                    selected_action="answer",
                    reason="The provider response is ready to return.",
                    evidence_refs=("provider_response",),
                )
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
        assert span_names.count("decision.answer_readiness") == 1
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
            return await task

        with pytest.raises(asyncio.CancelledError):
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
