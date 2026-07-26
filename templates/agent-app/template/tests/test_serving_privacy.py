"""Serving-boundary tests for MLflow's automatic ResponsesAgent trace."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def served_agent(monkeypatch):
    """Import serving/model.py against a tiny ResponsesAgent trace simulator."""

    active_span = {"value": None}
    recorded_spans = []
    registered_model = {}

    class FakeSpan:
        def __init__(self, inputs):
            self.inputs = inputs

        def set_inputs(self, inputs):
            self.inputs = inputs

    class FakeResponsesAgent:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            original = cls.__dict__.get("predict")
            if original is None:
                return

            @wraps(original)
            def auto_traced(self, request):
                # This is the privacy-sensitive behavior in MLflow 3.14:
                # ResponsesAgent traces the complete request before entering
                # the application method.
                span = FakeSpan({"request": request})
                recorded_spans.append(span)
                active_span["value"] = span
                try:
                    return original(self, request)
                finally:
                    active_span["value"] = None

            cls.predict = auto_traced

        @staticmethod
        def create_text_output_item(*, text, id):
            return {"type": "message", "id": id, "content": text}

    class FakeModelConfig:
        def __init__(self, **kwargs):
            pass

        def get(self, section):
            raise KeyError(section)

    class FakeResponsesAgentResponse:
        def __init__(self, *, output, custom_outputs):
            self.output = output
            self.custom_outputs = custom_outputs

    @dataclass
    class FakeAgentRequest:
        messages: list[dict]
        session_id: str | None = None

    class FakeToolAgent:
        def __init__(self, context):
            self.request = None

        def invoke(self, request):
            self.request = request
            return SimpleNamespace(content="answer", metadata={"safe": True})

    models = ModuleType("mlflow.models")
    models.ModelConfig = FakeModelConfig
    models.set_model = lambda model: registered_model.update(model=model)
    pyfunc = ModuleType("mlflow.pyfunc")
    pyfunc.ResponsesAgent = FakeResponsesAgent
    responses = ModuleType("mlflow.types.responses")
    responses.ResponsesAgentRequest = object
    responses.ResponsesAgentResponse = FakeResponsesAgentResponse
    mlflow_types = ModuleType("mlflow.types")
    mlflow_types.responses = responses
    mlflow = ModuleType("mlflow")
    mlflow.models = models
    mlflow.pyfunc = pyfunc
    mlflow.types = mlflow_types
    mlflow.get_current_active_span = lambda: active_span["value"]

    aai_core = ModuleType("aai_core")
    aai_core.PlatformContext = object
    aai_core.bootstrap = lambda path: object()
    aai_agents = ModuleType("aai_core.agents")
    aai_agents.AgentRequest = FakeAgentRequest
    fake_agent_module = ModuleType("app.agent")
    fake_agent_module.ToolAgent = FakeToolAgent

    for name, module in {
        "mlflow": mlflow,
        "mlflow.models": models,
        "mlflow.pyfunc": pyfunc,
        "mlflow.types": mlflow_types,
        "mlflow.types.responses": responses,
        "aai_core": aai_core,
        "aai_core.agents": aai_agents,
        "app.agent": fake_agent_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "serving_model_privacy_test",
        ROOT / "serving" / "model.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return registered_model["model"], recorded_spans


def test_serving_overwrites_automatic_trace_with_sanitized_inputs(served_agent):
    model, spans = served_agent
    request = SimpleNamespace(
        input=[SimpleNamespace(role="user", content="Where is my order?")],
        context=SimpleNamespace(
            conversation_id="opaque-conversation",
            user_id="personal-user-id",
        ),
    )

    model.predict(request)

    assert spans[-1].inputs == {
        "input": [{"role": "user", "content": "Where is my order?"}],
        "context": {"conversation_id": "opaque-conversation"},
    }
    assert "personal-user-id" not in repr(spans[-1].inputs)
    assert model._agent.request.messages == [
        {"role": "user", "content": "Where is my order?"}
    ]
    assert model._agent.request.session_id == "opaque-conversation"


def test_failed_normalization_still_scrubs_automatic_trace(served_agent):
    model, spans = served_agent
    request = SimpleNamespace(
        input=[
            SimpleNamespace(
                role="user",
                content=[
                    {
                        "type": "input_image",
                        "image_url": "https://private.invalid/image",
                    }
                ],
            )
        ],
        context=SimpleNamespace(user_id="personal-user-id"),
    )

    with pytest.raises(ValueError, match="text-only"):
        model.predict(request)

    assert spans[-1].inputs == {"input": []}
    assert "personal-user-id" not in repr(spans[-1].inputs)
    assert "private.invalid" not in repr(spans[-1].inputs)
