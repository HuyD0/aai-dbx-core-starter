"""Hermetic agent tests: scripted tool calls, structured final answer."""

from types import SimpleNamespace

from aai_core.agents import AgentRequest
from aai_core.testing import FakeChatModel, fake_tool_call
from app.agent import ToolAgent
from app.config import PROMPT_NAME


class FakePrompt:
    version = 1

    def format(self, **values):
        return [
            {"role": "system", "content": "Use tools when they help."},
            {"role": "user", "content": values["question"]},
        ]


class FakePrompts:
    def load(self, name, **kwargs):
        assert name == PROMPT_NAME
        return FakePrompt()


class FakeProviders:
    def __init__(self, model):
        self._model = model

    def model(self, name):
        assert name == "general-chat"
        return self._model


def _context(model):
    return SimpleNamespace(
        providers=FakeProviders(model),
        prompts=FakePrompts(),
        settings=SimpleNamespace(resource=SimpleNamespace(environment="dev")),
    )


def test_agent_uses_tools_and_returns_structured_answer():
    model = FakeChatModel(
        reply='{"answer": "Order A-1001 has shipped; ETA 2 days.", '
        '"confidence": 0.92, "tools_used": ["lookup_order_status"]}',
        tool_call_script=[
            [fake_tool_call("lookup_order_status", {"order_id": "A-1001"})],
            [],
        ],
    )

    response = ToolAgent(_context(model)).invoke(
        AgentRequest(messages=[{"role": "user", "content": "Where is A-1001?"}])
    )

    assert response.content == "Order A-1001 has shipped; ETA 2 days."
    assert response.metadata["tools_used"] == ["lookup_order_status"]
    assert response.metadata["confidence"] == 0.92
    # The structured call received the full tool transcript.
    final_request = model.requests[-1]
    assert final_request["response_format"]["type"] == "json_schema"
    assert any(m.get("role") == "tool" for m in final_request["messages"])


def test_agent_answers_directly_when_no_tool_needed():
    model = FakeChatModel(
        reply='{"answer": "I can help with orders.", "confidence": 0.8}',
    )

    response = ToolAgent(_context(model)).invoke(
        AgentRequest(messages=[{"role": "user", "content": "What can you do?"}])
    )

    assert response.metadata["tools_used"] == []
    assert response.content == "I can help with orders."
