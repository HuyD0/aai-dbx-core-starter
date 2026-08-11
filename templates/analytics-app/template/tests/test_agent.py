"""Hermetic agent-loop tests over an OpenAI-shaped scripted client."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_core.testing import dev_context, fake_tool_call
from app.agent import AnalyticsAgent
from app.controls import DEFAULT_ANALYTICS_LIMITS, AnalyticsLimits
from app.knowledge import KnowledgeRouter

ROOT = Path(__file__).resolve().parents[1]

MARCH_PLAN = {
    "metrics": ["revenue"],
    "filters": [{"dimension": "order_date", "value": "2024-03", "grain": "month"}],
}
FINAL = json.dumps(
    {
        "answer": "Revenue for March 2024 was 600.75.",
        "caveats": ["Excludes cancelled orders."],
    }
)


class ScriptedAsyncClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)

    async def aclose(self):
        self.closed = True


def _response(content="", tool_calls=(), tokens=0):
    message = SimpleNamespace(content=content, tool_calls=list(tool_calls))
    usage = (
        SimpleNamespace(prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens)
        if tokens
        else None
    )
    return SimpleNamespace(
        model="scripted", choices=[SimpleNamespace(message=message)], usage=usage
    )


def _system_messages():
    payload = json.loads((ROOT / "prompts" / "system_prompt.json").read_text("utf-8"))
    return tuple(dict(message) for message in payload["messages"])


def _reviewer_messages():
    payload = json.loads((ROOT / "prompts" / "reviewer_prompt.json").read_text("utf-8"))
    return tuple(dict(message) for message in payload["messages"])


def _agent(model, seed_executor, client, **overrides):
    context = dev_context()
    context.providers.register_model(
        "general-chat",
        SimpleNamespace(provider="fake", logical_name="general-chat", model="fake"),
    )
    return AnalyticsAgent(
        context,
        semantic_model=model,
        knowledge=KnowledgeRouter(ROOT / "knowledge"),
        executor=seed_executor,
        system_messages=_system_messages(),
        reviewer_messages=_reviewer_messages(),
        enable_review=overrides.pop("enable_review", False),
        async_client=client,
        **overrides,
    )


def test_agent_answers_with_code_rendered_footer(model, seed_executor):
    client = ScriptedAsyncClient(
        [
            _response(
                tool_calls=[fake_tool_call("query_metrics", MARCH_PLAN)], tokens=900
            ),
            _response(tokens=400),
            _response(content=FINAL, tokens=250),
        ]
    )
    agent = _agent(model, seed_executor, client)

    answer = asyncio.run(agent.aanswer("What was revenue in March 2024?"))

    assert answer.prose.startswith("Revenue for March 2024 was 600.75.")
    assert "Caveats: Excludes cancelled orders." in answer.prose
    assert "[provenance]" in answer.answer
    assert answer.records[0].tier.value == "semantic_layer"
    assert answer.records[0].value == "600.75"
    assert answer.tools_used == ("query_metrics",)
    assert answer.usage.total_tokens == 1550
    assert answer.usage.review_tokens == 0
    assert answer.reviewed is False
    # The runbook system prompt carries the index, never the corpus.
    system = client.requests[0]["messages"][0]["content"]
    assert "RUNBOOK" in system
    assert "orders" in system
    assert all(request["max_tokens"] == 2048 for request in client.requests)


def test_agent_review_pass_replaces_prose_and_counts_tokens(model, seed_executor):
    verdict = json.dumps(
        {
            "approved": False,
            "revised_answer": "Revenue for March 2024 was 600.75 "
            "(excluding cancelled orders).",
            "objections": ["Caveat was missing"],
        }
    )
    client = ScriptedAsyncClient(
        [
            _response(
                tool_calls=[fake_tool_call("query_metrics", MARCH_PLAN)], tokens=900
            ),
            _response(tokens=400),
            _response(content=FINAL, tokens=250),
            _response(content=verdict, tokens=500),
        ]
    )
    agent = _agent(model, seed_executor, client, enable_review=True)

    answer = asyncio.run(agent.aanswer("What was revenue in March 2024?"))

    assert answer.reviewed is True
    assert answer.prose.endswith("(excluding cancelled orders).")
    assert answer.usage.review_tokens == 500
    assert answer.usage.total_tokens == 2050
    # The reviewer sees the evidence footer, not just the prose.
    review_request = client.requests[-1]["messages"][-1]["content"]
    assert "[provenance]" in review_request


def test_agent_without_tool_calls_omits_the_footer(model, seed_executor):
    refusal = json.dumps({"answer": "Out of scope.", "caveats": []})
    client = ScriptedAsyncClient(
        [_response(tokens=100), _response(content=refusal, tokens=50)]
    )
    agent = _agent(model, seed_executor, client)

    answer = asyncio.run(agent.aanswer("What's the weather?"))

    assert "[provenance]" not in answer.answer
    assert answer.records == ()


def test_agent_bounds_the_tool_loop(model, seed_executor):
    calls = [
        _response(tool_calls=[fake_tool_call("list_metrics", {})])
        for _ in range(DEFAULT_ANALYTICS_LIMITS.max_agent_turns)
    ]
    agent = _agent(model, seed_executor, ScriptedAsyncClient(calls))

    with pytest.raises(RuntimeError, match="did not converge"):
        asyncio.run(agent.aanswer("Loop forever"))


def test_agent_rejects_empty_questions(model, seed_executor):
    agent = _agent(model, seed_executor, ScriptedAsyncClient([]))
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(agent.aanswer("   "))


def test_agent_rejects_oversized_questions_before_a_model_call(model, seed_executor):
    client = ScriptedAsyncClient([])
    limits = AnalyticsLimits(max_question_chars=5)
    agent = _agent(model, seed_executor, client, limits=limits)

    with pytest.raises(ValueError, match="character bound"):
        asyncio.run(agent.aanswer("123456"))
    assert client.requests == []


def test_agent_bounds_tool_calls_per_turn(model, seed_executor):
    limits = AnalyticsLimits(max_tool_calls_per_turn=1)
    response = _response(
        tool_calls=[
            fake_tool_call("list_metrics", {}),
            fake_tool_call("list_metrics", {}),
        ]
    )
    agent = _agent(
        model,
        seed_executor,
        ScriptedAsyncClient([response]),
        limits=limits,
    )

    with pytest.raises(RuntimeError, match="tools in one turn"):
        asyncio.run(agent.aanswer("Use too many tools"))


def test_agent_closes_only_clients_it_owns(model, seed_executor):
    client = ScriptedAsyncClient([])
    agent = _agent(model, seed_executor, client)
    asyncio.run(agent.aclose())
    assert client.closed is False
