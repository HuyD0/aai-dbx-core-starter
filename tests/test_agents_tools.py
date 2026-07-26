"""Unit tests for the tool loop, structured output, and serving adapter."""

import json

import pytest

from aai_core.agents import ToolLoopError, ToolRegistry, ToolSpec, run_tool_loop
from aai_core.serving import ServingError, agent_resources
from aai_core.structured import StructuredOutputError, generate_structured
from aai_core.testing import FakeChatModel, dev_settings, fake_tool_call

LOOKUP = ToolSpec(
    name="lookup_order_status",
    description="Look up an order's status by id.",
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
    handler=lambda order_id: {"order_id": order_id, "status": "shipped"},
)


def _registry():
    registry = ToolRegistry()
    registry.register(LOOKUP)
    return registry


def test_tool_loop_executes_calls_and_returns_final_answer():
    model = FakeChatModel(
        reply="Order A-1 has shipped.",
        tool_call_script=[
            [fake_tool_call("lookup_order_status", {"order_id": "A-1"})],
            [],
        ],
    )

    result = run_tool_loop(
        model,
        [{"role": "user", "content": "Where is order A-1?"}],
        _registry(),
    )

    assert result.response.content == "Order A-1 has shipped."
    assert result.tool_names == ("lookup_order_status",)
    assert json.loads(result.tool_invocations[0].result)["status"] == "shipped"
    tool_message = next(m for m in model.requests[1]["messages"] if m["role"] == "tool")
    assert "shipped" in tool_message["content"]


def test_tool_loop_rejects_unknown_tools_and_runaway_loops():
    with pytest.raises(ToolLoopError, match="unknown tool"):
        run_tool_loop(
            FakeChatModel(tool_call_script=[[fake_tool_call("nope", {})]]),
            [{"role": "user", "content": "q"}],
            _registry(),
        )

    endless = FakeChatModel(
        tool_call_script=[[fake_tool_call("lookup_order_status", {"order_id": "A-1"})]]
        * 5
    )
    with pytest.raises(ToolLoopError, match="did not converge"):
        run_tool_loop(
            endless,
            [{"role": "user", "content": "q"}],
            _registry(),
            max_turns=3,
        )


def test_registry_rejects_duplicate_registration():
    registry = _registry()
    with pytest.raises(ToolLoopError, match="already registered"):
        registry.register(LOOKUP)


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["answer", "confidence"],
}


def test_generate_structured_parses_and_validates():
    model = FakeChatModel(reply='{"answer": "shipped", "confidence": 0.9}')

    parsed = generate_structured(
        model, [{"role": "user", "content": "q"}], json_schema=SCHEMA
    )

    assert parsed == {"answer": "shipped", "confidence": 0.9}
    request = model.requests[0]
    assert request["response_format"]["json_schema"]["schema"] == SCHEMA


def test_generate_structured_rejects_bad_json_and_missing_keys():
    with pytest.raises(StructuredOutputError, match="invalid JSON"):
        generate_structured(
            FakeChatModel(reply="not json"),
            [{"role": "user", "content": "q"}],
            json_schema=SCHEMA,
        )
    with pytest.raises(StructuredOutputError, match="missing required"):
        generate_structured(
            FakeChatModel(reply='{"answer": "x"}'),
            [{"role": "user", "content": "q"}],
            json_schema=SCHEMA,
        )


def test_agent_resources_maps_logical_names(monkeypatch):
    from conftest import install_fake_module

    captured = {}
    install_fake_module(
        monkeypatch,
        "mlflow.models.resources",
        DatabricksServingEndpoint=lambda endpoint_name: captured.setdefault(
            "endpoint", endpoint_name
        ),
        DatabricksVectorSearchIndex=lambda index_name: captured.setdefault(
            "index", index_name
        ),
        DatabricksFunction=lambda function_name: captured.setdefault(
            "function", function_name
        ),
    )
    settings = dev_settings(
        models={"general-chat": {"provider": "databricks", "deployment": "chat-ep"}},
        retrievers={
            "product-knowledge": {
                "provider": "databricks_ai_search",
                "index": "main.rag.knowledge",
            }
        },
    )

    resources = agent_resources(
        settings,
        models=["general-chat"],
        retrievers=["product-knowledge"],
        uc_functions=["main.tools.lookup"],
    )

    assert len(resources) == 3
    assert captured == {
        "endpoint": "chat-ep",
        "index": "main.rag.knowledge",
        "function": "main.tools.lookup",
    }


def test_agent_resources_rejects_non_databricks_models():
    settings = dev_settings(
        models={"general-chat": {"provider": "foundry", "deployment": "gpt"}}
    )

    with pytest.raises(ServingError, match="not a Databricks serving endpoint"):
        agent_resources(settings, models=["general-chat"])
