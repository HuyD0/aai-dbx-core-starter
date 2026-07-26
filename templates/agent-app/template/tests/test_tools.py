"""Tool unit tests — pure functions, zero cloud."""

import json

from app.tools import build_registry, lookup_order_status


def test_lookup_returns_status_or_explicit_not_found():
    assert lookup_order_status("A-1001")["status"] == "shipped"
    assert lookup_order_status("missing")["error"] == "order not found"


def test_registry_exposes_openai_tool_metadata_and_executes():
    registry = build_registry()

    tools = registry.openai_tools()
    assert tools[0]["function"]["name"] == "lookup_order_status"
    assert "order_id" in tools[0]["function"]["parameters"]["properties"]

    result = json.loads(registry.execute("lookup_order_status", {"order_id": "A-1002"}))
    assert result["status"] == "processing"
