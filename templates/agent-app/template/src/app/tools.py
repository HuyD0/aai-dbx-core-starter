"""Agent tools.

Code tools (plain Python, default) run inside the agent process. For tools
over governed structured data, prefer Unity Catalog FUNCTIONS: governed,
audited, and declared as serving resources — see `uc_function_tools` below.
"""

from __future__ import annotations

from aai_core.agents import ToolRegistry, ToolSpec

# Demo data — replace with your real backend call.
_ORDERS = {
    "A-1001": {"status": "shipped", "eta_days": 2},
    "A-1002": {"status": "processing", "eta_days": 5},
}


def lookup_order_status(order_id: str) -> dict:
    """Code tool: look up an order in the fulfillment system."""

    order = _ORDERS.get(order_id)
    if order is None:
        return {"order_id": order_id, "error": "order not found"}
    return {"order_id": order_id, **order}


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="lookup_order_status",
            description="Look up an order's fulfillment status by order id.",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order identifier, e.g. A-1001",
                    }
                },
                "required": ["order_id"],
            },
            handler=lookup_order_status,
        )
    )
    return registry


# Unity Catalog function tools (governed structured-data lookups): create the
# function in UC (human-run bootstrap), list it here, and pass it to
# aai_core.serving.agent_resources(uc_functions=...) at deploy time so the
# serving endpoint gets authenticated access. Execution goes through the
# databricks-openai UCFunctionToolkit — see the deployment README section.
UC_FUNCTION_TOOLS: tuple[str, ...] = ()
