"""Shared fixtures: an inline semantic model and the snapshot executor.

The inline payload mirrors semantics/semantic_model.yml but is independent
of template rendering, so unit tests exercise the engine without a
generated catalog/schema. The seed snapshot is the same file both
executors and the offline gate share.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from app.semantics.executor import FakeWarehouseExecutor
from app.semantics.models import SemanticModel

ROOT = Path(__file__).resolve().parents[1]

MODEL_PAYLOAD = {
    "semantic_model": {"name": "sales", "version": 1},
    "sources": {
        "orders": {
            "table": "demo.sales.analytics_orders",
            "grain": "one row per order",
            "owner": "group:test-owners",
            "freshness_sla_hours": 24,
            "loaded_at_column": "_loaded_at",
            "gotchas": ["status uses letter codes"],
        },
        "customers": {
            "table": "demo.sales.analytics_customers",
            "grain": "one row per customer",
            "owner": "group:test-owners",
            "freshness_sla_hours": 168,
            "loaded_at_column": "_loaded_at",
        },
    },
    "dimensions": {
        "order_status": {
            "source": "orders",
            "column": "status",
            "type": "string",
            "encodings": {"S": "shipped", "P": "processing", "C": "cancelled"},
        },
        "order_date": {"source": "orders", "column": "order_date", "type": "date"},
        "region": {
            "source": "customers",
            "column": "region",
            "type": "string",
            "join": {"from_column": "customer_id", "to_column": "customer_id"},
        },
    },
    "detail_fields": {
        "order_id": {"source": "orders", "column": "order_id", "type": "string"},
        "customer_id": {
            "source": "orders",
            "column": "customer_id",
            "type": "string",
        },
        "order_date": {"source": "orders", "column": "order_date", "type": "date"},
        "order_status": {"source": "orders", "column": "status", "type": "string"},
        "order_amount": {"source": "orders", "column": "amount", "type": "number"},
        "customer_region": {
            "source": "customers",
            "column": "region",
            "type": "string",
        },
    },
    "metrics": {
        "revenue": {
            "source": "orders",
            "aggregation": "sum",
            "expr": "amount",
            "filter": {"column": "status", "op": "<>", "value": "C"},
            "description": "Net revenue excluding cancelled orders.",
        },
        "order_count": {
            "source": "orders",
            "aggregation": "count",
            "expr": "order_id",
            "description": "Count of orders, including cancelled ones.",
        },
        "average_order_value": {
            "source": "orders",
            "aggregation": "avg",
            "expr": "amount",
            "filter": {"column": "status", "op": "<>", "value": "C"},
            "description": "Mean order amount excluding cancelled orders.",
        },
        "active_customers": {
            "source": "orders",
            "aggregation": "count_distinct",
            "expr": "customer_id",
            "description": "Distinct customers with at least one order.",
        },
        "customer_count": {
            "source": "customers",
            "aggregation": "count",
            "expr": "customer_id",
            "description": "Registered customers.",
        },
    },
}


@pytest.fixture()
def model_payload() -> dict:
    return copy.deepcopy(MODEL_PAYLOAD)


@pytest.fixture()
def model() -> SemanticModel:
    return SemanticModel.model_validate(MODEL_PAYLOAD)


@pytest.fixture()
def seed_executor() -> FakeWarehouseExecutor:
    return FakeWarehouseExecutor(ROOT / "evals" / "data" / "seed_data.json")
