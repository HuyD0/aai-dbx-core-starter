"""Tool boundaries: semantic-first routing, guards, budgets, evidence."""

import asyncio
import json
from pathlib import Path

import pytest

import app.tools as tools_module
from app.knowledge import KnowledgeRouter
from app.provenance import SourceTier
from app.tools import ProvenanceLog, ToolExecutionError, build_registry

ROOT = Path(__file__).resolve().parents[1]


def _registry(model, seed_executor):
    log = ProvenanceLog()
    registry = build_registry(
        model, KnowledgeRouter(ROOT / "knowledge"), seed_executor, log
    )
    return registry, log


def test_query_metrics_records_semantic_evidence(model, seed_executor):
    registry, log = _registry(model, seed_executor)
    payload = json.loads(
        asyncio.run(
            registry.execute(
                "query_metrics",
                {
                    "metrics": ["revenue"],
                    "filters": [
                        {
                            "dimension": "order_date",
                            "value": "2024-03",
                            "grain": "month",
                        }
                    ],
                },
            )
        )
    )
    assert payload["rows"] == [["600.75"]]
    assert payload["row_count"] == 1
    record = log.records[0]
    assert record.tier is SourceTier.SEMANTIC_LAYER
    assert record.value == "600.75"
    assert record.sql == payload["sql"]
    assert record.owner == "group:test-owners"


def test_query_metrics_returns_correctable_errors(model, seed_executor):
    registry, log = _registry(model, seed_executor)
    payload = json.loads(
        asyncio.run(registry.execute("query_metrics", {"metrics": ["profit"]}))
    )
    assert "unknown metrics" in payload["error"]
    assert "list_metrics" in payload["hint"]
    assert log.records == []


def test_result_rows_honor_the_context_budget(model, seed_executor, monkeypatch):
    monkeypatch.setattr(tools_module, "MAX_RESULT_ROWS_IN_CONTEXT", 2)
    registry, log = _registry(model, seed_executor)
    payload = json.loads(
        asyncio.run(
            registry.execute(
                "query_metrics",
                {"metrics": ["order_count"], "dimensions": ["order_status"]},
            )
        )
    )
    assert payload["row_count"] == 3
    assert len(payload["rows"]) == 2
    assert payload["truncated"] is True
    assert log.records[0].rows == 3


def test_lookup_reference_caps_doc_size_and_logs_tier(
    model, seed_executor, monkeypatch
):
    monkeypatch.setattr(tools_module, "MAX_REFERENCE_DOC_CHARS", 80)
    registry, log = _registry(model, seed_executor)
    payload = json.loads(
        asyncio.run(registry.execute("lookup_reference", {"topic": "orders"}))
    )
    assert payload["truncated"] is True
    assert len(payload["body"]) == 80
    assert log.records[0].tier is SourceTier.CURATED_REFERENCE
    assert log.records[0].sources == ("knowledge/orders.md",)


def test_query_rows_compiles_allowlisted_fields_and_logs_raw_tier(model, seed_executor):
    registry, log = _registry(model, seed_executor)
    payload = json.loads(
        asyncio.run(
            registry.execute(
                "query_rows",
                {
                    "source": "orders",
                    "fields": ["order_id", "order_amount"],
                    "filters": [
                        {"field": "order_status", "operator": "eq", "value": "C"}
                    ],
                    "order_by": [{"field": "order_amount", "direction": "desc"}],
                    "limit": 3,
                    "reason": "show cancelled row details",
                },
            )
        )
    )
    assert payload["rows"] == [["O-1004", "999.99"], ["O-1010", "75.00"]]
    assert ":r0" in payload["sql"]
    assert log.records[0].tier is SourceTier.RAW_TABLE
    assert log.records[0].sources == ("demo.sales.analytics_orders",)
    assert log.records[0].owner == "group:test-owners"

    unknown = json.loads(
        asyncio.run(
            registry.execute(
                "query_rows",
                {
                    "source": "orders",
                    "fields": ["password_hash"],
                    "reason": "attempt undeclared data",
                },
            )
        )
    )
    assert "unknown governed row field" in unknown["error"]
    assert len(log.records) == 1

    with pytest.raises(ToolExecutionError, match="schema validation"):
        asyncio.run(
            registry.execute(
                "query_rows",
                {
                    "source": "orders",
                    "fields": ["order_id"],
                    "sql": "DROP TABLE orders",
                    "reason": "attempt SQL injection",
                },
            )
        )
    assert len(log.records) == 1


def test_registry_never_exposes_a_model_facing_sql_tool(model, seed_executor):
    registry, _ = _registry(model, seed_executor)
    names = [item["function"]["name"] for item in registry.openai_tools()]

    assert "query_rows" in names
    assert "execute_sql" not in names


def test_check_freshness_notes_and_records_the_watermark(model, seed_executor):
    registry, log = _registry(model, seed_executor)
    payload = json.loads(
        asyncio.run(registry.execute("check_freshness", {"source": "orders"}))
    )
    assert payload["within_sla"] is False
    assert payload["loaded_at"] == "2024-05-01T06:00:00Z"
    assert log.records[0].tier is SourceTier.RAW_TABLE
    assert "OUTSIDE" in log.records[0].freshness
    assert "demo.sales.analytics_orders" in log.freshness_notes


def test_finalize_attaches_freshness_to_matching_records(model, seed_executor):
    registry, log = _registry(model, seed_executor)
    asyncio.run(registry.execute("query_metrics", {"metrics": ["revenue"]}))
    asyncio.run(registry.execute("check_freshness", {"source": "orders"}))
    finalized = log.finalize()
    semantic = finalized[0]
    assert semantic.tier is SourceTier.SEMANTIC_LAYER
    assert semantic.freshness is not None and "OUTSIDE" in semantic.freshness


def test_unknown_tools_and_bad_arguments_fail_closed(model, seed_executor):
    registry, _ = _registry(model, seed_executor)
    with pytest.raises(ToolExecutionError, match="unknown tool"):
        asyncio.run(registry.execute("drop_tables", {}))
    with pytest.raises(ToolExecutionError, match="schema validation"):
        asyncio.run(registry.execute("lookup_reference", {"topic": 7}))


def test_list_metrics_exposes_catalog_and_encodings(model, seed_executor):
    registry, _ = _registry(model, seed_executor)
    payload = json.loads(asyncio.run(registry.execute("list_metrics", {})))
    assert "revenue" in payload["catalog"]
    assert payload["dimension_encodings"]["order_status"]["S"] == "shipped"
