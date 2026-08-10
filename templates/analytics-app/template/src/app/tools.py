"""Application-owned async tools implementing the runbook's search order.

query_metrics is the aggregate path and query_rows is the governed row-level
fallback. Both accept constrained plans, never SQL. Every tool feeds the
ProvenanceLog so the agent can render an evidence footer the model cannot
fabricate. Tool outputs are context-budgeted: row sets truncate to
MAX_RESULT_ROWS_IN_CONTEXT (with the true row count stated) and reference
docs cap at MAX_REFERENCE_DOC_CHARS.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aai_core.exceptions import AaiCoreError
from aai_core.tracing import provider_span
from app.config import MAX_REFERENCE_DOC_CHARS, MAX_RESULT_ROWS_IN_CONTEXT
from app.controls import DEFAULT_ANALYTICS_LIMITS, AnalyticsLimits
from app.knowledge import KnowledgeRouter
from app.provenance import ProvenanceRecord, SourceTier
from app.semantics.compiler import (
    OrderDirection,
    QueryFilter,
    RowFilter,
    RowOperator,
    RowOrder,
    RowQuery,
    SemanticCompileError,
    SemanticQuery,
    TimeGrain,
    compile_query,
    compile_rows,
)
from app.semantics.executor import (
    AsyncWarehouseExecutor,
    QueryResult,
    WarehouseExecutionError,
    WarehouseExecutor,
)
from app.semantics.models import SemanticModel


class ToolExecutionError(AaiCoreError):
    code = "app.tool_execution"


@dataclass
class ProvenanceLog:
    """Per-request evidence collector; the footer is rendered from this."""

    records: list[ProvenanceRecord] = field(default_factory=list)
    freshness_notes: dict[str, str] = field(default_factory=dict)

    def add(self, record: ProvenanceRecord) -> None:
        self.records.append(record)

    def note_freshness(self, table: str, note: str) -> None:
        self.freshness_notes[table] = note

    def finalize(self) -> tuple[ProvenanceRecord, ...]:
        """Attach recorded freshness to the records that read those sources."""

        finalized = []
        for record in self.records:
            notes = [
                self.freshness_notes[source]
                for source in record.sources
                if source in self.freshness_notes
            ]
            if notes and record.freshness is None:
                record = record.model_copy(update={"freshness": "; ".join(notes)})
            finalized.append(record)
        return tuple(finalized)


class ListMetricsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FilterInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dimension: str = Field(description="Dimension name from the semantic model")
    value: str = Field(description="Value to match; ISO date for time filters")
    grain: str | None = Field(
        default=None,
        description="Optional time grain (day/week/month/quarter/year) for "
        "date dimensions",
    )


class QueryMetricsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    metrics: list[str] = Field(description="Metric names to aggregate")
    dimensions: list[str] = Field(
        default_factory=list, description="Dimensions to group by"
    )
    filters: list[FilterInput] = Field(
        default_factory=list, description="Equality or time-bucket filters"
    )
    time_dimension: str | None = Field(
        default=None, description="Date dimension to bucket the trend by"
    )
    time_grain: str | None = Field(
        default=None, description="Bucket size for time_dimension"
    )
    limit: int = Field(default=100, ge=1, le=100, description="Maximum rows")


class LookupReferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    topic: str = Field(description="Knowledge topic from the index")


ScalarInput = str | int | float | bool


class RowFilterInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    field: str = Field(description="Governed detail field name")
    operator: Literal["eq", "ne", "lt", "lte", "gt", "gte", "in", "is_null"]
    value: ScalarInput | list[ScalarInput] | None = Field(
        default=None,
        description="Typed value; use a list only with the in operator",
    )


class RowOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    field: str = Field(description="Governed detail field name")
    direction: Literal["asc", "desc"] = "asc"


class QueryRowsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source: str = Field(description="Source name from the semantic model")
    fields: list[str] = Field(min_length=1, max_length=20)
    filters: list[RowFilterInput] = Field(default_factory=list, max_length=20)
    order_by: list[RowOrderInput] = Field(default_factory=list, max_length=5)
    limit: int = Field(default=100, ge=1, le=100)
    reason: str = Field(
        min_length=3,
        max_length=500,
        description="Why row-level detail is required instead of a metric",
    )


class CheckFreshnessInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source: str = Field(description="Semantic model source name")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[..., Awaitable[Any]]
    timeout_seconds: float = 180.0
    max_output_chars: int = 32 * 1024

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


class AsyncToolRegistry:
    def __init__(self, specs: tuple[ToolSpec, ...]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Tool names must be unique")
        for spec in specs:
            if spec.input_model.model_config.get("extra") != "forbid":
                raise ValueError(
                    f"Tool input model {spec.input_model.__name__!r} must forbid extras"
                )
            if not inspect.iscoroutinefunction(spec.handler):
                raise TypeError(f"Tool handler {spec.name!r} must be async")

    def openai_tools(self) -> list[dict[str, Any]]:
        return [spec.as_openai_tool() for spec in self._specs.values()]

    async def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        try:
            spec = self._specs[name]
        except KeyError as error:
            raise ToolExecutionError(
                f"Model requested unknown tool {name!r}"
            ) from error
        try:
            inputs = spec.input_model.model_validate(dict(arguments), strict=True)
        except ValidationError as error:
            raise ToolExecutionError(
                f"Arguments for tool {name!r} failed schema validation"
            ) from error
        validated = inputs.model_dump(mode="python")
        with provider_span(
            name,
            span_type="TOOL",
            attributes={"gen_ai.tool.name": name},
        ) as span:
            if span is not None:
                span.set_inputs(validated)
            try:
                result = await asyncio.wait_for(
                    spec.handler(**validated),
                    timeout=spec.timeout_seconds,
                )
            except TimeoutError as error:
                raise ToolExecutionError(
                    f"Tool {name!r} exceeded its {spec.timeout_seconds:g}s timeout"
                ) from error
            except ToolExecutionError:
                raise
            except Exception as error:
                raise ToolExecutionError(f"Tool {name!r} failed") from error
            serialized = result if isinstance(result, str) else json.dumps(result)
            if len(serialized) > spec.max_output_chars:
                raise ToolExecutionError(
                    f"Tool {name!r} output exceeded its "
                    f"{spec.max_output_chars}-character bound"
                )
            if span is not None:
                span.set_outputs(serialized)
            return serialized


def build_analytics_registry(
    model: SemanticModel,
    knowledge: KnowledgeRouter,
    executor: WarehouseExecutor,
    log: ProvenanceLog,
    limits: AnalyticsLimits = DEFAULT_ANALYTICS_LIMITS,
) -> AsyncToolRegistry:
    """Bind the analytics tools to this request's model, docs, and executor."""

    return AsyncToolRegistry(
        (
            ToolSpec(
                name="list_metrics",
                description="List the governed metric catalog, available "
                "dimensions, and dimension value encodings. Cheap; call first.",
                input_model=ListMetricsInput,
                handler=_list_metrics_handler(model),
                timeout_seconds=10.0,
                max_output_chars=limits.max_tool_output_chars,
            ),
            ToolSpec(
                name="query_metrics",
                description="Answer through the governed semantic layer: "
                "aggregate declared metrics by declared dimensions with "
                "structured filters. This is the required first query path.",
                input_model=QueryMetricsInput,
                handler=_query_metrics_handler(
                    model, executor, log, limits.max_result_rows
                ),
                timeout_seconds=limits.tool_timeout_seconds,
                max_output_chars=limits.max_tool_output_chars,
            ),
            ToolSpec(
                name="lookup_reference",
                description="Load one curated reference doc (grain, scope, "
                "encodings, gotchas, patterns) by topic from the knowledge "
                "index in the system prompt.",
                input_model=LookupReferenceInput,
                handler=_lookup_reference_handler(knowledge, log),
                timeout_seconds=10.0,
                max_output_chars=limits.max_tool_output_chars,
            ),
            ToolSpec(
                name="query_rows",
                description="Governed row-level fallback. Select only declared "
                "sources and fields with typed filters and bounded ordering; "
                "SQL is compiled by the application and cannot be supplied.",
                input_model=QueryRowsInput,
                handler=_query_rows_handler(
                    model, executor, log, limits.max_result_rows
                ),
                timeout_seconds=limits.tool_timeout_seconds,
                max_output_chars=limits.max_tool_output_chars,
            ),
            ToolSpec(
                name="check_freshness",
                description="Check a source table's loaded_at watermark "
                "against its freshness SLA; cite the result when staleness "
                "would change the answer.",
                input_model=CheckFreshnessInput,
                handler=_freshness_handler(model, executor, log),
                timeout_seconds=limits.tool_timeout_seconds,
                max_output_chars=limits.max_tool_output_chars,
            ),
        )
    )


def _list_metrics_handler(model: SemanticModel) -> Callable[..., Awaitable[Any]]:
    async def list_metrics() -> dict[str, Any]:
        encodings = {
            name: dict(dimension.encodings)
            for name, dimension in model.dimensions.items()
            if dimension.encodings
        }
        return {
            "catalog": model.metric_catalog(),
            "dimension_encodings": encodings,
        }

    return list_metrics


def _query_metrics_handler(
    model: SemanticModel,
    executor: WarehouseExecutor,
    log: ProvenanceLog,
    max_result_rows: int,
) -> Callable[..., Awaitable[Any]]:

    async def query_metrics(**arguments: Any) -> dict[str, Any]:
        try:
            query = _semantic_query(arguments, max_result_rows)
            compiled = compile_query(model, query, executor.dialect)
            result = await _run_plan(executor, model, query)
        except (SemanticCompileError, ValidationError, ValueError, KeyError) as error:
            return {
                "error": str(error),
                "hint": "call list_metrics for valid metric and dimension names",
            }
        except WarehouseExecutionError as error:
            return {"error": str(error)}
        owners = sorted(
            {model.sources[model.metrics[name].source].owner for name in query.metrics}
        )
        log.add(
            ProvenanceRecord(
                tier=SourceTier.SEMANTIC_LAYER,
                sources=compiled.sources,
                owner=", ".join(owners),
                rows=len(result.rows),
                value=result.scalar,
                sql=result.sql,
            )
        )
        return _bounded_result(result)

    return query_metrics


def _lookup_reference_handler(
    knowledge: KnowledgeRouter,
    log: ProvenanceLog,
) -> Callable[..., Awaitable[Any]]:

    async def lookup_reference(topic: str) -> dict[str, Any]:
        try:
            doc = knowledge.load(topic)
        except KeyError as error:
            return {"error": str(error)}
        body = doc.body
        truncated = len(body) > MAX_REFERENCE_DOC_CHARS
        if truncated:
            body = body[:MAX_REFERENCE_DOC_CHARS]
        log.add(
            ProvenanceRecord(
                tier=SourceTier.CURATED_REFERENCE,
                sources=(f"knowledge/{doc.topic}.md",),
            )
        )
        return {"title": doc.title, "body": body, "truncated": truncated}

    return lookup_reference


def _query_rows_handler(
    model: SemanticModel,
    executor: WarehouseExecutor,
    log: ProvenanceLog,
    max_result_rows: int,
) -> Callable[..., Awaitable[Any]]:

    async def query_rows(**arguments: Any) -> dict[str, Any]:
        try:
            query = _row_query(arguments, max_result_rows)
            compiled = compile_rows(model, query, executor.dialect)
            result = await _run_rows(executor, model, query)
        except (SemanticCompileError, ValidationError, ValueError, KeyError) as error:
            return {
                "error": str(error),
                "hint": "call list_metrics for governed sources and row fields",
            }
        except WarehouseExecutionError as error:
            return {"error": str(error)}
        log.add(
            ProvenanceRecord(
                tier=SourceTier.RAW_TABLE,
                sources=compiled.sources,
                owner=model.sources[query.source].owner,
                rows=len(result.rows),
                value=result.scalar,
                sql=result.sql,
            )
        )
        payload = _bounded_result(result)
        payload["reason"] = query.reason
        return payload

    return query_rows


def _freshness_handler(
    model: SemanticModel,
    executor: WarehouseExecutor,
    log: ProvenanceLog,
) -> Callable[..., Awaitable[Any]]:

    async def check_freshness(source: str) -> dict[str, Any]:
        table = model.sources.get(source)
        if table is None:
            known = ", ".join(sorted(model.sources))
            return {"error": f"unknown source {source!r}; known: {known}"}
        if table.loaded_at_column is None:
            return {
                "source": source,
                "table": table.table,
                "loaded_at": None,
                "note": "source declares no loaded_at_column",
            }
        loaded_at = await _latest_loaded_at(executor, model, source)
        within = _within_sla(loaded_at, table.freshness_sla_hours)
        status = "within" if within else "OUTSIDE"
        note = f"loaded {loaded_at} ({status} the {table.freshness_sla_hours}h SLA)"
        log.note_freshness(table.table, note)
        log.add(
            ProvenanceRecord(
                tier=SourceTier.RAW_TABLE,
                sources=(table.table,),
                freshness=note,
                value=loaded_at,
            )
        )
        return {
            "source": source,
            "table": table.table,
            "loaded_at": loaded_at,
            "sla_hours": table.freshness_sla_hours,
            "within_sla": within,
        }

    return check_freshness


def _semantic_query(
    arguments: Mapping[str, Any], max_result_rows: int = 100
) -> SemanticQuery:
    filters = tuple(
        QueryFilter(
            dimension=item["dimension"],
            value=item["value"],
            grain=TimeGrain(item["grain"]) if item.get("grain") else None,
        )
        for item in arguments.get("filters", ())
    )
    grain = arguments.get("time_grain")
    limit = int(arguments.get("limit", max_result_rows))
    if limit > max_result_rows:
        raise SemanticCompileError(
            f"query limit exceeds the configured {max_result_rows}-row bound"
        )
    return SemanticQuery(
        metrics=tuple(arguments.get("metrics", ())),
        dimensions=tuple(arguments.get("dimensions", ())),
        filters=filters,
        time_dimension=arguments.get("time_dimension"),
        time_grain=TimeGrain(grain) if grain else None,
        limit=limit,
    )


def _row_query(arguments: Mapping[str, Any], max_result_rows: int = 100) -> RowQuery:
    filters = tuple(
        RowFilter(
            field=item["field"],
            operator=RowOperator(item["operator"]),
            value=(
                tuple(item["value"])
                if isinstance(item.get("value"), list)
                else item.get("value")
            ),
        )
        for item in arguments.get("filters", ())
    )
    orders = tuple(
        RowOrder(
            field=item["field"],
            direction=OrderDirection(item.get("direction", "asc")),
        )
        for item in arguments.get("order_by", ())
    )
    limit = int(arguments.get("limit", max_result_rows))
    if limit > max_result_rows:
        raise SemanticCompileError(
            f"query limit exceeds the configured {max_result_rows}-row bound"
        )
    return RowQuery(
        source=arguments["source"],
        fields=tuple(arguments["fields"]),
        filters=filters,
        order_by=orders,
        limit=limit,
        reason=arguments["reason"],
    )


def _bounded_result(result: QueryResult) -> dict[str, Any]:
    rows = [list(row) for row in result.rows[:MAX_RESULT_ROWS_IN_CONTEXT]]
    return {
        "columns": list(result.columns),
        "rows": rows,
        "row_count": len(result.rows),
        "truncated": len(result.rows) > len(rows),
        "sql": result.sql,
    }


async def _run_plan(
    executor: WarehouseExecutor,
    model: SemanticModel,
    query: SemanticQuery,
) -> QueryResult:
    if isinstance(executor, AsyncWarehouseExecutor):
        return await executor.arun_plan(model, query)
    return await asyncio.to_thread(executor.run_plan, model, query)


async def _run_rows(
    executor: WarehouseExecutor,
    model: SemanticModel,
    query: RowQuery,
) -> QueryResult:
    if isinstance(executor, AsyncWarehouseExecutor):
        return await executor.aquery_rows(model, query)
    return await asyncio.to_thread(executor.query_rows, model, query)


async def _latest_loaded_at(
    executor: WarehouseExecutor,
    model: SemanticModel,
    source: str,
) -> str | None:
    if isinstance(executor, AsyncWarehouseExecutor):
        return await executor.alatest_loaded_at(model, source)
    return await asyncio.to_thread(executor.latest_loaded_at, model, source)


def _within_sla(loaded_at: str | None, sla_hours: int) -> bool:
    if not loaded_at:
        return False
    normalized = loaded_at.strip().replace(" ", "T")
    try:
        stamp = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.UTC)
    age = dt.datetime.now(dt.UTC) - stamp
    return age <= dt.timedelta(hours=sla_hours)
