"""Compile constrained semantic query plans into portable SQL.

The agent's semantic-first path never writes SQL: it emits a ``SemanticQuery``
plan (metrics, dimensions, structured filters, an optional time grain) and
this pure-function compiler renders deterministic SQL for the executor's
dialect. The grammar is deliberately narrow — standard aggregations,
equality/time-bucket filters, ``DATE_TRUNC``, ``GROUP BY``, ``LIMIT`` — which
is what keeps it portable across warehouses and testable offline.
"""

from __future__ import annotations

import datetime as dt
import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.semantics.models import Aggregation, Join, SemanticModel

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_{}.-]+$")
_YEAR = re.compile(r"^\d{4}$")
_YEAR_MONTH = re.compile(r"^\d{4}-\d{2}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class TimeGrain(StrEnum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class Dialect(StrEnum):
    # Snowflake or another warehouse is added by implementing a
    # WarehouseExecutor with its own dialect value; the compiled grammar is
    # already portable (DATE_TRUNC and standard aggregates).
    DATABRICKS = "databricks"


class QueryFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: str = Field(min_length=1)
    value: str = Field(min_length=1)
    grain: TimeGrain | None = None


class SemanticQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metrics: tuple[str, ...] = Field(min_length=1)
    dimensions: tuple[str, ...] = ()
    filters: tuple[QueryFilter, ...] = ()
    time_dimension: str | None = None
    time_grain: TimeGrain | None = None
    limit: int = Field(default=100, ge=1, le=1000)


class QueryParameter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    type: Literal["STRING", "DATE"]


class CompiledQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sql: str = Field(min_length=1)
    parameters: tuple[QueryParameter, ...] = ()
    sources: tuple[str, ...] = Field(min_length=1)


class SemanticCompileError(ValueError):
    """The plan does not resolve against the semantic model."""


def truncate_date(grain: TimeGrain, iso_date: str) -> str:
    """Python twin of SQL DATE_TRUNC, shared with the offline executor."""

    day = dt.date.fromisoformat(iso_date)
    if grain is TimeGrain.DAY:
        result = day
    elif grain is TimeGrain.WEEK:
        result = day - dt.timedelta(days=day.weekday())
    elif grain is TimeGrain.MONTH:
        result = day.replace(day=1)
    elif grain is TimeGrain.QUARTER:
        result = day.replace(month=((day.month - 1) // 3) * 3 + 1, day=1)
    else:
        result = day.replace(month=1, day=1)
    return result.isoformat()


def normalize_time_value(grain: TimeGrain, value: str) -> str:
    """Accept the shorthand a model naturally produces for a period."""

    if grain is TimeGrain.YEAR and _YEAR.match(value):
        value = f"{value}-01-01"
    elif grain is TimeGrain.MONTH and _YEAR_MONTH.match(value):
        value = f"{value}-01"
    if not _ISO_DATE.match(value):
        raise SemanticCompileError(
            f"time filter value {value!r} must be an ISO date for grain "
            f"{grain.value}"
        )
    return truncate_date(grain, value)


def compile_query(
    model: SemanticModel,
    query: SemanticQuery,
    dialect: Dialect = Dialect.DATABRICKS,
) -> CompiledQuery:
    base_source = _resolve_base_source(model, query)
    joined = _resolve_joined_sources(model, query, base_source)
    qualify = bool(joined)

    select_items: list[str] = []
    group_items = 0
    if query.time_dimension is not None:
        dimension = model.dimensions[query.time_dimension]
        column = _column_reference(dimension.source, dimension.column, qualify)
        grain = query.time_grain
        assert grain is not None  # enforced by _resolve_base_source
        alias = f"{query.time_dimension}_{grain.value}"
        select_items.append(
            f"DATE_TRUNC('{grain.value.upper()}', {column}) AS {_quote(alias)}"
        )
        group_items += 1
    for name in query.dimensions:
        dimension = model.dimensions[name]
        column = _column_reference(dimension.source, dimension.column, qualify)
        select_items.append(f"{column} AS {_quote(name)}")
        group_items += 1
    for name in query.metrics:
        select_items.append(f"{_aggregate(model, name, qualify)} AS {_quote(name)}")

    sql = "SELECT " + ", ".join(select_items)
    sql += f" FROM {_table_reference(model, base_source, qualify)}"
    sources = [model.sources[base_source].table]
    for source_name, join in joined:
        left = _column_reference(base_source, join.from_column, qualify)
        right = _column_reference(source_name, join.to_column, qualify)
        sql += (
            f" JOIN {_table_reference(model, source_name, qualify)}"
            f" ON {left} = {right}"
        )
        sources.append(model.sources[source_name].table)

    clauses, parameters = _filter_clauses(model, query, qualify)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if group_items:
        ordinals = ", ".join(str(index + 1) for index in range(group_items))
        sql += f" GROUP BY {ordinals} ORDER BY 1"
    sql += f" LIMIT {query.limit}"
    return CompiledQuery(sql=sql, parameters=tuple(parameters), sources=tuple(sources))


def _resolve_base_source(model: SemanticModel, query: SemanticQuery) -> str:
    unknown = [name for name in query.metrics if name not in model.metrics]
    if unknown:
        raise SemanticCompileError(f"unknown metrics: {unknown}")
    sources = {model.metrics[name].source for name in query.metrics}
    if len(sources) != 1:
        raise SemanticCompileError(
            "all metrics in one plan must share a source; split the question"
        )
    if (query.time_dimension is None) != (query.time_grain is None):
        raise SemanticCompileError(
            "time_dimension and time_grain must be provided together"
        )
    if query.time_dimension is not None:
        if query.time_dimension in query.dimensions:
            raise SemanticCompileError(
                "time_dimension must not also be listed in dimensions"
            )
        _require_date_dimension(model, query.time_dimension)
    for name in query.dimensions:
        if name not in model.dimensions:
            raise SemanticCompileError(f"unknown dimension {name!r}")
    return next(iter(sources))


def _resolve_joined_sources(
    model: SemanticModel, query: SemanticQuery, base_source: str
) -> list[tuple[str, Join]]:
    joined: dict[str, Join] = {}
    referenced = list(query.dimensions)
    if query.time_dimension is not None:
        referenced.append(query.time_dimension)
    referenced.extend(item.dimension for item in query.filters)
    for name in referenced:
        dimension = model.dimensions.get(name)
        if dimension is None:
            raise SemanticCompileError(f"unknown dimension {name!r}")
        if dimension.source == base_source:
            continue
        if dimension.join is None:
            raise SemanticCompileError(
                f"dimension {name!r} lives on {dimension.source!r} and "
                "declares no join to reach it"
            )
        joined[dimension.source] = dimension.join
    return sorted(joined.items())


def _filter_clauses(
    model: SemanticModel, query: SemanticQuery, qualify: bool
) -> tuple[list[str], list[QueryParameter]]:
    clauses: list[str] = []
    parameters: list[QueryParameter] = []
    for index, item in enumerate(query.filters):
        dimension = model.dimensions[item.dimension]
        column = _column_reference(dimension.source, dimension.column, qualify)
        parameter_name = f"p{index}"
        if item.grain is not None:
            _require_date_dimension(model, item.dimension)
            value = normalize_time_value(item.grain, item.value)
            clauses.append(
                f"DATE_TRUNC('{item.grain.value.upper()}', {column}) "
                f"= :{parameter_name}"
            )
            parameters.append(
                QueryParameter(name=parameter_name, value=value, type="DATE")
            )
        else:
            clauses.append(f"{column} = :{parameter_name}")
            parameters.append(
                QueryParameter(name=parameter_name, value=item.value, type="STRING")
            )
    return clauses, parameters


def _aggregate(model: SemanticModel, metric_name: str, qualify: bool) -> str:
    metric = model.metrics[metric_name]
    expr = _column_reference(metric.source, metric.expr, qualify)
    if metric.filter is not None:
        condition_column = _column_reference(
            metric.source, metric.filter.column, qualify
        )
        literal = metric.filter.value.replace("'", "''")
        expr = (
            f"CASE WHEN {condition_column} {metric.filter.op} "
            f"'{literal}' THEN {expr} END"
        )
    if metric.aggregation is Aggregation.COUNT_DISTINCT:
        return f"COUNT(DISTINCT {expr})"
    return f"{metric.aggregation.value.upper()}({expr})"


def _require_date_dimension(model: SemanticModel, name: str) -> None:
    dimension = model.dimensions.get(name)
    if dimension is None:
        raise SemanticCompileError(f"unknown dimension {name!r}")
    if dimension.type != "date":
        raise SemanticCompileError(
            f"dimension {name!r} is {dimension.type}, not date; time grains "
            "apply only to date dimensions"
        )


def _table_reference(model: SemanticModel, source_name: str, qualify: bool) -> str:
    table = model.sources[source_name].table
    quoted = ".".join(_quote(part) for part in table.split("."))
    if qualify:
        return f"{quoted} AS {_quote(source_name)}"
    return quoted


def _column_reference(source_name: str, column: str, qualify: bool) -> str:
    if qualify:
        return f"{_quote(source_name)}.{_quote(column)}"
    return _quote(column)


def _quote(identifier: str) -> str:
    if not _IDENTIFIER.match(identifier):
        raise SemanticCompileError(f"identifier {identifier!r} is not quotable")
    return f"`{identifier}`"
