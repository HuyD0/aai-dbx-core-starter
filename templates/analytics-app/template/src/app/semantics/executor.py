"""Warehouse execution behind a neutral protocol.

``WarehouseExecutor`` is the seam that keeps the semantic layer portable:
``run_plan`` executes a compiled semantic plan, ``execute`` runs guarded
read-only SQL, and ``latest_loaded_at`` reads a source's freshness watermark.
The Databricks adapter speaks the statement-execution API through the
already-certified SDK; another warehouse (Snowflake, anything with a SQL
API) is supported by implementing this protocol in application code — no
platform change required. ``FakeWarehouseExecutor`` evaluates plans in pure
Python over the versioned snapshot in evals/data/seed_data.json, which is
what lets evaluations and notebooks run credential-free.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from aai_core.exceptions import AaiCoreError
from app.semantics.compiler import (
    Dialect,
    QueryParameter,
    SemanticQuery,
    compile_query,
    normalize_time_value,
    truncate_date,
)
from app.semantics.models import Aggregation, Metric, SemanticModel

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LIMIT = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)
_WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE"
    r"|COPY|VACUUM|OPTIMIZE|REFRESH|SET)\b",
    re.IGNORECASE,
)
_READ_STARTERS = ("SELECT", "WITH", "EXPLAIN", "SHOW", "DESCRIBE")
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")


class WarehouseExecutionError(AaiCoreError):
    code = "app.warehouse_execution"


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: tuple[str, ...]
    types: tuple[str, ...] = ()
    rows: tuple[tuple[str | None, ...], ...]
    sql: str = Field(min_length=1)
    warehouse_id: str | None = None

    @property
    def scalar(self) -> str | None:
        """The single value when the result is one row by one column."""

        if len(self.rows) == 1 and len(self.columns) == 1:
            return self.rows[0][0]
        return None


def ensure_read_only(sql: str, *, row_limit: int = 1000) -> str:
    """Reject anything but a single read statement; enforce a LIMIT.

    The guard is a tool boundary for model-authored SQL, deliberately
    conservative: a quoted string containing a write keyword is rejected
    rather than parsed around.
    """

    stripped = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql))
    collapsed = " ".join(stripped.split())
    if not collapsed:
        raise WarehouseExecutionError("SQL statement is empty")
    if ";" in collapsed:
        raise WarehouseExecutionError(
            "multi-statement SQL is not allowed; submit one statement "
            "without a terminator"
        )
    first_token = collapsed.split(" ", 1)[0].upper()
    if first_token not in _READ_STARTERS:
        raise WarehouseExecutionError(
            f"statement must start with one of {_READ_STARTERS}"
        )
    if match := _WRITE_KEYWORDS.search(collapsed):
        raise WarehouseExecutionError(
            f"write keyword {match.group(1).upper()} is not allowed on the "
            "analytics path"
        )
    if not _LIMIT.search(collapsed):
        collapsed = f"{collapsed} LIMIT {row_limit}"
    return collapsed


@runtime_checkable
class WarehouseExecutor(Protocol):
    """Capability seam for any SQL warehouse; see the module docstring."""

    dialect: Dialect

    def run_plan(self, model: SemanticModel, query: SemanticQuery) -> QueryResult:
        """Compile and execute a semantic plan (the governed path)."""
        ...

    def execute(
        self,
        sql: str,
        *,
        parameters: tuple[QueryParameter, ...] = (),
        row_limit: int = 1000,
    ) -> QueryResult:
        """Run one guarded read-only statement (the fallback path)."""
        ...

    def latest_loaded_at(self, model: SemanticModel, source_name: str) -> str | None:
        """Freshness watermark for a source, or None when undeclared."""
        ...


class DatabricksWarehouseExecutor:
    """Statement-execution adapter for Databricks SQL warehouses."""

    dialect = Dialect.DATABRICKS

    def __init__(
        self,
        workspace_client: Any | None = None,
        *,
        warehouse_id: str,
        catalog: str | None = None,
        schema: str | None = None,
        wait_timeout: str = "30s",
        poll_interval_seconds: float = 1.0,
        max_poll_seconds: float = 120.0,
    ) -> None:
        if not warehouse_id.strip():
            raise WarehouseExecutionError("warehouse_id must be configured")
        if workspace_client is None:
            from aai_core.identity import databricks_workspace_client

            workspace_client = databricks_workspace_client()
        self._client = workspace_client
        self._warehouse_id = warehouse_id.strip()
        self._catalog = catalog
        self._schema = schema
        self._wait_timeout = wait_timeout
        self._poll_interval_seconds = poll_interval_seconds
        self._max_poll_seconds = max_poll_seconds

    @property
    def native_client(self) -> Any:
        """Provider escape hatch, mirroring the SDK's native_client rule."""

        return self._client

    def run_plan(self, model: SemanticModel, query: SemanticQuery) -> QueryResult:
        compiled = compile_query(model, query, self.dialect)
        return self.execute(
            compiled.sql, parameters=compiled.parameters, row_limit=query.limit
        )

    def execute(
        self,
        sql: str,
        *,
        parameters: tuple[QueryParameter, ...] = (),
        row_limit: int = 1000,
    ) -> QueryResult:
        guarded = ensure_read_only(sql, row_limit=row_limit)
        return self._submit(guarded, parameters=parameters, row_limit=row_limit)

    def execute_unguarded(self, sql: str) -> QueryResult:
        """Platform statements (seeding) only — never expose to the agent."""

        return self._submit(sql, parameters=(), row_limit=None)

    def latest_loaded_at(self, model: SemanticModel, source_name: str) -> str | None:
        source = model.sources[source_name]
        if source.loaded_at_column is None:
            return None
        table = ".".join(f"`{part}`" for part in source.table.split("."))
        result = self.execute(
            f"SELECT MAX(`{source.loaded_at_column}`) FROM {table}", row_limit=1
        )
        return result.scalar

    def _submit(
        self,
        sql: str,
        *,
        parameters: tuple[QueryParameter, ...],
        row_limit: int | None,
    ) -> QueryResult:
        from databricks.sdk.service.sql import (
            Disposition,
            ExecuteStatementRequestOnWaitTimeout,
            Format,
            StatementParameterListItem,
            StatementState,
        )

        response = self._client.statement_execution.execute_statement(
            statement=sql,
            warehouse_id=self._warehouse_id,
            catalog=self._catalog,
            schema=self._schema,
            wait_timeout=self._wait_timeout,
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
            disposition=Disposition.INLINE,
            format=Format.JSON_ARRAY,
            parameters=[
                StatementParameterListItem(
                    name=item.name, value=item.value, type=item.type
                )
                for item in parameters
            ]
            or None,
            row_limit=row_limit,
        )
        deadline = time.monotonic() + self._max_poll_seconds
        while _state(response) in (StatementState.PENDING, StatementState.RUNNING):
            if time.monotonic() > deadline:
                raise WarehouseExecutionError(
                    f"statement did not finish within {self._max_poll_seconds:g}s"
                )
            time.sleep(self._poll_interval_seconds)
            response = self._client.statement_execution.get_statement(
                response.statement_id
            )
        if _state(response) is not StatementState.SUCCEEDED:
            raise WarehouseExecutionError(_failure_message(response))
        columns: tuple[str, ...] = ()
        types: tuple[str, ...] = ()
        manifest = getattr(response, "manifest", None)
        manifest_schema = getattr(manifest, "schema", None)
        if manifest_schema is not None and manifest_schema.columns:
            columns = tuple(column.name for column in manifest_schema.columns)
            types = tuple(
                str(getattr(column.type_name, "value", column.type_name))
                for column in manifest_schema.columns
            )
        result = getattr(response, "result", None)
        data = getattr(result, "data_array", None) or []
        rows = tuple(
            tuple(None if value is None else str(value) for value in row)
            for row in data
        )
        return QueryResult(
            columns=columns,
            types=types,
            rows=rows,
            sql=sql,
            warehouse_id=self._warehouse_id,
        )


class FakeWarehouseExecutor:
    """Pure-Python plan interpreter over the versioned seed snapshot.

    ``run_plan`` mirrors the compiler's SQL semantics so golden answers stay
    pinned to evals/data/seed_data.json; ``execute`` serves canned results
    for recorded raw-SQL fixtures and enforces the same read-only guard.
    """

    dialect = Dialect.DATABRICKS

    def __init__(
        self,
        seed: str | Path | Mapping[str, Any],
        *,
        canned: Mapping[str, QueryResult] | None = None,
    ) -> None:
        if isinstance(seed, (str, Path)):
            seed = json.loads(Path(seed).read_text(encoding="utf-8"))
        tables = seed["tables"]
        self._tables: dict[str, list[dict[str, str | None]]] = {}
        for name, payload in tables.items():
            names = [column["name"] for column in payload["columns"]]
            self._tables[name] = [
                dict(zip(names, row, strict=True)) for row in payload["rows"]
            ]
        self._canned = {
            " ".join(sql.split()): result for sql, result in (canned or {}).items()
        }

    def run_plan(self, model: SemanticModel, query: SemanticQuery) -> QueryResult:
        compiled = compile_query(model, query, self.dialect)
        rows = self._plan_rows(model, query)
        columns = _plan_columns(query)
        return QueryResult(
            columns=columns,
            types=("STRING",) * len(columns),
            rows=tuple(rows[: query.limit]),
            sql=compiled.sql,
        )

    def execute(
        self,
        sql: str,
        *,
        parameters: tuple[QueryParameter, ...] = (),
        row_limit: int = 1000,
    ) -> QueryResult:
        guarded = ensure_read_only(sql, row_limit=row_limit)
        canned = self._canned.get(" ".join(sql.split())) or self._canned.get(guarded)
        if canned is None:
            raise WarehouseExecutionError(
                "FakeWarehouseExecutor has no canned result for this SQL; "
                "raw statements need a recorded fixture"
            )
        return canned

    def latest_loaded_at(self, model: SemanticModel, source_name: str) -> str | None:
        source = model.sources[source_name]
        if source.loaded_at_column is None:
            return None
        rows = self._rows_for_table(source.table)
        values = [
            row[source.loaded_at_column]
            for row in rows
            if row.get(source.loaded_at_column) is not None
        ]
        return max(values) if values else None

    # -- plan interpretation -------------------------------------------------

    def _rows_for_table(self, table_fqn: str) -> list[dict[str, str | None]]:
        name = table_fqn.rsplit(".", 1)[-1]
        if name not in self._tables:
            raise WarehouseExecutionError(f"seed data has no table {name!r}")
        return self._tables[name]

    def _plan_rows(
        self, model: SemanticModel, query: SemanticQuery
    ) -> list[tuple[str | None, ...]]:
        base_source = model.metrics[query.metrics[0]].source
        base_rows = self._rows_for_table(model.sources[base_source].table)

        def dimension_value(name: str, row: Mapping[str, str | None]) -> str | None:
            dimension = model.dimensions[name]
            if dimension.source == base_source:
                return row.get(dimension.column)
            join = dimension.join
            if join is None:
                raise WarehouseExecutionError(
                    f"dimension {name!r} requires a declared join"
                )
            key = row.get(join.from_column)
            for other in self._rows_for_table(model.sources[dimension.source].table):
                if other.get(join.to_column) == key:
                    return other.get(dimension.column)
            return None

        selected = []
        for row in base_rows:
            keep = True
            for item in query.filters:
                value = dimension_value(item.dimension, row)
                if item.grain is not None:
                    expected = normalize_time_value(item.grain, item.value)
                    keep = value is not None and (
                        truncate_date(item.grain, value) == expected
                    )
                else:
                    keep = value == item.value
                if not keep:
                    break
            if keep:
                selected.append(row)

        groups: dict[tuple[str | None, ...], list[Mapping[str, str | None]]] = {}
        for row in selected:
            key: list[str | None] = []
            if query.time_dimension is not None and query.time_grain is not None:
                value = dimension_value(query.time_dimension, row)
                key.append(
                    None if value is None else truncate_date(query.time_grain, value)
                )
            for name in query.dimensions:
                key.append(dimension_value(name, row))
            groups.setdefault(tuple(key), []).append(row)

        output: list[tuple[str | None, ...]] = []
        for key in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
            rows = groups[key]
            values = [
                _aggregate_rows(model, metric_name, rows)
                for metric_name in query.metrics
            ]
            output.append((*key, *values))
        return output


def _plan_columns(query: SemanticQuery) -> tuple[str, ...]:
    columns: list[str] = []
    if query.time_dimension is not None and query.time_grain is not None:
        columns.append(f"{query.time_dimension}_{query.time_grain.value}")
    columns.extend(query.dimensions)
    columns.extend(query.metrics)
    return tuple(columns)


def _aggregate_rows(
    model: SemanticModel,
    metric_name: str,
    rows: Sequence[Mapping[str, str | None]],
) -> str | None:
    metric = model.metrics[metric_name]
    values: list[str] = []
    for row in rows:
        if metric.filter is not None and not _passes(metric, row):
            continue
        value = row.get(metric.expr)
        if value is not None:
            values.append(value)
    if metric.aggregation is Aggregation.COUNT:
        return str(len(values))
    if metric.aggregation is Aggregation.COUNT_DISTINCT:
        return str(len(set(values)))
    if not values:
        return None
    numbers = [_decimal(value) for value in values]
    if metric.aggregation is Aggregation.SUM:
        return _render(sum(numbers, Decimal(0)))
    if metric.aggregation is Aggregation.AVG:
        return _render(sum(numbers, Decimal(0)) / Decimal(len(numbers)))
    if metric.aggregation is Aggregation.MIN:
        return _render(min(numbers))
    return _render(max(numbers))


def _passes(metric: Metric, row: Mapping[str, str | None]) -> bool:
    condition = metric.filter
    if condition is None:
        return True
    observed = row.get(condition.column)
    if observed is None:
        return False
    left: Any = observed
    right: Any = condition.value
    if _NUMBER.match(observed) and _NUMBER.match(condition.value):
        left, right = _decimal(observed), _decimal(condition.value)
    if condition.op == "=":
        return left == right
    if condition.op == "<>":
        return left != right
    if condition.op == ">":
        return left > right
    if condition.op == ">=":
        return left >= right
    if condition.op == "<":
        return left < right
    return left <= right


def _decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise WarehouseExecutionError(
            f"value {value!r} is not numeric; check the metric expr"
        ) from error


def _render(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _state(response: Any) -> Any:
    status = getattr(response, "status", None)
    return getattr(status, "state", None)


def _failure_message(response: Any) -> str:
    status = getattr(response, "status", None)
    state = getattr(getattr(status, "state", None), "value", "unknown")
    error = getattr(status, "error", None)
    message = getattr(error, "message", None) or "no error detail"
    return f"statement finished in state {state}: {str(message)[:300]}"
