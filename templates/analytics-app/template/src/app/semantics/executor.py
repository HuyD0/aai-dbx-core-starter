"""Warehouse execution behind a neutral protocol.

``WarehouseExecutor`` is the seam that keeps the semantic layer portable:
``run_plan`` and ``query_rows`` execute compiled governed plans, ``execute``
retains a read-only defense for application-owned diagnostics, and
``latest_loaded_at`` reads a source's freshness watermark.
The Databricks adapter speaks the statement-execution API through the
already-certified SDK; another warehouse (Snowflake, anything with a SQL
API) is supported by implementing this protocol in application code — no
platform change required. ``FakeWarehouseExecutor`` evaluates plans in pure
Python over the versioned snapshot in evals/data/seed_data.json, which is
what lets evaluations and notebooks run credential-free.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from decimal import Decimal, InvalidOperation
from functools import partial
from pathlib import Path
from threading import Condition
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from aai_core.exceptions import AaiCoreError
from app.semantics.compiler import (
    Dialect,
    QueryParameter,
    RowOperator,
    RowQuery,
    ScalarValue,
    SemanticQuery,
    compile_query,
    compile_rows,
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

    This is defense in depth for application-owned diagnostic statements.
    Models never receive an SQL tool. The guard stays deliberately
    conservative: a quoted string containing a write keyword is rejected.
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
        raise NotImplementedError

    def query_rows(self, model: SemanticModel, query: RowQuery) -> QueryResult:
        """Compile and execute a governed row-level plan."""
        raise NotImplementedError

    def execute(
        self,
        sql: str,
        *,
        parameters: tuple[QueryParameter, ...] = (),
        row_limit: int = 1000,
    ) -> QueryResult:
        """Run one application-owned guarded read statement."""
        raise NotImplementedError

    def latest_loaded_at(self, model: SemanticModel, source_name: str) -> str | None:
        """Freshness watermark for a source, or None when undeclared."""
        raise NotImplementedError


@runtime_checkable
class AsyncWarehouseExecutor(Protocol):
    """Async application boundary with cancellation-aware remote execution."""

    dialect: Dialect

    async def arun_plan(
        self, model: SemanticModel, query: SemanticQuery
    ) -> QueryResult:
        raise NotImplementedError

    async def aquery_rows(self, model: SemanticModel, query: RowQuery) -> QueryResult:
        raise NotImplementedError

    async def alatest_loaded_at(
        self, model: SemanticModel, source_name: str
    ) -> str | None:
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError


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
        max_concurrent_operations: int = 8,
        cancel_grace_seconds: float = 10.0,
    ) -> None:
        if not warehouse_id.strip():
            raise WarehouseExecutionError("warehouse_id must be configured")
        self._owns_client = workspace_client is None
        if workspace_client is None:
            from aai_core.identity import databricks_workspace_client

            workspace_client = databricks_workspace_client()
        self._client: Any = workspace_client
        self._warehouse_id = warehouse_id.strip()
        self._catalog = catalog
        self._schema = schema
        self._wait_timeout = wait_timeout
        self._poll_interval_seconds = poll_interval_seconds
        self._max_poll_seconds = max_poll_seconds
        if max_concurrent_operations < 1:
            raise WarehouseExecutionError("max_concurrent_operations must be positive")
        if cancel_grace_seconds <= 0:
            raise WarehouseExecutionError("cancel_grace_seconds must be positive")
        self._cancel_grace_seconds = cancel_grace_seconds
        # The semaphore limits statement work to the configured bound; the
        # extra worker keeps a control-plane slot available for cancellation.
        self._pool = ThreadPoolExecutor(
            max_workers=max_concurrent_operations + 1,
            thread_name_prefix="analytics-warehouse",
        )
        self._semaphore = asyncio.Semaphore(max_concurrent_operations)
        self._lifecycle = Condition()
        self._active_operations: dict[int, str | None] = {}
        self._next_operation_id = 0
        self._closed = False
        self._close_finished = False

    @property
    def native_client(self) -> Any:
        """Provider escape hatch, mirroring the SDK's native_client rule."""

        return self._client

    def run_plan(self, model: SemanticModel, query: SemanticQuery) -> QueryResult:
        compiled = compile_query(model, query, self.dialect)
        return self.execute(
            compiled.sql, parameters=compiled.parameters, row_limit=query.limit
        )

    def query_rows(self, model: SemanticModel, query: RowQuery) -> QueryResult:
        compiled = compile_rows(model, query, self.dialect)
        return self.execute(
            compiled.sql,
            parameters=compiled.parameters,
            row_limit=query.limit,
        )

    async def arun_plan(
        self, model: SemanticModel, query: SemanticQuery
    ) -> QueryResult:
        compiled = compile_query(model, query, self.dialect)
        return await self._asubmit(
            compiled.sql,
            parameters=compiled.parameters,
            row_limit=query.limit,
        )

    async def aquery_rows(self, model: SemanticModel, query: RowQuery) -> QueryResult:
        compiled = compile_rows(model, query, self.dialect)
        return await self._asubmit(
            compiled.sql,
            parameters=compiled.parameters,
            row_limit=query.limit,
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

    async def alatest_loaded_at(
        self, model: SemanticModel, source_name: str
    ) -> str | None:
        source = model.sources[source_name]
        if source.loaded_at_column is None:
            return None
        table = ".".join(f"`{part}`" for part in source.table.split("."))
        result = await self._asubmit(
            f"SELECT MAX(`{source.loaded_at_column}`) FROM {table} LIMIT 1",
            parameters=(),
            row_limit=1,
        )
        return result.scalar

    async def aclose(self) -> None:
        """Wait for owned worker threads; the injected client remains caller-owned."""

        await asyncio.to_thread(self.close)

    def close(self) -> None:
        """Cancel active statements, then release owned resources exactly once."""

        with self._lifecycle:
            if self._close_finished:
                return
            if self._closed:
                while not self._close_finished:
                    self._lifecycle.wait()
                return
            self._closed = True
        try:
            # A control-plane failure cannot make it safe to close the client
            # under active data-plane calls. Those calls retain their own poll
            # deadline and cancellation path.
            with suppress(Exception):
                self._cancel_active_statements()
            with self._lifecycle:
                while self._active_operations:
                    self._lifecycle.wait()
            self._pool.shutdown(wait=True, cancel_futures=True)
        finally:
            try:
                if self._owns_client:
                    _close_workspace_client(self._client)
            finally:
                with self._lifecycle:
                    self._close_finished = True
                    self._lifecycle.notify_all()

    def _submit(
        self,
        sql: str,
        *,
        parameters: tuple[QueryParameter, ...],
        row_limit: int | None,
    ) -> QueryResult:
        from databricks.sdk.service.sql import StatementState

        operation_id = self._begin_operation()
        try:
            response = self._client.statement_execution.execute_statement(
                **self._request(sql, parameters, row_limit)
            )
            statement_id = response.statement_id
            if _state(response) in (
                StatementState.PENDING,
                StatementState.RUNNING,
            ) and self._track_statement(operation_id, statement_id):
                self._cancel_and_confirm_sync(statement_id)
                raise WarehouseExecutionError(
                    "warehouse executor closed while the active statement "
                    "was being cancelled"
                )
            deadline = time.monotonic() + self._max_poll_seconds
            while _state(response) in (StatementState.PENDING, StatementState.RUNNING):
                if time.monotonic() > deadline:
                    self._cancel_and_confirm_sync(statement_id)
                    raise WarehouseExecutionError(
                        "statement did not finish within "
                        f"{self._max_poll_seconds:g}s and cancellation was requested"
                    )
                time.sleep(self._poll_interval_seconds)
                response = self._client.statement_execution.get_statement(statement_id)
            if _state(response) is not StatementState.SUCCEEDED:
                raise WarehouseExecutionError(_failure_message(response))
            return self._result(response, sql)
        finally:
            self._finish_operation(operation_id)

    def _cancel_and_confirm_sync(self, statement_id: str) -> None:
        from databricks.sdk.service.sql import StatementState

        try:
            self._client.statement_execution.cancel_execution(statement_id)
        except Exception:  # preserve the original timeout or close failure
            return
        deadline = time.monotonic() + self._cancel_grace_seconds
        while time.monotonic() < deadline:
            try:
                response = self._client.statement_execution.get_statement(statement_id)
            except Exception:
                return
            if _state(response) not in (StatementState.PENDING, StatementState.RUNNING):
                return
            time.sleep(min(self._poll_interval_seconds, 0.25))

    def _begin_operation(self) -> int:
        with self._lifecycle:
            if self._closed:
                raise WarehouseExecutionError("warehouse executor is closed")
            operation_id = self._next_operation_id
            self._next_operation_id += 1
            self._active_operations[operation_id] = None
            return operation_id

    def _track_statement(self, operation_id: int, statement_id: str) -> bool:
        with self._lifecycle:
            self._active_operations[operation_id] = statement_id
            self._lifecycle.notify_all()
            return self._closed

    def _finish_operation(self, operation_id: int) -> None:
        with self._lifecycle:
            self._active_operations.pop(operation_id, None)
            self._lifecycle.notify_all()

    def _cancel_active_statements(self) -> None:
        from databricks.sdk.service.sql import StatementState

        deadline = time.monotonic() + self._cancel_grace_seconds
        requested: set[str] = set()
        terminal: set[str] = set()
        while time.monotonic() < deadline:
            with self._lifecycle:
                if not self._active_operations:
                    return
                statement_ids = {
                    statement_id
                    for statement_id in self._active_operations.values()
                    if statement_id is not None
                }
            for statement_id in statement_ids - requested:
                try:
                    self._client.statement_execution.cancel_execution(statement_id)
                except Exception:
                    terminal.add(statement_id)
                requested.add(statement_id)
            for statement_id in (statement_ids & requested) - terminal:
                try:
                    response = self._client.statement_execution.get_statement(
                        statement_id
                    )
                except Exception:
                    terminal.add(statement_id)
                    continue
                if _state(response) not in (
                    StatementState.PENDING,
                    StatementState.RUNNING,
                ):
                    terminal.add(statement_id)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            with self._lifecycle:
                if not self._active_operations:
                    return
                self._lifecycle.wait(
                    timeout=min(self._poll_interval_seconds, 0.25, remaining)
                )

    async def _asubmit(
        self,
        sql: str,
        *,
        parameters: tuple[QueryParameter, ...],
        row_limit: int | None,
    ) -> QueryResult:
        from databricks.sdk.service.sql import StatementState

        async with self._semaphore:
            operation_id = self._begin_operation()
            loop = asyncio.get_running_loop()
            statement_id: str | None = None
            try:
                submit = loop.run_in_executor(
                    self._pool,
                    partial(
                        self._client.statement_execution.execute_statement,
                        **self._request(sql, parameters, row_limit),
                    ),
                )
                response = await asyncio.shield(submit)
                statement_id = response.statement_id
                if _state(response) in (
                    StatementState.PENDING,
                    StatementState.RUNNING,
                ) and self._track_statement(operation_id, statement_id):
                    await self._cancel_and_confirm(statement_id)
                    raise WarehouseExecutionError(
                        "warehouse executor closed while the active statement "
                        "was being cancelled"
                    )
                deadline = loop.time() + self._max_poll_seconds
                while _state(response) in (
                    StatementState.PENDING,
                    StatementState.RUNNING,
                ):
                    if loop.time() > deadline:
                        await self._cancel_and_confirm(statement_id)
                        raise WarehouseExecutionError(
                            "statement did not finish within "
                            f"{self._max_poll_seconds:g}s and was cancelled"
                        )
                    await asyncio.sleep(self._poll_interval_seconds)
                    response = await loop.run_in_executor(
                        self._pool,
                        self._client.statement_execution.get_statement,
                        statement_id,
                    )
                if _state(response) is not StatementState.SUCCEEDED:
                    raise WarehouseExecutionError(_failure_message(response))
                return self._result(response, sql)
            except asyncio.CancelledError:
                if statement_id is None:
                    statement_id = await self._submitted_statement_id(submit)
                if statement_id is not None:
                    self._track_statement(operation_id, statement_id)
                    await self._cancel_and_confirm(statement_id)
                raise
            finally:
                self._finish_operation(operation_id)

    async def _submitted_statement_id(self, submit: asyncio.Future[Any]) -> str | None:
        try:
            response = await asyncio.wait_for(
                asyncio.shield(submit), timeout=self._cancel_grace_seconds
            )
        except Exception:  # provider failure leaves nothing to cancel
            return None
        value = getattr(response, "statement_id", None)
        return value if isinstance(value, str) and value else None

    async def _cancel_and_confirm(self, statement_id: str) -> None:
        from databricks.sdk.service.sql import StatementState

        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(
                    self._pool,
                    self._client.statement_execution.cancel_execution,
                    statement_id,
                ),
                timeout=self._cancel_grace_seconds,
            )
        except Exception:  # cancellation is best-effort; preserve original failure
            return
        deadline = loop.time() + self._cancel_grace_seconds
        while loop.time() < deadline:
            remaining = deadline - loop.time()
            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._pool,
                        self._client.statement_execution.get_statement,
                        statement_id,
                    ),
                    timeout=remaining,
                )
            except Exception:
                return
            if _state(response) not in (StatementState.PENDING, StatementState.RUNNING):
                return
            await asyncio.sleep(min(self._poll_interval_seconds, 0.25))

    def _request(
        self,
        sql: str,
        parameters: tuple[QueryParameter, ...],
        row_limit: int | None,
    ) -> dict[str, Any]:
        from databricks.sdk.service.sql import (
            Disposition,
            ExecuteStatementRequestOnWaitTimeout,
            Format,
            StatementParameterListItem,
        )

        return {
            "statement": sql,
            "warehouse_id": self._warehouse_id,
            "catalog": self._catalog,
            "schema": self._schema,
            "wait_timeout": self._wait_timeout,
            "on_wait_timeout": ExecuteStatementRequestOnWaitTimeout.CONTINUE,
            "disposition": Disposition.INLINE,
            "format": Format.JSON_ARRAY,
            "parameters": [
                StatementParameterListItem(
                    name=item.name, value=item.value, type=item.type
                )
                for item in parameters
            ]
            or None,
            "row_limit": row_limit,
        }

    def _result(self, response: Any, sql: str) -> QueryResult:
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
        payload: Any = (
            json.loads(Path(seed).read_text(encoding="utf-8"))
            if isinstance(seed, (str, Path))
            else seed
        )
        if not isinstance(payload, Mapping):
            raise TypeError("seed data must be a JSON object")
        tables = payload.get("tables")
        if not isinstance(tables, Mapping):
            raise TypeError("seed data requires a tables object")
        self._tables: dict[str, list[dict[str, str | None]]] = {}
        for name, table in tables.items():
            if not isinstance(name, str) or not isinstance(table, Mapping):
                raise TypeError("seed table names and definitions must be objects")
            columns = table.get("columns")
            rows = table.get("rows")
            if not isinstance(columns, list) or not isinstance(rows, list):
                raise TypeError("seed tables require columns and rows arrays")
            names = []
            for column in columns:
                column_name = (
                    column.get("name") if isinstance(column, Mapping) else None
                )
                if not isinstance(column_name, str):
                    raise TypeError("seed columns require string names")
                names.append(column_name)
            if not all(isinstance(row, list) for row in rows):
                raise TypeError("seed table rows must be arrays")
            self._tables[name] = [dict(zip(names, row, strict=True)) for row in rows]
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

    def query_rows(self, model: SemanticModel, query: RowQuery) -> QueryResult:
        compiled = compile_rows(model, query, self.dialect)
        source = model.sources[query.source]
        rows = list(self._rows_for_table(source.table))
        for item in query.filters:
            field = model.detail_fields[item.field]
            rows = [
                row
                for row in rows
                if _row_matches(
                    row.get(field.column), field.type, item.operator, item.value
                )
            ]
        for order in reversed(query.order_by):
            field = model.detail_fields[order.field]
            rows.sort(
                key=lambda row: _sort_value(row.get(field.column), field.type),
                reverse=order.direction.value == "desc",
            )
        values = tuple(
            tuple(row.get(model.detail_fields[name].column) for name in query.fields)
            for row in rows[: query.limit]
        )
        return QueryResult(
            columns=query.fields,
            types=tuple(
                model.detail_fields[name].type.upper() for name in query.fields
            ),
            rows=values,
            sql=compiled.sql,
        )

    async def arun_plan(
        self, model: SemanticModel, query: SemanticQuery
    ) -> QueryResult:
        return self.run_plan(model, query)

    async def aquery_rows(self, model: SemanticModel, query: RowQuery) -> QueryResult:
        return self.query_rows(model, query)

    async def alatest_loaded_at(
        self, model: SemanticModel, source_name: str
    ) -> str | None:
        return self.latest_loaded_at(model, source_name)

    async def aclose(self) -> None:
        return None

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
        values: list[str] = []
        for row in rows:
            value = row.get(source.loaded_at_column)
            if value is not None:
                values.append(value)
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
        selected = [
            row
            for row in base_rows
            if self._passes_query_filters(model, query, base_source, row)
        ]
        groups: dict[tuple[str | None, ...], list[Mapping[str, str | None]]] = {}
        for row in selected:
            key = self._group_key(model, query, base_source, row)
            groups.setdefault(key, []).append(row)

        output: list[tuple[str | None, ...]] = []
        for key in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
            rows = groups[key]
            values = [
                _aggregate_rows(model, metric_name, rows)
                for metric_name in query.metrics
            ]
            output.append((*key, *values))
        return output

    def _dimension_value(
        self,
        model: SemanticModel,
        base_source: str,
        name: str,
        row: Mapping[str, str | None],
    ) -> str | None:
        dimension = model.dimensions[name]
        if dimension.source == base_source:
            return row.get(dimension.column)
        join = dimension.join
        if join is None:
            raise WarehouseExecutionError(
                f"dimension {name!r} requires a declared join"
            )
        key = row.get(join.from_column)
        other_rows = self._rows_for_table(model.sources[dimension.source].table)
        return next(
            (
                other.get(dimension.column)
                for other in other_rows
                if other.get(join.to_column) == key
            ),
            None,
        )

    def _passes_query_filters(
        self,
        model: SemanticModel,
        query: SemanticQuery,
        base_source: str,
        row: Mapping[str, str | None],
    ) -> bool:
        for item in query.filters:
            value = self._dimension_value(model, base_source, item.dimension, row)
            if item.grain is None:
                matches = value == item.value
            else:
                expected = normalize_time_value(item.grain, item.value)
                matches = value is not None and (
                    truncate_date(item.grain, value) == expected
                )
            if not matches:
                return False
        return True

    def _group_key(
        self,
        model: SemanticModel,
        query: SemanticQuery,
        base_source: str,
        row: Mapping[str, str | None],
    ) -> tuple[str | None, ...]:
        key: list[str | None] = []
        if query.time_dimension is not None and query.time_grain is not None:
            value = self._dimension_value(model, base_source, query.time_dimension, row)
            key.append(
                None if value is None else truncate_date(query.time_grain, value)
            )
        key.extend(
            self._dimension_value(model, base_source, name, row)
            for name in query.dimensions
        )
        return tuple(key)


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
    if _NUMBER.match(observed) and _NUMBER.match(condition.value):
        return _compare_values(
            _decimal(observed),
            _decimal(condition.value),
            condition.op,
        )
    return _compare_values(observed, condition.value, condition.op)


ComparableScalar = str | Decimal | bool


def _typed_row_value(value: str | None, field_type: str) -> ComparableScalar | None:
    if value is None:
        return None
    if field_type == "number":
        return _decimal(value)
    if field_type == "boolean":
        return value.strip().lower() == "true"
    if field_type == "date":
        return value[:10]
    return value


def _row_matches(
    observed: str | None,
    field_type: str,
    operator: RowOperator,
    expected: ScalarValue | tuple[ScalarValue, ...] | None,
) -> bool:
    if operator is RowOperator.IS_NULL:
        return observed is None
    left = _typed_row_value(observed, field_type)
    if left is None:
        return False
    if operator is RowOperator.IN:
        if not isinstance(expected, tuple):
            raise WarehouseExecutionError("in filters require a tuple of values")
        right_values = tuple(
            _typed_filter_value(value, field_type) for value in expected
        )
        return left in right_values
    if isinstance(expected, tuple) or expected is None:
        raise WarehouseExecutionError("scalar filters require exactly one value")
    right = _typed_filter_value(expected, field_type)
    if operator is RowOperator.EQ:
        return left == right
    if operator is RowOperator.NE:
        return left != right
    if isinstance(left, bool) or isinstance(right, bool):
        raise WarehouseExecutionError("boolean fields do not support ordering")
    return _compare_values(left, right, operator.value)


def _typed_filter_value(value: ScalarValue, field_type: str) -> ComparableScalar:
    if field_type == "number":
        return Decimal(str(value))
    if field_type == "boolean":
        if not isinstance(value, bool):
            raise WarehouseExecutionError("boolean filters require boolean values")
        return value
    if not isinstance(value, str):
        raise WarehouseExecutionError("string and date filters require string values")
    return value


def _compare_values(
    left: str | Decimal,
    right: str | Decimal,
    operator: str,
) -> bool:
    if isinstance(left, Decimal):
        if not isinstance(right, Decimal):
            raise WarehouseExecutionError("cannot compare values of different types")
        return _compare_decimals(left, right, operator)
    if not isinstance(right, str):
        raise WarehouseExecutionError("cannot compare values of different types")
    return _compare_strings(left, right, operator)


def _compare_decimals(left: Decimal, right: Decimal, operator: str) -> bool:
    if operator == "=":
        return left == right
    if operator == "<>":
        return left != right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "<":
        return left < right
    return left <= right


def _compare_strings(left: str, right: str, operator: str) -> bool:
    if operator == "=":
        return left == right
    if operator == "<>":
        return left != right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "<":
        return left < right
    return left <= right


def _sort_value(value: str | None, field_type: str) -> tuple[bool, Any]:
    return value is None, _typed_row_value(value, field_type)


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


def _close_workspace_client(client: Any) -> None:
    """Close an owned WorkspaceClient without relying on garbage collection."""

    close = getattr(client, "close", None)
    if callable(close):
        close()
        return
    # databricks-sdk 0.122 has no public WorkspaceClient.close(). Keep the
    # certified-version compatibility access isolated to lifecycle cleanup.
    api_client = getattr(client, "api_client", None)
    transport = getattr(api_client, "_api_client", None)
    session = getattr(transport, "_session", None)
    close = getattr(session, "close", None)
    if callable(close):
        close()
