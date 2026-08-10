"""Executor behavior: read-only guard, snapshot interpreter, API adapter."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from app.semantics.compiler import (
    OrderDirection,
    QueryFilter,
    RowFilter,
    RowOperator,
    RowOrder,
    RowQuery,
    SemanticQuery,
    TimeGrain,
)
from app.semantics.executor import (
    DatabricksWarehouseExecutor,
    FakeWarehouseExecutor,
    QueryResult,
    WarehouseExecutionError,
    ensure_read_only,
)

MARCH = QueryFilter(dimension="order_date", value="2024-03", grain=TimeGrain.MONTH)


# ---------------------------------------------------------------- guard


def test_guard_appends_limit_and_collapses_whitespace():
    assert (
        ensure_read_only("SELECT *\nFROM t", row_limit=50) == "SELECT * FROM t LIMIT 50"
    )
    assert ensure_read_only("SELECT 1 LIMIT 5") == "SELECT 1 LIMIT 5"


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "CREATE TABLE t (x INT)",
        "SELECT 1; DROP TABLE t",
        "WITH x AS (SELECT 1) DELETE FROM t",
        "EXPLAIN MERGE INTO t USING s ON 1=1",
        "GRANT SELECT ON t TO everyone",
        "",
    ],
)
def test_guard_rejects_writes_and_multi_statements(sql):
    with pytest.raises(WarehouseExecutionError):
        ensure_read_only(sql)


def test_guard_strips_comments_before_judging():
    assert (
        ensure_read_only("SELECT 1 -- DROP TABLE t", row_limit=10)
        == "SELECT 1 LIMIT 10"
    )
    with pytest.raises(WarehouseExecutionError):
        ensure_read_only("SELECT/**/1; DROP TABLE t")


def test_guard_allows_write_keywords_inside_words():
    guarded = ensure_read_only("SELECT updated_at FROM t", row_limit=10)
    assert guarded == "SELECT updated_at FROM t LIMIT 10"


# ---------------------------------------------------------------- fake


def test_snapshot_reproduces_pinned_march_revenue(model, seed_executor):
    result = seed_executor.run_plan(
        model, SemanticQuery(metrics=("revenue",), filters=(MARCH,))
    )
    assert result.scalar == "600.75"
    assert "SUM(CASE WHEN" in result.sql


def test_snapshot_grouped_join_matches_hand_computation(model, seed_executor):
    result = seed_executor.run_plan(
        model,
        SemanticQuery(metrics=("revenue",), dimensions=("region",), filters=(MARCH,)),
    )
    assert result.rows == (
        ("north", "200.5"),
        ("south", "260.25"),
        ("west", "140"),
    )


def test_snapshot_count_distinct_and_filtered_counts(model, seed_executor):
    distinct = seed_executor.run_plan(
        model, SemanticQuery(metrics=("active_customers",))
    )
    assert distinct.scalar == "6"
    shipped_march = seed_executor.run_plan(
        model,
        SemanticQuery(
            metrics=("order_count",),
            filters=(QueryFilter(dimension="order_status", value="S"), MARCH),
        ),
    )
    assert shipped_march.scalar == "4"


def test_snapshot_average_excludes_cancelled(model, seed_executor):
    result = seed_executor.run_plan(
        model, SemanticQuery(metrics=("average_order_value",), filters=(MARCH,))
    )
    assert result.scalar == "120.15"


def test_snapshot_trend_buckets_by_month(model, seed_executor):
    result = seed_executor.run_plan(
        model,
        SemanticQuery(
            metrics=("revenue",),
            time_dimension="order_date",
            time_grain=TimeGrain.MONTH,
        ),
    )
    assert result.rows == (("2024-03-01", "600.75"), ("2024-04-01", "711.25"))


def test_fake_freshness_watermark(model, seed_executor):
    assert seed_executor.latest_loaded_at(model, "orders") == "2024-05-01T06:00:00Z"


def test_fake_execute_requires_canned_fixture(model, seed_executor):
    with pytest.raises(WarehouseExecutionError, match="canned"):
        seed_executor.execute("SELECT `order_id` FROM t")
    with pytest.raises(WarehouseExecutionError):
        seed_executor.execute("DROP TABLE t")


def test_fake_execute_serves_canned_result():
    canned_sql = "SELECT `order_id` FROM `demo`.`sales`.`analytics_orders` LIMIT 3"
    canned = QueryResult(columns=("order_id",), rows=(("O-1004",),), sql=canned_sql)
    executor = FakeWarehouseExecutor({"tables": {}}, canned={canned_sql: canned})
    assert executor.execute(canned_sql).rows == (("O-1004",),)


def test_fake_governed_row_query_filters_orders_and_bounds_rows(model, seed_executor):
    result = seed_executor.query_rows(
        model,
        RowQuery(
            source="orders",
            fields=("order_id", "order_amount"),
            filters=(
                RowFilter(
                    field="order_status",
                    operator=RowOperator.EQ,
                    value="S",
                ),
            ),
            order_by=(
                RowOrder(
                    field="order_amount",
                    direction=OrderDirection.DESC,
                ),
            ),
            limit=2,
            reason="largest shipped orders",
        ),
    )

    assert result.rows == (("O-1009", "310.00"), ("O-1012", "210.00"))


# ---------------------------------------------------------------- adapter


class _StubStatementApi:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.polls = []
        self.cancellations = []

    def execute_statement(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)

    def get_statement(self, statement_id):
        self.polls.append(statement_id)
        return self.responses.pop(0)

    def cancel_execution(self, statement_id):
        self.cancellations.append(statement_id)


def _response(state, rows=None, columns=(), error=None):
    from databricks.sdk.service.sql import StatementState

    manifest = None
    result = None
    if rows is not None:
        manifest = SimpleNamespace(
            schema=SimpleNamespace(
                columns=[
                    SimpleNamespace(
                        name=name, type_name=SimpleNamespace(value="STRING")
                    )
                    for name in columns
                ]
            )
        )
        result = SimpleNamespace(data_array=rows)
    return SimpleNamespace(
        statement_id="stmt-1",
        status=SimpleNamespace(state=StatementState[state], error=error),
        manifest=manifest,
        result=result,
    )


def _executor(responses):
    client = SimpleNamespace(statement_execution=_StubStatementApi(responses))
    return (
        DatabricksWarehouseExecutor(
            client,
            warehouse_id="wh-1",
            catalog="demo",
            schema="sales",
            poll_interval_seconds=0.001,
        ),
        client,
    )


def test_adapter_maps_inline_results_and_parameters(model):
    executor, client = _executor(
        [_response("SUCCEEDED", rows=[["600.75"]], columns=("revenue",))]
    )
    result = executor.run_plan(
        model, SemanticQuery(metrics=("revenue",), filters=(MARCH,))
    )
    assert result.scalar == "600.75"
    assert result.warehouse_id == "wh-1"
    request = client.statement_execution.requests[0]
    assert request["warehouse_id"] == "wh-1"
    assert request["catalog"] == "demo"
    assert [p.value for p in request["parameters"]] == ["2024-03-01"]
    assert request["row_limit"] == 100


def test_adapter_polls_until_terminal_state():
    executor, client = _executor(
        [
            _response("PENDING"),
            _response("RUNNING"),
            _response("SUCCEEDED", rows=[["1"]], columns=("one",)),
        ]
    )
    assert executor.execute("SELECT 1").scalar == "1"
    assert client.statement_execution.polls == ["stmt-1", "stmt-1"]


def test_sync_adapter_cancels_remote_statement_on_poll_deadline():
    executor, client = _executor([_response("PENDING"), _response("CANCELED")])
    executor._max_poll_seconds = 0

    with pytest.raises(WarehouseExecutionError, match="cancellation was requested"):
        executor.execute("SELECT 1")

    assert client.statement_execution.cancellations == ["stmt-1"]
    assert client.statement_execution.polls == ["stmt-1"]


def test_sync_adapter_refuses_work_after_close():
    executor, client = _executor([])
    executor.close()

    with pytest.raises(WarehouseExecutionError, match="closed"):
        executor.execute("SELECT 1")

    assert client.statement_execution.requests == []


def test_adapter_closes_only_a_client_it_created(monkeypatch):
    owned = SimpleNamespace(
        statement_execution=_StubStatementApi([]),
        close_calls=0,
    )
    owned.close = lambda: setattr(owned, "close_calls", owned.close_calls + 1)
    monkeypatch.setattr(
        "aai_core.identity.databricks_workspace_client",
        lambda: owned,
    )
    executor = DatabricksWarehouseExecutor(warehouse_id="wh-1")

    executor.close()
    executor.close()

    assert owned.close_calls == 1

    injected = SimpleNamespace(
        statement_execution=_StubStatementApi([]),
        closed=False,
    )
    injected.close = lambda: setattr(injected, "closed", True)
    injected_executor = DatabricksWarehouseExecutor(injected, warehouse_id="wh-1")

    injected_executor.close()

    assert injected.closed is False


def test_concurrent_close_waits_for_active_sync_call_and_closes_session_once(
    monkeypatch,
):
    call_started = Event()
    release_call = Event()

    class StatementApi:
        def execute_statement(self, **request):
            call_started.set()
            assert release_call.wait(timeout=2)
            return _response("SUCCEEDED", rows=[["1"]], columns=("one",))

    session = SimpleNamespace(close_calls=0)
    session.close = lambda: setattr(session, "close_calls", session.close_calls + 1)
    owned = SimpleNamespace(
        statement_execution=StatementApi(),
        api_client=SimpleNamespace(
            _api_client=SimpleNamespace(_session=session),
        ),
    )
    monkeypatch.setattr(
        "aai_core.identity.databricks_workspace_client",
        lambda: owned,
    )
    executor = DatabricksWarehouseExecutor(
        warehouse_id="wh-1",
        cancel_grace_seconds=0.02,
        poll_interval_seconds=0.001,
    )

    with ThreadPoolExecutor(max_workers=3) as pool:
        execution = pool.submit(executor.execute, "SELECT 1")
        assert call_started.wait(timeout=2)
        first_close = pool.submit(executor.close)
        second_close = pool.submit(executor.close)
        time.sleep(0.03)
        assert session.close_calls == 0
        release_call.set()
        assert execution.result(timeout=2).scalar == "1"
        first_close.result(timeout=2)
        second_close.result(timeout=2)

    assert session.close_calls == 1


def test_close_cancels_active_async_work_and_rejects_semaphore_queue(model):
    async def scenario():
        call_started = Event()
        release_call = Event()

        class StatementApi:
            def __init__(self):
                self.requests = []
                self.cancellations = []
                self.polls = []
                self.cancelled = Event()

            def execute_statement(self, **request):
                self.requests.append(request)
                call_started.set()
                assert release_call.wait(timeout=2)
                return _response("PENDING")

            def cancel_execution(self, statement_id):
                self.cancellations.append(statement_id)
                self.cancelled.set()

            def get_statement(self, statement_id):
                self.polls.append(statement_id)
                return _response("CANCELED" if self.cancelled.is_set() else "PENDING")

        api = StatementApi()
        executor = DatabricksWarehouseExecutor(
            SimpleNamespace(statement_execution=api),
            warehouse_id="wh-1",
            max_concurrent_operations=1,
            cancel_grace_seconds=0.2,
            poll_interval_seconds=0.001,
        )
        first = asyncio.create_task(
            executor.arun_plan(model, SemanticQuery(metrics=("revenue",)))
        )
        assert await asyncio.to_thread(call_started.wait, 2)
        queued = asyncio.create_task(
            executor.arun_plan(model, SemanticQuery(metrics=("revenue",)))
        )
        closing = asyncio.create_task(executor.aclose())
        await asyncio.sleep(0.01)
        release_call.set()

        with pytest.raises(WarehouseExecutionError, match="closed"):
            _ = await first
        with pytest.raises(WarehouseExecutionError, match="closed"):
            _ = await queued
        _ = await closing

        assert len(api.requests) == 1
        assert api.cancellations
        assert api.polls

    asyncio.run(scenario())


def test_adapter_surfaces_failure_state_with_bounded_message():
    error = SimpleNamespace(message="PERMISSION_DENIED: no CAN USE on warehouse")
    executor, _ = _executor([_response("FAILED", error=error)])
    with pytest.raises(WarehouseExecutionError, match="PERMISSION_DENIED"):
        executor.execute("SELECT 1")


def test_adapter_guards_before_submitting():
    executor, client = _executor([])
    with pytest.raises(WarehouseExecutionError):
        executor.execute("DELETE FROM t")
    assert client.statement_execution.requests == []


def test_adapter_requires_a_warehouse_id():
    with pytest.raises(WarehouseExecutionError, match="warehouse_id"):
        DatabricksWarehouseExecutor(SimpleNamespace(), warehouse_id="  ")


def test_async_adapter_cancels_remote_statement_on_task_cancellation(model):
    async def scenario():
        executor, client = _executor([_response("PENDING"), _response("CANCELED")])
        executor._poll_interval_seconds = 0.05
        task = asyncio.create_task(
            executor.arun_plan(model, SemanticQuery(metrics=("revenue",)))
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            _ = await task
        assert client.statement_execution.cancellations == ["stmt-1"]
        assert client.statement_execution.polls == ["stmt-1"]
        await executor.aclose()

    asyncio.run(scenario())


def test_async_adapter_cancels_remote_statement_on_poll_deadline(model):
    async def scenario():
        executor, client = _executor([_response("PENDING"), _response("CANCELED")])
        executor._max_poll_seconds = 0
        with pytest.raises(WarehouseExecutionError, match="was cancelled"):
            await executor.arun_plan(model, SemanticQuery(metrics=("revenue",)))
        assert client.statement_execution.cancellations == ["stmt-1"]
        await executor.aclose()

    asyncio.run(scenario())


def test_async_adapter_refuses_work_after_close(model):
    async def scenario():
        executor, _ = _executor([])
        await executor.aclose()
        with pytest.raises(WarehouseExecutionError, match="closed"):
            await executor.aquery_rows(
                model,
                RowQuery(
                    source="orders",
                    fields=("order_id",),
                    reason="closed executor check",
                ),
            )

    asyncio.run(scenario())
