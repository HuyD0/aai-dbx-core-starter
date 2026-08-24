"""SQL snapshots: the compiler is deterministic or it is broken."""

import pytest
from pydantic import ValidationError

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
    normalize_time_value,
    truncate_date,
)

MARCH = QueryFilter(dimension="order_date", value="2024-03", grain=TimeGrain.MONTH)


def test_scalar_metric_with_month_filter(model):
    compiled = compile_query(
        model, SemanticQuery(metrics=("revenue",), filters=(MARCH,))
    )
    assert compiled.sql == (
        "SELECT SUM(CASE WHEN `status` <> 'C' THEN `amount` END) AS `revenue` "
        "FROM `demo`.`sales`.`analytics_orders` "
        "WHERE DATE_TRUNC('MONTH', `order_date`) = :p0 LIMIT 100"
    )
    assert [(p.name, p.value, p.type) for p in compiled.parameters] == [
        ("p0", "2024-03-01", "DATE")
    ]
    assert compiled.sources == ("demo.sales.analytics_orders",)


def test_joined_dimension_qualifies_and_groups(model):
    compiled = compile_query(
        model,
        SemanticQuery(metrics=("revenue",), dimensions=("region",), filters=(MARCH,)),
    )
    assert compiled.sql == (
        "SELECT `customers`.`region` AS `region`, "
        "SUM(CASE WHEN `orders`.`status` <> 'C' THEN `orders`.`amount` END) "
        "AS `revenue` "
        "FROM `demo`.`sales`.`analytics_orders` AS `orders` "
        "JOIN `demo`.`sales`.`analytics_customers` AS `customers` "
        "ON `orders`.`customer_id` = `customers`.`customer_id` "
        "WHERE DATE_TRUNC('MONTH', `orders`.`order_date`) = :p0 "
        "GROUP BY 1 ORDER BY 1 LIMIT 100"
    )
    assert compiled.sources == (
        "demo.sales.analytics_orders",
        "demo.sales.analytics_customers",
    )


def test_time_trend_uses_date_trunc_bucket(model):
    compiled = compile_query(
        model,
        SemanticQuery(
            metrics=("revenue",),
            time_dimension="order_date",
            time_grain=TimeGrain.MONTH,
        ),
    )
    assert compiled.sql == (
        "SELECT DATE_TRUNC('MONTH', `order_date`) AS `order_date_month`, "
        "SUM(CASE WHEN `status` <> 'C' THEN `amount` END) AS `revenue` "
        "FROM `demo`.`sales`.`analytics_orders` "
        "GROUP BY 1 ORDER BY 1 LIMIT 100"
    )


def test_unknown_metric_is_rejected(model):
    with pytest.raises(SemanticCompileError, match="unknown metrics"):
        compile_query(model, SemanticQuery(metrics=("profit",)))


def test_mixed_metric_sources_are_rejected(model):
    with pytest.raises(SemanticCompileError, match="share a source"):
        compile_query(model, SemanticQuery(metrics=("revenue", "customer_count")))


def test_time_grain_requires_date_dimension(model):
    query = SemanticQuery(
        metrics=("revenue",),
        time_dimension="order_status",
        time_grain=TimeGrain.MONTH,
    )
    with pytest.raises(SemanticCompileError, match="not date"):
        compile_query(model, query)


def test_time_dimension_requires_grain(model):
    query = SemanticQuery(metrics=("revenue",), time_dimension="order_date")
    with pytest.raises(SemanticCompileError, match="together"):
        compile_query(model, query)


def test_limit_bounds_are_enforced():
    with pytest.raises(ValidationError):
        SemanticQuery(metrics=("revenue",), limit=0)
    with pytest.raises(ValidationError):
        SemanticQuery(metrics=("revenue",), limit=1001)


def test_normalize_time_value_accepts_period_shorthand():
    assert normalize_time_value(TimeGrain.MONTH, "2024-03") == "2024-03-01"
    assert normalize_time_value(TimeGrain.YEAR, "2024") == "2024-01-01"
    assert normalize_time_value(TimeGrain.DAY, "2024-03-15") == "2024-03-15"
    with pytest.raises(SemanticCompileError, match="ISO date"):
        normalize_time_value(TimeGrain.MONTH, "March 2024")


def test_truncate_date_matches_sql_semantics():
    assert truncate_date(TimeGrain.WEEK, "2024-03-15") == "2024-03-11"
    assert truncate_date(TimeGrain.QUARTER, "2024-05-20") == "2024-04-01"
    assert truncate_date(TimeGrain.YEAR, "2024-05-20") == "2024-01-01"


def test_row_query_uses_only_declared_identifiers_and_typed_parameters(model):
    compiled = compile_rows(
        model,
        RowQuery(
            source="orders",
            fields=("order_id", "order_amount"),
            filters=(
                RowFilter(
                    field="order_status",
                    operator=RowOperator.IN,
                    value=("S", "P"),
                ),
                RowFilter(
                    field="order_amount",
                    operator=RowOperator.GTE,
                    value=100,
                ),
            ),
            order_by=(
                RowOrder(
                    field="order_amount",
                    direction=OrderDirection.DESC,
                ),
            ),
            limit=10,
            reason="show governed order details",
        ),
    )

    assert compiled.sql == (
        "SELECT `order_id` AS `order_id`, `amount` AS `order_amount` "
        "FROM `demo`.`sales`.`analytics_orders` "
        "WHERE `status` IN (:r0_0, :r0_1) AND `amount` >= :r1 "
        "ORDER BY `amount` DESC LIMIT 10"
    )
    assert [(item.value, item.type) for item in compiled.parameters] == [
        ("S", "STRING"),
        ("P", "STRING"),
        ("100", "DECIMAL"),
    ]


@pytest.mark.parametrize(
    "query",
    [
        RowQuery(
            source="orders",
            fields=("customer_region",),
            reason="cross-source field",
        ),
        RowQuery(
            source="orders",
            fields=("order_id",),
            filters=(
                RowFilter(
                    field="order_status",
                    operator=RowOperator.GT,
                    value="S",
                ),
            ),
            reason="invalid string comparison",
        ),
    ],
)
def test_row_query_rejects_cross_source_and_type_unsafe_plans(model, query):
    with pytest.raises(SemanticCompileError):
        compile_rows(model, query)


def test_row_query_rejects_non_finite_numeric_values(model):
    query = RowQuery(
        source="orders",
        fields=("order_id",),
        filters=(
            RowFilter(
                field="order_amount",
                operator=RowOperator.EQ,
                value=float("nan"),
            ),
        ),
        reason="invalid numeric value",
    )

    with pytest.raises(SemanticCompileError, match="finite"):
        compile_rows(model, query)
