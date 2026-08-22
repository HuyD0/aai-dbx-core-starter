"""Bounded-window loader for daily spend from ``system.billing``.

Only this module touches Spark, and only inside a function, so the SDK
imports cleanly without pyspark and unit tests inject a fake session. The
query is one bounded scan (never a page-load or fleet-wide sweep), and the
amounts are list prices from ``system.billing.list_prices`` — a consistent
bias across baseline and observation, which is all relative detection
needs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aai_core.billing.anomaly import (
    ACCOUNT_KEY,
    UNTAGGED_PROJECT,
    DailySpend,
    DetectionConfig,
    Dimension,
)
from aai_core.billing.errors import (
    REQUIRED_GRANTS,
    BillingConfigError,
    UsageQueryError,
)


def build_daily_spend_sql(config: DetectionConfig, evaluation_date: date) -> str:
    """Return the one bounded-window statement producing per-series spend.

    Only validated ``date`` objects are interpolated — no caller string
    ever reaches the SQL text. ``SUM`` nets ``record_type`` corrections
    (retraction rows carry negative quantities). Data-level NULLs are
    normalized inside the CTE so the ``GROUPING`` flags stay unambiguous.
    """

    window_start = evaluation_date - timedelta(days=config.lookback_days - 1)
    return f"""
WITH priced AS (
  SELECT
    u.usage_date,
    COALESCE(CAST(u.workspace_id AS STRING), 'none') AS workspace_id,
    COALESCE(u.billing_origin_product, 'unknown') AS billing_origin_product,
    COALESCE(u.custom_tags['project'], '{UNTAGGED_PROJECT}') AS project,
    lp.currency_code AS currency,
    u.usage_quantity * lp.pricing.effective_list.default AS amount
  FROM system.billing.usage AS u
  JOIN system.billing.list_prices AS lp
    ON u.sku_name = lp.sku_name
   AND u.cloud = lp.cloud
   AND u.usage_unit = lp.usage_unit
   AND u.usage_end_time >= lp.price_start_time
   AND (lp.price_end_time IS NULL OR u.usage_end_time < lp.price_end_time)
  WHERE u.usage_date BETWEEN DATE'{window_start.isoformat()}'
    AND DATE'{evaluation_date.isoformat()}'
)
SELECT
  usage_date,
  currency,
  CASE
    WHEN GROUPING(workspace_id) = 0 THEN '{Dimension.WORKSPACE.value}'
    WHEN GROUPING(billing_origin_product) = 0 THEN '{Dimension.PRODUCT.value}'
    WHEN GROUPING(project) = 0 THEN '{Dimension.PROJECT.value}'
    ELSE '{Dimension.ACCOUNT.value}'
  END AS dimension,
  CASE
    WHEN GROUPING(workspace_id) = 0 THEN workspace_id
    WHEN GROUPING(billing_origin_product) = 0 THEN billing_origin_product
    WHEN GROUPING(project) = 0 THEN project
    ELSE '{ACCOUNT_KEY}'
  END AS series_key,
  SUM(amount) AS amount
FROM priced
GROUP BY GROUPING SETS (
  (usage_date, currency),
  (usage_date, currency, workspace_id),
  (usage_date, currency, billing_origin_product),
  (usage_date, currency, project)
)
"""


def load_daily_spend(
    config: DetectionConfig,
    *,
    evaluation_date: date,
    spark: Any | None = None,
) -> list[DailySpend]:
    """Run the bounded query on the injected or ambient Spark session."""

    session = spark if spark is not None else _spark_session()
    statement = build_daily_spend_sql(config, evaluation_date)
    try:
        collected = session.sql(statement).collect()
    except Exception as error:  # noqa: BLE001 - single query boundary
        raise UsageQueryError(
            "querying system.billing failed",
            remediation=REQUIRED_GRANTS,
        ) from error
    return [_to_daily_spend(row) for row in collected]


def rows_from_json(path: Path) -> list[DailySpend]:
    """Load spend rows from a JSON array file (offline runs and tests)."""

    try:
        payload = json.loads(path.read_text())
    except OSError as error:
        raise BillingConfigError(f"cannot read input file: {path}") from error
    except json.JSONDecodeError as error:
        raise BillingConfigError(f"input file is not valid JSON: {path}") from error
    if not isinstance(payload, list):
        raise BillingConfigError("input file must contain a JSON array of rows")
    try:
        return [DailySpend.model_validate(item) for item in payload]
    except ValidationError as error:
        raise BillingConfigError(f"invalid spend row: {error}") from error


def _to_daily_spend(row: Any) -> DailySpend:
    data: Mapping[str, Any] = row.asDict() if hasattr(row, "asDict") else row
    return DailySpend(
        usage_date=data["usage_date"],
        dimension=Dimension(str(data["dimension"])),
        key=str(data["series_key"]),
        amount=float(data["amount"]),
        currency=str(data["currency"]),
    )


def _spark_session() -> Any:  # pragma: no cover - requires a Databricks runtime
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()
