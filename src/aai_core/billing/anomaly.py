"""Robust anomaly detection over daily observed spend (pure stdlib math).

``system.billing`` rows arrive through :mod:`aai_core.billing.usage`; this
module owns the detection contract: strict input and evidence models, the
median + MAD baseline, the new-spend rule, and the fail-loud stale-data
guard. The exit codes are the same stable CI contract ``agentkit`` uses:
``0`` pass, ``2`` anomaly detected, ``1`` configuration or runtime error.

The baseline is deliberately plain: per series, the median of the trailing
daily amounts plus ``sensitivity`` times the raw median absolute deviation
(no 1.4826 normality constant). A constant baseline has MAD 0, so the
absolute ``min_delta`` floor is what separates signal from noise there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, timedelta
from enum import StrEnum
from statistics import median
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from aai_core.billing.errors import StaleUsageDataError
from aai_core.contracts import ContractModel

EXIT_PASS = 0
EXIT_ERROR = 1
EXIT_THRESHOLD_FAILED = 2

ACCOUNT_KEY = "account"
UNTAGGED_PROJECT = "untagged"

_SeriesIdentity = tuple["Dimension", str, str]


class Dimension(StrEnum):
    """Spend series grouping derived from ``system.billing.usage``."""

    ACCOUNT = "account"
    WORKSPACE = "workspace"
    PRODUCT = "product"
    PROJECT = "project"


def _parse_date(value: Any) -> Any:
    if isinstance(value, str):
        return date.fromisoformat(value)
    return value


def _coerce_float(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("amounts must be numeric, not boolean")
    if isinstance(value, int):
        return float(value)
    return value


class DailySpend(ContractModel):
    """One day of observed spend for one series (dimension, key, currency)."""

    usage_date: date
    dimension: Dimension
    key: str = Field(min_length=1)
    amount: float
    currency: str = Field(min_length=1)

    @field_validator("usage_date", mode="before")
    @classmethod
    def parse_usage_date(cls, value: Any) -> Any:
        return _parse_date(value)

    @field_validator("dimension", mode="before")
    @classmethod
    def parse_dimension(cls, value: Any) -> Any:
        if isinstance(value, str) and not isinstance(value, Dimension):
            return Dimension(value)
        return value

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, value: Any) -> Any:
        return _coerce_float(value)

    @field_validator("amount")
    @classmethod
    def finite_amount(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("amount must be finite")
        return value


class DetectionConfig(ContractModel):
    """Thresholds for one daily cost-anomaly evaluation.

    ``min_delta`` and ``new_spend_floor`` are denominated in the list-price
    currency of the billing rows (list price, not a negotiated rate — the
    baseline and the observed day carry the same bias, so detection stays
    relative).
    """

    lookback_days: int = Field(default=28, ge=8, le=365)
    lag_days: int = Field(default=1, ge=0, le=30)
    sensitivity: float = Field(default=4.0, gt=0.0, le=100.0)
    min_history: int = Field(default=7, ge=1)
    min_delta: float = Field(default=10.0, ge=0.0)
    new_spend_floor: float = Field(default=10.0, ge=0.0)

    @field_validator("sensitivity", "min_delta", "new_spend_floor", mode="before")
    @classmethod
    def coerce_thresholds(cls, value: Any) -> Any:
        return _coerce_float(value)

    @field_validator("sensitivity", "min_delta", "new_spend_floor")
    @classmethod
    def finite_thresholds(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("thresholds must be finite")
        return value

    @model_validator(mode="after")
    def baseline_fits_window(self) -> DetectionConfig:
        if self.min_history > self.lookback_days - 1:
            raise ValueError(
                "min_history must leave at least one baseline day inside "
                "lookback_days (the evaluation day is not baseline)"
            )
        return self


class Anomaly(ContractModel):
    """One flagged series on the evaluation day (persisted evidence)."""

    dimension: Dimension
    key: str = Field(min_length=1)
    currency: str = Field(min_length=1)
    evaluation_date: date
    observed: float
    baseline_median: float
    baseline_mad: float
    threshold: float
    delta: float
    kind: Literal["spike", "new_spend"]
    history_days: int = Field(ge=0)


class AnomalyReport(ContractModel):
    """Outcome of one evaluation run."""

    evaluation_date: date
    config: DetectionConfig
    anomalies: tuple[Anomaly, ...] = ()
    series_evaluated: int = Field(ge=0)
    skipped_short_history: tuple[str, ...] = ()

    @field_validator("anomalies", "skipped_short_history", mode="before")
    @classmethod
    def coerce_sequences(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @property
    def passed(self) -> bool:
        """True when no series breached its baseline."""

        return not self.anomalies

    @property
    def exit_code(self) -> int:
        """The CI-contract code for this outcome (0 pass, 2 anomaly)."""

        return EXIT_PASS if self.passed else EXIT_THRESHOLD_FAILED


def detect_anomalies(
    rows: Sequence[DailySpend],
    config: DetectionConfig,
    *,
    evaluation_date: date,
) -> AnomalyReport:
    """Evaluate the latest day of spend against each series' robust baseline.

    Raises :class:`StaleUsageDataError` when no account-level row exists for
    ``evaluation_date`` — unknown cost is never reported as zero.
    """

    window_start = evaluation_date - timedelta(days=config.lookback_days - 1)
    series = _series(rows, window_start=window_start, evaluation_date=evaluation_date)
    _require_fresh_account_row(series, evaluation_date)
    anomalies: list[Anomaly] = []
    skipped: list[str] = []
    evaluated = 0
    for identity in sorted(series):
        outcome = _evaluate_series(
            identity,
            series[identity],
            config,
            window_start=window_start,
            evaluation_date=evaluation_date,
        )
        if outcome == "short-history":
            skipped.append(":".join(identity))
            continue
        evaluated += 1
        if isinstance(outcome, Anomaly):
            anomalies.append(outcome)
    return AnomalyReport(
        evaluation_date=evaluation_date,
        config=config,
        anomalies=tuple(anomalies),
        series_evaluated=evaluated,
        skipped_short_history=tuple(skipped),
    )


def render_report(report: AnomalyReport) -> str:
    """Human-readable summary shown in the job run output."""

    lines = [
        f"cost anomaly evaluation for {report.evaluation_date.isoformat()}",
        f"series evaluated: {report.series_evaluated} "
        f"(short history skipped: {len(report.skipped_short_history)})",
    ]
    if report.passed:
        lines.append("no anomalies detected")
    else:
        lines.append(f"ANOMALIES DETECTED: {len(report.anomalies)}")
        for anomaly in report.anomalies:
            lines.append(
                f"  [{anomaly.kind}] {anomaly.dimension.value}={anomaly.key} "
                f"({anomaly.currency}): observed {anomaly.observed:.2f}, "
                f"baseline median {anomaly.baseline_median:.2f}, threshold "
                f"{anomaly.threshold:.2f}, history {anomaly.history_days}d"
            )
    lines.append(f"exit code: {report.exit_code}")
    return "\n".join(lines)


def _series(
    rows: Sequence[DailySpend],
    *,
    window_start: date,
    evaluation_date: date,
) -> dict[_SeriesIdentity, dict[date, float]]:
    """Sum in-window rows into per-series daily totals (corrections net out)."""

    totals: dict[_SeriesIdentity, dict[date, float]] = {}
    for row in rows:
        if not window_start <= row.usage_date <= evaluation_date:
            continue
        identity = (row.dimension, row.key, row.currency)
        days = totals.setdefault(identity, {})
        days[row.usage_date] = days.get(row.usage_date, 0.0) + row.amount
    return totals


def _require_fresh_account_row(
    series: dict[_SeriesIdentity, dict[date, float]],
    evaluation_date: date,
) -> None:
    for (dimension, _, _), days in series.items():
        if dimension is Dimension.ACCOUNT and evaluation_date in days:
            return
    raise StaleUsageDataError(
        f"system.billing.usage has no account-level rows for "
        f"{evaluation_date.isoformat()}; refusing to report zero spend",
        remediation=(
            "confirm the system.billing grants, allow for billing-data "
            "latency (raise --lag-days), or inspect the query"
        ),
    )


def _evaluate_series(
    identity: _SeriesIdentity,
    days: dict[date, float],
    config: DetectionConfig,
    *,
    window_start: date,
    evaluation_date: date,
) -> Anomaly | Literal["short-history"] | None:
    """Apply the new-spend or spike rule to one series."""

    dimension, key, currency = identity
    observed = days.get(evaluation_date, 0.0)
    baseline_start = max(min(days), window_start)
    history_days = (evaluation_date - baseline_start).days
    if history_days == 0:
        if observed > config.new_spend_floor:
            return Anomaly(
                dimension=dimension,
                key=key,
                currency=currency,
                evaluation_date=evaluation_date,
                observed=observed,
                baseline_median=0.0,
                baseline_mad=0.0,
                threshold=config.new_spend_floor,
                delta=observed,
                kind="new_spend",
                history_days=0,
            )
        return None
    if history_days < config.min_history:
        return "short-history"
    # A baseline day with no row is a $0 day; omitting it would inflate the
    # median for sparse series.
    baseline = [
        days.get(baseline_start + timedelta(days=offset), 0.0)
        for offset in range(history_days)
    ]
    baseline_median = median(baseline)
    baseline_mad = median(abs(value - baseline_median) for value in baseline)
    threshold = baseline_median + config.sensitivity * baseline_mad
    delta = observed - baseline_median
    if observed > threshold and delta > config.min_delta:
        return Anomaly(
            dimension=dimension,
            key=key,
            currency=currency,
            evaluation_date=evaluation_date,
            observed=observed,
            baseline_median=baseline_median,
            baseline_mad=baseline_mad,
            threshold=threshold,
            delta=delta,
            kind="spike",
            history_days=history_days,
        )
    return None
