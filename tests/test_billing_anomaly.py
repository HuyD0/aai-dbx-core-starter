"""Detection-math contract for the observed-cost anomaly watch."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from aai_core.billing.anomaly import (
    EXIT_PASS,
    EXIT_THRESHOLD_FAILED,
    DailySpend,
    DetectionConfig,
    Dimension,
    detect_anomalies,
    render_report,
)
from aai_core.billing.errors import StaleUsageDataError

EVALUATION = date(2026, 8, 18)
CONFIG = DetectionConfig()


def _row(dimension, key, day, amount, currency="USD"):
    return DailySpend(
        usage_date=day,
        dimension=dimension,
        key=key,
        amount=amount,
        currency=currency,
    )


def _series(dimension, key, baseline, observed, currency="USD"):
    """Rows for one series: ``baseline`` amounts backwards from yesterday."""

    rows = [
        _row(dimension, key, EVALUATION - timedelta(days=offset), amount, currency)
        for offset, amount in enumerate(baseline, start=1)
    ]
    rows.append(_row(dimension, key, EVALUATION, observed, currency))
    return rows


def _account(observed=100.0):
    return _series(Dimension.ACCOUNT, "account", [100.0] * 27, observed)


def test_constant_baseline_spike_is_flagged():
    rows = _account() + _series(Dimension.PROJECT, "checkout", [100.0] * 27, 240.0)
    report = detect_anomalies(rows, CONFIG, evaluation_date=EVALUATION)
    assert report.exit_code == EXIT_THRESHOLD_FAILED
    assert not report.passed
    (anomaly,) = report.anomalies
    assert anomaly.kind == "spike"
    assert anomaly.dimension is Dimension.PROJECT
    assert anomaly.key == "checkout"
    assert anomaly.baseline_median == 100.0
    assert anomaly.baseline_mad == 0.0
    assert anomaly.threshold == 100.0
    assert anomaly.delta == 140.0
    assert anomaly.history_days == 27


def test_min_delta_floor_suppresses_small_spikes():
    rows = _account() + _series(Dimension.PROJECT, "checkout", [100.0] * 27, 105.0)
    report = detect_anomalies(rows, CONFIG, evaluation_date=EVALUATION)
    assert report.passed
    assert report.exit_code == EXIT_PASS


def test_mad_scales_the_threshold_for_noisy_baselines():
    baseline = [100.0, 100.0, 100.0, 105.0, 95.0, 110.0, 90.0, 95.0]
    # median 100; deviations sorted [0,0,0,5,5,5,10,10] -> MAD 5 ->
    # threshold 100 + 4 * 5 = 120.
    within = _account() + _series(Dimension.PRODUCT, "JOBS", baseline, 118.0)
    report = detect_anomalies(within, CONFIG, evaluation_date=EVALUATION)
    assert report.passed
    beyond = _account() + _series(Dimension.PRODUCT, "JOBS", baseline, 125.0)
    report = detect_anomalies(beyond, CONFIG, evaluation_date=EVALUATION)
    assert [anomaly.key for anomaly in report.anomalies] == ["JOBS"]


def test_missing_baseline_days_are_zero_filled_from_first_seen():
    # 3 spend days out of 10 since first seen: the median must count the
    # seven $0 days, so a $50 day is a spike; without zero-fill the median
    # would be 300 and nothing would flag.
    first_seen = EVALUATION - timedelta(days=10)
    rows = _account() + [
        _row(Dimension.PROJECT, "sparse", first_seen + timedelta(days=offset), 300.0)
        for offset in (0, 3, 6)
    ]
    rows.append(_row(Dimension.PROJECT, "sparse", EVALUATION, 50.0))
    report = detect_anomalies(rows, CONFIG, evaluation_date=EVALUATION)
    (anomaly,) = report.anomalies
    assert anomaly.key == "sparse"
    assert anomaly.baseline_median == 0.0
    assert anomaly.history_days == 10


def test_rows_outside_the_lookback_window_are_ignored():
    ancient = _row(
        Dimension.PROJECT,
        "checkout",
        EVALUATION - timedelta(days=400),
        1_000_000.0,
    )
    rows = [
        ancient,
        *_account(),
        *_series(Dimension.PROJECT, "checkout", [100.0] * 27, 105.0),
    ]
    report = detect_anomalies(rows, CONFIG, evaluation_date=EVALUATION)
    assert report.passed


def test_short_history_series_is_recorded_not_evaluated():
    rows = _account() + _series(Dimension.PROJECT, "newproj", [10.0] * 3, 5_000.0)
    report = detect_anomalies(rows, CONFIG, evaluation_date=EVALUATION)
    assert report.passed
    assert report.skipped_short_history == ("project:newproj:USD",)
    assert report.series_evaluated == 1  # the account series


def test_new_spend_rule_flags_first_day_spend_above_the_floor():
    rows = _account() + [_row(Dimension.PROJECT, "brand-new", EVALUATION, 50.0)]
    report = detect_anomalies(rows, CONFIG, evaluation_date=EVALUATION)
    (anomaly,) = report.anomalies
    assert anomaly.kind == "new_spend"
    assert anomaly.history_days == 0
    assert anomaly.threshold == CONFIG.new_spend_floor
    quiet = _account() + [_row(Dimension.PROJECT, "brand-new", EVALUATION, 5.0)]
    report = detect_anomalies(quiet, CONFIG, evaluation_date=EVALUATION)
    assert report.passed
    assert report.skipped_short_history == ()


def test_untagged_project_bucket_is_an_ordinary_series():
    rows = _account() + _series(Dimension.PROJECT, "untagged", [20.0] * 27, 500.0)
    report = detect_anomalies(rows, CONFIG, evaluation_date=EVALUATION)
    assert [anomaly.key for anomaly in report.anomalies] == ["untagged"]


def test_missing_account_row_on_evaluation_day_is_stale_not_zero():
    without_account = _series(Dimension.PROJECT, "checkout", [100.0] * 27, 100.0)
    with pytest.raises(StaleUsageDataError):
        detect_anomalies(without_account, CONFIG, evaluation_date=EVALUATION)
    yesterday_only = _account()[:-1]
    with pytest.raises(StaleUsageDataError):
        detect_anomalies(yesterday_only, CONFIG, evaluation_date=EVALUATION)
    with pytest.raises(StaleUsageDataError):
        detect_anomalies([], CONFIG, evaluation_date=EVALUATION)


def test_anomalies_are_deterministically_ordered():
    rows = (
        _account(5_000.0)
        + _series(Dimension.PROJECT, "zeta", [100.0] * 27, 400.0)
        + _series(Dimension.PROJECT, "alpha", [100.0] * 27, 400.0)
    )
    report = detect_anomalies(rows, CONFIG, evaluation_date=EVALUATION)
    keys = [(anomaly.dimension.value, anomaly.key) for anomaly in report.anomalies]
    assert keys == sorted(keys)
    assert ("account", "account") in keys


def test_detection_config_rejects_invalid_thresholds():
    with pytest.raises(ValidationError):
        DetectionConfig(min_history=28, lookback_days=28)
    with pytest.raises(ValidationError):
        DetectionConfig(lookback_days=7)
    with pytest.raises(ValidationError):
        DetectionConfig(sensitivity=0.0)
    with pytest.raises(ValidationError):
        DetectionConfig(min_delta=-1.0)
    with pytest.raises(ValidationError):
        DetectionConfig(new_spend_floor=float("inf"))


def test_daily_spend_rejects_non_finite_amounts_and_parses_iso_dates():
    with pytest.raises(ValidationError):
        _row(Dimension.ACCOUNT, "account", EVALUATION, float("nan"))
    parsed = DailySpend(
        usage_date="2026-08-18",
        dimension="project",
        key="checkout",
        amount=7,
        currency="USD",
    )
    assert parsed.usage_date == EVALUATION
    assert parsed.dimension is Dimension.PROJECT
    assert parsed.amount == 7.0


def test_render_report_names_the_findings_and_the_exit_code():
    rows = _account() + _series(Dimension.PROJECT, "checkout", [100.0] * 27, 240.0)
    report = detect_anomalies(rows, CONFIG, evaluation_date=EVALUATION)
    rendered = render_report(report)
    assert "ANOMALIES DETECTED: 1" in rendered
    assert "project=checkout" in rendered
    assert "exit code: 2" in rendered
    quiet = detect_anomalies(_account(), CONFIG, evaluation_date=EVALUATION)
    rendered = render_report(quiet)
    assert "no anomalies detected" in rendered
    assert "exit code: 0" in rendered
