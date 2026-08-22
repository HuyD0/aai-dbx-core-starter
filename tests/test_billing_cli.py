"""CLI wiring, data seams, and the SQL builder for the cost anomaly watch."""

from __future__ import annotations

import json
import runpy
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_core.billing.anomaly import DetectionConfig, Dimension
from aai_core.billing.cli import EXIT_ERROR, EXIT_PASS, EXIT_THRESHOLD_FAILED, main
from aai_core.billing.errors import BillingConfigError, UsageQueryError
from aai_core.billing.usage import (
    build_daily_spend_sql,
    load_daily_spend,
    rows_from_json,
)

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = date(2026, 8, 18)


def _rows(observed=100.0, project_observed=None):
    rows = []
    for offset in range(28):
        day = (EVALUATION - timedelta(days=offset)).isoformat()
        amount = observed if offset == 0 else 100.0
        rows.append(
            {
                "usage_date": day,
                "dimension": "account",
                "key": "account",
                "amount": amount,
                "currency": "USD",
            }
        )
        if project_observed is not None:
            rows.append(
                {
                    "usage_date": day,
                    "dimension": "project",
                    "key": "checkout",
                    "amount": project_observed if offset == 0 else 100.0,
                    "currency": "USD",
                }
            )
    return rows


def _input_file(tmp_path, rows):
    path = tmp_path / "spend.json"
    path.write_text(json.dumps(rows))
    return str(path)


def _detect(path, *extra):
    return main(["detect", "--input", path, "--evaluation-date", "2026-08-18", *extra])


def test_detect_passes_on_steady_spend(tmp_path, capsys):
    path = _input_file(tmp_path, _rows())
    assert _detect(path) == EXIT_PASS
    output = capsys.readouterr().out
    assert "no anomalies detected" in output


def test_detect_flags_a_spike_and_emits_one_json_document(tmp_path, capsys):
    path = _input_file(tmp_path, _rows(project_observed=240.0))
    assert _detect(path, "--json") == EXIT_THRESHOLD_FAILED
    output = capsys.readouterr().out.strip()
    document = json.loads(output)
    assert "\n" not in output
    assert document["exit_code"] == EXIT_THRESHOLD_FAILED
    assert document["passed"] is False
    assert document["evaluation_date"] == "2026-08-18"
    (anomaly,) = document["anomalies"]
    assert anomaly["kind"] == "spike"
    assert anomaly["key"] == "checkout"


def test_stale_billing_data_exits_error_with_remediation(tmp_path, capsys):
    rows = [row for row in _rows() if row["usage_date"] != "2026-08-18"]
    path = _input_file(tmp_path, rows)
    assert _detect(path) == EXIT_ERROR
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "Remediation" in captured.err


def test_invalid_configuration_exits_error(tmp_path, capsys):
    path = _input_file(tmp_path, _rows())
    assert _detect(path, "--min-history", "40") == EXIT_ERROR
    assert "invalid detection configuration" in capsys.readouterr().err


def test_invalid_evaluation_date_exits_error(tmp_path, capsys):
    path = _input_file(tmp_path, _rows())
    code = main(["detect", "--input", path, "--evaluation-date", "not-a-date"])
    assert code == EXIT_ERROR
    assert "ISO date" in capsys.readouterr().err


def test_threshold_flags_change_the_outcome(tmp_path):
    path = _input_file(tmp_path, _rows(project_observed=115.0))
    assert _detect(path) == EXIT_THRESHOLD_FAILED
    assert _detect(path, "--min-delta", "20.0") == EXIT_PASS


def test_sql_builder_bounds_the_window_and_covers_every_dimension():
    statement = build_daily_spend_sql(DetectionConfig(), EVALUATION)
    assert "system.billing.usage" in statement
    assert "system.billing.list_prices" in statement
    assert "DATE'2026-07-22'" in statement
    assert "DATE'2026-08-18'" in statement
    assert "GROUPING SETS" in statement
    assert "'untagged'" in statement
    for join_key in ("sku_name", "cloud", "usage_unit"):
        assert f"u.{join_key} = lp.{join_key}" in statement
    assert "lp.price_end_time IS NULL" in statement
    for dimension in Dimension:
        assert f"'{dimension.value}'" in statement


class _FakeSession:
    def __init__(self, rows=None, error=None):
        self.rows = rows or []
        self.error = error
        self.statements = []

    def sql(self, statement):
        self.statements.append(statement)
        if self.error is not None:
            raise self.error
        rows = self.rows
        return SimpleNamespace(collect=lambda: rows)


def test_load_daily_spend_uses_the_injected_session():
    session = _FakeSession(
        rows=[
            {
                "usage_date": EVALUATION,
                "dimension": "workspace",
                "series_key": "12345",
                "amount": 41.5,
                "currency": "USD",
            }
        ]
    )
    config = DetectionConfig()
    loaded = load_daily_spend(config, evaluation_date=EVALUATION, spark=session)
    (row,) = loaded
    assert row.dimension is Dimension.WORKSPACE
    assert row.key == "12345"
    assert row.amount == 41.5
    (statement,) = session.statements
    assert statement == build_daily_spend_sql(config, EVALUATION)


def test_load_daily_spend_wraps_query_failures_with_the_grant_ask():
    session = _FakeSession(error=RuntimeError("TABLE_OR_VIEW_NOT_FOUND"))
    with pytest.raises(UsageQueryError) as failure:
        load_daily_spend(DetectionConfig(), evaluation_date=EVALUATION, spark=session)
    assert "system.billing" in str(failure.value)
    assert "SELECT" in str(failure.value)


def test_rows_from_json_rejects_bad_payloads(tmp_path):
    not_a_list = tmp_path / "object.json"
    not_a_list.write_text("{}")
    with pytest.raises(BillingConfigError):
        rows_from_json(not_a_list)
    bad_row = tmp_path / "rows.json"
    bad_row.write_text(json.dumps([{"usage_date": "2026-08-18"}]))
    with pytest.raises(BillingConfigError):
        rows_from_json(bad_row)
    with pytest.raises(BillingConfigError):
        rows_from_json(tmp_path / "missing.json")


def test_job_runner_exits_with_the_cli_code(tmp_path, monkeypatch):
    path = _input_file(tmp_path, _rows())
    runner = ROOT / "src" / "jobs" / "detect_cost_anomalies.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(runner), "detect", "--input", path, "--evaluation-date", "2026-08-18"],
    )
    with pytest.raises(SystemExit) as outcome:
        runpy.run_path(str(runner), run_name="__main__")
    assert outcome.value.code == EXIT_PASS
