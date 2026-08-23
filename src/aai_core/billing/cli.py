"""The observed-cost anomaly watch command line.

    python -m aai_core.billing.cli detect    evaluate the latest complete day

Exit codes are the same CI contract as ``agentkit``: 0 passed, 2 ran and at
least one series is anomalous, 1 runtime or configuration error — including
stale billing data, because unknown cost is not zero. The scheduled bundle
job treats both non-zero codes as a failed run, so its email notification
alerts the owning group even when the monitor itself breaks.

Only the standard library is imported at module load; the ``detect``
handler imports what it needs, so ``--help`` stays instant and pyspark is
touched only inside a Databricks runtime (or replaced by ``--input``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

EXIT_PASS = 0
EXIT_ERROR = 1
EXIT_THRESHOLD_FAILED = 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], arguments.handler)
    try:
        return handler(arguments)
    except Exception as error:  # noqa: BLE001 - single CLI boundary
        from aai_core.exceptions import AaiCoreError

        if isinstance(error, AaiCoreError):
            print(f"error: {error}", file=sys.stderr)
            return EXIT_ERROR
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aai-billing",
        description=(
            "Detect anomalous observed spend in system.billing.usage; a "
            "non-zero exit fails the scheduled job so its notification "
            "alerts the owning group."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    detect = subcommands.add_parser(
        "detect", help="evaluate the latest complete day of spend"
    )
    detect.add_argument("--lookback-days", type=int, default=28)
    detect.add_argument("--lag-days", type=int, default=1)
    detect.add_argument("--sensitivity", type=float, default=4.0)
    detect.add_argument("--min-history", type=int, default=7)
    detect.add_argument("--min-delta", type=float, default=10.0)
    detect.add_argument("--new-spend-floor", type=float, default=10.0)
    detect.add_argument(
        "--evaluation-date",
        default=None,
        help="ISO date to evaluate (default: UTC today minus --lag-days)",
    )
    detect.add_argument(
        "--input",
        default=None,
        help="JSON array of spend rows instead of querying system.billing",
    )
    detect.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit exactly one JSON document on stdout",
    )
    detect.set_defaults(handler=_cmd_detect)
    return parser


def _cmd_detect(arguments: argparse.Namespace) -> int:
    from aai_core.billing.anomaly import detect_anomalies, render_report
    from aai_core.billing.usage import load_daily_spend, rows_from_json

    config = _config_from(arguments)
    evaluation_date = _evaluation_date(arguments, lag_days=config.lag_days)
    if arguments.input:
        rows = rows_from_json(Path(arguments.input))
    else:
        rows = load_daily_spend(config, evaluation_date=evaluation_date)
    report = detect_anomalies(rows, config, evaluation_date=evaluation_date)
    if arguments.as_json:
        document = report.model_dump(mode="json")
        document["passed"] = report.passed
        document["exit_code"] = report.exit_code
        print(json.dumps(document, sort_keys=True))
    else:
        print(render_report(report))
    return report.exit_code


def _config_from(arguments: argparse.Namespace) -> Any:
    from pydantic import ValidationError

    from aai_core.billing.anomaly import DetectionConfig
    from aai_core.billing.errors import BillingConfigError

    try:
        return DetectionConfig(
            lookback_days=arguments.lookback_days,
            lag_days=arguments.lag_days,
            sensitivity=arguments.sensitivity,
            min_history=arguments.min_history,
            min_delta=arguments.min_delta,
            new_spend_floor=arguments.new_spend_floor,
        )
    except ValidationError as error:
        raise BillingConfigError(f"invalid detection configuration: {error}") from error


def _evaluation_date(arguments: argparse.Namespace, *, lag_days: int) -> Any:
    from datetime import UTC, date, datetime, timedelta

    from aai_core.billing.errors import BillingConfigError

    if arguments.evaluation_date:
        try:
            return date.fromisoformat(arguments.evaluation_date)
        except ValueError as error:
            raise BillingConfigError(
                "--evaluation-date must be an ISO date (YYYY-MM-DD)"
            ) from error
    return datetime.now(UTC).date() - timedelta(days=lag_days)


if __name__ == "__main__":
    raise SystemExit(main())
