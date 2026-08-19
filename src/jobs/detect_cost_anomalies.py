"""Scheduled cost-anomaly watch entry point (``spark_python_task``).

Deployed by ``resources/cost_anomaly_job.yml`` with the bundle-built
aai-core wheel installed on the job cluster. The CLI's exit code is the
job result: 0 pass, 2 anomaly detected, 1 configuration or runtime error.
Both non-zero codes fail the run, so ``email_notifications.on_failure``
alerts the owning group even when the monitor itself breaks.
"""

from __future__ import annotations

import sys

from aai_core.billing.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
