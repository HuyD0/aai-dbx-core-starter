"""Observed-cost anomaly detection over the ``system.billing`` tables.

The scheduled bundle job (``resources/cost_anomaly_job.yml``) runs
``python -m aai_core.billing.cli detect`` daily. Exit code 2 marks an
anomaly and exit code 1 a broken or blind monitor; both fail the run so
the job's email notification alerts the owning group. Detection math is
pure stdlib and unit-tested offline; only :mod:`aai_core.billing.usage`
touches Spark, lazily. See ``docs/platform-operations.md``.
"""

from aai_core.billing.anomaly import (
    Anomaly,
    AnomalyReport,
    DailySpend,
    DetectionConfig,
    Dimension,
    detect_anomalies,
)

__all__ = [
    "Anomaly",
    "AnomalyReport",
    "DailySpend",
    "DetectionConfig",
    "Dimension",
    "detect_anomalies",
]
