"""Stable error taxonomy for the observed-cost anomaly watch.

Every failure carries a machine-readable ``code`` and, where the fix is
known, a ``remediation`` hint. The CLI maps any :class:`BillingError` to
exit code 1 (configuration/state/runtime error); an anomaly finding is not
an error — it is a detection result and exits with code 2.
"""

from __future__ import annotations

from aai_core.exceptions import AaiCoreError

REQUIRED_GRANTS = (
    "ask the platform owner to grant the job's run-as principal USE CATALOG "
    "on `system`, USE SCHEMA on `system.billing`, and SELECT on "
    "`system.billing.usage` and `system.billing.list_prices` through the "
    "approved external platform process (see docs/cloud-setup.md)"
)


class BillingError(AaiCoreError):
    """Base class for every billing-watch failure."""

    code = "aai_core.billing.error"


class BillingConfigError(BillingError):
    """The detection configuration or an input file is invalid."""

    code = "aai_core.billing.config_invalid"


class UsageQueryError(BillingError):
    """The billing system tables could not be queried."""

    code = "aai_core.billing.usage_query_failed"


class StaleUsageDataError(BillingError):
    """No account-level spend row exists for the evaluation day.

    Unknown cost is not zero: absent billing rows mean the monitor is blind
    (missing grants, billing-data latency, or a broken query), never that
    nothing was spent. The run fails loudly instead of reporting "no
    anomaly".
    """

    code = "aai_core.billing.usage_stale"
