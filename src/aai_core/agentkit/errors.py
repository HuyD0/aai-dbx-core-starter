"""Stable error taxonomy for the agent-evaluation toolkit.

Every failure carries a machine-readable ``code`` and, where the fix is
known, a ``remediation`` hint. The CLI maps any :class:`AgentkitError` to
exit code 1 (configuration/state/runtime error); threshold failures are not
errors — they are gate results and exit with code 2.
"""

from __future__ import annotations

from aai_core.exceptions import AaiCoreError


class AgentkitError(AaiCoreError):
    """Base class for every agentkit failure."""

    code = "aai_core.agentkit.error"


class ConfigError(AgentkitError):
    code = "aai_core.agentkit.config_invalid"


class UnknownScorerError(AgentkitError):
    code = "aai_core.agentkit.unknown_scorer"


class TargetResolutionError(AgentkitError):
    code = "aai_core.agentkit.target_unresolved"


class TargetContractError(AgentkitError):
    code = "aai_core.agentkit.target_contract"


class TargetInvocationError(AgentkitError):
    code = "aai_core.agentkit.target_invocation"


class BaselineMissingError(AgentkitError):
    code = "aai_core.agentkit.baseline_missing"


class EvidenceMissingError(AgentkitError):
    code = "aai_core.agentkit.evidence_missing"


class BudgetExceededError(AgentkitError):
    code = "aai_core.agentkit.budget_exceeded"


class MissingExtraError(AgentkitError):
    code = "aai_core.agentkit.missing_extra"


def missing_extra(feature: str, extra: str) -> MissingExtraError:
    """Build the standard missing-optional-dependency error."""

    return MissingExtraError(
        f"{feature} requires the `{extra}` extra. From an aai-core checkout "
        "run `make examples-install` and use `.venv/bin/python`; in a "
        f"consuming environment install `aai-core[{extra}]`."
    )
