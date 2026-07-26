"""Root error type for the SDK.

Every aai-core error derives from :class:`AaiCoreError`, so applications can
catch one type at their boundary. Errors carry a stable machine-readable
``code`` and, where the fix is known, a human ``remediation`` hint — the error
message is the platform's first support channel.
"""

from __future__ import annotations


class AaiCoreError(RuntimeError):
    """Base class for every error raised by aai-core."""

    code: str = "aai_core.error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        remediation: str | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.remediation = remediation

    def __str__(self) -> str:
        base = super().__str__()
        if self.remediation:
            return f"{base} Remediation: {self.remediation}"
        return base
