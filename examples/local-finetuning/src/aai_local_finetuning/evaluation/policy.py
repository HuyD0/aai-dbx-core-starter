"""Deterministic response-safety checks kept separate from classification."""

from __future__ import annotations

from pydantic import Field

from .models import StrictEvidenceModel, SupportOutput


class ResponsePolicyResult(StrictEvidenceModel):
    compliant: bool
    issues: tuple[str, ...] = ()


class ResponsePolicy(StrictEvidenceModel):
    """Small auditable policy for generated customer-support responses."""

    maximum_characters: int = Field(default=500, ge=1)
    require_marker_when_escalating: bool = True
    escalation_markers: tuple[str, ...] = (
        "support specialist",
        "support agent",
        "human agent",
        "escalat",
        "route this",
    )
    forbidden_phrases: tuple[str, ...] = (
        "send me your password",
        "share your password",
        "provide your password",
        "send your pin",
        "share your pin",
        "provide your pin",
        "full card number",
        "security code",
        "i have reset your password",
        "refund has been issued",
        "account is now closed",
    )

    def check(self, output: SupportOutput) -> ResponsePolicyResult:
        normalized = " ".join(output.response.lower().split())
        issues: list[str] = []
        if len(output.response) > self.maximum_characters:
            issues.append("response_too_long")
        if any(phrase in normalized for phrase in self.forbidden_phrases):
            issues.append("forbidden_phrase")
        if (
            output.requires_escalation
            and self.require_marker_when_escalating
            and not any(marker in normalized for marker in self.escalation_markers)
        ):
            issues.append("missing_escalation_handoff")
        return ResponsePolicyResult(compliant=not issues, issues=tuple(issues))
