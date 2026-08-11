"""De-identified review and outcome signals for the improvement loop."""

from __future__ import annotations

import re
from enum import StrEnum
from hashlib import sha256
from typing import Any

from pydantic import Field, field_validator

from aai_core.contracts import ContractModel
from aai_core.monitoring import FeedbackSourceKind, log_feedback
from email_support_agent.contracts import ReviewAction, ReviewReason

_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class DeliveryOutcome(StrEnum):
    DELIVERED = "delivered"
    BOUNCED = "bounced"
    COMPLAINED = "complained"
    FAILED = "failed"


class SignalLinkage(ContractModel):
    trace_id: str = Field(min_length=1, max_length=256)
    session_ref: str
    case_ref: str
    application_release: str = Field(min_length=1, max_length=128)
    proposal_digest: str | None = None
    occurred_at: str

    @field_validator("trace_id")
    @classmethod
    def require_opaque_trace(cls, value: str) -> str:
        if "@" in value or value.strip() != value:
            raise ValueError("trace_id must be opaque and normalized")
        return value

    @field_validator("session_ref", "case_ref", "proposal_digest")
    @classmethod
    def require_digest(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_REF.fullmatch(value):
            raise ValueError("feedback linkage must use sha256:<hex> references")
        return value

    @field_validator("occurred_at")
    @classmethod
    def require_timestamp(cls, value: str) -> str:
        if not _TIMESTAMP.fullmatch(value):
            raise ValueError("occurred_at must be an ISO-8601 timestamp with timezone")
        return value


class ReviewFeedbackSignal(ContractModel):
    linkage: SignalLinkage
    action: ReviewAction
    reason: ReviewReason
    reviewer_group: str = Field(pattern=r"^group:[A-Za-z0-9._-]{1,64}$")
    draft_edit_distance: float = Field(ge=0.0, le=1.0)


class OutcomeFeedbackSignal(ContractModel):
    linkage: SignalLinkage
    delivery_outcome: DeliveryOutcome
    resolved_first_contact: bool
    customer_reopened_7d: bool
    source_id: str = Field(pattern=r"^code:[A-Za-z0-9._-]{1,64}$")


def feedback_ref(namespace: str, value: str) -> str:
    """Create a trace-safe join key without exposing an operational id."""

    return (
        "sha256:"
        + sha256(f"email-support-feedback:v1:{namespace}:{value}".encode()).hexdigest()
    )


def log_review_feedback(
    signal: ReviewFeedbackSignal,
    *,
    mlflow_module: Any | None = None,
) -> None:
    metadata = _metadata(signal.linkage)
    for name, value in (
        ("human_review_decision", signal.action.value),
        ("review_reason", signal.reason.value),
        ("draft_edit_distance", signal.draft_edit_distance),
        ("approved_unchanged", signal.action is ReviewAction.APPROVE),
    ):
        log_feedback(
            trace_id=signal.linkage.trace_id,
            name=name,
            value=value,
            source_kind=FeedbackSourceKind.HUMAN,
            source_id=signal.reviewer_group,
            metadata=metadata,
            mlflow_module=mlflow_module,
        )


def log_outcome_feedback(
    signal: OutcomeFeedbackSignal,
    *,
    mlflow_module: Any | None = None,
) -> None:
    metadata = _metadata(signal.linkage)
    for name, value in (
        ("delivery_outcome", signal.delivery_outcome.value),
        ("resolved_first_contact", signal.resolved_first_contact),
        ("customer_reopened_7d", signal.customer_reopened_7d),
    ):
        log_feedback(
            trace_id=signal.linkage.trace_id,
            name=name,
            value=value,
            source_kind=FeedbackSourceKind.CODE,
            source_id=signal.source_id,
            metadata=metadata,
            mlflow_module=mlflow_module,
        )


def _metadata(linkage: SignalLinkage) -> dict[str, str]:
    metadata = {
        "session_ref": linkage.session_ref,
        "case_ref": linkage.case_ref,
        "application_release": linkage.application_release,
        "occurred_at": linkage.occurred_at,
    }
    if linkage.proposal_digest is not None:
        metadata["proposal_digest"] = linkage.proposal_digest
    return metadata
