"""Strict application and evidence boundaries for the solution accelerator."""

from __future__ import annotations

import re
from enum import StrEnum
from hashlib import sha256
from typing import Self

from pydantic import Field, field_validator, model_validator

from aai_core.contracts import ContractModel


class Intent(StrEnum):
    QUESTION = "question"
    FEATURE_REQUEST = "feature_request"
    BUG = "bug"
    BILLING = "billing"
    ACCOUNT = "account"
    SECURITY = "security"
    OTHER = "other"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Route(StrEnum):
    KNOWLEDGE_REPLY = "knowledge_reply"
    BUG_TRACKING = "bug_tracking"
    HUMAN_REVIEW = "human_review"
    ESCALATION = "escalation"


class ReviewAction(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class ReviewReason(StrEnum):
    APPROVED = "approved"
    FACTUAL_EDIT = "factual_edit"
    POLICY_EDIT = "policy_edit"
    NEEDS_INVESTIGATION = "needs_investigation"
    WRONG_ROUTE = "wrong_route"
    UNSAFE = "unsafe"


class ActionKind(StrEnum):
    UPSERT_TICKET = "upsert_ticket"
    ENQUEUE_REPLY = "enqueue_reply"


class OutboxStatus(StrEnum):
    ENQUEUED = "enqueued"
    ALREADY_ENQUEUED = "already_enqueued"


class Disposition(StrEnum):
    PENDING_REVIEW = "pending_review"
    READY = "ready"
    QUEUED = "queued"
    HANDLED_BY_HUMAN = "handled_by_human"


class MeasurementSource(StrEnum):
    OFFLINE_FIXTURE = "offline_fixture"
    CONNECTED = "connected"


_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_PSEUDONYMOUS_REF = re.compile(r"^(?:sha256|hmac-sha256):[0-9a-f]{64}$")
_IDEMPOTENCY_KEY = re.compile(r"^[0-9a-f]{64}$")
_REVIEWER_GROUP = re.compile(r"^group:[A-Za-z0-9._-]{1,64}$")
_DATE_TIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_EMAIL_ADDRESS = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
_PAYMENT_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_BEARER_TOKEN = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}={0,2}\b", re.I)


def obvious_sensitive_fragments(value: str) -> tuple[str, ...]:
    """Return cheap fail-closed signals; an approved DLP layer remains required."""

    findings: list[str] = []
    if _EMAIL_ADDRESS.search(value):
        findings.append("email_address")
    if _PAYMENT_CARD.search(value):
        findings.append("payment_card")
    if _BEARER_TOKEN.search(value):
        findings.append("bearer_token")
    return tuple(findings)


class RedactedEmail(ContractModel):
    """The only email representation allowed into durable graph state.

    MIME bytes, the sender address, and attachments belong in an approved
    encrypted ingress store. The graph receives opaque references and text
    already processed by malware scanning and the enterprise DLP service.
    """

    case_id: str = Field(min_length=3, max_length=128)
    message_id: str = Field(min_length=3, max_length=128)
    thread_id: str = Field(min_length=3, max_length=128)
    tenant_id: str = Field(min_length=3, max_length=128)
    ingress_provider: str = Field(min_length=3, max_length=64)
    access_context_ref: str = Field(min_length=4, max_length=512)
    sender_ref: str
    raw_email_ref: str = Field(min_length=4, max_length=512)
    subject: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=12_000)
    received_at: str
    attachments_scanned: bool
    redaction_complete: bool
    sanitization_version: str = Field(min_length=1, max_length=64)

    @field_validator(
        "case_id",
        "message_id",
        "thread_id",
        "tenant_id",
        "ingress_provider",
    )
    @classmethod
    def require_opaque_identifier(cls, value: str) -> str:
        if not _OPAQUE_ID.fullmatch(value) or "@" in value:
            raise ValueError("identifier must be opaque and contain no email address")
        return value

    @field_validator("sender_ref")
    @classmethod
    def require_hashed_sender(cls, value: str) -> str:
        if not _PSEUDONYMOUS_REF.fullmatch(value):
            raise ValueError("sender_ref must be a pseudonymous digest reference")
        return value

    @field_validator("received_at")
    @classmethod
    def require_timezone(cls, value: str) -> str:
        if not _DATE_TIME.fullmatch(value):
            raise ValueError("received_at must be an ISO-8601 timestamp with timezone")
        return value

    @field_validator("raw_email_ref")
    @classmethod
    def require_safe_raw_reference(cls, value: str) -> str:
        if not value.startswith(("secure://", "synthetic://")) or any(
            character in value for character in ("@", "?", "#")
        ):
            raise ValueError(
                "raw_email_ref must be an opaque secure:// or synthetic:// reference"
            )
        return value

    @field_validator("access_context_ref")
    @classmethod
    def require_safe_access_reference(cls, value: str) -> str:
        if not value.startswith(("secure://access/", "synthetic://access/")):
            raise ValueError(
                "access_context_ref must be an opaque access-service reference"
            )
        if any(character in value for character in ("@", "?", "#")):
            raise ValueError(
                "access_context_ref must not contain identity or query data"
            )
        return value

    @model_validator(mode="after")
    def require_safe_checkpoint_payload(self) -> Self:
        if not self.attachments_scanned:
            raise ValueError("attachments must be scanned before graph admission")
        if not self.redaction_complete:
            raise ValueError("DLP redaction must complete before graph admission")
        if self.raw_email_ref.startswith(
            "secure://"
        ) and not self.sender_ref.startswith("hmac-sha256:"):
            raise ValueError("production sender_ref must use keyed hmac-sha256")
        findings = obvious_sensitive_fragments(f"{self.subject}\n{self.body}")
        if findings:
            raise ValueError(
                "redacted graph text still contains obvious sensitive data: "
                + ", ".join(findings)
            )
        return self


class AccessContext(ContractModel):
    """Claims resolved by an injected trusted authorizer, never from email text."""

    access_context_ref: str = Field(min_length=4, max_length=512)
    tenant_id: str = Field(min_length=3, max_length=128)
    groups: tuple[str, ...] = Field(min_length=1, max_length=32)
    authorization_evidence_ref: str = Field(min_length=4, max_length=512)

    @field_validator("tenant_id")
    @classmethod
    def require_opaque_tenant(cls, value: str) -> str:
        if not _OPAQUE_ID.fullmatch(value) or "@" in value:
            raise ValueError("authorized tenant must be opaque")
        return value

    @field_validator("groups")
    @classmethod
    def require_authorized_groups(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("authorized groups must be unique")
        if any(not _REVIEWER_GROUP.fullmatch(group) for group in value):
            raise ValueError("authorized groups must use group:<identifier>")
        return value

    @field_validator("access_context_ref", "authorization_evidence_ref")
    @classmethod
    def require_safe_authorization_reference(cls, value: str) -> str:
        if not value.startswith(
            (
                "secure://access/",
                "secure://authorization/",
                "synthetic://access/",
                "synthetic://authorization/",
            )
        ) or any(character in value for character in ("@", "?", "#")):
            raise ValueError(
                "authorization references must be opaque secure references"
            )
        return value


class ModelUsage(ContractModel):
    model_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    measurement_source: MeasurementSource = MeasurementSource.OFFLINE_FIXTURE
    model_release: str | None = Field(default=None, min_length=1, max_length=256)
    price_evidence_ref: str | None = Field(default=None, min_length=4, max_length=512)

    @field_validator("price_evidence_ref")
    @classmethod
    def require_safe_price_reference(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.startswith(("secure://pricing/", "synthetic://pricing/"))
            or any(character in value for character in ("@", "?", "#"))
        ):
            raise ValueError("price_evidence_ref must be an opaque pricing reference")
        return value


class Classification(ContractModel):
    intent: Intent
    urgency: Urgency
    risk: RiskTier
    topic: str = Field(min_length=1, max_length=120)
    confidence: float = Field(ge=0.0, le=1.0)
    complex_issue: bool = False
    usage: ModelUsage = Field(default_factory=ModelUsage)


class EvidenceDocument(ContractModel):
    document_id: str = Field(min_length=1, max_length=128)
    page_content: str = Field(min_length=1, max_length=8_000)
    doc_uri: str = Field(min_length=1, max_length=512)
    chunk_id: str = Field(min_length=1, max_length=128)
    score: float = Field(ge=0.0)
    tenant_id: str = Field(min_length=1, max_length=128)
    allowed_groups: tuple[str, ...] = Field(min_length=1, max_length=32)
    active: bool
    release: str = Field(min_length=1, max_length=128)

    @field_validator("allowed_groups")
    @classmethod
    def require_document_groups(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(
            not _REVIEWER_GROUP.fullmatch(group) for group in value
        ):
            raise ValueError("document groups must be unique group:<identifier> values")
        return value

    def as_mlflow_document(self) -> dict[str, object]:
        """Return the exact document fields consumed by MLflow RAG scorers."""

        return {
            "id": self.document_id,
            "page_content": self.page_content,
            "metadata": {
                "doc_uri": self.doc_uri,
                "chunk_id": self.chunk_id,
                "score": self.score,
                "tenant_scope": (
                    "shared"
                    if self.tenant_id == "shared"
                    else "sha256:" + _sha256(self.tenant_id)
                ),
                "active": self.active,
                "release": self.release,
            },
        }


class DraftResponse(ContractModel):
    body: str = Field(min_length=1, max_length=4_000)
    citations: tuple[str, ...] = ()
    abstained: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    prompt_version: str = Field(min_length=1, max_length=128)
    usage: ModelUsage = Field(default_factory=ModelUsage)
    quarantined: bool = False

    @field_validator("citations")
    @classmethod
    def require_unique_citations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("citations must not contain duplicates")
        return value

    @model_validator(mode="after")
    def reject_sensitive_durable_output(self) -> Self:
        findings = obvious_sensitive_fragments(self.body)
        if findings:
            raise ValueError(
                "draft failed sensitive-output admission: " + ", ".join(findings)
            )
        if self.quarantined and not self.abstained:
            raise ValueError("a quarantined draft must abstain")
        return self


class GateFinding(ContractModel):
    name: str = Field(min_length=1, max_length=128)
    passed: bool
    detail: str = Field(min_length=1, max_length=500)


class PlannedAction(ContractModel):
    kind: ActionKind
    idempotency_key: str = Field(min_length=16, max_length=128)
    case_id: str = Field(min_length=3, max_length=128)
    reply_body: str | None = Field(default=None, min_length=1, max_length=4_000)
    citations: tuple[str, ...] = ()
    ticket_summary: str | None = Field(default=None, min_length=1, max_length=500)

    @field_validator("idempotency_key")
    @classmethod
    def require_stable_idempotency_key(cls, value: str) -> str:
        if not _IDEMPOTENCY_KEY.fullmatch(value):
            raise ValueError("idempotency_key must be a lowercase SHA-256 digest")
        return value

    @field_validator("case_id")
    @classmethod
    def require_opaque_case_id(cls, value: str) -> str:
        if not _OPAQUE_ID.fullmatch(value) or "@" in value:
            raise ValueError("action case_id must be opaque")
        return value

    @field_validator("citations")
    @classmethod
    def require_unique_action_citations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("action citations must not contain duplicates")
        return value

    @model_validator(mode="after")
    def require_kind_payload(self) -> Self:
        if self.kind is ActionKind.ENQUEUE_REPLY:
            if self.reply_body is None or self.ticket_summary is not None:
                raise ValueError("enqueue_reply requires only reply_body")
        elif self.ticket_summary is None or self.reply_body is not None:
            raise ValueError("upsert_ticket requires only ticket_summary")
        content = (
            self.reply_body if self.reply_body is not None else self.ticket_summary
        )
        findings = obvious_sensitive_fragments(content or "")
        if findings:
            raise ValueError(
                "action payload failed sensitive-output admission: "
                + ", ".join(findings)
            )
        return self


class PreparedCase(ContractModel):
    email: RedactedEmail
    access_context_digest: str
    knowledge_release: str = Field(min_length=1, max_length=128)
    classification: Classification
    route: Route
    route_reasons: tuple[str, ...]
    evidence: tuple[EvidenceDocument, ...]
    draft: DraftResponse
    gates: tuple[GateFinding, ...]
    planned_actions: tuple[PlannedAction, ...]
    requires_review: bool
    disposition: Disposition
    total_usage: ModelUsage
    application_release: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def disposition_matches_review(self) -> Self:
        if not _SHA256_REF.fullmatch(self.access_context_digest):
            raise ValueError("access_context_digest must be a sha256:<hex> reference")
        expected = (
            Disposition.PENDING_REVIEW if self.requires_review else Disposition.READY
        )
        if self.disposition is not expected:
            raise ValueError("prepared disposition does not match review requirement")
        if any(action.case_id != self.email.case_id for action in self.planned_actions):
            raise ValueError("all planned actions must belong to the prepared case")
        return self


class ReviewDecision(ContractModel):
    case_id: str = Field(min_length=3, max_length=128)
    proposal_digest: str
    application_release: str = Field(min_length=1, max_length=128)
    authorization_ref: str = Field(min_length=4, max_length=512)
    action: ReviewAction
    reason: ReviewReason
    edited_response: str | None = Field(default=None, min_length=1, max_length=4_000)

    @field_validator("authorization_ref")
    @classmethod
    def require_review_authorization_reference(cls, value: str) -> str:
        if not value.startswith(("secure://review/", "synthetic://review/")) or any(
            character in value for character in ("@", "?", "#")
        ):
            raise ValueError(
                "authorization_ref must be an opaque review-service reference"
            )
        return value

    @field_validator("case_id")
    @classmethod
    def require_opaque_case_id(cls, value: str) -> str:
        if not _OPAQUE_ID.fullmatch(value) or "@" in value:
            raise ValueError("review case_id must be opaque")
        return value

    @field_validator("proposal_digest")
    @classmethod
    def require_proposal_digest(cls, value: str) -> str:
        if not _SHA256_REF.fullmatch(value):
            raise ValueError("proposal_digest must be a sha256:<hex> reference")
        return value

    @model_validator(mode="after")
    def action_matches_payload(self) -> Self:
        if self.action is ReviewAction.EDIT and self.edited_response is None:
            raise ValueError("edit requires edited_response")
        if self.action is not ReviewAction.EDIT and self.edited_response is not None:
            raise ValueError("edited_response is valid only for edit")
        if (
            self.action is ReviewAction.APPROVE
            and self.reason is not ReviewReason.APPROVED
        ):
            raise ValueError("approve must use the approved reason")
        if self.action is ReviewAction.EDIT and self.reason not in {
            ReviewReason.FACTUAL_EDIT,
            ReviewReason.POLICY_EDIT,
        }:
            raise ValueError("edit requires a factual_edit or policy_edit reason")
        if self.action is ReviewAction.REJECT and self.reason is ReviewReason.APPROVED:
            raise ValueError("reject requires a rejection reason")
        if self.edited_response is not None:
            findings = obvious_sensitive_fragments(self.edited_response)
            if findings:
                raise ValueError(
                    "review edit failed sensitive-output admission: "
                    + ", ".join(findings)
                )
        return self


class VerifiedReviewerContext(ContractModel):
    """Authorization evidence returned by the trusted review-service port."""

    authorization_ref: str = Field(min_length=4, max_length=512)
    reviewer_subject_ref: str
    reviewer_group: str
    authorized_case_id: str = Field(min_length=3, max_length=128)
    authorized_proposal_digest: str
    application_release: str = Field(min_length=1, max_length=128)
    allowed_actions: tuple[ReviewAction, ...] = Field(min_length=1)

    @field_validator("authorization_ref")
    @classmethod
    def require_verified_authorization_ref(cls, value: str) -> str:
        if not value.startswith(("secure://review/", "synthetic://review/")) or any(
            character in value for character in ("@", "?", "#")
        ):
            raise ValueError("verified review reference must be opaque")
        return value

    @field_validator("reviewer_subject_ref", "authorized_proposal_digest")
    @classmethod
    def require_hashed_references(cls, value: str) -> str:
        if not _SHA256_REF.fullmatch(value):
            raise ValueError("review authorization identity must be sha256:<hex>")
        return value

    @field_validator("authorized_case_id")
    @classmethod
    def require_opaque_authorized_case(cls, value: str) -> str:
        if not _OPAQUE_ID.fullmatch(value) or "@" in value:
            raise ValueError("authorized_case_id must be opaque")
        return value

    @field_validator("reviewer_group")
    @classmethod
    def require_verified_group(cls, value: str) -> str:
        if not _REVIEWER_GROUP.fullmatch(value):
            raise ValueError("reviewer_group must use group:<identifier>")
        return value

    @field_validator("allowed_actions")
    @classmethod
    def require_unique_actions(
        cls, value: tuple[ReviewAction, ...]
    ) -> tuple[ReviewAction, ...]:
        if len(value) != len(set(value)):
            raise ValueError("allowed review actions must be unique")
        return value


class ActionReceipt(ContractModel):
    kind: ActionKind
    idempotency_key: str
    status: OutboxStatus
    external_ref: str = Field(min_length=1, max_length=256)
    duplicate: bool = False

    @field_validator("idempotency_key")
    @classmethod
    def require_stable_idempotency_key(cls, value: str) -> str:
        if not _IDEMPOTENCY_KEY.fullmatch(value):
            raise ValueError("receipt idempotency_key must be a SHA-256 digest")
        return value

    @field_validator("external_ref")
    @classmethod
    def require_opaque_outbox_reference(cls, value: str) -> str:
        if not value.startswith(
            ("outbox://", "secure://outbox/", "synthetic://outbox/")
        ) or any(character in value for character in ("@", "?", "#")):
            raise ValueError("external_ref must be an opaque outbox reference")
        return value

    @model_validator(mode="after")
    def status_matches_duplicate(self) -> Self:
        expected = (
            OutboxStatus.ALREADY_ENQUEUED if self.duplicate else OutboxStatus.ENQUEUED
        )
        if self.status is not expected:
            raise ValueError("outbox receipt status does not match duplicate flag")
        return self


class ExecutionResult(ContractModel):
    case_id: str
    application_release: str = Field(min_length=1, max_length=128)
    disposition: Disposition
    receipts: tuple[ActionReceipt, ...] = ()
    final_response: str | None = None
    review_action: ReviewAction | None = None
    review_reason: ReviewReason | None = None
    reviewer_group: str | None = None
    reviewer_subject_ref: str | None = None
    proposal_digest: str | None = None

    @field_validator("case_id")
    @classmethod
    def require_opaque_case_id(cls, value: str) -> str:
        if not _OPAQUE_ID.fullmatch(value) or "@" in value:
            raise ValueError("execution case_id must be opaque")
        return value

    @field_validator("final_response")
    @classmethod
    def require_safe_final_response(cls, value: str | None) -> str | None:
        findings = obvious_sensitive_fragments(value or "")
        if findings:
            raise ValueError(
                "execution response failed sensitive-output admission: "
                + ", ".join(findings)
            )
        return value

    @model_validator(mode="after")
    def review_provenance_is_complete(self) -> Self:
        provenance = (
            self.review_reason,
            self.reviewer_group,
            self.reviewer_subject_ref,
            self.proposal_digest,
        )
        if self.review_action is None and any(
            value is not None for value in provenance
        ):
            raise ValueError("automatic execution must not invent review provenance")
        if self.review_action is not None and any(
            value is None for value in provenance
        ):
            raise ValueError(
                "reviewed execution requires complete authorization evidence"
            )
        if self.reviewer_group is not None and not _REVIEWER_GROUP.fullmatch(
            self.reviewer_group
        ):
            raise ValueError("execution reviewer_group must use group:<identifier>")
        for value in (self.reviewer_subject_ref, self.proposal_digest):
            if value is not None and not _SHA256_REF.fullmatch(value):
                raise ValueError("execution review references must be sha256:<hex>")
        keys = [receipt.idempotency_key for receipt in self.receipts]
        if len(keys) != len(set(keys)):
            raise ValueError("execution receipts must have unique idempotency keys")
        if self.disposition is Disposition.QUEUED:
            if not self.receipts:
                raise ValueError("queued execution requires at least one receipt")
        elif self.disposition is Disposition.HANDLED_BY_HUMAN:
            if self.receipts or self.final_response is not None:
                raise ValueError("human-owned execution cannot contain queued output")
            if self.review_action is not ReviewAction.REJECT:
                raise ValueError("human-owned execution requires a rejected review")
        else:
            raise ValueError("execution result must be queued or handled by a human")
        has_reply_receipt = any(
            receipt.kind is ActionKind.ENQUEUE_REPLY for receipt in self.receipts
        )
        if has_reply_receipt is not (self.final_response is not None):
            raise ValueError("final_response must match an enqueue-reply receipt")
        if self.review_action is ReviewAction.REJECT and (
            self.disposition is not Disposition.HANDLED_BY_HUMAN
        ):
            raise ValueError("a rejected review cannot queue an action")
        return self


class RuntimeBudget(ContractModel):
    max_model_calls: int = Field(default=2, ge=0, le=10)
    max_input_tokens: int = Field(default=8_000, ge=0, le=1_000_000)
    max_output_tokens: int = Field(default=1_200, ge=0, le=100_000)
    max_retrieved_documents: int = Field(default=4, ge=1, le=20)


class PolicyConfig(ContractModel):
    auto_send_low_risk: bool = False
    minimum_classification_confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    minimum_draft_confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    maximum_auto_send_risk: RiskTier = RiskTier.LOW
    application_release: str = Field(default="email-support-reference-v1", min_length=1)
    knowledge_release: str = Field(default="kb-2026-08-01", min_length=1)
    required_reviewer_group: str = "group:support-quality-reviewers"

    @field_validator("required_reviewer_group")
    @classmethod
    def require_controlled_reviewer_group(cls, value: str) -> str:
        if not _REVIEWER_GROUP.fullmatch(value):
            raise ValueError("required_reviewer_group must use group:<identifier>")
        return value


def _sha256(value: str) -> str:
    return sha256(value.encode()).hexdigest()
