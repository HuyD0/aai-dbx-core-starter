"""Deterministic safety, routing, and output policy."""

from __future__ import annotations

import re
from hashlib import sha256

from email_support_agent.contracts import (
    AccessContext,
    ActionKind,
    Classification,
    DraftResponse,
    EvidenceDocument,
    GateFinding,
    Intent,
    ModelUsage,
    PlannedAction,
    PolicyConfig,
    RedactedEmail,
    RiskTier,
    Route,
    Urgency,
    obvious_sensitive_fragments,
)

_INJECTION = re.compile(
    r"\b(?:ignore (?:all |the |your )?(?:previous|prior|system) instructions|"
    r"reveal (?:the )?(?:system prompt|hidden instructions)|developer message|"
    r"jailbreak)\b",
    re.IGNORECASE,
)
_SECURITY_PREFLIGHT = re.compile(
    r"\b(?:api key|root password|security incident|system prompt|"
    r"hidden instructions)\b",
    re.IGNORECASE,
)
_UNSAFE_COMMITMENT = re.compile(
    r"\b(?:guarantee(?:d)?|definitely refund|waive all charges|"
    r"share (?:a )?password)\b",
    re.IGNORECASE,
)
_CITATION_MARKER = re.compile(r"\[([A-Za-z0-9._:-]+)\]")
_RISK_ORDER = {
    RiskTier.LOW: 0,
    RiskTier.MEDIUM: 1,
    RiskTier.HIGH: 2,
    RiskTier.CRITICAL: 3,
}


def preflight_classification(email: RedactedEmail) -> Classification | None:
    """Short-circuit known hostile/security inputs before any model call."""

    text = f"{email.subject}\n{email.body}"
    if not (_INJECTION.search(text) or _SECURITY_PREFLIGHT.search(text)):
        return None
    return Classification(
        intent=Intent.SECURITY,
        urgency=Urgency.CRITICAL,
        risk=RiskTier.CRITICAL,
        topic="security_preflight",
        confidence=1.0,
        complex_issue=False,
        usage=ModelUsage(),
    )


def route_email(
    email: RedactedEmail,
    classification: Classification,
) -> tuple[Route, tuple[str, ...]]:
    """Select a route with deterministic policy taking precedence over the model."""

    text = f"{email.subject}\n{email.body}"
    if _INJECTION.search(text):
        return Route.ESCALATION, ("prompt_injection_signal",)
    if classification.intent is Intent.SECURITY:
        return Route.ESCALATION, ("security_intent",)
    if classification.urgency is Urgency.CRITICAL:
        return Route.ESCALATION, ("critical_urgency",)
    if classification.complex_issue:
        return Route.ESCALATION, ("complex_issue",)
    if classification.intent in {Intent.BILLING, Intent.ACCOUNT}:
        return Route.HUMAN_REVIEW, (f"controlled_{classification.intent.value}",)
    if classification.intent is Intent.BUG:
        return Route.BUG_TRACKING, ("bug_requires_idempotent_ticket",)
    if classification.intent in {Intent.QUESTION, Intent.FEATURE_REQUEST}:
        return Route.KNOWLEDGE_REPLY, ("knowledge_grounding_required",)
    return Route.HUMAN_REVIEW, ("unrecognized_intent",)


def verify_evidence_scope(
    documents: tuple[EvidenceDocument, ...],
    *,
    access: AccessContext,
    release: str,
) -> tuple[EvidenceDocument, ...]:
    """Fail closed on tenant, entitlement, active-state, or release drift."""

    for document in documents:
        if document.tenant_id not in {access.tenant_id, "shared"}:
            raise ValueError("retriever returned cross-tenant evidence")
        if not document.active:
            raise ValueError("retriever returned inactive evidence")
        if document.release != release:
            raise ValueError("retriever returned an unauthorized knowledge release")
        if not set(document.allowed_groups).intersection(access.groups):
            raise ValueError("retriever returned evidence outside caller entitlements")
    return documents


def policy_draft(
    classification: Classification,
    route: Route,
) -> DraftResponse | None:
    """Use safe zero-model acknowledgements on controlled routes."""

    if route not in {Route.HUMAN_REVIEW, Route.ESCALATION}:
        return None
    security = classification.intent is Intent.SECURITY
    abstained = route is Route.ESCALATION and not security
    body = (
        "A support specialist will review this security request. No security "
        "change has been made and hidden instructions are not disclosed."
        if security
        else (
            "A support specialist needs to investigate this critical or complex "
            "request before a response can be sent."
            if abstained
            else (
                "A support specialist will review this request. No account or "
                "billing change has been made."
            )
        )
    )
    return DraftResponse(
        body=body,
        citations=(),
        abstained=abstained,
        confidence=1.0,
        prompt_version="policy-template-v1",
        usage=ModelUsage(),
    )


def evaluate_draft(
    draft: DraftResponse,
    *,
    route: Route,
    evidence: tuple[EvidenceDocument, ...],
) -> tuple[GateFinding, ...]:
    evidence_ids = {document.document_id for document in evidence}
    citations = set(draft.citations)
    markers = set(_CITATION_MARKER.findall(draft.body))
    sensitive = obvious_sensitive_fragments(draft.body)
    citation_integrity = citations == markers and citations.issubset(evidence_ids)
    knowledge_grounded = (
        route is not Route.KNOWLEDGE_REPLY
        or draft.abstained
        or (bool(citations) and citation_integrity)
    )
    return (
        GateFinding(
            name="draft_admission",
            passed=not draft.quarantined,
            detail=(
                "draft passed durable-state admission"
                if not draft.quarantined
                else "draft content was quarantined before persistence"
            ),
        ),
        GateFinding(
            name="privacy",
            passed=not sensitive,
            detail=(
                "no obvious sensitive fragments"
                if not sensitive
                else "found " + ", ".join(sensitive)
            ),
        ),
        GateFinding(
            name="citation_integrity",
            passed=citation_integrity,
            detail="all citation markers came from and appear in final context",
        ),
        GateFinding(
            name="knowledge_grounding",
            passed=knowledge_grounded,
            detail="knowledge answers cite final context or abstain",
        ),
        GateFinding(
            name="unsafe_commitment",
            passed=_UNSAFE_COMMITMENT.search(draft.body) is None,
            detail="no prohibited guarantee or account commitment",
        ),
        GateFinding(
            name="response_length",
            passed=len(draft.body) <= 4_000,
            detail="response is within the bounded output contract",
        ),
    )


def requires_human_review(
    *,
    classification: Classification,
    route: Route,
    draft: DraftResponse,
    findings: tuple[GateFinding, ...],
    policy: PolicyConfig,
) -> bool:
    if not all(finding.passed for finding in findings):
        return True
    if draft.abstained:
        return True
    if route is not Route.KNOWLEDGE_REPLY:
        return True
    if classification.urgency is not Urgency.LOW:
        return True
    if classification.confidence < policy.minimum_classification_confidence:
        return True
    if draft.confidence < policy.minimum_draft_confidence:
        return True
    if _RISK_ORDER[classification.risk] > _RISK_ORDER[policy.maximum_auto_send_risk]:
        return True
    return not policy.auto_send_low_risk


def plan_actions(
    email: RedactedEmail,
    classification: Classification,
    route: Route,
    draft: DraftResponse,
) -> tuple[PlannedAction, ...]:
    """Describe writes without executing them; each key is stable across retries."""

    actions: list[PlannedAction] = []
    if route is Route.BUG_TRACKING:
        actions.append(
            PlannedAction(
                kind=ActionKind.UPSERT_TICKET,
                idempotency_key=_idempotency_key(email, "ticket"),
                case_id=email.case_id,
                ticket_summary=(f"{classification.topic}: {email.subject}"[:500]),
            )
        )
    if not draft.abstained:
        actions.append(
            PlannedAction(
                kind=ActionKind.ENQUEUE_REPLY,
                idempotency_key=_idempotency_key(email, "reply"),
                case_id=email.case_id,
                reply_body=draft.body,
                citations=draft.citations,
            )
        )
    return tuple(actions)


def _idempotency_key(email: RedactedEmail, action: str) -> str:
    material = ":".join(
        (
            "email-support",
            "v2",
            email.ingress_provider,
            email.tenant_id,
            email.message_id,
            action,
        )
    )
    return sha256(material.encode()).hexdigest()
