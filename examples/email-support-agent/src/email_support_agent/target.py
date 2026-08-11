"""Synchronous, side-effect-free target used by AgentKit live evaluation."""

from __future__ import annotations

import asyncio
from pathlib import Path

from email_support_agent.contracts import RedactedEmail
from email_support_agent.offline import build_offline_workflow


def respond(
    *,
    case_id: str,
    message_id: str,
    thread_id: str,
    tenant_id: str,
    ingress_provider: str,
    access_context_ref: str,
    sender_ref: str,
    raw_email_ref: str,
    subject: str,
    body: str,
    received_at: str,
    attachments_scanned: bool,
    redaction_complete: bool,
    sanitization_version: str,
) -> dict[str, object]:
    """Evaluate preparation only; ticket and reply side effects stay disabled."""

    root = Path(__file__).resolve().parents[2]
    workflow, outbox = build_offline_workflow(root, auto_send_low_risk=True)
    prepared = asyncio.run(
        workflow.prepare(
            RedactedEmail(
                case_id=case_id,
                message_id=message_id,
                thread_id=thread_id,
                tenant_id=tenant_id,
                ingress_provider=ingress_provider,
                access_context_ref=access_context_ref,
                sender_ref=sender_ref,
                raw_email_ref=raw_email_ref,
                subject=subject,
                body=body,
                received_at=received_at,
                attachments_scanned=attachments_scanned,
                redaction_complete=redaction_complete,
                sanitization_version=sanitization_version,
            )
        )
    )
    if outbox.actions:
        raise RuntimeError("evaluation preparation must never execute side effects")
    return {
        "response": prepared.draft.body,
        "intent": prepared.classification.intent.value,
        "urgency": prepared.classification.urgency.value,
        "route": prepared.route.value,
        "requires_review": prepared.requires_review,
        "citations": list(prepared.draft.citations),
        "planned_actions": [action.kind.value for action in prepared.planned_actions],
        "total_usage": prepared.total_usage.model_dump(mode="json"),
        "measurement_source": prepared.total_usage.measurement_source.value,
        "application_release": prepared.application_release,
        "knowledge_release": prepared.knowledge_release,
    }
