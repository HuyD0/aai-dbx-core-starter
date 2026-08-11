"""Deterministic adapters and command-line check for the reference design."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from email_support_agent.contracts import (
    AccessContext,
    ActionReceipt,
    Classification,
    DraftResponse,
    EvidenceDocument,
    Intent,
    MeasurementSource,
    ModelUsage,
    OutboxStatus,
    PlannedAction,
    PolicyConfig,
    RedactedEmail,
    ReviewAction,
    ReviewDecision,
    RiskTier,
    RuntimeBudget,
    Urgency,
    VerifiedReviewerContext,
)
from email_support_agent.workflow import EmailSupportWorkflow

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "and",
    "can",
    "for",
    "how",
    "i",
    "is",
    "my",
    "of",
    "the",
    "to",
    "what",
    "with",
}


class OfflineClassifier:
    """Transparent rules stand in for a small structured-output classifier."""

    async def classify(
        self,
        email: RedactedEmail,
        *,
        max_output_tokens: int,
    ) -> Classification:
        await asyncio.sleep(0)
        if max_output_tokens < 1:
            raise ValueError("classifier requires a positive output-token limit")
        text = f"{email.subject} {email.body}".lower()
        if any(
            phrase in text
            for phrase in (
                "reveal the api key",
                "root password",
                "system prompt",
                "ignore your instructions",
                "security incident",
            )
        ):
            values = (Intent.SECURITY, Urgency.CRITICAL, RiskTier.CRITICAL, "security")
        elif any(word in text for word in ("refund", "invoice", "charged", "billing")):
            values = (Intent.BILLING, Urgency.HIGH, RiskTier.HIGH, "billing")
        elif any(
            phrase in text
            for phrase in ("delete my account", "privacy request", "export my data")
        ):
            values = (Intent.ACCOUNT, Urgency.HIGH, RiskTier.HIGH, "account_privacy")
        elif any(word in text for word in ("crash", "bug", "error", "outage", "down")):
            critical = any(
                phrase in text
                for phrase in ("all users", "production down", "critical outage")
            )
            values = (
                Intent.BUG,
                Urgency.CRITICAL if critical else Urgency.HIGH,
                RiskTier.CRITICAL if critical else RiskTier.MEDIUM,
                "product_defect",
            )
        elif any(
            phrase in text
            for phrase in (
                "feature request",
                "add dark mode",
                "please add",
                "wish the product",
            )
        ):
            values = (
                Intent.FEATURE_REQUEST,
                Urgency.MEDIUM,
                RiskTier.LOW,
                "product_feature",
            )
        elif any(
            phrase in text
            for phrase in ("data loss", "legal notice", "multiple systems")
        ):
            values = (Intent.OTHER, Urgency.HIGH, RiskTier.HIGH, "complex_case")
        else:
            values = (Intent.QUESTION, Urgency.LOW, RiskTier.LOW, "product_question")
        return Classification(
            intent=values[0],
            urgency=values[1],
            risk=values[2],
            topic=values[3],
            confidence=0.99,
            complex_issue=values[3] == "complex_case",
            usage=ModelUsage(measurement_source=MeasurementSource.OFFLINE_FIXTURE),
        )


class OfflineKnowledgeRetriever:
    def __init__(self, documents: Sequence[EvidenceDocument]) -> None:
        self.documents = tuple(documents)
        self.calls = 0

    async def retrieve(
        self,
        query: str,
        *,
        access: AccessContext,
        release: str,
        top_k: int,
    ) -> tuple[EvidenceDocument, ...]:
        await asyncio.sleep(0)
        self.calls += 1
        query_terms = _terms(query)
        ranked = []
        for document in self.documents:
            if (
                document.tenant_id not in {access.tenant_id, "shared"}
                or not document.active
                or document.release != release
                or not set(document.allowed_groups).intersection(access.groups)
            ):
                continue
            overlap = len(query_terms.intersection(_terms(document.page_content)))
            # One generic token (for example, "support") is not enough to
            # turn an unrelated positive-scored candidate into model context.
            if overlap >= 2:
                ranked.append((overlap, document.score, document.document_id, document))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return tuple(item[3] for item in ranked[:top_k])


class OfflineDrafter:
    async def draft(
        self,
        email: RedactedEmail,
        classification: Classification,
        evidence: tuple[EvidenceDocument, ...],
        *,
        max_output_tokens: int,
    ) -> DraftResponse:
        await asyncio.sleep(0)
        if max_output_tokens < 1:
            raise ValueError("drafter requires a positive output-token limit")
        if classification.intent in {Intent.SECURITY, Intent.BILLING, Intent.ACCOUNT}:
            body = (
                "A support specialist will review this request. No account, "
                "billing, or security change has been made."
            )
            citations: tuple[str, ...] = ()
            abstained = False
            confidence = 0.95
        elif evidence:
            selected = evidence[0]
            body = f"{selected.page_content} Source: [{selected.document_id}]."
            citations = (selected.document_id,)
            abstained = False
            confidence = 0.98
        else:
            body = (
                "I could not find current, authorized knowledge for this request. "
                "A support specialist needs to investigate."
            )
            citations = ()
            abstained = True
            confidence = 1.0
        return DraftResponse(
            body=body,
            citations=citations,
            abstained=abstained,
            confidence=confidence,
            prompt_version="offline-deterministic-v1",
            usage=ModelUsage(measurement_source=MeasurementSource.OFFLINE_FIXTURE),
        )


class InMemoryTransactionalOutbox:
    """Test-only outbox with the same duplicate-suppression contract as production."""

    def __init__(self) -> None:
        self.attempts = 0
        self.actions: dict[str, PlannedAction] = {}

    async def enqueue_once(self, action: PlannedAction) -> ActionReceipt:
        """Compatibility helper for focused tests; workflow uses batches."""

        return (await self.enqueue_batch_once((action,)))[0]

    async def enqueue_batch_once(
        self,
        actions: tuple[PlannedAction, ...],
    ) -> tuple[ActionReceipt, ...]:
        """Atomically validate and insert one complete business action set."""

        await asyncio.sleep(0)
        if not actions:
            return ()
        if len({action.idempotency_key for action in actions}) != len(actions):
            raise ValueError("an outbox batch cannot repeat an idempotency key")
        for action in actions:
            existing = self.actions.get(action.idempotency_key)
            if existing is not None and existing != action:
                raise ValueError(
                    "idempotency key collision has a different action payload"
                )

        # Nothing mutates until every collision check succeeds. A real adapter
        # implements this same invariant with one database transaction.
        self.attempts += len(actions)
        receipts = []
        for action in actions:
            duplicate = action.idempotency_key in self.actions
            self.actions.setdefault(action.idempotency_key, action)
            receipts.append(
                ActionReceipt(
                    kind=action.kind,
                    idempotency_key=action.idempotency_key,
                    status=(
                        OutboxStatus.ALREADY_ENQUEUED
                        if duplicate
                        else OutboxStatus.ENQUEUED
                    ),
                    external_ref=f"outbox://{action.idempotency_key}",
                    duplicate=duplicate,
                )
            )
        return tuple(receipts)


class OfflineAccessAuthorizer:
    """Test-only resolver standing in for verified ingress authorization."""

    def __init__(self) -> None:
        self.calls = 0

    async def authorize(self, email: RedactedEmail) -> AccessContext:
        await asyncio.sleep(0)
        self.calls += 1
        expected = f"synthetic://access/{email.tenant_id}"
        if email.access_context_ref != expected:
            raise PermissionError("synthetic access reference is not authorized")
        groups = ["group:support-all"]
        if email.tenant_id == "tenant-alpha":
            groups.append("group:tenant-alpha-admin")
        return AccessContext(
            access_context_ref=email.access_context_ref,
            tenant_id=email.tenant_id,
            groups=tuple(groups),
            authorization_evidence_ref=(
                f"synthetic://authorization/{email.tenant_id}/current"
            ),
        )


class OfflineReviewAuthorizer:
    """Test-only resolver; production derives claims from its identity provider."""

    def __init__(self) -> None:
        self.calls = 0

    async def authorize(
        self,
        decision: ReviewDecision,
    ) -> VerifiedReviewerContext:
        await asyncio.sleep(0)
        self.calls += 1
        if decision.authorization_ref != "synthetic://review/support-quality":
            raise PermissionError("review authorization is missing or revoked")
        return VerifiedReviewerContext(
            authorization_ref=decision.authorization_ref,
            reviewer_subject_ref="sha256:"
            + sha256(b"synthetic-reviewer-subject").hexdigest(),
            reviewer_group="group:support-quality-reviewers",
            authorized_case_id=decision.case_id,
            authorized_proposal_digest=decision.proposal_digest,
            application_release=decision.application_release,
            allowed_actions=tuple(ReviewAction),
        )


def load_knowledge(path: str | Path) -> tuple[EvidenceDocument, ...]:
    return tuple(
        EvidenceDocument.model_validate_json(line, strict=True)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def build_offline_workflow(
    root: str | Path,
    *,
    auto_send_low_risk: bool = False,
    budget: RuntimeBudget | None = None,
) -> tuple[EmailSupportWorkflow, InMemoryTransactionalOutbox]:
    project_root = Path(root)
    outbox = InMemoryTransactionalOutbox()
    workflow = EmailSupportWorkflow(
        access_authorizer=OfflineAccessAuthorizer(),
        review_authorizer=OfflineReviewAuthorizer(),
        classifier=OfflineClassifier(),
        retriever=OfflineKnowledgeRetriever(
            load_knowledge(project_root / "data" / "synthetic_knowledge.jsonl")
        ),
        drafter=OfflineDrafter(),
        outbox=outbox,
        policy=PolicyConfig(auto_send_low_risk=auto_send_low_risk),
        budget=budget,
    )
    return workflow, outbox


def _terms(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="run the offline release gate",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.check:
        from email_support_agent.evaluation import evaluate_release_cases

        report, gate = asyncio.run(evaluate_release_cases(root))
        print(
            json.dumps(
                {
                    "metrics": dict(report.metrics),
                    "case_count": len(report.cases),
                    "gate_passed": gate.passed,
                    "failures": [
                        item.model_dump(mode="json") for item in gate.failures
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        gate.require_passed()
        return

    workflow, _ = build_offline_workflow(root)
    example = RedactedEmail(
        case_id="case-demo-001",
        message_id="message-demo-001",
        thread_id="thread-demo-001",
        tenant_id="tenant-demo",
        ingress_provider="synthetic-mail",
        access_context_ref="synthetic://access/tenant-demo",
        sender_ref="sha256:" + "a" * 64,
        raw_email_ref="synthetic://email/message-demo-001",
        subject="How do I reset my password?",
        body="The reset link has expired. What should I do?",
        received_at="2026-08-10T12:00:00Z",
        attachments_scanned=True,
        redaction_complete=True,
        sanitization_version="synthetic-dlp-v1",
    )
    prepared = asyncio.run(workflow.prepare(example))
    print(prepared.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
