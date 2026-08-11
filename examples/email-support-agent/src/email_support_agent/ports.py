"""Application-owned ports; infrastructure implementations live outside the SDK."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from email_support_agent.contracts import (
    AccessContext,
    ActionReceipt,
    Classification,
    DraftResponse,
    EvidenceDocument,
    PlannedAction,
    RedactedEmail,
    ReviewDecision,
    VerifiedReviewerContext,
)


class Classifier(Protocol):
    async def classify(
        self,
        email: RedactedEmail,
        *,
        max_output_tokens: int,
    ) -> Classification: ...


class AccessAuthorizer(Protocol):
    async def authorize(self, email: RedactedEmail) -> AccessContext: ...


class ReviewAuthorizer(Protocol):
    async def authorize(self, decision: ReviewDecision) -> VerifiedReviewerContext: ...


class KnowledgeRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        access: AccessContext,
        release: str,
        top_k: int,
    ) -> tuple[EvidenceDocument, ...]: ...


class Drafter(Protocol):
    async def draft(
        self,
        email: RedactedEmail,
        classification: Classification,
        evidence: tuple[EvidenceDocument, ...],
        *,
        max_output_tokens: int,
    ) -> DraftResponse | Mapping[str, object]: ...


class TransactionalOutbox(Protocol):
    async def enqueue_batch_once(
        self,
        actions: tuple[PlannedAction, ...],
    ) -> tuple[ActionReceipt, ...]: ...
