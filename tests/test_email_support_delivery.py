"""Failure-injection tests for the idempotent outbox delivery worker."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "email-support-agent" / "src"
_ADDED_SOURCE = str(SOURCE) not in sys.path
if _ADDED_SOURCE:
    sys.path.insert(0, str(SOURCE))

from email_support_agent.contracts import ActionKind, PlannedAction  # noqa: E402
from email_support_agent.delivery import (  # noqa: E402
    DeliveryDisposition,
    DeliveryLease,
    DeliveryProviderError,
    DeliveryWorker,
    ProviderAcceptance,
    ProviderDeliveryReceipt,
    RetryPolicy,
    action_payload_digest,
)

if _ADDED_SOURCE:
    sys.path.remove(str(SOURCE))

NOW = "2026-08-11T12:00:00Z"


def _action(kind: ActionKind = ActionKind.ENQUEUE_REPLY) -> PlannedAction:
    values = {
        "kind": kind,
        "idempotency_key": ("a" if kind is ActionKind.ENQUEUE_REPLY else "b") * 64,
        "case_id": "case-delivery-001",
    }
    if kind is ActionKind.ENQUEUE_REPLY:
        values["reply_body"] = "A reviewed response ready for delivery."
        values["citations"] = ()
    else:
        values["ticket_summary"] = "Reviewed issue requires a ticket."
    return PlannedAction(**values)


def _lease(
    *,
    kind: ActionKind = ActionKind.ENQUEUE_REPLY,
    attempt: int = 1,
) -> DeliveryLease:
    action = _action(kind)
    return DeliveryLease(
        action=action,
        payload_digest=action_payload_digest(action),
        lease_ref="sha256:" + "c" * 64,
        attempt=attempt,
        max_attempts=5,
        claimed_at=NOW,
        lease_expires_at="2026-08-11T12:01:00Z",
    )


class MemoryDeliveryStore:
    def __init__(self, *leases: DeliveryLease) -> None:
        self.leases = list(leases)
        self.delivered = []
        self.retries = []
        self.dead_letters = []
        self.fail_mark_once = False

    async def claim_next(self, *, now):
        assert now == NOW
        return self.leases.pop(0) if self.leases else None

    async def mark_delivered(self, lease, receipt):
        if self.fail_mark_once:
            self.fail_mark_once = False
            raise RuntimeError("synthetic crash after provider acceptance")
        self.delivered.append((lease, receipt))

    async def schedule_retry(self, lease, *, error_code, retry_at):
        self.retries.append((lease, error_code, retry_at))

    async def dead_letter(self, lease, *, error_code):
        self.dead_letters.append((lease, error_code))


class IdempotentProvider:
    def __init__(self) -> None:
        self.calls = []
        self.effects: set[str] = set()
        self.failures: list[Exception] = []
        self.receipt_mutation = None

    async def deliver(self, action):
        self.calls.append(action)
        if self.failures:
            raise self.failures.pop(0)
        duplicate = action.idempotency_key in self.effects
        self.effects.add(action.idempotency_key)
        receipt = ProviderDeliveryReceipt(
            kind=action.kind,
            idempotency_key=action.idempotency_key,
            payload_digest=action_payload_digest(action),
            acceptance=(
                ProviderAcceptance.DUPLICATE
                if duplicate
                else ProviderAcceptance.ACCEPTED
            ),
            provider_receipt_ref=(
                "synthetic://provider-receipt/" + action.idempotency_key
            ),
            accepted_at=NOW,
        )
        return self.receipt_mutation(receipt) if self.receipt_mutation else receipt


def _worker(store, reply, ticket=None):
    return DeliveryWorker(
        store=store,
        reply_provider=reply,
        ticket_provider=ticket or IdempotentProvider(),
        retry_policy=RetryPolicy(
            max_attempts=5,
            base_delay_seconds=5,
            max_delay_seconds=60,
        ),
    )


def test_delivery_routes_actions_and_records_bound_terminal_receipts():
    async def scenario():
        reply = IdempotentProvider()
        ticket = IdempotentProvider()
        store = MemoryDeliveryStore(
            _lease(kind=ActionKind.ENQUEUE_REPLY),
            _lease(kind=ActionKind.UPSERT_TICKET),
        )
        worker = _worker(store, reply, ticket)

        reply_result = await worker.process_one(now=NOW)
        ticket_result = await worker.process_one(now=NOW)
        empty = await worker.process_one(now=NOW)

        assert reply_result.disposition is DeliveryDisposition.DELIVERED
        assert ticket_result.disposition is DeliveryDisposition.DELIVERED
        assert len(reply.calls) == len(ticket.calls) == 1
        assert len(store.delivered) == 2
        assert empty is None

    asyncio.run(scenario())


def test_retry_is_bounded_and_uses_controlled_exponential_backoff():
    async def scenario():
        provider = IdempotentProvider()
        provider.failures.append(
            DeliveryProviderError(code="rate_limited", retryable=True)
        )
        store = MemoryDeliveryStore(_lease(attempt=1))

        result = await _worker(store, provider).process_one(now=NOW)

        assert result.disposition is DeliveryDisposition.RETRY_SCHEDULED
        assert result.retry_at == "2026-08-11T12:00:05Z"
        assert result.error_code == "rate_limited"
        assert len(store.retries) == 1
        assert not store.delivered and not store.dead_letters

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    [
        DeliveryProviderError(code="policy_rejected", retryable=False),
        DeliveryProviderError(code="provider_unavailable", retryable=True),
    ],
)
def test_terminal_or_exhausted_failure_goes_to_dead_letter(failure):
    async def scenario():
        provider = IdempotentProvider()
        provider.failures.append(failure)
        attempt = 5 if failure.retryable else 1
        store = MemoryDeliveryStore(_lease(attempt=attempt))

        result = await _worker(store, provider).process_one(now=NOW)

        assert result.disposition is DeliveryDisposition.DEAD_LETTER
        assert len(store.dead_letters) == 1
        assert not store.retries and not store.delivered

    asyncio.run(scenario())


def test_tampered_lease_or_unbound_provider_receipt_fails_before_acknowledgement():
    async def tampered_lease_scenario():
        lease = _lease().model_copy(update={"payload_digest": "sha256:" + "9" * 64})
        store = MemoryDeliveryStore(lease)
        provider = IdempotentProvider()
        with pytest.raises(ValueError, match="payload digest"):
            await _worker(store, provider).process_one(now=NOW)
        assert not provider.calls and not store.delivered

    async def bad_receipt_scenario():
        store = MemoryDeliveryStore(_lease())
        provider = IdempotentProvider()
        provider.receipt_mutation = lambda receipt: receipt.model_copy(
            update={"payload_digest": "sha256:" + "8" * 64}
        )
        with pytest.raises(ValueError, match="not bound"):
            await _worker(store, provider).process_one(now=NOW)
        assert not store.delivered

    asyncio.run(tampered_lease_scenario())
    asyncio.run(bad_receipt_scenario())


def test_provider_accept_then_worker_crash_replays_without_duplicate_effect():
    async def scenario():
        provider = IdempotentProvider()
        store = MemoryDeliveryStore(_lease(attempt=1))
        store.fail_mark_once = True
        worker = _worker(store, provider)

        with pytest.raises(RuntimeError, match="synthetic crash"):
            await worker.process_one(now=NOW)
        assert len(provider.effects) == 1
        assert not store.delivered

        store.leases.append(_lease(attempt=2))
        result = await worker.process_one(now=NOW)

        assert result.disposition is DeliveryDisposition.DELIVERED
        assert len(provider.calls) == 2
        assert len(provider.effects) == 1
        assert store.delivered[0][1].acceptance is ProviderAcceptance.DUPLICATE

    asyncio.run(scenario())


def test_unclassified_provider_exception_is_not_silently_retried():
    async def scenario():
        provider = IdempotentProvider()
        provider.failures.append(RuntimeError("unexpected provider exception"))
        store = MemoryDeliveryStore(_lease())

        with pytest.raises(RuntimeError, match="unexpected provider"):
            await _worker(store, provider).process_one(now=NOW)

        assert not store.retries and not store.dead_letters and not store.delivered

    asyncio.run(scenario())


def test_sensitive_provider_reference_is_rejected_by_strict_read_boundary():
    async def scenario():
        provider = IdempotentProvider()
        provider.receipt_mutation = lambda receipt: receipt.model_copy(
            update={
                "provider_receipt_ref": (
                    "synthetic://provider-receipt/person@example.test"
                )
            }
        )
        store = MemoryDeliveryStore(_lease())

        with pytest.raises(ValueError, match="invalid strict contract"):
            await _worker(store, provider).process_one(now=NOW)
        assert not store.delivered

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("claimed_at", "lease_expires_at"),
    [
        ("2026-08-11T12:01:00Z", "2026-08-11T12:02:00Z"),
        ("2026-08-11T11:58:00Z", "2026-08-11T12:00:00Z"),
    ],
)
def test_future_or_expired_lease_never_reaches_provider(
    claimed_at,
    lease_expires_at,
):
    async def scenario():
        lease = _lease().model_copy(
            update={
                "claimed_at": claimed_at,
                "lease_expires_at": lease_expires_at,
            }
        )
        store = MemoryDeliveryStore(lease)
        provider = IdempotentProvider()

        with pytest.raises(ValueError, match="not active"):
            await _worker(store, provider).process_one(now=NOW)

        assert not provider.calls and not store.delivered

    asyncio.run(scenario())
