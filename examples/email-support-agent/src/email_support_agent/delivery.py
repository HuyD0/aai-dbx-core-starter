"""Idempotent delivery-worker state machine for queued support actions.

The workflow commits a complete business action set to an outbox. Separate
workers claim rows, call providers with the same idempotency key, and record a
terminal receipt or a bounded retry. Queue/provider storage remains an injected
production capability; this module makes its behavioral contract executable.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol, Self

from pydantic import Field, field_validator, model_validator

from aai_core.contracts import ContractModel
from email_support_agent.contracts import ActionKind, PlannedAction

_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDEMPOTENCY_KEY = re.compile(r"^[0-9a-f]{64}$")
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class ProviderAcceptance(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"


class DeliveryDisposition(StrEnum):
    DELIVERED = "delivered"
    RETRY_SCHEDULED = "retry_scheduled"
    DEAD_LETTER = "dead_letter"


class DeliveryLease(ContractModel):
    action: PlannedAction
    payload_digest: str
    lease_ref: str
    attempt: int = Field(ge=1)
    max_attempts: int = Field(ge=1, le=20)
    claimed_at: str
    lease_expires_at: str

    @field_validator("payload_digest", "lease_ref")
    @classmethod
    def require_digest_refs(cls, value: str) -> str:
        if not _SHA256_REF.fullmatch(value):
            raise ValueError("delivery lease references must use sha256:<hex>")
        return value

    @field_validator("claimed_at", "lease_expires_at")
    @classmethod
    def require_timestamp(cls, value: str) -> str:
        if not _TIMESTAMP.fullmatch(value):
            raise ValueError("delivery timestamps must include an ISO-8601 timezone")
        return value

    @model_validator(mode="after")
    def require_valid_attempt_and_lease_window(self) -> Self:
        if self.attempt > self.max_attempts:
            raise ValueError("delivery attempt exceeds the terminal retry bound")
        if _parse_timestamp(self.lease_expires_at) <= _parse_timestamp(self.claimed_at):
            raise ValueError("delivery lease must expire after it is claimed")
        return self


class ProviderDeliveryReceipt(ContractModel):
    kind: ActionKind
    idempotency_key: str
    payload_digest: str
    acceptance: ProviderAcceptance
    provider_receipt_ref: str = Field(min_length=4, max_length=512)
    accepted_at: str

    @field_validator("idempotency_key")
    @classmethod
    def require_idempotency_key(cls, value: str) -> str:
        if not _IDEMPOTENCY_KEY.fullmatch(value):
            raise ValueError("provider receipt key must be a SHA-256 digest")
        return value

    @field_validator("payload_digest")
    @classmethod
    def require_payload_digest(cls, value: str) -> str:
        if not _SHA256_REF.fullmatch(value):
            raise ValueError("provider payload_digest must use sha256:<hex>")
        return value

    @field_validator("provider_receipt_ref")
    @classmethod
    def require_provider_ref(cls, value: str) -> str:
        if not value.startswith(
            ("secure://provider-receipt/", "synthetic://provider-receipt/")
        ) or any(character in value for character in ("@", "?", "#")):
            raise ValueError("provider receipt must be an opaque secure reference")
        return value

    @field_validator("accepted_at")
    @classmethod
    def require_timestamp(cls, value: str) -> str:
        if not _TIMESTAMP.fullmatch(value):
            raise ValueError("accepted_at must include an ISO-8601 timezone")
        return value


class DeliveryResult(ContractModel):
    disposition: DeliveryDisposition
    kind: ActionKind
    idempotency_key: str
    attempt: int = Field(ge=1)
    provider_receipt_ref: str | None = Field(default=None, max_length=512)
    retry_at: str | None = None
    error_code: str | None = None

    @field_validator("idempotency_key")
    @classmethod
    def require_idempotency_key(cls, value: str) -> str:
        if not _IDEMPOTENCY_KEY.fullmatch(value):
            raise ValueError("delivery result key must be a SHA-256 digest")
        return value

    @field_validator("provider_receipt_ref")
    @classmethod
    def require_provider_ref(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.startswith(
                ("secure://provider-receipt/", "synthetic://provider-receipt/")
            )
            or any(character in value for character in ("@", "?", "#"))
        ):
            raise ValueError("delivery result provider receipt must be opaque")
        return value

    @field_validator("retry_at")
    @classmethod
    def require_retry_timestamp(cls, value: str | None) -> str | None:
        if value is not None and not _TIMESTAMP.fullmatch(value):
            raise ValueError("retry_at must include an ISO-8601 timezone")
        return value

    @field_validator("error_code")
    @classmethod
    def require_safe_error_code(cls, value: str | None) -> str | None:
        if value is not None and not _ERROR_CODE.fullmatch(value):
            raise ValueError("error_code must be a controlled low-cardinality value")
        return value

    @model_validator(mode="after")
    def require_disposition_fields(self) -> Self:
        if self.disposition is DeliveryDisposition.DELIVERED:
            if self.provider_receipt_ref is None or any(
                value is not None for value in (self.retry_at, self.error_code)
            ):
                raise ValueError("delivered result requires only a provider receipt")
        elif self.disposition is DeliveryDisposition.RETRY_SCHEDULED:
            if (
                self.retry_at is None
                or self.error_code is None
                or self.provider_receipt_ref is not None
            ):
                raise ValueError("retry result requires retry_at and error_code")
        elif (
            self.error_code is None
            or self.retry_at is not None
            or self.provider_receipt_ref is not None
        ):
            raise ValueError("dead-letter result requires only error_code")
        return self


class RetryPolicy(ContractModel):
    max_attempts: int = Field(default=5, ge=1, le=20)
    base_delay_seconds: int = Field(default=5, ge=1, le=3_600)
    max_delay_seconds: int = Field(default=300, ge=1, le=86_400)

    @model_validator(mode="after")
    def require_delay_order(self) -> Self:
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max retry delay must not be below the base delay")
        return self

    def delay_seconds(self, attempt: int) -> int:
        return min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2 ** max(0, attempt - 1)),
        )


class DeliveryProviderError(RuntimeError):
    """A classified provider failure without raw provider response text."""

    def __init__(self, *, code: str, retryable: bool) -> None:
        if not _ERROR_CODE.fullmatch(code):
            raise ValueError("provider failure code is not controlled")
        self.code = code
        self.retryable = retryable
        super().__init__("delivery provider returned a classified failure")


class OutboxDeliveryStore(Protocol):
    async def claim_next(self, *, now: str) -> DeliveryLease | None: ...

    async def mark_delivered(
        self,
        lease: DeliveryLease,
        receipt: ProviderDeliveryReceipt,
    ) -> None: ...

    async def schedule_retry(
        self,
        lease: DeliveryLease,
        *,
        error_code: str,
        retry_at: str,
    ) -> None: ...

    async def dead_letter(
        self,
        lease: DeliveryLease,
        *,
        error_code: str,
    ) -> None: ...


class ActionDeliveryProvider(Protocol):
    async def deliver(self, action: PlannedAction) -> ProviderDeliveryReceipt: ...


class DeliveryWorker:
    """Process at most one leased action with bounded, provider-idempotent retry."""

    def __init__(
        self,
        *,
        store: OutboxDeliveryStore,
        reply_provider: ActionDeliveryProvider,
        ticket_provider: ActionDeliveryProvider,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.store = store
        self.reply_provider = reply_provider
        self.ticket_provider = ticket_provider
        self.retry_policy = retry_policy or RetryPolicy()

    async def process_one(self, *, now: str) -> DeliveryResult | None:
        observed_at = _parse_timestamp(now)
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("delivery now must include a timezone")
        lease_value = await self.store.claim_next(now=now)
        if lease_value is None:
            return None
        lease = _admit(DeliveryLease, lease_value, "outbox lease")
        if not (
            _parse_timestamp(lease.claimed_at)
            <= observed_at
            < _parse_timestamp(lease.lease_expires_at)
        ):
            raise ValueError("outbox lease is not active at the delivery time")
        if lease.payload_digest != action_payload_digest(lease.action):
            raise ValueError("outbox lease payload digest does not match its action")
        if lease.max_attempts != self.retry_policy.max_attempts:
            raise ValueError("outbox lease retry bound differs from worker policy")

        provider = (
            self.reply_provider
            if lease.action.kind is ActionKind.ENQUEUE_REPLY
            else self.ticket_provider
        )
        try:
            receipt = _admit(
                ProviderDeliveryReceipt,
                await provider.deliver(lease.action),
                "delivery provider receipt",
            )
            _validate_receipt(lease, receipt)
            await self.store.mark_delivered(lease, receipt)
            return DeliveryResult(
                disposition=DeliveryDisposition.DELIVERED,
                kind=lease.action.kind,
                idempotency_key=lease.action.idempotency_key,
                attempt=lease.attempt,
                provider_receipt_ref=receipt.provider_receipt_ref,
            )
        except DeliveryProviderError as error:
            if error.retryable and lease.attempt < self.retry_policy.max_attempts:
                retry_at = _add_seconds(
                    now,
                    self.retry_policy.delay_seconds(lease.attempt),
                )
                await self.store.schedule_retry(
                    lease,
                    error_code=error.code,
                    retry_at=retry_at,
                )
                return DeliveryResult(
                    disposition=DeliveryDisposition.RETRY_SCHEDULED,
                    kind=lease.action.kind,
                    idempotency_key=lease.action.idempotency_key,
                    attempt=lease.attempt,
                    retry_at=retry_at,
                    error_code=error.code,
                )
            await self.store.dead_letter(lease, error_code=error.code)
            return DeliveryResult(
                disposition=DeliveryDisposition.DEAD_LETTER,
                kind=lease.action.kind,
                idempotency_key=lease.action.idempotency_key,
                attempt=lease.attempt,
                error_code=error.code,
            )


def action_payload_digest(action: PlannedAction) -> str:
    """Bind queue rows and provider receipts to an immutable action payload."""

    material = json.dumps(
        action.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + sha256(material.encode()).hexdigest()


def _validate_receipt(
    lease: DeliveryLease,
    receipt: ProviderDeliveryReceipt,
) -> None:
    expected = (
        lease.action.kind,
        lease.action.idempotency_key,
        lease.payload_digest,
    )
    observed = (receipt.kind, receipt.idempotency_key, receipt.payload_digest)
    if observed != expected:
        raise ValueError("provider receipt is not bound to the leased action")


def _admit(model: Any, value: Any, boundary: str) -> Any:
    payload = value.model_dump(mode="json") if isinstance(value, model) else value
    try:
        return model.model_validate_json(json.dumps(payload), strict=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{boundary} returned an invalid strict contract") from exc


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _add_seconds(value: str, seconds: int) -> str:
    result = _parse_timestamp(value) + timedelta(seconds=seconds)
    return result.isoformat().replace("+00:00", "Z")
