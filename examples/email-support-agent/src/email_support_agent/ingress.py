"""Fail-closed ingress orchestration before durable agent state.

Provider signature algorithms, malware engines, DLP, encrypted storage, and
identity resolution are injected capabilities. This module owns their ordering,
strict evidence admission, replay semantics, and the only conversion into a
``RedactedEmail``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from typing import Any, Protocol, Self

from pydantic import Field, SecretStr, ValidationInfo, field_validator, model_validator

from aai_core.contracts import ContractModel
from email_support_agent.contracts import RedactedEmail, obvious_sensitive_fragments

_OPAQUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_PSEUDONYM = re.compile(r"^(?:sha256|hmac-sha256):[0-9a-f]{64}$")
_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class VerifiedWebhook(ContractModel):
    """Metadata established by a provider-specific signature verifier."""

    ingress_provider: str = Field(min_length=3, max_length=64)
    event_id: str = Field(min_length=3, max_length=128)
    message_id: str = Field(min_length=3, max_length=128)
    thread_id: str = Field(min_length=3, max_length=128)
    received_at: str
    raw_mime_digest: str
    signature_evidence_ref: str = Field(min_length=4, max_length=512)

    @field_validator("ingress_provider", "event_id", "message_id", "thread_id")
    @classmethod
    def require_opaque_ids(cls, value: str) -> str:
        if not _OPAQUE.fullmatch(value) or "@" in value:
            raise ValueError("verified webhook identifiers must be opaque")
        return value

    @field_validator("received_at")
    @classmethod
    def require_timestamp(cls, value: str) -> str:
        if not _TIMESTAMP.fullmatch(value):
            raise ValueError("received_at must include an ISO-8601 timezone")
        return value

    @field_validator("raw_mime_digest")
    @classmethod
    def require_content_digest(cls, value: str) -> str:
        if not _SHA256_REF.fullmatch(value):
            raise ValueError("raw_mime_digest must be sha256:<hex>")
        return value

    @field_validator("signature_evidence_ref")
    @classmethod
    def require_signature_evidence(cls, value: str) -> str:
        if not value.startswith(
            ("secure://signature/", "synthetic://signature/")
        ) or any(character in value for character in ("@", "?", "#")):
            raise ValueError("signature evidence must be an opaque secure reference")
        return value


class ReplayReservation(ContractModel):
    """A replay claim or a pointer to an already completed ingress result."""

    event_digest: str
    reservation_ref: str | None = Field(default=None, max_length=512)
    prior_result_ref: str | None = Field(default=None, max_length=512)

    @field_validator("event_digest")
    @classmethod
    def require_event_digest(cls, value: str) -> str:
        if not _SHA256_REF.fullmatch(value):
            raise ValueError("event_digest must be sha256:<hex>")
        return value

    @field_validator("reservation_ref")
    @classmethod
    def require_reservation_ref(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.startswith(("secure://replay/", "synthetic://replay/"))
            or any(character in value for character in ("@", "?", "#"))
        ):
            raise ValueError("reservation_ref must be opaque")
        return value

    @field_validator("prior_result_ref")
    @classmethod
    def require_prior_result_ref(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.startswith(
                ("secure://ingress-result/", "synthetic://ingress-result/")
            )
            or any(character in value for character in ("@", "?", "#"))
        ):
            raise ValueError("prior_result_ref must be opaque")
        return value

    @model_validator(mode="after")
    def require_exactly_one_outcome(self) -> Self:
        if (self.reservation_ref is None) == (self.prior_result_ref is None):
            raise ValueError(
                "replay result must contain either a reservation or prior result"
            )
        return self


class ParsedEmail(ContractModel):
    """Transient parsed content. Its sensitive fields are excluded from repr."""

    message_id: str = Field(min_length=3, max_length=128)
    sender: SecretStr = Field(repr=False)
    subject: str = Field(min_length=1, max_length=500, repr=False)
    body: str = Field(min_length=1, max_length=100_000, repr=False)
    attachment_digests: tuple[str, ...] = Field(default=(), max_length=32)

    @field_validator("message_id")
    @classmethod
    def require_opaque_message(cls, value: str) -> str:
        if not _OPAQUE.fullmatch(value) or "@" in value:
            raise ValueError("parsed message_id must be opaque")
        return value

    @field_validator("attachment_digests")
    @classmethod
    def require_attachment_digests(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(
            not _SHA256_REF.fullmatch(item) for item in value
        ):
            raise ValueError("attachments must use unique sha256:<hex> digests")
        return value


class AttachmentScanEvidence(ContractModel):
    raw_mime_digest: str
    attachment_count: int = Field(ge=0, le=32)
    complete: bool
    malware_detected: bool
    engine_release: str = Field(min_length=1, max_length=128)
    evidence_ref: str = Field(min_length=4, max_length=512)

    @field_validator("raw_mime_digest")
    @classmethod
    def require_digest(cls, value: str) -> str:
        if not _SHA256_REF.fullmatch(value):
            raise ValueError("scan digest must be sha256:<hex>")
        return value

    @field_validator("evidence_ref")
    @classmethod
    def require_scan_ref(cls, value: str) -> str:
        if not value.startswith(("secure://scan/", "synthetic://scan/")) or any(
            character in value for character in ("@", "?", "#")
        ):
            raise ValueError("scan evidence must be an opaque reference")
        return value


class DlpRedactionResult(ContractModel):
    redacted_subject: str = Field(min_length=1, max_length=500, repr=False)
    redacted_body: str = Field(min_length=1, max_length=12_000, repr=False)
    complete: bool
    sanitization_version: str = Field(min_length=1, max_length=64)
    finding_count: int = Field(ge=0)
    evidence_ref: str = Field(min_length=4, max_length=512)

    @field_validator("evidence_ref")
    @classmethod
    def require_dlp_ref(cls, value: str) -> str:
        if not value.startswith(("secure://dlp/", "synthetic://dlp/")) or any(
            character in value for character in ("@", "?", "#")
        ):
            raise ValueError("DLP evidence must be an opaque reference")
        return value

    @model_validator(mode="after")
    def require_admitted_text(self) -> Self:
        if not self.complete:
            raise ValueError("DLP processing did not complete")
        findings = obvious_sensitive_fragments(
            f"{self.redacted_subject}\n{self.redacted_body}"
        )
        if findings:
            raise ValueError(
                "DLP output still contains obvious sensitive data: "
                + ", ".join(findings)
            )
        return self


class IngressIdentity(ContractModel):
    """Tenant/access claims resolved by a trusted identity mapping service."""

    event_id: str = Field(min_length=3, max_length=128)
    case_id: str = Field(min_length=3, max_length=128)
    tenant_id: str = Field(min_length=3, max_length=128)
    access_context_ref: str = Field(min_length=4, max_length=512)
    sender_ref: str
    authorization_evidence_ref: str = Field(min_length=4, max_length=512)

    @field_validator("event_id", "case_id", "tenant_id")
    @classmethod
    def require_opaque_ids(cls, value: str) -> str:
        if not _OPAQUE.fullmatch(value) or "@" in value:
            raise ValueError("ingress identity fields must be opaque")
        return value

    @field_validator("sender_ref")
    @classmethod
    def require_sender_pseudonym(cls, value: str) -> str:
        if not _PSEUDONYM.fullmatch(value):
            raise ValueError("sender_ref must be a pseudonymous digest")
        return value

    @field_validator("access_context_ref", "authorization_evidence_ref")
    @classmethod
    def require_access_refs(cls, value: str) -> str:
        if not value.startswith(
            (
                "secure://access/",
                "secure://authorization/",
                "synthetic://access/",
                "synthetic://authorization/",
            )
        ) or any(character in value for character in ("@", "?", "#")):
            raise ValueError("identity evidence must use opaque secure references")
        return value


class RawStoreReceipt(ContractModel):
    raw_email_ref: str = Field(min_length=4, max_length=512)
    raw_mime_digest: str
    encrypted: bool

    @field_validator("raw_email_ref")
    @classmethod
    def require_raw_ref(cls, value: str) -> str:
        if not value.startswith(("secure://raw/", "synthetic://raw/")) or any(
            character in value for character in ("@", "?", "#")
        ):
            raise ValueError("raw store receipt must use an opaque reference")
        return value

    @field_validator("raw_mime_digest")
    @classmethod
    def require_digest(cls, value: str) -> str:
        if not _SHA256_REF.fullmatch(value):
            raise ValueError("raw store digest must be sha256:<hex>")
        return value


class IngressEvidence(ContractModel):
    """Content-free audit evidence for one admitted email."""

    result_ref: str = Field(min_length=4, max_length=512)
    event_digest: str
    signature_evidence_ref: str = Field(min_length=4, max_length=512)
    replay_reservation_ref: str = Field(min_length=4, max_length=512)
    scan_evidence_ref: str = Field(min_length=4, max_length=512)
    dlp_evidence_ref: str = Field(min_length=4, max_length=512)
    authorization_evidence_ref: str = Field(min_length=4, max_length=512)
    raw_email_ref: str = Field(min_length=4, max_length=512)
    raw_mime_digest: str
    redacted_email_digest: str
    completed_at: str

    @field_validator("event_digest", "raw_mime_digest", "redacted_email_digest")
    @classmethod
    def require_digests(cls, value: str) -> str:
        if not _SHA256_REF.fullmatch(value):
            raise ValueError("ingress evidence digests must use sha256:<hex>")
        return value

    @field_validator(
        "result_ref",
        "signature_evidence_ref",
        "replay_reservation_ref",
        "scan_evidence_ref",
        "dlp_evidence_ref",
        "authorization_evidence_ref",
        "raw_email_ref",
    )
    @classmethod
    def require_opaque_evidence_refs(
        cls,
        value: str,
        info: ValidationInfo,
    ) -> str:
        suffixes = {
            "result_ref": "ingress-result/",
            "signature_evidence_ref": "signature/",
            "replay_reservation_ref": "replay/",
            "scan_evidence_ref": "scan/",
            "dlp_evidence_ref": "dlp/",
            "authorization_evidence_ref": "authorization/",
            "raw_email_ref": "raw/",
        }
        suffix = suffixes[info.field_name]
        if not value.startswith((f"secure://{suffix}", f"synthetic://{suffix}")):
            raise ValueError(f"{info.field_name} has an unapproved reference scheme")
        if any(character in value for character in ("@", "?", "#")):
            raise ValueError(f"{info.field_name} must be opaque and query-free")
        return value

    @field_validator("completed_at")
    @classmethod
    def require_timestamp(cls, value: str) -> str:
        if not _TIMESTAMP.fullmatch(value):
            raise ValueError("completed_at must include an ISO-8601 timezone")
        return value

    @model_validator(mode="after")
    def require_one_evidence_origin(self) -> Self:
        expected = (
            "secure://" if self.result_ref.startswith("secure://") else "synthetic://"
        )
        references = (
            self.signature_evidence_ref,
            self.replay_reservation_ref,
            self.scan_evidence_ref,
            self.dlp_evidence_ref,
            self.authorization_evidence_ref,
            self.raw_email_ref,
        )
        if any(not value.startswith(expected) for value in references):
            raise ValueError("ingress evidence cannot mix secure and synthetic origins")
        return self


class IngressPolicy(ContractModel):
    """Cheap admission limits applied before parsing or durable graph work."""

    max_raw_mime_bytes: int = Field(default=10_000_000, ge=1_024, le=50_000_000)
    max_event_age_seconds: int = Field(default=900, ge=1, le=86_400)
    max_clock_skew_seconds: int = Field(default=60, ge=0, le=3_600)


class WebhookVerifier(Protocol):
    async def verify(
        self,
        *,
        ingress_provider: str,
        headers: Mapping[str, str],
        raw_mime: bytes,
        observed_at: str,
    ) -> VerifiedWebhook: ...


class ReplayRegistry(Protocol):
    async def reserve(self, webhook: VerifiedWebhook) -> ReplayReservation: ...

    async def complete(
        self,
        reservation: ReplayReservation,
        *,
        result_ref: str,
    ) -> None: ...

    async def abandon(
        self,
        reservation: ReplayReservation,
        *,
        reason_code: str,
    ) -> None: ...


class MimeParser(Protocol):
    async def parse(self, raw_mime: bytes) -> ParsedEmail: ...


class AttachmentScanner(Protocol):
    async def scan(
        self,
        parsed: ParsedEmail,
        *,
        raw_mime_digest: str,
    ) -> AttachmentScanEvidence: ...


class DlpRedactor(Protocol):
    async def redact(self, parsed: ParsedEmail) -> DlpRedactionResult: ...


class IngressIdentityResolver(Protocol):
    async def resolve(
        self,
        webhook: VerifiedWebhook,
        parsed: ParsedEmail,
    ) -> IngressIdentity: ...


class SecureRawEmailStore(Protocol):
    async def put_once(
        self,
        webhook: VerifiedWebhook,
        *,
        raw_mime: bytes,
    ) -> RawStoreReceipt: ...


class DuplicateIngressError(RuntimeError):
    """Raised with only an opaque pointer to the prior completed result."""

    def __init__(self, prior_result_ref: str) -> None:
        self.prior_result_ref = prior_result_ref
        super().__init__("verified webhook was already processed")


class IngressCoordinator:
    """Verify, deduplicate, scan, redact, authorize, and persist in that order."""

    def __init__(
        self,
        *,
        verifier: WebhookVerifier,
        replay_registry: ReplayRegistry,
        parser: MimeParser,
        scanner: AttachmentScanner,
        dlp: DlpRedactor,
        identity_resolver: IngressIdentityResolver,
        raw_store: SecureRawEmailStore,
        policy: IngressPolicy | None = None,
    ) -> None:
        self.verifier = verifier
        self.replay_registry = replay_registry
        self.parser = parser
        self.scanner = scanner
        self.dlp = dlp
        self.identity_resolver = identity_resolver
        self.raw_store = raw_store
        self.policy = policy or IngressPolicy()

    async def process(
        self,
        *,
        ingress_provider: str,
        headers: Mapping[str, str],
        raw_mime: bytes,
        observed_at: str,
    ) -> tuple[RedactedEmail, IngressEvidence]:
        if not raw_mime:
            raise ValueError("raw MIME payload must not be empty")
        if len(raw_mime) > self.policy.max_raw_mime_bytes:
            raise ValueError("raw MIME payload exceeds the configured size limit")
        observed_timestamp = _parse_aware_timestamp(observed_at, "observed_at")
        webhook = _admit(
            VerifiedWebhook,
            await self.verifier.verify(
                ingress_provider=ingress_provider,
                headers=headers,
                raw_mime=raw_mime,
                observed_at=observed_at,
            ),
            "webhook verification",
        )
        if webhook.ingress_provider != ingress_provider:
            raise ValueError("verified webhook belongs to a different provider")
        observed_digest = _digest_bytes(raw_mime)
        if webhook.raw_mime_digest != observed_digest:
            raise ValueError("verified webhook is not bound to the received MIME bytes")
        received_timestamp = _parse_aware_timestamp(
            webhook.received_at,
            "verified received_at",
        )
        age_seconds = (observed_timestamp - received_timestamp).total_seconds()
        if age_seconds > self.policy.max_event_age_seconds:
            raise ValueError("verified webhook is outside the configured replay window")
        if age_seconds < -self.policy.max_clock_skew_seconds:
            raise ValueError(
                "verified webhook timestamp exceeds the clock-skew allowance"
            )

        reservation = _admit(
            ReplayReservation,
            await self.replay_registry.reserve(webhook),
            "replay reservation",
        )
        expected_event_digest = _event_digest(webhook)
        if reservation.event_digest != expected_event_digest:
            raise ValueError("replay reservation belongs to a different webhook")
        if reservation.prior_result_ref is not None:
            raise DuplicateIngressError(reservation.prior_result_ref)
        reservation_ref = reservation.reservation_ref
        if reservation_ref is None:  # Defensive narrowing after strict admission.
            raise ValueError("replay reservation did not return an active claim")

        try:
            parsed = _admit(
                ParsedEmail,
                await self.parser.parse(raw_mime),
                "MIME parser",
            )
            if parsed.message_id != webhook.message_id:
                raise ValueError("parsed MIME message does not match verified metadata")
            scan = _admit(
                AttachmentScanEvidence,
                await self.scanner.scan(
                    parsed,
                    raw_mime_digest=webhook.raw_mime_digest,
                ),
                "attachment scanner",
            )
            if (
                scan.raw_mime_digest != webhook.raw_mime_digest
                or scan.attachment_count != len(parsed.attachment_digests)
            ):
                raise ValueError("scan evidence is not bound to the parsed message")
            if not scan.complete or scan.malware_detected:
                raise ValueError("attachment scan did not produce a clean result")

            redaction = _admit(
                DlpRedactionResult,
                await self.dlp.redact(parsed),
                "DLP redaction",
            )
            identity = _admit(
                IngressIdentity,
                await self.identity_resolver.resolve(webhook, parsed),
                "ingress identity resolution",
            )
            if identity.event_id != webhook.event_id:
                raise ValueError("identity authorization belongs to another webhook")
            stored = _admit(
                RawStoreReceipt,
                await self.raw_store.put_once(webhook, raw_mime=raw_mime),
                "raw email store",
            )
            if (
                not stored.encrypted
                or stored.raw_mime_digest != webhook.raw_mime_digest
            ):
                raise ValueError(
                    "raw email was not stored with bound encryption evidence"
                )

            email = RedactedEmail(
                case_id=identity.case_id,
                message_id=webhook.message_id,
                thread_id=webhook.thread_id,
                tenant_id=identity.tenant_id,
                ingress_provider=webhook.ingress_provider,
                access_context_ref=identity.access_context_ref,
                sender_ref=identity.sender_ref,
                raw_email_ref=stored.raw_email_ref,
                subject=redaction.redacted_subject,
                body=redaction.redacted_body,
                received_at=webhook.received_at,
                attachments_scanned=True,
                redaction_complete=True,
                sanitization_version=redaction.sanitization_version,
            )
            result_ref = (
                "secure://ingress-result/"
                if stored.raw_email_ref.startswith("secure://")
                else "synthetic://ingress-result/"
            ) + expected_event_digest.removeprefix("sha256:")
            evidence = IngressEvidence(
                result_ref=result_ref,
                event_digest=expected_event_digest,
                signature_evidence_ref=webhook.signature_evidence_ref,
                replay_reservation_ref=reservation_ref,
                scan_evidence_ref=scan.evidence_ref,
                dlp_evidence_ref=redaction.evidence_ref,
                authorization_evidence_ref=identity.authorization_evidence_ref,
                raw_email_ref=stored.raw_email_ref,
                raw_mime_digest=webhook.raw_mime_digest,
                redacted_email_digest=_digest_model(email),
                completed_at=observed_at,
            )
            await self.replay_registry.complete(
                reservation,
                result_ref=evidence.result_ref,
            )
            return email, evidence
        except BaseException:
            try:
                await self.replay_registry.abandon(
                    reservation,
                    reason_code="ingress_failed",
                )
            except Exception:
                pass
            raise


def _admit(model: Any, value: Any, boundary: str) -> Any:
    payload = value.model_dump(mode="python") if isinstance(value, model) else value
    try:
        return model.model_validate(payload, strict=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{boundary} returned an invalid strict contract") from exc


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + sha256(value).hexdigest()


def _event_digest(webhook: VerifiedWebhook) -> str:
    material = json.dumps(
        {
            "provider": webhook.ingress_provider,
            "event_id": webhook.event_id,
            "raw_mime_digest": webhook.raw_mime_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + sha256(material.encode()).hexdigest()


def _digest_model(value: ContractModel) -> str:
    material = json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + sha256(material.encode()).hexdigest()


def _parse_aware_timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed
