"""Credential-free tests for the production ingress trust boundary."""

from __future__ import annotations

import asyncio
import sys
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "email-support-agent" / "src"
_ADDED_SOURCE = str(SOURCE) not in sys.path
if _ADDED_SOURCE:
    sys.path.insert(0, str(SOURCE))

import email_support_agent.ingress as ingress_module  # noqa: E402
from email_support_agent.ingress import (  # noqa: E402
    AttachmentScanEvidence,
    DlpRedactionResult,
    DuplicateIngressError,
    IngressCoordinator,
    IngressEvidence,
    IngressIdentity,
    IngressPolicy,
    ParsedEmail,
    RawStoreReceipt,
    ReplayReservation,
    VerifiedWebhook,
)

if _ADDED_SOURCE:
    sys.path.remove(str(SOURCE))

RAW_MIME = b"From: person@example.test\r\nSubject: Reset\r\n\r\nHelp me reset it."
RAW_DIGEST = "sha256:" + sha256(RAW_MIME).hexdigest()
OBSERVED_AT = "2026-08-11T12:00:00Z"


class IngressHarness:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.webhook = VerifiedWebhook(
            ingress_provider="synthetic-mail",
            event_id="event-001",
            message_id="message-001",
            thread_id="thread-001",
            received_at=OBSERVED_AT,
            raw_mime_digest=RAW_DIGEST,
            signature_evidence_ref="synthetic://signature/event-001",
        )
        self.parsed = ParsedEmail(
            message_id="message-001",
            sender="person@example.test",
            subject="Reset for person@example.test",
            body="Please help person@example.test reset the account.",
            attachment_digests=("sha256:" + "1" * 64,),
        )
        self.scan_result = AttachmentScanEvidence(
            raw_mime_digest=RAW_DIGEST,
            attachment_count=1,
            complete=True,
            malware_detected=False,
            engine_release="scanner-2026-08",
            evidence_ref="synthetic://scan/event-001",
        )
        self.dlp_result = DlpRedactionResult(
            redacted_subject="Password reset request",
            redacted_body="Please help the customer reset the account.",
            complete=True,
            sanitization_version="dlp-2026-08",
            finding_count=2,
            evidence_ref="synthetic://dlp/event-001",
        )
        self.identity = IngressIdentity(
            event_id="event-001",
            case_id="case-001",
            tenant_id="tenant-alpha",
            access_context_ref="synthetic://access/tenant-alpha",
            sender_ref="sha256:" + "2" * 64,
            authorization_evidence_ref=(
                "synthetic://authorization/tenant-alpha/current"
            ),
        )
        self.store_result = RawStoreReceipt(
            raw_email_ref="synthetic://raw/event-001",
            raw_mime_digest=RAW_DIGEST,
            encrypted=True,
        )
        self.prior_result_ref: str | None = None
        self.complete_error: Exception | None = None

    async def verify(self, **kwargs):
        self.calls.append("verify")
        assert kwargs["raw_mime"] == RAW_MIME
        return self.webhook

    async def reserve(self, webhook):
        self.calls.append("reserve")
        event_digest = ingress_module._event_digest(webhook)
        if self.prior_result_ref is not None:
            return ReplayReservation(
                event_digest=event_digest,
                prior_result_ref=self.prior_result_ref,
            )
        return ReplayReservation(
            event_digest=event_digest,
            reservation_ref="synthetic://replay/event-001",
        )

    async def parse(self, raw_mime):
        self.calls.append("parse")
        assert raw_mime == RAW_MIME
        return self.parsed

    async def scan(self, parsed, *, raw_mime_digest):
        self.calls.append("scan")
        assert parsed is not None and raw_mime_digest == RAW_DIGEST
        return self.scan_result

    async def redact(self, parsed):
        self.calls.append("dlp")
        assert parsed is not None
        return self.dlp_result

    async def resolve(self, webhook, parsed):
        self.calls.append("identity")
        assert webhook is not None and parsed is not None
        return self.identity

    async def put_once(self, webhook, *, raw_mime):
        self.calls.append("store")
        assert webhook is not None and raw_mime == RAW_MIME
        return self.store_result

    async def complete(self, reservation, *, result_ref):
        self.calls.append("complete")
        assert reservation.reservation_ref and result_ref
        if self.complete_error is not None:
            raise self.complete_error

    async def abandon(self, reservation, *, reason_code):
        self.calls.append("abandon")
        assert reservation.reservation_ref and reason_code == "ingress_failed"


def _coordinator(harness: IngressHarness) -> IngressCoordinator:
    return IngressCoordinator(
        verifier=harness,
        replay_registry=harness,
        parser=harness,
        scanner=harness,
        dlp=harness,
        identity_resolver=harness,
        raw_store=harness,
    )


async def _process(harness: IngressHarness):
    return await _coordinator(harness).process(
        ingress_provider="synthetic-mail",
        headers={"x-synthetic-signature": "not-persisted"},
        raw_mime=RAW_MIME,
        observed_at=OBSERVED_AT,
    )


def test_ingress_orders_all_controls_and_emits_only_redacted_durable_state():
    async def scenario():
        harness = IngressHarness()
        email, evidence = await _process(harness)

        assert harness.calls == [
            "verify",
            "reserve",
            "parse",
            "scan",
            "dlp",
            "identity",
            "store",
            "complete",
        ]
        assert email.subject == "Password reset request"
        assert email.body == "Please help the customer reset the account."
        assert "person@example.test" not in repr(email)
        assert "person@example.test" not in repr(harness.parsed)
        assert evidence.raw_mime_digest == RAW_DIGEST
        assert RAW_MIME.decode() not in evidence.model_dump_json()
        assert evidence.result_ref.startswith("synthetic://ingress-result/")

    asyncio.run(scenario())


def test_duplicate_webhook_returns_only_prior_opaque_result_pointer():
    async def scenario():
        harness = IngressHarness()
        harness.prior_result_ref = "synthetic://ingress-result/prior"

        with pytest.raises(DuplicateIngressError) as error:
            await _process(harness)

        assert error.value.prior_result_ref == harness.prior_result_ref
        assert harness.calls == ["verify", "reserve"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mutation", "message", "last_call"),
    [
        (
            lambda h: setattr(
                h,
                "parsed",
                h.parsed.model_copy(update={"message_id": "message-other"}),
            ),
            "parsed MIME message",
            "abandon",
        ),
        (
            lambda h: setattr(
                h,
                "scan_result",
                h.scan_result.model_copy(update={"malware_detected": True}),
            ),
            "clean result",
            "abandon",
        ),
        (
            lambda h: setattr(
                h,
                "identity",
                h.identity.model_copy(update={"event_id": "event-other"}),
            ),
            "another webhook",
            "abandon",
        ),
        (
            lambda h: setattr(
                h,
                "store_result",
                h.store_result.model_copy(update={"encrypted": False}),
            ),
            "bound encryption",
            "abandon",
        ),
    ],
)
def test_ingress_failures_abandon_replay_claim_without_emitting_state(
    mutation,
    message,
    last_call,
):
    async def scenario():
        harness = IngressHarness()
        mutation(harness)
        with pytest.raises(ValueError, match=message):
            await _process(harness)
        assert harness.calls[-1] == last_call
        assert "complete" not in harness.calls

    asyncio.run(scenario())


def test_dlp_and_production_pseudonym_contracts_fail_closed():
    async def dlp_scenario():
        harness = IngressHarness()
        # model_copy intentionally simulates a provider bypassing constructor checks;
        # the coordinator revalidates every provider object.
        harness.dlp_result = harness.dlp_result.model_copy(
            update={"redacted_body": "Contact person@example.test"}
        )
        with pytest.raises(ValueError, match="DLP redaction"):
            await _process(harness)
        assert "store" not in harness.calls
        assert harness.calls[-1] == "abandon"

    async def pseudonym_scenario():
        harness = IngressHarness()
        harness.store_result = RawStoreReceipt(
            raw_email_ref="secure://raw/event-001",
            raw_mime_digest=RAW_DIGEST,
            encrypted=True,
        )
        harness.identity = harness.identity.model_copy(
            update={
                "access_context_ref": "secure://access/tenant-alpha",
                "authorization_evidence_ref": (
                    "secure://authorization/tenant-alpha/current"
                ),
                "sender_ref": "sha256:" + "3" * 64,
            }
        )
        with pytest.raises(ValueError, match="keyed hmac"):
            await _process(harness)
        assert harness.calls[-1] == "abandon"

    asyncio.run(dlp_scenario())
    asyncio.run(pseudonym_scenario())


def test_completion_failure_releases_claim_for_safe_retry():
    async def scenario():
        harness = IngressHarness()
        harness.complete_error = RuntimeError("synthetic completion failure")
        with pytest.raises(RuntimeError, match="completion failure"):
            await _process(harness)
        assert harness.calls[-2:] == ["complete", "abandon"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("result_ref", "synthetic://ingress-result/person@example.test"),
        ("signature_evidence_ref", "synthetic://signature/event?token=raw"),
        ("replay_reservation_ref", "secure://replay/event-001"),
        ("scan_evidence_ref", "synthetic://other/event-001"),
    ],
)
def test_persisted_ingress_evidence_revalidates_opaque_origin_bound_refs(
    field,
    value,
):
    async def scenario():
        harness = IngressHarness()
        _, evidence = await _process(harness)
        payload = evidence.model_dump(mode="json")
        payload[field] = value

        with pytest.raises(ValueError):
            IngressEvidence.model_validate(payload, strict=True)

    asyncio.run(scenario())


def test_size_and_replay_window_fail_before_parser_or_replay_registry():
    async def size_scenario():
        harness = IngressHarness()
        coordinator = IngressCoordinator(
            verifier=harness,
            replay_registry=harness,
            parser=harness,
            scanner=harness,
            dlp=harness,
            identity_resolver=harness,
            raw_store=harness,
            policy=IngressPolicy(max_raw_mime_bytes=1_024),
        )
        with pytest.raises(ValueError, match="size limit"):
            await coordinator.process(
                ingress_provider="synthetic-mail",
                headers={},
                raw_mime=b"x" * 1_025,
                observed_at=OBSERVED_AT,
            )
        assert not harness.calls

    async def stale_scenario():
        harness = IngressHarness()
        harness.webhook = harness.webhook.model_copy(
            update={"received_at": "2026-08-11T11:00:00Z"}
        )
        with pytest.raises(ValueError, match="replay window"):
            await _process(harness)
        assert harness.calls == ["verify"]

    asyncio.run(size_scenario())
    asyncio.run(stale_scenario())
