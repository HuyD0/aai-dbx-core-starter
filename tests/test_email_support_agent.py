"""Credential-free contract tests for the email support solution accelerator."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aai_core.agentkit.config import load_config
from aai_core.tracing import TraceCaptureMode, TracePolicy, TraceState

ROOT = Path(__file__).resolve().parents[1]
ACCELERATOR = ROOT / "examples" / "email-support-agent"
SOURCE = ACCELERATOR / "src"
_ADDED_SOURCE = str(SOURCE) not in sys.path
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import email_support_agent.workflow as workflow_module  # noqa: E402
from email_support_agent import (  # noqa: E402
    AccessContext,
    BudgetExceededError,
    DeliveryOutcome,
    Disposition,
    OutcomeFeedbackSignal,
    PolicyConfig,
    RedactedEmail,
    ReviewAction,
    ReviewDecision,
    ReviewFeedbackSignal,
    ReviewReason,
    RuntimeBudget,
    SignalLinkage,
    checkpoint_state,
    feedback_ref,
    log_outcome_feedback,
    log_review_feedback,
    proposal_digest,
)
from email_support_agent.contracts import (  # noqa: E402
    ActionKind,
    Classification,
    DraftResponse,
    ExecutionResult,
    MeasurementSource,
    ModelUsage,
    PlannedAction,
)
from email_support_agent.evaluation import (  # noqa: E402
    evaluate_release_cases,
    keyword_coverage,
    load_release_cases,
    mlflow_rows,
)
from email_support_agent.offline import (  # noqa: E402
    InMemoryTransactionalOutbox,
    OfflineClassifier,
    OfflineDrafter,
    OfflineKnowledgeRetriever,
    build_offline_workflow,
    load_knowledge,
)
from email_support_agent.policy import plan_actions  # noqa: E402
from email_support_agent.workflow import EmailSupportWorkflow  # noqa: E402

if _ADDED_SOURCE:
    # The imported modules remain loaded, while unrelated tests see only the
    # repository's governed import roots. The fine-tuning evidence tests
    # deliberately reject an extra ambient sys.path entry.
    sys.path.remove(str(SOURCE))


def _cases():
    return load_release_cases(ACCELERATOR / "evals" / "data" / "release_cases.jsonl")


def _case(case_id: str):
    return next(item for item in _cases() if item.inputs.case_id == case_id)


def _decision(
    prepared,
    *,
    action: ReviewAction,
    reason: ReviewReason,
    edited_response: str | None = None,
) -> ReviewDecision:
    return ReviewDecision(
        case_id=prepared.email.case_id,
        proposal_digest=proposal_digest(prepared),
        application_release=prepared.application_release,
        authorization_ref="synthetic://review/support-quality",
        action=action,
        reason=reason,
        edited_response=edited_response,
    )


def test_release_data_uses_native_nested_mlflow_contract_and_is_synthetic():
    cases = _cases()
    rows = mlflow_rows(cases)
    assert len(cases) >= 10
    assert all(set(row) == {"inputs", "expectations"} for row in rows)
    assert all(
        row["inputs"]["raw_email_ref"].startswith("synthetic://") for row in rows
    )
    assert all("expected_response" in row["expectations"] for row in rows)
    assert not any("@" in row["inputs"]["sender_ref"] for row in rows)


def test_graph_boundary_rejects_unknown_raw_or_obviously_sensitive_content():
    valid = _case("case-faq-reset").inputs.model_dump(mode="json")

    with pytest.raises(ValidationError):
        RedactedEmail.model_validate({**valid, "raw_mime": "From: raw"}, strict=True)
    with pytest.raises(ValidationError, match="email_address"):
        RedactedEmail.model_validate(
            {**valid, "body": "Contact person@example.test for help"}, strict=True
        )
    with pytest.raises(ValidationError, match="attachments must be scanned"):
        RedactedEmail.model_validate(
            {**valid, "attachments_scanned": False}, strict=True
        )
    with pytest.raises(ValidationError, match="pseudonymous"):
        RedactedEmail.model_validate(
            {**valid, "sender_ref": "person@example.test"}, strict=True
        )
    with pytest.raises(ValidationError, match="opaque"):
        RedactedEmail.model_validate(
            {**valid, "raw_email_ref": "secure://mail/person@example.test"},
            strict=True,
        )


def test_prepare_routes_cases_without_executing_side_effects():
    async def scenario():
        workflow, outbox = build_offline_workflow(ACCELERATOR, auto_send_low_risk=True)
        prepared = [await workflow.prepare(case.inputs) for case in _cases()]
        assert not outbox.actions
        for case, result in zip(_cases(), prepared, strict=True):
            assert result.classification.intent is case.expectations.expected_intent
            assert result.classification.urgency is case.expectations.expected_urgency
            assert result.route is case.expectations.expected_route
            assert result.requires_review is case.expectations.requires_review
            assert tuple(action.kind for action in result.planned_actions) == (
                case.expectations.expected_actions
            )

    asyncio.run(scenario())


def test_checkpoint_state_is_json_compatible_and_contains_no_raw_identity():
    async def scenario():
        workflow, _ = build_offline_workflow(ACCELERATOR)
        prepared = await workflow.prepare(_case("case-billing-refund").inputs)
        state = checkpoint_state(prepared)
        serialized = json.dumps(state)
        assert "sender_ref" in state["email"]
        assert "sender_email" not in serialized
        assert "raw_mime" not in serialized
        assert "@" not in serialized
        assert "model_fields" not in serialized

    asyncio.run(scenario())


def test_retrieval_is_active_release_and_tenant_scoped():
    async def scenario():
        retriever = OfflineKnowledgeRetriever(
            load_knowledge(ACCELERATOR / "data" / "synthetic_knowledge.jsonl")
        )
        results = await retriever.retrieve(
            "export an audit report with a bounded date range",
            access=AccessContext(
                access_context_ref="synthetic://access/tenant-alpha",
                tenant_id="tenant-alpha",
                groups=("group:support-all", "group:tenant-alpha-admin"),
                authorization_evidence_ref=(
                    "synthetic://authorization/tenant-alpha/current"
                ),
            ),
            release="kb-2026-08-01",
            top_k=4,
        )
        assert "tenant-alpha-export-v1" in {item.document_id for item in results}
        assert "tenant-beta-export-v1" not in {item.document_id for item in results}
        assert all(item.active for item in results)
        for result in results:
            document = result.as_mlflow_document()
            assert document["page_content"]
            assert document["metadata"]["doc_uri"]
            assert document["metadata"]["chunk_id"]
            assert "tenant_id" not in document["metadata"]
            assert document["metadata"]["tenant_scope"] == "sha256:" + (
                "d10b4f3ef504e2c900c137014165a6dd82a8582a9d872c9711f0c62c4a157dda"
            )

    asyncio.run(scenario())


def test_malicious_retriever_outputs_fail_entitlement_and_release_checks():
    class MaliciousRetriever:
        def __init__(self, document):
            self.document = document

        async def retrieve(self, query, *, access, release, top_k):
            return (self.document,)

    async def scenario():
        source = load_knowledge(ACCELERATOR / "data" / "synthetic_knowledge.jsonl")[0]
        variants = (
            (source.model_copy(update={"tenant_id": "tenant-beta"}), "cross-tenant"),
            (source.model_copy(update={"active": False}), "inactive"),
            (
                source.model_copy(update={"allowed_groups": ("group:finance",)}),
                "entitlements",
            ),
            (source.model_copy(update={"release": "kb-2024-01-01"}), "release"),
        )
        for document, message in variants:
            base, outbox = build_offline_workflow(ACCELERATOR)
            workflow = EmailSupportWorkflow(
                access_authorizer=base.access_authorizer,
                review_authorizer=base.review_authorizer,
                classifier=base.classifier,
                retriever=MaliciousRetriever(document),
                drafter=base.drafter,
                outbox=outbox,
            )
            with pytest.raises(ValueError, match=message):
                await workflow.prepare(_case("case-faq-reset").inputs)
            assert not outbox.actions

    asyncio.run(scenario())


def test_review_interrupt_boundary_and_duplicate_commit_are_safe():
    async def scenario():
        workflow, outbox = build_offline_workflow(ACCELERATOR)
        prepared = await workflow.prepare(_case("case-bug-crash").inputs)
        assert prepared.disposition is Disposition.PENDING_REVIEW
        with pytest.raises(ValueError, match="human review is required"):
            await workflow.commit(prepared)
        assert not outbox.actions

        decision = _decision(
            prepared,
            action=ReviewAction.APPROVE,
            reason=ReviewReason.APPROVED,
        )
        # Exercise the real durable boundary: JSON-compatible checkpoint and
        # review payloads, not the original in-memory Pydantic instances.
        first = await workflow.commit(
            checkpoint_state(prepared),
            review=decision.model_dump(mode="json"),
        )
        repeated = await workflow.commit(prepared, review=decision)
        assert first.disposition is Disposition.QUEUED
        assert len(first.receipts) == 2
        assert len(outbox.actions) == 2
        assert all(receipt.duplicate for receipt in repeated.receipts)

    asyncio.run(scenario())


def test_rejection_and_unsafe_human_edit_never_enqueue():
    async def scenario():
        workflow, outbox = build_offline_workflow(ACCELERATOR)
        prepared = await workflow.prepare(_case("case-billing-refund").inputs)
        rejected = await workflow.commit(
            prepared,
            review=_decision(
                prepared,
                action=ReviewAction.REJECT,
                reason=ReviewReason.NEEDS_INVESTIGATION,
            ),
        )
        assert rejected.disposition is Disposition.HANDLED_BY_HUMAN
        assert not outbox.actions

        unsafe = _decision(
            prepared,
            action=ReviewAction.EDIT,
            reason=ReviewReason.POLICY_EDIT,
            edited_response="Safe placeholder for strict serialization.",
        ).model_dump(mode="json")
        unsafe["edited_response"] = "Email the details to person@example.test."
        with pytest.raises(ValueError, match="strict admission"):
            await workflow.commit(
                prepared,
                review=unsafe,
            )
        assert not outbox.actions

    asyncio.run(scenario())


def test_sensitive_model_output_is_quarantined_before_checkpointing():
    class LeakyDrafter:
        async def draft(
            self,
            email,
            classification,
            evidence,
            *,
            max_output_tokens,
        ):
            return {
                "body": "Contact person@example.test for the answer.",
                "citations": [],
                "abstained": False,
                "confidence": 0.99,
                "prompt_version": "leaky-v1",
                "usage": {
                    "model_calls": 1,
                    "input_tokens": 10,
                    "output_tokens": 10,
                    "cost_usd": 0.01,
                    "measurement_source": "connected",
                },
                "quarantined": False,
            }

    async def scenario():
        base, outbox = build_offline_workflow(ACCELERATOR)
        workflow = EmailSupportWorkflow(
            access_authorizer=base.access_authorizer,
            review_authorizer=base.review_authorizer,
            classifier=base.classifier,
            retriever=base.retriever,
            drafter=LeakyDrafter(),
            outbox=outbox,
        )
        prepared = await workflow.prepare(_case("case-faq-reset").inputs)
        serialized = json.dumps(checkpoint_state(prepared))

        assert "person@example.test" not in serialized
        assert prepared.draft.quarantined
        assert prepared.draft.abstained
        assert any(
            finding.name == "draft_admission" and not finding.passed
            for finding in prepared.gates
        )
        assert not prepared.planned_actions
        assert not outbox.actions

    asyncio.run(scenario())


def test_action_and_execution_evidence_reject_sensitive_or_inconsistent_state():
    with pytest.raises(ValidationError, match="sensitive-output admission"):
        PlannedAction(
            kind=ActionKind.ENQUEUE_REPLY,
            idempotency_key="a" * 64,
            case_id="case-safe-001",
            reply_body="Contact person@example.test for help.",
        )

    with pytest.raises(ValidationError, match="at least one receipt"):
        ExecutionResult(
            case_id="case-safe-001",
            application_release="release-001",
            disposition=Disposition.QUEUED,
        )

    with pytest.raises(ValidationError, match="sensitive-output admission"):
        ExecutionResult(
            case_id="case-safe-001",
            application_release="release-001",
            disposition=Disposition.QUEUED,
            final_response="Email person@example.test.",
        )


def test_review_identity_is_resolved_by_authorizer_not_caller_claims():
    async def scenario():
        workflow, outbox = build_offline_workflow(ACCELERATOR)
        prepared = await workflow.prepare(_case("case-billing-refund").inputs)
        authorizer = workflow.review_authorizer

        forged = _decision(
            prepared,
            action=ReviewAction.APPROVE,
            reason=ReviewReason.APPROVED,
        ).model_dump(mode="json")
        forged["authorization_ref"] = "synthetic://review/forged"
        with pytest.raises(PermissionError, match="missing or revoked"):
            await workflow.commit(prepared, review=forged)
        assert not outbox.actions

        self_asserted = _decision(
            prepared,
            action=ReviewAction.APPROVE,
            reason=ReviewReason.APPROVED,
        ).model_dump(mode="json")
        self_asserted["reviewer_group"] = "group:support-quality-reviewers"
        calls_before = authorizer.calls
        with pytest.raises(ValueError, match="strict admission"):
            await workflow.commit(prepared, review=self_asserted)
        assert authorizer.calls == calls_before
        assert not outbox.actions

    asyncio.run(scenario())


def test_human_edit_cannot_claim_a_citation_missing_from_the_response():
    async def scenario():
        workflow, outbox = build_offline_workflow(ACCELERATOR)
        prepared = await workflow.prepare(_case("case-faq-reset").inputs)
        with pytest.raises(ValueError, match="citation_integrity"):
            await workflow.commit(
                prepared,
                review=_decision(
                    prepared,
                    action=ReviewAction.EDIT,
                    reason=ReviewReason.FACTUAL_EDIT,
                    edited_response="Request another reset link after it expires.",
                ),
            )
        assert not outbox.actions

    asyncio.run(scenario())


def test_wrong_but_cited_answer_is_caught_by_deterministic_coverage():
    class WrongButCitedDrafter:
        async def draft(
            self,
            email,
            classification,
            evidence,
            *,
            max_output_tokens,
        ):
            document_id = evidence[0].document_id
            return DraftResponse(
                body=f"The lunar archive is blue. Source: [{document_id}].",
                citations=(document_id,),
                abstained=False,
                confidence=0.99,
                prompt_version="wrong-but-cited-v1",
                usage=ModelUsage(),
            )

    async def scenario():
        base, outbox = build_offline_workflow(ACCELERATOR)
        workflow = EmailSupportWorkflow(
            access_authorizer=base.access_authorizer,
            review_authorizer=base.review_authorizer,
            classifier=base.classifier,
            retriever=base.retriever,
            drafter=WrongButCitedDrafter(),
            outbox=outbox,
        )
        case = _case("case-faq-reset")
        prepared = await workflow.prepare(case.inputs)
        assert all(finding.passed for finding in prepared.gates)
        assert (
            keyword_coverage(
                case.expectations.expected_response,
                prepared.draft.body,
            )
            < 0.55
        )
        assert not outbox.actions

    asyncio.run(scenario())


def test_commit_rederives_actions_instead_of_trusting_checkpoint_state():
    async def scenario():
        workflow, outbox = build_offline_workflow(ACCELERATOR)
        prepared = await workflow.prepare(_case("case-billing-refund").inputs)
        checkpoint = checkpoint_state(prepared)
        checkpoint["planned_actions"][0][
            "reply_body"
        ] = "We will definitely refund every charge."

        with pytest.raises(ValueError, match="planned actions changed"):
            await workflow.commit(
                checkpoint,
                review=_decision(
                    prepared,
                    action=ReviewAction.APPROVE,
                    reason=ReviewReason.APPROVED,
                ),
            )
        assert not outbox.actions

    asyncio.run(scenario())


def test_access_revocation_between_prepare_and_commit_fails_closed():
    class RevocableAuthorizer:
        def __init__(self, delegate):
            self.delegate = delegate
            self.revoked = False

        async def authorize(self, email):
            if self.revoked:
                raise PermissionError("access authorization was revoked")
            return await self.delegate.authorize(email)

    async def scenario():
        base, outbox = build_offline_workflow(ACCELERATOR)
        authorizer = RevocableAuthorizer(base.access_authorizer)
        workflow = EmailSupportWorkflow(
            access_authorizer=authorizer,
            review_authorizer=base.review_authorizer,
            classifier=base.classifier,
            retriever=base.retriever,
            drafter=base.drafter,
            outbox=outbox,
        )
        prepared = await workflow.prepare(_case("case-billing-refund").inputs)
        authorizer.revoked = True
        with pytest.raises(PermissionError, match="revoked"):
            await workflow.commit(
                prepared,
                review=_decision(
                    prepared,
                    action=ReviewAction.APPROVE,
                    reason=ReviewReason.APPROVED,
                ),
            )
        assert not outbox.actions

    asyncio.run(scenario())


def test_safe_human_edit_can_replace_an_abstention_with_a_reply():
    async def scenario():
        workflow, outbox = build_offline_workflow(ACCELERATOR)
        prepared = await workflow.prepare(_case("case-critical-outage").inputs)
        assert prepared.draft.abstained
        assert not prepared.planned_actions

        result = await workflow.commit(
            prepared,
            review=_decision(
                prepared,
                action=ReviewAction.EDIT,
                reason=ReviewReason.FACTUAL_EDIT,
                edited_response=(
                    "A support specialist reviewed the outage and is "
                    "investigating. Verified updates will appear on the status page."
                ),
            ),
        )

        assert result.disposition is Disposition.QUEUED
        assert len(result.receipts) == 1
        assert len(outbox.actions) == 1

    asyncio.run(scenario())


def test_idempotency_is_namespaced_and_payload_collisions_fail_closed():
    async def scenario():
        workflow, _ = build_offline_workflow(ACCELERATOR)
        prepared = await workflow.prepare(_case("case-faq-reset").inputs)
        original = prepared.planned_actions[0]
        other_tenant = prepared.email.model_copy(
            update={
                "case_id": "case-beta-reset",
                "thread_id": "thread-beta-reset",
                "tenant_id": "tenant-beta",
                "access_context_ref": "synthetic://access/tenant-beta",
            }
        )
        tenant_actions = plan_actions(
            other_tenant,
            prepared.classification,
            prepared.route,
            prepared.draft,
        )
        assert tenant_actions[0].idempotency_key != original.idempotency_key

        other_provider = prepared.email.model_copy(
            update={"ingress_provider": "synthetic-mail-secondary"}
        )
        provider_actions = plan_actions(
            other_provider,
            prepared.classification,
            prepared.route,
            prepared.draft,
        )
        assert provider_actions[0].idempotency_key != original.idempotency_key

        outbox = InMemoryTransactionalOutbox()
        await outbox.enqueue_once(original)
        conflicting = PlannedAction(
            kind=ActionKind.ENQUEUE_REPLY,
            idempotency_key=original.idempotency_key,
            case_id=original.case_id,
            reply_body="A different reviewed payload.",
            citations=(),
        )
        with pytest.raises(ValueError, match="collision"):
            await outbox.enqueue_once(conflicting)
        assert outbox.actions[original.idempotency_key] == original

    asyncio.run(scenario())


def test_outbox_batch_collision_is_atomic():
    async def scenario():
        workflow, _ = build_offline_workflow(ACCELERATOR)
        prepared = await workflow.prepare(_case("case-bug-crash").inputs)
        assert len(prepared.planned_actions) == 2
        ticket, reply = prepared.planned_actions

        outbox = InMemoryTransactionalOutbox()
        await outbox.enqueue_once(reply)
        conflicting_reply = reply.model_copy(
            update={"reply_body": "A different payload must not reuse this key."}
        )

        with pytest.raises(ValueError, match="collision"):
            await outbox.enqueue_batch_once((ticket, conflicting_reply))

        assert set(outbox.actions) == {reply.idempotency_key}
        assert ticket.idempotency_key not in outbox.actions

    asyncio.run(scenario())


def test_workflow_rejects_partial_or_unbound_outbox_receipts():
    class PartialOutbox:
        async def enqueue_batch_once(self, actions):
            return ()

    class SensitiveReceiptOutbox:
        async def enqueue_batch_once(self, actions):
            action = actions[0]
            return (
                {
                    "kind": action.kind.value,
                    "idempotency_key": action.idempotency_key,
                    "status": "enqueued",
                    "external_ref": "outbox://person@example.test",
                    "duplicate": False,
                },
            )

    async def scenario():
        for outbox, message in (
            (PartialOutbox(), "receipt set"),
            (SensitiveReceiptOutbox(), "invalid strict contract"),
        ):
            workflow, _ = build_offline_workflow(ACCELERATOR)
            prepared = await workflow.prepare(_case("case-faq-reset").inputs)
            workflow.outbox = outbox
            with pytest.raises(ValueError, match=message):
                await workflow.commit(
                    prepared,
                    review=_decision(
                        prepared,
                        action=ReviewAction.APPROVE,
                        reason=ReviewReason.APPROVED,
                    ),
                )

    asyncio.run(scenario())


def test_only_explicit_low_risk_policy_can_bypass_review():
    async def scenario():
        default_workflow, default_outbox = build_offline_workflow(ACCELERATOR)
        default = await default_workflow.prepare(_case("case-faq-reset").inputs)
        assert default.requires_review

        canary, canary_outbox = build_offline_workflow(
            ACCELERATOR, auto_send_low_risk=True
        )
        prepared = await canary.prepare(_case("case-faq-reset").inputs)
        assert not prepared.requires_review
        completed = await canary.commit(prepared)
        assert completed.disposition is Disposition.QUEUED
        assert len(canary_outbox.actions) == 1
        assert not default_outbox.actions

    asyncio.run(scenario())


def test_request_budget_fails_before_commit():
    class ExpensiveClassifier(OfflineClassifier):
        async def classify(self, email, *, max_output_tokens):
            result = await super().classify(
                email,
                max_output_tokens=max_output_tokens,
            )
            return Classification(
                intent=result.intent,
                urgency=result.urgency,
                risk=result.risk,
                topic=result.topic,
                confidence=result.confidence,
                complex_issue=result.complex_issue,
                usage=ModelUsage(
                    model_calls=3,
                    input_tokens=10,
                    output_tokens=10,
                    cost_usd=0.01,
                    measurement_source=MeasurementSource.CONNECTED,
                ),
            )

    async def scenario():
        base, outbox = build_offline_workflow(ACCELERATOR)
        workflow = EmailSupportWorkflow(
            access_authorizer=base.access_authorizer,
            review_authorizer=base.review_authorizer,
            classifier=ExpensiveClassifier(),
            retriever=OfflineKnowledgeRetriever(
                load_knowledge(ACCELERATOR / "data" / "synthetic_knowledge.jsonl")
            ),
            drafter=OfflineDrafter(),
            outbox=outbox,
            policy=PolicyConfig(auto_send_low_risk=True),
            budget=RuntimeBudget(max_model_calls=2),
        )
        with pytest.raises(BudgetExceededError, match="model calls"):
            await workflow.prepare(_case("case-faq-reset").inputs)
        assert not outbox.actions

    asyncio.run(scenario())


def test_zero_budget_and_security_preflight_avoid_provider_calls():
    class CountingClassifier(OfflineClassifier):
        def __init__(self):
            self.calls = 0

        async def classify(self, email, *, max_output_tokens):
            self.calls += 1
            return await super().classify(
                email,
                max_output_tokens=max_output_tokens,
            )

    class CountingDrafter(OfflineDrafter):
        def __init__(self):
            self.calls = 0

        async def draft(
            self,
            email,
            classification,
            evidence,
            *,
            max_output_tokens,
        ):
            self.calls += 1
            return await super().draft(
                email,
                classification,
                evidence,
                max_output_tokens=max_output_tokens,
            )

    async def scenario():
        base, outbox = build_offline_workflow(ACCELERATOR)
        classifier = CountingClassifier()
        drafter = CountingDrafter()
        workflow = EmailSupportWorkflow(
            access_authorizer=base.access_authorizer,
            review_authorizer=base.review_authorizer,
            classifier=classifier,
            retriever=base.retriever,
            drafter=drafter,
            outbox=outbox,
            budget=RuntimeBudget(max_model_calls=0),
        )

        with pytest.raises(BudgetExceededError, match="classification"):
            await workflow.prepare(_case("case-faq-reset").inputs)
        assert classifier.calls == 0
        assert drafter.calls == 0

        security = await workflow.prepare(_case("case-injection").inputs)
        assert security.route.value == "escalation"
        assert classifier.calls == 0
        assert drafter.calls == 0
        assert security.total_usage.model_calls == 0
        assert not outbox.actions

    asyncio.run(scenario())


def test_classifier_usage_can_exhaust_budget_before_drafting():
    class FullBudgetClassifier(OfflineClassifier):
        async def classify(self, email, *, max_output_tokens):
            result = await super().classify(
                email,
                max_output_tokens=max_output_tokens,
            )
            return result.model_copy(
                update={
                    "usage": ModelUsage(
                        model_calls=2,
                        input_tokens=10,
                        output_tokens=10,
                        cost_usd=0.01,
                        measurement_source=MeasurementSource.CONNECTED,
                    )
                }
            )

    class NeverDrafter(OfflineDrafter):
        def __init__(self):
            self.calls = 0

        async def draft(
            self,
            email,
            classification,
            evidence,
            *,
            max_output_tokens,
        ):
            self.calls += 1
            return await super().draft(
                email,
                classification,
                evidence,
                max_output_tokens=max_output_tokens,
            )

    async def scenario():
        base, outbox = build_offline_workflow(ACCELERATOR)
        drafter = NeverDrafter()
        workflow = EmailSupportWorkflow(
            access_authorizer=base.access_authorizer,
            review_authorizer=base.review_authorizer,
            classifier=FullBudgetClassifier(),
            retriever=base.retriever,
            drafter=drafter,
            outbox=outbox,
            budget=RuntimeBudget(max_model_calls=2),
        )
        with pytest.raises(BudgetExceededError, match="drafting"):
            await workflow.prepare(_case("case-faq-reset").inputs)
        assert drafter.calls == 0
        assert not outbox.actions

    asyncio.run(scenario())


def test_domain_release_gate_passes_and_cost_is_explicitly_unknown():
    report, gate = asyncio.run(evaluate_release_cases(ACCELERATOR))
    assert gate.passed, gate.failures
    assert report.metrics["classification/critical_recall"] == 1.0
    assert report.metrics["classification/intent_macro_f1"] == 1.0
    assert report.metrics["safety/false_auto_send_rate"] == 0.0
    assert report.metrics["trajectory/no_preapproval_side_effects"] == 1.0
    assert report.metrics["trajectory/idempotency"] == 1.0
    assert report.metrics["answer/keyword_coverage"] >= 0.55
    assert report.metrics["cost/coverage"] == 0.0


def test_agentkit_and_platform_configs_are_portable_and_secret_free():
    config = load_config(ACCELERATOR / "agentkit.yaml")
    assert config.agent == "src/email_support_agent/target.py:respond"
    assert config.budget.max_judge_calls == 100
    assert config.budget.retrieved_chunks_per_row == 4
    assert "safety" in config.thresholds
    assert {
        "correctness",
        "safety",
        "guidelines",
        "retrieval_groundedness",
        "retrieval_relevance",
        "retrieval_sufficiency",
    }.issubset(set(config.scorers.add))

    smoke = load_config(ACCELERATOR / "agentkit.smoke.yaml")
    assert set(smoke.thresholds).isdisjoint({"correctness", "safety", "guidelines"})
    assert smoke.smoke.rows == 11

    platform_path = ACCELERATOR / "config" / "aai-platform.example.yml"
    platform = yaml.safe_load(platform_path.read_text(encoding="utf-8"))
    assert platform["secrets"] == {}
    assert set(platform["providers"]["models"]) == {
        "email-triage",
        "email-draft",
        "email-complex-assist",
        "judge-model",
    }
    source = platform_path.read_text(encoding="utf-8").lower()
    assert "client_secret:" not in source
    assert "api_key:" not in source
    assert "password:" not in source


def test_semantic_spans_honor_aai_capture_policy_and_hash_references(monkeypatch):
    class FakeSpan:
        def __init__(self):
            self.inputs = None
            self.outputs = None

        def set_inputs(self, value):
            self.inputs = value

        def set_outputs(self, value):
            self.outputs = value

    class SpanContext:
        def __init__(self, span):
            self.span = span

        def __enter__(self):
            return self.span

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeMlflow:
        def __init__(self):
            self.spans = []

        def start_span(self, **options):
            span = FakeSpan()
            self.spans.append(span)
            return SpanContext(span)

    fake = FakeMlflow()
    monkeypatch.setattr(workflow_module, "native_mlflow", lambda: fake)
    monkeypatch.setattr(workflow_module, "native_active_span", object)
    monkeypatch.setattr(
        workflow_module,
        "current_trace_state",
        lambda: TraceState(
            metadata={},
            policy=TracePolicy(capture_mode=TraceCaptureMode.OFF),
        ),
    )
    with workflow_module._semantic_span(
        "blocked",
        "CHAIN",
        inputs={"password": "raw-secret"},
    ) as span:
        assert span is None
    assert not fake.spans

    monkeypatch.setattr(
        workflow_module,
        "current_trace_state",
        lambda: TraceState(
            metadata={},
            policy=TracePolicy(capture_mode=TraceCaptureMode.METADATA_ONLY),
        ),
    )
    with workflow_module._semantic_span(
        "metadata",
        "CHAIN",
        inputs={"password": "raw-secret", "body": "customer text"},
    ) as span:
        workflow_module._set_outputs(span, {"external_ref": "secure://raw/value"})
    captured = json.dumps(
        {"inputs": fake.spans[0].inputs, "outputs": fake.spans[0].outputs}
    )
    assert "raw-secret" not in captured
    assert "customer text" not in captured
    assert "secure://raw/value" not in captured
    assert "provider://sensitive/receipt" not in workflow_module._trace_ref(
        "receipt", "provider://sensitive/receipt"
    )


def test_feedback_signals_are_deidentified_and_keep_release_lineage():
    class AssessmentSource:
        def __init__(self, **values):
            self.values = values

    class Entities:
        pass

    class FakeMlflow:
        def __init__(self):
            self.calls = []
            self.entities = Entities()
            self.entities.AssessmentSource = AssessmentSource

        def log_feedback(self, **values):
            self.calls.append(values)

    linkage = SignalLinkage(
        trace_id="trace-opaque-001",
        session_ref=feedback_ref("session", "thread-internal-001"),
        case_ref=feedback_ref("case", "case-internal-001"),
        application_release="email-support-reference-v1",
        proposal_digest="sha256:" + "a" * 64,
        occurred_at="2026-08-11T04:00:00Z",
    )
    fake = FakeMlflow()
    log_review_feedback(
        ReviewFeedbackSignal(
            linkage=linkage,
            action=ReviewAction.EDIT,
            reason=ReviewReason.FACTUAL_EDIT,
            reviewer_group="group:support-quality-reviewers",
            draft_edit_distance=0.25,
        ),
        mlflow_module=fake,
    )
    log_outcome_feedback(
        OutcomeFeedbackSignal(
            linkage=linkage,
            delivery_outcome=DeliveryOutcome.DELIVERED,
            resolved_first_contact=True,
            customer_reopened_7d=False,
            source_id="code:delivery-outcome-worker",
        ),
        mlflow_module=fake,
    )

    assert {call["name"] for call in fake.calls} == {
        "human_review_decision",
        "review_reason",
        "draft_edit_distance",
        "approved_unchanged",
        "delivery_outcome",
        "resolved_first_contact",
        "customer_reopened_7d",
    }
    serialized = json.dumps(fake.calls, default=lambda value: value.values)
    assert "thread-internal-001" not in serialized
    assert "case-internal-001" not in serialized
    assert "email-support-reference-v1" in serialized


def test_native_langgraph_adapter_is_optional_and_keeps_the_safety_contract():
    path = ACCELERATOR / "recipes" / "langgraph" / "graph.py"
    source = path.read_text(encoding="utf-8")
    assert "interrupt(" in source
    assert "await workflow.prepare" in source
    assert "await workflow.commit" in source
    assert "BaseCheckpointSaver" in source
    assert "InMemorySaver" not in source
    assert '"proposal_digest"' in source

    if importlib.util.find_spec("langgraph") is None:
        return
    spec = importlib.util.spec_from_file_location("email_support_graph", path)
    assert spec is not None and spec.loader is not None


def test_reference_design_names_release_feedback_cost_and_trace_controls():
    design = (ACCELERATOR / "REFERENCE_DESIGN.md").read_text(encoding="utf-8")
    for phrase in (
        "solution accelerator",
        "transactional outbox",
        "knowledge.retrieve",
        "RETRIEVER",
        "mlflow.genai.evaluate()",
        "false auto-send",
        "cost per safely resolved case",
        "aai_core.monitoring.log_feedback()",
        "agent-app",
    ):
        assert (
            phrase in (ACCELERATOR / "README.md").read_text(encoding="utf-8") + design
        )
