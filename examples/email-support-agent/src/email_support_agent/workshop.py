"""Credential-free teaching runners for the email-support accelerator.

The workshop deliberately orchestrates the accelerator's real contracts,
workflow, offline adapters, evaluation policy, and lifecycle evidence.  It does
not carry a second implementation of routing, review, retrieval, or release
logic.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    Field,
    JsonValue,
    ValidationError,
    field_serializer,
    field_validator,
)

from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.decisions import Decision, DecisionRecord
from aai_core.deployment import ApplicationRelease
from aai_core.evaluation import apply_gate
from aai_core.tracing import TraceCaptureMode, TracePolicy, sanitize_trace_payload
from email_support_agent.contracts import (
    PolicyConfig,
    PreparedCase,
    RedactedEmail,
    ReviewAction,
    ReviewDecision,
    ReviewReason,
)
from email_support_agent.evaluation import (
    evaluate_release_cases,
    load_release_cases,
    mlflow_rows,
    release_policy,
)
from email_support_agent.feedback import (
    DeliveryOutcome,
    OutcomeFeedbackSignal,
    ReviewFeedbackSignal,
    SignalLinkage,
    feedback_ref,
)
from email_support_agent.offline import build_offline_workflow
from email_support_agent.workflow import checkpoint_state, proposal_digest

ACCELERATOR_ROOT = Path(__file__).resolve().parents[2]


class WorkshopResult(ContractModel):
    """Stable output contract shared by all four executable lessons."""

    level: int = Field(ge=1, le=4)
    slug: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    title: str = Field(min_length=1, max_length=120)
    credential_mode: Literal["credential_free"] = "credential_free"
    fake_boundaries: tuple[str, ...] = Field(min_length=1)
    observations: Mapping[str, JsonValue]
    expected_observations: tuple[str, ...] = Field(min_length=1)
    failure_exercise: str = Field(min_length=1)

    @field_validator("fake_boundaries")
    @classmethod
    def require_explicit_fake_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any("TEST-ONLY" not in item and "NO REMOTE" not in item for item in value):
            raise ValueError("every workshop fake boundary must be labelled")
        return value

    @field_validator("observations", mode="after")
    @classmethod
    def freeze_observations(
        cls, value: Mapping[str, JsonValue]
    ) -> Mapping[str, JsonValue]:
        return freeze_value(value)

    @field_serializer("observations")
    def serialize_observations(
        self, value: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return thaw_value(value)

    @field_validator("failure_exercise")
    @classmethod
    def require_prediction_prompt(cls, value: str) -> str:
        if not value.startswith("Predict "):
            raise ValueError("failure exercises must begin with a prediction")
        return value


LESSON_SPECS = {
    "graph_basics": (1, "Graph basics and checkpoint-safe state"),
    "reliability_hitl_idempotency": (
        2,
        "Reliability, human review, and idempotency",
    ),
    "mlflow_trace_evaluation": (3, "MLflow trace and evaluation contracts"),
    "improvement_release_decision": (
        4,
        "Improvement signals and release decisions",
    ),
}

TRACE_SPAN_CONTRACT = (
    "input.guardrail",
    "intent.classify",
    "route.select",
    "knowledge.retrieve",
    "response.draft",
    "response.policy_gate",
)

_Runner = Callable[[], Awaitable[WorkshopResult]]


def run_lesson(slug: str) -> WorkshopResult:
    """Execute one workshop level without credentials or external writes."""

    runners: dict[str, _Runner] = {
        "graph_basics": _graph_basics,
        "reliability_hitl_idempotency": _reliability_hitl_idempotency,
        "mlflow_trace_evaluation": _mlflow_trace_evaluation,
        "improvement_release_decision": _improvement_release_decision,
    }
    try:
        runner = runners[slug]
    except KeyError as error:
        raise ValueError(
            f"unknown workshop lesson {slug!r}; choose one of: " + ", ".join(runners)
        ) from error
    return asyncio.run(runner())


def emit_lesson(slug: str) -> None:
    """Print a readable summary plus a machine-verifiable result line."""

    result = run_lesson(slug)
    print(f"Level {result.level}: {result.title}")
    print("Credential mode: credential-free; external writes: disabled")
    print("Expected observations:")
    for observation in result.expected_observations:
        print(f"- {observation}")
    print(f"Failure exercise: {result.failure_exercise}")
    print("WORKSHOP_RESULT=" + result.model_dump_json())


async def _graph_basics() -> WorkshopResult:
    case = _case("case-faq-reset")
    workflow, outbox = build_offline_workflow(ACCELERATOR_ROOT)
    prepared = await workflow.prepare(case.inputs)
    state = checkpoint_state(prepared)
    restored = PreparedCase.model_validate_json(json.dumps(state), strict=True)

    malformed = case.inputs.model_dump(mode="json")
    malformed["raw_mime"] = "From: untrusted raw content"
    strict_failure = _captures_validation_failure(
        lambda: RedactedEmail.model_validate_json(
            json.dumps(malformed),
            strict=True,
        )
    )

    conditional_edge = "review" if prepared.requires_review else "commit"
    return _result(
        "graph_basics",
        fake_boundaries=(
            "OfflineAccessAuthorizer (TEST-ONLY verified-ingress stand-in)",
            "OfflineClassifier (TEST-ONLY deterministic model stand-in)",
            "OfflineKnowledgeRetriever (TEST-ONLY search stand-in)",
            "OfflineDrafter (TEST-ONLY generation stand-in)",
            "InMemoryTransactionalOutbox (TEST-ONLY, not durable)",
        ),
        observations={
            "case_id": prepared.email.case_id,
            "state_contract": type(prepared).__name__,
            "state_json_round_trip": restored == prepared,
            "route": prepared.route.value,
            "conditional_edge": conditional_edge,
            "node_sequence": list(TRACE_SPAN_CONTRACT),
            "evidence_document_ids": [
                document.document_id for document in prepared.evidence
            ],
            "planned_action_kinds": [
                action.kind.value for action in prepared.planned_actions
            ],
            "outbox_writes_before_commit": len(outbox.actions),
            "strict_admission_failure_observed": strict_failure,
        },
        expected_observations=(
            "prepare() returns immutable PreparedCase state that round-trips as JSON.",
            "The route selects the review edge under the safe default policy.",
            "No ticket or reply is written while the graph is preparing state.",
            "An unknown raw MIME field is rejected at the strict graph boundary.",
        ),
        failure_exercise=(
            "Predict whether adding raw_mime to RedactedEmail is ignored or rejected; "
            "then confirm strict_admission_failure_observed is true."
        ),
    )


async def _reliability_hitl_idempotency() -> WorkshopResult:
    case = _case("case-bug-crash")
    workflow, outbox = build_offline_workflow(ACCELERATOR_ROOT)
    prepared = await workflow.prepare(case.inputs)

    preapproval_blocked = False
    try:
        await workflow.commit(prepared)
    except ValueError as error:
        preapproval_blocked = "human review is required" in str(error)
    writes_before_approval = len(outbox.actions)

    decision = ReviewDecision(
        case_id=prepared.email.case_id,
        proposal_digest=proposal_digest(prepared),
        application_release=prepared.application_release,
        authorization_ref="synthetic://review/support-quality",
        action=ReviewAction.APPROVE,
        reason=ReviewReason.APPROVED,
    )
    first = await workflow.commit(
        checkpoint_state(prepared),
        review=decision.model_dump(mode="json"),
    )
    repeated = await workflow.commit(prepared, review=decision)

    forged_workflow, forged_outbox = build_offline_workflow(ACCELERATOR_ROOT)
    forged_prepared = await forged_workflow.prepare(case.inputs)
    forged_decision = ReviewDecision(
        case_id=forged_prepared.email.case_id,
        proposal_digest=proposal_digest(forged_prepared),
        application_release=forged_prepared.application_release,
        authorization_ref="synthetic://review/schema-valid-but-untrusted",
        action=ReviewAction.APPROVE,
        reason=ReviewReason.APPROVED,
    )
    forged_authorization_blocked = False
    try:
        await forged_workflow.commit(forged_prepared, review=forged_decision)
    except PermissionError:
        forged_authorization_blocked = True

    return _result(
        "reliability_hitl_idempotency",
        fake_boundaries=(
            "OfflineAccessAuthorizer (TEST-ONLY verified-ingress stand-in)",
            "OfflineReviewAuthorizer (TEST-ONLY identity-provider stand-in)",
            "InMemoryTransactionalOutbox (TEST-ONLY, not transactional storage)",
            "Offline model and retrieval adapters (TEST-ONLY deterministic fakes)",
        ),
        observations={
            "case_id": prepared.email.case_id,
            "pending_review": prepared.requires_review,
            "preapproval_commit_blocked": preapproval_blocked,
            "outbox_writes_before_approval": writes_before_approval,
            "proposal_digest": proposal_digest(prepared),
            "first_receipts_duplicate": [
                receipt.duplicate for receipt in first.receipts
            ],
            "retry_receipts_duplicate": [
                receipt.duplicate for receipt in repeated.receipts
            ],
            "unique_outbox_actions": len(outbox.actions),
            "outbox_attempts": outbox.attempts,
            "verified_reviewer_group": first.reviewer_group,
            "forged_authorization_blocked": forged_authorization_blocked,
            "forged_outbox_actions": len(forged_outbox.actions),
        },
        expected_observations=(
            "The write boundary refuses a review-controlled case before approval.",
            "The first approved commit enqueues one ticket and one reply.",
            "Replaying the proposal produces duplicate receipts, not duplicate rows.",
            "A schema-valid authorization reference is denied by the trusted port.",
        ),
        failure_exercise=(
            "Predict whether a well-formed but unknown review authorization can "
            "approve the case; then confirm the trusted authorizer blocks it."
        ),
    )


async def _mlflow_trace_evaluation() -> WorkshopResult:
    cases = load_release_cases(
        ACCELERATOR_ROOT / "evals" / "data" / "release_cases.jsonl"
    )
    rows = mlflow_rows(cases)
    report, gate = await evaluate_release_cases(ACCELERATOR_ROOT)

    case = next(item for item in cases if item.inputs.case_id == "case-tenant-export")
    workflow, outbox = build_offline_workflow(
        ACCELERATOR_ROOT,
        auto_send_low_risk=True,
    )
    prepared = await workflow.prepare(case.inputs)
    if not prepared.evidence:
        raise RuntimeError("trace lesson requires scorer-visible retrieval evidence")
    document = prepared.evidence[0].as_mlflow_document()
    trace_shape = sanitize_trace_payload(
        {
            "subject": case.inputs.subject,
            "body": case.inputs.body,
            "retrieved_documents": [document],
        },
        policy=TracePolicy(capture_mode=TraceCaptureMode.METADATA_ONLY),
    )

    failing_metrics = dict(report.metrics)
    failing_metrics["safety/false_auto_send_rate"] = 1.0
    failure_gate = apply_gate(failing_metrics, policy=release_policy())

    selected_metrics = {
        name: report.metrics[name]
        for name in (
            "classification/critical_recall",
            "safety/false_auto_send_rate",
            "retrieval/recall_at_k",
            "trajectory/idempotency",
            "cost/coverage",
        )
    }
    return _result(
        "mlflow_trace_evaluation",
        fake_boundaries=(
            "Offline evaluation target (TEST-ONLY deterministic application)",
            "Offline provider adapters (TEST-ONLY; zero model or judge calls)",
            "NO REMOTE trace is persisted; span output is a labelled contract",
            "InMemoryTransactionalOutbox (TEST-ONLY, used only by safety checks)",
        ),
        observations={
            "evaluation_case_count": len(cases),
            "mlflow_row_top_level_keys": sorted(rows[0]),
            "mlflow_row_inputs_are_nested": isinstance(rows[0]["inputs"], dict),
            "mlflow_row_expectations_are_nested": isinstance(
                rows[0]["expectations"], dict
            ),
            "span_contract": list(TRACE_SPAN_CONTRACT),
            "trace_claim": "offline_contract_only_no_trace_id",
            "trace_capture_mode": TraceCaptureMode.METADATA_ONLY.value,
            "trace_payload_shape": trace_shape,
            "retriever_document_top_level_keys": sorted(document),
            "retriever_document_metadata_keys": sorted(document["metadata"]),
            "gate_passed": gate.passed,
            "selected_metrics": selected_metrics,
            "false_auto_send_failure_gate_passed": failure_gate.passed,
            "outbox_writes_from_prepare": len(outbox.actions),
        },
        expected_observations=(
            "Evaluation rows use MLflow's nested inputs and expectations contract.",
            "Retriever evidence carries page_content, doc_uri, and chunk_id fields.",
            "The deterministic release gate passes the checked-in synthetic set.",
            "A non-zero false-auto-send rate fails the hard safety policy.",
            "Metadata-only capture describes payload shape without storing email text.",
        ),
        failure_exercise=(
            "Predict the gate result when false_auto_send_rate becomes 1.0; then "
            "confirm false_auto_send_failure_gate_passed is false."
        ),
    )


async def _improvement_release_decision() -> WorkshopResult:
    report, gate = await evaluate_release_cases(ACCELERATOR_ROOT)
    policy = PolicyConfig()
    release = ApplicationRelease(
        application="email-support-agent",
        release=policy.application_release,
        source_commit="synthetic-workshop-source",
        core_sdk_version="worktree",
        model={
            "classifier": "offline-deterministic-v1",
            "drafter": "offline-deterministic-v1",
        },
        prompt={"draft": "offline-deterministic-v1"},
        retrieval={
            "logical_name": "support-knowledge",
            "knowledge_release": policy.knowledge_release,
            "index_release": "synthetic-index-v1",
            "embedding_release": "not_used_by_offline_fixture",
            "chunking_release": "synthetic-chunks-v1",
        },
        evaluation={
            "dataset": "evals/data/release_cases.jsonl",
            "case_count": len(report.cases),
            "measurement_source": "offline_fixture",
        },
        environment="offline-workshop",
    )
    decision = DecisionRecord(
        decision=Decision.INCONCLUSIVE,
        change_id="enable_low_risk_canary",
        change_summary="Permit only policy-qualified low-risk knowledge replies.",
        rationale=(
            "The deterministic gate passed, but an offline fake supplies neither "
            "a connected baseline, production outcomes, nor provider cost evidence."
        ),
        baseline_run_id="baseline_fixture_001",
        change_run_id="change_fixture_001",
        gate=gate,
        release_digest=release.digest,
        decided_by=policy.required_reviewer_group,
    )

    linkage = SignalLinkage(
        trace_id="trace-workshop-synthetic",
        session_ref=feedback_ref("session", "thread-workshop"),
        case_ref=feedback_ref("case", "case-workshop"),
        application_release=policy.application_release,
        occurred_at="2026-08-10T12:30:00Z",
    )
    review_signal = ReviewFeedbackSignal(
        linkage=linkage,
        action=ReviewAction.EDIT,
        reason=ReviewReason.FACTUAL_EDIT,
        reviewer_group=policy.required_reviewer_group,
        draft_edit_distance=0.25,
    )
    outcome_signal = OutcomeFeedbackSignal(
        linkage=linkage,
        delivery_outcome=DeliveryOutcome.DELIVERED,
        resolved_first_contact=True,
        customer_reopened_7d=False,
        source_id="code:delivery-worker",
    )

    failing_metrics = dict(report.metrics)
    failing_metrics["safety/false_auto_send_rate"] = 1.0
    failing_gate = apply_gate(failing_metrics, policy=release_policy())
    adopt_with_failing_gate_blocked = _captures_validation_failure(
        lambda: DecisionRecord(
            decision=Decision.ADOPT,
            change_id="unsafe_auto_send",
            change_summary="Unsafe counterexample used only by the workshop.",
            rationale="This construction must fail because its gate rejects.",
            baseline_run_id="baseline_fixture_001",
            change_run_id="unsafe_fixture_001",
            gate=failing_gate,
            release_digest=release.digest,
            decided_by=policy.required_reviewer_group,
        )
    )

    return _result(
        "improvement_release_decision",
        fake_boundaries=(
            "Synthetic review and delivery signals (TEST-ONLY evidence shape)",
            "Synthetic run identifiers (TEST-ONLY; no MLflow run is claimed)",
            "Offline evaluation target (TEST-ONLY deterministic application)",
            "NO REMOTE prompt alias, model, index, or deployment is changed",
        ),
        observations={
            "lifecycle": {
                "hypothesis": (
                    "A tightly bounded low-risk canary can reduce review load "
                    "without increasing false auto-send."
                ),
                "baseline": "review_every_reply",
                "change": "policy_qualified_low_risk_canary",
                "result": {
                    "deterministic_gate_passed": gate.passed,
                    "false_auto_send_rate": report.metrics[
                        "safety/false_auto_send_rate"
                    ],
                    "cost_coverage": report.metrics["cost/coverage"],
                },
                "decision": decision.decision.value,
                "release": "blocked_until_connected_evidence",
            },
            "application_release_digest": release.digest,
            "review_feedback_action": review_signal.action.value,
            "review_feedback_edit_distance": review_signal.draft_edit_distance,
            "outcome_feedback": {
                "delivery": outcome_signal.delivery_outcome.value,
                "resolved_first_contact": outcome_signal.resolved_first_contact,
                "customer_reopened_7d": outcome_signal.customer_reopened_7d,
            },
            "adopt_with_failing_gate_blocked": adopt_with_failing_gate_blocked,
            "production_promotion_authorized": False,
        },
        expected_observations=(
            "The lesson records baseline, change, result, decision, and release.",
            "A passing fixture gate still ends inconclusive for production.",
            "Review corrections and verified outcomes are typed improvement signals.",
            "The application release digest binds retrieval and evaluation evidence.",
            "An adopt decision cannot cite a failing gate.",
        ),
        failure_exercise=(
            "Predict whether DecisionRecord permits adopt with a false-auto-send "
            "failure; then confirm adopt_with_failing_gate_blocked is true."
        ),
    )


def _case(case_id: str) -> Any:
    cases = load_release_cases(
        ACCELERATOR_ROOT / "evals" / "data" / "release_cases.jsonl"
    )
    return next(item for item in cases if item.inputs.case_id == case_id)


def _captures_validation_failure(operation: Callable[[], Any]) -> bool:
    try:
        operation()
    except (TypeError, ValueError, ValidationError):
        return True
    return False


def _result(
    slug: str,
    *,
    fake_boundaries: tuple[str, ...],
    observations: Mapping[str, JsonValue],
    expected_observations: tuple[str, ...],
    failure_exercise: str,
) -> WorkshopResult:
    level, title = LESSON_SPECS[slug]
    return WorkshopResult(
        level=level,
        slug=slug,
        title=title,
        fake_boundaries=fake_boundaries,
        observations=observations,
        expected_observations=expected_observations,
        failure_exercise=failure_exercise,
    )
