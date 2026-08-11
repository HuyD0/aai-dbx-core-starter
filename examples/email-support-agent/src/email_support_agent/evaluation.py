"""Credential-free domain evaluation and deterministic release gate."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import Field, field_serializer, field_validator

from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.evaluation import (
    GatePolicy,
    GateResult,
    MetricDirection,
    MetricRule,
    apply_gate,
)
from email_support_agent.contracts import (
    ActionKind,
    Intent,
    RedactedEmail,
    ReviewAction,
    ReviewDecision,
    ReviewReason,
    Route,
    Urgency,
)
from email_support_agent.offline import build_offline_workflow
from email_support_agent.workflow import proposal_digest


class ReleaseExpectations(ContractModel):
    expected_response: str = Field(min_length=1)
    expected_intent: Intent
    expected_urgency: Urgency
    expected_route: Route
    requires_review: bool
    expected_document_ids: tuple[str, ...] = ()
    expected_actions: tuple[ActionKind, ...] = ()
    expected_abstention: bool = False


class ReleaseCase(ContractModel):
    inputs: RedactedEmail
    expectations: ReleaseExpectations


class CaseResult(ContractModel):
    case_id: str
    intent_correct: bool
    urgency_correct: bool
    route_correct: bool
    review_correct: bool
    false_auto_send: bool
    retrieval_recall: float = Field(ge=0.0, le=1.0)
    citation_integrity: bool
    privacy_passed: bool
    action_plan_correct: bool
    abstention_correct: bool
    policy_gates_passed: bool
    answer_keyword_coverage: float = Field(ge=0.0, le=1.0)
    model_calls: int = Field(ge=0)


class EvaluationReport(ContractModel):
    metrics: Mapping[str, float]
    cases: tuple[CaseResult, ...]

    @field_validator("metrics", mode="after")
    @classmethod
    def freeze_metrics(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return freeze_value(value)

    @field_serializer("metrics")
    def serialize_metrics(self, value: Mapping[str, float]) -> dict[str, float]:
        return thaw_value(value)


def load_release_cases(path: str | Path) -> tuple[ReleaseCase, ...]:
    cases = tuple(
        ReleaseCase.model_validate_json(line, strict=True)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not cases:
        raise ValueError("release dataset must not be empty")
    identifiers = [case.inputs.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("release case ids must be unique")
    return cases


async def evaluate_release_cases(
    root: str | Path,
) -> tuple[EvaluationReport, GateResult]:
    """Evaluate the canary automation policy without committing any action."""

    project_root = Path(root)
    cases = load_release_cases(project_root / "evals" / "data" / "release_cases.jsonl")
    workflow, outbox = build_offline_workflow(
        project_root,
        auto_send_low_risk=True,
    )
    results: list[CaseResult] = []
    intent_pairs: list[tuple[Intent, Intent]] = []
    retrieval_scores: list[float] = []
    human_required_auto_send: list[float] = []
    critical_expected = 0
    critical_recalled = 0
    for case in cases:
        prepared = await workflow.prepare(case.inputs)
        expected = case.expectations
        expected_documents = set(expected.expected_document_ids)
        observed_documents = {item.document_id for item in prepared.evidence}
        retrieval_recall = (
            len(expected_documents.intersection(observed_documents))
            / len(expected_documents)
            if expected_documents
            else 1.0
        )
        if expected.expected_urgency is Urgency.CRITICAL:
            critical_expected += 1
            critical_recalled += int(
                prepared.classification.urgency is Urgency.CRITICAL
            )
        intent_pairs.append((expected.expected_intent, prepared.classification.intent))
        if expected_documents:
            retrieval_scores.append(retrieval_recall)
        if expected.requires_review:
            human_required_auto_send.append(float(not prepared.requires_review))
        results.append(
            CaseResult(
                case_id=case.inputs.case_id,
                intent_correct=(
                    prepared.classification.intent is expected.expected_intent
                ),
                urgency_correct=(
                    prepared.classification.urgency is expected.expected_urgency
                ),
                route_correct=prepared.route is expected.expected_route,
                review_correct=prepared.requires_review is expected.requires_review,
                false_auto_send=(
                    expected.requires_review and not prepared.requires_review
                ),
                retrieval_recall=retrieval_recall,
                citation_integrity=next(
                    finding.passed
                    for finding in prepared.gates
                    if finding.name == "citation_integrity"
                ),
                privacy_passed=next(
                    finding.passed
                    for finding in prepared.gates
                    if finding.name == "privacy"
                ),
                action_plan_correct=(
                    tuple(action.kind for action in prepared.planned_actions)
                    == expected.expected_actions
                ),
                abstention_correct=(
                    prepared.draft.abstained is expected.expected_abstention
                ),
                policy_gates_passed=all(item.passed for item in prepared.gates),
                answer_keyword_coverage=keyword_coverage(
                    expected.expected_response,
                    prepared.draft.body,
                ),
                model_calls=prepared.total_usage.model_calls,
            )
        )
    idempotency_passed, rejection_passed = await _side_effect_contracts(
        project_root, cases
    )
    metrics = {
        "schema/validity": 1.0,
        "classification/intent_accuracy": _ratio(results, "intent_correct"),
        "classification/intent_macro_f1": _macro_f1(intent_pairs),
        "classification/urgency_accuracy": _ratio(results, "urgency_correct"),
        "classification/critical_recall": (
            critical_recalled / critical_expected if critical_expected else 1.0
        ),
        "routing/accuracy": _ratio(results, "route_correct"),
        "safety/review_policy_accuracy": _ratio(results, "review_correct"),
        "safety/false_auto_send_rate": (
            mean(human_required_auto_send) if human_required_auto_send else 0.0
        ),
        "retrieval/recall_at_k": (mean(retrieval_scores) if retrieval_scores else 1.0),
        "answer/citation_integrity": _ratio(results, "citation_integrity"),
        "answer/abstention_accuracy": _ratio(results, "abstention_correct"),
        "answer/policy_gate_pass_rate": _ratio(results, "policy_gates_passed"),
        "answer/keyword_coverage": mean(
            item.answer_keyword_coverage for item in results
        ),
        "privacy/obvious_pii_free": _ratio(results, "privacy_passed"),
        "trajectory/action_plan_accuracy": _ratio(results, "action_plan_correct"),
        "trajectory/no_preapproval_side_effects": float(not outbox.actions),
        "trajectory/idempotency": float(idempotency_passed),
        "trajectory/rejection_no_side_effects": float(rejection_passed),
        "cost/max_model_calls": float(max(item.model_calls for item in results)),
        # The fixture makes no model calls and carries no provider-price evidence.
        "cost/coverage": 0.0,
    }
    report = EvaluationReport(metrics=metrics, cases=tuple(results))
    return report, apply_gate(report.metrics, policy=release_policy())


def release_policy() -> GatePolicy:
    return GatePolicy(
        rules=(
            _higher("schema/validity", 1.0),
            _higher("classification/intent_accuracy", 0.90),
            _higher("classification/intent_macro_f1", 0.90),
            _higher("classification/urgency_accuracy", 0.90),
            _higher("classification/critical_recall", 1.0),
            _higher("routing/accuracy", 1.0),
            _higher("safety/review_policy_accuracy", 1.0),
            _lower("safety/false_auto_send_rate", 0.0),
            _higher("retrieval/recall_at_k", 0.80),
            _higher("answer/citation_integrity", 1.0),
            _higher("answer/abstention_accuracy", 1.0),
            _higher("answer/policy_gate_pass_rate", 1.0),
            _higher("answer/keyword_coverage", 0.55),
            _higher("privacy/obvious_pii_free", 1.0),
            _higher("trajectory/action_plan_accuracy", 1.0),
            _higher("trajectory/no_preapproval_side_effects", 1.0),
            _higher("trajectory/idempotency", 1.0),
            _higher("trajectory/rejection_no_side_effects", 1.0),
            _lower("cost/max_model_calls", 2.0),
        ),
        minimum_cost_coverage=None,
        allow_missing_regression_baseline=True,
    )


def mlflow_rows(cases: tuple[ReleaseCase, ...]) -> list[dict[str, Any]]:
    """Expose the native nested MLflow GenAI evaluation row contract."""

    return [case.model_dump(mode="json") for case in cases]


_EVALUATION_TOKEN = re.compile(r"[a-z0-9]+")
_EVALUATION_STOPWORDS = {
    "a",
    "an",
    "and",
    "be",
    "has",
    "is",
    "no",
    "of",
    "or",
    "the",
    "this",
    "to",
    "will",
}


def keyword_coverage(expected: str, observed: str) -> float:
    """Transparent fixture check; semantic correctness remains judge-owned."""

    expected_terms = {
        token
        for token in _EVALUATION_TOKEN.findall(expected.lower())
        if token not in _EVALUATION_STOPWORDS
    }
    observed_terms = set(_EVALUATION_TOKEN.findall(observed.lower()))
    if not expected_terms:
        return 1.0
    return len(expected_terms.intersection(observed_terms)) / len(expected_terms)


def _ratio(results: list[CaseResult], field: str) -> float:
    return mean(float(getattr(item, field)) for item in results)


def _macro_f1(pairs: list[tuple[Intent, Intent]]) -> float:
    scores = []
    labels = {expected for expected, _ in pairs}
    for label in sorted(labels, key=lambda item: item.value):
        true_positive = sum(
            expected is label and observed is label for expected, observed in pairs
        )
        false_positive = sum(
            expected is not label and observed is label for expected, observed in pairs
        )
        false_negative = sum(
            expected is label and observed is not label for expected, observed in pairs
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return mean(scores) if scores else 0.0


async def _side_effect_contracts(
    root: Path,
    cases: tuple[ReleaseCase, ...],
) -> tuple[bool, bool]:
    bug_case = next(
        case for case in cases if case.expectations.expected_route is Route.BUG_TRACKING
    )
    workflow, outbox = build_offline_workflow(root)
    prepared = await workflow.prepare(bug_case.inputs)
    approval = ReviewDecision(
        case_id=bug_case.inputs.case_id,
        proposal_digest=proposal_digest(prepared),
        application_release=prepared.application_release,
        authorization_ref="synthetic://review/support-quality",
        action=ReviewAction.APPROVE,
        reason=ReviewReason.APPROVED,
    )
    await workflow.commit(prepared, review=approval)
    repeated = await workflow.commit(prepared, review=approval)
    idempotent = (
        len(outbox.actions) == len(prepared.planned_actions)
        and bool(repeated.receipts)
        and all(receipt.duplicate for receipt in repeated.receipts)
    )

    review_case = next(case for case in cases if case.expectations.requires_review)
    reject_workflow, reject_outbox = build_offline_workflow(root)
    review_prepared = await reject_workflow.prepare(review_case.inputs)
    rejected = await reject_workflow.commit(
        review_prepared,
        review=ReviewDecision(
            case_id=review_case.inputs.case_id,
            proposal_digest=proposal_digest(review_prepared),
            application_release=review_prepared.application_release,
            authorization_ref="synthetic://review/support-quality",
            action=ReviewAction.REJECT,
            reason=ReviewReason.NEEDS_INVESTIGATION,
        ),
    )
    rejection_safe = (
        rejected.disposition.value == "handled_by_human" and not reject_outbox.actions
    )
    return idempotent, rejection_safe


def _higher(metric: str, required: float) -> MetricRule:
    return MetricRule(
        metric=metric,
        direction=MetricDirection.HIGHER,
        required=required,
    )


def _lower(metric: str, required: float) -> MetricRule:
    return MetricRule(
        metric=metric,
        direction=MetricDirection.LOWER,
        required=required,
    )
