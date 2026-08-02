"""Conservative LoRA promotion decisions over frozen evaluation reports."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from .models import EvaluationReport, StrictEvidenceModel


class PromotionDecision(StrEnum):
    ADOPT = "adopt"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


class BaselineEvaluation(StrictEvidenceModel):
    name: str = Field(min_length=1)
    report: EvaluationReport
    meaningful: bool = True


class PromotionThresholds(StrictEvidenceModel):
    """Versioned course gates, including a one-point minimum useful F1 gain.

    These defaults are conservative teaching choices, not universal production
    thresholds.  A real owner must derive thresholds and uncertainty policy from
    domain risk, sample size, and operational budgets before frozen evaluation.
    """

    minimum_schema_validity_rate: float = Field(default=0.98, ge=0.0, le=1.0)
    minimum_policy_compliance_rate: float = Field(default=0.95, ge=0.0, le=1.0)
    maximum_unsupported_intent_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_macro_f1_gain: float = Field(default=0.01, ge=0.0)


class EvaluationSnapshot(StrictEvidenceModel):
    name: str = Field(min_length=1)
    total_examples: int = Field(ge=1)
    evaluation_fingerprint: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    supported_intents: tuple[str, ...] = Field(min_length=1)
    macro_f1: float = Field(ge=0.0, le=1.0)
    schema_validity_rate: float = Field(ge=0.0, le=1.0)
    policy_compliance_rate: float = Field(ge=0.0, le=1.0)
    unsupported_intent_rate: float = Field(ge=0.0, le=1.0)


class ChangeEvidence(StrictEvidenceModel):
    name: str = Field(min_length=1)
    method: Literal["lora_fine_tune"] = "lora_fine_tune"
    training_manifest_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )


class PromotionResult(StrictEvidenceModel):
    evaluated_change: EvaluationSnapshot
    macro_f1_gain: float | None = None
    comparable: bool
    beats_strongest_meaningful_baseline: bool | None = None
    passes_schema_threshold: bool
    passes_policy_threshold: bool
    passes_unsupported_intent_threshold: bool
    reasons: tuple[str, ...]


class PromotionAssessment(StrictEvidenceModel):
    """Persisted baseline -> change -> result -> decision evidence."""

    baseline: EvaluationSnapshot | None
    change: ChangeEvidence
    result: PromotionResult
    decision: PromotionDecision


def decide_lora_promotion(
    *,
    change_name: str,
    training_manifest_sha256: str,
    change_report: EvaluationReport,
    baselines: tuple[BaselineEvaluation, ...] | list[BaselineEvaluation],
    thresholds: PromotionThresholds | None = None,
) -> PromotionAssessment:
    """Adopt only when a comparable LoRA result clears every required gate."""

    if not change_name.strip():
        raise ValueError("change_name must not be blank")
    if change_report.training_manifest_sha256 != training_manifest_sha256:
        raise ValueError(
            "change report must carry the supplied training manifest SHA-256"
        )
    policy = thresholds or PromotionThresholds()
    evaluated_change = _snapshot(change_name, change_report)
    schema_pass = (
        evaluated_change.schema_validity_rate >= policy.minimum_schema_validity_rate
    )
    policy_pass = (
        evaluated_change.policy_compliance_rate >= policy.minimum_policy_compliance_rate
    )
    unsupported_pass = (
        evaluated_change.unsupported_intent_rate
        <= policy.maximum_unsupported_intent_rate
    )
    meaningful = [baseline for baseline in baselines if baseline.meaningful]
    if not meaningful:
        reasons = ["no meaningful baseline evaluation was supplied"]
        if not schema_pass:
            reasons.append("change missed the absolute schema-validity threshold")
        if not policy_pass:
            reasons.append("change missed the absolute response-policy threshold")
        if not unsupported_pass:
            reasons.append("change exceeded the unsupported-intent threshold")
        return PromotionAssessment(
            baseline=None,
            change=ChangeEvidence(
                name=change_name,
                training_manifest_sha256=training_manifest_sha256,
            ),
            result=PromotionResult(
                evaluated_change=evaluated_change,
                comparable=False,
                passes_schema_threshold=schema_pass,
                passes_policy_threshold=policy_pass,
                passes_unsupported_intent_threshold=unsupported_pass,
                reasons=tuple(reasons),
            ),
            decision=PromotionDecision.INCONCLUSIVE,
        )

    strongest = min(
        meaningful,
        key=lambda baseline: (
            -baseline.report.classification.macro_f1,
            baseline.name,
        ),
    )
    baseline_snapshot = _snapshot(strongest.name, strongest.report)
    comparable = (
        baseline_snapshot.total_examples == evaluated_change.total_examples
        and baseline_snapshot.supported_intents == evaluated_change.supported_intents
        and baseline_snapshot.evaluation_fingerprint
        == evaluated_change.evaluation_fingerprint
    )
    if not comparable:
        return PromotionAssessment(
            baseline=baseline_snapshot,
            change=ChangeEvidence(
                name=change_name,
                training_manifest_sha256=training_manifest_sha256,
            ),
            result=PromotionResult(
                evaluated_change=evaluated_change,
                comparable=False,
                passes_schema_threshold=schema_pass,
                passes_policy_threshold=policy_pass,
                passes_unsupported_intent_threshold=unsupported_pass,
                reasons=(
                    "change and strongest meaningful baseline were not scored "
                    "on comparable evaluation sets",
                ),
            ),
            decision=PromotionDecision.INCONCLUSIVE,
        )

    gain = evaluated_change.macro_f1 - baseline_snapshot.macro_f1
    beats_baseline = gain > 0.0 and gain >= policy.minimum_macro_f1_gain
    reasons: list[str] = []
    if not beats_baseline:
        reasons.append(
            "change did not beat the strongest meaningful baseline by at least "
            f"{policy.minimum_macro_f1_gain:g} macro-F1"
        )
    if not schema_pass:
        reasons.append(
            f"schema validity {evaluated_change.schema_validity_rate:.3f} is below "
            f"{policy.minimum_schema_validity_rate:.3f}"
        )
    if not policy_pass:
        reasons.append(
            "response-policy compliance "
            f"{evaluated_change.policy_compliance_rate:.3f} "
            f"is below {policy.minimum_policy_compliance_rate:.3f}"
        )
    if not unsupported_pass:
        reasons.append(
            "unsupported-intent rate "
            f"{evaluated_change.unsupported_intent_rate:.3f} is above "
            f"{policy.maximum_unsupported_intent_rate:.3f}"
        )
    if not reasons:
        reasons.append(
            "change beat the strongest meaningful baseline and passed all "
            "absolute output gates"
        )
    return PromotionAssessment(
        baseline=baseline_snapshot,
        change=ChangeEvidence(
            name=change_name,
            training_manifest_sha256=training_manifest_sha256,
        ),
        result=PromotionResult(
            evaluated_change=evaluated_change,
            macro_f1_gain=gain,
            comparable=True,
            beats_strongest_meaningful_baseline=beats_baseline,
            passes_schema_threshold=schema_pass,
            passes_policy_threshold=policy_pass,
            passes_unsupported_intent_threshold=unsupported_pass,
            reasons=tuple(reasons),
        ),
        decision=(
            PromotionDecision.ADOPT
            if beats_baseline and schema_pass and policy_pass and unsupported_pass
            else PromotionDecision.REJECT
        ),
    )


def _snapshot(name: str, report: EvaluationReport) -> EvaluationSnapshot:
    return EvaluationSnapshot(
        name=name,
        total_examples=report.total_examples,
        evaluation_fingerprint=report.evaluation_fingerprint,
        supported_intents=report.supported_intents,
        macro_f1=report.classification.macro_f1,
        schema_validity_rate=report.output_quality.json_schema_validity_rate,
        policy_compliance_rate=(report.output_quality.response_policy_compliance_rate),
        unsupported_intent_rate=report.output_quality.unsupported_intent_rate,
    )
