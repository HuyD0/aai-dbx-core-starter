"""Deterministic evaluation for structured customer-support predictions."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, median
from typing import Any

from pydantic import ValidationError

from .models import (
    ClassificationMetrics,
    DistributionSummary,
    ErrorAnalysis,
    ErrorExample,
    ErrorKind,
    EvaluationRecord,
    EvaluationReport,
    InferenceConfig,
    LocalMLXInferenceConfig,
    OutputQualityMetrics,
    PerformanceMetrics,
    Prediction,
    SliceMetrics,
    SupportOutput,
)
from .policy import ResponsePolicy
from .session import (
    EvaluationSession,
    recheck_evaluation_session,
)


@dataclass(frozen=True)
class _ScoredPrediction:
    record: EvaluationRecord
    prediction: Prediction
    parsed: bool
    schema_valid: bool
    unsupported_intent: bool
    output: SupportOutput | None
    policy_compliant: bool
    policy_issues: tuple[str, ...]
    error_kinds: tuple[ErrorKind, ...]


class Evaluator:
    """Score frozen records without model, MLX, network, or tracker imports."""

    def __init__(
        self,
        *,
        supported_intents: Sequence[str] | None = None,
        response_policy: ResponsePolicy | None = None,
        error_limit: int = 20,
    ) -> None:
        if error_limit < 0:
            raise ValueError("error_limit must be non-negative")
        if supported_intents is None:
            self._supported_intents = None
        else:
            normalized = tuple(sorted(supported_intents))
            if not normalized or any(not value.strip() for value in normalized):
                raise ValueError("supported_intents must contain non-blank values")
            if len(set(normalized)) != len(normalized):
                raise ValueError("supported_intents must not contain duplicates")
            self._supported_intents = normalized
        self.response_policy = response_policy or ResponsePolicy()
        self.error_limit = error_limit

    def evaluate(
        self,
        records: Sequence[EvaluationRecord],
        predictions: Sequence[Prediction],
        *,
        evaluation_session: EvaluationSession,
        inference_config: InferenceConfig,
    ) -> EvaluationReport:
        recheck_evaluation_session(evaluation_session)
        _require_session_model_contract(evaluation_session, inference_config)
        if not records:
            raise ValueError("records must not be empty")
        ordered_predictions = _align_predictions(records, predictions)
        target_intents = {record.target.intent for record in records}
        supported_intents = self._supported_intents or tuple(sorted(target_intents))
        missing = target_intents.difference(supported_intents)
        if missing:
            raise ValueError(
                "supported_intents omits target labels: " + ", ".join(sorted(missing))
            )

        scored = tuple(
            self._score(record, prediction, supported_intents)
            for record, prediction in zip(records, ordered_predictions, strict=True)
        )
        classification = _classification_metrics(scored, supported_intents)
        output_quality = _output_quality_metrics(scored)
        recheck_evaluation_session(evaluation_session)
        _require_session_model_contract(evaluation_session, inference_config)
        return EvaluationReport(
            total_examples=len(scored),
            inference_config=inference_config,
            evaluation_fingerprint=_evaluation_fingerprint(records),
            evaluation_execution_contract_sha256=(
                evaluation_session.execution_contract_sha256
            ),
            supported_intents=supported_intents,
            classification=classification,
            output_quality=output_quality,
            performance=PerformanceMetrics(
                latency_ms=_summarize([item.prediction.latency_ms for item in scored]),
                output_tokens=_summarize(
                    [float(item.prediction.output_tokens) for item in scored]
                ),
                peak_memory_mb=_summarize(
                    [item.prediction.peak_memory_mb for item in scored]
                ),
            ),
            by_intent=_group_slices(scored, lambda item: (item.record.target.intent,)),
            by_flag=_group_slices(
                scored,
                lambda item: item.record.flags or ("__none__",),
            ),
            by_difficulty=_group_slices(scored, lambda item: (item.record.difficulty,)),
            error_analysis=_error_analysis(scored, limit=self.error_limit),
        )

    def _score(
        self,
        record: EvaluationRecord,
        prediction: Prediction,
        supported_intents: tuple[str, ...],
    ) -> _ScoredPrediction:
        parsed_value: Any = None
        parsed = False
        try:
            parsed_value = json.loads(prediction.raw_text)
            parsed = True
        except (json.JSONDecodeError, TypeError):
            pass

        output = None
        if parsed:
            try:
                output = SupportOutput.model_validate(parsed_value, strict=True)
            except ValidationError:
                pass
        schema_valid = output is not None
        raw_intent = (
            parsed_value.get("intent") if isinstance(parsed_value, dict) else None
        )
        unsupported = isinstance(raw_intent, str) and raw_intent not in set(
            supported_intents
        )
        if output is None:
            policy_compliant = False
            policy_issues: tuple[str, ...] = ()
        else:
            policy_result = self.response_policy.check(output)
            policy_compliant = policy_result.compliant
            policy_issues = policy_result.issues

        kinds: list[ErrorKind] = []
        if not parsed:
            kinds.append(ErrorKind.JSON_PARSE)
        elif not schema_valid:
            kinds.append(ErrorKind.JSON_SCHEMA)
        if unsupported:
            kinds.append(ErrorKind.UNSUPPORTED_INTENT)
        if output is not None:
            if output.intent != record.target.intent:
                kinds.append(ErrorKind.INTENT)
            if output.category != record.target.category:
                kinds.append(ErrorKind.CATEGORY)
            if output.requires_escalation != record.target.requires_escalation:
                kinds.append(ErrorKind.ESCALATION)
            if not policy_compliant:
                kinds.append(ErrorKind.RESPONSE_POLICY)
        return _ScoredPrediction(
            record=record,
            prediction=prediction,
            parsed=parsed,
            schema_valid=schema_valid,
            unsupported_intent=unsupported,
            output=output,
            policy_compliant=policy_compliant,
            policy_issues=policy_issues,
            error_kinds=tuple(kinds),
        )


def evaluate_predictions(
    records: Sequence[EvaluationRecord],
    predictions: Sequence[Prediction],
    *,
    evaluation_session: EvaluationSession,
    inference_config: InferenceConfig,
    supported_intents: Sequence[str] | None = None,
    response_policy: ResponsePolicy | None = None,
    error_limit: int = 20,
) -> EvaluationReport:
    """Score predictions within an explicitly prestarted inference session."""

    return Evaluator(
        supported_intents=supported_intents,
        response_policy=response_policy,
        error_limit=error_limit,
    ).evaluate(
        records,
        predictions,
        evaluation_session=evaluation_session,
        inference_config=inference_config,
    )


def _require_session_model_contract(
    evaluation_session: EvaluationSession,
    inference_config: InferenceConfig,
) -> None:
    if not isinstance(inference_config, LocalMLXInferenceConfig):
        return
    captured = evaluation_session.base_model_execution_contract
    if captured is None:
        raise ValueError(
            "local MLX inference evidence requires a model-aware evaluation session"
        )
    if inference_config.base_model != captured:
        raise ValueError(
            "inference configuration does not match the evaluation session's "
            "base-model contract"
        )


def format_error_analysis(report: EvaluationReport) -> str:
    """Render compact, bounded error evidence for a terminal or notebook."""

    analysis = report.error_analysis
    nonzero = [
        f"{kind.value}={analysis.counts.get(kind, 0)}"
        for kind in ErrorKind
        if analysis.counts.get(kind, 0)
    ]
    summary = f"{analysis.total_errors}/{report.total_examples} examples with errors"
    if nonzero:
        summary += " (" + ", ".join(nonzero) + ")"
    lines = [summary]
    for example in analysis.examples:
        kinds = ",".join(kind.value for kind in example.kinds)
        predicted = example.predicted_intent or "<invalid>"
        lines.append(
            f"- {example.example_id}: {kinds}; "
            f"intent={example.expected_intent}->{predicted}"
        )
    if analysis.truncated:
        lines.append("- additional examples omitted")
    return "\n".join(lines)


def _align_predictions(
    records: Sequence[EvaluationRecord], predictions: Sequence[Prediction]
) -> tuple[Prediction, ...]:
    record_ids = [record.example_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("records contains duplicate example_id values")
    prediction_by_id: dict[str, Prediction] = {}
    for prediction in predictions:
        if prediction.example_id in prediction_by_id:
            raise ValueError(
                f"predictions contains duplicate example_id {prediction.example_id!r}"
            )
        prediction_by_id[prediction.example_id] = prediction
    missing = set(record_ids).difference(prediction_by_id)
    extra = set(prediction_by_id).difference(record_ids)
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing=" + ",".join(sorted(missing)))
        if extra:
            parts.append("extra=" + ",".join(sorted(extra)))
        raise ValueError(
            "prediction identifiers do not match records: " + "; ".join(parts)
        )
    return tuple(prediction_by_id[example_id] for example_id in record_ids)


def _evaluation_fingerprint(records: Sequence[EvaluationRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda value: value.example_id):
        payload = {
            "example_id": record.example_id,
            "input_text": record.input_text,
            "target": record.target.model_dump(mode="json"),
            "flags": record.flags,
            "difficulty": record.difficulty,
        }
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _classification_metrics(
    scored: Sequence[_ScoredPrediction], supported_intents: tuple[str, ...]
) -> ClassificationMetrics:
    targets = [item.record.target.intent for item in scored]
    predictions = [
        item.output.intent if item.output is not None else None for item in scored
    ]
    per_intent_f1: dict[str, float] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    weighted_total = 0.0
    for intent in supported_intents:
        true_positive = sum(
            target == intent and predicted == intent
            for target, predicted in zip(targets, predictions, strict=True)
        )
        false_positive = sum(
            target != intent and predicted == intent
            for target, predicted in zip(targets, predictions, strict=True)
        )
        false_negative = sum(
            target == intent and predicted != intent
            for target, predicted in zip(targets, predictions, strict=True)
        )
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _f1(precision, recall)
        support = targets.count(intent)
        per_intent_f1[intent] = f1
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
        weighted_total += f1 * support

    total = len(scored)
    return ClassificationMetrics(
        intent_accuracy=_safe_ratio(
            sum(
                target == prediction
                for target, prediction in zip(targets, predictions, strict=True)
            ),
            total,
        ),
        macro_precision=fmean(precisions),
        macro_recall=fmean(recalls),
        macro_f1=fmean(f1_values),
        weighted_f1=weighted_total / total,
        per_intent_f1=per_intent_f1,
        category_accuracy=_safe_ratio(
            sum(
                item.output is not None
                and item.output.category == item.record.target.category
                for item in scored
            ),
            total,
        ),
        escalation_accuracy=_safe_ratio(
            sum(
                item.output is not None
                and item.output.requires_escalation
                == item.record.target.requires_escalation
                for item in scored
            ),
            total,
        ),
    )


def _output_quality_metrics(
    scored: Sequence[_ScoredPrediction],
) -> OutputQualityMetrics:
    total = len(scored)
    return OutputQualityMetrics(
        json_parse_rate=_safe_ratio(sum(item.parsed for item in scored), total),
        json_schema_validity_rate=_safe_ratio(
            sum(item.schema_valid for item in scored), total
        ),
        unsupported_intent_rate=_safe_ratio(
            sum(item.unsupported_intent for item in scored), total
        ),
        response_policy_compliance_rate=_safe_ratio(
            sum(item.policy_compliant for item in scored), total
        ),
    )


def _group_slices(scored, keys_for) -> dict[str, SliceMetrics]:
    grouped: dict[str, list[_ScoredPrediction]] = defaultdict(list)
    for item in scored:
        for key in keys_for(item):
            grouped[key].append(item)
    return {key: _slice_metrics(grouped[key]) for key in sorted(grouped)}


def _slice_metrics(scored: Sequence[_ScoredPrediction]) -> SliceMetrics:
    total = len(scored)
    return SliceMetrics(
        count=total,
        intent_accuracy=_safe_ratio(
            sum(
                item.output is not None
                and item.output.intent == item.record.target.intent
                for item in scored
            ),
            total,
        ),
        category_accuracy=_safe_ratio(
            sum(
                item.output is not None
                and item.output.category == item.record.target.category
                for item in scored
            ),
            total,
        ),
        escalation_accuracy=_safe_ratio(
            sum(
                item.output is not None
                and item.output.requires_escalation
                == item.record.target.requires_escalation
                for item in scored
            ),
            total,
        ),
        json_parse_rate=_safe_ratio(sum(item.parsed for item in scored), total),
        json_schema_validity_rate=_safe_ratio(
            sum(item.schema_valid for item in scored), total
        ),
        unsupported_intent_rate=_safe_ratio(
            sum(item.unsupported_intent for item in scored), total
        ),
        response_policy_compliance_rate=_safe_ratio(
            sum(item.policy_compliant for item in scored), total
        ),
    )


def _summarize(values: Sequence[float]) -> DistributionSummary:
    if not values:
        raise ValueError("cannot summarize an empty collection")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("performance evidence must contain finite values")
    ordered = sorted(values)
    return DistributionSummary(
        count=len(values),
        minimum=float(ordered[0]),
        mean=float(fmean(ordered)),
        median=float(median(ordered)),
        p95=float(_percentile(ordered, 0.95)),
        maximum=float(ordered[-1]),
    )


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _error_analysis(
    scored: Sequence[_ScoredPrediction], *, limit: int
) -> ErrorAnalysis:
    failures = [item for item in scored if item.error_kinds]
    counts: Counter[ErrorKind] = Counter(
        kind for item in failures for kind in item.error_kinds
    )
    examples = tuple(
        ErrorExample(
            example_id=item.record.example_id,
            kinds=item.error_kinds,
            expected_intent=item.record.target.intent,
            predicted_intent=(item.output.intent if item.output is not None else None),
            input_preview=_preview(item.record.input_text, 120),
            output_preview=_preview(item.prediction.raw_text, 160),
            policy_issues=item.policy_issues,
        )
        for item in failures[:limit]
    )
    return ErrorAnalysis(
        total_errors=len(failures),
        counts={kind: counts[kind] for kind in ErrorKind},
        examples=examples,
        truncated=len(failures) > limit,
    )


def _preview(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
