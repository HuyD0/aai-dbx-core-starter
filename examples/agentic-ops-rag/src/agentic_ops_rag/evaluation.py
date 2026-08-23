"""Deterministic retrieval metrics and aai-core release-gate policy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import Field, field_serializer, field_validator

from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.evaluation import (
    GateFailure,
    GatePolicy,
    GateResult,
    MetricDirection,
    MetricRule,
    apply_gate,
)
from agentic_ops_rag.contracts import EvaluationCase, RetrievalMode


class ComparisonRecord(ContractModel):
    """Strict evidence for one named baseline/change decision."""

    hypothesis: str = Field(min_length=1)
    baseline_configuration: str = Field(min_length=1)
    change_configuration: str = Field(min_length=1)
    baseline: Mapping[str, float]
    change: Mapping[str, float]
    result: Mapping[str, float]
    decision: Literal["adopt", "reject", "inconclusive"]
    failures: tuple[GateFailure, ...] = ()

    @field_validator("baseline", "change", "result", mode="after")
    @classmethod
    def freeze_metrics(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return freeze_value(value)

    @field_serializer("baseline", "change", "result")
    def serialize_metrics(self, value: Mapping[str, float]) -> dict[str, float]:
        return thaw_value(value)


def load_cases(path: str | Path) -> tuple[EvaluationCase, ...]:
    cases = tuple(
        EvaluationCase.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    identifiers = [case.case_id for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("case_id values must be unique")
    return cases


def benchmark_samples(
    pipeline,
    cases: tuple[EvaluationCase, ...],
    *,
    mode: RetrievalMode,
    semantic_rerank: bool = False,
) -> dict[str, tuple[float | None, ...]]:
    """Per-case scores in dataset order; ``None`` marks an out-of-scope case.

    Retrieval recall and MRR are ``None`` for cases that expect no documents:
    skipping the case is honest, scoring it 1.0 would reward not retrieving.
    Keeping every metric the same length as ``cases`` is what lets
    ``aai_core.agentkit.statistics.build_statistical_evidence`` pair rows
    against a baseline scored on the same ordered dataset.
    """

    samples: dict[str, list[float | None]] = {
        "retrieval/recall_at_3": [],
        "retrieval/mrr": [],
        "answer/abstention_accuracy": [],
        "answer/citation_integrity": [],
        "security/tenant_isolation": [],
        "security/region_isolation": [],
        "security/group_authorization": [],
        "security/current_evidence": [],
        "safety/action_approval": [],
        "latency/ms": [],
    }
    for case in cases:
        result = pipeline.invoke(
            case.question,
            tenant_id=case.tenant_id,
            region=case.region,
            allowed_groups=case.allowed_groups,
            mode=mode,
            semantic_rerank=semantic_rerank,
        )
        expected = set(case.expected_document_ids)
        retrieved = list(result.retrieved_document_ids)
        if expected:
            samples["retrieval/recall_at_3"].append(
                len(expected.intersection(retrieved)) / len(expected)
            )
            ranks = [
                retrieved.index(item) + 1 for item in expected if item in retrieved
            ]
            samples["retrieval/mrr"].append(1.0 / min(ranks) if ranks else 0.0)
        else:
            samples["retrieval/recall_at_3"].append(None)
            samples["retrieval/mrr"].append(None)
        samples["answer/abstention_accuracy"].append(
            float(result.abstained == (not case.answerable))
        )
        samples["answer/citation_integrity"].append(
            float(
                (not case.answerable and not result.citations)
                or (
                    case.answerable
                    and bool(result.citations)
                    and set(result.citations).issubset(set(retrieved))
                )
            )
        )
        samples["security/tenant_isolation"].append(
            float(
                len(result.retrieved_tenants) == len(retrieved)
                and all(tenant == case.tenant_id for tenant in result.retrieved_tenants)
            )
        )
        samples["security/region_isolation"].append(
            float(
                len(result.retrieved_regions) == len(retrieved)
                and all(region == case.region for region in result.retrieved_regions)
            )
        )
        requested_groups = set(case.allowed_groups)
        samples["security/group_authorization"].append(
            float(
                len(result.retrieved_allowed_groups) == len(retrieved)
                and all(
                    bool(requested_groups.intersection(document_groups))
                    for document_groups in result.retrieved_allowed_groups
                )
            )
        )
        samples["security/current_evidence"].append(
            float(
                len(result.retrieved_active) == len(retrieved)
                and all(result.retrieved_active)
            )
        )
        samples["safety/action_approval"].append(
            float(
                (not case.expects_action_proposal and not result.requires_approval)
                or (
                    case.expects_action_proposal
                    and result.requires_approval
                    and result.proposed_action is not None
                )
            )
        )
        samples["latency/ms"].append(result.latency_ms)
    return {metric: tuple(values) for metric, values in samples.items()}


def benchmark(
    pipeline,
    cases: tuple[EvaluationCase, ...],
    *,
    mode: RetrievalMode,
    semantic_rerank: bool = False,
) -> dict[str, float]:
    samples = benchmark_samples(
        pipeline, cases, mode=mode, semantic_rerank=semantic_rerank
    )
    latencies = _present(samples["latency/ms"])
    return {
        "retrieval/recall_at_3": _safe_mean(_present(samples["retrieval/recall_at_3"])),
        "retrieval/mrr": _safe_mean(_present(samples["retrieval/mrr"])),
        "answer/abstention_accuracy": mean(
            _present(samples["answer/abstention_accuracy"])
        ),
        "answer/citation_integrity": mean(
            _present(samples["answer/citation_integrity"])
        ),
        "security/tenant_isolation": mean(
            _present(samples["security/tenant_isolation"])
        ),
        "security/region_isolation": mean(
            _present(samples["security/region_isolation"])
        ),
        "security/group_authorization": mean(
            _present(samples["security/group_authorization"])
        ),
        "security/current_evidence": mean(
            _present(samples["security/current_evidence"])
        ),
        "safety/action_approval": mean(_present(samples["safety/action_approval"])),
        "latency/p95_ms": _percentile_95(latencies),
        "cost/coverage": 0.0,
    }


def release_gate(
    metrics: dict[str, float],
    *,
    baseline_metrics: dict[str, float] | None = None,
) -> GateResult:
    policy = GatePolicy(
        rules=(
            MetricRule(
                metric="retrieval/recall_at_3",
                direction=MetricDirection.HIGHER,
                required=0.8,
                max_regression=0.05,
            ),
            MetricRule(
                metric="retrieval/mrr",
                direction=MetricDirection.HIGHER,
                required=0.75,
                max_regression=0.05,
            ),
            MetricRule(
                metric="answer/abstention_accuracy",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
            MetricRule(
                metric="answer/citation_integrity",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
            MetricRule(
                metric="security/tenant_isolation",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
            MetricRule(
                metric="security/region_isolation",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
            MetricRule(
                metric="security/group_authorization",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
            MetricRule(
                metric="security/current_evidence",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
            MetricRule(
                metric="safety/action_approval",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
            MetricRule(
                metric="latency/p95_ms",
                direction=MetricDirection.LOWER,
                required=75.0,
                max_regression=10.0,
            ),
        ),
        # The offline fixture has no provider pricing and must not invent it.
        minimum_cost_coverage=None,
        allow_missing_regression_baseline=baseline_metrics is None,
    )
    return apply_gate(metrics, policy=policy, baseline_metrics=baseline_metrics)


def comparison_record(
    baseline: dict[str, float],
    change: dict[str, float],
    *,
    baseline_configuration: str = "baseline",
    change_configuration: str = "change",
) -> ComparisonRecord:
    report = release_gate(change, baseline_metrics=baseline)
    return ComparisonRecord(
        hypothesis=(
            f"Changing retrieval from {baseline_configuration} to "
            f"{change_configuration} improves retrieval quality without violating "
            "access, action-approval, or latency policy."
        ),
        baseline_configuration=baseline_configuration,
        change_configuration=change_configuration,
        baseline=baseline,
        change=change,
        result=report.metrics,
        decision="adopt" if report.passed else "reject",
        failures=report.failures,
    )


def is_release_eligible(
    selected_configuration: str,
    *,
    absolute_gate: GateResult,
    baseline_metrics: Mapping[str, float],
    comparison: ComparisonRecord,
    source_state: str,
) -> bool:
    """Bind release eligibility to the exact recorded comparison decision."""

    if comparison.__class__ is not ComparisonRecord:
        return False
    gate_metrics = dict(absolute_gate.metrics)
    trusted_baseline = dict(baseline_metrics)
    if dict(comparison.baseline) != trusted_baseline:
        return False
    comparison_gate = release_gate(
        dict(comparison.change),
        baseline_metrics=trusted_baseline,
    )
    return (
        source_state == "clean"
        and absolute_gate.passed
        and comparison.change_configuration == selected_configuration
        and dict(comparison.change) == gate_metrics
        and dict(comparison.result) == gate_metrics
        and dict(comparison_gate.metrics) == gate_metrics
        and comparison_gate.failures == comparison.failures
        and comparison_gate.passed
        and comparison.decision == "adopt"
        and not comparison.failures
    )


def _present(values: tuple[float | None, ...]) -> list[float]:
    return [value for value in values if value is not None]


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 1.0


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math_ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if integer == value else integer + 1
