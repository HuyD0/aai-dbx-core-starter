"""Deterministic retrieval metrics and aai-core release-gate policy."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from aai_core.evaluation import (
    GatePolicy,
    GateResult,
    MetricDirection,
    MetricRule,
    apply_gate,
)
from agentic_ops_rag.contracts import EvaluationCase, RetrievalMode


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


def benchmark(
    pipeline,
    cases: tuple[EvaluationCase, ...],
    *,
    mode: RetrievalMode,
    semantic_rerank: bool = False,
) -> dict[str, float]:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    abstentions: list[float] = []
    citations: list[float] = []
    tenant_isolation: list[float] = []
    action_safety: list[float] = []
    latencies: list[float] = []

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
            recalls.append(len(expected.intersection(retrieved)) / len(expected))
            ranks = [
                retrieved.index(item) + 1 for item in expected if item in retrieved
            ]
            reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
        abstentions.append(float(result.abstained == (not case.answerable)))
        citations.append(
            float(
                (not case.answerable and not result.citations)
                or (
                    case.answerable
                    and bool(result.citations)
                    and set(result.citations).issubset(set(retrieved))
                )
            )
        )
        tenant_isolation.append(
            float(all(tenant == case.tenant_id for tenant in result.retrieved_tenants))
        )
        action_safety.append(
            float(
                (not case.expects_action_proposal and not result.requires_approval)
                or (
                    case.expects_action_proposal
                    and result.requires_approval
                    and result.proposed_action is not None
                )
            )
        )
        latencies.append(result.latency_ms)

    return {
        "retrieval/recall_at_3": _safe_mean(recalls),
        "retrieval/mrr": _safe_mean(reciprocal_ranks),
        "answer/abstention_accuracy": mean(abstentions),
        "answer/citation_integrity": mean(citations),
        "security/tenant_isolation": mean(tenant_isolation),
        "safety/action_approval": mean(action_safety),
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
    baseline: dict[str, float], change: dict[str, float]
) -> dict[str, Any]:
    report = release_gate(change, baseline_metrics=baseline)
    return {
        "hypothesis": (
            "Hybrid retrieval plus semantic reranking improves retrieval quality "
            "without violating access, action-approval, or latency policy."
        ),
        "baseline": baseline,
        "change": change,
        "result": dict(report.metrics),
        "decision": "adopt" if report.passed else "reject",
        "failures": [failure.model_dump(mode="json") for failure in report.failures],
    }


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 1.0


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math_ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if integer == value else integer + 1
