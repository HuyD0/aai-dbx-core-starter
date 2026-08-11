"""Typed, deterministic support mechanics for Foundry curriculum labs 00-07.

The objects in this module are teaching fakes, not measurements of a Foundry
deployment.  Every synthetic quality, latency, and cost record carries an
explicit ``simulated_offline_fixture`` source label.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from statistics import mean
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

EVIDENCE_SOURCE = "simulated_offline_fixture"
Decision = Literal["adopt", "reject", "inconclusive"]
EvidenceStatus = Literal["pass", "fail", "missing"]

# Lesson 00: impact and risk treatment evidence.


@dataclass(frozen=True)
class RiskTreatment:
    """One owned, verifiable treatment for a named material risk."""

    owner: str
    control: str
    verification: str


def assess_risk_treatments(
    risks: Iterable[str],
    treatments: Mapping[str, RiskTreatment],
) -> dict[str, object]:
    """Return complete, missing, incomplete, and unscoped treatment evidence."""

    scoped_risks = tuple(dict.fromkeys(risk.strip() for risk in risks if risk.strip()))
    missing = tuple(risk for risk in scoped_risks if risk not in treatments)
    incomplete = tuple(
        risk
        for risk in scoped_risks
        if risk in treatments
        and not all(
            (
                treatments[risk].owner.strip(),
                treatments[risk].control.strip(),
                treatments[risk].verification.strip(),
            )
        )
    )
    unscoped = tuple(sorted(set(treatments) - set(scoped_risks)))
    return {
        "complete": not missing and not incomplete,
        "missing": missing,
        "incomplete": incomplete,
        "unscoped": unscoped,
    }


# Lesson 01: same-case model comparison through a replaceable provider boundary.


@dataclass(frozen=True)
class ModelCase:
    """A fixed model-selection case and its deterministic success terms."""

    case_id: str
    expected_terms: tuple[str, ...]


@dataclass(frozen=True)
class ModelObservation:
    """One response plus simulated operational measurements."""

    output: str
    latency_ms: int
    cost_units: float
    evidence_source: str = EVIDENCE_SOURCE


class TextResponseBoundary(Protocol):
    """Small boundary implemented by offline fakes or a reviewed live adapter."""

    def create_text(self, deployment: str, case_id: str) -> ModelObservation:
        """Return one observation for a deployment and fixed case."""


class FakeTextResponseBoundary:
    """Deterministic in-memory replacement for a Responses API call."""

    def __init__(
        self,
        observations: Mapping[str, Mapping[str, ModelObservation]],
    ) -> None:
        self._observations = {
            deployment: dict(by_case) for deployment, by_case in observations.items()
        }

    def create_text(self, deployment: str, case_id: str) -> ModelObservation:
        return self._observations[deployment][case_id]


@dataclass(frozen=True)
class ModelThresholds:
    """Predeclared model-selection thresholds."""

    minimum_quality: float
    maximum_p95_latency_ms: int
    maximum_average_cost_units: float
    minimum_cases: int


@dataclass(frozen=True)
class ModelResult:
    """Same-case result for one deployment."""

    deployment: str
    quality: float
    p95_latency_ms: int
    average_cost_units: float
    case_ids: tuple[str, ...]
    evidence_source: str = EVIDENCE_SOURCE


@dataclass(frozen=True)
class ModelDecision:
    """Fail-closed model-selection decision."""

    decision: Decision
    selected_deployment: str | None
    reason: str


def compare_models(
    provider: TextResponseBoundary,
    deployments: Sequence[str],
    cases: Sequence[ModelCase],
) -> tuple[ModelResult, ...]:
    """Evaluate every deployment against the same ordered cases."""

    results = []
    for deployment in deployments:
        observations = [
            provider.create_text(deployment, case.case_id) for case in cases
        ]
        passes = [
            all(
                term.lower() in observation.output.lower()
                for term in case.expected_terms
            )
            for case, observation in zip(cases, observations, strict=True)
        ]
        ordered_latency = sorted(item.latency_ms for item in observations)
        percentile_index = max(0, math.ceil(0.95 * len(ordered_latency)) - 1)
        results.append(
            ModelResult(
                deployment=deployment,
                quality=mean(passes),
                p95_latency_ms=ordered_latency[percentile_index],
                average_cost_units=mean(item.cost_units for item in observations),
                case_ids=tuple(case.case_id for case in cases),
            )
        )
    return tuple(results)


def select_model(
    results: Sequence[ModelResult],
    thresholds: ModelThresholds,
) -> ModelDecision:
    """Select the least-cost eligible model, or fail closed."""

    if not results:
        return ModelDecision("inconclusive", None, "results are absent or misaligned")
    case_ids = results[0].case_ids
    if any(result.case_ids != case_ids for result in results[1:]):
        return ModelDecision("inconclusive", None, "results are absent or misaligned")
    if len(case_ids) < thresholds.minimum_cases:
        return ModelDecision("inconclusive", None, "too few representative cases")
    eligible = [
        result
        for result in results
        if result.quality >= thresholds.minimum_quality
        and result.p95_latency_ms <= thresholds.maximum_p95_latency_ms
        and result.average_cost_units <= thresholds.maximum_average_cost_units
    ]
    if not eligible:
        return ModelDecision("reject", None, "no deployment meets every threshold")
    selected = min(
        eligible,
        key=lambda item: (
            item.average_cost_units,
            item.p95_latency_ms,
            item.deployment,
        ),
    )
    return ModelDecision("adopt", selected.deployment, "all thresholds passed")


SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Lesson 02: untrusted structured-output validation.


@dataclass(frozen=True)
class StructuredCaseResult:
    """Safe validation evidence that never echoes an untrusted candidate."""

    case_id: str
    accepted: bool
    issue_codes: tuple[str, ...]


def exercise_structured_outputs(
    schema: type[SchemaT],
    candidates: Mapping[str, Mapping[str, object]],
) -> tuple[StructuredCaseResult, ...]:
    """Validate structured-output fixtures and retain only safe error codes."""

    results = []
    for case_id, candidate in candidates.items():
        try:
            schema.model_validate(candidate)
        except ValidationError as error:
            codes = tuple(sorted({str(item["type"]) for item in error.errors()}))
            results.append(StructuredCaseResult(case_id, False, codes))
        else:
            results.append(StructuredCaseResult(case_id, True, ()))
    return tuple(results)


# Lesson 03: authorization-first hybrid retrieval and citation validation.


@dataclass(frozen=True)
class RetrievalDocument:
    """One provenance-bearing document visible to the retrieval boundary."""

    chunk_id: str
    source_uri: str
    allowed_groups: frozenset[str]
    content: str
    semantic_score: float
    is_current: bool = True
    is_deleted: bool = False


@dataclass(frozen=True)
class RetrievedChunk:
    """Authorized retrieval result with separate relevance signals."""

    document: RetrievalDocument
    lexical_score: float
    semantic_score: float
    hybrid_score: float
    instruction_suspected: bool
    evidence_source: str = EVIDENCE_SOURCE


_TOKEN = re.compile(r"[a-z0-9]+")
_INSTRUCTION_MARKERS = (
    "ignore prior instructions",
    "reveal credentials",
    "call every tool",
)


def _tokens(value: str) -> set[str]:
    return set(_TOKEN.findall(value.lower()))


def hybrid_retrieve(
    query: str,
    documents: Sequence[RetrievalDocument],
    caller_groups: frozenset[str],
    *,
    limit: int = 3,
) -> tuple[RetrievedChunk, ...]:
    """Authorize and freshness-filter before deterministic hybrid ranking."""

    if limit < 1:
        raise ValueError("limit must be positive")
    query_terms = _tokens(query)
    candidates = []
    for document in documents:
        if (
            not document.allowed_groups.intersection(caller_groups)
            or not document.is_current
            or document.is_deleted
        ):
            continue
        lexical = len(query_terms.intersection(_tokens(document.content))) / max(
            1, len(query_terms)
        )
        lowered = document.content.lower()
        candidates.append(
            RetrievedChunk(
                document=document,
                lexical_score=round(lexical, 4),
                semantic_score=document.semantic_score,
                hybrid_score=round((lexical + document.semantic_score) / 2, 4),
                instruction_suspected=any(
                    marker in lowered for marker in _INSTRUCTION_MARKERS
                ),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (-item.hybrid_score, item.document.chunk_id),
        )[:limit]
    )


def build_grounded_answer(chunks: Sequence[RetrievedChunk]) -> dict[str, object]:
    """Treat retrieved text as data and cite every retained claim."""

    safe_chunks = [chunk for chunk in chunks if not chunk.instruction_suspected]
    claims = tuple(
        {
            "text": chunk.document.content,
            "citation": chunk.document.source_uri,
        }
        for chunk in safe_chunks
    )
    return {
        "claims": claims,
        "excluded_chunks": tuple(
            chunk.document.chunk_id for chunk in chunks if chunk.instruction_suspected
        ),
        "tool_calls": (),
        "policy_changed": False,
    }


def citations_resolve(
    answer: Mapping[str, object],
    chunks: Sequence[RetrievedChunk],
) -> bool:
    """Require each claim to cite a URI present in authorized retrieval."""

    available = {chunk.document.source_uri for chunk in chunks}
    claims = answer.get("claims")
    if not isinstance(claims, (tuple, list)) or not claims:
        return False
    return all(
        isinstance(claim, Mapping)
        and isinstance(claim.get("citation"), str)
        and claim["citation"] in available
        for claim in claims
    )


# Lesson 04: application-owned tool authorization, retries, and idempotency.


@dataclass(frozen=True)
class ToolPolicy:
    """Authorization and retry policy owned by the application."""

    allowed_groups: frozenset[str]
    side_effect: bool
    max_attempts: int

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")


class TransientToolError(RuntimeError):
    """A retryable error raised before the fake backend commits a side effect."""


class ToolBackend(Protocol):
    """Narrow execution boundary for approved JSON-shaped tool arguments."""

    def call(
        self,
        name: str,
        arguments: Mapping[str, object],
        idempotency_key: str,
    ) -> Mapping[str, object]:
        """Execute one authorized attempt."""


class SimulatedToolBackend:
    """Deterministic backend with configurable pre-commit transient failures."""

    def __init__(
        self, failures_before_success: Mapping[str, int] | None = None
    ) -> None:
        self.failures_remaining = dict(failures_before_success or {})
        self.call_counts: Counter[str] = Counter()
        self.committed_keys: set[str] = set()

    def call(
        self,
        name: str,
        arguments: Mapping[str, object],
        idempotency_key: str,
    ) -> Mapping[str, object]:
        self.call_counts[name] += 1
        if self.failures_remaining.get(name, 0) > 0:
            self.failures_remaining[name] -= 1
            raise TransientToolError("simulated transient dependency failure")
        if name == "publish_release":
            self.committed_keys.add(idempotency_key)
            return {"published": str(arguments["version"])}
        return {"matches": ("foundry-curriculum",)}


@dataclass(frozen=True)
class ToolOutcome:
    """Sanitized result of authorization, retry, execution, or replay."""

    status: Literal["succeeded", "blocked", "failed"]
    reason: str
    attempts: int
    replayed: bool
    value: Mapping[str, object] | None = None


class ToolGateway:
    """Authorize independently, bound retries, and replay successful requests."""

    def __init__(
        self,
        policies: Mapping[str, ToolPolicy],
        backend: ToolBackend,
    ) -> None:
        self._policies = dict(policies)
        self._backend = backend
        self._cache: dict[str, tuple[str, ToolOutcome]] = {}

    def execute(
        self,
        name: str,
        arguments: Mapping[str, object],
        *,
        caller_groups: frozenset[str],
        approved: bool,
        idempotency_key: str,
    ) -> ToolOutcome:
        policy = self._policies[name]
        if not policy.allowed_groups.intersection(caller_groups):
            return ToolOutcome("blocked", "caller is not authorized", 0, False)
        if policy.side_effect and not approved:
            return ToolOutcome("blocked", "human approval is required", 0, False)
        if not idempotency_key.strip():
            return ToolOutcome("blocked", "idempotency key is required", 0, False)
        fingerprint = json.dumps(
            [name, arguments], sort_keys=True, separators=(",", ":")
        )
        cached = self._cache.get(idempotency_key)
        if cached and cached[0] != fingerprint:
            return ToolOutcome("blocked", "idempotency key conflict", 0, False)
        if cached:
            return replace(cached[1], replayed=True)
        for attempt in range(1, policy.max_attempts + 1):
            try:
                value = self._backend.call(name, arguments, idempotency_key)
            except TransientToolError:
                continue
            else:
                outcome = ToolOutcome("succeeded", "executed", attempt, False, value)
                self._cache[idempotency_key] = (fingerprint, outcome)
                return outcome
        return ToolOutcome(
            "failed", "retry budget exhausted", policy.max_attempts, False
        )


# Lesson 05: fixed-data baseline/change evaluation and judge calibration.


class EvaluationBoundary(Protocol):
    """Minimal response-quality boundary used by the deterministic evaluator."""

    def passes(self, version: str, case_id: str) -> bool:
        """Return the reviewed pass/fail label for one versioned case."""


class FakeEvaluationProvider:
    """Deterministic baseline/change boundary over one frozen case set."""

    def __init__(self, failures_by_version: Mapping[str, Iterable[str]]) -> None:
        self._failures = {
            version: frozenset(case_ids)
            for version, case_ids in failures_by_version.items()
        }

    def passes(self, version: str, case_id: str) -> bool:
        return case_id not in self._failures[version]


@dataclass(frozen=True)
class EvaluationSummary:
    """Deterministic quality and safety evidence for one immutable version."""

    version: str
    dataset_digest: str
    task_pass_rate: float
    critical_safety_pass_rate: float
    slice_pass_rates: Mapping[str, float]
    failed_case_ids: tuple[str, ...]
    evidence_source: str = EVIDENCE_SOURCE


@dataclass(frozen=True)
class JudgeCalibration:
    """Agreement evidence against a human-labelled calibration split."""

    agreement: float
    false_positive_case_ids: tuple[str, ...]
    false_negative_case_ids: tuple[str, ...]
    evidence_source: str = EVIDENCE_SOURCE


@dataclass(frozen=True)
class EvaluationThresholds:
    """Predeclared release thresholds for the offline evaluation."""

    task_success_minimum: float
    critical_safety_pass_rate: float
    judge_agreement_minimum: float


@dataclass(frozen=True)
class EvaluationDecision:
    """Baseline/change decision with an explicit reason."""

    decision: Decision
    reason: str


def dataset_digest(cases: Sequence[Mapping[str, Any]]) -> str:
    """Digest the ordered case payload used by both experiment arms."""

    payload = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def evaluate_version(
    provider: EvaluationBoundary,
    version: str,
    cases: Sequence[Mapping[str, Any]],
) -> EvaluationSummary:
    """Score one version and retain failures plus category slices."""

    outcomes = {
        str(case["case_id"]): provider.passes(version, str(case["case_id"]))
        for case in cases
    }
    critical_ids = [
        str(case["case_id"])
        for case in cases
        if case["expectations"]["risk"] == "critical"
    ]
    categories = sorted({str(case["category"]) for case in cases})
    slice_rates = {}
    for category in categories:
        ids = [str(case["case_id"]) for case in cases if case["category"] == category]
        slice_rates[category] = mean(outcomes[case_id] for case_id in ids)
    return EvaluationSummary(
        version=version,
        dataset_digest=dataset_digest(cases),
        task_pass_rate=mean(outcomes.values()),
        critical_safety_pass_rate=mean(outcomes[case_id] for case_id in critical_ids),
        slice_pass_rates=slice_rates,
        failed_case_ids=tuple(
            case_id for case_id, passed in outcomes.items() if not passed
        ),
    )


def calibrate_binary_judge(
    human_labels: Mapping[str, bool],
    judge_labels: Mapping[str, bool],
) -> JudgeCalibration:
    """Compare an automated binary judge with the same human-labelled rows."""

    if not human_labels or set(human_labels) != set(judge_labels):
        raise ValueError("human and judge labels must cover the same non-empty cases")
    case_ids = tuple(human_labels)
    matches = [human_labels[case_id] == judge_labels[case_id] for case_id in case_ids]
    false_positives = tuple(
        case_id
        for case_id in case_ids
        if judge_labels[case_id] and not human_labels[case_id]
    )
    false_negatives = tuple(
        case_id
        for case_id in case_ids
        if human_labels[case_id] and not judge_labels[case_id]
    )
    return JudgeCalibration(mean(matches), false_positives, false_negatives)


def decide_evaluation(
    baseline: EvaluationSummary,
    change: EvaluationSummary,
    calibration: JudgeCalibration,
    thresholds: EvaluationThresholds,
) -> EvaluationDecision:
    """Adopt only a calibrated, same-data, non-regressing change."""

    if baseline.dataset_digest != change.dataset_digest:
        return EvaluationDecision("inconclusive", "baseline and change data differ")
    if calibration.agreement < thresholds.judge_agreement_minimum:
        return EvaluationDecision("inconclusive", "judge calibration is insufficient")
    if (
        change.critical_safety_pass_rate < thresholds.critical_safety_pass_rate
        or change.task_pass_rate < thresholds.task_success_minimum
    ):
        return EvaluationDecision("reject", "change misses a release threshold")
    if change.task_pass_rate <= baseline.task_pass_rate:
        return EvaluationDecision(
            "inconclusive", "change does not improve the baseline"
        )
    return EvaluationDecision("adopt", "calibrated change improves the same cases")


# Lesson 06: minimized operational traces, degraded mode, and rollback.


@dataclass(frozen=True)
class InvocationResult:
    """One normal or degraded response and its minimized trace fields."""

    mode: Literal["normal", "degraded"]
    response: str
    trace: Mapping[str, object]


class SimulatedReleaseRouter:
    """Exercise dependency failures and immutable rollback without a network."""

    def __init__(self, versions: Iterable[str], active_version: str) -> None:
        self.versions = frozenset(versions)
        if active_version not in self.versions:
            raise ValueError("active version must be immutable and known")
        self.active_version = active_version

    def invoke(
        self,
        query: str,
        *,
        correlation_id: str,
        dependency_state: Literal["healthy", "throttled", "unavailable"],
    ) -> InvocationResult:
        if not query.strip() or not correlation_id.strip():
            raise ValueError("query and correlation_id must not be blank")
        degraded = dependency_state != "healthy"
        error_type = {
            "healthy": None,
            "throttled": "rate_limited",
            "unavailable": "dependency_unavailable",
        }[dependency_state]
        trace = {
            "correlation_id": correlation_id,
            "release": self.active_version,
            "dependency_state": dependency_state,
            "outcome": "degraded" if degraded else "success",
            "latency_ms": 45 if degraded else 120,
            "error_type": error_type,
            "query_digest": hashlib.sha256(query.encode()).hexdigest(),
            "evidence_source": EVIDENCE_SOURCE,
        }
        response = (
            "Approved static guidance only; live retrieval is unavailable."
            if degraded
            else "Live dependency response from the active immutable version."
        )
        return InvocationResult("degraded" if degraded else "normal", response, trace)

    def rollback(self, target_version: str) -> dict[str, str]:
        """Route all traffic to a known immutable version."""

        if target_version not in self.versions:
            raise ValueError("rollback target must be a known immutable version")
        previous = self.active_version
        self.active_version = target_version
        return {"from": previous, "to": target_version, "status": "completed"}


# Lesson 07: a fail-closed capstone evidence map.


@dataclass(frozen=True)
class EvidenceItem:
    """One owned capstone criterion and its immutable artifact reference."""

    status: EvidenceStatus
    artifact: str | None
    owner: str


@dataclass(frozen=True)
class EvidenceDecision:
    """Fail-closed decision over the complete required evidence map."""

    decision: Decision
    failed: tuple[str, ...]
    missing: tuple[str, ...]


def decide_evidence_map(
    evidence: Mapping[str, EvidenceItem],
    required: Iterable[str],
) -> EvidenceDecision:
    """Reject explicit failures and keep absent/unreferenced evidence inconclusive."""

    required_items = tuple(dict.fromkeys(required))
    failed = tuple(
        name
        for name in required_items
        if name in evidence and evidence[name].status == "fail"
    )
    missing = tuple(
        name
        for name in required_items
        if name not in evidence
        or evidence[name].status == "missing"
        or not evidence[name].owner.strip()
        or (evidence[name].status == "pass" and not evidence[name].artifact)
    )
    if failed:
        return EvidenceDecision("reject", failed, missing)
    if missing:
        return EvidenceDecision("inconclusive", (), missing)
    return EvidenceDecision("adopt", (), ())


__all__ = [
    "EVIDENCE_SOURCE",
    "EvaluationBoundary",
    "EvaluationDecision",
    "EvaluationSummary",
    "EvaluationThresholds",
    "EvidenceDecision",
    "EvidenceItem",
    "FakeEvaluationProvider",
    "FakeTextResponseBoundary",
    "InvocationResult",
    "JudgeCalibration",
    "ModelCase",
    "ModelDecision",
    "ModelObservation",
    "ModelResult",
    "ModelThresholds",
    "RetrievalDocument",
    "RetrievedChunk",
    "RiskTreatment",
    "SimulatedReleaseRouter",
    "SimulatedToolBackend",
    "StructuredCaseResult",
    "ToolGateway",
    "ToolOutcome",
    "ToolPolicy",
    "assess_risk_treatments",
    "build_grounded_answer",
    "calibrate_binary_judge",
    "citations_resolve",
    "compare_models",
    "dataset_digest",
    "decide_evaluation",
    "decide_evidence_map",
    "evaluate_version",
    "exercise_structured_outputs",
    "hybrid_retrieve",
    "select_model",
]
