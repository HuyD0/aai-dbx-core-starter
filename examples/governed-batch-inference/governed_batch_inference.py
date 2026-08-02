"""Governed batch inference: spec, sampling, scoring, gate, and provenance.

Batch inference over a large table is a model deployment wearing a SQL
query's clothes. This module supplies the reusable, locally testable pieces
of the governed pipeline demonstrated in ``example_notebook.py``:

- ``BatchInferenceSpec`` — a strict, frozen Pydantic v2 spec declared and
  committed *before* any results exist. Tolerances set after looking at
  output are not tolerances.
- Cost estimation that fails *before* execution when the projected spend
  exceeds the declared ceiling.
- Stratified sample allocation with deliberate over-sampling of rare strata,
  sized against human labelling capacity.
- Per-field, per-stratum precision/recall scoring with Wilson score
  confidence intervals.
- A gate that compares the *lower bound* of the interval — never the point
  estimate — against the declared tolerance, and for ``criticality: high``
  fields gates on the *worst-performing stratum*, never the weighted
  average.
- SQL builders for the ``ai_query`` execute step (structured output with an
  abstention path, idempotent anti-join restart) and for three-layer
  provenance (column naming, Unity Catalog column tags, run metadata).

Everything here is pure Python over plain data so the statistics can be
unit-tested without Spark, MLflow, or a workspace. Spark-side work (reading
tables, drawing the sample, running ``ai_query``) lives in the notebook.

The module intentionally has no aai-core dependency so a team can copy the
two files into any Databricks project; align the vocabulary (``adopt`` /
``reject`` / ``inconclusive`` decisions) with your platform standards.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from enum import IntEnum, StrEnum
from statistics import NormalDist

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants and small helpers
# ---------------------------------------------------------------------------

#: Stratum label used for scores computed over the whole (pooled) sample.
POOLED = "__pooled__"

#: Prefix that travels with AI-derived columns through a ``SELECT *``.
AI_COLUMN_PREFIX = "ai_"

#: Rough characters-per-token used for pre-run cost estimation. This is an
#: estimate, not a measurement: exact tokenization is model-specific and
#: ``ai_query`` does not return usage. Refine from
#: ``system.billing.usage`` (offering type BATCH_INFERENCE) after a run.
CHARS_PER_TOKEN = 4.0

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_identifier(value: str, what: str) -> str:
    """SQL builders interpolate these names, so restrict them hard."""
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{what} {value!r} must match {_IDENTIFIER.pattern}")
    return value


def _require_three_part_name(value: str, what: str) -> str:
    parts = value.split(".")
    if len(parts) != 3:
        raise ValueError(f"{what} {value!r} must be a catalog.schema.table name")
    for part in parts:
        _require_identifier(part, f"{what} part")
    return value


def sql_string_literal(text: str) -> str:
    """Escape ``text`` as a single-quoted Spark SQL string literal."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


class CostCeilingExceeded(RuntimeError):
    """Raised before execution when projected cost exceeds the ceiling."""


class GateNotPassed(RuntimeError):
    """Raised when execution is attempted without an adopting gate decision."""


# ---------------------------------------------------------------------------
# 1. Declare — the spec is written before any results exist
# ---------------------------------------------------------------------------


class Criticality(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UseTier(IntEnum):
    """The tier is set by what consumes the output, not by table size."""

    CONSEQUENTIAL = 1  # member-facing, valuation, reporting, regulatory
    OPERATIONAL = 2  # persisted, consumed by internal processes
    EXPLORATORY = 3  # notebook-only, one consumer: cost tracking, no gate


class GateDecision(StrEnum):
    """Aligned with the platform decision vocabulary."""

    ADOPT = "adopt"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"
    PENDING_APPROVAL = "pending_approval"


class FieldSpec(BaseModel):
    """One extracted field and the error its consumers can tolerate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = Field(min_length=1)
    criticality: Criticality
    tolerable_error_rate: float = Field(gt=0.0, lt=1.0)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        return _require_identifier(value, "field name")

    @property
    def required_rate(self) -> float:
        """The success rate the interval lower bound must clear."""
        return 1.0 - self.tolerable_error_rate


class BatchInferenceSpec(BaseModel):
    """Typed, versioned declaration of one governed batch inference job.

    The single most important property of this model is *when* it exists:
    it is authored, reviewed, and committed before any inference results
    are seen. ``evaluate_gate`` reads tolerances from here, never from a
    number typed in after looking at the output.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_version: str = "1"
    name: str
    source_table: str
    target_table: str
    run_metadata_table: str
    document_column: str
    key_column: str
    use_tier: UseTier
    consumed_by: tuple[str, ...] = Field(min_length=1)
    fields: tuple[FieldSpec, ...] = Field(min_length=1)
    strata: tuple[str, ...] = Field(min_length=1)
    endpoint: str
    model_version: str
    prompt_version: str
    cost_ceiling_cad: float = Field(gt=0.0)
    abstain_threshold: float = Field(ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    rollback_plan: str | None = None

    @field_validator("name")
    @classmethod
    def _valid_spec_name(cls, value: str) -> str:
        return _require_identifier(value, "spec name")

    @field_validator("source_table", "target_table", "run_metadata_table")
    @classmethod
    def _valid_table(cls, value: str) -> str:
        return _require_three_part_name(value, "table")

    @field_validator("document_column", "key_column")
    @classmethod
    def _valid_column(cls, value: str) -> str:
        return _require_identifier(value, "column")

    @field_validator("strata")
    @classmethod
    def _valid_strata(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for column in value:
            _require_identifier(column, "stratum column")
        if len(set(value)) != len(value):
            raise ValueError("strata columns must be unique")
        return value

    @model_validator(mode="after")
    def _cross_checks(self) -> BatchInferenceSpec:
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("field names must be unique")
        if self.target_table == self.source_table:
            raise ValueError("target_table must differ from source_table")
        if self.use_tier == UseTier.CONSEQUENTIAL and not self.rollback_plan:
            raise ValueError(
                "tier 1 (consequential) requires a documented rollback_plan"
            )
        return self

    def field_named(self, name: str) -> FieldSpec:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)

    @property
    def gate_required(self) -> bool:
        """Tier 3 is cost tracking only — do not over-control it."""
        return self.use_tier != UseTier.EXPLORATORY

    @property
    def spec_digest(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> BatchInferenceSpec:
        return cls.model_validate(yaml.safe_load(text))


# ---------------------------------------------------------------------------
# 2. Estimate — convert "we discovered a bill" into "we approved a budget"
# ---------------------------------------------------------------------------


class CostEstimate(BaseModel):
    """Projected spend for the full run, computed from a small probe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_count: int = Field(ge=0)
    probe_row_count: int = Field(gt=0)
    mean_input_tokens_per_row: float = Field(ge=0.0)
    mean_output_tokens_per_row: float = Field(ge=0.0)
    cad_per_million_input_tokens: float = Field(ge=0.0)
    cad_per_million_output_tokens: float = Field(ge=0.0)
    safety_factor: float = Field(ge=1.0)
    projected_cost_cad: float = Field(ge=0.0)
    cost_ceiling_cad: float = Field(gt=0.0)

    @property
    def within_ceiling(self) -> bool:
        return self.projected_cost_cad <= self.cost_ceiling_cad


def estimate_tokens_from_text(text: str) -> int:
    """Cheap token estimate for probing; see ``CHARS_PER_TOKEN``."""
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def estimate_cost(
    spec: BatchInferenceSpec,
    *,
    row_count: int,
    probe_input_tokens: Sequence[int],
    probe_output_tokens: Sequence[int],
    cad_per_million_input_tokens: float,
    cad_per_million_output_tokens: float,
    safety_factor: float = 1.2,
) -> CostEstimate:
    """Project full-run cost from per-row probe token counts.

    Rate limits do not stop a runaway ``ai_query`` batch job on a
    pay-per-token endpoint; only this estimate, made before execution,
    turns the spend into a decision instead of a discovery.
    """
    if not probe_input_tokens or not probe_output_tokens:
        raise ValueError("probe token sequences must be non-empty")
    mean_in = sum(probe_input_tokens) / len(probe_input_tokens)
    mean_out = sum(probe_output_tokens) / len(probe_output_tokens)
    projected = (
        safety_factor
        * row_count
        * (
            mean_in * cad_per_million_input_tokens
            + mean_out * cad_per_million_output_tokens
        )
        / 1_000_000.0
    )
    return CostEstimate(
        row_count=row_count,
        probe_row_count=len(probe_input_tokens),
        mean_input_tokens_per_row=mean_in,
        mean_output_tokens_per_row=mean_out,
        cad_per_million_input_tokens=cad_per_million_input_tokens,
        cad_per_million_output_tokens=cad_per_million_output_tokens,
        safety_factor=safety_factor,
        projected_cost_cad=projected,
        cost_ceiling_cad=spec.cost_ceiling_cad,
    )


def require_within_ceiling(estimate: CostEstimate) -> CostEstimate:
    """Fail before executing — the whole point of estimating first."""
    if not estimate.within_ceiling:
        raise CostCeilingExceeded(
            f"projected cost {estimate.projected_cost_cad:.2f} CAD exceeds the "
            f"declared ceiling {estimate.cost_ceiling_cad:.2f} CAD; do not run. "
            "Reduce scope, choose a smaller endpoint, or get the ceiling raised "
            "explicitly."
        )
    return estimate


# ---------------------------------------------------------------------------
# Wilson score intervals — the statistical core of the gate
# ---------------------------------------------------------------------------


class ConfidenceInterval(BaseModel):
    """A proportion with the uncertainty its sample size actually supports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    successes: int = Field(ge=0)
    trials: int = Field(gt=0)
    confidence: float = Field(gt=0.5, lt=1.0)
    point: float
    lower: float
    upper: float


def _z_value(confidence: float) -> float:
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> ConfidenceInterval:
    """Wilson score interval for a binomial proportion.

    Preferred over the naive normal approximation because it stays inside
    [0, 1], behaves sensibly at 0 and 100 percent observed success, and is
    accurate at the small per-stratum sample sizes human labelling budgets
    actually produce.
    """
    if trials <= 0:
        raise ValueError("trials must be positive; no evidence is not evidence")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1.0)")
    z = _z_value(confidence)
    n = float(trials)
    p_hat = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denominator
    half_width = (
        z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) / denominator
    )
    return ConfidenceInterval(
        successes=successes,
        trials=trials,
        confidence=confidence,
        point=p_hat,
        lower=max(0.0, centre - half_width),
        upper=min(1.0, centre + half_width),
    )


def min_labelled_rows_for_tolerance(
    tolerable_error_rate: float, confidence: float = 0.95
) -> int:
    """Smallest n at which even a *flawless* sample can clear the tolerance.

    With k = n successes the Wilson lower bound reduces to n / (n + z²);
    below this n the gate cannot pass no matter how good the model is, so
    labelling fewer rows than this per stratum buys no decision at all.
    This is the number that turns "how many can we label?" into an explicit
    trade against the declared tolerance.
    """
    required = 1.0 - tolerable_error_rate
    z2 = _z_value(confidence) ** 2
    return math.ceil(required * z2 / (1.0 - required))


# ---------------------------------------------------------------------------
# 3. Sample — stratified, sized by human labelling capacity
# ---------------------------------------------------------------------------


def allocate_stratified_sample(
    population: Mapping[str, int],
    labelling_budget: int,
    min_per_stratum: int,
) -> dict[str, int]:
    """Allocate a labelling budget across strata, over-sampling rare ones.

    Every stratum first receives ``min_per_stratum`` rows (or its whole
    population if smaller) — this is the deliberate over-sampling of rare
    strata. A proportional sample from a population where 2 percent of rows
    are the hard case yields almost no information about the hard case,
    which is the case that matters. The remaining budget is then split
    proportionally to remaining population using largest-remainder
    rounding, so the result is deterministic and sums exactly.

    Raises ``ValueError`` when the budget cannot cover the floors: that is
    a real capacity conversation (label more, relax the tolerance, or merge
    strata), not something to paper over silently.
    """
    if labelling_budget <= 0:
        raise ValueError("labelling_budget must be positive")
    if min_per_stratum <= 0:
        raise ValueError("min_per_stratum must be positive")
    if not population:
        raise ValueError("population must contain at least one stratum")
    for stratum, count in population.items():
        if count < 0:
            raise ValueError(f"stratum {stratum!r} has negative population")

    total_population = sum(population.values())
    if labelling_budget >= total_population:
        return {stratum: count for stratum, count in population.items()}

    floors = {
        stratum: min(count, min_per_stratum) for stratum, count in population.items()
    }
    floor_total = sum(floors.values())
    if labelling_budget < floor_total:
        raise ValueError(
            f"labelling budget {labelling_budget} cannot give every stratum its "
            f"floor of {min_per_stratum} (needs {floor_total}). Raise the "
            "budget, merge strata, or accept an inconclusive gate for the "
            "missing strata — explicitly."
        )

    allocation = dict(floors)
    remaining_budget = labelling_budget - floor_total
    spare = {
        stratum: population[stratum] - allocation[stratum] for stratum in population
    }
    spare_total = sum(spare.values())
    if remaining_budget and spare_total:
        quotas = {
            stratum: remaining_budget * spare[stratum] / spare_total
            for stratum in population
        }
        for stratum in population:
            extra = min(int(quotas[stratum]), spare[stratum])
            allocation[stratum] += extra
            spare[stratum] -= extra
            remaining_budget -= extra
        by_remainder = sorted(
            population,
            key=lambda stratum: (-(quotas[stratum] % 1.0), stratum),
        )
        while remaining_budget > 0:
            progressed = False
            for stratum in by_remainder:
                if remaining_budget == 0:
                    break
                if spare[stratum] > 0:
                    allocation[stratum] += 1
                    spare[stratum] -= 1
                    remaining_budget -= 1
                    progressed = True
            if not progressed:  # pragma: no cover - guarded by budget cap above
                break
    return allocation


# ---------------------------------------------------------------------------
# 4. Evaluate — per field, per stratum; precision and recall separately
# ---------------------------------------------------------------------------


class EvaluationRecord(BaseModel):
    """One labelled sample row: gold values, model values, abstentions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stratum: str
    gold: Mapping[str, str | int | float | None]
    predicted: Mapping[str, str | int | float | None]
    abstained: frozenset[str] = frozenset()


class FieldStratumScore(BaseModel):
    """Precision/recall evidence for one field within one stratum.

    ``precision`` is None when the model asserted nothing in the stratum
    and ``recall`` is None when gold has no values there — "no evidence",
    which the gate treats as inconclusive, never as a pass.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    stratum: str
    n_rows: int = Field(ge=0)
    n_gold: int = Field(ge=0)
    n_asserted: int = Field(ge=0)
    n_correct: int = Field(ge=0)
    precision: ConfidenceInterval | None
    recall: ConfidenceInterval | None
    abstention_rate: float = Field(ge=0.0, le=1.0)


_AMOUNT_JUNK = re.compile(r"[,$\s]")


def values_match(gold: object, predicted: object) -> bool:
    """Field-level equality: numeric-aware, whitespace/case tolerant.

    ``$12,345.60`` matches ``12345.6``; issuer names match after case and
    whitespace normalisation. Adapt per field in real use (dates, currency
    codes); keep the comparator deterministic so scoring is reproducible.
    """
    if gold is None or predicted is None:
        return False
    gold_text = str(gold).strip()
    predicted_text = str(predicted).strip()
    try:
        return float(_AMOUNT_JUNK.sub("", gold_text)) == float(
            _AMOUNT_JUNK.sub("", predicted_text)
        )
    except ValueError:
        pass
    normalise = re.compile(r"\s+")
    return (
        normalise.sub(" ", gold_text).casefold()
        == normalise.sub(" ", predicted_text).casefold()
    )


def _score_group(
    field: FieldSpec,
    stratum: str,
    records: Sequence[EvaluationRecord],
    confidence: float,
) -> FieldStratumScore:
    asserted = 0
    correct = 0
    gold_present = 0
    abstained = 0
    for record in records:
        gold_value = record.gold.get(field.name)
        predicted_value = record.predicted.get(field.name)
        did_abstain = field.name in record.abstained
        if gold_value is not None:
            gold_present += 1
        if did_abstain:
            abstained += 1
            continue
        if predicted_value is None:
            continue
        asserted += 1
        # Asserting a value where gold has none is a hallucination: it
        # lands in the precision denominator and can never be correct.
        if gold_value is not None and values_match(gold_value, predicted_value):
            correct += 1
    return FieldStratumScore(
        field=field.name,
        stratum=stratum,
        n_rows=len(records),
        n_gold=gold_present,
        n_asserted=asserted,
        n_correct=correct,
        precision=(
            wilson_interval(correct, asserted, confidence) if asserted else None
        ),
        recall=(
            wilson_interval(correct, gold_present, confidence) if gold_present else None
        ),
        abstention_rate=abstained / len(records) if records else 0.0,
    )


def score_extraction(
    records: Sequence[EvaluationRecord],
    fields: Sequence[FieldSpec],
    confidence: float = 0.95,
) -> tuple[FieldStratumScore, ...]:
    """Score every field in every stratum, plus a pooled row per field.

    Precision — of the values the model *asserted*, how many were right —
    and recall — of the values that truly exist, how many were correctly
    produced — are reported separately because extraction fails in two
    different ways: hallucinating and missing. A single accuracy figure
    hides whichever failure mode your consumers care about. Abstentions
    count against recall (the value exists and was not produced) but not
    against precision (nothing false was asserted): the abstention path
    converts silent errors into visible work.

    The pooled row (stratum ``POOLED``) is reported for context only. A
    stratified sample deliberately over-weights rare strata, so its pooled
    proportion estimates nothing about the population — and the gate for
    high-criticality fields never uses it.
    """
    if not records:
        raise ValueError("cannot score an empty sample")
    by_stratum: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        by_stratum.setdefault(record.stratum, []).append(record)
    scores: list[FieldStratumScore] = []
    for field in fields:
        for stratum in sorted(by_stratum):
            scores.append(_score_group(field, stratum, by_stratum[stratum], confidence))
        scores.append(_score_group(field, POOLED, records, confidence))
    return tuple(scores)


# ---------------------------------------------------------------------------
# 5. Gate — lower bound vs declared tolerance; worst stratum for high fields
# ---------------------------------------------------------------------------


class FieldGateResult(BaseModel):
    """The gate's verdict for one field, with the evidence that binds it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    criticality: Criticality
    required_rate: float
    decision: GateDecision
    binding_stratum: str | None
    binding_metric: str | None
    binding_lower_bound: float | None
    binding_point_estimate: float | None
    reasons: tuple[str, ...]


class GateReport(BaseModel):
    """Immutable gate evidence for one spec + one evaluation sample."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_name: str
    spec_digest: str
    use_tier: UseTier
    confidence_level: float
    fields: tuple[FieldGateResult, ...]
    decision: GateDecision
    approved_by: str | None = None
    human_review_obligations: tuple[str, ...] = ()


def _gate_metric(
    score: FieldStratumScore,
    interval: ConfidenceInterval | None,
    metric: str,
    required: float,
) -> tuple[GateDecision, str | None, float | None, float | None]:
    """Judge one metric in one stratum. Returns (decision, reason, lower, point).

    Inconclusive is distinct from reject: when the sample is so small that
    even a flawless result could not clear the bar (max achievable lower
    bound = n/(n+z²) < required), the evidence is insufficient — label more
    rows; the model has not been shown to be bad.
    """
    if interval is None:
        return (
            GateDecision.INCONCLUSIVE,
            f"no {metric} evidence in stratum {score.stratum!r}",
            None,
            None,
        )
    if interval.lower >= required:
        return (GateDecision.ADOPT, None, interval.lower, interval.point)
    best_possible = wilson_interval(
        interval.trials, interval.trials, interval.confidence
    ).lower
    if best_possible < required:
        return (
            GateDecision.INCONCLUSIVE,
            (
                f"{metric} in stratum {score.stratum!r} has only "
                f"{interval.trials} labelled rows; even a flawless sample "
                f"tops out at a lower bound of {best_possible:.4f} < "
                f"{required:.4f} — insufficient evidence, label more rows"
            ),
            interval.lower,
            interval.point,
        )
    return (
        GateDecision.REJECT,
        (
            f"{metric} in stratum {score.stratum!r}: lower bound "
            f"{interval.lower:.4f} < required {required:.4f} "
            f"(point estimate {interval.point:.4f} from "
            f"{interval.successes}/{interval.trials} — the point estimate "
            "is not the evidence)"
        ),
        interval.lower,
        interval.point,
    )


def _gate_field(
    field: FieldSpec,
    scores: Sequence[FieldStratumScore],
) -> FieldGateResult:
    required = field.required_rate
    # criticality: high → every stratum must clear the bar on its own.
    # Aggregate performance is irrelevant if the failures concentrate in
    # the one stratum that matters, so the weighted average never appears
    # here. medium / low → the pooled sample decides.
    if field.criticality == Criticality.HIGH:
        considered = [score for score in scores if score.stratum != POOLED]
    else:
        considered = [score for score in scores if score.stratum == POOLED]
    if not considered:
        raise ValueError(f"no scores supplied for field {field.name!r}")

    reasons: list[str] = []
    verdicts: list[GateDecision] = []
    binding: tuple[float, str, str, float | None] | None = None
    for score in considered:
        for metric, interval in (
            ("precision", score.precision),
            ("recall", score.recall),
        ):
            decision, reason, lower, point = _gate_metric(
                score, interval, metric, required
            )
            verdicts.append(decision)
            if reason:
                reasons.append(reason)
            if lower is not None and (binding is None or lower < binding[0]):
                binding = (lower, score.stratum, metric, point)

    if GateDecision.REJECT in verdicts:
        decision = GateDecision.REJECT
    elif GateDecision.INCONCLUSIVE in verdicts:
        decision = GateDecision.INCONCLUSIVE
    else:
        decision = GateDecision.ADOPT
    return FieldGateResult(
        field=field.name,
        criticality=field.criticality,
        required_rate=required,
        decision=decision,
        binding_stratum=binding[1] if binding else None,
        binding_metric=binding[2] if binding else None,
        binding_lower_bound=binding[0] if binding else None,
        binding_point_estimate=binding[3] if binding else None,
        reasons=tuple(reasons),
    )


def evaluate_gate(
    spec: BatchInferenceSpec,
    scores: Sequence[FieldStratumScore],
) -> GateReport:
    """Compare interval lower bounds against the tolerances declared in the
    spec — never the point estimates.

    Tier semantics:

    - Tier 3 (exploratory) does not require this gate at all; calling it
      anyway is harmless but execution is not blocked on it.
    - Tier 2 (operational): all fields adopt → the run may proceed.
    - Tier 1 (consequential): even a fully passing result returns
      ``PENDING_APPROVAL``. Unreviewed acceptance is not available at this
      tier regardless of measured accuracy; a named human must call
      ``approve_gate``.
    """
    by_field: dict[str, list[FieldStratumScore]] = {}
    for score in scores:
        by_field.setdefault(score.field, []).append(score)
    results = tuple(
        _gate_field(field, by_field.get(field.name, ())) for field in spec.fields
    )
    field_decisions = {result.decision for result in results}
    if GateDecision.REJECT in field_decisions:
        decision = GateDecision.REJECT
    elif GateDecision.INCONCLUSIVE in field_decisions:
        decision = GateDecision.INCONCLUSIVE
    elif spec.use_tier == UseTier.CONSEQUENTIAL:
        decision = GateDecision.PENDING_APPROVAL
    else:
        decision = GateDecision.ADOPT
    obligations: tuple[str, ...] = ()
    if spec.use_tier == UseTier.CONSEQUENTIAL:
        obligations = (
            "human review of every abstained row before downstream use",
            "human review of all criticality=high fields on a fresh sample",
            f"rollback path on file: {spec.rollback_plan}",
        )
    return GateReport(
        spec_name=spec.name,
        spec_digest=spec.spec_digest,
        use_tier=spec.use_tier,
        confidence_level=spec.confidence_level,
        fields=results,
        decision=decision,
        human_review_obligations=obligations,
    )


def approve_gate(report: GateReport, approver: str) -> GateReport:
    """Record the named human sign-off a tier 1 run requires.

    Only a ``PENDING_APPROVAL`` report can be approved: approval is a
    person accepting a passing result's residual risk, never a way to
    override a rejection or to substitute for missing evidence.
    """
    if report.decision != GateDecision.PENDING_APPROVAL:
        raise GateNotPassed(
            f"only a pending_approval report can be approved, got "
            f"{report.decision.value!r}"
        )
    if not approver.strip():
        raise ValueError("approver must be a named person or group")
    return report.model_copy(
        update={"decision": GateDecision.ADOPT, "approved_by": approver.strip()}
    )


def require_executable(spec: BatchInferenceSpec, report: GateReport | None) -> None:
    """Refuse to execute without an adopting gate (tiers 1 and 2)."""
    if not spec.gate_required:
        return
    if report is None:
        raise GateNotPassed(f"tier {spec.use_tier} runs require a gate report")
    if report.spec_digest != spec.spec_digest:
        raise GateNotPassed(
            "gate report was produced for a different spec revision; "
            "re-evaluate against the current spec"
        )
    if report.decision != GateDecision.ADOPT:
        raise GateNotPassed(
            f"gate decision is {report.decision.value!r}; execution requires " "'adopt'"
        )
    if spec.use_tier == UseTier.CONSEQUENTIAL and not report.approved_by:
        raise GateNotPassed("tier 1 execution requires a named approver")


# ---------------------------------------------------------------------------
# 6. Execute — ai_query with structured output, abstention, idempotent restart
# ---------------------------------------------------------------------------


def ai_column(field_name: str) -> str:
    """Layer 1 of provenance: a prefix that survives ``SELECT *``."""
    return f"{AI_COLUMN_PREFIX}{field_name}"


def response_format(spec: BatchInferenceSpec) -> dict:
    """The ``responseFormat`` JSON schema for ``ai_query`` structured output.

    Databricks structured outputs support only a subset of JSON Schema:
    no ``anyOf``/``oneOf``/``allOf``, no ``pattern``, and the *only*
    permitted type union is ``[type, "null"]`` for nullable fields — which
    is exactly what the abstention path needs. ``strict`` sits inside
    ``json_schema`` (verified against the structured-outputs documentation;
    re-check when upgrading, this surface has changed before).
    """
    properties: dict[str, dict] = {}
    required: list[str] = []
    for field in spec.fields:
        properties[field.name] = {
            "type": ["string", "null"],
            "description": (
                f"{field.description} Null when absent from the document or "
                "when abstaining."
            ),
        }
        properties[f"{field.name}_confidence"] = {
            "type": ["number", "null"],
            "description": f"Confidence from 0 to 1 for {field.name}.",
        }
        required.extend([field.name, f"{field.name}_confidence"])
    properties["abstained_fields"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "Names of requested fields deliberately not answered because "
            f"confidence was below {spec.abstain_threshold}. Do not guess."
        ),
    }
    properties["abstain_reason"] = {
        "type": ["string", "null"],
        "description": "Why the abstained fields could not be read.",
    }
    required.extend(["abstained_fields", "abstain_reason"])
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{spec.name}_extraction",
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "strict": True,
        },
    }


def response_format_sql_literal(spec: BatchInferenceSpec) -> str:
    return sql_string_literal(json.dumps(response_format(spec)))


def response_struct_type(spec: BatchInferenceSpec) -> str:
    """Spark SQL type used by ``from_json`` to parse the model response.

    Values stay STRING at the extraction boundary; casting to DECIMAL or
    DATE is a downstream, per-consumer decision that should fail visibly
    there, not silently here.
    """
    parts = []
    for field in spec.fields:
        parts.append(f"{field.name}: STRING")
        parts.append(f"{field.name}_confidence: DOUBLE")
    parts.append("abstained_fields: ARRAY<STRING>")
    parts.append("abstain_reason: STRING")
    return "STRUCT<" + ", ".join(parts) + ">"


def create_target_table_sql(spec: BatchInferenceSpec) -> str:
    """DDL for the output table, provenance columns included from birth."""
    body = ",\n".join(f"  {name} {sql_type}" for name, sql_type in target_columns(spec))
    return f"CREATE TABLE IF NOT EXISTS {spec.target_table} (\n{body}\n)"


def target_columns(spec: BatchInferenceSpec) -> tuple[tuple[str, str], ...]:
    """(name, SQL type) for every target column, in canonical order.

    One definition serves both the DDL and the INSERT column list so the
    two can never drift apart.
    """
    columns: list[tuple[str, str]] = [(spec.key_column, "STRING")]
    columns.extend((column, "STRING") for column in spec.strata)
    for field in spec.fields:
        name = ai_column(field.name)
        columns.append((name, "STRING"))
        columns.append((f"{name}_confidence", "DOUBLE"))
        columns.append((f"{name}_abstained", "BOOLEAN"))
    columns.extend(
        [
            ("ai_abstained_fields", "ARRAY<STRING>"),
            ("ai_abstain_reason", "STRING"),
            ("ai_error", "STRING"),
            ("ai_run_id", "STRING"),
            ("ai_model_version", "STRING"),
            ("ai_prompt_version", "STRING"),
            ("ai_executed_at", "TIMESTAMP"),
        ]
    )
    return tuple(columns)


def build_execute_sql(
    spec: BatchInferenceSpec,
    *,
    run_id: str,
    prompt_sql: str,
) -> str:
    """The full-table execute statement, restartable by construction.

    - ``prompt_sql`` is a SQL expression producing the request string (the
      notebook builds it with ``concat`` from an escaped instruction
      literal and the document column).
    - The anti-join means a re-run after a partial failure processes only
      rows that have not landed yet: a million-row job will fail partway
      at some point, and this is what makes that boring. Current guidance
      is to submit the remaining set as one query — AI Functions manage
      parallelization and retries — rather than hand-chunking it.
    - ``failOnError => false`` keeps one poisoned document from killing
      the run; its error message lands in ``ai_error`` and the row flows
      to the exception queue instead of blocking everything else.
    """
    value_lines = []
    for field in spec.fields:
        name = ai_column(field.name)
        value_lines.append(f"  parsed.{field.name} AS {name}")
        value_lines.append(f"  parsed.{field.name}_confidence AS {name}_confidence")
        value_lines.append(
            f"  coalesce(array_contains(parsed.abstained_fields, "
            f"'{field.name}'), false) AS {name}_abstained"
        )
    value_block = ",\n".join(value_lines)
    pending_strata = "".join(f", source.{column}" for column in spec.strata)
    plain_strata = "".join(f", {column}" for column in spec.strata)
    insert_columns = ", ".join(name for name, _ in target_columns(spec))
    struct_type = sql_string_literal(response_struct_type(spec))
    return f"""INSERT INTO {spec.target_table} ({insert_columns})
WITH pending AS (
  SELECT source.{spec.key_column}, source.{spec.document_column}{pending_strata}
  FROM {spec.source_table} AS source
  LEFT ANTI JOIN {spec.target_table} AS done
    ON source.{spec.key_column} = done.{spec.key_column}
),
scored AS (
  SELECT
    *,
    ai_query(
      {sql_string_literal(spec.endpoint)},
      {prompt_sql},
      responseFormat => {response_format_sql_literal(spec)},
      failOnError => false
    ) AS raw
  FROM pending
),
parsed AS (
  SELECT
    * EXCEPT (raw),
    from_json(raw.response, {struct_type}) AS parsed,
    raw.errorMessage AS error_message
  FROM scored
)
SELECT
  {spec.key_column}{plain_strata},
{value_block},
  parsed.abstained_fields AS ai_abstained_fields,
  parsed.abstain_reason AS ai_abstain_reason,
  error_message AS ai_error,
  {sql_string_literal(run_id)} AS ai_run_id,
  {sql_string_literal(spec.model_version)} AS ai_model_version,
  {sql_string_literal(spec.prompt_version)} AS ai_prompt_version,
  current_timestamp() AS ai_executed_at
FROM parsed"""


# ---------------------------------------------------------------------------
# 7. Land — provenance that survives joins, three layers deep
# ---------------------------------------------------------------------------


def column_tag_statements(spec: BatchInferenceSpec, run_id: str) -> tuple[str, ...]:
    """Layer 2: Unity Catalog column tags, queryable across the estate.

    ``SELECT * FROM <catalog>.information_schema.column_tags WHERE
    tag_name = 'data_source' AND tag_value = 'ai_generated'`` answers
    "what in this estate is AI-derived?" long after column names stop
    being read. Requires APPLY TAG on the table.
    """
    statements = []
    tags = (
        "'data_source' = 'ai_generated', "
        f"'ai_endpoint' = {sql_string_literal(spec.endpoint)}, "
        f"'ai_model_version' = {sql_string_literal(spec.model_version)}, "
        f"'ai_prompt_version' = {sql_string_literal(spec.prompt_version)}, "
        f"'ai_run_id' = {sql_string_literal(run_id)}"
    )
    for field in spec.fields:
        statements.append(
            f"ALTER TABLE {spec.target_table} ALTER COLUMN "
            f"{ai_column(field.name)} SET TAGS ({tags})"
        )
    return tuple(statements)


def create_run_metadata_table_sql(spec: BatchInferenceSpec) -> str:
    """Layer 3: the run record everything joins back to by ``run_id``."""
    return f"""CREATE TABLE IF NOT EXISTS {spec.run_metadata_table} (
  run_id STRING,
  spec_name STRING,
  spec_digest STRING,
  spec_yaml STRING,
  use_tier INT,
  endpoint STRING,
  model_version STRING,
  prompt_version STRING,
  gate_decision STRING,
  approved_by STRING,
  projected_cost_cad DOUBLE,
  target_table STRING,
  target_table_version BIGINT,
  executed_at TIMESTAMP
)"""


def run_metadata_insert_sql(
    spec: BatchInferenceSpec,
    report: GateReport,
    *,
    run_id: str,
    projected_cost_cad: float,
    target_table_version: int,
) -> str:
    approved = sql_string_literal(report.approved_by) if report.approved_by else "NULL"
    return f"""INSERT INTO {spec.run_metadata_table} VALUES (
  {sql_string_literal(run_id)},
  {sql_string_literal(spec.name)},
  {sql_string_literal(spec.spec_digest)},
  {sql_string_literal(spec.to_yaml())},
  {int(spec.use_tier)},
  {sql_string_literal(spec.endpoint)},
  {sql_string_literal(spec.model_version)},
  {sql_string_literal(spec.prompt_version)},
  {sql_string_literal(report.decision.value)},
  {approved},
  {float(projected_cost_cad)},
  {sql_string_literal(spec.target_table)},
  {int(target_table_version)},
  current_timestamp()
)"""


# ---------------------------------------------------------------------------
# 8. Monitor — abstention rate leads, re-sampling verifies
# ---------------------------------------------------------------------------


def exception_queue_view_sql(spec: BatchInferenceSpec, view_name: str) -> str:
    """Abstained and errored rows are work, and someone owns clearing them."""
    _require_three_part_name(view_name, "view")
    return f"""CREATE OR REPLACE VIEW {view_name} AS
SELECT *
FROM {spec.target_table}
WHERE size(ai_abstained_fields) > 0 OR ai_error IS NOT NULL"""


def abstention_trend_sql(spec: BatchInferenceSpec) -> str:
    """Abstention rate per stratum per day: the cheapest early warning.

    It moves before accuracy does and needs no labels — a rising rate
    means the inputs are drifting away from what the prompt was validated
    on. Alert on it; investigate with a labelled re-sample.
    """
    strata_list = ", ".join(spec.strata)
    return f"""SELECT
  date_trunc('DAY', ai_executed_at) AS day,
  {strata_list},
  count(*) AS rows,
  avg(CAST(size(ai_abstained_fields) > 0 AS DOUBLE)) AS abstention_rate,
  avg(CAST(ai_error IS NOT NULL AS DOUBLE)) AS error_rate
FROM {spec.target_table}
GROUP BY ALL
ORDER BY day DESC, {strata_list}"""


# ---------------------------------------------------------------------------
# MLflow evidence — one run holds the whole story
# ---------------------------------------------------------------------------


def metric_key(field: str, stratum: str, metric: str) -> str:
    """MLflow-safe metric name for one field/stratum/metric triple."""
    safe = re.sub(r"[^A-Za-z0-9_\-. :/]", "-", f"{field}/{stratum}/{metric}")
    return safe


def log_gate_evidence(
    spec: BatchInferenceSpec,
    estimate: CostEstimate,
    allocation: Mapping[str, int],
    scores: Sequence[FieldStratumScore],
    report: GateReport,
) -> None:
    """Log spec, sample, per-field per-stratum intervals, and the decision
    to the *active* MLflow run. The caller owns the run lifecycle so the
    same run can later receive the output table version. Imported lazily so
    this module stays importable without MLflow installed.
    """
    import mlflow

    mlflow.log_params(
        {
            "spec_name": spec.name,
            "spec_digest": spec.spec_digest,
            "use_tier": int(spec.use_tier),
            "source_table": spec.source_table,
            "target_table": spec.target_table,
            "endpoint": spec.endpoint,
            "model_version": spec.model_version,
            "prompt_version": spec.prompt_version,
            "confidence_level": spec.confidence_level,
            "consumed_by": ",".join(spec.consumed_by),
        }
    )
    mlflow.log_text(spec.to_yaml(), "governed_batch_inference/spec.yaml")
    mlflow.log_dict(dict(allocation), "governed_batch_inference/sample_allocation.json")
    mlflow.log_metric("projected_cost_cad", estimate.projected_cost_cad)
    for score in scores:
        for metric, interval in (
            ("precision", score.precision),
            ("recall", score.recall),
        ):
            if interval is None:
                continue
            base = metric_key(score.field, score.stratum, metric)
            mlflow.log_metric(f"{base}_point", interval.point)
            mlflow.log_metric(f"{base}_lower", interval.lower)
        mlflow.log_metric(
            metric_key(score.field, score.stratum, "abstention_rate"),
            score.abstention_rate,
        )
    mlflow.log_dict(
        report.model_dump(mode="json"), "governed_batch_inference/gate_report.json"
    )
    mlflow.set_tags(
        {
            "gate_decision": report.decision.value,
            "approved_by": report.approved_by or "",
        }
    )
