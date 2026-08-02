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
  fields gates on the *worst-performing stratum*, never an aggregate.
  Lower-criticality fields use a population-weighted estimate, because the
  raw pool of a stratified sample describes no real population.
- Evidence bound to the release that produced it: scores carry a
  ``ReleaseIdentity`` and the gate refuses numbers measured for a
  different prompt, model, or spec revision.
- SQL builders for the ``ai_query`` execute step (structured output with an
  abstention path, release-aware restartable landing) and for three-layer
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

#: Label for the population-weighted estimate across all strata. It is not
#: a raw pool of the sample: a stratified sample deliberately over-weights
#: rare strata, so pooling its rows estimates nothing about the population.
WEIGHTED = "__population_weighted__"

#: Prefix that travels with AI-derived columns through a ``SELECT *``.
AI_COLUMN_PREFIX = "ai_"

#: Provenance columns every landed row carries, beside the extracted values.
PROVENANCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ai_abstained_fields", "ARRAY<STRING>"),
    ("ai_abstain_reason", "STRING"),
    ("ai_error", "STRING"),
    ("ai_run_id", "STRING"),
    ("ai_spec_digest", "STRING"),
    ("ai_model_version", "STRING"),
    ("ai_prompt_version", "STRING"),
    ("ai_release_sequence", "BIGINT"),
    ("ai_executed_at", "TIMESTAMP"),
)

#: Response-schema keys the extraction contract reserves for itself. Each
#: has a matching ``ai_``-prefixed provenance column above, which is what
#: makes the column collision check cover the response schema as well.
RESERVED_RESPONSE_KEYS = frozenset({"abstained_fields", "abstain_reason"})

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


def ai_column(field_name: str) -> str:
    """Layer 1 of provenance: a prefix that survives ``SELECT *``."""
    return f"{AI_COLUMN_PREFIX}{field_name}"


def _generated_column_names(field_name: str) -> tuple[str, ...]:
    """The three target columns one extracted field expands into."""
    name = ai_column(field_name)
    return (name, f"{name}_confidence", f"{name}_abstained")


def sql_string_literal(text: str) -> str:
    """Escape ``text`` as a single-quoted Spark SQL string literal."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


class CostCeilingExceeded(RuntimeError):
    """Raised before execution when projected cost exceeds the ceiling."""


class GateNotPassed(RuntimeError):
    """Raised when execution is attempted without an adopting gate decision."""


class EvidenceMismatch(RuntimeError):
    """Raised when evidence does not belong to the release being gated."""


class TargetSchemaMismatch(RuntimeError):
    """Raised when the target table still holds columns a release dropped."""


class UnusableSourceKeys(RuntimeError):
    """Raised when source keys cannot support restartable, idempotent landing."""


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


class ReleaseIdentity(BaseModel):
    """What produced a piece of evidence.

    A prompt, model, or spec change is an application release, so evidence
    from one release says nothing about another. Scores carry this stamp
    and the gate refuses evidence that does not match the spec it is
    gating — otherwise a passing v1 sample could authorise a v2 run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_digest: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)


class InferenceIdentity(BaseModel):
    """What produced a *prediction*, as distinct from how it is judged.

    A spec holds two kinds of thing, and conflating them is expensive:

    - what determines the model's output — endpoint, model version, prompt
      version, the fields and descriptions that build the response schema,
      and the abstention threshold that decides what may be asserted;
    - how that output is judged — tolerances, criticality, strata, tier,
      consumers, the rollback plan, the release sequence.

    Predictions are bound to the first. Binding them to the whole spec
    would mean a pure policy change — raising a tier, naming another
    consumer — invalidated model output that is byte-for-byte what the new
    policy would produce, forcing a paid re-run to learn nothing. Scores
    still carry the full ``ReleaseIdentity``, because *judging* the same
    predictions under a new policy does require re-scoring; that is
    arithmetic over records already held.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inference_digest: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)


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
    #: Monotonic release counter, incremented whenever any part of the
    #: release changes. It is what lets the pipeline tell *newer* from
    #: merely *different*: identity alone cannot, and without an ordering
    #: an old job resuming late would treat newer rows as unprocessed and
    #: overwrite them with its own older output.
    release_sequence: int = Field(ge=0)
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
        # All three tables have different physical roles, and Spark
        # identifiers are case-insensitive — so `Main.X.Y` and `main.x.y`
        # are one table. Pointing two roles at it makes the target DDL
        # operate on the source, or turns the metadata DDL into a no-op
        # followed by a MERGE against an incompatible schema.
        roles = {
            "source_table": self.source_table,
            "target_table": self.target_table,
            "run_metadata_table": self.run_metadata_table,
        }
        seen_tables: dict[str, str] = {}
        for role, table in roles.items():
            key = table.casefold()
            if key in seen_tables:
                raise ValueError(
                    f"{role} and {seen_tables[key]} both name {table!r} "
                    "(table names are case-insensitive); each role needs its "
                    "own table"
                )
            seen_tables[key] = role
        if self.use_tier == UseTier.CONSEQUENTIAL and not self.rollback_plan:
            raise ValueError(
                "tier 1 (consequential) requires a documented rollback_plan"
            )
        self._reject_expanded_name_collisions()
        return self

    def _reject_expanded_name_collisions(self) -> None:
        """Unique field names are not enough — the *expanded* names must be
        unique too.

        Each field becomes three target columns, so distinct fields can
        still collide: a field named ``error`` produces ``ai_error``, which
        is a provenance column, and fields ``x`` and ``x_confidence`` both
        produce ``ai_x_confidence``. Catch it here, at the configuration
        boundary, rather than as a confusing SQL error or — worse — a
        silently overwritten provenance column.

        Checking the columns is enough to cover the response schema too:
        the ``ai_`` prefix is injective and every response key maps onto a
        checked column, so a duplicate response key cannot exist without a
        duplicate column.

        Comparison is case-insensitive because Spark SQL identifiers are:
        fields ``x`` and ``X`` are two distinct names here but one column
        there.
        """
        seen: dict[str, str] = {}

        def claim(name: str, owner: str) -> None:
            key = name.casefold()
            if key in seen:
                raise ValueError(
                    f"name collision: {owner} and {seen[key]} both produce "
                    f"{name!r} (SQL identifiers are case-insensitive). "
                    "Rename one of the fields."
                )
            seen[key] = owner

        claim(self.key_column, "the key column")
        # The document column is selected alongside the key and strata in
        # the pending CTE, so it has to be distinct from them or the
        # projection carries the same physical column twice and every
        # later reference to it is ambiguous.
        claim(self.document_column, "the document column")
        for column in self.strata:
            claim(column, f"stratum column {column!r}")
        for name, _ in PROVENANCE_COLUMNS:
            claim(name, "a reserved provenance column")
        for field in self.fields:
            for column in _generated_column_names(field.name):
                claim(column, f"field {field.name!r}")

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

    @property
    def release(self) -> ReleaseIdentity:
        """The release this spec describes; stamped onto its scores."""
        return ReleaseIdentity(
            spec_digest=self.spec_digest,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
        )

    @property
    def inference_digest(self) -> str:
        """Digest of only the parts that determine the model's output.

        Deliberately excludes tolerances, criticality, strata, tier,
        consumers, tables, cost ceiling, and release sequence: none of
        them change a single character the model returns.
        """
        canonical = json.dumps(
            {
                "endpoint": self.endpoint,
                "model_version": self.model_version,
                "prompt_version": self.prompt_version,
                "abstain_threshold": self.abstain_threshold,
                # The response schema is built from these, so they are part
                # of the request the model actually sees.
                "fields": [
                    {"name": field.name, "description": field.description}
                    for field in self.fields
                ],
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def inference(self) -> InferenceIdentity:
        """What produced a prediction; stamped onto every record."""
        return InferenceIdentity(
            inference_digest=self.inference_digest,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
        )

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> BatchInferenceSpec:
        return cls.model_validate(yaml.safe_load(text))


# ---------------------------------------------------------------------------
# 2. Estimate — convert "we discovered a bill" into "we approved a budget"
# ---------------------------------------------------------------------------


class CostEstimate(BaseModel):
    """Projected spend for the full run, computed from a small probe.

    ``release`` is what the estimate was measured for. A longer prompt, a
    different model, or a repriced endpoint is a different budget, so an
    estimate cannot be carried across releases — reusing one would let a
    release clear the declared ceiling on another release's assumptions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    release: ReleaseIdentity
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
        release=spec.release,
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


def source_key_check_sql(spec: BatchInferenceSpec) -> str:
    """Count null and duplicated source keys — run this before spending."""
    return f"""SELECT
  count_if({spec.key_column} IS NULL) AS null_keys,
  count(*) - count(DISTINCT {spec.key_column})
    - count_if({spec.key_column} IS NULL) AS duplicate_keys
FROM {spec.source_table}"""


def require_usable_source_keys(
    spec: BatchInferenceSpec, null_count: int, duplicate_count: int = 0
) -> None:
    """Refuse to run when the source keys cannot carry the landing contract.

    Every idempotence guarantee in this pipeline rests on key equality,
    and ``NULL = NULL`` is not true. A null-keyed row therefore never
    matches the restart anti-join — so it is re-inferred, and paid for, on
    every run — and never matches the MERGE, so each run inserts another
    copy of it. The restart guarantee does not degrade gracefully here; it
    silently stops holding while appearing to work.

    A duplicated key breaks the same contract from the other side: both
    rows are sent to the paid endpoint, the MERGE then finds two source
    rows for one target row and fails outright, and against an empty
    target it lands two rows under one key — so "one current row per key",
    which every provenance join assumes, was never true.

    The check is deliberately a refusal rather than a filter. Skipping
    those rows would quietly shrink coverage of the very table the gate
    just certified; the key contract is broken and someone has to fix it
    upstream.
    """
    if null_count:
        raise UnusableSourceKeys(
            f"{null_count} row(s) in {spec.source_table} have a null "
            f"{spec.key_column}. Key equality drives both the restart "
            "anti-join and the MERGE, so those rows would be re-inferred "
            "and re-inserted on every run. Give them keys upstream, or "
            "narrow the source to rows that have one."
        )
    if duplicate_count:
        raise UnusableSourceKeys(
            f"{duplicate_count} duplicate {spec.key_column} value(s) in "
            f"{spec.source_table}. The MERGE cannot resolve two source rows "
            "onto one target row, and one row per key is what every "
            "provenance join assumes. De-duplicate upstream, or add the "
            "column that makes the key unique."
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
    """A proportion with the uncertainty its sample size actually supports.

    ``point``, ``lower`` and ``upper`` are *derived*, so they are verified
    against the counts rather than trusted. This model is a persisted
    evidence boundary — gate reports round-trip through MLflow as JSON —
    and the gate reads ``lower`` directly, so evidence reconstructed with
    `successes=0, lower=1.0` would otherwise adopt a release on numbers
    nothing produced.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    successes: int = Field(ge=0)
    trials: int = Field(gt=0)
    confidence: float = Field(gt=0.5, lt=1.0)
    point: float = Field(ge=0.0, le=1.0)
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _bounds_match_the_counts(self) -> ConfidenceInterval:
        if self.successes > self.trials:
            raise ValueError("successes cannot exceed trials")
        expected_point, expected_lower, expected_upper = _wilson_bounds(
            self.successes, self.trials, self.confidence
        )
        for name, given, expected in (
            ("point", self.point, expected_point),
            ("lower", self.lower, expected_lower),
            ("upper", self.upper, expected_upper),
        ):
            if not math.isclose(given, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"{name} {given!r} does not match the Wilson value "
                    f"{expected:.12g} implied by {self.successes}/"
                    f"{self.trials} at confidence {self.confidence}"
                )
        return self


def _z_value(confidence: float) -> float:
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def _wilson_bounds(
    successes: int, trials: int, confidence: float
) -> tuple[float, float, float]:
    """(point, lower, upper) for a Wilson score interval.

    Split out from ``wilson_interval`` so ``ConfidenceInterval`` can check
    a reconstructed interval against the same arithmetic that produced it,
    without recursing through the model it is validating.
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
    return p_hat, max(0.0, centre - half_width), min(1.0, centre + half_width)


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> ConfidenceInterval:
    """Wilson score interval for a binomial proportion.

    Preferred over the naive normal approximation because it stays inside
    [0, 1], behaves sensibly at 0 and 100 percent observed success, and is
    accurate at the small per-stratum sample sizes human labelling budgets
    actually produce.
    """
    point, lower, upper = _wilson_bounds(successes, trials, confidence)
    return ConfidenceInterval(
        successes=successes,
        trials=trials,
        confidence=confidence,
        point=point,
        lower=lower,
        upper=upper,
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
    """One labelled sample row: gold values, model values, abstentions.

    ``inference`` records what actually produced ``predicted``. It is set
    where the prediction is made, not where it is scored: taking it from
    the spec handed to ``score_extraction`` would let v1 predictions
    certify themselves as v2 evidence.

    It is the *inference* identity rather than the full release, so the
    same predictions can be re-judged under a changed policy — a new tier,
    another consumer — without paying to regenerate output that would come
    back identical.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stratum: str
    inference: InferenceIdentity
    gold: Mapping[str, str | int | float | None]
    predicted: Mapping[str, str | int | float | None]
    abstained: frozenset[str] = frozenset()


class FieldStratumScore(BaseModel):
    """Precision/recall evidence for one field within one stratum.

    ``precision`` is None when the model asserted nothing in the stratum
    and ``recall`` is None when gold has no values there — "no evidence",
    which the gate treats as inconclusive, never as a pass.

    ``release`` and ``confidence`` record *what* the evidence measured and
    *at what confidence level*, so the gate can refuse evidence produced
    for a different release or computed at a different level than the spec
    declares. ``sample_strata`` lists every stratum the sample covered, so
    the gate can tell a complete evidence set from a filtered one — the
    worst-stratum rule is only as good as the strata it is given.

    On the ``WEIGHTED`` row the counts remain raw sample totals, while the
    intervals are the population-weighted estimate expressed through an
    effective sample size — so ``precision.trials`` there is that
    effective n, not a row count.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    stratum: str
    release: ReleaseIdentity
    confidence: float = Field(gt=0.5, lt=1.0)
    sample_strata: tuple[str, ...] = Field(min_length=1)
    n_rows: int = Field(ge=0)
    n_gold: int = Field(ge=0)
    n_asserted: int = Field(ge=0)
    n_correct: int = Field(ge=0)
    precision: ConfidenceInterval | None
    recall: ConfidenceInterval | None
    abstention_rate: float = Field(ge=0.0, le=1.0)


def apply_abstention_policy(
    spec: BatchInferenceSpec,
    values: Mapping[str, object],
    confidences: Mapping[str, float | None],
    model_abstained: Sequence[str] | None = None,
) -> tuple[dict[str, object], frozenset[str]]:
    """Decide what a response is actually allowed to assert.

    The model can contradict itself — return a value *and* list the field
    as abstained — and it can return a value below the confidence
    threshold it was told to respect. Neither may reach a consumer: a
    value the evaluation treated as an abstention has never been through
    the precision gate, so landing it would put unmeasured output in the
    table. The rule is therefore applied identically here and in the
    generated SQL, and both evaluation and execution call it, so what was
    measured is exactly what lands.

    Returns the permitted values (abstained ones nulled) and the effective
    abstention set. A field that is simply absent from the document stays
    null without being an abstention — that is a legitimate answer, not a
    declined one.

    A confidence outside [0, 1] is malformed rather than high: structured
    output constrains the JSON *type*, while "from 0 to 1" is only prose in
    the field description, so a model can return 5. Such a value is
    declined, not trusted — an unusable confidence is exactly the case
    where guessing is least defensible.
    """
    permitted: dict[str, object] = {}
    abstained: set[str] = set()
    listed = set(model_abstained or ())
    for field in spec.fields:
        value = values.get(field.name)
        confidence = confidences.get(field.name)
        unusable_confidence = value is not None and not (
            confidence is not None and spec.abstain_threshold <= confidence <= 1.0
        )
        if field.name in listed or unusable_confidence:
            abstained.add(field.name)
            permitted[field.name] = None
        else:
            permitted[field.name] = value
    return permitted, frozenset(abstained)


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
    release: ReleaseIdentity,
    sample_strata: tuple[str, ...],
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
        release=release,
        confidence=confidence,
        sample_strata=sample_strata,
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


def _weighted_interval(
    per_stratum: Sequence[tuple[float, int, int]],
    confidence: float,
) -> ConfidenceInterval | None:
    """Combine per-stratum rates into one population-level interval.

    ``per_stratum`` is ``(population_weight, successes, denominator)`` per
    stratum. The estimate is the standard stratified one,
    ``p̂ = Σ wₕ p̂ₕ`` with ``wₕ`` normalised population weights, and its
    variance is ``V = Σ wₕ² p̂ₕ(1-p̂ₕ)/nₕ``.

    Rather than report a Wald interval — which misbehaves exactly where
    this pipeline lives, near a rate of 1 and at small n — the variance is
    converted to an *effective sample size*, ``n_eff = p̂(1-p̂)/V``, and the
    Wilson interval is taken at ``(p̂·n_eff, n_eff)``. That is the usual
    design-effect correction: it keeps the interval inside [0, 1], stays
    sensible at 100 percent observed success, and shrinks the interval by
    exactly the amount the unequal sampling probabilities justify. When
    every stratum rate is 0 or 1 the variance vanishes, and the design's
    own effective size ``1/Σ(wₕ²/nₕ)`` is used instead.
    """
    usable = [(w, s, n) for w, s, n in per_stratum if n > 0 and w > 0]
    if not usable:
        return None
    total_weight = sum(weight for weight, _, _ in usable)
    normalised = [(weight / total_weight, s, n) for weight, s, n in usable]

    estimate = sum(weight * (s / n) for weight, s, n in normalised)
    variance = sum(
        weight**2 * (s / n) * (1.0 - s / n) / n for weight, s, n in normalised
    )
    design_size = 1.0 / sum(weight**2 / n for weight, _, n in normalised)
    if variance > 0.0 and 0.0 < estimate < 1.0:
        effective = estimate * (1.0 - estimate) / variance
    else:
        effective = design_size
    # Never claim more evidence than was actually labelled.
    effective = min(effective, float(sum(n for _, _, n in usable)))
    trials = max(1, round(effective))
    successes = min(trials, max(0, round(estimate * trials)))
    return wilson_interval(successes, trials, confidence)


def score_extraction(
    records: Sequence[EvaluationRecord],
    spec: BatchInferenceSpec,
    stratum_population: Mapping[str, int],
) -> tuple[FieldStratumScore, ...]:
    """Score every field in every stratum, plus one population-weighted row.

    Precision — of the values the model *asserted*, how many were right —
    and recall — of the values that truly exist, how many were correctly
    produced — are reported separately because extraction fails in two
    different ways: hallucinating and missing. A single accuracy figure
    hides whichever failure mode your consumers care about. Abstentions
    count against recall (the value exists and was not produced) but not
    against precision (nothing false was asserted): the abstention path
    converts silent errors into visible work.

    ``stratum_population`` is the row count of each stratum in the *source
    table* — the same mapping handed to ``allocate_stratified_sample``. It
    is required because the sample is not a simple random sample: rare
    strata are deliberately over-sampled, so pooling the sample's rows
    would estimate the population rate of nothing. The ``WEIGHTED`` row
    re-weights each stratum back to its true share, which is what the gate
    uses for medium- and low-criticality fields.

    The fields, the confidence level, and the release stamp all come from
    ``spec``. Passing the spec rather than a loose field list and a
    defaulted confidence is deliberate: it makes it impossible to compute
    intervals at one confidence level and gate them at another, or to hand
    the gate evidence from a different prompt or model version.
    """
    if not records:
        raise ValueError("cannot score an empty sample")
    confidence = spec.confidence_level
    release = spec.release
    # The records say what produced them; the spec says what is being
    # gated. If the *inference* identities disagree, scoring would mint
    # evidence that certifies itself — the one thing a stamp exists to
    # prevent. Policy-only differences are fine and deliberately so: the
    # same predictions may be re-judged under a new tier or tolerance.
    inference = spec.inference
    for record in records:
        if record.inference != inference:
            raise EvidenceMismatch(
                f"a sample row in stratum {record.stratum!r} was produced by "
                f"prompt {record.inference.prompt_version} / model "
                f"{record.inference.model_version} (inference digest "
                f"{record.inference.inference_digest[:12]}…), but is being "
                f"scored as prompt {inference.prompt_version} / model "
                f"{inference.model_version} (inference digest "
                f"{inference.inference_digest[:12]}…). Re-run inference for "
                "the release being gated."
            )
    by_stratum: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        by_stratum.setdefault(record.stratum, []).append(record)

    unknown = sorted(set(by_stratum) - set(stratum_population))
    if unknown:
        raise ValueError(
            f"sampled strata {unknown} have no population count; the "
            "population-weighted estimate cannot be computed without one"
        )
    unmeasured = sorted(
        stratum
        for stratum, count in stratum_population.items()
        if count > 0 and stratum not in by_stratum
    )
    if unmeasured:
        raise ValueError(
            f"strata {unmeasured} exist in the population but not in the "
            "sample; there is no evidence about them. Sample them or "
            "restate the population deliberately."
        )

    sample_strata = tuple(sorted(by_stratum))
    scores: list[FieldStratumScore] = []
    for field in spec.fields:
        per_stratum = [
            _score_group(
                field, stratum, by_stratum[stratum], confidence, release, sample_strata
            )
            for stratum in sample_strata
        ]
        scores.extend(per_stratum)

        # Weight each stratum by the population share of the rows that
        # actually enter the metric: for precision that is the estimated
        # population of asserted values, for recall the estimated
        # population of true values. Both collapse to the plain population
        # share when the rates are equal across strata.
        precision_inputs = []
        recall_inputs = []
        for score in per_stratum:
            population = float(stratum_population[score.stratum])
            share = population / score.n_rows if score.n_rows else 0.0
            precision_inputs.append(
                (share * score.n_asserted, score.n_correct, score.n_asserted)
            )
            recall_inputs.append((share * score.n_gold, score.n_correct, score.n_gold))
        # The abstention rate is a population rate too, so it is weighted
        # the same way rather than pooled.
        measured = [score for score in per_stratum if score.n_rows]
        population_total = sum(stratum_population[s.stratum] for s in measured)
        weighted_abstention = (
            sum(
                stratum_population[score.stratum] * score.abstention_rate
                for score in measured
            )
            / population_total
            if population_total
            else 0.0
        )
        scores.append(
            FieldStratumScore(
                field=field.name,
                stratum=WEIGHTED,
                release=release,
                confidence=confidence,
                sample_strata=sample_strata,
                n_rows=sum(score.n_rows for score in per_stratum),
                n_gold=sum(score.n_gold for score in per_stratum),
                n_asserted=sum(score.n_asserted for score in per_stratum),
                n_correct=sum(score.n_correct for score in per_stratum),
                precision=_weighted_interval(precision_inputs, confidence),
                recall=_weighted_interval(recall_inputs, confidence),
                abstention_rate=weighted_abstention,
            )
        )
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
    """Judge one metric in one group. Returns (decision, reason, lower, point).

    Three outcomes, in the order they are decided:

    1. ``adopt`` — the interval's lower bound clears the tolerance.
    2. ``reject`` — the interval's *upper* bound is below the tolerance.
       Even the optimistic end of the estimate misses the bar, so this is
       a demonstrated failure and more labelling will not rescue it. This
       test comes before the power check on purpose: 0/30 and 30/30 are
       both "too small to pass", but only one of them is uninformative.
    3. ``inconclusive`` — the interval straddles the tolerance and the
       group is too small for any result to clear it (a flawless n/n tops
       out at a lower bound of n/(n+z²)). The evidence is insufficient;
       label more rows. The model has not been shown to be bad.

    Anything else — straddling with adequate power — is a ``reject``: the
    run did not demonstrate the declared rate, and the default is not to
    ship.
    """
    where = (
        "the population-weighted estimate"
        if score.stratum == WEIGHTED
        else f"stratum {score.stratum!r}"
    )
    if interval is None:
        return (
            GateDecision.INCONCLUSIVE,
            f"no {metric} evidence for {where}",
            None,
            None,
        )
    if interval.lower >= required:
        return (GateDecision.ADOPT, None, interval.lower, interval.point)
    if interval.upper < required:
        return (
            GateDecision.REJECT,
            (
                f"{metric} for {where}: the whole interval "
                f"[{interval.lower:.4f}, {interval.upper:.4f}] sits below the "
                f"required {required:.4f} ({interval.successes}/"
                f"{interval.trials}) — a demonstrated failure, not a small "
                "sample; more labelling will not change it"
            ),
            interval.lower,
            interval.point,
        )
    best_possible = wilson_interval(
        interval.trials, interval.trials, interval.confidence
    ).lower
    if best_possible < required:
        return (
            GateDecision.INCONCLUSIVE,
            (
                f"{metric} for {where} has only {interval.trials} labelled "
                f"rows; even a flawless sample tops out at a lower bound of "
                f"{best_possible:.4f} < {required:.4f} — insufficient "
                "evidence, label more rows"
            ),
            interval.lower,
            interval.point,
        )
    return (
        GateDecision.REJECT,
        (
            f"{metric} for {where}: lower bound "
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
    # the one stratum that matters, so no average appears here at all.
    # medium / low → the population-weighted estimate decides. Never the
    # raw pool of a stratified sample, which over-weights rare strata and
    # so estimates nothing about the population.
    if field.criticality == Criticality.HIGH:
        considered = [score for score in scores if score.stratum != WEIGHTED]
    else:
        considered = [score for score in scores if score.stratum == WEIGHTED]
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


def require_matching_evidence(
    spec: BatchInferenceSpec,
    scores: Sequence[FieldStratumScore],
) -> None:
    """Refuse evidence that was not produced for exactly this release, or
    that does not cover everything the gate is supposed to judge.

    Without the release check, ``evaluate_gate(spec_v2, scores_from_v1)``
    would stamp v2's digest onto v1's numbers and ``require_executable``
    would then happily accept it — an unvalidated release executing on an
    earlier release's evidence.

    Without the completeness check, a filtered score list would quietly
    weaken the gate instead of failing it: drop the failing stratum's rows
    and the worst-stratum rule has nothing bad left to find. Evidence must
    therefore cover every field in the spec and every stratum the sample
    measured.
    """
    if not scores:
        raise EvidenceMismatch("no scores supplied; a gate needs evidence")
    expected = spec.release
    for score in scores:
        if score.release != expected:
            raise EvidenceMismatch(
                f"score for {score.field!r}/{score.stratum!r} was produced for "
                f"prompt {score.release.prompt_version} / model "
                f"{score.release.model_version} (spec digest "
                f"{score.release.spec_digest[:12]}…), but this gate is for "
                f"prompt {expected.prompt_version} / model "
                f"{expected.model_version} (spec digest "
                f"{expected.spec_digest[:12]}…). Re-score the sample against "
                "the release being gated."
            )
        if score.confidence != spec.confidence_level:
            raise EvidenceMismatch(
                f"score for {score.field!r}/{score.stratum!r} was computed at "
                f"confidence {score.confidence}, but the spec declares "
                f"{spec.confidence_level}. Re-score at the declared level."
            )
        # The intervals are what the gate actually reads, and they carry
        # their own confidence. Checking only the score's outer label would
        # let evidence relabelled to 99% be gated on bounds computed at
        # 95% — the very substitution the outer check exists to stop.
        for metric, interval in (
            ("precision", score.precision),
            ("recall", score.recall),
        ):
            if interval is not None and interval.confidence != spec.confidence_level:
                raise EvidenceMismatch(
                    f"the {metric} interval for {score.field!r}/"
                    f"{score.stratum!r} was computed at confidence "
                    f"{interval.confidence}, but the spec declares "
                    f"{spec.confidence_level}. Its bounds do not mean what "
                    "the score claims; re-score at the declared level."
                )
        # The intervals are what the gate actually reads, and each carries
        # its own confidence. Checking only the score's outer label would
        # let relabelled or hand-assembled evidence present 95% bounds as
        # 99% ones — the same false adoption the release check closes.
        for metric, interval in (
            ("precision", score.precision),
            ("recall", score.recall),
        ):
            if interval is not None and interval.confidence != spec.confidence_level:
                raise EvidenceMismatch(
                    f"the {metric} interval for {score.field!r}/"
                    f"{score.stratum!r} was computed at confidence "
                    f"{interval.confidence}, but the spec declares "
                    f"{spec.confidence_level}. Its bounds do not mean what "
                    "the score claims; re-score at the declared level."
                )

    manifests = {score.sample_strata for score in scores}
    if len(manifests) != 1:
        raise EvidenceMismatch(
            f"scores disagree about which strata the sample covered: "
            f"{sorted(manifests)}. They did not come from one scoring run."
        )
    (manifest,) = manifests
    required_groups = set(manifest) | {WEIGHTED}
    present: dict[str, set[str]] = {}
    for score in scores:
        present.setdefault(score.field, set()).add(score.stratum)
    for field in spec.fields:
        groups = present.get(field.name, set())
        missing = sorted(required_groups - groups)
        if missing:
            raise EvidenceMismatch(
                f"evidence for field {field.name!r} is incomplete: no scores "
                f"for {missing}. The sample covered {list(manifest)}, and a "
                "worst-stratum gate cannot be applied to a filtered set."
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

    Evidence is checked before it is judged: every score must carry this
    spec's release stamp and its declared confidence level. A prompt or
    model change is a new release, so evidence from the previous one is
    not evidence about this one — and intervals computed at 95% cannot
    satisfy a spec that declared 99%.
    """
    require_matching_evidence(spec, scores)
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

    Prefer an accountable role or group over an individual's email. The
    value is retained in the gate artifact and the run metadata table
    (both access controlled) and deliberately never reaches a tag.
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


def response_format(spec: BatchInferenceSpec) -> dict:
    """The ``responseFormat`` JSON schema for ``ai_query`` structured output.

    Databricks structured outputs support only a subset of JSON Schema:
    no ``anyOf``/``oneOf``/``allOf``, no ``pattern``, and the *only*
    permitted type union is ``[type, "null"]`` for nullable fields — which
    is exactly what the abstention path needs. ``strict`` sits inside
    ``json_schema`` (verified against the structured-outputs documentation;
    re-check when upgrading, this surface has changed before).

    ``additionalProperties: false`` closes the object. The Databricks
    documentation neither requires nor mentions it, but the endpoint is
    OpenAI-compatible and strict mode there expects a closed schema — and
    closed is what is wanted regardless: an extraction contract should not
    silently accept fields nobody declared.
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
                "additionalProperties": False,
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
    """DDL for the output table, provenance columns included from birth.

    ``IF NOT EXISTS`` means this creates the table once and never changes
    it. When the field set changes — which is a new release — the existing
    table needs migrating; see ``plan_target_migration``.
    """
    body = ",\n".join(f"  {name} {sql_type}" for name, sql_type in target_columns(spec))
    return f"CREATE TABLE IF NOT EXISTS {spec.target_table} (\n{body}\n)"


class TargetMigration(BaseModel):
    """What an existing target table needs before this release can land.

    ``add`` is applied automatically; ``blocking`` is not. ``UPDATE SET *``
    and ``INSERT *`` expand over the *target's* columns and require every
    one of them to resolve in the source, so any column the release no
    longer produces stops the MERGE at analysis. ``stale`` are the
    ``ai_``-prefixed ones — output from a previous release — and
    ``foreign`` are columns this pipeline never owned.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    add: tuple[tuple[str, str], ...] = ()
    stale: tuple[str, ...] = ()
    foreign: tuple[str, ...] = ()
    statements: tuple[str, ...] = ()

    @property
    def blocking(self) -> tuple[str, ...]:
        """Columns a human must resolve before the release can execute."""
        return self.stale + self.foreign

    @property
    def required(self) -> bool:
        return bool(self.add or self.blocking)


def plan_target_migration(
    spec: BatchInferenceSpec, existing_columns: Sequence[str]
) -> TargetMigration:
    """Diff the target table against the columns this release produces.

    Adding or removing a field changes the MERGE source, and
    ``CREATE TABLE IF NOT EXISTS`` will not touch a table that already
    exists. Without this step, adding a field makes ``INSERT *`` fail
    analysis against the old schema, and removing one leaves a column that
    the current release no longer writes — so it keeps serving a previous
    release's values under an ``ai_`` name, which is exactly the
    provenance failure the three-layer scheme exists to prevent.

    Columns to add come back as ready statements. Columns gone stale are
    *reported, not dropped*: deleting a column is destructive and losing
    the old values may itself be a governance event, so a human decides.
    Comparison is case-insensitive, like Spark identifiers.
    """
    existing = {column.casefold() for column in existing_columns}
    expected = target_columns(spec)
    expected_names = {name.casefold() for name, _ in expected}

    add = tuple(
        (name, sql_type)
        for name, sql_type in expected
        if name.casefold() not in existing
    )
    unexpected = [
        column for column in existing_columns if column.casefold() not in expected_names
    ]
    stale = tuple(
        column
        for column in unexpected
        if column.casefold().startswith(AI_COLUMN_PREFIX)
    )
    foreign = tuple(column for column in unexpected if column not in stale)
    statements: list[str] = []
    if add:
        columns = ", ".join(f"{name} {sql_type}" for name, sql_type in add)
        statements.append(f"ALTER TABLE {spec.target_table} ADD COLUMNS ({columns})")
    for column in stale:
        # Emitted for a human to run deliberately, never executed for them.
        statements.append(
            f"-- review before running: {spec.target_table}.{column} is no "
            "longer produced by this release and now holds values from an "
            f"earlier one\n-- ALTER TABLE {spec.target_table} DROP COLUMN {column}"
        )
    for column in foreign:
        statements.append(
            f"-- review before running: {spec.target_table}.{column} is not "
            "produced by this pipeline at all; move it to its own table or "
            f"drop it\n-- ALTER TABLE {spec.target_table} DROP COLUMN {column}"
        )
    return TargetMigration(
        add=add, stale=stale, foreign=foreign, statements=tuple(statements)
    )


def require_migrated_target(
    spec: BatchInferenceSpec, existing_columns: Sequence[str]
) -> TargetMigration:
    """Refuse to execute against a target that still has unresolved columns.

    ``UPDATE SET *`` / ``INSERT *`` expand over the target's columns and
    need each one to resolve in the source, so a leftover column does not
    degrade the run — it stops the MERGE at analysis. Failing here, with
    the statements to fix it, beats failing later inside a SQL error that
    does not mention releases at all.

    Additive migration is returned for the caller to apply; anything
    destructive stays a human decision.
    """
    migration = plan_target_migration(spec, existing_columns)
    if migration.blocking:
        raise TargetSchemaMismatch(
            f"{spec.target_table} still has columns this release does not "
            f"produce: {list(migration.blocking)}. `INSERT *` cannot resolve "
            "them, so the run would fail at analysis. Resolve them "
            "deliberately first:\n"
            + "\n".join(
                statement
                for statement in migration.statements
                if statement.startswith("--")
            )
        )
    return migration


def target_columns(spec: BatchInferenceSpec) -> tuple[tuple[str, str], ...]:
    """(name, SQL type) for every target column, in canonical order.

    One definition serves both the DDL and the INSERT column list so the
    two can never drift apart.
    """
    columns: list[tuple[str, str]] = [(spec.key_column, "STRING")]
    columns.extend((column, "STRING") for column in spec.strata)
    for field in spec.fields:
        value, confidence, abstained = _generated_column_names(field.name)
        columns.append((value, "STRING"))
        columns.append((confidence, "DOUBLE"))
        columns.append((abstained, "BOOLEAN"))
    columns.extend(PROVENANCE_COLUMNS)
    return tuple(columns)


def build_execute_sql(
    spec: BatchInferenceSpec,
    *,
    run_id: str,
    prompt_sql: str,
) -> str:
    """The full-table execute statement: restartable *and* release-aware.

    - ``prompt_sql`` is a SQL expression producing the request string (the
      notebook builds it with ``concat`` from an escaped instruction
      literal and the document column).
    - **Restart**: the anti-join drops rows this release has already
      landed, so a re-run after a partial failure finishes the job instead
      of paying for inference twice. A million-row job will fail partway
      at some point, and this is what makes that boring. Current guidance
      is to submit the remaining set as one query — AI Functions manage
      parallelization and retries — rather than hand-chunking it.
    - **Release awareness**: that anti-join matches on the key *and* the
      full release identity — spec digest, model version, prompt version.
      Any of those changing is a new application release, so rows carrying
      an older one must be reprocessed. A key-only anti-join would skip
      every previously landed row and let a newly gated release report
      success while the table still held the old release's values and
      provenance; matching on model and prompt alone would do the same
      whenever the spec changed while those two labels stayed put (new
      abstention threshold, changed field set, edited instructions). The
      MERGE then updates stale rows in place and inserts genuinely new
      keys, keeping one current row per key.

    - **Newer, not merely different.** Release identity says whether two
      runs differ; it cannot say which is later. Without an ordering, an
      old job resuming after a newer release has landed would see every
      newer row as unprocessed, re-infer it, and the key-only MERGE would
      write the old model's values back over the new ones — a silent
      rollback of a production table from a delayed retry or an
      overlapping deploy. So ``release_sequence`` orders releases: the
      anti-join also treats a strictly newer sequence as done, and the
      MERGE updates a row only when its sequence is not being lowered.

      The digest covers the spec, not this module's code. Bump
      ``spec_version`` when a code change alters what the pipeline
      produces — that is what makes the release identity honest.
    - ``failOnError => false`` keeps one poisoned document from killing
      the run; its error message lands in ``ai_error`` and the row flows
      to the exception queue instead of blocking everything else. The
      struct field is ``errorMessage`` (``response`` is null on failure).
    """
    # The abstention rule from `apply_abstention_policy`, expressed in SQL
    # so that what the gate measured is exactly what lands: a field the
    # model listed as abstained, or answered below the declared threshold,
    # is nulled rather than written. Landing such a value would put output
    # the precision gate never saw into a consumer's table.
    threshold = spec.abstain_threshold
    value_lines = []
    abstained_items = []
    for field in spec.fields:
        value, confidence, abstained = _generated_column_names(field.name)
        literal = sql_string_literal(field.name)
        # Below the threshold *or* outside [0, 1]: the schema constrains
        # the JSON type, not the range, so a confidence of 5 is malformed
        # output rather than a very confident answer.
        confidence_sql = f"coalesce(parsed.{field.name}_confidence, -1)"
        rule = (
            f"coalesce(array_contains(parsed.abstained_fields, {literal}), false) "
            f"OR (parsed.{field.name} IS NOT NULL "
            f"AND NOT ({confidence_sql} BETWEEN {threshold} AND 1))"
        )
        value_lines.append(
            f"    CASE WHEN {rule} THEN NULL ELSE parsed.{field.name} END AS {value}"
        )
        # Confidence is kept even when abstaining: it is diagnostic for
        # whoever works the exception queue, not an asserted value.
        value_lines.append(f"    parsed.{field.name}_confidence AS {confidence}")
        value_lines.append(f"    ({rule}) AS {abstained}")
        abstained_items.append(f"CASE WHEN {rule} THEN {literal} END")
    value_block = ",\n".join(value_lines)
    # Re-derive the landed list from the same rule, so a threshold
    # abstention the model did not declare still reaches the queue.
    effective_abstained = (
        "array_compact(array(\n      " + ",\n      ".join(abstained_items) + "\n    ))"
    )
    pending_strata = "".join(f", source.{column}" for column in spec.strata)
    plain_strata = "".join(f", {column}" for column in spec.strata)
    struct_type = sql_string_literal(response_struct_type(spec))
    digest_literal = sql_string_literal(spec.spec_digest)
    model_literal = sql_string_literal(spec.model_version)
    prompt_literal = sql_string_literal(spec.prompt_version)
    sequence = int(spec.release_sequence)
    return f"""MERGE INTO {spec.target_table} AS target
USING (
  WITH pending AS (
    SELECT source.{spec.key_column}, source.{spec.document_column}{pending_strata}
    FROM {spec.source_table} AS source
    LEFT ANTI JOIN {spec.target_table} AS done
      ON source.{spec.key_column} = done.{spec.key_column}
     AND (
       (
         done.ai_spec_digest = {digest_literal}
         AND done.ai_model_version = {model_literal}
         AND done.ai_prompt_version = {prompt_literal}
       )
       OR done.ai_release_sequence > {sequence}
     )
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
    {effective_abstained} AS ai_abstained_fields,
    parsed.abstain_reason AS ai_abstain_reason,
    error_message AS ai_error,
    {sql_string_literal(run_id)} AS ai_run_id,
    {digest_literal} AS ai_spec_digest,
    {model_literal} AS ai_model_version,
    {prompt_literal} AS ai_prompt_version,
    {sequence} AS ai_release_sequence,
    current_timestamp() AS ai_executed_at
  FROM parsed
) AS source
ON target.{spec.key_column} = source.{spec.key_column}
WHEN MATCHED AND target.ai_release_sequence <= source.ai_release_sequence
  THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *"""


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


def run_metadata_upsert_sql(
    spec: BatchInferenceSpec,
    report: GateReport,
    *,
    run_id: str,
    projected_cost_cad: float,
    target_table_version: int,
) -> str:
    """Write the run record, keyed on ``run_id`` so a retry cannot duplicate it.

    A plain INSERT would add a second row whenever the client lost the
    response and the cell was re-run. ``ai_run_id`` is the join key every
    landed row uses to reach this table, so a duplicate fans out every
    downstream join and can tie one run to two recorded table versions —
    it corrupts the provenance record rather than merely repeating it.
    """
    approved = sql_string_literal(report.approved_by) if report.approved_by else "NULL"
    return f"""MERGE INTO {spec.run_metadata_table} AS target
USING (
  SELECT
    {sql_string_literal(run_id)} AS run_id,
    {sql_string_literal(spec.name)} AS spec_name,
    {sql_string_literal(spec.spec_digest)} AS spec_digest,
    {sql_string_literal(spec.to_yaml())} AS spec_yaml,
    {int(spec.use_tier)} AS use_tier,
    {sql_string_literal(spec.endpoint)} AS endpoint,
    {sql_string_literal(spec.model_version)} AS model_version,
    {sql_string_literal(spec.prompt_version)} AS prompt_version,
    {sql_string_literal(report.decision.value)} AS gate_decision,
    {approved} AS approved_by,
    {float(projected_cost_cad)} AS projected_cost_cad,
    {sql_string_literal(spec.target_table)} AS target_table,
    {int(target_table_version)} AS target_table_version,
    current_timestamp() AS executed_at
) AS source
ON target.run_id = source.run_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *"""


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

    # The estimate is what authorised the spend, so it has to belong to
    # the release being recorded. A longer prompt is a different budget.
    if estimate.release != spec.release:
        raise EvidenceMismatch(
            f"the cost estimate was computed for prompt "
            f"{estimate.release.prompt_version} / model "
            f"{estimate.release.model_version} (spec digest "
            f"{estimate.release.spec_digest[:12]}…), but this run is "
            f"prompt {spec.release.prompt_version} / model "
            f"{spec.release.model_version} (spec digest "
            f"{spec.release.spec_digest[:12]}…). Re-estimate before "
            "authorising the spend."
        )

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
    # The approver identity stays out of tags. Tags are broadly readable,
    # get copied onto downstream objects, and carry no sensitive data by
    # platform rule — an individual's name or email is exactly what that
    # rule excludes. The audit trail lives in the gate_report.json artifact
    # above and in the run metadata table, both of which are access
    # controlled. Only whether an approval exists is tagged.
    mlflow.set_tags(
        {
            "gate_decision": report.decision.value,
            "human_approved": "yes" if report.approved_by else "no",
        }
    )
