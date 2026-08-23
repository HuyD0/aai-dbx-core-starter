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
from typing import Annotated, TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

#: A string that must carry actual content. `min_length=1` alone rejects
#: `""` and accepts `" "`, which is how a whitespace approver once
#: satisfied a named-approver check and a whitespace rollback plan once
#: satisfied tier 1. Surrounding whitespace is stripped as well, so two
#: spellings of one identifier cannot both exist.
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_ArtifactT = TypeVar("_ArtifactT", bound=BaseModel)

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
    #: The run whose *policy* governs this row, when that is not the run
    #: that produced it. A policy-only release re-stamps the spec digest
    #: without re-inferring, so `ai_run_id` still points at the run that
    #: made the values — which is true and worth keeping — while the gate
    #: and cost record that authorise the row today live under a
    #: different run. Both references are real; collapsing them would
    #: lose whichever one was overwritten.
    ("ai_policy_run_id", "STRING"),
    ("ai_spec_digest", "STRING"),
    ("ai_model_version", "STRING"),
    ("ai_prompt_version", "STRING"),
    ("ai_release_sequence", "BIGINT"),
    #: Delta version of the source this row was produced from, or -1 when
    #: the run was not pinned. `release_sequence` orders application
    #: releases and says nothing about which *data* a run saw, so two
    #: cycles of one release over different snapshots would otherwise tie
    #: — and the older one could overwrite the newer. Ordering is on the
    #: pair.
    ("ai_source_version", "BIGINT"),
    #: The snapshot whose *stratum labels* this row carries. Strata come
    #: from source columns rather than from the model, so their ordering
    #: has nothing to do with release identity and everything to do with
    #: which snapshot was read — they need their own column. Advancing
    #: `ai_release_sequence` instead would be actively wrong: it would
    #: make a row whose labels were resynced look as though the newer
    #: release had inferred it, and that release's anti-join would then
    #: skip the inference the row still needs.
    ("ai_strata_version", "BIGINT"),
    #: What produced the values: endpoint, model, prompt text, abstention
    #: threshold and field descriptions. Restart keys on this rather than
    #: on the spec digest, so a pure policy change — a tolerance, a tier,
    #: a consumer — re-scores the predictions it already has instead of
    #: paying to regenerate identical ones.
    ("ai_inference_digest", "STRING"),
    #: Digest of the document text this row was derived from, so an
    #: edit-in-place makes the row pending instead of silently stale.
    ("ai_source_digest", "STRING"),
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
    """Escape ``text`` as a single-quoted Spark SQL string literal.

    Newlines and tabs are escaped rather than embedded raw: prompt
    templates and serialised specs are multi-line, and a statement that
    breaks across lines mid-literal is unreadable in logs and error
    messages. Spark processes these escapes in string literals by default.
    """
    return (
        "'"
        + text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        + "'"
    )


class CostCeilingExceeded(RuntimeError):
    """Raised before execution when projected cost exceeds the ceiling."""


class GateNotPassed(RuntimeError):
    """Raised when execution is attempted without an adopting gate decision."""


class EvidenceMismatch(RuntimeError):
    """Raised when evidence does not belong to the release being gated."""


class TargetSchemaMismatch(RuntimeError):
    """Raised when the target table still holds columns a release dropped."""


class UnusableSourceRows(RuntimeError):
    """Raised when source rows cannot support restartable, idempotent landing."""


class UnknownAbstainedField(RuntimeError):
    """Raised when a response declines a field the spec never declared."""


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

    spec_digest: NonBlank
    model_version: NonBlank
    prompt_version: NonBlank


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

    inference_digest: NonBlank
    model_version: NonBlank
    prompt_version: NonBlank


class FieldSpec(BaseModel):
    """One extracted field and the error its consumers can tolerate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: NonBlank
    description: NonBlank
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

    spec_version: NonBlank = "1"
    name: NonBlank
    source_table: str
    target_table: str
    run_metadata_table: str
    document_column: NonBlank
    key_column: NonBlank
    use_tier: UseTier
    consumed_by: tuple[NonBlank, ...] = Field(min_length=1)
    fields: tuple[FieldSpec, ...] = Field(min_length=1)
    strata: tuple[NonBlank, ...] = Field(min_length=1)
    endpoint: NonBlank
    model_version: str
    prompt_version: str
    #: The instruction text itself, not just its label. The document is
    #: appended to it. It lives here so the identity can cover what the
    #: model actually received: ``prompt_version`` is a string the author
    #: types, and nothing stops it staying "1.0.0" across an edit.
    #: Not `NonBlank`: the template must have content, but it must not be
    #: stripped. Its trailing newline separates the instructions from the
    #: document appended after it, so trimming would silently change the
    #: request the model receives — and the inference digest with it.
    prompt_template: str = Field(min_length=1)
    #: Monotonic release counter, incremented whenever any part of the
    #: release changes. It is what lets the pipeline tell *newer* from
    #: merely *different*: identity alone cannot, and without an ordering
    #: an old job resuming late would treat newer rows as unprocessed and
    #: overwrite them with its own older output.
    release_sequence: int = Field(ge=0)
    cost_ceiling_cad: float = Field(gt=0.0)
    abstain_threshold: float = Field(ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    rollback_plan: NonBlank | None = None

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

    @field_validator("prompt_template")
    @classmethod
    def _prompt_has_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt_template must contain instructions")
        return value

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
                # The text, not the label: an edited prompt is different
                # output whether or not anyone bumped the version string.
                "prompt_template": self.prompt_template,
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

    ``source_snapshot`` is what it was measured *over*. Release identity
    alone does not pin the row count: the same spec estimated against
    Monday's snapshot and executed against Friday's clears a ceiling
    approved for a table that has since grown. Cost is a function of the
    data as much as of the release, so the estimate records both.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    release: ReleaseIdentity
    source_snapshot: SourceSnapshot | None = None
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

    @model_validator(mode="after")
    def _projection_follows_from_its_inputs(self) -> CostEstimate:
        """`within_ceiling` compares two numbers the artifact supplies.

        Every other piece of evidence here is recomputed rather than
        read; the budget was the one still taken at its word. A
        reconstructed estimate could keep an honest release, snapshot and
        token counts while declaring a zero projection, and the ceiling
        check would wave through a run those very token counts price in
        the millions.
        """
        expected = (
            self.safety_factor
            * self.row_count
            * (
                self.mean_input_tokens_per_row * self.cad_per_million_input_tokens
                + self.mean_output_tokens_per_row * self.cad_per_million_output_tokens
            )
            / 1_000_000.0
        )
        if not math.isclose(
            self.projected_cost_cad, expected, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(
                f"projected cost {self.projected_cost_cad} does not follow from "
                f"the recorded inputs, which price this run at {expected}"
            )
        return self


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
    source_snapshot: SourceSnapshot | None = None,
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
        source_snapshot=source_snapshot,
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


class SourceSnapshot(BaseModel):
    """The exact Delta version of the source that evidence describes.

    Population counts, the stratified sample, the gate and the cost
    estimate are all computed against the source *as it was* when the
    cycle started. A tier 1 spec then waits for a human to sign off, and
    that wait is where the table moves: rows land, and with them strata
    the sample never covered. Executing against "latest" would infer and
    write those rows on evidence that predates them, and spend past a
    ceiling approved for a smaller population.

    Recording the version turns "we evaluated this table" into "we
    evaluated these rows". Delta time travel then makes the run read them
    back exactly.

    Rows that arrive after the snapshot are not lost — they are simply
    not *this* run's work. The next cycle re-samples, re-gates and picks
    them up with evidence of their own.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    table: str
    version: int = Field(ge=0)


def _source_relation(spec: BatchInferenceSpec, snapshot: SourceSnapshot | None) -> str:
    """The source table, pinned to the evaluated version when one is given.

    ``VERSION AS OF`` reads a historical Delta commit, which requires the
    files backing it to still exist. ``VACUUM`` removes them after
    ``delta.deletedFileRetentionDuration`` (7 days by default), so a
    review that outlasts retention makes the pinned read fail rather than
    silently return current data — a loud failure that says re-gate, not
    a quiet one that says ship.
    """
    if snapshot is None:
        return spec.source_table
    if snapshot.table != spec.source_table:
        raise EvidenceMismatch(
            f"the snapshot describes {snapshot.table!r}, but this spec reads "
            f"{spec.source_table!r}. Evidence about one table cannot pin a "
            "run over another."
        )
    return f"{spec.source_table} VERSION AS OF {snapshot.version}"


def source_preflight_sql(
    spec: BatchInferenceSpec, snapshot: SourceSnapshot | None = None
) -> str:
    """Count unusable source rows — run this before spending.

    Reads the same pinned snapshot the run will process, so the counts
    describe exactly the rows that are about to be paid for.
    """
    source = _source_relation(spec, snapshot)
    return f"""SELECT
  count(*) AS row_count,
  count_if({spec.key_column} IS NULL) AS null_keys,
  count(*) - count(DISTINCT {spec.key_column})
    - count_if({spec.key_column} IS NULL) AS duplicate_keys,
  count_if({spec.document_column} IS NULL) AS null_documents
FROM {source}"""


class SourcePreflight(BaseModel):
    """Proof that a specific snapshot was measured and found usable.

    This is the module's one measurement boundary, and naming it is the
    point. Everything else here is recomputed from something else — the
    projection from its token counts, the weighted row from the physical
    rows, each verdict from the scores — but *how many rows a Delta
    version contains* cannot be derived from anything; it has to be
    counted by the warehouse. Recomputing the projection therefore proved
    only that the arithmetic was honest, not that ``row_count`` was: an
    estimate could halve the count and the price together and stay
    perfectly self-consistent while the pinned snapshot held a million
    rows.

    So the count enters once, here, from the query the preflight ran, and
    the estimate must agree with it. That does not make the number
    unforgeable — nothing in a pure-Python module can — but it collapses
    a diffuse "any caller may assert any row count" into a single object
    with a single origin, which is the difference between a boundary and
    a leak.

    Holding one is also proof the usability checks passed: it is returned
    by ``require_usable_source_rows`` and constructed nowhere else in the
    normal flow, so ``build_execute_sql`` can require it instead of
    hoping the caller ran the preflight first.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: SourceSnapshot
    row_count: int = Field(ge=0)
    #: Rows per stratum in that snapshot. The weighted gate row is an
    #: estimate *of this population*, and recomputing it from the weights
    #: the same report carries proves only that the report agrees with
    #: itself: claim 1,000,000 good rows and 1 failing one and evidence
    #: from a 50/50 sample adopts. Measured counts are the only thing
    #: that makes those weights mean anything.
    stratum_population: tuple[tuple[str, int], ...] = ()

    @model_validator(mode="after")
    def _strata_account_for_every_row(self) -> SourcePreflight:
        """The parts must add up to the whole they were measured beside.

        Names must be unique first, or the sum proves nothing: every
        consumer reads these pairs as a mapping, and
        ``(("standard", 50), ("standard", 50))`` sums to 100 while
        `dict()` collapses it to 50 — so a stratum could be omitted from
        the population entirely and the arithmetic would still balance.
        A tuple of pairs standing in for a mapping has to enforce the one
        property a mapping guarantees.

        A mapping that omits a group is not obviously wrong anywhere: the
        sample is drawn from it, the weighted row is computed from it, and
        the report/preflight comparison then agrees with itself — while
        the omitted rows are executed and priced without ever reaching the
        gate that certified the run.
        """
        if not self.stratum_population:
            return self
        names = [name for name, _ in self.stratum_population]
        if len(names) != len(set(names)):
            raise ValueError(
                "the measured population names a stratum more than once; "
                "read as a mapping the duplicates collapse, and the row "
                "count would balance against a population missing a group"
            )
        counts = [count for _, count in self.stratum_population]
        if any(count < 0 for count in counts):
            raise ValueError("a stratum cannot contain a negative number of rows")
        if sum(counts) != self.row_count:
            raise ValueError(
                f"the measured strata hold {sum(counts)} row(s) but the "
                f"snapshot holds {self.row_count}. Some rows belong to no "
                "stratum, so they would be processed without being gated."
            )
        return self


def require_usable_source_rows(
    spec: BatchInferenceSpec,
    null_count: int,
    duplicate_count: int = 0,
    null_documents: int = 0,
    *,
    snapshot: SourceSnapshot,
    row_count: int,
    stratum_population: Mapping[str, int],
) -> SourcePreflight:
    """Refuse to run when the source rows cannot carry the landing contract.

    Returns the ``SourcePreflight`` the builder requires, so that passing
    this check is something a caller can *hold* rather than something
    they are trusted to have done.

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

    A null *document* breaks it a third way, and quietly. ``sha2(NULL)``
    is NULL, so the row lands with a null ``ai_source_digest``; the
    restart anti-join then compares NULL to a real digest, which is
    unknown rather than true, and the row is selected again on every
    later run. The request itself is ``concat(prompt, NULL)`` — also NULL
    — so the pipeline pays an endpoint to process nothing, forever.

    The check is deliberately a refusal rather than a filter. Skipping
    those rows would quietly shrink coverage of the very table the gate
    just certified; the contract is broken and someone has to fix it
    upstream.
    """
    if null_count:
        raise UnusableSourceRows(
            f"{null_count} row(s) in {spec.source_table} have a null "
            f"{spec.key_column}. Key equality drives both the restart "
            "anti-join and the MERGE, so those rows would be re-inferred "
            "and re-inserted on every run. Give them keys upstream, or "
            "narrow the source to rows that have one."
        )
    if null_documents:
        raise UnusableSourceRows(
            f"{null_documents} row(s) in {spec.source_table} have a null "
            f"{spec.document_column}. Their content digest would be null, "
            "so the restart anti-join could never match them and each run "
            "would pay to infer over an empty request again. Populate the "
            "column upstream, or narrow the source to rows that have text."
        )
    if duplicate_count:
        raise UnusableSourceRows(
            f"{duplicate_count} duplicate {spec.key_column} value(s) in "
            f"{spec.source_table}. The MERGE cannot resolve two source rows "
            "onto one target row, and one row per key is what every "
            "provenance join assumes. De-duplicate upstream, or add the "
            "column that makes the key unique."
        )
    if snapshot.table != spec.source_table:
        raise EvidenceMismatch(
            f"the preflight measured {snapshot.table!r}, but this spec reads "
            f"{spec.source_table!r}"
        )
    return SourcePreflight(
        snapshot=snapshot,
        row_count=row_count,
        stratum_population=tuple(sorted(stratum_population.items())),
    )


def source_population_sql(
    spec: BatchInferenceSpec, snapshot: SourceSnapshot | None = None
) -> str:
    """Rows per stratum in the pinned snapshot.

    This is the population the weighted gate row estimates, so it is
    measured rather than asserted — for the same reason the total row
    count is. Feed the result into ``require_usable_source_rows``.
    """
    columns = ", ".join(spec.strata)
    return f"""SELECT {columns}, count(*) AS n
FROM {_source_relation(spec, snapshot)}
GROUP BY {columns}"""


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


def _validate_sample_allocation(
    population: Mapping[str, int],
    labelling_budget: int,
    min_per_stratum: int,
) -> None:
    """Validate the capacity inputs before allocating any rows."""

    if labelling_budget <= 0:
        raise ValueError("labelling_budget must be positive")
    if min_per_stratum <= 0:
        raise ValueError("min_per_stratum must be positive")
    if not population:
        raise ValueError("population must contain at least one stratum")
    for stratum, count in population.items():
        if count < 0:
            raise ValueError(f"stratum {stratum!r} has negative population")


def _distribute_remaining_sample(
    population: Mapping[str, int],
    allocation: dict[str, int],
    remaining_budget: int,
) -> None:
    """Distribute remaining rows deterministically by largest remainder."""

    spare = {
        stratum: population[stratum] - allocation[stratum] for stratum in population
    }
    spare_total = sum(spare.values())
    if not remaining_budget or not spare_total:
        return
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
    _validate_sample_allocation(population, labelling_budget, min_per_stratum)

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
    _distribute_remaining_sample(population, allocation, remaining_budget)
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

    #: The sampled source key this record judges. Carried so that
    #: duplicated evidence can be *detected*: an evaluation set is
    #: assembled by joining the sample to a gold table, and a gold table
    #: with two adjudications for one document silently yields two
    #: records. Scoring counts both, the Wilson interval narrows as
    #: though the sample were larger than it is, and a gate adopts on
    #: fewer distinct documents than its own numbers claim. The source
    #: preflight cannot see this: the duplication is created by the join,
    #: not present in the table.
    key: NonBlank
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
    sample_strata: tuple[NonBlank, ...] = Field(min_length=1)
    stratum_population: tuple[tuple[str, int], ...] = ()
    n_rows: int = Field(ge=0)
    n_gold: int = Field(ge=0)
    n_asserted: int = Field(ge=0)
    n_correct: int = Field(ge=0)
    precision: ConfidenceInterval | None
    recall: ConfidenceInterval | None
    abstention_rate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _intervals_match_the_counts(self) -> FieldStratumScore:
        """A physical stratum's intervals must be *its own* observations.

        The gate reads the intervals and never the raw counts, so a score
        reconstructed with `n_asserted=0` and a borrowed 200/200 precision
        interval would adopt on evidence from some other group. Each
        interval is therefore tied back to the counts printed beside it.

        The WEIGHTED row's intervals cannot be checked this way — they are
        the population estimate expressed through an effective sample
        size, so their trials are deliberately not a row count. It is
        checked instead by recomputing it from the physical rows and the
        weights it carries, in `require_matching_evidence`; that is why
        those weights are persisted rather than discarded after scoring.
        """
        _validate_score_counts(self)
        if self.stratum == WEIGHTED:
            _validate_weighted_score(self)
            return self
        _validate_physical_score(self)
        return self


def _validate_score_counts(score: FieldStratumScore) -> None:
    """Bind every score's event counts to its sampled row count."""

    if score.n_gold > score.n_rows or score.n_asserted > score.n_rows:
        raise ValueError(
            f"stratum {score.stratum!r} sampled {score.n_rows} row(s) but "
            f"reports {score.n_gold} gold and {score.n_asserted} asserted "
            "value(s); a row cannot carry more than one of either"
        )
    if score.n_correct > min(score.n_asserted, score.n_gold):
        raise ValueError(
            "n_correct cannot exceed the values asserted or the values that exist"
        )


def _validate_weighted_score(score: FieldStratumScore) -> None:
    """Validate the population weights carried by an aggregate score."""

    if not score.stratum_population:
        raise ValueError(
            "the population-weighted row must carry the weights it was "
            "computed from, so the gate can recompute it"
        )
    weight_names = [name for name, _ in score.stratum_population]
    if len(weight_names) != len(set(weight_names)):
        raise ValueError(
            "the weighted row names a stratum more than once in its population weights"
        )
    if tuple(sorted(weight_names)) != tuple(sorted(score.sample_strata)):
        raise ValueError(
            "the weighted row's population weights must cover exactly the "
            "strata the sample covered"
        )


def _validate_physical_score(score: FieldStratumScore) -> None:
    """Bind physical-stratum intervals to their raw observations."""

    if score.stratum_population:
        raise ValueError(
            f"stratum {score.stratum!r} is a physical stratum and carries "
            "no population weights; only the aggregate row does"
        )
    for metric, interval, denominator in (
        ("precision", score.precision, score.n_asserted),
        ("recall", score.recall, score.n_gold),
    ):
        if denominator == 0:
            if interval is not None:
                raise ValueError(
                    f"{metric} has no denominator in stratum {score.stratum!r}, "
                    "so it must be absent, not an interval"
                )
            continue
        if interval is None:
            raise ValueError(
                f"{metric} in stratum {score.stratum!r} has {denominator} "
                "observations but no interval"
            )
        if (interval.successes, interval.trials) != (score.n_correct, denominator):
            raise ValueError(
                f"the {metric} interval in stratum {score.stratum!r} reports "
                f"{interval.successes}/{interval.trials}, but this score counted "
                f"{score.n_correct}/{denominator}"
            )


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
    # A declined name that matches no declared field cannot be honoured:
    # the model may have been withholding a value while misspelling its
    # name, and silently ignoring the entry would land exactly the value
    # it was trying to withhold. Here — at evaluation time, on a labelled
    # sample — that is a bug to fix before the release ships, so it
    # raises. The generated SQL cannot raise over a million rows without
    # taking the whole run down with it, so it nulls the row's values and
    # routes it to the exception queue instead: same refusal to assert,
    # scaled to where it happens.
    unknown = sorted(listed - {field.name for field in spec.fields})
    if unknown:
        raise UnknownAbstainedField(
            f"the response declined {unknown}, which {spec.name!r} never "
            "declared. The intended field cannot be recovered, so nothing "
            "from this response may be asserted."
        )
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
    # One document, one vote. An evaluation set is built by joining the
    # sample to a gold table, and a gold table holding two adjudications
    # for a document yields two records of it — which scoring would count
    # as two independent observations. Nothing downstream could notice:
    # the counts are consistent, the intervals are correctly computed for
    # the n they are given, and that n is simply wrong. The interval
    # narrows, the lower bound rises, and the gate adopts on fewer
    # distinct documents than its own evidence claims.
    #
    # This is deliberately a refusal, not a de-duplication. Two rows for
    # one key mean the join is wrong or the gold set has an unresolved
    # disagreement, and silently keeping one of them would pick an
    # adjudication arbitrarily.
    seen: dict[str, int] = {}
    for record in records:
        seen[record.key] = seen.get(record.key, 0) + 1
    repeated = sorted(key for key, count in seen.items() if count > 1)
    if repeated:
        raise EvidenceMismatch(
            f"{len(repeated)} document(s) appear more than once in the "
            f"evaluation set, first {repeated[:3]}. Each would be counted as "
            "independent evidence, narrowing every interval built from it. "
            "Resolve the duplicate gold rows or fix the join before scoring."
        )

    by_stratum: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        by_stratum.setdefault(record.stratum, []).append(record)

    # WEIGHTED labels the synthetic all-strata row. A real stratum with
    # that value would collide with it: high-criticality gating filters
    # both away and finds no evidence, while medium/low would consume the
    # physical stratum as if it were the population estimate.
    if WEIGHTED in by_stratum or WEIGHTED in stratum_population:
        raise ValueError(
            f"{WEIGHTED!r} is reserved for the population-weighted row and "
            "cannot be a stratum value; rename the stratum in the source."
        )

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
        scores.append(_weighted_row(field.name, per_stratum, stratum_population))
    return tuple(scores)


def _weighted_row(
    field_name: str,
    per_stratum: Sequence[FieldStratumScore],
    stratum_population: Mapping[str, int],
) -> FieldStratumScore:
    """Build the population-weighted row from the physical stratum rows.

    Deliberately a pure function of the physical scores plus the
    population weights, and the *only* place the aggregate is produced.
    `require_matching_evidence` calls it again on persisted evidence and
    compares, so scoring and verification cannot drift apart: if the
    aggregate is ever computed a second way, the check that recomputes it
    is computing the same thing.

    Weight each stratum by the population share of the rows that actually
    enter the metric: for precision that is the estimated population of
    asserted values, for recall the estimated population of true values.
    Both collapse to the plain population share when the rates are equal
    across strata.
    """
    reference = per_stratum[0]
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
    return FieldStratumScore(
        field=field_name,
        stratum=WEIGHTED,
        release=reference.release,
        confidence=reference.confidence,
        sample_strata=reference.sample_strata,
        stratum_population=tuple(
            (stratum, stratum_population[stratum])
            for stratum in reference.sample_strata
        ),
        n_rows=sum(score.n_rows for score in per_stratum),
        n_gold=sum(score.n_gold for score in per_stratum),
        n_asserted=sum(score.n_asserted for score in per_stratum),
        n_correct=sum(score.n_correct for score in per_stratum),
        precision=_weighted_interval(precision_inputs, reference.confidence),
        recall=_weighted_interval(recall_inputs, reference.confidence),
        abstention_rate=weighted_abstention,
    )


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
    """Immutable gate evidence for one spec + one evaluation sample.

    Nothing in this report is trusted; every verdict in it is *derived*
    from the scores it carries. It is a persisted artifact that later
    authorises a paid, table-mutating run, and ``require_executable``
    reads only the aggregate, so anything left merely asserted here is
    the whole gate's weakest point.

    The chain reconstructs from the raw counts up, and each link is
    checked where it can be:

    - a ``ConfidenceInterval`` against its own successes and trials,
    - a physical ``FieldStratumScore``'s intervals against the counts
      printed beside them,
    - the population-weighted row by recomputing it from the physical
      rows and the weights it carries,
    - each ``FieldGateResult`` by re-running ``_gate_field`` over those
      scores,
    - and ``decision`` from the field results.

    ``require_executable`` closes it by binding ``scores`` to the spec —
    the one thing a self-contained report cannot know about itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_name: NonBlank
    spec_digest: str
    use_tier: UseTier
    confidence_level: float
    scores: tuple[FieldStratumScore, ...] = Field(min_length=1)
    fields: tuple[FieldGateResult, ...] = Field(min_length=1)
    decision: GateDecision
    source_snapshot: SourceSnapshot | None = None
    approved_by: str | None = None

    @field_validator("approved_by")
    @classmethod
    def _approver_is_a_name(cls, value: str | None) -> str | None:
        """`approve_gate` strips and refuses blanks; the model must too.

        A whitespace approver is truthy, so a reconstructed tier 1 report
        carrying `" "` derives ADOPT and satisfies the named-approver
        check while recording that nobody accepted the risk. Persisted
        evidence reaches this class without going through `approve_gate`.
        """
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "approved_by records the person or group who accepted the "
                "residual risk; blank is not an approver"
            )
        return stripped

    human_review_obligations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _gated_evidence_names_its_snapshot(self) -> GateReport:
        """`evaluate_gate` requires this, but a report is more often
        *reconstructed* than produced — and an unpinned adopting report
        read back from JSON would pass `require_executable` and let
        `build_execute_sql` fall through to the live table. The invariant
        belongs on the artifact, not only on the function that mints it.
        """
        if self.use_tier != UseTier.EXPLORATORY and self.source_snapshot is None:
            raise ValueError(
                f"tier {self.use_tier} evidence must name the source version "
                "it describes; an unpinned report cannot authorise a run"
            )
        return self

    @model_validator(mode="after")
    def _every_verdict_follows_from_the_scores(self) -> GateReport:
        by_field: dict[str, list[FieldStratumScore]] = {}
        for score in self.scores:
            by_field.setdefault(score.field, []).append(score)
        for result in self.fields:
            recomputed = _gate_field(
                result.field,
                result.criticality,
                result.required_rate,
                by_field.get(result.field, ()),
            )
            if recomputed != result:
                raise ValueError(
                    f"the gate result for {result.field!r} does not follow "
                    "from the scores in this report; re-running the gate over "
                    "them reaches a different verdict"
                )
        verdicts = {result.decision for result in self.fields}
        if GateDecision.REJECT in verdicts:
            implied = GateDecision.REJECT
        elif GateDecision.INCONCLUSIVE in verdicts:
            implied = GateDecision.INCONCLUSIVE
        elif self.use_tier == UseTier.CONSEQUENTIAL:
            # Tier 1 passes only into pending_approval; `approve_gate`
            # then records the named human who accepted the residual risk.
            implied = (
                GateDecision.ADOPT
                if self.approved_by
                else GateDecision.PENDING_APPROVAL
            )
        else:
            implied = GateDecision.ADOPT
        if self.decision != implied:
            raise ValueError(
                f"decision {self.decision.value!r} does not follow from the "
                f"field results, which imply {implied.value!r}"
            )
        if self.approved_by is not None and self.use_tier != UseTier.CONSEQUENTIAL:
            raise ValueError(
                "approved_by is a tier 1 obligation; a lower tier records no "
                "named approver"
            )
        return self


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
    name: str,
    criticality: Criticality,
    required: float,
    scores: Sequence[FieldStratumScore],
) -> FieldGateResult:
    """Judge one field from its scores.

    Takes the three things it actually uses rather than a `FieldSpec`, so
    that `GateReport` — which has a `FieldGateResult` recording exactly
    those three, and no spec — can call this same function to recompute a
    persisted verdict and compare. One implementation, two callers: a
    second implementation written for verification would only be a second
    thing to keep in sync.
    """
    # criticality: high → every stratum must clear the bar on its own.
    # Aggregate performance is irrelevant if the failures concentrate in
    # the one stratum that matters, so no average appears here at all.
    # medium / low → the population-weighted estimate decides. Never the
    # raw pool of a stratified sample, which over-weights rare strata and
    # so estimates nothing about the population.
    if criticality == Criticality.HIGH:
        considered = [score for score in scores if score.stratum != WEIGHTED]
    else:
        considered = [score for score in scores if score.stratum == WEIGHTED]
    if not considered:
        raise ValueError(f"no scores supplied for field {name!r}")

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
        field=name,
        criticality=criticality,
        required_rate=required,
        decision=decision,
        binding_stratum=binding[1] if binding else None,
        binding_metric=binding[2] if binding else None,
        binding_lower_bound=binding[0] if binding else None,
        binding_point_estimate=binding[3] if binding else None,
        reasons=tuple(reasons),
    )


def _require_score_matches_spec(
    spec: BatchInferenceSpec,
    score: FieldStratumScore,
) -> None:
    """Bind one persisted score to the release and confidence it claims."""

    expected = spec.release
    if score.release != expected:
        raise EvidenceMismatch(
            f"score for {score.field!r}/{score.stratum!r} was produced for "
            f"prompt {score.release.prompt_version} / model "
            f"{score.release.model_version} (spec digest "
            f"{score.release.spec_digest[:12]}…), but this gate is for prompt "
            f"{expected.prompt_version} / model {expected.model_version} "
            f"(spec digest {expected.spec_digest[:12]}…). Re-score the sample "
            "against the release being gated."
        )
    if score.confidence != spec.confidence_level:
        raise EvidenceMismatch(
            f"score for {score.field!r}/{score.stratum!r} was computed at "
            f"confidence {score.confidence}, but the spec declares "
            f"{spec.confidence_level}. Re-score at the declared level."
        )
    for metric, interval in (
        ("precision", score.precision),
        ("recall", score.recall),
    ):
        if interval is not None and interval.confidence != spec.confidence_level:
            raise EvidenceMismatch(
                f"the {metric} interval for {score.field!r}/{score.stratum!r} "
                f"was computed at confidence {interval.confidence}, but the "
                f"spec declares {spec.confidence_level}. Its bounds do not "
                "mean what the score claims; re-score at the declared level."
            )


def _single_evidence_manifest(
    scores: Sequence[FieldStratumScore],
) -> tuple[str, ...]:
    """Return the one sample manifest shared by every score."""

    manifests = {score.sample_strata for score in scores}
    if len(manifests) != 1:
        raise EvidenceMismatch(
            f"scores disagree about which strata the sample covered: "
            f"{sorted(manifests)}. They did not come from one scoring run."
        )
    return next(iter(manifests))


def _require_evidence_coverage(
    spec: BatchInferenceSpec,
    scores: Sequence[FieldStratumScore],
    manifest: tuple[str, ...],
) -> None:
    """Require every declared field and sampled stratum to be represented."""

    required_groups = set(manifest) | {WEIGHTED}
    present: dict[str, set[str]] = {}
    for score in scores:
        present.setdefault(score.field, set()).add(score.stratum)
    for field in spec.fields:
        missing = sorted(required_groups - present.get(field.name, set()))
        if missing:
            raise EvidenceMismatch(
                f"evidence for field {field.name!r} is incomplete: no scores "
                f"for {missing}. The sample covered {list(manifest)}, and a "
                "worst-stratum gate cannot be applied to a filtered set."
            )


def _index_score_evidence(
    scores: Sequence[FieldStratumScore],
) -> dict[str, dict[str, FieldStratumScore]]:
    """Index one unambiguous score per field and stratum."""

    indexed: dict[str, dict[str, FieldStratumScore]] = {}
    for score in scores:
        group = indexed.setdefault(score.field, {})
        if score.stratum in group:
            raise EvidenceMismatch(
                f"two scores describe {score.field!r} in stratum "
                f"{score.stratum!r}. One measurement per group is what makes "
                "the aggregate recomputable; with two, the gate reads "
                "whichever happens to be last."
            )
        group[score.stratum] = score
    return indexed


def _require_weighted_evidence(
    spec: BatchInferenceSpec,
    indexed: Mapping[str, Mapping[str, FieldStratumScore]],
    manifest: tuple[str, ...],
) -> None:
    """Recompute every claimed population-weighted aggregate."""

    for field in spec.fields:
        group = indexed.get(field.name, {})
        claimed = group.get(WEIGHTED)
        if claimed is None:
            continue
        physical = [group[stratum] for stratum in manifest if stratum in group]
        recomputed = _weighted_row(
            field.name,
            physical,
            dict(claimed.stratum_population),
        )
        if recomputed != claimed:
            raise EvidenceMismatch(
                f"the population-weighted row for {field.name!r} does not "
                "follow from its stratum scores. Recomputing it from those "
                "rows and the weights it carries gives different evidence, "
                "so the aggregate was not produced by this sample."
            )


def require_matching_evidence(
    spec: BatchInferenceSpec,
    scores: Sequence[FieldStratumScore],
) -> None:
    """Require complete evidence from exactly the release being gated."""

    if not scores:
        raise EvidenceMismatch("no scores supplied; a gate needs evidence")
    for score in scores:
        _require_score_matches_spec(spec, score)
    manifest = _single_evidence_manifest(scores)
    _require_evidence_coverage(spec, scores, manifest)
    _require_weighted_evidence(spec, _index_score_evidence(scores), manifest)


def evaluate_gate(
    spec: BatchInferenceSpec,
    scores: Sequence[FieldStratumScore],
    *,
    source_snapshot: SourceSnapshot | None = None,
) -> GateReport:
    """Compare interval lower bounds against the tolerances declared in the
    spec — never the point estimates.

    ``source_snapshot`` records which Delta version of the source this
    evidence describes. ``build_execute_sql`` reads it back off the report
    and pins the run to it, so the rows that get inferred are the rows the
    sample was drawn from.

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
    if source_snapshot is None:
        if spec.gate_required:
            raise EvidenceMismatch(
                f"tier {spec.use_tier} evidence must record which version of "
                f"{spec.source_table} it describes: the run is pinned to it, "
                "and without one the gate certifies a table rather than a set "
                "of rows. Capture it before sampling."
            )
    else:
        _source_relation(spec, source_snapshot)  # refuses a foreign table
    by_field: dict[str, list[FieldStratumScore]] = {}
    for score in scores:
        by_field.setdefault(score.field, []).append(score)
    results = tuple(
        _gate_field(
            field.name,
            field.criticality,
            field.required_rate,
            by_field.get(field.name, ()),
        )
        for field in spec.fields
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
        scores=tuple(scores),
        fields=results,
        decision=decision,
        source_snapshot=source_snapshot,
        human_review_obligations=obligations,
    )


def revalidated(artifact: _ArtifactT, error: type[Exception]) -> _ArtifactT:
    """Re-run an artifact's validators, because `model_copy` does not.

    `model_copy(update=...)` produces an object of the right type that
    never satisfied the invariants its own model declares —
    `rejecting.model_copy(update={"decision": ADOPT})` is a valid
    `GateReport` whose aggregate its field results do not support. Every
    "the artifact validates itself" guard in this module is only true at
    the boundaries that do this round trip.

    It is a function rather than a line repeated at each boundary because
    repeating it is how the persistence path came to be missing it while
    the authorisation path had it: a forged adoption was refused at
    execution and written to the durable evaluation record anyway.
    """
    try:
        return type(artifact).model_validate(artifact.model_dump(mode="python"))
    except ValidationError as invalid:
        raise error(
            f"the {type(artifact).__name__} does not satisfy its own "
            f"invariants: {invalid}"
        ) from invalid


def approve_gate(report: GateReport, approver: str) -> GateReport:
    """Record the named human sign-off a tier 1 run requires.

    Only a ``PENDING_APPROVAL`` report can be approved: approval is a
    person accepting a passing result's residual risk, never a way to
    override a rejection or to substitute for missing evidence.

    Prefer an accountable role or group over an individual's email. The
    value is retained in the gate artifact and the run metadata table
    (both access controlled) and deliberately never reaches a tag.
    """
    revalidated(report, GateNotPassed)
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


def _require_report_matches_spec(
    spec: BatchInferenceSpec,
    report: GateReport,
) -> None:
    """Bind a self-consistent report to the policy it authorises."""

    if report.spec_digest != spec.spec_digest:
        raise GateNotPassed(
            "gate report was produced for a different spec revision; "
            "re-evaluate against the current spec"
        )
    if report.use_tier != spec.use_tier:
        raise GateNotPassed(
            f"the gate report claims tier {report.use_tier}, but this run is "
            f"tier {spec.use_tier}. Tier selects which controls apply, so a "
            "report from another tier cannot authorise it."
        )
    require_matching_evidence(spec, report.scores)
    judged = {result.field for result in report.fields}
    unjudged = sorted(field.name for field in spec.fields if field.name not in judged)
    if unjudged:
        raise GateNotPassed(
            f"gate report does not judge {unjudged}; it cannot authorise a "
            "run that writes those fields"
        )
    by_name = {result.field: result for result in report.fields}
    for field in spec.fields:
        result = by_name[field.name]
        if (result.criticality, result.required_rate) != (
            field.criticality,
            field.required_rate,
        ):
            raise GateNotPassed(
                f"the gate report judged {field.name!r} as "
                f"{result.criticality.value} at {result.required_rate}, but "
                f"the spec declares {field.criticality.value} at "
                f"{field.required_rate}. It certifies a policy this run does "
                "not run under."
            )


def _require_report_population(
    report: GateReport,
    preflight: SourcePreflight,
) -> None:
    """Bind persisted population weights to warehouse-measured counts."""

    measured = dict(preflight.stratum_population)
    weighted_rows = [score for score in report.scores if score.stratum == WEIGHTED]
    if weighted_rows and not measured:
        raise GateNotPassed(
            "the preflight carries no measured stratum counts, and this "
            "report is weighted by population. Measure them with "
            "`source_population_sql` — an unmeasured population cannot "
            "certify an estimate of it."
        )
    if measured:
        for score in weighted_rows:
            claimed = dict(score.stratum_population)
            if claimed != {k: v for k, v in measured.items() if k in claimed}:
                raise GateNotPassed(
                    f"the weighted evidence for {score.field!r} claims "
                    f"population {sorted(claimed.items())}, but the snapshot "
                    f"holds {sorted(measured.items())}. The estimate is "
                    "weighted for a population that does not exist."
                )
            unsampled = sorted(set(measured) - set(claimed))
            if unsampled:
                raise GateNotPassed(
                    f"the snapshot contains strata {unsampled} that the "
                    f"weighted evidence for {score.field!r} does not cover; "
                    "the population estimate omits part of the population."
                )


def require_executable(
    spec: BatchInferenceSpec,
    report: GateReport | None,
    preflight: SourcePreflight,
) -> None:
    """Refuse to execute without an adopting gate (tiers 1 and 2)."""

    if not spec.gate_required:
        return
    if report is None:
        raise GateNotPassed(f"tier {spec.use_tier} runs require a gate report")
    revalidated(report, GateNotPassed)
    _require_report_matches_spec(spec, report)
    _require_report_population(report, preflight)
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
        # Constrained to the declared vocabulary: a free-form string lets a
        # near-miss like "issuer_nam" satisfy the schema while matching no
        # field, so the abstention is dropped and the value the model was
        # declining lands anyway. The generated SQL rejects unknown names
        # too — this narrows what the model can emit, that catches what it
        # emits regardless.
        "items": {"type": "string", "enum": [field.name for field in spec.fields]},
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
    #: (column, found type, expected type) for columns that exist under
    #: the right name and the wrong type. Not fixable by this pipeline:
    #: changing a column's type rewrites landed values, which is a
    #: governance event, so it blocks and a human decides.
    mistyped: tuple[tuple[str, str, str], ...] = ()
    statements: tuple[str, ...] = ()

    @property
    def blocking(self) -> tuple[str, ...]:
        """Columns a human must resolve before the release can execute."""
        return self.stale + self.foreign + tuple(name for name, _, _ in self.mistyped)

    @property
    def required(self) -> bool:
        return bool(self.add or self.blocking)


def plan_target_migration(
    spec: BatchInferenceSpec,
    existing_columns: Sequence[str] | Mapping[str, str],
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

    Pass a ``{name: type}`` mapping to have the *types* checked too.
    Matching names alone declares the table ready when a confidence
    column is STRING rather than DOUBLE, or an ordering column is not a
    BIGINT: ``UPDATE SET *`` then either casts silently, leaving the
    advertised schema wrong, or fails inside the paid statement instead
    of before it. A plain sequence of names still works, and skips the
    type check, for callers that genuinely only have names.
    """
    declared_types = (
        {name.casefold(): sql_type for name, sql_type in existing_columns.items()}
        if isinstance(existing_columns, Mapping)
        else {}
    )
    existing = {column.casefold() for column in existing_columns}
    expected = target_columns(spec)
    expected_names = {name.casefold() for name, _ in expected}

    mistyped = tuple(
        (name, declared_types[name.casefold()], sql_type)
        for name, sql_type in expected
        if name.casefold() in declared_types
        and declared_types[name.casefold()].casefold() != sql_type.casefold()
    )

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
    for ordering_column in (
        "ai_release_sequence",
        "ai_source_version",
        "ai_strata_version",
    ):
        if any(name == ordering_column for name, _ in add):
            # Rows that predate sequencing get -1 rather than NULL. The SQL
            # comparisons coalesce anyway, but a real value keeps the column
            # honest for anyone reading the table directly.
            statements.append(
                f"UPDATE {spec.target_table} SET {ordering_column} = -1 "
                f"WHERE {ordering_column} IS NULL"
            )
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
    for column, found, expected_type in mistyped:
        statements.append(
            f"-- review before running: {spec.target_table}.{column} is "
            f"{found}, but this release writes {expected_type}. Changing it "
            "rewrites landed values, so decide deliberately."
        )
    return TargetMigration(
        add=add,
        stale=stale,
        foreign=foreign,
        mistyped=mistyped,
        statements=tuple(statements),
    )


def require_migrated_target(
    spec: BatchInferenceSpec,
    existing_columns: Sequence[str] | Mapping[str, str],
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


def resync_strata_sql(
    spec: BatchInferenceSpec,
    snapshot: SourceSnapshot | None = None,
    *,
    preflight: SourcePreflight | None = None,
) -> str:
    """Refresh landed stratum values from the source, without inference.

    Strata are row *metadata*, not model output: correcting a document's
    ``layout`` from ``standard`` to ``legacy_scan`` changes how the row is
    grouped for monitoring and re-sampling, but not a character the model
    would return. The restart predicate keys on content and release, so it
    rightly treats such a row as done — which would leave the obsolete
    stratum in place forever.

    Fixing that by widening the anti-join would work, but it would pay an
    endpoint to regenerate identical output in order to correct a label.
    This statement updates the labels directly instead. Run it before the
    execute stage; it is cheap and touches only stratum columns.

    **It deliberately does not require an adopting gate**, unlike
    ``resync_policy_sql`` next door. Stratum labels are copied from source
    columns, so this asserts nothing about model quality — and a release
    the gate *rejected* is exactly when you may still need the labels
    correct, to re-sample the stratum that failed. It does require the
    preflight to describe the snapshot it reads, because relabelling from
    rows the run never examined is still wrong. If a later change makes
    the two resyncs symmetric for tidiness, this paragraph is the reason
    not to.
    """
    if preflight is not None and preflight.snapshot != snapshot:
        raise EvidenceMismatch(
            f"the preflight measured {preflight.snapshot} but this resync "
            f"reads {snapshot}; the labels would come from rows that were "
            "never checked"
        )
    version = snapshot.version if snapshot else -1
    # The update advances the column it orders on, so the guard actually
    # moves: a corrected label with unchanged document text is skipped by
    # inference (the row really is done), so nothing else would ever
    # raise it and a delayed older resync would keep winning.
    assignments = ",\n    ".join(
        [f"target.{column} = source.{column}" for column in spec.strata]
        + [f"target.ai_strata_version = {version}"]
    )
    differs = "\n     OR ".join(
        f"NOT (target.{column} <=> source.{column})" for column in spec.strata
    )
    # Strata order on a column of their own, and it is the only ordering
    # that applies to them. They come from source columns rather than
    # from the model, so release identity is irrelevant here — and
    # borrowing the inference pair, as this statement used to, ordered on
    # numbers it could not consistently advance: it wrote
    # `ai_source_version` while leaving `ai_release_sequence` at whatever
    # release last inferred the row, so a newer release resyncing from an
    # older snapshot moved the two halves in opposite directions and a
    # delayed older resync could win again at an intermediate version.
    #
    # Advancing `ai_release_sequence` here instead would be worse than
    # the bug: a row whose labels were resynced would look as though the
    # newer release had inferred it, and that release's anti-join would
    # skip the inference the row still needs. Metadata gets metadata
    # ordering.
    return f"""MERGE INTO {spec.target_table} AS target
USING {_source_relation(spec, snapshot)} AS source
ON target.{spec.key_column} = source.{spec.key_column}
WHEN MATCHED
  AND coalesce(target.ai_strata_version, -1) <= {version}
  AND (
     {differs}
   ) THEN UPDATE SET
    {assignments}"""


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


def _require_authorised_write(
    spec: BatchInferenceSpec,
    report: GateReport | None,
    estimate: CostEstimate,
    preflight: SourcePreflight,
) -> SourceSnapshot:
    """Everything that must hold before a statement may mutate the target.

    This exists as one function because the alternative was found four
    separate times: `build_execute_sql` learned to enforce the gate, then
    the ceiling, then the preflight, each after a review noticed that the
    notebook's call ordering — not the builder — was what had been
    protecting it. `resync_policy_sql` was then written and inherited
    none of it, because there was nothing to inherit. A new mutating
    statement calls this and starts out safe; that is the point.

    Returns the snapshot the caller must read, so the authorisation and
    the data can never come from different places.
    """
    # `model_copy(update=...)` skips validators, so *any* artifact reaching
    # here may never have satisfied the invariants its own model declares.
    # `require_executable` round-trips the report for exactly that reason —
    # and the estimate and preflight were left out, which meant
    # `estimate.model_copy(update={"projected_cost_cad": 0})` cleared the
    # ceiling with every other check still matching. Revalidating one
    # artifact and not its siblings is the same mistake three rounds
    # running, so this revalidates the whole bundle.
    for artifact in (estimate, preflight):
        revalidated(artifact, EvidenceMismatch)
    if spec.gate_required and report is None:
        raise GateNotPassed(
            f"tier {spec.use_tier} runs execute against the snapshot their "
            "gate report describes, so the report is required to build the "
            "statement"
        )
    # Checked for any supplied report, including tier 3's optional one,
    # which `require_executable` deliberately waves through.
    if report is not None and report.spec_digest != spec.spec_digest:
        raise EvidenceMismatch(
            "the report was produced for a different spec revision; its "
            "source snapshot does not describe this run"
        )
    require_executable(spec, report, preflight)
    # Gated runs are pinned by their evidence. Tier 3 has no evidence, so
    # it is pinned by the thing that *is* its only control: the estimate.
    snapshot = report.source_snapshot if report else estimate.source_snapshot
    if snapshot is None:
        raise EvidenceMismatch(
            "a paid run reads a pinned snapshot; this estimate does not name "
            "one, so nothing bounds the rows the statement would process"
        )
    if estimate.release != spec.release:
        raise EvidenceMismatch(
            "the cost estimate was computed for a different release; "
            "re-estimate before authorising the spend"
        )
    # The projection is recomputed from its own inputs by the model, but
    # the *ceiling* it is compared against is not that estimate's to
    # choose. The spec declares the budget, exactly as it declares the
    # gate policy; an artifact that raises its own ceiling clears itself.
    if estimate.cost_ceiling_cad != spec.cost_ceiling_cad:
        raise EvidenceMismatch(
            f"the estimate was approved against a ceiling of "
            f"{estimate.cost_ceiling_cad} CAD, but this spec declares "
            f"{spec.cost_ceiling_cad} CAD. The budget is the spec's to set."
        )
    if estimate.source_snapshot != snapshot:
        raise EvidenceMismatch(
            f"the cost estimate describes {estimate.source_snapshot} but this "
            f"run reads {snapshot}. A projection over different rows cannot "
            "authorise this spend."
        )
    if preflight.snapshot != snapshot:
        raise EvidenceMismatch(
            f"the preflight measured {preflight.snapshot} but this run reads "
            f"{snapshot}; the rows checked are not the rows to be processed"
        )
    # The preflight carries the only row count that came from the
    # warehouse. Recomputing the projection proved the arithmetic was
    # honest, not that its `row_count` was: halve the count and the price
    # together and the estimate stays self-consistent while the pinned
    # snapshot still holds a million rows.
    if estimate.row_count != preflight.row_count:
        raise EvidenceMismatch(
            f"the estimate priced {estimate.row_count} row(s), but the "
            f"preflight counted {preflight.row_count} in that snapshot. The "
            "budget was approved for a different amount of work."
        )
    require_within_ceiling(estimate)
    return snapshot


def restart_predicate_sql(
    spec: BatchInferenceSpec, snapshot: SourceSnapshot | None = None
) -> str:
    """The "already done" test, as SQL, for a ``done`` / ``source`` pair.

    Exposed because anything that *counts* pending rows has to agree with
    what the run will actually process, and a hand-copied mirror of this
    predicate drifts the moment the real one changes — it silently did,
    twice, while the ordering rules were being worked out. One string,
    every caller.

    Matching is on the **inference identity**, not the spec digest. The
    digest covers tolerances, tier, consumers and the cost ceiling —
    none of which change a character the model returns — so keying on it
    sent an entire pinned table back through a paid endpoint to
    regenerate byte-identical predictions every time a tolerance moved.
    The inference digest still covers endpoint, model, prompt *text*,
    abstention threshold and field descriptions, so a changed threshold
    or field set re-infers exactly as before; ``resync_policy_sql``
    refreshes which policy governs a row without paying for it.
    """
    sequence = int(spec.release_sequence)
    version = snapshot.version if snapshot else -1
    return f"""
      AND (
        coalesce(done.ai_release_sequence, -1) > {sequence}
        OR (
          coalesce(done.ai_release_sequence, -1) = {sequence}
          AND coalesce(done.ai_source_version, -1) > {version}
        )
        OR (
          done.ai_source_digest = sha2(source.{spec.document_column}, 256)
          AND done.ai_inference_digest =
            {sql_string_literal(spec.inference_digest)}
        )
      )"""


def resync_policy_sql(
    spec: BatchInferenceSpec,
    *,
    run_id: str,
    estimate: CostEstimate,
    preflight: SourcePreflight,
    report: GateReport | None = None,
) -> str:
    """Refresh which policy release governs already-correct predictions.

    A release that changes only judgment — a tolerance, a tier, a
    consumer, the cost ceiling — produces exactly the predictions already
    landed, because none of it reaches the model. Restart therefore keys
    on the inference identity and leaves those rows alone, which is the
    whole point; but the table would then still say an older spec digest
    governs them.

    This statement corrects the provenance without paying an endpoint. It
    touches only the policy stamps, and only on rows whose values this
    release would have produced anyway — which means matching the source
    content too, not just the inference identity. A document edited since
    the row was inferred is *pending*, and the run may be interrupted or
    never started; stamping it here would leave production claiming this
    release governs values derived from text that no longer exists.

    Stamping ``ai_spec_digest`` onto a production row asserts that *this
    release governs these values* — an authorisation, not a cleanup — so
    this goes through the same check the paid statement does. Without it
    a rejecting or unapproved report could relabel the table as governed
    by a release the gate refused, and the restart predicate would then
    treat those rows as done.

    It also points ``ai_policy_run_id`` at this release's run, because
    otherwise the row would claim the new policy while every provenance
    join still resolved to the run that carried the old one — and no row
    would reference the new run record at all. ``ai_run_id`` keeps
    pointing at the run that produced the values, which remains true.
    """
    snapshot = _require_authorised_write(spec, report, estimate, preflight)
    return f"""MERGE INTO {spec.target_table} AS target
USING {_source_relation(spec, snapshot)} AS source
ON target.{spec.key_column} = source.{spec.key_column}
WHEN MATCHED
  AND target.ai_inference_digest = {sql_string_literal(spec.inference_digest)}
  AND target.ai_source_digest = sha2(source.{spec.document_column}, 256)
  AND coalesce(target.ai_release_sequence, -1) <= {int(spec.release_sequence)}
  AND NOT (target.ai_spec_digest <=> {sql_string_literal(spec.spec_digest)})
  THEN UPDATE SET
    target.ai_spec_digest = {sql_string_literal(spec.spec_digest)},
    target.ai_release_sequence = {int(spec.release_sequence)},
    target.ai_policy_run_id = {sql_string_literal(run_id)}"""


def build_execute_sql(
    spec: BatchInferenceSpec,
    *,
    run_id: str,
    estimate: CostEstimate,
    preflight: SourcePreflight,
    report: GateReport | None = None,
) -> str:
    """The full-table execute statement: restartable *and* release-aware.

    - **The evidence chooses the data.** A gated spec must hand over the
      report that authorised the run, and the source snapshot is read off
      *that* — never from a separate argument. Population, sample, gate
      and cost estimate were all computed against one version of the
      source, and a tier 1 spec then waits for a human; taking the version
      as its own parameter would let the run read a table that had moved
      underneath the evidence, inferring rows no sample covered and
      spending past a ceiling approved for fewer of them. With one source
      for it, that disagreement cannot be expressed — the same reason the
      prompt is not a parameter either.

    - The request is built from ``spec.prompt_template``, not from a
      caller-supplied expression. Taking the prompt as an argument meant
      the statement could run one prompt while every stamp claimed
      another; with one source of truth that disagreement cannot be
      expressed.
    - **Restart**: the anti-join drops rows this release has already
      landed *for this document content*, so a re-run after a partial
      failure finishes the job instead of paying for inference twice,
      while a document corrected in place becomes pending again. Keying
      on the row key alone would leave the target serving values derived
      from text that no longer exists. A million-row job will fail partway
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

      **Ordering is on the pair ``(release_sequence, source_version)``.**
      The release sequence orders *what ran*; it says nothing about
      *which rows it ran over*. Two cycles of the same spec — a nightly
      job, unchanged for months — snapshot different source versions and
      therefore tie on sequence alone, and a delayed Monday run finishing
      after Tuesday's would overwrite Tuesday's output for every document
      edited in between: the changed digest makes those rows pending, so
      they are re-inferred and written straight over newer content. The
      source version breaks that tie in both directions — the anti-join
      treats a newer snapshot of the same release as done, and the MERGE
      refuses to lower it.

      A strictly newer sequence excludes the row **on its own**, before
      content is considered. Testing the content digest first looks
      equivalent and is not: when the newer release landed the key from
      *edited* text, the digest disagrees, the row is no longer excluded,
      and the old job pays ``ai_query`` for output the MERGE then
      correctly refuses to write. Across two large overlapping releases
      that is a second full bill for every edited document, with nothing
      to show for it. Read the predicate as: newer wins outright;
      otherwise this exact content by this exact release.

      The digest covers the spec, not this module's code. Bump
      ``spec_version`` when a code change alters what the pipeline
      produces — that is what makes the release identity honest.
    - ``failOnError => false`` keeps one poisoned document from killing
      the run; its error message lands in ``ai_error`` and the row flows
      to the exception queue instead of blocking everything else. The
      struct field is ``errorMessage`` (``response`` is null on failure).
    """
    snapshot = _require_authorised_write(spec, report, estimate, preflight)
    source = _source_relation(spec, snapshot)
    source_version = snapshot.version if snapshot else -1
    restart = restart_predicate_sql(spec, snapshot)
    # The abstention rule from `apply_abstention_policy`, expressed in SQL
    # so that what the gate measured is exactly what lands: a field the
    # model listed as abstained, or answered below the declared threshold,
    # is nulled rather than written. Landing such a value would put output
    # the precision gate never saw into a consumer's table.
    threshold = spec.abstain_threshold
    # A name the spec never declared means the response is not about the
    # fields that were asked for, and the danger is asymmetric: the model
    # may have been declining `issuer_name` while misspelling it, in which
    # case the abstention silently evaporates and the value it was trying
    # to withhold lands as though confidently asserted. Since the intended
    # target cannot be recovered, nothing from the row is asserted and it
    # goes to the exception queue — the same treatment a poisoned document
    # gets, for the same reason.
    declared_array = (
        "array("
        + ", ".join(sql_string_literal(field.name) for field in spec.fields)
        + ")"
    )
    unknown_abstentions = f"array_except(parsed.abstained_fields, {declared_array})"
    has_unknown = f"coalesce(size({unknown_abstentions}), 0) > 0"
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
            f"{has_unknown} "
            f"OR coalesce(array_contains(parsed.abstained_fields, {literal}), false) "
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
    source_digest = f"sha2({spec.document_column}, 256)"
    request = (
        f"concat({sql_string_literal(spec.prompt_template)}, "
        f"{spec.document_column})"
    )
    return f"""MERGE INTO {spec.target_table} AS target
USING (
  WITH pending AS (
    SELECT source.{spec.key_column}, source.{spec.document_column}{pending_strata}
    FROM {source} AS source
    LEFT ANTI JOIN {spec.target_table} AS done
      ON source.{spec.key_column} = done.{spec.key_column}{restart}
  ),
  scored AS (
    SELECT
      *,
      ai_query(
        {sql_string_literal(spec.endpoint)},
        {request},
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
    CASE
      WHEN {has_unknown}
        THEN concat(
          'response declined fields the spec never declared: ',
          concat_ws(', ', {unknown_abstentions})
        )
      ELSE error_message
    END AS ai_error,
    {sql_string_literal(run_id)} AS ai_run_id,
    {sql_string_literal(run_id)} AS ai_policy_run_id,
    {digest_literal} AS ai_spec_digest,
    {model_literal} AS ai_model_version,
    {prompt_literal} AS ai_prompt_version,
    {sequence} AS ai_release_sequence,
    {source_version} AS ai_source_version,
    {source_version} AS ai_strata_version,
    {sql_string_literal(spec.inference_digest)} AS ai_inference_digest,
    {source_digest} AS ai_source_digest,
    current_timestamp() AS ai_executed_at
  FROM parsed
) AS source
ON target.{spec.key_column} = source.{spec.key_column}
WHEN MATCHED
  AND (
    coalesce(target.ai_release_sequence, -1) < source.ai_release_sequence
    OR (
      coalesce(target.ai_release_sequence, -1) = source.ai_release_sequence
      AND coalesce(target.ai_source_version, -1) <= source.ai_source_version
    )
  )
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


#: Columns of the run-metadata table, in order. One definition so the
#: DDL and the migration cannot disagree about what the table holds.
RUN_METADATA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run_id", "STRING"),
    ("spec_name", "STRING"),
    ("spec_digest", "STRING"),
    ("spec_yaml", "STRING"),
    ("use_tier", "INT"),
    ("endpoint", "STRING"),
    ("model_version", "STRING"),
    ("prompt_version", "STRING"),
    ("gate_decision", "STRING"),
    ("approved_by", "STRING"),
    ("projected_cost_cad", "DOUBLE"),
    ("source_table", "STRING"),
    ("source_table_version", "BIGINT"),
    ("target_table", "STRING"),
    ("target_table_version", "BIGINT"),
    ("executed_at", "TIMESTAMP"),
)


def run_metadata_migration_sql(
    spec: BatchInferenceSpec, existing_columns: Sequence[str]
) -> tuple[str, ...]:
    """Additive migration for a run-metadata table that predates a column.

    ``CREATE TABLE IF NOT EXISTS`` does nothing to a table that already
    exists, so adding a column to the DDL silently breaks every existing
    installation: the reservation and conflict queries reference it and
    fail on an unresolved column before anything runs. The target table
    has had this treatment since early on; the metadata table was given
    new columns without it, which is the whole defect.

    Returns the statements needed, empty when the table is current or
    absent. Additive only — a column this pipeline no longer writes is
    left alone, because dropping it would destroy recorded provenance.
    """
    present = {column.casefold() for column in existing_columns}
    if not present:  # table does not exist yet; the DDL covers it
        return ()
    missing = [
        f"{name} {sql_type}"
        for name, sql_type in RUN_METADATA_COLUMNS
        if name.casefold() not in present
    ]
    if not missing:
        return ()
    return (
        f"ALTER TABLE {spec.run_metadata_table} ADD COLUMNS ({', '.join(missing)})",
    )


def create_run_metadata_table_sql(spec: BatchInferenceSpec) -> str:
    """Layer 3: the run record everything joins back to by ``run_id``.

    Pair this with ``run_metadata_migration_sql`` for tables that already
    exist: this statement will not alter one.
    """
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
  source_table STRING,
  source_table_version BIGINT,
  target_table STRING,
  target_table_version BIGINT,
  executed_at TIMESTAMP
)"""


def run_metadata_reserve_sql(
    spec: BatchInferenceSpec,
    report: GateReport | None,
    *,
    run_id: str,
    estimate: CostEstimate,
    preflight: SourcePreflight,
) -> str:
    """Write the run record, keyed on ``run_id`` so a retry cannot duplicate it.

    A plain INSERT would add a second row whenever the client lost the
    response and the cell was re-run. ``ai_run_id`` is the join key every
    landed row uses to reach this table, so a duplicate fans out every
    downstream join and can tie one run to two recorded table versions —
    it corrupts the provenance record rather than merely repeating it.

    The matched branch is deliberately absent. A run record describes
    something that already happened, so there is nothing about it a later
    call may legitimately revise: updating on match meant a retry after
    another commit — or an accidentally reused ``run_id`` — rewrote the
    spec, gate, cost and table version that the rows carrying that
    ``ai_run_id`` were produced under, silently repointing them at a
    release they never ran. Insert-only makes the retry a true no-op and
    the record immutable; ``run_metadata_conflict_sql`` is how a
    conflicting reuse is *found*, because SQL cannot raise here.

    **Run this before execution, not after.** Every landed row carries
    ``ai_run_id``, and the policy resync writes ``ai_policy_run_id``, so
    writing the record last leaves a window where an interruption strands
    rows pointing at a run that does not exist — permanently, since
    nothing revisits them. Reserving it first closes that window.

    ``target_table_version`` is left NULL here: it is the one fact that
    genuinely cannot be known until the write has happened.
    ``run_metadata_finalize_sql`` fills it in exactly once, and only from
    NULL, so nothing already recorded is ever revised.

    The report and estimate go through the same authorisation check the
    execute statement uses, rather than arriving as loose fragments. A
    bare ``projected_cost_cad`` float and an unvalidated report let one
    record combine this spec's YAML with another evaluation's gate
    decision and an unrelated cost — and every landed row's ``ai_run_id``
    resolves to that record, so the audit trail would be wrong precisely
    where it is most consulted.

    ``report`` may be ``None`` only for tier 3, which has no gate to
    produce one; those rows still carry ``ai_run_id``, so they still need
    a record to point at. The decision is stored as ``not_required``,
    which is the truth rather than a fabricated adoption.
    """
    snapshot = _require_authorised_write(spec, report, estimate, preflight)
    decision = report.decision.value if report else "not_required"
    approved = (
        sql_string_literal(report.approved_by)
        if report and report.approved_by
        else "NULL"
    )
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
    {sql_string_literal(decision)} AS gate_decision,
    {approved} AS approved_by,
    {float(estimate.projected_cost_cad)} AS projected_cost_cad,
    {sql_string_literal(spec.source_table)} AS source_table,
    {int(snapshot.version)} AS source_table_version,
    {sql_string_literal(spec.target_table)} AS target_table,
    CAST(NULL AS BIGINT) AS target_table_version,
    current_timestamp() AS executed_at
) AS source
ON target.run_id = source.run_id
WHEN NOT MATCHED THEN INSERT *"""


def run_metadata_finalize_sql(
    spec: BatchInferenceSpec,
    *,
    run_id: str,
    target_table_version: int,
) -> str:
    """Record the table version this run produced — once, and only once.

    The version cannot be known before the write, so it is the single
    field the reserved record leaves open. Setting it only where it is
    still NULL keeps the record append-only in effect: a retry after a
    later commit finds it already filled and changes nothing, rather than
    repointing rows at a version they were not produced under.
    """
    return f"""UPDATE {spec.run_metadata_table}
SET target_table_version = {int(target_table_version)}
WHERE run_id = {sql_string_literal(run_id)}
  AND target_table_version IS NULL"""


def run_metadata_conflict_sql(
    spec: BatchInferenceSpec,
    *,
    run_id: str,
    source_snapshot: SourceSnapshot | None = None,
    target_table_version: int | None = None,
) -> str:
    """Find an existing run record this run would have contradicted.

    The reserve is insert-only, so a reused ``run_id`` leaves the original
    record standing rather than overwriting it — correct, but silent. Run
    this after it: any row returned means two different runs claimed one
    id, and every landed row carrying it now resolves to the wrong one.

    A still-NULL ``target_table_version`` is not a conflict: that is this
    run's own reserved record, waiting to be finalised.

    Pass ``source_snapshot`` before execution. Identity alone accepts a
    run id reused for a *later snapshot of the same unchanged spec* — a
    nightly job is exactly that — so the reservation would stay tied to
    the old cycle while the new MERGE landed rows carrying its id, and
    the clash would surface only after the production write. The source
    version is knowable before anything is spent, which is the point.

    Omit ``target_table_version`` before execution, when this run has not
    produced one yet. Passing a placeholder there compared every real
    version against it and flagged a conflict on any same-id retry that
    reached stage 7 the first time round — turning the documented
    idempotent retry into a refusal. Identity is the only thing knowable
    that early, and identity is what makes a reuse a conflict.
    """
    snapshot_clash = (
        ""
        if source_snapshot is None
        else f"""
    OR NOT (source_table_version <=> {int(source_snapshot.version)})"""
    )
    version_clash = (
        ""
        if target_table_version is None
        else f"""
    OR (
      target_table_version IS NOT NULL
      AND target_table_version <> {int(target_table_version)}
    )"""
    )
    digest_clash = f"NOT (spec_digest <=> {sql_string_literal(spec.spec_digest)})"
    return f"""SELECT run_id, spec_digest, target_table_version
FROM {spec.run_metadata_table}
WHERE run_id = {sql_string_literal(run_id)}
  AND (
    {digest_clash}{snapshot_clash}{version_clash}
  )"""


def require_unique_run_id(spec: BatchInferenceSpec, conflicting_rows: int) -> None:
    """Refuse to carry on when a run id already describes a different run."""
    if conflicting_rows:
        raise EvidenceMismatch(
            f"{spec.run_metadata_table} already holds a different run under "
            "this run_id. The record is immutable and was left intact, so "
            "the rows this run just landed now point at another run's spec "
            "and table version. Use a fresh run id."
        )


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
    report: GateReport,
) -> None:
    """Log spec, sample, per-field per-stratum intervals, and the decision
    to the *active* MLflow run. The caller owns the run lifecycle so the
    same run can later receive the output table version. Imported lazily so
    this module stays importable without MLflow installed.

    The intervals come from ``report.scores``, not from a separate
    argument. Taking the scores independently meant a caller holding two
    evaluations — which the notebook does, by design — could log v1's
    metrics beside v2's adoption, and the record would look complete.
    The report already carries the scores its verdicts were derived from;
    a second source for them could only ever disagree.
    """
    import mlflow

    # The durable evaluation record is as much a boundary as execution
    # is: a forged adoption refused by `require_executable` was still
    # being written here, leaving MLflow claiming an adoption the gate
    # would reject.
    #
    # Every artifact, not just the report. Naming the round trip and
    # applying it to reports was supposed to end this class of defect,
    # and the estimate at this same boundary was missed in that very
    # change — so the loop runs over the arguments rather than a list of
    # names someone has to remember to extend.
    for artifact in (report, estimate):
        revalidated(artifact, EvidenceMismatch)
    require_matching_evidence(spec, report.scores)
    # The report is what the decision came from, so it has to belong to
    # the release being recorded. Checking only the snapshot let
    # `log_gate_evidence(spec_v2, estimate_v2, allocation, report_v1)`
    # write v2's parameters and YAML beside v1's decision and intervals —
    # and this notebook holds both reports for one snapshot by design, so
    # the mistake is available rather than hypothetical. Same binding the
    # reserve helper does; it was simply not swept here.
    if report.spec_digest != spec.spec_digest:
        raise EvidenceMismatch(
            f"the gate report describes spec digest "
            f"{report.spec_digest[:12]}… but this run records "
            f"{spec.spec_digest[:12]}…. The evidence would name one release "
            "and the decision another."
        )
    if report.use_tier != spec.use_tier:
        raise EvidenceMismatch(
            f"the gate report claims tier {report.use_tier}, but this run is "
            f"tier {spec.use_tier}"
        )
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
    # And to the same rows. Release identity does not pin the row count,
    # so an estimate made against an older, smaller snapshot clears a
    # ceiling the newer one would not — the run then executes against the
    # newer snapshot while this record shows the stale projection. This
    # is the one place that sees the estimate and the report together,
    # which makes it the place the bundle has to agree.
    if estimate.source_snapshot != report.source_snapshot:
        raise EvidenceMismatch(
            f"the cost estimate describes {estimate.source_snapshot} but the "
            f"gate report describes {report.source_snapshot}. The projection "
            "was made over different rows than the gate certified, so it "
            "cannot authorise this run's spend."
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
    # Derived from the evidence rather than passed alongside it. A stale
    # allocation could claim 20 documents were sampled while the Wilson
    # intervals logged beside it were computed from 200 — the persisted
    # account of the experiment contradicting its own numbers. This is
    # the realised sample, which is what the evidence actually rests on.
    realised = {
        score.stratum: score.n_rows
        for score in report.scores
        if score.stratum != WEIGHTED
    }
    mlflow.log_dict(realised, "governed_batch_inference/sample_allocation.json")
    mlflow.log_metric("projected_cost_cad", estimate.projected_cost_cad)
    for score in report.scores:
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
