"""Statistics and gate contracts for the governed batch inference example.

The gate math is the part of that example most likely to be quietly broken
by a future "simplification", so it is pinned here: Wilson lower bounds
(never point estimates), worst-stratum gating for high-criticality fields,
deliberate over-sampling of rare strata, and the tier-1 human sign-off.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "examples" / "governed-batch-inference" / "governed_batch_inference.py"

_spec = importlib.util.spec_from_file_location("governed_batch_inference", MODULE)
assert _spec is not None and _spec.loader is not None
gbi = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gbi
_spec.loader.exec_module(gbi)


def make_spec(**overrides):
    base = dict(
        name="tax_document_extraction",
        source_table="main.finance_docs.document_text",
        target_table="main.finance_docs.document_entities",
        run_metadata_table="main.finance_docs.batch_inference_runs",
        document_column="doc_text",
        key_column="doc_id",
        use_tier=2,
        consumed_by=("reconciliation_pipeline",),
        fields=(
            dict(
                name="issuer_name",
                description="Issuing organisation.",
                criticality="high",
                tolerable_error_rate=0.05,
            ),
            dict(
                name="account_id",
                description="Account identifier.",
                criticality="medium",
                tolerable_error_rate=0.05,
            ),
        ),
        strata=("layout",),
        endpoint="databricks-gpt-oss-20b",
        model_version="gpt-oss-20b-2025-08",
        prompt_version="1.0.0",
        prompt_template="Extract the fields from the document.\n\nDOCUMENT:\n",
        release_sequence=1,
        cost_ceiling_cad=50.0,
        abstain_threshold=0.7,
    )
    base.update(overrides)
    return gbi.BatchInferenceSpec.model_validate(base)


# Records must name the release that produced them. Tests build them with
# this placeholder and `score()` re-stamps to the spec under test, so the
# release-binding tests below stay the only place the stamp is meaningful.
PLACEHOLDER_INFERENCE = gbi.InferenceIdentity(
    inference_digest="placeholder",
    model_version="placeholder",
    prompt_version="placeholder",
)


def records_for(stratum, *, correct=0, wrong=0, abstained=0, hallucinated=0):
    """Single-field evaluation records with controlled outcome counts."""

    def record(**kwargs):
        return gbi.EvaluationRecord(
            stratum=stratum, inference=PLACEHOLDER_INFERENCE, **kwargs
        )

    rows = []
    for _ in range(correct):
        rows.append(record(gold={"f": "v"}, predicted={"f": "v"}))
    for _ in range(wrong):
        rows.append(record(gold={"f": "v"}, predicted={"f": "x"}))
    for _ in range(abstained):
        rows.append(
            record(
                gold={"f": "v"},
                predicted={"f": None},
                abstained=frozenset({"f"}),
            )
        )
    for _ in range(hallucinated):
        rows.append(record(gold={"f": None}, predicted={"f": "made-up"}))
    return rows


def one_field(criticality="high", tolerable_error_rate=0.05):
    return gbi.FieldSpec(
        name="f",
        description="test field",
        criticality=criticality,
        tolerable_error_rate=tolerable_error_rate,
    )


def one_field_spec(criticality="high", tolerable_error_rate=0.05, **overrides):
    """A spec whose single field is the one `records_for` produces."""
    return make_spec(
        fields=(
            dict(
                name="f",
                description="test field",
                criticality=criticality,
                tolerable_error_rate=tolerable_error_rate,
            ),
        ),
        **overrides,
    )


_UNSET = object()


def gate(spec, scores, **kwargs):
    """`evaluate_gate`, supplying the snapshot gated evidence now requires.

    Tiers 1 and 2 must record which source version their evidence
    describes, so the run can be pinned to it. That is uninteresting to
    most tests here, so it is defaulted; the tests that are *about* the
    snapshot call `gbi.evaluate_gate` directly.
    """
    if spec.gate_required:
        kwargs.setdefault("source_snapshot", snapshot_for(spec))
    return gbi.evaluate_gate(spec, scores, **kwargs)


ESTIMATED_ROWS = 1_000


def preflight_matching(report, spec, row_count=ESTIMATED_ROWS):
    """A preflight whose measured population matches this report's weights.

    `require_executable` compares the two, so a test that builds a report
    over stratum "s" needs a preflight that measured stratum "s".
    """
    weights: dict[str, int] = {}
    for score in report.scores:
        if score.stratum == gbi.WEIGHTED:
            weights.update(dict(score.stratum_population))
    return preflight_for(
        spec, report.source_snapshot or snapshot_for(spec), row_count, weights
    )


def preflight_for(
    spec, source_snapshot=_UNSET, row_count=ESTIMATED_ROWS, population=None
):
    """A clean preflight for the snapshot `estimate_for` prices.

    Holding one is proof the usability checks passed, which is what the
    builder now requires instead of trusting call order.
    """
    if source_snapshot is _UNSET:
        # Every paid run is pinned now, gated or not — the preflight
        # describes the table, and tier does not change that.
        source_snapshot = snapshot_for(spec)
    if population is None:
        # Weighted evidence is checked against measured counts, so a
        # preflight needs a population that matches the strata the
        # evidence covers. Both sample builders here use 200 rows.
        population = {"standard": 200}
    return gbi.require_usable_source_rows(
        spec,
        0,
        0,
        0,
        snapshot=source_snapshot,
        row_count=row_count,
        stratum_population=population,
    )


def estimate_for(spec, source_snapshot=_UNSET):
    """A within-ceiling estimate matching what `adopting_report` pins.

    The builder emits a paid statement, so it now requires an approved
    budget for exactly these rows — release and snapshot both.
    """
    if source_snapshot is _UNSET:
        source_snapshot = snapshot_for(spec)
    return gbi.estimate_cost(
        spec,
        row_count=ESTIMATED_ROWS,
        probe_input_tokens=[100],
        probe_output_tokens=[100],
        cad_per_million_input_tokens=0.1,
        cad_per_million_output_tokens=0.1,
        source_snapshot=source_snapshot,
    )


def snapshot_for(spec, version=7):
    """Gated evidence must record the source version it describes."""
    return gbi.SourceSnapshot(table=spec.source_table, version=version)


def adopting_report(spec, *, source_snapshot=None):
    """A passing gate report for `spec` — what execution now requires.

    A gated spec builds its execute statement from the report that
    authorised it, so tests exercising the SQL need one. Every declared
    field scores perfectly on a single stratum.
    """
    records = [
        gbi.EvaluationRecord(
            stratum="standard",
            inference=PLACEHOLDER_INFERENCE,
            gold={field.name: "v" for field in spec.fields},
            predicted={field.name: "v" for field in spec.fields},
        )
    ] * 200
    if source_snapshot is None and spec.gate_required:
        source_snapshot = snapshot_for(spec)
    report = gbi.evaluate_gate(
        spec, score(records, spec), source_snapshot=source_snapshot
    )
    if report.decision == gbi.GateDecision.PENDING_APPROVAL:
        report = gbi.approve_gate(report, "platform-team")
    return report


def score(records, spec, population=None):
    """Scoring is bound to the spec being gated and to the population.

    Default population: each sampled stratum weighted by its own sample
    size, i.e. the sample is treated as proportional. Tests that care
    about weighting pass a real population.
    """
    if population is None:
        counts: dict[str, int] = {}
        for record in records:
            counts[record.stratum] = counts.get(record.stratum, 0) + 1
        population = counts
    stamped = [
        record.model_copy(update={"inference": spec.inference}) for record in records
    ]
    return gbi.score_extraction(stamped, spec, population)


# ---------------------------------------------------------------------------
# Wilson score intervals
# ---------------------------------------------------------------------------


def test_wilson_97_of_100_fails_a_95_percent_tolerance():
    """The canonical lesson: an encouraging point estimate with too little
    evidence behind it. 97% from n=100 has a lower bound near 91.5%."""
    interval = gbi.wilson_interval(97, 100, 0.95)
    assert interval.point == pytest.approx(0.97)
    assert interval.lower == pytest.approx(0.91548, abs=5e-5)
    assert interval.point >= 0.95
    assert interval.lower < 0.95


def test_wilson_known_values_and_bounds():
    interval = gbi.wilson_interval(490, 500, 0.95)
    assert interval.lower == pytest.approx(0.96358, abs=5e-5)
    perfect = gbi.wilson_interval(100, 100, 0.95)
    # Closed form at p_hat = 1: lower bound reduces to n / (n + z^2).
    assert perfect.lower == pytest.approx(0.963007, abs=5e-5)
    assert perfect.upper == 1.0
    nothing = gbi.wilson_interval(0, 50, 0.95)
    assert nothing.lower == 0.0
    assert nothing.upper > 0.0


def test_wilson_narrows_with_evidence_and_confidence():
    small = gbi.wilson_interval(97, 100, 0.95)
    large = gbi.wilson_interval(970, 1000, 0.95)
    assert large.lower > small.lower
    stricter = gbi.wilson_interval(97, 100, 0.99)
    assert stricter.lower < small.lower


def test_an_interval_cannot_claim_bounds_its_counts_do_not_support():
    """Gate reports round-trip through MLflow as JSON, and the gate reads
    `lower` directly — so a reconstructed interval must be checked against
    its own counts, not trusted."""
    honest = gbi.wilson_interval(0, 10, 0.95)
    assert honest.lower == pytest.approx(0.0, abs=1e-12)

    # 0/10 with a lower bound of 1.0 would adopt any release.
    with pytest.raises(ValidationError, match="does not match the Wilson value"):
        gbi.ConfidenceInterval(
            successes=0, trials=10, confidence=0.95, point=1.0, lower=1.0, upper=1.0
        )
    # Subtler: honest counts, one nudged bound.
    with pytest.raises(ValidationError, match="does not match the Wilson value"):
        honest.model_copy(update={"lower": 0.96}).model_validate(
            honest.model_copy(update={"lower": 0.96}).model_dump()
        )
    # Out-of-range values are refused before the arithmetic check.
    with pytest.raises(ValidationError):
        gbi.ConfidenceInterval(
            successes=5, trials=10, confidence=0.95, point=1.5, lower=0.0, upper=1.0
        )
    # A faithful round-trip still validates.
    assert (
        gbi.ConfidenceInterval.model_validate(honest.model_dump(mode="json")) == honest
    )


def test_wilson_rejects_bad_input():
    with pytest.raises(ValueError):
        gbi.wilson_interval(0, 0)
    with pytest.raises(ValueError):
        gbi.wilson_interval(5, 4)
    with pytest.raises(ValueError):
        gbi.wilson_interval(1, 10, confidence=1.5)


def test_min_labelled_rows_is_the_exact_feasibility_boundary():
    assert gbi.min_labelled_rows_for_tolerance(0.05, 0.95) == 73
    assert gbi.min_labelled_rows_for_tolerance(0.01, 0.95) == 381
    for tolerance in (0.05, 0.10, 0.02):
        required = 1.0 - tolerance
        n = gbi.min_labelled_rows_for_tolerance(tolerance, 0.95)
        assert gbi.wilson_interval(n, n, 0.95).lower >= required
        assert gbi.wilson_interval(n - 1, n - 1, 0.95).lower < required


# ---------------------------------------------------------------------------
# Stratified sample allocation
# ---------------------------------------------------------------------------


def test_rare_strata_are_deliberately_over_sampled():
    population = {"standard": 24500, "legacy_scan": 500}
    allocation = gbi.allocate_stratified_sample(population, 400, 150)
    assert sum(allocation.values()) == 400
    # Proportional would give legacy_scan 400 * 500/25000 = 8 rows —
    # almost no information about the hard case. The floor forces 150+.
    assert allocation["legacy_scan"] >= 150
    assert allocation["standard"] > allocation["legacy_scan"]


def test_allocation_never_exceeds_stratum_population():
    allocation = gbi.allocate_stratified_sample({"tiny": 10, "big": 1000}, 100, 30)
    assert allocation == {"tiny": 10, "big": 90}


def test_allocation_is_deterministic_and_exact():
    population = {"a": 700, "b": 200, "c": 100}
    first = gbi.allocate_stratified_sample(population, 250, 50)
    second = gbi.allocate_stratified_sample(population, 250, 50)
    assert first == second
    assert sum(first.values()) == 250
    assert all(first[s] >= min(50, population[s]) for s in population)


def test_allocation_fails_loudly_when_budget_cannot_cover_floors():
    with pytest.raises(ValueError, match="labelling budget"):
        gbi.allocate_stratified_sample({"a": 500, "b": 500, "c": 500}, 100, 50)


def test_allocation_caps_at_full_population():
    population = {"a": 30, "b": 20}
    assert gbi.allocate_stratified_sample(population, 500, 10) == population


# ---------------------------------------------------------------------------
# Scoring semantics: precision and recall are different failures
# ---------------------------------------------------------------------------


def test_hallucination_hurts_precision_and_abstention_hurts_recall():
    records = records_for("s", correct=8, abstained=1, hallucinated=1)
    (result,) = [s for s in score(records, one_field_spec()) if s.stratum == "s"]
    # 9 assertions (8 correct + 1 hallucinated); the abstention is not one.
    assert result.n_asserted == 9
    assert result.precision.successes == 8
    assert result.precision.trials == 9
    # 9 rows have a true value (hallucination row has gold None); the
    # abstained row counts as a miss for recall — visible, but a miss.
    assert result.recall.successes == 8
    assert result.recall.trials == 9
    assert result.abstention_rate == pytest.approx(0.1)


def test_scores_include_pooled_and_per_stratum_rows():
    records = records_for("a", correct=5) + records_for("b", correct=5, wrong=5)
    scores = score(records, one_field_spec())
    strata = {s.stratum for s in scores}
    assert strata == {"a", "b", gbi.WEIGHTED}
    pooled = next(s for s in scores if s.stratum == gbi.WEIGHTED)
    assert pooled.n_rows == 15
    assert pooled.precision.successes == 10


def test_scores_carry_the_release_and_confidence_they_measured():
    spec = one_field_spec()
    scores = score(records_for("s", correct=10), spec)
    assert all(s.release == spec.release for s in scores)
    assert all(s.confidence == spec.confidence_level for s in scores)
    assert spec.release.prompt_version == spec.prompt_version
    assert spec.release.spec_digest == spec.spec_digest


def test_scoring_honours_a_non_default_confidence_from_the_spec():
    strict_spec = one_field_spec(confidence_level=0.99)
    (result,) = [
        s
        for s in score(records_for("s", correct=97, wrong=3), strict_spec)
        if s.stratum == "s"
    ]
    assert result.confidence == 0.99
    assert result.precision.confidence == 0.99
    # A 99% interval is wider, so its lower bound sits below the 95% one.
    assert result.precision.lower < gbi.wilson_interval(97, 100, 0.95).lower


def test_values_match_is_numeric_aware_and_case_tolerant():
    assert gbi.values_match("$12,345.60", "12345.6")
    assert gbi.values_match("Maple Grove  Capital", " maple grove capital ")
    assert not gbi.values_match("12345.60", "12345.61")
    assert not gbi.values_match(None, "anything")
    assert not gbi.values_match("anything", None)


# ---------------------------------------------------------------------------
# The gate: lower bounds, worst stratum, tiers
# ---------------------------------------------------------------------------


CENTRAL_LESSON_RECORDS = records_for("standard", correct=795, wrong=5) + records_for(
    "legacy_scan", correct=80, wrong=20
)
"""Aggregate looks fine; the minority stratum is broken.

standard: 795/800 correct (lower ~0.985). legacy_scan: 80/100
(lower ~0.711). Pooled: 875/900, lower ~0.959 — which *passes* 0.95.
"""


def test_high_criticality_gates_on_the_worst_stratum_not_the_average():
    spec = one_field_spec(criticality="high")
    report = gate(spec, score(CENTRAL_LESSON_RECORDS, spec))
    assert report.decision == gbi.GateDecision.REJECT
    (result,) = report.fields
    assert result.decision == gbi.GateDecision.REJECT
    assert result.binding_stratum == "legacy_scan"
    assert result.binding_lower_bound == pytest.approx(0.71117, abs=5e-5)


def test_medium_criticality_gates_on_the_population_weighted_estimate():
    """Same evidence, medium criticality: the all-strata estimate decides."""
    spec = one_field_spec(criticality="medium")
    report = gate(spec, score(CENTRAL_LESSON_RECORDS, spec))
    assert report.decision == gbi.GateDecision.ADOPT
    (result,) = report.fields
    assert result.binding_stratum == gbi.WEIGHTED


def test_weighted_estimate_is_not_the_raw_pool_of_a_stratified_sample():
    """Over-sampling a hard stratum must not distort the population rate.

    Population is 1% hard, but the sample is 50% hard by design. Pooling
    the sample's rows would describe a population that does not exist and
    reject a field whose true rate is fine.
    """
    spec = one_field_spec(criticality="medium", tolerable_error_rate=0.10)
    records = records_for("easy", correct=100) + records_for(
        "hard", correct=60, wrong=40
    )
    population = {"easy": 9900, "hard": 100}

    raw_pool = gbi.wilson_interval(160, 200, spec.confidence_level)
    assert raw_pool.lower < 0.90  # pooling would reject

    weighted = next(
        s for s in score(records, spec, population) if s.stratum == gbi.WEIGHTED
    )
    assert weighted.precision.point > 0.99
    assert weighted.precision.lower >= 0.90
    assert gate(spec, score(records, spec, population)).decision == (
        gbi.GateDecision.ADOPT
    )
    # And the effective sample size never exceeds what was labelled.
    assert weighted.precision.trials <= 200


def test_weighted_estimate_needs_a_population_for_every_stratum():
    spec = one_field_spec(criticality="medium")
    records = [
        record.model_copy(update={"inference": spec.inference})
        for record in records_for("easy", correct=10) + records_for("hard", correct=10)
    ]
    with pytest.raises(ValueError, match="no population count"):
        gbi.score_extraction(records, spec, {"easy": 100})
    with pytest.raises(ValueError, match="not in the sample"):
        gbi.score_extraction(
            records, spec, {"easy": 100, "hard": 100, "never_sampled": 50}
        )


def test_gate_compares_lower_bound_never_the_point_estimate():
    """97/100 with a 95% tolerance: the point estimate passes, the run
    does not — it produced an encouraging number with too little evidence."""
    spec = one_field_spec(criticality="medium")
    scores = score(records_for("standard", correct=97, wrong=3), spec)
    report = gate(spec, scores)
    (result,) = report.fields
    assert result.binding_point_estimate >= 0.95
    assert result.binding_lower_bound < 0.95
    assert report.decision == gbi.GateDecision.REJECT
    assert any("point estimate" in reason for reason in result.reasons)


def test_too_small_a_sample_is_inconclusive_not_a_rejection():
    """30/30 correct cannot clear a 95% bar (best possible lower bound is
    ~0.886): the model was not shown to be bad — the sample was too small."""
    spec = one_field_spec(criticality="high")
    scores = score(records_for("legacy", correct=30), spec)
    report = gate(spec, scores)
    assert report.decision == gbi.GateDecision.INCONCLUSIVE
    (result,) = report.fields
    assert any("label more rows" in reason for reason in result.reasons)


def test_a_decisive_failure_is_rejected_even_at_the_same_small_size():
    """0/30 is the same size as 30/30 but not the same evidence.

    Its whole interval sits below the bar, so it is a demonstrated
    failure — telling the team to label more rows would waste a week
    confirming what the first 30 rows already showed.
    """
    spec = one_field_spec(criticality="high")
    interval = gbi.wilson_interval(0, 30, spec.confidence_level)
    assert interval.upper < 0.95  # decisive, despite n = 30

    report = gate(spec, score(records_for("legacy", wrong=30), spec))
    assert report.decision == gbi.GateDecision.REJECT
    (result,) = report.fields
    assert any("demonstrated failure" in reason for reason in result.reasons)
    assert not any("label more rows" in reason for reason in result.reasons)


# ---------------------------------------------------------------------------
# Evidence binding: scores belong to the release that produced them
# ---------------------------------------------------------------------------


def test_gate_refuses_evidence_from_a_different_release():
    """A passing v1 sample must not be able to authorise a v2 run.

    Without this the report would carry v2's digest over v1's numbers and
    require_executable would accept it — an unvalidated prompt executing
    on an earlier release's evidence.
    """
    spec_v1 = one_field_spec(criticality="medium", prompt_version="1.0.0")
    spec_v2 = spec_v1.model_copy(update={"prompt_version": "2.0.0"})
    scores_v1 = score(records_for("standard", correct=200), spec_v1)

    assert gate(spec_v1, scores_v1).decision == gbi.GateDecision.ADOPT
    with pytest.raises(gbi.EvidenceMismatch, match="prompt 1.0.0"):
        gate(spec_v2, scores_v1)


def test_gate_refuses_a_model_version_change_on_stale_evidence():
    spec = one_field_spec(criticality="medium")
    scores = score(records_for("standard", correct=200), spec)
    retrained = spec.model_copy(update={"model_version": "next-model-build"})
    with pytest.raises(gbi.EvidenceMismatch):
        gate(retrained, scores)


def test_gate_refuses_intervals_computed_at_another_confidence_level():
    """Declared 99%, scored at 95% — 485/500 clears 95% but not 99%."""
    spec = one_field_spec(criticality="medium", confidence_level=0.99)
    records = records_for("standard", correct=485, wrong=15)
    at_95 = gbi.wilson_interval(485, 500, 0.95)
    at_99 = gbi.wilson_interval(485, 500, 0.99)
    assert at_95.lower >= 0.95 > at_99.lower  # the gap the check protects

    mislabelled = tuple(
        s.model_copy(update={"confidence": 0.95}) for s in score(records, spec)
    )
    with pytest.raises(gbi.EvidenceMismatch, match="confidence"):
        gate(spec, mislabelled)
    # Scored properly at the declared level, the same sample is rejected.
    assert gate(spec, score(records, spec)).decision == (gbi.GateDecision.REJECT)


def test_gate_refuses_an_empty_evidence_set():
    with pytest.raises(gbi.EvidenceMismatch):
        gate(one_field_spec(), ())


def test_gate_refuses_evidence_with_a_stratum_filtered_out():
    """Dropping the failing stratum must fail the gate, not weaken it.

    The worst-stratum rule is only as good as the strata it is handed, so
    the evidence carries the sample's stratum manifest and the gate checks
    the set is complete.
    """
    spec = one_field_spec(criticality="high")
    scores = score(CENTRAL_LESSON_RECORDS, spec)
    assert gate(spec, scores).decision == gbi.GateDecision.REJECT

    without_the_bad_one = [s for s in scores if s.stratum != "legacy_scan"]
    with pytest.raises(gbi.EvidenceMismatch, match="incomplete"):
        gate(spec, without_the_bad_one)


def test_gate_refuses_evidence_missing_a_field_or_the_weighted_row():
    spec = make_spec()  # two fields
    records = [
        gbi.EvaluationRecord(
            stratum="standard",
            inference=PLACEHOLDER_INFERENCE,
            gold={"issuer_name": "v", "account_id": "v"},
            predicted={"issuer_name": "v", "account_id": "v"},
        )
    ] * 200
    scores = score(records, spec)
    assert gate(spec, scores).decision == gbi.GateDecision.ADOPT

    with pytest.raises(gbi.EvidenceMismatch, match="incomplete"):
        gate(spec, [s for s in scores if s.field != "account_id"])
    with pytest.raises(gbi.EvidenceMismatch, match="incomplete"):
        gate(spec, [s for s in scores if s.stratum != gbi.WEIGHTED])


def test_scoring_refuses_records_produced_by_a_different_release():
    """The stamp has to come from where the prediction was made.

    Taking it from the spec handed to `score_extraction` would let v1
    output certify itself as v2 evidence, and the gate's release check —
    which reads that same stamp — would wave it through.
    """
    spec_v1 = one_field_spec(criticality="high")
    spec_v2 = spec_v1.model_copy(update={"prompt_version": "2.0.0"})
    v1_records = [
        record.model_copy(update={"inference": spec_v1.inference})
        for record in records_for("standard", correct=200)
    ]
    population = {"standard": 200}

    gbi.score_extraction(v1_records, spec_v1, population)  # its own output: fine
    with pytest.raises(gbi.EvidenceMismatch, match="Re-run inference"):
        gbi.score_extraction(v1_records, spec_v2, population)

    # One stale row among fresh ones is caught too.
    mixed = [
        record.model_copy(update={"inference": spec_v2.inference})
        for record in v1_records
    ]
    mixed[7] = mixed[7].model_copy(update={"inference": spec_v1.inference})
    with pytest.raises(gbi.EvidenceMismatch):
        gbi.score_extraction(mixed, spec_v2, population)


def test_editing_the_prompt_invalidates_records_even_at_the_same_version():
    """`prompt_version` is a label someone types; the text is the fact.

    Binding evidence to the label alone means an edited prompt keeps
    certifying itself as the release that was actually measured.
    """
    original = one_field_spec()
    edited = original.model_copy(
        update={"prompt_template": original.prompt_template + " Be concise."}
    )
    assert edited.prompt_version == original.prompt_version
    assert edited.inference != original.inference

    records = [
        record.model_copy(update={"inference": original.inference})
        for record in records_for("standard", correct=200)
    ]
    with pytest.raises(gbi.EvidenceMismatch):
        gbi.score_extraction(records, edited, {"standard": 200})


def test_execute_sql_uses_the_spec_prompt_and_takes_no_prompt_argument():
    """One source of truth: the statement cannot run a prompt the stamps
    do not describe, because there is nowhere else to supply one."""
    spec = make_spec(prompt_template="Read this slip.\n\nDOCUMENT:\n")
    sql = gbi.build_execute_sql(
        spec,
        run_id="run-1",
        estimate=estimate_for(spec),
        preflight=preflight_for(spec),
        report=adopting_report(spec),
    )
    assert "concat('Read this slip.\\n\\nDOCUMENT:\\n', doc_text)" in sql
    with pytest.raises(TypeError):
        gbi.build_execute_sql(spec, run_id="run-1", **{"prompt_sql": "anything"})


def test_a_changed_source_document_becomes_pending_again():
    """Correct a document in place and the target must not keep serving
    values derived from text that no longer exists."""
    spec = make_spec()
    sql = gbi.build_execute_sql(
        spec,
        run_id="run-1",
        estimate=estimate_for(spec),
        preflight=preflight_for(spec),
        report=adopting_report(spec),
    )
    anti_join = sql.split("scored AS")[0]
    assert "done.ai_source_digest = sha2(source.doc_text, 256)" in anti_join
    assert "sha2(doc_text, 256) AS ai_source_digest" in sql
    assert ("ai_source_digest", "STRING") in gbi.target_columns(spec)


def test_release_sequence_comparisons_survive_migrated_null_rows():
    """Migration adds the column to legacy rows as NULL, and NULL
    comparisons are unknown — so without coalesce the MERGE silently
    declines to update those rows while the anti-join keeps re-selecting
    (and re-paying for) them."""
    spec = make_spec()
    sql = gbi.build_execute_sql(
        spec,
        run_id="run-1",
        estimate=estimate_for(spec),
        preflight=preflight_for(spec),
        report=adopting_report(spec),
    )
    assert "coalesce(done.ai_release_sequence, -1) > 1" in sql
    # Both halves of the ordering pair coalesce, on both sides.
    for column in ("ai_release_sequence", "ai_source_version"):
        assert f"coalesce(done.{column}, -1)" in sql
        assert f"coalesce(target.{column}, -1)" in sql

    # Either ordering column arriving by migration is backfilled, so a
    # legacy row sorts below every real run instead of poisoning the
    # comparison with NULL.
    for column in ("ai_release_sequence", "ai_source_version"):
        legacy = [name for name, _ in gbi.target_columns(spec) if name != column]
        migration = gbi.plan_target_migration(spec, legacy)
        assert any("ADD COLUMNS" in statement for statement in migration.statements)
        assert any(
            statement.startswith(f"UPDATE {spec.target_table} SET {column} = -1")
            for statement in migration.statements
        )


def test_a_gate_report_cannot_claim_a_decision_its_fields_do_not_support():
    """The report authorises a paid, table-mutating run, and
    `require_executable` reads only the aggregate — so the aggregate is
    derived from the field results rather than trusted."""
    spec = one_field_spec(criticality="high")
    honest = gate(spec, score(records_for("s", correct=200), spec))
    assert honest.decision == gbi.GateDecision.ADOPT

    # A truncated artifact: no field results, confident verdict.
    with pytest.raises(ValidationError):
        gbi.GateReport(
            spec_name=spec.name,
            spec_digest=spec.spec_digest,
            use_tier=spec.use_tier,
            confidence_level=spec.confidence_level,
            fields=(),
            decision=gbi.GateDecision.ADOPT,
        )
    # A rejecting field result relabelled as an adoption.
    rejected = gate(spec, score(records_for("s", correct=80, wrong=20), spec))
    assert rejected.decision == gbi.GateDecision.REJECT
    with pytest.raises(ValidationError, match="does not follow"):
        gbi.GateReport.model_validate(
            {**rejected.model_dump(mode="json"), "decision": "adopt"}
        )
    # A round-trip of an honest report still validates.
    assert gbi.GateReport.model_validate(honest.model_dump(mode="json")) == honest


def test_execution_needs_a_report_that_judged_every_field():
    """A matching digest says the report describes this spec; it does not
    say the report judged all of it."""
    spec = make_spec()  # two fields
    records = [
        gbi.EvaluationRecord(
            stratum="standard",
            inference=PLACEHOLDER_INFERENCE,
            gold={"issuer_name": "v", "account_id": "v"},
            predicted={"issuer_name": "v", "account_id": "v"},
        )
    ] * 200
    report = gate(spec, score(records, spec))
    gbi.require_executable(spec, report, preflight_for(spec))

    partial = report.model_copy(
        update={"fields": tuple(f for f in report.fields if f.field != "account_id")}
    )
    with pytest.raises(gbi.GateNotPassed, match="account_id"):
        gbi.require_executable(spec, partial, preflight_for(spec))


def test_a_score_cannot_borrow_another_groups_intervals():
    """The gate reads intervals, never the raw counts beside them."""
    spec = one_field_spec()
    real = next(
        s
        for s in score(records_for("s", correct=200), spec)
        if s.stratum != gbi.WEIGHTED
    )
    payload = real.model_dump(mode="json")

    # Claim no observations while keeping the 200/200 intervals: refused
    # for having an interval where there is no denominator at all.
    with pytest.raises(ValidationError, match="must be absent, not an interval"):
        gbi.FieldStratumScore.model_validate(
            {**payload, "n_correct": 0, "n_asserted": 0, "n_gold": 0}
        )
    # Claim fewer observations than the interval reports.
    with pytest.raises(ValidationError, match="this score counted"):
        gbi.FieldStratumScore.model_validate(
            {**payload, "n_correct": 5, "n_asserted": 5, "n_gold": 5}
        )
    # Or keep the counts and swap in a stronger interval.
    stronger = gbi.wilson_interval(400, 400, spec.confidence_level)
    with pytest.raises(ValidationError, match="this score counted"):
        gbi.FieldStratumScore.model_validate(
            {**payload, "precision": stronger.model_dump(mode="json")}
        )
    # An honest round-trip is unaffected.
    assert gbi.FieldStratumScore.model_validate(payload) == real


def test_strata_are_resynced_without_paying_for_inference():
    """A corrected label is metadata drift, not new model output."""
    spec = make_spec()
    sql = gbi.resync_strata_sql(spec)
    assert sql.startswith("MERGE INTO main.finance_docs.document_entities AS target")
    assert "USING main.finance_docs.document_text AS source" in sql
    assert "NOT (target.layout <=> source.layout)" in sql  # null-safe
    assert "target.layout = source.layout" in sql
    # It touches only strata — no ai_ column and no inference call.
    assert "ai_query" not in sql
    # It advances the ordering column, but never an extracted value.
    assert "target.ai_strata_version" in sql.split("THEN UPDATE SET")[1]
    assert not any(
        gbi.ai_column(f.name) in sql.split("THEN UPDATE SET")[1] for f in spec.fields
    )


def test_the_weighted_row_is_recomputed_not_trusted():
    """Medium/low fields are gated on the aggregate alone, and its
    intervals cannot be tied to a row count — so it is rebuilt from the
    physical rows before it is read as evidence."""
    spec = one_field_spec(criticality="medium")
    population = {"standard": 9000, "legacy_scan": 1000}
    honest = score(
        records_for("standard", correct=200) + records_for("legacy_scan", wrong=200),
        spec,
        population,
    )
    # The physical strata disagree, and the honest aggregate reflects it.
    assert gate(spec, honest).decision == gbi.GateDecision.REJECT

    weighted = next(s for s in honest if s.stratum == gbi.WEIGHTED)
    others = [s for s in honest if s.stratum != gbi.WEIGHTED]

    # Forge the aggregate's interval while keeping every stamp intact.
    strong = gbi.wilson_interval(200, 200, spec.confidence_level)
    forged = weighted.model_copy(update={"precision": strong, "recall": strong})
    with pytest.raises(gbi.EvidenceMismatch, match="does not follow from its stratum"):
        gate(spec, [*others, forged])

    # Forging the counts underneath it fails the same way.
    with pytest.raises(gbi.EvidenceMismatch, match="does not follow from its stratum"):
        gate(spec, [*others, weighted.model_copy(update={"n_correct": 400})])

    # So does quietly dropping the failing stratum's weight, which would
    # otherwise re-weight the population around the passing rows.
    lightened = weighted.model_copy(
        update={"stratum_population": (("legacy_scan", 1), ("standard", 9000))}
    )
    with pytest.raises(gbi.EvidenceMismatch, match="does not follow from its stratum"):
        gate(spec, [*others, lightened])

    # An honest set round-trips through JSON and still gates identically.
    revived = [
        gbi.FieldStratumScore.model_validate(s.model_dump(mode="json")) for s in honest
    ]
    assert gate(spec, revived).decision == gbi.GateDecision.REJECT


def test_the_weighted_row_must_carry_the_weights_it_used():
    """Recomputation is only possible because the weights are persisted."""
    spec = one_field_spec()
    scores = score(records_for("s", correct=200), spec)
    weighted = next(s for s in scores if s.stratum == gbi.WEIGHTED)
    physical = next(s for s in scores if s.stratum != gbi.WEIGHTED)

    assert dict(weighted.stratum_population) == {"s": 200}
    assert physical.stratum_population == ()
    with pytest.raises(ValidationError, match="must carry the weights"):
        weighted.model_copy(update={"stratum_population": ()}).model_validate(
            {**weighted.model_dump(mode="json"), "stratum_population": []}
        )
    # A physical row carrying weights is the aggregate wearing a disguise.
    with pytest.raises(ValidationError, match="carries no population weights"):
        gbi.FieldStratumScore.model_validate(
            {**physical.model_dump(mode="json"), "stratum_population": [["s", 200]]}
        )


def test_execution_reads_the_delta_version_the_evidence_describes():
    """Population, sample, gate and cost all describe one snapshot; the run
    must read that one, not whatever the table has become since."""
    spec = make_spec()
    snapshot = gbi.SourceSnapshot(table=spec.source_table, version=42)
    report = adopting_report(spec, source_snapshot=snapshot)
    assert report.source_snapshot == snapshot

    sql = gbi.build_execute_sql(
        spec,
        run_id="run-1",
        estimate=estimate_for(spec, snapshot),
        preflight=preflight_for(spec, snapshot),
        report=report,
    )
    assert "FROM main.finance_docs.document_text VERSION AS OF 42 AS source" in sql
    # The strata resync and the preflight read the same pinned rows.
    assert "VERSION AS OF 42" in gbi.resync_strata_sql(spec, snapshot)
    assert "VERSION AS OF 42" in gbi.source_preflight_sql(spec, snapshot)

    # A gated spec cannot build a statement without the authorising report,
    # so the version can never come from somewhere the evidence did not.
    assert spec.gate_required
    with pytest.raises(gbi.GateNotPassed, match="report is required"):
        gbi.build_execute_sql(
            spec,
            run_id="run-1",
            estimate=estimate_for(spec),
            preflight=preflight_for(spec),
        )
    # Nor from a report for a different spec revision.
    other = make_spec(prompt_version="9.9.9")
    with pytest.raises(gbi.EvidenceMismatch, match="different spec revision"):
        gbi.build_execute_sql(
            spec,
            run_id="run-1",
            estimate=estimate_for(spec),
            preflight=preflight_for(spec),
            report=adopting_report(other),
        )
    # Nor can evidence about one table pin a run over another.
    with pytest.raises(gbi.EvidenceMismatch, match="cannot pin a run"):
        gate(
            spec,
            score(
                [
                    gbi.EvaluationRecord(
                        stratum="standard",
                        inference=PLACEHOLDER_INFERENCE,
                        gold={f.name: "v" for f in spec.fields},
                        predicted={f.name: "v" for f in spec.fields},
                    )
                ]
                * 200,
                spec,
            ),
            source_snapshot=gbi.SourceSnapshot(table="main.other.docs", version=1),
        )


def test_tier_three_is_pinned_by_its_estimate():
    """Tier 3 has no gate to pin it, so the estimate does it instead.

    Cost is the only control tier 3 has, and an unpinned read would let
    the table grow past the projection that authorised the spend — the
    control defeated by exactly the mechanism the gated tiers are
    protected from.
    """
    spec = make_spec(use_tier=3)
    assert not spec.gate_required
    sql = gbi.build_execute_sql(
        spec,
        run_id="run-1",
        estimate=estimate_for(spec, snapshot_for(spec, 5)),
        preflight=preflight_for(spec, snapshot_for(spec, 5)),
    )
    assert "FROM main.finance_docs.document_text VERSION AS OF 5 AS source" in sql

    # And an estimate that names no snapshot cannot authorise a paid run.
    with pytest.raises(gbi.EvidenceMismatch, match="pinned snapshot"):
        gbi.build_execute_sql(
            spec,
            run_id="run-1",
            estimate=estimate_for(spec, None),
            preflight=preflight_for(spec, snapshot_for(spec, 5)),
        )


def test_a_declined_field_the_spec_never_declared_is_refused():
    """A near-miss name passes a free-form schema, matches no field, and
    the value the model was declining lands as though asserted."""
    spec = make_spec()

    # The schema no longer permits it in the first place.
    schema = gbi.response_format(spec)["json_schema"]["schema"]
    assert schema["properties"]["abstained_fields"]["items"]["enum"] == [
        "issuer_name",
        "account_id",
    ]

    # And the policy refuses it even so, rather than dropping it silently.
    with pytest.raises(gbi.UnknownAbstainedField, match="issuer_nam"):
        gbi.apply_abstention_policy(
            spec,
            {"issuer_name": "Acme", "account_id": "123"},
            {"issuer_name": 0.99, "account_id": 0.99},
            ["issuer_nam"],
        )
    # The declared spelling still works normally.
    permitted, effective = gbi.apply_abstention_policy(
        spec,
        {"issuer_name": "Acme", "account_id": "123"},
        {"issuer_name": 0.99, "account_id": 0.99},
        ["issuer_name"],
    )
    assert permitted["issuer_name"] is None and effective == {"issuer_name"}


def test_an_unknown_abstention_sends_the_whole_row_to_the_queue():
    """SQL cannot raise over a million rows, so it refuses to assert
    anything from the response and routes it to the exception queue."""
    spec = make_spec()
    sql = gbi.build_execute_sql(
        spec,
        run_id="run-1",
        estimate=estimate_for(spec),
        preflight=preflight_for(spec),
        report=adopting_report(spec),
    )
    unknown = (
        "coalesce(size(array_except(parsed.abstained_fields, "
        "array('issuer_name', 'account_id'))), 0) > 0"
    )
    assert unknown in sql
    # Every value is nulled by it, and the reason reaches ai_error.
    assert sql.count(unknown) >= len(spec.fields) + 1
    assert "response declined fields the spec never declared" in sql
    # The exception queue already selects on ai_error, so it is picked up.
    assert "ai_error IS NOT NULL" in gbi.exception_queue_view_sql(
        spec, "main.finance_docs.v_extraction_queue"
    )


def test_a_nested_field_verdict_cannot_be_flipped():
    """Round 9 made the aggregate derived; the per-field verdicts under it
    were still whatever the artifact said they were."""
    spec = one_field_spec()
    rejecting = gate(spec, score(records_for("s", correct=80, wrong=20), spec))
    assert rejecting.decision == gbi.GateDecision.REJECT
    assert rejecting.fields[0].decision == gbi.GateDecision.REJECT

    # Flip the leaf and the aggregate together, so they agree with each
    # other and disagree only with the evidence.
    payload = rejecting.model_dump(mode="json")
    for result in payload["fields"]:
        result["decision"] = "adopt"
    payload["decision"] = "adopt"
    with pytest.raises(ValidationError, match="does not follow from the scores"):
        gbi.GateReport.model_validate(payload)

    # Raising the recorded bound instead is refused the same way: the
    # verdict is recomputed from the scores, not read off the summary.
    payload = rejecting.model_dump(mode="json")
    payload["fields"][0]["binding_lower_bound"] = 0.99
    with pytest.raises(ValidationError, match="does not follow from the scores"):
        gbi.GateReport.model_validate(payload)

    # An honest report survives the round trip intact.
    assert gbi.GateReport.model_validate(rejecting.model_dump(mode="json")) == rejecting


def test_execution_binds_the_reports_scores_to_the_spec():
    """The report proves its verdicts follow from its scores; only the
    caller knows whether those scores are about this release."""
    spec = one_field_spec()
    report = adopting_report(spec)
    gbi.require_executable(spec, report, preflight_for(spec))

    # Scores from a different release, re-judged into an adopting report,
    # cannot authorise this one even though that report is self-consistent.
    other = one_field_spec(prompt_version="9.9.9")
    smuggled = report.model_copy(
        update={
            "scores": adopting_report(other).scores,
            "spec_digest": spec.spec_digest,
        }
    )
    with pytest.raises(gbi.EvidenceMismatch):
        gbi.require_executable(spec, smuggled, preflight_for(spec))


def test_gated_evidence_must_record_its_source_version():
    """Pinning the run is worthless if the evidence need not say to what."""
    spec = make_spec()
    scores = score(
        [
            gbi.EvaluationRecord(
                stratum="standard",
                inference=PLACEHOLDER_INFERENCE,
                gold={f.name: "v" for f in spec.fields},
                predicted={f.name: "v" for f in spec.fields},
            )
        ]
        * 200,
        spec,
    )
    assert spec.gate_required
    with pytest.raises(gbi.EvidenceMismatch, match="must record which version"):
        gbi.evaluate_gate(spec, scores)

    # Tier 3 is ungated, so there is no evidence to pin and none is asked for.
    exploratory = make_spec(use_tier=3)
    assert not exploratory.gate_required
    assert (
        gbi.evaluate_gate(exploratory, score_for(exploratory)).source_snapshot is None
    )


def score_for(spec):
    return score(
        [
            gbi.EvaluationRecord(
                stratum="standard",
                inference=PLACEHOLDER_INFERENCE,
                gold={f.name: "v" for f in spec.fields},
                predicted={f.name: "v" for f in spec.fields},
            )
        ]
        * 200,
        spec,
    )


def test_the_strata_resync_will_not_relabel_a_newer_release():
    """It runs from a pinned historical snapshot, so without the guard a
    delayed job regresses the grouping monitoring depends on."""
    spec = make_spec(release_sequence=3)
    sql = gbi.resync_strata_sql(spec, snapshot_for(spec, 11))
    # The same pair the inference MERGE orders on — two cycles of one
    # unchanged spec tie on the release sequence and differ only in
    # which snapshot they read.
    # Strata order on their own column: release identity says nothing
    # about which snapshot a label came from.
    assert "coalesce(target.ai_strata_version, -1) <= 11" in sql
    assert "ai_release_sequence" not in sql
    # Still only touching strata — no inference, no value columns.
    assert "ai_query" not in sql
    # It advances the ordering column, but never an extracted value.
    assert "target.ai_strata_version" in sql.split("THEN UPDATE SET")[1]
    assert not any(
        gbi.ai_column(f.name) in sql.split("THEN UPDATE SET")[1] for f in spec.fields
    )


def test_a_report_cannot_bring_its_own_gate_policy():
    """The verdicts are derived from the scores — but the bar they are
    derived against was still whatever the artifact claimed."""
    spec = one_field_spec(criticality="high", tolerable_error_rate=0.02)
    honest = adopting_report(spec)
    gbi.require_executable(spec, honest, preflight_for(spec))

    def with_policy(criticality, required):
        """A report with honest, correctly stamped scores — and a
        different bar applied to them. It validates: the verdict really
        does follow from these scores at the policy it states."""
        result = gbi._gate_field("f", criticality, required, honest.scores)
        return gbi.GateReport.model_validate(
            {
                **honest.model_dump(mode="json"),
                "fields": [result.model_dump(mode="json")],
            }
        )

    # A lowered bar. Nothing here is inconsistent; it simply certifies a
    # policy other than the one this run executes under.
    loosened = with_policy(gbi.Criticality.HIGH, 0.5)
    assert loosened.decision == gbi.GateDecision.ADOPT
    with pytest.raises(gbi.GateNotPassed, match="certifies a policy"):
        gbi.require_executable(spec, loosened, preflight_for(spec))

    # Downgrading criticality is the same attack by another route: it
    # silently swaps worst-stratum gating for the population-weighted row.
    downgraded = with_policy(gbi.Criticality.MEDIUM, spec.fields[0].required_rate)
    with pytest.raises(gbi.GateNotPassed, match="certifies a policy"):
        gbi.require_executable(spec, downgraded, preflight_for(spec))


def test_an_older_snapshot_cannot_overwrite_a_newer_one():
    """`release_sequence` orders what ran, not which rows it ran over. A
    nightly job unchanged for months ties with itself on sequence alone."""
    spec = make_spec(release_sequence=4)
    monday = gbi.build_execute_sql(
        spec,
        run_id="mon",
        estimate=estimate_for(spec, snapshot_for(spec, 10)),
        preflight=preflight_for(spec, snapshot_for(spec, 10)),
        report=adopting_report(spec, source_snapshot=snapshot_for(spec, 10)),
    )
    tuesday = gbi.build_execute_sql(
        spec,
        run_id="tue",
        estimate=estimate_for(spec, snapshot_for(spec, 20)),
        preflight=preflight_for(spec, snapshot_for(spec, 20)),
        report=adopting_report(spec, source_snapshot=snapshot_for(spec, 20)),
    )
    # Each run stamps the version it read.
    assert "10 AS ai_source_version" in monday
    assert "20 AS ai_source_version" in tuesday
    # Monday, resuming late, treats Tuesday's rows as done rather than
    # re-inferring them and writing its older content over the top.
    assert "coalesce(done.ai_source_version, -1) > 10" in monday
    assert "coalesce(done.ai_source_version, -1) > 20" in tuesday
    # And if it gets that far, the MERGE will not lower the version.
    assert (
        "coalesce(target.ai_source_version, -1) <= source.ai_source_version" in monday
    )
    assert ("ai_source_version", "BIGINT") in gbi.target_columns(spec)


def test_the_notebook_pins_every_gate_call():
    """A source-level check, because CI cannot execute the notebook.

    Making the snapshot mandatory broke four call sites at once and
    nothing caught it: `compile()` only parses, and the wiring test below
    mirrors the notebook's derivation rather than reading it. Two of the
    four were `try/except EvidenceMismatch` demonstrations, which would
    have gone on printing a refusal while proving nothing.
    """
    source = (
        ROOT / "examples" / "governed-batch-inference" / "example_notebook.py"
    ).read_text()
    calls = source.split("gbi.evaluate_gate(")[1:]
    assert len(calls) >= 5
    for call in calls:
        # The kwarg must appear before the call closes.
        head = call[: call.index(")\n")]
        assert "source_snapshot=" in head, head[:120]


def test_the_sql_builder_enforces_the_gate_itself():
    """It *is* the paid statement. A guard that works only because the
    caller happened to check first is not a guard."""
    spec = one_field_spec()
    rejecting = gate(spec, score(records_for("s", correct=80, wrong=20), spec))
    assert rejecting.decision == gbi.GateDecision.REJECT
    assert rejecting.spec_digest == spec.spec_digest  # names the right spec

    with pytest.raises(gbi.GateNotPassed, match="execution requires 'adopt'"):
        gbi.build_execute_sql(
            spec,
            run_id="run-1",
            estimate=estimate_for(spec),
            preflight=preflight_matching(rejecting, spec),
            report=rejecting,
        )

    # Tier 1 passing-but-unapproved is refused here too, not just by the
    # notebook's separate call.
    tier1 = one_field_spec(use_tier=1, rollback_plan="Restore prior version.")
    pending = gate(tier1, score(records_for("s", correct=200), tier1))
    assert pending.decision == gbi.GateDecision.PENDING_APPROVAL
    with pytest.raises(gbi.GateNotPassed):
        gbi.build_execute_sql(
            tier1,
            run_id="run-1",
            estimate=estimate_for(tier1),
            preflight=preflight_matching(pending, tier1),
            report=pending,
        )
    # Signed off, it builds.
    approved = gbi.approve_gate(pending, "governance-board")
    assert "ai_query" in gbi.build_execute_sql(
        tier1,
        run_id="run-1",
        estimate=estimate_for(tier1),
        preflight=preflight_matching(approved, tier1),
        report=approved,
    )


def test_a_persisted_gated_report_must_name_its_snapshot():
    """`evaluate_gate` enforces this, but reports are read back far more
    often than they are minted."""
    spec = one_field_spec()
    report = adopting_report(spec)
    payload = report.model_dump(mode="json")
    assert payload["source_snapshot"] is not None

    with pytest.raises(ValidationError, match="must name the source version"):
        gbi.GateReport.model_validate({**payload, "source_snapshot": None})

    # Tier 3 has nothing to pin, and is allowed to say so.
    exploratory = one_field_spec(use_tier=3)
    unpinned = gate(exploratory, score(records_for("s", correct=200), exploratory))
    assert unpinned.source_snapshot is None
    assert gbi.GateReport.model_validate(unpinned.model_dump(mode="json")) == unpinned


def test_a_cost_estimate_is_bound_to_the_rows_it_measured(monkeypatch):
    """Release identity fixes the price per row, not how many rows."""
    spec = one_field_spec()
    scores = score(records_for("s", correct=200), spec)
    report = gate(spec, scores)

    def estimate_over(snapshot, rows):
        return gbi.estimate_cost(
            spec,
            row_count=rows,
            probe_input_tokens=[10],
            probe_output_tokens=[10],
            cad_per_million_input_tokens=0.1,
            cad_per_million_output_tokens=0.1,
            source_snapshot=snapshot,
        )

    recorder = _RecordingMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", recorder)
    # Matching snapshots log fine.
    gbi.log_gate_evidence(
        spec, estimate_over(report.source_snapshot, 100), {"s": 200}, scores, report
    )
    # A projection made over an older, smaller snapshot cannot authorise
    # a run gated on a newer one, however cheap it claims to be.
    stale = estimate_over(snapshot_for(spec, 1), 10)
    assert stale.release == report.scores[0].release
    with pytest.raises(gbi.EvidenceMismatch, match="different rows"):
        gbi.log_gate_evidence(spec, stale, {"s": 200}, scores, report)


def test_the_sql_builder_enforces_the_ceiling_itself():
    """Same argument as the gate, applied to money — and it binds at
    every tier, since tier 3 is controlled by cost alone."""
    spec = one_field_spec(cost_ceiling_cad=1.0)
    report = adopting_report(spec)
    over = gbi.estimate_cost(
        spec,
        row_count=100_000_000,
        probe_input_tokens=[5_000],
        probe_output_tokens=[5_000],
        cad_per_million_input_tokens=7.0,
        cad_per_million_output_tokens=21.0,
        source_snapshot=report.source_snapshot,
    )
    assert not over.within_ceiling
    with pytest.raises(gbi.CostCeilingExceeded):
        gbi.build_execute_sql(
            spec,
            run_id="r",
            estimate=over,
            preflight=preflight_for(spec, report.source_snapshot, over.row_count),
            report=report,
        )

    # An estimate over different rows cannot authorise this spend either,
    # however cheap it is.
    elsewhere = estimate_for(spec, snapshot_for(spec, 999))
    assert elsewhere.within_ceiling
    with pytest.raises(gbi.EvidenceMismatch, match="different rows"):
        gbi.build_execute_sql(
            spec,
            run_id="r",
            estimate=elsewhere,
            preflight=preflight_for(spec, snapshot_for(spec, 999)),
            report=report,
        )

    # Tier 3 has no gate at all, so the ceiling is its only control — and
    # it is still enforced here.
    tier3 = one_field_spec(use_tier=3, cost_ceiling_cad=1.0)
    with pytest.raises(gbi.CostCeilingExceeded):
        gbi.build_execute_sql(
            tier3,
            run_id="r",
            estimate=gbi.estimate_cost(
                tier3,
                row_count=100_000_000,
                probe_input_tokens=[5_000],
                probe_output_tokens=[5_000],
                cad_per_million_input_tokens=7.0,
                cad_per_million_output_tokens=21.0,
                source_snapshot=snapshot_for(tier3),
            ),
            preflight=preflight_for(tier3, snapshot_for(tier3), 100_000_000),
        )


def test_a_report_cannot_relabel_its_own_tier():
    """The snapshot invariant keys off the report's tier, so the tier is
    the thing worth forging: claim exploratory and the pin switches off."""
    spec = one_field_spec()  # tier 2
    report = adopting_report(spec)
    payload = report.model_dump(mode="json")

    # Exploratory reports need no snapshot, so this reconstructs cleanly
    # while keeping the operational digest and real scores.
    unpinned = gbi.GateReport.model_validate(
        {**payload, "use_tier": 3, "source_snapshot": None}
    )
    assert unpinned.decision == gbi.GateDecision.ADOPT
    with pytest.raises(gbi.GateNotPassed, match="claims tier"):
        gbi.require_executable(spec, unpinned, preflight_for(spec))
    # And so the builder will not read the live table on its say-so.
    with pytest.raises(gbi.GateNotPassed, match="claims tier"):
        gbi.build_execute_sql(
            spec,
            run_id="r",
            estimate=estimate_for(spec, None),
            preflight=preflight_for(spec),
            report=unpinned,
        )


def test_the_resync_advances_the_version_it_compares():
    """Otherwise the guard never moves: a label corrected without a
    content change is skipped by inference, so the row keeps the version
    of whichever run last inferred it and an older resync wins again."""
    spec = make_spec(release_sequence=3)
    tuesday = gbi.resync_strata_sql(spec, snapshot_for(spec, 20))
    body = tuesday.split("THEN UPDATE SET")[1]
    assert "target.ai_strata_version = 20" in body
    assert "target.layout = source.layout" in body
    # Monday, arriving late, is now excluded by the version it wrote.
    monday = gbi.resync_strata_sql(spec, snapshot_for(spec, 10))
    assert "coalesce(target.ai_strata_version, -1) <= 10" in monday


def test_a_cost_estimate_cannot_declare_its_own_projection():
    """Every other piece of evidence is recomputed; the budget was the
    one still taken at its word."""
    spec = one_field_spec()
    honest = estimate_for(spec)
    payload = honest.model_dump(mode="json")
    assert gbi.CostEstimate.model_validate(payload) == honest

    # Same release, same snapshot, same token counts — a free run.
    with pytest.raises(ValidationError, match="does not follow from"):
        gbi.CostEstimate.model_validate({**payload, "projected_cost_cad": 0.0})
    # Or the same projection with the row count quietly reduced.
    with pytest.raises(ValidationError, match="does not follow from"):
        gbi.CostEstimate.model_validate({**payload, "row_count": 1})

    # The ceiling is the spec's to set, so raising it in the artifact
    # does not buy headroom either.
    roomier = gbi.CostEstimate.model_validate(
        {**payload, "cost_ceiling_cad": payload["cost_ceiling_cad"] * 1000}
    )
    assert roomier.within_ceiling
    with pytest.raises(gbi.EvidenceMismatch, match="budget is the spec's"):
        gbi.build_execute_sql(
            spec,
            run_id="r",
            estimate=roomier,
            preflight=preflight_for(spec),
            report=adopting_report(spec),
        )


def test_score_counts_cannot_exceed_the_rows_they_came_from():
    """Tying an interval to its counts is worth nothing if the counts
    themselves are impossible."""
    spec = one_field_spec()
    real = next(
        s
        for s in score(records_for("s", correct=200), spec)
        if s.stratum != gbi.WEIGHTED
    )
    payload = real.model_dump(mode="json")
    assert gbi.FieldStratumScore.model_validate(payload) == real

    # One sampled document, two hundred correct answers, and intervals
    # that agree with the counts all the way to an adopting gate.
    with pytest.raises(ValidationError, match="cannot carry more than one"):
        gbi.FieldStratumScore.model_validate({**payload, "n_rows": 1})
    # The aggregate row's counts are sums of the physical rows, so the
    # same inequality binds there.
    weighted = next(
        s
        for s in score(records_for("s", correct=200), spec)
        if s.stratum == gbi.WEIGHTED
    )
    with pytest.raises(ValidationError, match="cannot carry more than one"):
        gbi.FieldStratumScore.model_validate(
            {**weighted.model_dump(mode="json"), "n_rows": 2}
        )


def test_the_builder_requires_proof_the_preflight_ran():
    """Null keys, duplicate keys and null documents each break restart or
    the MERGE in a way that looks like it is working — so passing the
    check is something the caller holds, not something they are trusted
    to have done."""
    spec = one_field_spec()
    report = adopting_report(spec)

    # The preflight has to describe the rows this run will process.
    with pytest.raises(gbi.EvidenceMismatch, match="not the rows to be processed"):
        gbi.build_execute_sql(
            spec,
            run_id="r",
            estimate=estimate_for(spec),
            preflight=preflight_for(spec, snapshot_for(spec, 4321)),
            report=report,
        )
    # It cannot be minted for a table this spec does not read.
    with pytest.raises(gbi.EvidenceMismatch, match="preflight measured"):
        gbi.require_usable_source_rows(
            spec,
            0,
            0,
            0,
            snapshot=gbi.SourceSnapshot(table="main.other.docs", version=7),
            row_count=10,
            stratum_population={"standard": 10},
        )


def test_the_priced_row_count_must_be_the_counted_one():
    """Recomputing the projection proved the arithmetic, not the inputs:
    halve the row count and the price together and the estimate stays
    perfectly self-consistent."""
    spec = one_field_spec()
    report = adopting_report(spec)
    cheap = gbi.estimate_cost(
        spec,
        row_count=1,  # the snapshot really holds ESTIMATED_ROWS
        probe_input_tokens=[100],
        probe_output_tokens=[100],
        cad_per_million_input_tokens=0.1,
        cad_per_million_output_tokens=0.1,
        source_snapshot=report.source_snapshot,
    )
    # Internally consistent, correctly stamped, and within ceiling.
    assert gbi.CostEstimate.model_validate(cheap.model_dump(mode="json")) == cheap
    assert cheap.within_ceiling

    with pytest.raises(gbi.EvidenceMismatch, match="priced 1 row"):
        gbi.build_execute_sql(
            spec,
            run_id="r",
            estimate=cheap,
            preflight=preflight_for(spec),
            report=report,
        )


def test_model_copy_cannot_smuggle_a_decision_past_the_validators():
    """`model_copy` does not re-run validators, so every "the artifact
    validates itself" guard needs a round trip where it is relied upon."""
    spec = one_field_spec()
    rejecting = gate(spec, score(records_for("s", correct=80, wrong=20), spec))
    assert rejecting.decision == gbi.GateDecision.REJECT

    # Pydantic builds this happily: no validator runs.
    flipped = rejecting.model_copy(update={"decision": gbi.GateDecision.ADOPT})
    assert flipped.decision == gbi.GateDecision.ADOPT
    with pytest.raises(gbi.GateNotPassed, match="does not satisfy its own"):
        gbi.require_executable(spec, flipped, preflight_matching(rejecting, spec))
    # And the builder refuses it for the same reason.
    with pytest.raises(gbi.GateNotPassed, match="does not satisfy its own"):
        gbi.build_execute_sql(
            spec,
            run_id="r",
            estimate=estimate_for(spec),
            preflight=preflight_matching(rejecting, spec),
            report=flipped,
        )
    # The legitimate `model_copy` transition still survives the round trip.
    tier1 = one_field_spec(use_tier=1, rollback_plan="Restore prior version.")
    approved = gbi.approve_gate(
        gate(tier1, score(records_for("s", correct=200), tier1)), "board"
    )
    gbi.require_executable(tier1, approved, preflight_matching(approved, tier1))


def test_population_weights_are_checked_against_the_measured_snapshot():
    """Recomputing the weighted row from the weights the same report
    carries proves it agrees with itself and nothing more."""
    spec = one_field_spec(criticality="medium")
    real = {"standard": 500, "legacy_scan": 500}
    scores = score(
        records_for("standard", correct=200) + records_for("legacy_scan", wrong=200),
        spec,
        real,
    )
    honest = gate(spec, scores)
    assert honest.decision == gbi.GateDecision.REJECT

    # Re-weight the same evidence so the failing stratum all but vanishes.
    forged = gate(
        spec,
        score(
            records_for("standard", correct=200)
            + records_for("legacy_scan", wrong=200),
            spec,
            {"standard": 1_000_000, "legacy_scan": 1},
        ),
    )
    assert forged.decision == gbi.GateDecision.ADOPT  # internally consistent

    measured = preflight_for(spec, snapshot_for(spec), 1000)
    measured = measured.model_copy(
        update={"stratum_population": tuple(sorted(real.items()))}
    )
    with pytest.raises(gbi.GateNotPassed, match="population that does not exist"):
        gbi.require_executable(spec, forged, measured)


def test_a_policy_only_release_reuses_the_predictions_it_has():
    """None of tolerance, tier, consumers or ceiling reaches the model, so
    re-inferring for them buys byte-identical output at full price."""
    spec = one_field_spec(tolerable_error_rate=0.05)
    relaxed = one_field_spec(tolerable_error_rate=0.10)
    assert relaxed.spec_digest != spec.spec_digest
    assert relaxed.inference_digest == spec.inference_digest

    # Both statements call the same rows done, so the second is a no-op.
    predicate = gbi.restart_predicate_sql(relaxed, snapshot_for(relaxed))
    assert f"done.ai_inference_digest =\n            '{spec.inference_digest}'" in (
        predicate
    )
    assert "ai_spec_digest" not in predicate

    # The policy stamp is refreshed separately, without inference.
    policy = gbi.resync_policy_sql(relaxed, snapshot_for(relaxed))
    assert f"target.ai_spec_digest = '{relaxed.spec_digest}'" in policy
    assert f"target.ai_inference_digest = '{relaxed.inference_digest}'" in policy
    assert "ai_query" not in policy


def test_an_unmeasured_population_cannot_certify_weighted_evidence():
    """An empty measurement is the absence of the check, not a pass."""
    spec = one_field_spec(criticality="medium")
    forged = gate(
        spec,
        score(
            records_for("standard", correct=200)
            + records_for("legacy_scan", wrong=200),
            spec,
            {"standard": 1_000_000, "legacy_scan": 1},
        ),
    )
    assert forged.decision == gbi.GateDecision.ADOPT

    blank = gbi.SourcePreflight(snapshot=snapshot_for(spec), row_count=1000)
    assert blank.stratum_population == ()
    with pytest.raises(gbi.GateNotPassed, match="no measured stratum counts"):
        gbi.require_executable(spec, forged, blank)

    # The public helper can no longer produce one by omission.
    with pytest.raises(TypeError):
        gbi.require_usable_source_rows(
            spec, 0, 0, 0, snapshot=snapshot_for(spec), row_count=10
        )


def test_a_blank_approver_is_not_an_approver():
    """`approve_gate` strips and refuses; persisted evidence reaches the
    model without passing through it."""
    tier1 = one_field_spec(use_tier=1, rollback_plan="Restore prior version.")
    approved = gbi.approve_gate(
        gate(tier1, score(records_for("s", correct=200), tier1)), "  board  "
    )
    assert approved.approved_by == "board"  # stripped on the way in

    payload = approved.model_dump(mode="json")
    for blank in (" ", "", "\t\n"):
        with pytest.raises(ValidationError, match="blank is not an approver"):
            gbi.GateReport.model_validate({**payload, "approved_by": blank})


def test_the_policy_resync_leaves_edited_documents_alone():
    """A row whose source text changed is pending, and the run may never
    happen — stamping it would claim this release governs stale values."""
    spec = one_field_spec()
    sql = gbi.resync_policy_sql(spec, snapshot_for(spec))
    assert "target.ai_source_digest = sha2(source.doc_text, 256)" in sql
    assert f"target.ai_inference_digest = '{spec.inference_digest}'" in sql
    assert "ai_query" not in sql


def test_the_weighted_label_is_reserved_for_the_aggregate_row():
    spec = one_field_spec(criticality="high")
    records = [
        record.model_copy(update={"inference": spec.inference})
        for record in records_for(gbi.WEIGHTED, correct=200)
    ]
    with pytest.raises(ValueError, match="reserved"):
        gbi.score_extraction(records, spec, {gbi.WEIGHTED: 200})


def test_a_policy_change_re_judges_the_same_predictions_without_re_inference():
    """Changing how output is judged does not change the output.

    Tier, consumers, tolerances, strata and the release sequence do not
    alter a single character the model returns, so binding predictions to
    the whole spec would force a paid re-run to obtain identical results.
    Predictions bind to the inference identity; scores still carry the
    full release, because re-*judging* does require re-scoring.
    """
    operational = one_field_spec(criticality="high")
    records = [
        record.model_copy(update={"inference": operational.inference})
        for record in records_for("standard", correct=200)
    ]
    population = {"standard": 200}

    consequential = gbi.BatchInferenceSpec.model_validate(
        {
            **operational.model_dump(mode="json"),
            "use_tier": 1,
            "consumed_by": ["member_statements"],
            "rollback_plan": "restore the previous table version",
            "release_sequence": operational.release_sequence + 1,
        }
    )
    # A different release, but the same inference — so the same rows.
    assert consequential.release != operational.release
    assert consequential.inference == operational.inference

    scores = gbi.score_extraction(records, consequential, population)
    assert all(score.release == consequential.release for score in scores)
    assert gate(consequential, scores).decision == gbi.GateDecision.PENDING_APPROVAL

    # But anything that changes the request still invalidates the records.
    for change in (
        {"prompt_version": "9.9.9"},
        {"model_version": "another-model"},
        {"endpoint": "another-endpoint"},
        {"abstain_threshold": 0.99},
    ):
        with pytest.raises(gbi.EvidenceMismatch):
            gbi.score_extraction(
                records, operational.model_copy(update=change), population
            )


def test_gate_refuses_intervals_computed_at_the_wrong_confidence():
    """The gate reads the intervals, so relabelling the score is not enough.

    Evidence round-trips through MLflow as JSON, so a score whose outer
    confidence says 99% while its intervals were computed at 95% is a real
    shape, and gating it would reinstate the false adoption.
    """
    spec = one_field_spec(criticality="high", confidence_level=0.99)
    scores = list(score(records_for("standard", correct=400), spec))
    gate(spec, scores)  # consistent evidence is fine

    tampered = []
    for item in scores:
        relabelled = item.precision.model_copy(update={"confidence": 0.95})
        tampered.append(item.model_copy(update={"precision": relabelled}))
    with pytest.raises(gbi.EvidenceMismatch, match="precision interval"):
        gate(spec, tampered)


def test_gate_refuses_scores_spliced_from_two_scoring_runs():
    spec = one_field_spec(criticality="high")
    two_strata = score(CENTRAL_LESSON_RECORDS, spec)
    one_stratum = score(records_for("standard", correct=200), spec)
    with pytest.raises(gbi.EvidenceMismatch, match="disagree"):
        gate(spec, list(two_strata) + list(one_stratum))


def test_rejection_outranks_inconclusive_across_fields():
    spec = make_spec(
        fields=(
            dict(
                name="f",
                description="clearly failing field",
                criticality="high",
                tolerable_error_rate=0.05,
            ),
            dict(
                name="g",
                description="under-evidenced field",
                criticality="high",
                tolerable_error_rate=0.05,
            ),
        )
    )
    records = []
    for row in records_for("s", correct=80, wrong=20):
        records.append(
            gbi.EvaluationRecord(
                stratum="s",
                inference=PLACEHOLDER_INFERENCE,
                gold={**row.gold, "g": "v"},
                predicted={**row.predicted, "g": "v"},
            )
        )
    scores = score(records, spec)
    # f: 80/100 rejects. g would pass on 100 rows, so re-score it over only
    # 30 to make it under-evidenced.
    thin = [s for s in score(records[:30], spec) if s.field == "g"]
    combined = [s for s in scores if s.field == "f"] + thin
    report = gate(spec, combined)
    assert report.decision == gbi.GateDecision.REJECT


def test_tier_one_passing_gate_requires_a_named_human():
    spec = one_field_spec(
        use_tier=1,
        rollback_plan="Restore document_entities from the previous table "
        "version and re-point consumers.",
    )
    scores = score(records_for("standard", correct=200), spec)
    report = gate(spec, scores)
    # Every check passed, and the decision is still not adopt.
    assert all(f.decision == gbi.GateDecision.ADOPT for f in report.fields)
    assert report.decision == gbi.GateDecision.PENDING_APPROVAL
    assert report.human_review_obligations
    with pytest.raises(gbi.GateNotPassed):
        gbi.require_executable(spec, report, preflight_for(spec))
    approved = gbi.approve_gate(report, "analytics-approvers group")
    assert approved.decision == gbi.GateDecision.ADOPT
    assert approved.approved_by == "analytics-approvers group"
    gbi.require_executable(spec, approved, preflight_for(spec))


def test_a_rejected_gate_cannot_be_approved_into_adoption():
    spec = one_field_spec(
        use_tier=1,
        rollback_plan="Restore previous table version.",
    )
    scores = score(records_for("standard", correct=80, wrong=20), spec)
    report = gate(spec, scores)
    assert report.decision == gbi.GateDecision.REJECT
    with pytest.raises(gbi.GateNotPassed):
        gbi.approve_gate(report, "anyone")
    with pytest.raises(gbi.GateNotPassed):
        gbi.require_executable(spec, report, preflight_for(spec))


def test_execution_guard_checks_tier_gate_and_spec_digest():
    spec = one_field_spec(criticality="high")
    with pytest.raises(gbi.GateNotPassed):
        gbi.require_executable(spec, None, preflight_for(spec))
    exploratory = make_spec(use_tier=3, target_table="main.sandbox.scratch")
    gbi.require_executable(
        exploratory, None, preflight_for(exploratory)
    )  # tier 3: no gate to demand
    scores = score(records_for("standard", correct=200), spec)
    report = gate(spec, scores)
    assert report.decision == gbi.GateDecision.ADOPT
    drifted = spec.model_copy(update={"prompt_version": "2.0.0"})
    with pytest.raises(gbi.GateNotPassed, match="different spec revision"):
        gbi.require_executable(drifted, report, preflight_for(drifted))


# ---------------------------------------------------------------------------
# Spec, cost, and SQL builders
# ---------------------------------------------------------------------------


def test_spec_round_trips_through_yaml_with_a_stable_digest():
    spec = make_spec()
    restored = gbi.BatchInferenceSpec.from_yaml(spec.to_yaml())
    assert restored == spec
    assert restored.spec_digest == spec.spec_digest


def test_spec_is_strict_and_frozen():
    spec = make_spec()
    with pytest.raises(ValidationError):
        spec.endpoint = "another-endpoint"
    with pytest.raises(ValidationError):
        make_spec(unexpected_key=True)
    with pytest.raises(ValidationError):
        make_spec(source_table="only.two")
    with pytest.raises(ValidationError):
        make_spec(use_tier=1)  # tier 1 without a rollback plan
    with pytest.raises(ValidationError):
        make_spec(
            fields=(
                dict(
                    name="dup",
                    description="x",
                    criticality="low",
                    tolerable_error_rate=0.1,
                ),
                dict(
                    name="dup",
                    description="y",
                    criticality="low",
                    tolerable_error_rate=0.1,
                ),
            )
        )


def test_spec_rejects_expanded_column_name_collisions():
    """Unique field names are not enough — the expanded names must be too."""

    def field(name, **kwargs):
        return dict(
            name=name,
            description="d",
            criticality="low",
            tolerable_error_rate=0.1,
            **kwargs,
        )

    # ai_error would overwrite the reserved provenance column.
    with pytest.raises(ValidationError, match="collision"):
        make_spec(fields=(field("error"),))
    # ai_x_confidence generated twice, from `x` and from `x_confidence`.
    with pytest.raises(ValidationError, match="collision"):
        make_spec(fields=(field("x"), field("x_confidence")))
    # A field colliding with the key or a stratum column.
    with pytest.raises(ValidationError, match="collision"):
        make_spec(fields=(field("issuer"),), key_column="ai_issuer")
    with pytest.raises(ValidationError, match="collision"):
        make_spec(fields=(field("layout"),), strata=("ai_layout",))
    # Spark identifiers are case-insensitive, so `x` and `X` are one column.
    with pytest.raises(ValidationError, match="case-insensitive"):
        make_spec(fields=(field("x"), field("X")))
    with pytest.raises(ValidationError, match="case-insensitive"):
        make_spec(fields=(field("ERROR"),))
    # Reserved response-schema keys are covered by the same check, because
    # each one has a matching ai_-prefixed provenance column.
    for reserved in gbi.RESERVED_RESPONSE_KEYS:
        with pytest.raises(ValidationError, match="collision"):
            make_spec(fields=(field(reserved),))
    # The ordinary case still validates, and expands without duplicates.
    spec = make_spec()
    names = [name for name, _ in gbi.target_columns(spec)]
    assert len(names) == len(set(names))


def test_cost_estimate_fails_before_execution_when_over_ceiling():
    spec = make_spec(cost_ceiling_cad=5.0)
    estimate = gbi.estimate_cost(
        spec,
        row_count=1_000_000,
        probe_input_tokens=[600, 620, 580, 600],
        probe_output_tokens=[120, 110, 130, 120],
        cad_per_million_input_tokens=0.20,
        cad_per_million_output_tokens=0.60,
        safety_factor=1.2,
    )
    # 1.2 * 1e6 * (600 * 0.20 + 120 * 0.60) / 1e6 = 230.4 CAD
    assert estimate.projected_cost_cad == pytest.approx(230.4)
    assert not estimate.within_ceiling
    with pytest.raises(gbi.CostCeilingExceeded):
        gbi.require_within_ceiling(estimate)
    roomy = make_spec(cost_ceiling_cad=500.0)
    fine = gbi.estimate_cost(
        roomy,
        row_count=1_000_000,
        probe_input_tokens=[600],
        probe_output_tokens=[120],
        cad_per_million_input_tokens=0.20,
        cad_per_million_output_tokens=0.60,
    )
    assert gbi.require_within_ceiling(fine) is fine


def test_cost_estimate_is_bound_to_the_release_it_measured(monkeypatch):
    """A longer prompt is a different budget, so an estimate cannot be
    carried from one release to the next."""
    spec_v1 = one_field_spec()
    spec_v2 = spec_v1.model_copy(update={"prompt_version": "2.0.0"})
    estimate = gbi.estimate_cost(
        spec_v1,
        row_count=1000,
        probe_input_tokens=[600],
        probe_output_tokens=[120],
        cad_per_million_input_tokens=0.20,
        cad_per_million_output_tokens=0.60,
    )
    assert estimate.release == spec_v1.release

    scores = score(records_for("standard", correct=200), spec_v2)
    report = gate(spec_v2, scores)
    recorder = _RecordingMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", recorder)
    with pytest.raises(gbi.EvidenceMismatch, match="Re-estimate"):
        gbi.log_gate_evidence(spec_v2, estimate, {"standard": 200}, scores, report)
    # The mismatch is caught before anything is written.
    assert not recorder.params


def test_unusable_source_keys_are_refused_before_the_paid_query():
    """Both halves of the key contract, checked before anything is spent.

    A null key never matches (so the row is re-inferred and re-inserted
    every run); a duplicated key matches twice (so the MERGE cannot
    resolve it, and one-row-per-key was never true).
    """
    spec = make_spec()
    check = gbi.source_preflight_sql(spec)
    assert "count_if(doc_id IS NULL) AS null_keys" in check
    assert "count(DISTINCT doc_id)" in check
    assert "count_if(doc_text IS NULL) AS null_documents" in check
    assert "count(*) AS row_count" in check
    assert "FROM main.finance_docs.document_text" in check

    snap = snapshot_for(spec)

    def check_rows(nulls=0, dupes=0, null_docs=0):
        return gbi.require_usable_source_rows(
            spec,
            nulls,
            dupes,
            null_docs,
            snapshot=snap,
            row_count=500,
            stratum_population={"standard": 500},
        )

    # The clean case proceeds — and hands back the proof the builder wants.
    clean = check_rows()
    assert clean == gbi.SourcePreflight(
        snapshot=snap, row_count=500, stratum_population=(("standard", 500),)
    )
    with pytest.raises(gbi.UnusableSourceRows, match="3 row"):
        check_rows(nulls=3)
    with pytest.raises(gbi.UnusableSourceRows, match="2 duplicate"):
        check_rows(dupes=2)
    # A null document breaks the same contract a third way: sha2(NULL) is
    # NULL, so the anti-join can never match the row and every run pays to
    # infer over an empty request again.
    with pytest.raises(gbi.UnusableSourceRows, match="4 row.*null doc_text"):
        check_rows(null_docs=4)
    # And the preflight reads the same pinned snapshot the run will use.
    pinned = gbi.source_preflight_sql(
        spec, gbi.SourceSnapshot(table=spec.source_table, version=17)
    )
    assert "FROM main.finance_docs.document_text VERSION AS OF 17" in pinned


def test_the_notebooks_own_wiring_scores_both_releases():
    """Mirrors how the notebook derives v2 and stamps its records.

    The previous round added `release_sequence` and bumped it for v2,
    which changed v2's spec digest — and the notebook stamped records from
    `spec_v1.model_copy(prompt_version=...)`, so scoring raised and the
    walkthrough could not run past the gate. Nothing tested the notebook's
    own wiring, so nothing caught it. This does.
    """
    spec_v1 = one_field_spec(criticality="high")
    spec_v2 = gbi.BatchInferenceSpec.from_yaml(
        spec_v1.to_yaml()
        .replace("prompt_version: 1.0.0", "prompt_version: 2.0.0")
        .replace("release_sequence: 1", "release_sequence: 2")
    )
    population = {"standard": 200}

    for spec in (spec_v1, spec_v2):
        records = [
            record.model_copy(update={"inference": spec.inference})
            for record in records_for("standard", correct=200)
        ]
        scores = gbi.score_extraction(records, spec, population)
        assert gate(spec, scores).decision == gbi.GateDecision.ADOPT

        # And the tier-1 demonstration re-judges those same predictions.
        tier1 = gbi.BatchInferenceSpec.model_validate(
            {
                **spec.model_dump(mode="json"),
                "use_tier": 1,
                "consumed_by": ["member_statements"],
                "rollback_plan": "restore the previous table version",
            }
        )
        tier1_scores = gbi.score_extraction(records, tier1, population)
        assert gate(tier1, tier1_scores).decision == gbi.GateDecision.PENDING_APPROVAL


def test_an_older_release_cannot_overwrite_newer_output():
    """Identity says two runs differ; only an ordering says which is later.

    A delayed retry or an overlapping deploy would otherwise see every
    newer row as unprocessed, re-infer it, and write the older model's
    values back over production.
    """
    old = make_spec(prompt_version="1.0.0", release_sequence=1)
    new = make_spec(prompt_version="2.0.0", release_sequence=2)
    old_sql = gbi.build_execute_sql(
        old,
        run_id="run-old",
        estimate=estimate_for(old),
        preflight=preflight_for(old),
        report=adopting_report(old),
    )
    new_sql = gbi.build_execute_sql(
        new,
        run_id="run-new",
        estimate=estimate_for(new),
        preflight=preflight_for(new),
        report=adopting_report(new),
    )

    # The old job treats anything from a newer release as already done,
    # so it never even pays to re-infer those rows. The test is on the
    # *ordering*: the sequence check stands alone, ahead of the content
    # digest, so an edited document a newer release already landed is
    # excluded rather than inferred and then thrown away by the MERGE.
    assert "coalesce(done.ai_release_sequence, -1) > 1\n        OR (" in old_sql
    assert "coalesce(done.ai_release_sequence, -1) > 2\n        OR (" in new_sql
    for sql in (old_sql, new_sql):
        anti_join = sql.split("scored AS")[0]
        assert anti_join.index("ai_release_sequence") < anti_join.index(
            "ai_source_digest"
        )
    # And the MERGE refuses to lower a row's release sequence.
    guard = (
        "coalesce(target.ai_release_sequence, -1) < source.ai_release_sequence\n"
        "    OR (\n"
        "      coalesce(target.ai_release_sequence, -1) "
        "= source.ai_release_sequence\n"
        "      AND coalesce(target.ai_source_version, -1) "
        "<= source.ai_source_version\n"
        "    )"
    )
    assert guard in old_sql and guard in new_sql
    assert "WHEN MATCHED THEN UPDATE SET *" not in old_sql
    # The sequence is persisted, so the comparison has something to read.
    assert "1 AS ai_release_sequence" in old_sql
    assert ("ai_release_sequence", "BIGINT") in gbi.target_columns(old)


def test_document_column_must_not_collide_with_key_or_strata():
    """`pending` selects key, document, and strata together, so a repeat
    projects one physical column twice and every later use is ambiguous."""
    with pytest.raises(ValidationError, match="collision"):
        make_spec(document_column="doc_id")  # equals key_column
    with pytest.raises(ValidationError, match="collision"):
        make_spec(document_column="layout")  # equals a stratum column


def test_each_table_role_needs_its_own_table():
    same = "main.finance_docs.one_table"
    with pytest.raises(ValidationError, match="each role needs its own table"):
        make_spec(source_table=same, target_table=same)
    with pytest.raises(ValidationError, match="each role needs its own table"):
        make_spec(target_table=same, run_metadata_table=same)
    with pytest.raises(ValidationError, match="each role needs its own table"):
        make_spec(source_table=same, run_metadata_table=same)
    # Spark identifiers are case-insensitive, so these are one table too.
    with pytest.raises(ValidationError, match="case-insensitive"):
        make_spec(
            source_table="main.finance_docs.docs",
            target_table="MAIN.Finance_Docs.DOCS",
        )


def test_response_format_uses_only_supported_schema_features():
    """Databricks structured outputs: no anyOf/oneOf/allOf/pattern; the only
    union allowed is [type, "null"] — which is what abstention needs."""
    spec = make_spec()
    payload = gbi.response_format(spec)
    text = json.dumps(payload)
    assert "anyOf" not in text and "oneOf" not in text and "allOf" not in text
    assert payload["json_schema"]["strict"] is True
    schema = payload["json_schema"]["schema"]
    assert schema["properties"]["issuer_name"]["type"] == ["string", "null"]
    assert schema["properties"]["issuer_name_confidence"]["type"] == [
        "number",
        "null",
    ]
    assert schema["properties"]["abstained_fields"]["type"] == "array"
    assert set(schema["required"]) == set(schema["properties"])
    # Closed object: strict mode on an OpenAI-compatible endpoint expects
    # it, and an extraction contract should not accept undeclared fields.
    assert schema["additionalProperties"] is False


def test_execute_sql_is_idempotent_and_carries_row_provenance():
    spec = make_spec()
    sql = gbi.build_execute_sql(
        spec,
        run_id="run-123",
        estimate=estimate_for(spec),
        preflight=preflight_for(spec),
        report=adopting_report(spec),
    )
    assert "LEFT ANTI JOIN main.finance_docs.document_entities" in sql
    assert "failOnError => false" in sql
    assert "responseFormat =>" in sql
    assert "'run-123' AS ai_run_id" in sql
    assert "ai_query(" in sql and "'databricks-gpt-oss-20b'" in sql
    # failOnError => false returns STRUCT(response, errorMessage).
    assert "raw.errorMessage AS error_message" in sql
    ddl = gbi.create_target_table_sql(spec)
    for name, _ in gbi.target_columns(spec):
        assert name in ddl
        assert name in sql


def test_execute_sql_reprocesses_rows_from_an_earlier_release():
    """The anti-join is release-aware and the write is a MERGE.

    Matching on the key alone would exclude every row landed by an older
    prompt or model, so a newly gated release could report success while
    the table still served the previous release's values and provenance.
    """
    spec = make_spec(prompt_version="2.0.0", model_version="model-b")
    sql = gbi.build_execute_sql(
        spec,
        run_id="run-9",
        estimate=estimate_for(spec),
        preflight=preflight_for(spec),
        report=adopting_report(spec),
    )
    anti_join = sql.split("scored AS")[0]
    assert "ON source.doc_id = done.doc_id" in anti_join
    # Restart keys on the inference identity, which covers the model and
    # prompt labels, the prompt text, the abstention threshold and the
    # field descriptions — everything that changes what the model returns.
    assert f"done.ai_inference_digest =\n            '{spec.inference_digest}'" in (
        anti_join
    )
    assert f"'{spec.inference_digest}' AS ai_inference_digest" in sql
    # The policy stamp is still landed, but is not what restart matches.
    assert f"'{spec.spec_digest}' AS ai_spec_digest" in sql

    def digest_of(**changes):
        changed = spec.model_copy(update=changes)
        return changed.inference_digest

    # A changed threshold, prompt text, model or field set is different
    # output, so each still re-infers.
    for change in (
        {"abstain_threshold": 0.8},
        {"prompt_template": "Different instructions.\n\nDOC:\n"},
        {"model_version": "model-c"},
    ):
        assert digest_of(**change) != spec.inference_digest, change
    # A pure policy change is not, so it must not.
    for change in (
        {"consumed_by": ("some_other_pipeline",)},
        {"cost_ceiling_cad": 999.0},
    ):
        changed = spec.model_copy(update=change)
        assert changed.spec_digest != spec.spec_digest, change
        assert changed.inference_digest == spec.inference_digest, change
    assert sql.startswith("MERGE INTO main.finance_docs.document_entities AS target")
    # Updating in place, but never downwards — see the release-ordering test.
    assert "THEN UPDATE SET *" in sql
    assert "WHEN NOT MATCHED THEN INSERT *" in sql


def test_merge_source_projects_exactly_the_target_columns():
    """`UPDATE SET *` / `INSERT *` are only correct when the source's
    columns are the target's columns, in order — and the document text
    used to build the prompt must not leak into the output table."""
    spec = make_spec(strata=("layout", "doc_type"))
    sql = gbi.build_execute_sql(
        spec,
        run_id="run-1",
        estimate=estimate_for(spec),
        preflight=preflight_for(spec),
        report=adopting_report(spec),
    )
    head = sql[: sql.index("\n  FROM parsed")]
    projection = head[head.index("\n", head.rindex("SELECT\n")) :]
    # Split on top-level commas only: projected expressions contain their
    # own (CASE WHEN coalesce(...), false) ... END), so a naive split by
    # comma or by line would mis-count the columns.
    items, depth, current = [], 0, []
    for character in projection:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if character == "," and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(character)
    items.append("".join(current))

    produced = []
    for item in items:
        item = " ".join(item.split())
        if item:
            produced.append(item.rsplit(" AS ", 1)[1] if " AS " in item else item)
    assert produced == [name for name, _ in gbi.target_columns(spec)]
    assert spec.document_column not in produced


def test_abstained_and_low_confidence_values_are_never_landed():
    """A value the gate treated as an abstention must not reach the table.

    Scoring ignores abstained predictions, so landing one would put output
    the precision gate never saw in front of a consumer. The rule is
    applied identically in Python and in the generated SQL.
    """
    spec = make_spec(abstain_threshold=0.7)
    permitted, abstained = gbi.apply_abstention_policy(
        spec,
        {"issuer_name": "Contradictory Co", "account_id": "AC1"},
        {"issuer_name": 0.99, "account_id": 0.42},
        ["issuer_name"],  # model asserted a value AND declared abstention
    )
    assert permitted == {"issuer_name": None, "account_id": None}
    assert abstained == frozenset({"issuer_name", "account_id"})

    # A confident value passes through; a genuinely absent one stays null
    # without being counted as an abstention.
    permitted, abstained = gbi.apply_abstention_policy(
        spec,
        {"issuer_name": "Maple Grove", "account_id": None},
        {"issuer_name": 0.95, "account_id": None},
        [],
    )
    assert permitted == {"issuer_name": "Maple Grove", "account_id": None}
    assert abstained == frozenset()

    sql = gbi.build_execute_sql(
        spec,
        run_id="run-1",
        estimate=estimate_for(spec),
        preflight=preflight_for(spec),
        report=adopting_report(spec),
    )
    for name in ("issuer_name", "account_id"):
        assert f"array_contains(parsed.abstained_fields, '{name}')" in sql
        assert f"coalesce(parsed.{name}_confidence, -1) BETWEEN 0.7 AND 1" in sql
        assert f"THEN NULL ELSE parsed.{name} END AS ai_{name}" in sql
    # The landed list is re-derived, so a threshold abstention the model
    # never declared still reaches the exception queue.
    assert "array_compact(array(" in sql
    assert "parsed.abstained_fields AS ai_abstained_fields" not in sql


def test_a_confidence_outside_zero_to_one_is_declined_not_trusted():
    """The schema constrains the JSON type; "0 to 1" is only prose."""
    spec = make_spec(abstain_threshold=0.7)
    permitted, abstained = gbi.apply_abstention_policy(
        spec,
        {"issuer_name": "Impossible Co", "account_id": "AC1"},
        {"issuer_name": 5.0, "account_id": -0.5},
        [],
    )
    assert permitted == {"issuer_name": None, "account_id": None}
    assert abstained == frozenset({"issuer_name", "account_id"})
    # Exactly 1.0 is a legitimate confidence and must still land.
    permitted, abstained = gbi.apply_abstention_policy(
        spec, {"issuer_name": "Certain Co"}, {"issuer_name": 1.0}, []
    )
    assert permitted["issuer_name"] == "Certain Co"
    assert abstained == frozenset()

    sql = gbi.build_execute_sql(
        spec,
        run_id="run-1",
        estimate=estimate_for(spec),
        preflight=preflight_for(spec),
        report=adopting_report(spec),
    )
    for name in ("issuer_name", "account_id"):
        assert f"NOT (coalesce(parsed.{name}_confidence, -1) BETWEEN 0.7 AND 1)" in sql


def test_run_metadata_write_is_idempotent():
    """ai_run_id is the provenance join key, so a duplicate row fans out
    every downstream join — a retry must not create one."""
    spec = make_spec()
    scores = score(records_for("standard", correct=200), one_field_spec())
    report = gate(one_field_spec(), scores)
    sql = gbi.run_metadata_upsert_sql(
        spec,
        report,
        run_id="run-1",
        projected_cost_cad=12.5,
        target_table_version=4,
    )
    assert sql.startswith("MERGE INTO main.finance_docs.batch_inference_runs")
    assert "ON target.run_id = source.run_id" in sql
    assert "WHEN MATCHED THEN UPDATE SET *" in sql
    assert "WHEN NOT MATCHED THEN INSERT *" in sql
    assert "INSERT INTO" not in sql
    # Every column of the metadata table is supplied, by name.
    ddl = gbi.create_run_metadata_table_sql(spec)
    for line in ddl.split("\n")[1:-1]:
        column = line.strip().split(" ")[0]
        assert f" AS {column}" in sql


def test_target_migration_plans_added_columns_and_flags_stale_ones():
    """A changed field set is a new release, and CREATE TABLE IF NOT EXISTS
    will not touch a table that already exists."""
    spec = make_spec()
    expected = [name for name, _ in gbi.target_columns(spec)]

    assert not gbi.plan_target_migration(spec, expected).required

    # Table built for a previous spec: missing the account_id columns,
    # carrying columns for a field this release no longer produces.
    previous = [
        column for column in expected if not column.startswith("ai_account_id")
    ] + ["ai_legacy_field", "ai_legacy_field_confidence"]
    migration = gbi.plan_target_migration(spec, previous)
    assert migration.required
    assert [name for name, _ in migration.add] == [
        "ai_account_id",
        "ai_account_id_confidence",
        "ai_account_id_abstained",
    ]
    assert migration.stale == ("ai_legacy_field", "ai_legacy_field_confidence")
    add_statement = migration.statements[0]
    assert add_statement.startswith(
        "ALTER TABLE main.finance_docs.document_entities ADD COLUMNS ("
    )
    assert "ai_account_id STRING" in add_statement
    # Stale columns are reported for a human, never dropped automatically.
    drops = [s for s in migration.statements if "DROP COLUMN" in s]
    assert all(line.strip().startswith("--") for s in drops for line in s.split("\n"))


def test_target_migration_compares_case_insensitively():
    spec = make_spec()
    shouting = [name.upper() for name, _ in gbi.target_columns(spec)]
    assert not gbi.plan_target_migration(spec, shouting).required


def test_execution_is_blocked_until_removed_columns_are_resolved():
    """`INSERT *` expands over the target's columns and needs each one in
    the source, so a leftover column stops the MERGE at analysis. Fail
    here with the statements instead of there with a SQL error."""
    spec = make_spec()
    expected = [name for name, _ in gbi.target_columns(spec)]

    # A clean table migrates cleanly, and an additive change is applied.
    assert gbi.require_migrated_target(spec, expected).statements == ()
    additive = [c for c in expected if not c.startswith("ai_account_id")]
    assert gbi.require_migrated_target(spec, additive).add

    # A column the release no longer produces blocks, ai_-prefixed or not.
    with pytest.raises(gbi.TargetSchemaMismatch, match="ai_dropped_field"):
        gbi.require_migrated_target(spec, expected + ["ai_dropped_field"])
    with pytest.raises(gbi.TargetSchemaMismatch, match="reviewed_by"):
        gbi.require_migrated_target(spec, expected + ["reviewed_by"])

    migration = gbi.plan_target_migration(
        spec, expected + ["ai_dropped_field", "reviewed_by"]
    )
    assert migration.stale == ("ai_dropped_field",)
    assert migration.foreign == ("reviewed_by",)
    assert migration.blocking == ("ai_dropped_field", "reviewed_by")


def test_provenance_layers_are_generated():
    spec = make_spec()
    statements = gbi.column_tag_statements(spec, "run-123")
    assert len(statements) == len(spec.fields)
    for statement in statements:
        assert statement.startswith(
            "ALTER TABLE main.finance_docs.document_entities ALTER COLUMN ai_"
        )
        assert "SET TAGS ('data_source' = 'ai_generated'" in statement
        assert "'ai_run_id' = 'run-123'" in statement
    ddl = gbi.create_run_metadata_table_sql(spec)
    assert "spec_yaml STRING" in ddl and "gate_decision STRING" in ddl
    records = [
        gbi.EvaluationRecord(
            stratum="standard",
            inference=PLACEHOLDER_INFERENCE,
            gold={"issuer_name": "v", "account_id": "v"},
            predicted={"issuer_name": "v", "account_id": "v"},
        )
    ] * 200
    insert = gbi.run_metadata_upsert_sql(
        spec,
        gate(spec, score(records, spec)),
        run_id="run-123",
        projected_cost_cad=12.5,
        target_table_version=4,
    )
    assert "'run-123' AS run_id" in insert
    assert "12.5 AS projected_cost_cad" in insert
    assert "4 AS target_table_version" in insert


def test_sql_string_literal_escapes_quotes_and_backslashes():
    assert gbi.sql_string_literal("O'Brien") == "'O\\'Brien'"
    assert gbi.sql_string_literal("a\\b") == "'a\\\\b'"


def test_monitoring_sql_tracks_abstention_by_stratum():
    spec = make_spec()
    trend = gbi.abstention_trend_sql(spec)
    assert "abstention_rate" in trend and "layout" in trend
    view = gbi.exception_queue_view_sql(spec, "main.finance_docs.ai_exceptions")
    assert "size(ai_abstained_fields) > 0 OR ai_error IS NOT NULL" in view
    with pytest.raises(ValueError):
        gbi.exception_queue_view_sql(spec, "not_three_part")


def test_metric_keys_are_mlflow_safe():
    key = gbi.metric_key("issuer_name", "K-1|legacy", "precision")
    assert "|" not in key
    assert key.startswith("issuer_name/")


class _RecordingMlflow:
    """Minimal stand-in so evidence logging is testable without MLflow."""

    def __init__(self):
        self.params, self.tags, self.metrics = {}, {}, {}
        self.texts, self.dicts = {}, {}

    def log_params(self, values):
        self.params.update(values)

    def log_metric(self, key, value):
        self.metrics[key] = value

    def log_text(self, text, path):
        self.texts[path] = text

    def log_dict(self, payload, path):
        self.dicts[path] = payload

    def set_tags(self, values):
        self.tags.update(values)


def test_approver_identity_is_recorded_in_evidence_but_never_in_a_tag(monkeypatch):
    """Platform rule: tags carry no personal data.

    Tags are broadly readable and propagate onto other objects, so the
    approver's identity belongs in the access-controlled gate artifact and
    run metadata table — not in a tag.
    """
    approver = "j.reviewer@example.invalid"
    spec = one_field_spec(use_tier=1, rollback_plan="Restore prior version.")
    scores = score(records_for("standard", correct=200), spec)
    approved = gbi.approve_gate(gate(spec, scores), approver)
    estimate = gbi.estimate_cost(
        spec,
        row_count=10,
        probe_input_tokens=[10],
        probe_output_tokens=[10],
        cad_per_million_input_tokens=0.1,
        cad_per_million_output_tokens=0.1,
        source_snapshot=approved.source_snapshot,
    )

    recorder = _RecordingMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", recorder)
    gbi.log_gate_evidence(spec, estimate, {"standard": 200}, scores, approved)

    assert approver not in json.dumps(recorder.tags)
    assert approver not in json.dumps(recorder.params)
    assert recorder.tags["human_approved"] == "yes"
    # ...but the audit trail is intact in the logged gate report.
    report_artifact = recorder.dicts["governed_batch_inference/gate_report.json"]
    assert report_artifact["approved_by"] == approver

    metadata_sql = gbi.run_metadata_upsert_sql(
        spec,
        approved,
        run_id="run-1",
        projected_cost_cad=1.0,
        target_table_version=1,
    )
    assert approver in metadata_sql
