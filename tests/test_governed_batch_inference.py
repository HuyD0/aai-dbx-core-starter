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
        cost_ceiling_cad=50.0,
        abstain_threshold=0.7,
    )
    base.update(overrides)
    return gbi.BatchInferenceSpec.model_validate(base)


def records_for(stratum, *, correct=0, wrong=0, abstained=0, hallucinated=0):
    """Single-field evaluation records with controlled outcome counts."""
    rows = []
    for _ in range(correct):
        rows.append(
            gbi.EvaluationRecord(stratum=stratum, gold={"f": "v"}, predicted={"f": "v"})
        )
    for _ in range(wrong):
        rows.append(
            gbi.EvaluationRecord(stratum=stratum, gold={"f": "v"}, predicted={"f": "x"})
        )
    for _ in range(abstained):
        rows.append(
            gbi.EvaluationRecord(
                stratum=stratum,
                gold={"f": "v"},
                predicted={"f": None},
                abstained=frozenset({"f"}),
            )
        )
    for _ in range(hallucinated):
        rows.append(
            gbi.EvaluationRecord(
                stratum=stratum, gold={"f": None}, predicted={"f": "made-up"}
            )
        )
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
    return gbi.score_extraction(records, spec, population)


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
    report = gbi.evaluate_gate(spec, score(CENTRAL_LESSON_RECORDS, spec))
    assert report.decision == gbi.GateDecision.REJECT
    (result,) = report.fields
    assert result.decision == gbi.GateDecision.REJECT
    assert result.binding_stratum == "legacy_scan"
    assert result.binding_lower_bound == pytest.approx(0.71117, abs=5e-5)


def test_medium_criticality_gates_on_the_population_weighted_estimate():
    """Same evidence, medium criticality: the all-strata estimate decides."""
    spec = one_field_spec(criticality="medium")
    report = gbi.evaluate_gate(spec, score(CENTRAL_LESSON_RECORDS, spec))
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
    assert gbi.evaluate_gate(spec, score(records, spec, population)).decision == (
        gbi.GateDecision.ADOPT
    )
    # And the effective sample size never exceeds what was labelled.
    assert weighted.precision.trials <= 200


def test_weighted_estimate_needs_a_population_for_every_stratum():
    spec = one_field_spec(criticality="medium")
    records = records_for("easy", correct=10) + records_for("hard", correct=10)
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
    report = gbi.evaluate_gate(spec, scores)
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
    report = gbi.evaluate_gate(spec, scores)
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

    report = gbi.evaluate_gate(spec, score(records_for("legacy", wrong=30), spec))
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

    assert gbi.evaluate_gate(spec_v1, scores_v1).decision == gbi.GateDecision.ADOPT
    with pytest.raises(gbi.EvidenceMismatch, match="prompt 1.0.0"):
        gbi.evaluate_gate(spec_v2, scores_v1)


def test_gate_refuses_a_model_version_change_on_stale_evidence():
    spec = one_field_spec(criticality="medium")
    scores = score(records_for("standard", correct=200), spec)
    retrained = spec.model_copy(update={"model_version": "next-model-build"})
    with pytest.raises(gbi.EvidenceMismatch):
        gbi.evaluate_gate(retrained, scores)


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
        gbi.evaluate_gate(spec, mislabelled)
    # Scored properly at the declared level, the same sample is rejected.
    assert gbi.evaluate_gate(spec, score(records, spec)).decision == (
        gbi.GateDecision.REJECT
    )


def test_gate_refuses_an_empty_evidence_set():
    with pytest.raises(gbi.EvidenceMismatch):
        gbi.evaluate_gate(one_field_spec(), ())


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
                gold={**row.gold, "g": "v"},
                predicted={**row.predicted, "g": "v"},
            )
        )
    scores = score(records, spec)
    # f: 80/100 rejects. g would pass on 100 rows, so re-score it over only
    # 30 to make it under-evidenced.
    thin = [s for s in score(records[:30], spec) if s.field == "g"]
    combined = [s for s in scores if s.field == "f"] + thin
    report = gbi.evaluate_gate(spec, combined)
    assert report.decision == gbi.GateDecision.REJECT


def test_tier_one_passing_gate_requires_a_named_human():
    spec = one_field_spec(
        use_tier=1,
        rollback_plan="Restore document_entities from the previous table "
        "version and re-point consumers.",
    )
    scores = score(records_for("standard", correct=200), spec)
    report = gbi.evaluate_gate(spec, scores)
    # Every check passed, and the decision is still not adopt.
    assert all(f.decision == gbi.GateDecision.ADOPT for f in report.fields)
    assert report.decision == gbi.GateDecision.PENDING_APPROVAL
    assert report.human_review_obligations
    with pytest.raises(gbi.GateNotPassed):
        gbi.require_executable(spec, report)
    approved = gbi.approve_gate(report, "analytics-approvers group")
    assert approved.decision == gbi.GateDecision.ADOPT
    assert approved.approved_by == "analytics-approvers group"
    gbi.require_executable(spec, approved)


def test_a_rejected_gate_cannot_be_approved_into_adoption():
    spec = one_field_spec(
        use_tier=1,
        rollback_plan="Restore previous table version.",
    )
    scores = score(records_for("standard", correct=80, wrong=20), spec)
    report = gbi.evaluate_gate(spec, scores)
    assert report.decision == gbi.GateDecision.REJECT
    with pytest.raises(gbi.GateNotPassed):
        gbi.approve_gate(report, "anyone")
    with pytest.raises(gbi.GateNotPassed):
        gbi.require_executable(spec, report)


def test_execution_guard_checks_tier_gate_and_spec_digest():
    spec = one_field_spec(criticality="high")
    with pytest.raises(gbi.GateNotPassed):
        gbi.require_executable(spec, None)
    exploratory = make_spec(use_tier=3, target_table="main.sandbox.scratch")
    gbi.require_executable(exploratory, None)  # tier 3: no gate to demand
    scores = score(records_for("standard", correct=200), spec)
    report = gbi.evaluate_gate(spec, scores)
    assert report.decision == gbi.GateDecision.ADOPT
    drifted = spec.model_copy(update={"prompt_version": "2.0.0"})
    with pytest.raises(gbi.GateNotPassed, match="different spec revision"):
        gbi.require_executable(drifted, report)


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
        spec, run_id="run-123", prompt_sql="concat('extract: ', doc_text)"
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
        spec, run_id="run-9", prompt_sql="concat('extract: ', doc_text)"
    )
    anti_join = sql.split("scored AS")[0]
    assert "ON source.doc_id = done.doc_id" in anti_join
    assert "done.ai_model_version = 'model-b'" in anti_join
    assert "done.ai_prompt_version = '2.0.0'" in anti_join
    # The spec digest too: a changed abstention threshold or field set is
    # also a new release, even when the model and prompt labels hold still.
    assert f"done.ai_spec_digest = '{spec.spec_digest}'" in anti_join
    assert f"'{spec.spec_digest}' AS ai_spec_digest" in sql
    edited = spec.model_copy(update={"abstain_threshold": 0.8})
    assert edited.spec_digest != spec.spec_digest
    assert f"done.ai_spec_digest = '{edited.spec_digest}'" in gbi.build_execute_sql(
        edited, run_id="run-9", prompt_sql="concat('extract: ', doc_text)"
    )
    assert sql.startswith("MERGE INTO main.finance_docs.document_entities AS target")
    assert "WHEN MATCHED THEN UPDATE SET *" in sql
    assert "WHEN NOT MATCHED THEN INSERT *" in sql


def test_merge_source_projects_exactly_the_target_columns():
    """`UPDATE SET *` / `INSERT *` are only correct when the source's
    columns are the target's columns, in order — and the document text
    used to build the prompt must not leak into the output table."""
    spec = make_spec(strata=("layout", "doc_type"))
    sql = gbi.build_execute_sql(
        spec, run_id="run-1", prompt_sql="concat('extract: ', doc_text)"
    )
    head = sql[: sql.index("\n  FROM parsed")]
    projection = head[head.rindex("SELECT\n") :]
    produced = []
    for line in projection.split("\n")[1:]:
        # One projected column per line, except the bare key/strata line.
        # Expressions can contain commas, so alias first, split second.
        line = line.strip().rstrip(",")
        if " AS " in line:
            produced.append(line.rsplit(" AS ", 1)[1])
        else:
            produced.extend(item.strip() for item in line.split(",") if item.strip())
    assert produced == [name for name, _ in gbi.target_columns(spec)]
    assert spec.document_column not in produced


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
            gold={"issuer_name": "v", "account_id": "v"},
            predicted={"issuer_name": "v", "account_id": "v"},
        )
    ] * 200
    insert = gbi.run_metadata_insert_sql(
        spec,
        gbi.evaluate_gate(spec, score(records, spec)),
        run_id="run-123",
        projected_cost_cad=12.5,
        target_table_version=4,
    )
    assert "'run-123'" in insert and "12.5" in insert and "4," in insert


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
    approved = gbi.approve_gate(gbi.evaluate_gate(spec, scores), approver)
    estimate = gbi.estimate_cost(
        spec,
        row_count=10,
        probe_input_tokens=[10],
        probe_output_tokens=[10],
        cad_per_million_input_tokens=0.1,
        cad_per_million_output_tokens=0.1,
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

    metadata_sql = gbi.run_metadata_insert_sql(
        spec,
        approved,
        run_id="run-1",
        projected_cost_cad=1.0,
        target_table_version=1,
    )
    assert approver in metadata_sql
