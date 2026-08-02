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
    field = one_field()
    records = records_for("s", correct=8, abstained=1, hallucinated=1)
    (score,) = [
        s for s in gbi.score_extraction(records, [field], 0.95) if s.stratum == "s"
    ]
    # 9 assertions (8 correct + 1 hallucinated); the abstention is not one.
    assert score.n_asserted == 9
    assert score.precision.successes == 8
    assert score.precision.trials == 9
    # 9 rows have a true value (hallucination row has gold None); the
    # abstained row counts as a miss for recall — visible, but a miss.
    assert score.recall.successes == 8
    assert score.recall.trials == 9
    assert score.abstention_rate == pytest.approx(0.1)


def test_scores_include_pooled_and_per_stratum_rows():
    field = one_field()
    records = records_for("a", correct=5) + records_for("b", correct=5, wrong=5)
    scores = gbi.score_extraction(records, [field], 0.95)
    strata = {score.stratum for score in scores}
    assert strata == {"a", "b", gbi.POOLED}
    pooled = next(s for s in scores if s.stratum == gbi.POOLED)
    assert pooled.n_rows == 15
    assert pooled.precision.successes == 10


def test_values_match_is_numeric_aware_and_case_tolerant():
    assert gbi.values_match("$12,345.60", "12345.6")
    assert gbi.values_match("Maple Grove  Capital", " maple grove capital ")
    assert not gbi.values_match("12345.60", "12345.61")
    assert not gbi.values_match(None, "anything")
    assert not gbi.values_match("anything", None)


# ---------------------------------------------------------------------------
# The gate: lower bounds, worst stratum, tiers
# ---------------------------------------------------------------------------


def central_lesson_scores(field):
    """Aggregate looks fine; the minority stratum is broken.

    standard: 795/800 correct (lower ~0.985). legacy_scan: 80/100
    (lower ~0.711). Pooled: 875/900, lower ~0.959 — which *passes* 0.95.
    """
    records = records_for("standard", correct=795, wrong=5) + records_for(
        "legacy_scan", correct=80, wrong=20
    )
    return gbi.score_extraction(records, [field], 0.95)


def test_high_criticality_gates_on_the_worst_stratum_not_the_average():
    field = one_field(criticality="high")
    spec = make_spec(
        fields=(
            dict(
                name="f",
                description="test field",
                criticality="high",
                tolerable_error_rate=0.05,
            ),
        )
    )
    report = gbi.evaluate_gate(spec, central_lesson_scores(field))
    assert report.decision == gbi.GateDecision.REJECT
    (result,) = report.fields
    assert result.decision == gbi.GateDecision.REJECT
    assert result.binding_stratum == "legacy_scan"
    assert result.binding_lower_bound == pytest.approx(0.71117, abs=5e-5)


def test_medium_criticality_gates_on_the_pooled_sample():
    """Same evidence, medium criticality: the pooled interval decides."""
    field = one_field(criticality="medium")
    spec = make_spec(
        fields=(
            dict(
                name="f",
                description="test field",
                criticality="medium",
                tolerable_error_rate=0.05,
            ),
        )
    )
    report = gbi.evaluate_gate(spec, central_lesson_scores(field))
    assert report.decision == gbi.GateDecision.ADOPT


def test_gate_compares_lower_bound_never_the_point_estimate():
    """97/100 with a 95% tolerance: the point estimate passes, the run
    does not — it produced an encouraging number with too little evidence."""
    field = one_field(criticality="medium")
    spec = make_spec(
        fields=(
            dict(
                name="f",
                description="test field",
                criticality="medium",
                tolerable_error_rate=0.05,
            ),
        )
    )
    scores = gbi.score_extraction(
        records_for("standard", correct=97, wrong=3), [field], 0.95
    )
    report = gbi.evaluate_gate(spec, scores)
    (result,) = report.fields
    assert result.binding_point_estimate >= 0.95
    assert result.binding_lower_bound < 0.95
    assert report.decision == gbi.GateDecision.REJECT
    assert any("point estimate" in reason for reason in result.reasons)


def test_too_small_a_sample_is_inconclusive_not_a_rejection():
    """30/30 correct cannot clear a 95% bar (best possible lower bound is
    ~0.886): the model was not shown to be bad — the sample was too small."""
    field = one_field(criticality="high")
    spec = make_spec(
        fields=(
            dict(
                name="f",
                description="test field",
                criticality="high",
                tolerable_error_rate=0.05,
            ),
        )
    )
    scores = gbi.score_extraction(records_for("legacy", correct=30), [field], 0.95)
    report = gbi.evaluate_gate(spec, scores)
    assert report.decision == gbi.GateDecision.INCONCLUSIVE
    (result,) = report.fields
    assert any("label more rows" in reason for reason in result.reasons)


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
    scores = gbi.score_extraction(records, spec.fields, 0.95)
    # f: 80/100 rejects; g: 100/100 but you'd need... 100 >= 73, so make g
    # under-evidenced by scoring only 30 rows for it.
    thin = gbi.score_extraction(records[:30], [spec.field_named("g")], 0.95)
    combined = [s for s in scores if s.field == "f"] + list(thin)
    report = gbi.evaluate_gate(spec, combined)
    assert report.decision == gbi.GateDecision.REJECT


def test_tier_one_passing_gate_requires_a_named_human():
    spec = make_spec(
        use_tier=1,
        rollback_plan="Restore document_entities from the previous table "
        "version and re-point consumers.",
        fields=(
            dict(
                name="f",
                description="test field",
                criticality="high",
                tolerable_error_rate=0.05,
            ),
        ),
    )
    scores = gbi.score_extraction(
        records_for("standard", correct=200), [spec.fields[0]], 0.95
    )
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
    spec = make_spec(
        use_tier=1,
        rollback_plan="Restore previous table version.",
        fields=(
            dict(
                name="f",
                description="test field",
                criticality="high",
                tolerable_error_rate=0.05,
            ),
        ),
    )
    scores = gbi.score_extraction(
        records_for("standard", correct=80, wrong=20), [spec.fields[0]], 0.95
    )
    report = gbi.evaluate_gate(spec, scores)
    assert report.decision == gbi.GateDecision.REJECT
    with pytest.raises(gbi.GateNotPassed):
        gbi.approve_gate(report, "anyone")
    with pytest.raises(gbi.GateNotPassed):
        gbi.require_executable(spec, report)


def test_execution_guard_checks_tier_gate_and_spec_digest():
    spec = make_spec(
        fields=(
            dict(
                name="f",
                description="test field",
                criticality="high",
                tolerable_error_rate=0.05,
            ),
        )
    )
    with pytest.raises(gbi.GateNotPassed):
        gbi.require_executable(spec, None)
    exploratory = make_spec(use_tier=3, target_table="main.sandbox.scratch")
    gbi.require_executable(exploratory, None)  # tier 3: no gate to demand
    scores = gbi.score_extraction(
        records_for("standard", correct=200),
        spec.fields,
        0.95,
    )
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
    # INSERT column list matches the DDL column order exactly.
    ddl = gbi.create_target_table_sql(spec)
    names = [name for name, _ in gbi.target_columns(spec)]
    assert f"({', '.join(names)})" in sql
    for name in names:
        assert name in ddl


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
    insert = gbi.run_metadata_insert_sql(
        spec,
        gbi.evaluate_gate(
            spec,
            gbi.score_extraction(
                records_for("standard", correct=200), spec.fields, 0.95
            ),
        ),
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
