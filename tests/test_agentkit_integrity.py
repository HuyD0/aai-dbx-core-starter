"""Unit tests for the judge-integrity checks: self-consistency and anchors."""

import json

import pytest

from aai_core.agentkit.errors import ConfigError
from aai_core.agentkit.integrity import (
    ANCHOR_DRIFT_METRIC,
    RESCORE_FAILURES_METRIC,
    SELF_INCONSISTENCY_METRIC,
    AnchorRow,
    IntegrityConfig,
    RowJudge,
    anchor_rows_digest,
    build_anchor_rows,
    eligible_row_indices,
    estimate_integrity_calls,
    extend_rules_with_integrity,
    integrity_metric,
    invoke_judge,
    is_integrity_metric,
    is_row_level_judge,
    load_anchors,
    run_integrity_checks,
    sample_indices,
    write_anchors,
)
from aai_core.evaluation import MetricDirection, MetricRule


class _SteadyJudge:
    """Returns the same verdict every time — a perfectly stable instrument."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self, *, inputs=None, outputs=None, expectations=None):
        self.calls += 1
        return self.value


class _FlippingJudge:
    """Disagrees with every recorded score — a maximally unstable judge."""

    def __call__(self, *, inputs=None, outputs=None, expectations=None):
        return 0.0


class _RaisingJudge:
    def __call__(self, *, inputs=None, outputs=None, expectations=None):
        raise RuntimeError("judge endpoint unreachable")


class _DecliningJudge:
    def __call__(self, *, inputs=None, outputs=None, expectations=None):
        return None


class _OutputsOnlyJudge:
    """A scorer whose __call__ accepts only some of the row fields."""

    def __init__(self):
        self.seen = []

    def __call__(self, *, outputs=None):
        self.seen.append(outputs)
        return "yes"


class _VarKwargsJudge:
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return True


def _rows(n=6):
    return [
        {
            "inputs": {"question": f"question {index}"},
            "expectations": {"expected_response": f"answer {index}"},
        }
        for index in range(n)
    ]


def _samples(values, metric="correctness/mean"):
    return {metric: tuple(values)}


def _judge(scorer, name="correctness", metric="correctness/mean"):
    return RowJudge(name=name, metric=metric, scorer=scorer)


def _config(**overrides):
    return IntegrityConfig(**overrides)


# --- metric naming ----------------------------------------------------------


def test_integrity_metric_names_use_a_stable_segment():
    assert integrity_metric("correctness/mean", "flip_rate") == (
        "correctness/mean/integrity/flip_rate"
    )
    assert is_integrity_metric(SELF_INCONSISTENCY_METRIC)
    assert is_integrity_metric(ANCHOR_DRIFT_METRIC)
    assert is_integrity_metric(RESCORE_FAILURES_METRIC)
    assert not is_integrity_metric("correctness/mean")
    assert not is_integrity_metric("correctness/mean/statistics/sample_count")


# --- rule extension ---------------------------------------------------------


def _base_rule():
    return MetricRule(
        metric="correctness/mean", direction=MetricDirection.HIGHER, required=0.7
    )


def test_no_rules_are_added_by_default():
    extended = extend_rules_with_integrity(
        (_base_rule(),), _config(), judges_enabled=True
    )
    assert [rule.metric for rule in extended] == ["correctness/mean"]


def test_consistency_rule_requires_a_positive_sample():
    extended = extend_rules_with_integrity(
        (_base_rule(),), _config(consistency_sample=8), judges_enabled=True
    )
    rule = {rule.metric: rule for rule in extended}[SELF_INCONSISTENCY_METRIC]
    assert rule.direction is MetricDirection.LOWER
    assert rule.required == pytest.approx(0.2)


def test_anchor_rule_requires_the_explicit_opt_in():
    silent = extend_rules_with_integrity(
        (_base_rule(),), _config(consistency_sample=8), judges_enabled=True
    )
    assert ANCHOR_DRIFT_METRIC not in {rule.metric for rule in silent}
    enforced = extend_rules_with_integrity(
        (_base_rule(),),
        _config(require_anchors=True, max_anchor_drift=0.05),
        judges_enabled=True,
    )
    rule = {rule.metric: rule for rule in enforced}[ANCHOR_DRIFT_METRIC]
    assert rule.required == pytest.approx(0.05)


def test_judges_disabled_strips_every_integrity_rule():
    stale = MetricRule(
        metric=SELF_INCONSISTENCY_METRIC,
        direction=MetricDirection.LOWER,
        required=0.2,
    )
    extended = extend_rules_with_integrity(
        (_base_rule(), stale),
        _config(consistency_sample=8, require_anchors=True),
        judges_enabled=False,
    )
    assert [rule.metric for rule in extended] == ["correctness/mean"]


# --- sampling and eligibility -----------------------------------------------


def test_sample_indices_are_deterministic_and_sorted():
    eligible = list(range(50))
    first = sample_indices(eligible, 8, seed=7)
    second = sample_indices(eligible, 8, seed=7)
    assert first == second == sorted(first)
    assert sample_indices(eligible, 100, seed=7) == eligible


def test_eligible_rows_need_an_output_and_a_recorded_score():
    rows = _rows(4)
    outputs = ["a", None, "c", "d"]
    samples = _samples([1.0, 1.0, None, 1.0])
    judges = [_judge(_SteadyJudge(1.0))]
    assert eligible_row_indices(rows, outputs, samples, judges) == [0, 3]


# --- invocation -------------------------------------------------------------


def test_invoke_judge_filters_kwargs_to_the_scorer_signature():
    scorer = _OutputsOnlyJudge()
    value = invoke_judge(_judge(scorer), _rows(1)[0], "the answer")
    assert value == 1.0
    assert scorer.seen == ["the answer"]


def test_invoke_judge_passes_everything_to_var_kwargs():
    scorer = _VarKwargsJudge()
    value = invoke_judge(_judge(scorer), _rows(1)[0], "the answer")
    assert value == 1.0
    assert set(scorer.kwargs) == {"inputs", "outputs", "expectations"}


def test_invoke_judge_unwraps_feedback_objects_and_lists():
    class _Feedback:
        value = "no"

    class _FeedbackJudge:
        def __call__(self, **kwargs):
            return [_Feedback()]

    assert invoke_judge(_judge(_FeedbackJudge()), _rows(1)[0], "answer") == 0.0


def test_is_row_level_judge_is_duck_typed():
    from aai_core.agentkit.catalog import CATALOG

    names = {spec.name for spec in CATALOG if is_row_level_judge(spec)}
    assert "correctness" in names
    assert "safety" in names
    assert "pension_domain_policy" in names
    # Code scorers have no judge; retrieval and tool judges need traces.
    assert "keyword_coverage" not in names
    assert "retrieval_groundedness" not in names
    assert "tool_call_correctness" not in names


# --- self-consistency -------------------------------------------------------


def test_stable_judge_reports_zero_inconsistency():
    rows = _rows()
    outputs = [f"answer {index}" for index in range(len(rows))]
    samples = _samples([1.0] * len(rows))
    evidence, metrics, warnings = run_integrity_checks(
        config=_config(consistency_sample=4),
        rows=rows,
        outputs_by_row=outputs,
        metric_samples=samples,
        judges=[_judge(_SteadyJudge("yes"))],
        anchors=None,
    )
    assert metrics[SELF_INCONSISTENCY_METRIC] == 0.0
    assert metrics["correctness/mean/integrity/flip_rate"] == 0.0
    assert metrics[RESCORE_FAILURES_METRIC] == 0.0
    assert evidence.consistency.sample_size == 4
    assert evidence.consistency.flip_rates == {"correctness": 0.0}
    assert not warnings


def test_flipping_judge_reports_full_inconsistency_and_warns():
    rows = _rows()
    outputs = [f"answer {index}" for index in range(len(rows))]
    samples = _samples([1.0] * len(rows))
    _, metrics, warnings = run_integrity_checks(
        config=_config(consistency_sample=4),
        rows=rows,
        outputs_by_row=outputs,
        metric_samples=samples,
        judges=[_judge(_FlippingJudge())],
        anchors=None,
    )
    assert metrics[SELF_INCONSISTENCY_METRIC] == 1.0
    assert any("disagreed with themselves" in warning for warning in warnings)


def test_failed_rescoring_leaves_the_metric_absent_and_says_so():
    rows = _rows()
    outputs = [f"answer {index}" for index in range(len(rows))]
    samples = _samples([1.0] * len(rows))
    evidence, metrics, warnings = run_integrity_checks(
        config=_config(consistency_sample=4),
        rows=rows,
        outputs_by_row=outputs,
        metric_samples=samples,
        judges=[_judge(_RaisingJudge())],
        anchors=None,
    )
    # Fail closed through the missing metric, never through a fake zero.
    assert SELF_INCONSISTENCY_METRIC not in metrics
    assert metrics[RESCORE_FAILURES_METRIC] == 4.0
    assert evidence.consistency.rescore_failures == 4
    assert any("re-scoring" in warning for warning in warnings)


def test_no_recoverable_outputs_is_reported_not_invented():
    rows = _rows()
    _, metrics, warnings = run_integrity_checks(
        config=_config(consistency_sample=4),
        rows=rows,
        outputs_by_row=[None] * len(rows),
        metric_samples=_samples([1.0] * len(rows)),
        judges=[_judge(_SteadyJudge("yes"))],
        anchors=None,
    )
    assert SELF_INCONSISTENCY_METRIC not in metrics
    assert any("could not be measured" in warning for warning in warnings)


def test_declined_rescoring_is_not_a_failure():
    rows = _rows()
    outputs = [f"answer {index}" for index in range(len(rows))]
    evidence, metrics, warnings = run_integrity_checks(
        config=_config(consistency_sample=4),
        rows=rows,
        outputs_by_row=outputs,
        metric_samples=_samples([1.0] * len(rows)),
        judges=[
            _judge(_DecliningJudge()),
            _judge(_SteadyJudge("yes"), name="safety", metric="safety/mean"),
        ],
        anchors=None,
    )
    # Every re-score was declined (the safety judge has no first-pass
    # scores to pair with), so the check measured nothing — declines are
    # not failures, and no failure count is invented.
    assert evidence is None
    assert RESCORE_FAILURES_METRIC not in metrics
    assert any("failed or was declined" in warning for warning in warnings)


# --- anchors ----------------------------------------------------------------


def _frozen_anchors(tmp_path, score=1.0, judge_name="correctness"):
    rows = (
        AnchorRow(
            inputs={"question": "question 0"},
            outputs="answer 0",
            expectations={"expected_response": "answer 0"},
            scores={judge_name: score},
        ),
        AnchorRow(
            inputs={"question": "question 1"},
            outputs="answer 1",
            expectations={"expected_response": "answer 1"},
            scores={judge_name: score},
        ),
    )
    path = tmp_path / "evals" / "judge_anchors.json"
    write_anchors(
        path,
        rows=rows,
        recorded_at="2026-08-19T10:00:00Z",
        recorded_by="agentkit compare --establish-baseline",
        change_id="abc123456789",
        judge_model="endpoints:/judge-endpoint",
        judge_model_identity="catalog.schema.judge/3",
        judge_prompts={},
        scorer_versions={judge_name: 1},
    )
    return path


def test_anchors_round_trip_through_the_digest(tmp_path):
    path = _frozen_anchors(tmp_path)
    anchors = load_anchors(path)
    assert len(anchors.rows) == 2
    assert anchors.digest == anchor_rows_digest(anchors.rows)
    assert anchors.scorer_versions == {"correctness": 1}


def test_edited_anchor_rows_are_refused(tmp_path):
    path = _frozen_anchors(tmp_path)
    document = json.loads(path.read_text())
    document["rows"][0]["scores"]["correctness"] = 0.0
    path.write_text(json.dumps(document))
    with pytest.raises(ConfigError, match="changed after it was frozen"):
        load_anchors(path)


def test_edited_anchor_digest_is_refused(tmp_path):
    path = _frozen_anchors(tmp_path)
    document = json.loads(path.read_text())
    document["digest"] = "0" * 64
    path.write_text(json.dumps(document))
    with pytest.raises(ConfigError, match="changed after it was frozen"):
        load_anchors(path)


def test_steady_judge_reports_zero_anchor_drift(tmp_path):
    anchors = load_anchors(_frozen_anchors(tmp_path))
    _, metrics, warnings = run_integrity_checks(
        config=_config(),
        rows=[],
        outputs_by_row=[],
        metric_samples={},
        judges=[_judge(_SteadyJudge("yes"))],
        anchors=anchors,
    )
    assert metrics[ANCHOR_DRIFT_METRIC] == 0.0
    assert metrics["correctness/mean/integrity/anchor_drift"] == 0.0
    assert not warnings


def test_drifting_judge_is_blamed_not_the_agent(tmp_path):
    anchors = load_anchors(_frozen_anchors(tmp_path, score=1.0))
    evidence, metrics, warnings = run_integrity_checks(
        config=_config(max_anchor_drift=0.1),
        rows=[],
        outputs_by_row=[],
        metric_samples={},
        judges=[_judge(_FlippingJudge())],
        anchors=anchors,
    )
    assert metrics[ANCHOR_DRIFT_METRIC] == 1.0
    assert evidence.anchor_drift.rows == 2
    joined = "\n".join(warnings)
    assert "the judge changed, not the agent" in joined
    assert "not an agent regression" in joined


def test_anchors_for_other_judges_cannot_measure_this_run(tmp_path):
    anchors = load_anchors(_frozen_anchors(tmp_path, judge_name="fluency"))
    _, metrics, warnings = run_integrity_checks(
        config=_config(),
        rows=[],
        outputs_by_row=[],
        metric_samples={},
        judges=[_judge(_SteadyJudge("yes"))],
        anchors=anchors,
    )
    assert ANCHOR_DRIFT_METRIC not in metrics
    assert any("no scores for any judge" in warning for warning in warnings)


def test_missing_required_anchors_warn(tmp_path):
    _, metrics, warnings = run_integrity_checks(
        config=_config(require_anchors=True),
        rows=_rows(),
        outputs_by_row=["answer"] * 6,
        metric_samples=_samples([1.0] * 6),
        judges=[_judge(_SteadyJudge("yes"))],
        anchors=None,
    )
    assert ANCHOR_DRIFT_METRIC not in metrics
    assert any("require_anchors" in warning for warning in warnings)


def test_no_judges_with_configured_checks_warns():
    evidence, metrics, warnings = run_integrity_checks(
        config=_config(consistency_sample=4),
        rows=_rows(),
        outputs_by_row=["answer"] * 6,
        metric_samples={},
        judges=[],
        anchors=None,
    )
    assert evidence is None
    assert metrics == {}
    assert any("no row-level judge ran" in warning for warning in warnings)


def test_build_anchor_rows_freezes_scores_by_judge_name():
    rows = _rows(4)
    outputs = [f"answer {index}" for index in range(4)]
    samples = _samples([1.0, 0.0, None, 1.0])
    frozen = build_anchor_rows(
        rows=rows,
        outputs_by_row=outputs,
        metric_samples=samples,
        judges=[_judge(_SteadyJudge("yes"))],
        limit=10,
    )
    # Row 2 has no recorded score, so it cannot anchor anything.
    assert len(frozen) == 3
    assert all("correctness" in row.scores for row in frozen)
    assert frozen[0].outputs == "answer 0"
    assert frozen[1].scores["correctness"] == 0.0


def test_estimate_counts_both_checks_per_judge():
    config = _config(consistency_sample=8)
    assert estimate_integrity_calls(
        config, row_judges=2, dataset_rows=100, anchor_rows=12
    ) == 2 * (8 + 12)
    assert (
        estimate_integrity_calls(config, row_judges=0, dataset_rows=100, anchor_rows=12)
        == 0
    )
    # A dataset smaller than the sample caps the re-scored rows.
    assert (
        estimate_integrity_calls(config, row_judges=1, dataset_rows=3, anchor_rows=0)
        == 3
    )
