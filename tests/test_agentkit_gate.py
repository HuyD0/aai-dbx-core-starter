"""Unit tests for the promotion gate and the CI exit-code contract."""

import pytest

from aai_core.agentkit.baseline import (
    BaselineDataset,
    BaselineRecord,
    BaselineScope,
    BaselineVersions,
    write_baseline,
)
from aai_core.agentkit.catalog import select_scorers
from aai_core.agentkit.config import AgentkitConfig, ProjectContext
from aai_core.agentkit.datasets import DatasetShape
from aai_core.agentkit.gate import (
    EXIT_ERROR,
    EXIT_PASS,
    EXIT_THRESHOLD_FAILED,
    build_policy,
    evaluate_gate,
    render_report,
    run_gate,
)
from aai_core.agentkit.results import (
    ResultsRecord,
    begin_results_attempt,
    complete_results_attempt,
    write_results,
)
from aai_core.testing import dev_settings


def _project(tmp_path, **config_overrides):
    values = {
        "version": 1,
        "agent": "src/app/example_agent.py:respond",
        "dataset": "evals/data/golden_cases.json",
    }
    values.update(config_overrides)
    return ProjectContext(
        config=AgentkitConfig(**values), settings=dev_settings(), root=tmp_path
    )


def _shape():
    return DatasetShape(
        row_count=10,
        input_keys=("question",),
        has_outputs=True,
        expectation_keys=("expected_response",),
        has_traces=False,
        strata_values={},
    )


def _plan(project, judges_enabled=False):
    return select_scorers(
        _shape(),
        project.config,
        mode="answer-sheet",
        judges_enabled=judges_enabled,
    )


def _results(**overrides):
    values = {
        "command": "compare",
        "recorded_at": "2026-08-02T10:00:00Z",
        "run_id": "run-1",
        "agent": "src/app/example_agent.py:respond",
        "dataset": BaselineDataset(ref="golden.json", digest="abc123", rows=10),
        "scope": BaselineScope(mode="full", rows=10),
        "mode": "answer-sheet",
        "metrics": {
            "keyword_coverage/mean": 0.8,
            "refusal_compliance/mean": 1.0,
            "response_length_ok/mean": 1.0,
        },
        "versions": BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 1},
            aai_core="0.4.0",
        ),
        "baseline_run_id": "run-0",
        "baseline_metrics": {
            "keyword_coverage/mean": 0.75,
            "refusal_compliance/mean": 1.0,
            "response_length_ok/mean": 1.0,
        },
        "decision": "inconclusive",
        "change_id": "abc1234",
        "gate_passed": True,
    }
    values.update(overrides)
    return ResultsRecord(**values)


def _baseline():
    return BaselineRecord(
        schema_version=1,
        run_id="run-0",
        recorded_at="2026-08-01T10:00:00Z",
        dataset=BaselineDataset(ref="golden.json", digest="abc123", rows=10),
        scope=BaselineScope(mode="full", rows=10),
        metrics={"keyword_coverage/mean": 0.75},
        versions=BaselineVersions(agent="agent", aai_core="0.4.0"),
        recorded_by="agentkit compare --establish-baseline",
        change_id="0000000",
    )


def test_passing_results_exit_zero(tmp_path):
    project = _project(tmp_path)

    report, code = evaluate_gate(
        project, results=_results(), baseline=_baseline(), plan=_plan(project)
    )

    assert code == EXIT_PASS
    assert report.passed
    text = render_report(report)
    assert "gate: PASSED" in text
    assert "compared against run-0" in text
    assert "keyword_coverage=v1" in text


def test_threshold_failure_exits_two(tmp_path):
    project = _project(tmp_path)
    results = _results(metrics={"keyword_coverage/mean": 0.2})

    report, code = evaluate_gate(
        project, results=results, baseline=_baseline(), plan=_plan(project)
    )

    assert code == EXIT_THRESHOLD_FAILED
    assert not report.passed
    assert any(
        failure.metric == "keyword_coverage/mean" for failure in report.result.failures
    )


def test_missing_thresholded_metric_fails_closed(tmp_path):
    project = _project(tmp_path, thresholds={"correctness/mean": ">=0.7"})
    # The judge never ran, so correctness/mean is absent from the results.
    results = _results()

    report, code = evaluate_gate(project, results=results, baseline=_baseline())

    assert code == EXIT_THRESHOLD_FAILED
    assert any(
        failure.metric == "correctness/mean" and "missing" in failure.reason
        for failure in report.result.failures
    )


def test_results_without_a_comparison_are_rejected(tmp_path):
    project = _project(tmp_path)
    results = _results(baseline_run_id=None, baseline_metrics={})

    report, code = evaluate_gate(project, results=results, baseline=None)

    assert code == EXIT_THRESHOLD_FAILED
    assert "not a comparison" in report.message
    assert "compare" in report.message
    # The verdict must agree with the refusal: everything downstream (the
    # --json output, the evidence pack) reads `passed` off this report.
    assert not report.passed
    assert [failure.metric for failure in report.result.failures] == ["comparison"]


def test_established_baseline_counts_as_named_evidence(tmp_path):
    project = _project(tmp_path)
    results = _results(
        baseline_run_id=None, baseline_metrics={}, established_baseline=True
    )

    report, code = evaluate_gate(
        project, results=results, baseline=None, plan=_plan(project)
    )

    assert code == EXIT_PASS
    assert "this run IS the recorded baseline" in render_report(report)


def test_regression_budget_composes_the_shared_gate_engine(tmp_path):
    project = _project(tmp_path, regression_budget={"keyword_coverage/mean": 0.02})
    results = _results(
        metrics={
            "keyword_coverage/mean": 0.70,
            "refusal_compliance/mean": 1.0,
            "response_length_ok/mean": 1.0,
        }
    )

    report, code = evaluate_gate(
        project, results=results, baseline=_baseline(), plan=_plan(project)
    )

    assert code == EXIT_THRESHOLD_FAILED
    assert any("regressed" in failure.reason for failure in report.result.failures)


def test_catalog_defaults_apply_when_no_thresholds_configured(tmp_path):
    project = _project(tmp_path)

    policy = build_policy(project, plan=_plan(project))

    metrics = {rule.metric for rule in policy.rules}
    assert "keyword_coverage/mean" in metrics
    assert "refusal_compliance/mean" in metrics
    assert "response_length_ok/mean" in metrics


def test_thresholds_accept_scorer_names_and_metric_keys(tmp_path):
    project = _project(
        tmp_path,
        thresholds={"keyword_coverage": ">=0.9", "response_length_ok/mean": ">=1.0"},
    )

    policy = build_policy(project, plan=_plan(project))
    by_metric = {rule.metric: rule for rule in policy.rules}

    assert by_metric["keyword_coverage/mean"].required == 0.9
    assert by_metric["response_length_ok/mean"].required == 1.0


def test_run_gate_without_results_exits_one(tmp_path):
    project = _project(tmp_path)

    report, code, message = run_gate(project)

    assert report is None
    assert code == EXIT_ERROR
    assert "agentkit compare" in message


def test_run_gate_reads_the_newest_results_record(tmp_path):
    project = _project(tmp_path)
    write_baseline(project.baseline_path, _baseline())
    write_results(project.results_dir, _results(recorded_at="2026-08-02T09:00:00Z"))
    write_results(
        project.results_dir,
        _results(
            recorded_at="2026-08-02T11:00:00Z",
            metrics={"keyword_coverage/mean": 0.1},
        ),
    )

    report, code, message = run_gate(project)

    assert code == EXIT_THRESHOLD_FAILED
    assert message is None
    assert not report.passed


def test_pending_latest_attempt_blocks_an_older_passing_result(tmp_path):
    project = _project(tmp_path)
    write_baseline(project.baseline_path, _baseline())
    write_results(project.results_dir, _results())
    begin_results_attempt(project.results_dir, command="compare")

    report, code, message = run_gate(project)

    assert report is None
    assert code == EXIT_ERROR
    assert "latest evaluation attempt" in message


def test_completed_attempt_binds_gate_to_exact_result_bytes(tmp_path):
    project = _project(tmp_path)
    write_baseline(project.baseline_path, _baseline())
    attempt = begin_results_attempt(project.results_dir, command="compare")
    path = write_results(project.results_dir, _results())
    complete_results_attempt(project.results_dir, attempt, path)

    report, code, message = run_gate(project)
    assert code == EXIT_PASS
    assert message is None
    assert report.results.run_id == "run-1"

    path.write_text(path.read_text() + " ")
    report, code, message = run_gate(project)
    assert report is None
    assert code == EXIT_ERROR
    assert "changed after it was recorded" in message


def test_older_concurrent_completion_cannot_replace_the_latest_attempt(tmp_path):
    project = _project(tmp_path)
    write_baseline(project.baseline_path, _baseline())
    older = begin_results_attempt(project.results_dir, command="compare")
    latest = begin_results_attempt(project.results_dir, command="compare")
    path = write_results(project.results_dir, _results())

    complete_results_attempt(project.results_dir, older, path)

    report, code, message = run_gate(project)
    assert report is None
    assert code == EXIT_ERROR
    assert "latest evaluation attempt" in message
    assert older.attempt_id != latest.attempt_id


def test_run_gate_with_explicit_missing_path_exits_one(tmp_path):
    project = _project(tmp_path)

    report, code, message = run_gate(project, results_path=tmp_path / "nope.json")

    assert report is None
    assert code == EXIT_ERROR
    assert "does not exist" in message


@pytest.mark.parametrize(
    ("expression", "value", "expected_code"),
    [
        (">=0.7", 0.7, EXIT_PASS),
        (">0.7", 0.7, EXIT_THRESHOLD_FAILED),
        ("<=2", 2.0, EXIT_PASS),
        ("<2", 2.0, EXIT_THRESHOLD_FAILED),
    ],
)
def test_strict_and_inclusive_thresholds(tmp_path, expression, value, expected_code):
    project = _project(tmp_path, thresholds={"custom/mean": expression})
    results = _results(
        metrics={"custom/mean": value},
        versions=BaselineVersions(agent="agent", scorers={}, aai_core="0.4.0"),
    )

    _, code = evaluate_gate(project, results=results, baseline=_baseline())

    assert code == expected_code


def test_standalone_regression_budget_respects_registry_direction(tmp_path):
    """Latency is lower-is-better: slower must fail, faster must pass."""

    project = _project(tmp_path, regression_budget={"latency_seconds/mean": 0.5})
    versions = BaselineVersions(
        agent="agent", scorers={"latency_seconds": 1}, aai_core="0.4.0"
    )

    slower = _results(
        metrics={"latency_seconds/mean": 2.0},
        baseline_metrics={"latency_seconds/mean": 1.0},
        versions=versions,
    )
    faster = _results(
        metrics={"latency_seconds/mean": 0.2},
        baseline_metrics={"latency_seconds/mean": 1.0},
        versions=versions,
    )

    _, slower_code = evaluate_gate(project, results=slower, baseline=_baseline())
    _, faster_code = evaluate_gate(project, results=faster, baseline=_baseline())

    assert slower_code == EXIT_THRESHOLD_FAILED
    assert faster_code == EXIT_PASS


def test_standalone_regression_budget_defaults_to_higher_is_better(tmp_path):
    project = _project(tmp_path, regression_budget={"keyword_coverage/mean": 0.05})
    versions = BaselineVersions(
        agent="agent", scorers={"keyword_coverage": 1}, aai_core="0.4.0"
    )

    worse = _results(
        metrics={"keyword_coverage/mean": 0.7},
        baseline_metrics={"keyword_coverage/mean": 0.9},
        versions=versions,
    )

    _, code = evaluate_gate(project, results=worse, baseline=_baseline())

    assert code == EXIT_THRESHOLD_FAILED


def test_recorded_rules_survive_a_relaxed_config(tmp_path):
    """Relaxing a threshold must not turn a failed run into evidence.

    The record is judged by the rules that were in force when it was
    scored; a reader with a different agentkit.yaml gets the same verdict.
    """

    from aai_core.evaluation import MetricDirection, MetricRule

    project = _project(tmp_path, thresholds={"keyword_coverage": ">=0.9"})
    results = _results(
        metrics={"keyword_coverage/mean": 0.5},
        baseline_run_id="run-0",
        policy_rules=(
            MetricRule(
                metric="keyword_coverage/mean",
                direction=MetricDirection.HIGHER,
                required=0.9,
            ),
        ),
    )
    relaxed = _project(tmp_path, thresholds={"keyword_coverage": ">=0.1"})

    report, code = evaluate_gate(relaxed, results=results, baseline=None)

    assert code == EXIT_THRESHOLD_FAILED
    assert not report.passed
    # The rules applied are the record's, not the reader's.
    assert report.rules[0].required == 0.9
    assert report.policy_note is None
    assert project is not None


def test_records_without_recorded_rules_fall_back_and_say_so(tmp_path):
    project = _project(tmp_path, thresholds={"keyword_coverage": ">=0.9"})
    results = _results(metrics={"keyword_coverage/mean": 0.5}, baseline_run_id="run-0")

    report, code = evaluate_gate(project, results=results, baseline=None)

    assert code == EXIT_THRESHOLD_FAILED
    assert report.policy_note is not None
    assert "predate recorded gate rules" in report.policy_note


def test_a_newly_added_thresholded_scorer_is_drift(tmp_path):
    """Adding a scorer to config must not let a stale record pass.

    `run_gate` supplies no plan, so the live policy used to be derived
    from the scorers the *old run* recorded. A scorer added since then
    contributed its registry-default rule to neither side of the
    comparison, so a record with no metric for it exited 0 — the gate
    reporting a pass on evidence that predates the requirement.
    """

    from aai_core.agentkit.gate import _policy_drift

    project = _project(tmp_path, scorers={"add": ["correctness"]})
    results = _results(
        policy_rules=build_policy(project, scorer_names=("keyword_coverage",)).rules
    )

    drift = _policy_drift(project, results, None)

    assert drift is not None
    assert "correctness/mean" in drift


def test_a_removed_scorer_is_drift(tmp_path):
    from aai_core.agentkit.gate import _policy_drift

    recorded = build_policy(
        _project(tmp_path), scorer_names=("keyword_coverage",)
    ).rules
    project = _project(tmp_path, scorers={"remove": ["keyword_coverage"]})

    drift = _policy_drift(project, _results(policy_rules=recorded), None)

    assert drift is not None
    assert "keyword_coverage/mean" in drift


def test_an_unchanged_selection_is_not_drift(tmp_path):
    from aai_core.agentkit.gate import _policy_drift

    project = _project(tmp_path)
    results = _results(
        policy_rules=build_policy(project, scorer_names=("keyword_coverage",)).rules
    )

    assert _policy_drift(project, results, None) is None
