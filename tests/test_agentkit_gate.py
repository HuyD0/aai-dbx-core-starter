"""Unit tests for the promotion gate and the CI exit-code contract."""

from pathlib import Path
from threading import Event, Thread

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
from aai_core.agentkit.errors import ConfigError
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
    RESULTS_ATTEMPT_FILE,
    RESULTS_ATTEMPT_LOCK_FILE,
    RESULTS_ATTEMPT_TRANSITION_FILE,
    ResultsRecord,
    begin_results_attempt,
    complete_results_attempt,
    load_gate_results,
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
            scorers={"keyword_coverage": 2},
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
    assert "keyword_coverage=v2" in text


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


def test_legacy_results_fallback_requires_no_attempt_metadata(tmp_path):
    project = _project(tmp_path)
    path = write_results(project.results_dir, _results())

    loaded = load_gate_results(project.results_dir)

    assert loaded is not None
    assert loaded[0].run_id == "run-1"
    assert loaded[1] == path
    assert {entry.name for entry in project.results_dir.iterdir()} == {
        path.name,
        RESULTS_ATTEMPT_LOCK_FILE,
    }

    # The reader-created lock is coordination, not attempt metadata; a later
    # legacy read must still use the same compatibility fallback.
    assert load_gate_results(project.results_dir) == loaded


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
    path = write_results(
        project.results_dir,
        _results(attempt_id=attempt.attempt_id),
        attempt=attempt,
    )
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


@pytest.mark.parametrize(
    "overrides",
    [
        {"attempt_id": "0" * 32},
        {"command": "smoke"},
    ],
)
def test_results_must_match_the_attempt_they_complete(tmp_path, overrides):
    project = _project(tmp_path)
    attempt = begin_results_attempt(project.results_dir, command="compare")
    values = {"attempt_id": attempt.attempt_id, **overrides}

    with pytest.raises(ConfigError, match="not bound"):
        write_results(
            project.results_dir,
            _results(**values),
            attempt=attempt,
        )


def test_older_concurrent_completion_cannot_replace_the_latest_attempt(tmp_path):
    project = _project(tmp_path)
    write_baseline(project.baseline_path, _baseline())
    older = begin_results_attempt(project.results_dir, command="compare")
    latest = begin_results_attempt(project.results_dir, command="compare")
    older_path = write_results(
        project.results_dir,
        _results(attempt_id=older.attempt_id, run_id="run-older"),
        attempt=older,
    )
    latest_path = write_results(
        project.results_dir,
        _results(attempt_id=latest.attempt_id, run_id="run-latest"),
        attempt=latest,
    )

    assert older_path != latest_path
    assert older.attempt_id in older_path.name
    assert latest.attempt_id in latest_path.name

    # The latest command finishes first, then the older concurrent command
    # completes out of order. Its state update must not replace or corrupt
    # the result the pointer names.
    complete_results_attempt(project.results_dir, latest, latest_path)
    complete_results_attempt(project.results_dir, older, older_path)

    record, path = load_gate_results(project.results_dir)

    assert path == latest_path
    assert record.run_id == "run-latest"
    assert older.attempt_id != latest.attempt_id


def test_gate_parses_the_same_result_bytes_it_hashes(tmp_path, monkeypatch):
    project = _project(tmp_path)
    attempt = begin_results_attempt(project.results_dir, command="compare")
    result_path = write_results(
        project.results_dir,
        _results(attempt_id=attempt.attempt_id),
        attempt=attempt,
    )
    complete_results_attempt(project.results_dir, attempt, result_path)
    original_read_bytes = Path.read_bytes

    def replace_after_read(path):
        payload = original_read_bytes(path)
        if path == result_path:
            path.write_text("[]", encoding="utf-8")
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)

    record, loaded_path = load_gate_results(project.results_dir)

    assert loaded_path == result_path
    assert record.run_id == "run-1"
    assert result_path.read_text(encoding="utf-8") == "[]"


def test_attempt_state_write_failure_leaves_the_gate_invalidated(tmp_path, monkeypatch):
    from aai_core.agentkit import results as results_module

    project = _project(tmp_path)
    old = begin_results_attempt(project.results_dir, command="compare")
    old_path = write_results(
        project.results_dir,
        _results(attempt_id=old.attempt_id),
        attempt=old,
    )
    complete_results_attempt(project.results_dir, old, old_path)
    calls = []
    write_contract = results_module._write_contract_file

    def fail_state(path, document):
        calls.append(path)
        if results_module._is_attempt_state_name(path.name):
            raise OSError("disk full")
        write_contract(path, document)

    monkeypatch.setattr(results_module, "_write_contract_file", fail_state)

    with pytest.raises(OSError, match="disk full"):
        begin_results_attempt(project.results_dir, command="compare")

    assert calls[0] == project.results_dir / RESULTS_ATTEMPT_TRANSITION_FILE
    assert results_module._is_attempt_state_name(calls[1].name)
    with pytest.raises(ConfigError, match="did not finish initializing"):
        load_gate_results(project.results_dir)


def test_transition_write_failure_first_moves_an_old_pass_out_of_service(
    tmp_path, monkeypatch
):
    from aai_core.agentkit import results as results_module

    project = _project(tmp_path)
    old = begin_results_attempt(project.results_dir, command="compare")
    old_path = write_results(
        project.results_dir,
        _results(attempt_id=old.attempt_id),
        attempt=old,
    )
    complete_results_attempt(project.results_dir, old, old_path)
    assert load_gate_results(project.results_dir)[0].gate_passed
    write_contract = results_module._write_contract_file

    def fail_transition(path, document):
        if path.name == RESULTS_ATTEMPT_TRANSITION_FILE:
            raise OSError("transition write failed")
        write_contract(path, document)

    monkeypatch.setattr(results_module, "_write_contract_file", fail_transition)

    with pytest.raises(OSError, match="transition write failed"):
        begin_results_attempt(project.results_dir, command="compare")

    assert not (project.results_dir / RESULTS_ATTEMPT_FILE).exists()
    assert (project.results_dir / RESULTS_ATTEMPT_TRANSITION_FILE).exists()
    with pytest.raises(ConfigError, match="did not finish initializing"):
        load_gate_results(project.results_dir)


def test_pointer_write_failure_cannot_leave_an_old_pass_gate_eligible(
    tmp_path, monkeypatch
):
    from aai_core.agentkit import results as results_module

    project = _project(tmp_path)
    old = begin_results_attempt(project.results_dir, command="compare")
    old_path = write_results(
        project.results_dir,
        _results(attempt_id=old.attempt_id),
        attempt=old,
    )
    complete_results_attempt(project.results_dir, old, old_path)
    assert load_gate_results(project.results_dir)[0].gate_passed
    write_contract = results_module._write_contract_file

    def fail_pointer(path, document):
        if path.name == RESULTS_ATTEMPT_FILE:
            raise OSError("pointer replace failed")
        write_contract(path, document)

    monkeypatch.setattr(results_module, "_write_contract_file", fail_pointer)

    with pytest.raises(OSError, match="pointer replace failed"):
        begin_results_attempt(project.results_dir, command="compare")

    assert not (project.results_dir / RESULTS_ATTEMPT_FILE).exists()
    assert (project.results_dir / RESULTS_ATTEMPT_TRANSITION_FILE).exists()
    with pytest.raises(ConfigError, match="did not finish initializing"):
        load_gate_results(project.results_dir)


def test_gate_read_and_attempt_begin_are_serialized_across_all_evidence_bytes(
    tmp_path, monkeypatch
):
    from aai_core.agentkit import results as results_module

    project = _project(tmp_path)
    old = begin_results_attempt(project.results_dir, command="compare")
    old_path = write_results(
        project.results_dir,
        _results(attempt_id=old.attempt_id),
        attempt=old,
    )
    complete_results_attempt(project.results_dir, old, old_path)

    reader_inside = Event()
    release_reader = Event()
    pointer_moved = Event()
    writer_started = Event()
    reader_results = []
    reader_errors = []
    writer_errors = []
    parse_results = results_module._parse_results_bytes
    replace = results_module.os.replace

    def blocking_parse(path, payload):
        if path == old_path:
            reader_inside.set()
            release_reader.wait(timeout=2)
        return parse_results(path, payload)

    def tracking_replace(source, destination):
        if Path(destination).name == RESULTS_ATTEMPT_TRANSITION_FILE:
            pointer_moved.set()
        return replace(source, destination)

    monkeypatch.setattr(results_module, "_parse_results_bytes", blocking_parse)
    monkeypatch.setattr(results_module.os, "replace", tracking_replace)

    def read_gate():
        try:
            reader_results.append(load_gate_results(project.results_dir))
        except Exception as error:  # pragma: no cover - asserted below
            reader_errors.append(error)

    def begin_attempt():
        writer_started.set()
        try:
            begin_results_attempt(project.results_dir, command="compare")
        except Exception as error:  # pragma: no cover - asserted below
            writer_errors.append(error)

    reader = Thread(target=read_gate)
    writer = Thread(target=begin_attempt)
    reader.start()
    assert reader_inside.wait(timeout=2)
    writer.start()
    assert writer_started.wait(timeout=2)
    moved_while_reader_held_lock = pointer_moved.wait(timeout=0.2)
    release_reader.set()
    reader.join(timeout=2)
    writer.join(timeout=2)

    assert not reader.is_alive()
    assert not writer.is_alive()
    assert not moved_while_reader_held_lock
    assert pointer_moved.is_set()
    assert reader_errors == []
    assert writer_errors == []
    assert reader_results[0][0].run_id == "run-1"
    with pytest.raises(ConfigError, match="did not publish"):
        load_gate_results(project.results_dir)


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
        agent="agent", scorers={"keyword_coverage": 2}, aai_core="0.4.0"
    )

    worse = _results(
        metrics={"keyword_coverage/mean": 0.7},
        baseline_metrics={"keyword_coverage/mean": 0.9},
        versions=versions,
    )

    _, code = evaluate_gate(project, results=worse, baseline=_baseline())

    assert code == EXIT_THRESHOLD_FAILED


def test_standalone_regression_budget_respects_economics_direction(tmp_path):
    """Cost is lower-is-better: pricier must fail, cheaper must pass.

    The registry answers "higher" for metrics it does not know, so without
    the economics direction table a falling cost would read as regression.
    """

    project = _project(
        tmp_path, regression_budget={"economics/cost_per_success_usd": 0.01}
    )
    versions = BaselineVersions(agent="agent", scorers={}, aai_core="0.4.0")

    pricier = _results(
        metrics={"economics/cost_per_success_usd": 0.05},
        baseline_metrics={"economics/cost_per_success_usd": 0.02},
        versions=versions,
    )
    cheaper = _results(
        metrics={"economics/cost_per_success_usd": 0.005},
        baseline_metrics={"economics/cost_per_success_usd": 0.02},
        versions=versions,
    )

    _, pricier_code = evaluate_gate(project, results=pricier, baseline=_baseline())
    _, cheaper_code = evaluate_gate(project, results=cheaper, baseline=_baseline())

    assert pricier_code == EXIT_THRESHOLD_FAILED
    assert cheaper_code == EXIT_PASS


def test_configured_economics_thresholds_gate_and_fail_closed(tmp_path):
    """Economics gating is opt-in through the ordinary threshold grammar."""

    project = _project(
        tmp_path,
        thresholds={
            "cost/coverage": ">=1.0",
            "economics/cost_per_success_usd": "<=0.03",
        },
    )
    versions = BaselineVersions(agent="agent", scorers={}, aai_core="0.4.0")

    passing = _results(
        metrics={"cost/coverage": 1.0, "economics/cost_per_success_usd": 0.02},
        versions=versions,
    )
    pricey = _results(
        metrics={"cost/coverage": 1.0, "economics/cost_per_success_usd": 0.05},
        versions=versions,
    )
    absent = _results(metrics={}, versions=versions)

    assert evaluate_gate(project, results=passing, baseline=_baseline())[1] == EXIT_PASS
    assert (
        evaluate_gate(project, results=pricey, baseline=_baseline())[1]
        == EXIT_THRESHOLD_FAILED
    )
    # Coverage the run never produced fails closed, not open.
    assert (
        evaluate_gate(project, results=absent, baseline=_baseline())[1]
        == EXIT_THRESHOLD_FAILED
    )


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


# --- release binding --------------------------------------------------------


def _bound_results(**overrides):
    """Results as a CI release-gate run would record them."""

    values = {
        "release": "a" * 40,
        "change_id": ("a" * 40)[:12],
    }
    values.update(overrides)
    return _results(**values)


def test_gate_refuses_results_from_another_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("AAI_RELEASE", "b" * 40)
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    project = _project(tmp_path)

    report, code = evaluate_gate(
        project,
        results=_bound_results(),
        baseline=_baseline(),
        plan=_plan(project),
        check_release_binding=True,
    )

    assert code == EXIT_THRESHOLD_FAILED
    assert not report.passed
    assert [failure.metric for failure in report.result.failures] == ["release"]
    assert "scored for commit" in report.message
    assert ("b" * 40)[:12] in report.message


def test_gate_accepts_results_for_the_gated_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("AAI_RELEASE", "a" * 40)
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    project = _project(tmp_path)

    report, code = evaluate_gate(
        project,
        results=_bound_results(),
        baseline=_baseline(),
        plan=_plan(project),
        check_release_binding=True,
    )

    assert code == EXIT_PASS
    assert report.passed


def test_gate_accepts_a_change_id_match_without_a_release(tmp_path, monkeypatch):
    # Older records carry only the 12-char change id; GIT_COMMIT-driven
    # environments must still bind on it.
    monkeypatch.delenv("AAI_RELEASE", raising=False)
    monkeypatch.setenv("GIT_COMMIT", "c" * 40)
    project = _project(tmp_path)

    report, code = evaluate_gate(
        project,
        results=_results(release=None, change_id=("c" * 40)[:12]),
        baseline=_baseline(),
        plan=_plan(project),
        check_release_binding=True,
    )

    assert code == EXIT_PASS


def test_local_dev_skips_the_release_binding(tmp_path, monkeypatch):
    monkeypatch.setenv("AAI_RELEASE", "local-dev")
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    project = _project(tmp_path)

    report, code = evaluate_gate(
        project,
        results=_results(),
        baseline=_baseline(),
        plan=_plan(project),
        check_release_binding=True,
    )

    assert code == EXIT_PASS


def test_release_binding_is_off_unless_requested(tmp_path, monkeypatch):
    # `agentkit evidence --run <id>` renders records from other machines;
    # only the promotion path (`run_gate`) opts in.
    monkeypatch.setenv("AAI_RELEASE", "b" * 40)
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    project = _project(tmp_path)

    report, code = evaluate_gate(
        project,
        results=_bound_results(),
        baseline=_baseline(),
        plan=_plan(project),
    )

    assert code == EXIT_PASS


def test_run_gate_binds_to_the_release_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("AAI_RELEASE", "b" * 40)
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    project = _project(tmp_path)
    directory = project.results_dir
    attempt = begin_results_attempt(directory, command="compare")
    path = write_results(
        directory, _bound_results(attempt_id=attempt.attempt_id), attempt=attempt
    )
    complete_results_attempt(directory, attempt, path)

    report, code, message = run_gate(project)

    assert code == EXIT_THRESHOLD_FAILED
    assert "scored for commit" in message


# --- judge-integrity rules --------------------------------------------------


def test_integrity_rules_join_the_policy_only_for_judged_runs(tmp_path):
    from aai_core.agentkit.integrity import (
        ANCHOR_DRIFT_METRIC,
        SELF_INCONSISTENCY_METRIC,
    )

    project = _project(
        tmp_path,
        integrity={"consistency_sample": 8, "require_anchors": True},
    )

    judged = build_policy(
        project, scorer_names=("keyword_coverage",), judges_enabled=True
    )
    judged_metrics = {rule.metric for rule in judged.rules}
    assert SELF_INCONSISTENCY_METRIC in judged_metrics
    assert ANCHOR_DRIFT_METRIC in judged_metrics

    unjudged = build_policy(
        project, scorer_names=("keyword_coverage",), judges_enabled=False
    )
    unjudged_metrics = {rule.metric for rule in unjudged.rules}
    assert SELF_INCONSISTENCY_METRIC not in unjudged_metrics
    assert ANCHOR_DRIFT_METRIC not in unjudged_metrics


def test_a_judged_record_missing_integrity_evidence_fails_closed(tmp_path):
    from aai_core.agentkit.integrity import SELF_INCONSISTENCY_METRIC

    project = _project(tmp_path, integrity={"consistency_sample": 8})
    # The record says judges ran, but carries no self-inconsistency metric
    # and predates recorded gate rules — the current policy applies.
    results = _results(judges_enabled=True)

    report, code = evaluate_gate(project, results=results, baseline=_baseline())

    assert code == EXIT_THRESHOLD_FAILED
    assert any(
        failure.metric == SELF_INCONSISTENCY_METRIC and "missing" in failure.reason
        for failure in report.result.failures
    )


def test_anchor_drift_failure_explains_the_instrument_moved(tmp_path):
    from aai_core.agentkit.integrity import ANCHOR_DRIFT_METRIC

    project = _project(
        tmp_path,
        integrity={"require_anchors": True, "max_anchor_drift": 0.1},
    )
    results = _results(
        judges_enabled=True,
        metrics={
            "keyword_coverage/mean": 0.8,
            "refusal_compliance/mean": 1.0,
            "response_length_ok/mean": 1.0,
            ANCHOR_DRIFT_METRIC: 0.4,
        },
    )

    report, code = evaluate_gate(project, results=results, baseline=_baseline())

    assert code == EXIT_THRESHOLD_FAILED
    text = render_report(report)
    assert "the judge changed, not the agent" in text


def test_enabling_integrity_is_policy_drift_for_older_records(tmp_path):
    from aai_core.agentkit.gate import _policy_drift
    from aai_core.agentkit.integrity import SELF_INCONSISTENCY_METRIC

    plain = _project(tmp_path)
    recorded = build_policy(
        plain, scorer_names=("keyword_coverage",), judges_enabled=True
    ).rules
    tightened = _project(tmp_path, integrity={"consistency_sample": 8})

    drift = _policy_drift(
        tightened,
        _results(policy_rules=recorded, judges_enabled=True),
        None,
    )

    assert drift is not None
    assert SELF_INCONSISTENCY_METRIC in drift


def test_run_gate_requires_calibration_when_configured(tmp_path, monkeypatch):
    from aai_core.agentkit.calibration import (
        CalibrationRecord,
        calibration_path,
        write_calibration,
    )

    monkeypatch.delenv("AAI_RELEASE", raising=False)
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    project = _project(tmp_path, integrity={"require_calibration": True})
    directory = project.results_dir
    attempt = begin_results_attempt(directory, command="compare")
    results = _results(
        attempt_id=attempt.attempt_id,
        judges_enabled=True,
        metrics={
            "keyword_coverage/mean": 0.8,
            "refusal_compliance/mean": 1.0,
            "response_length_ok/mean": 1.0,
            "safety/mean": 1.0,
        },
        versions=BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 2, "safety": 1},
            aai_core="0.4.0",
        ),
    )
    path = write_results(directory, results, attempt=attempt)
    complete_results_attempt(directory, attempt, path)

    report, code, message = run_gate(project)

    assert code == EXIT_THRESHOLD_FAILED
    assert [failure.metric for failure in report.result.failures] == ["calibration"]
    assert "agentkit judge calibrate" in message

    write_calibration(
        calibration_path(tmp_path, "evals/judges", "safety"),
        CalibrationRecord(
            scorer="safety",
            scorer_version=1,
            labels_digest="0" * 64,
            sample_size=20,
            annotator_count=2,
            percent_agreement=0.9,
            kappa=0.8,
            passed=True,
            recorded_at="2026-08-19T10:00:00Z",
        ),
    )

    report, code, message = run_gate(project)

    # The code-scorer keyword_coverage needs no calibration; the judged
    # safety scorer is now covered, so the ordinary policy applies.
    assert code == EXIT_PASS
