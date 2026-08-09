"""Unit tests for the baseline record, selection precedence, and drift."""

import json
from types import SimpleNamespace

import pytest

from aai_core.agentkit.baseline import (
    BaselineDataset,
    BaselineRecord,
    BaselineScope,
    BaselineVersions,
    comparability_failures,
    drift_warnings,
    load_baseline,
    select_baseline,
    write_baseline,
)
from aai_core.agentkit.datasets import (
    DatasetShape,
    LoadedDataset,
    dataset_digest,
    effective_dataset,
)
from aai_core.agentkit.errors import BaselineMissingError, ConfigError


def _record(**overrides):
    values = {
        "schema_version": 1,
        "run_id": "run-123",
        "experiment_id": "42",
        "recorded_at": "2026-08-02T10:00:00Z",
        "dataset": BaselineDataset(ref="golden.json", digest="abc123", rows=10),
        "scope": BaselineScope(mode="full", rows=10, seed=None),
        "metrics": {"keyword_coverage/mean": 0.7, "safety/mean": 1.0},
        "versions": BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 1},
            judge_model="endpoints:/judge",
            aai_core="0.4.0",
        ),
        "recorded_by": "agentkit compare --establish-baseline",
        "change_id": "9f31c2e",
    }
    values.update(overrides)
    return BaselineRecord(**values)


def _dataset(digest="abc123", rows=10):
    return LoadedDataset(
        ref="golden.json",
        source="local-json",
        rows=tuple({"inputs": {"q": str(i)}} for i in range(rows)),
        digest=digest,
        shape=DatasetShape(
            row_count=rows,
            input_keys=("q",),
            has_outputs=False,
            expectation_keys=(),
            has_traces=False,
            strata_values={},
        ),
    )


def test_round_trip_is_sorted_and_newline_terminated(tmp_path):
    path = tmp_path / "evals" / "baseline.json"
    record = _record()

    write_baseline(path, record)
    loaded, warnings = load_baseline(path)

    assert warnings == []
    assert loaded == record
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    document = json.loads(text)
    assert list(document) == sorted(document)


def test_legacy_metrics_only_file_upgrades_with_warning(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"metrics": {"correctness/mean": 0.8}}))

    record, warnings = load_baseline(path)

    assert record is not None
    assert dict(record.metrics) == {"correctness/mean": 0.8}
    assert record.run_id is None
    assert any("legacy" in warning for warning in warnings)
    assert any("--establish-baseline" in warning for warning in warnings)


def test_integer_metrics_coerce_to_floats():
    record = _record(metrics={"safety/mean": 1})
    assert record.metrics["safety/mean"] == 1.0


def test_corrupt_baseline_is_a_config_error(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text("{not json")

    with pytest.raises(ConfigError):
        load_baseline(path)

    path.write_text(json.dumps({"something": "else"}))
    with pytest.raises(ConfigError):
        load_baseline(path)


def test_missing_baseline_refuses_with_establish_guidance(tmp_path):
    with pytest.raises(BaselineMissingError) as excinfo:
        select_baseline(baseline_path=tmp_path / "evals" / "baseline.json")
    message = str(excinfo.value)
    assert "--establish-baseline" in message
    assert "IS the baseline" in message


def test_selection_precedence_flag_then_config_then_file(tmp_path):
    path = tmp_path / "baseline.json"
    write_baseline(path, _record())

    run = SimpleNamespace(
        info=SimpleNamespace(experiment_id="7"),
        data=SimpleNamespace(
            metrics={"correctness/mean": 0.9},
            tags={
                "aai.dataset": "golden.json",
                "aai.dataset_digest": "ddd",
                "aai.dataset_rows": "12",
                "aai.scorer_versions": "correctness=1,safety=1",
                "aai.judge_model": "endpoints:/judge",
                "aai.agent_target": "endpoints:/agent",
                "aai.agentkit_version": "0.4.0",
                "aai.change_id": "abc",
            },
        ),
    )
    fake_mlflow = SimpleNamespace(get_run=lambda run_id: run)

    from_flag, _ = select_baseline(
        baseline_path=path,
        flag_run_id="flag-run",
        config_run_id="config-run",
        mlflow_module=fake_mlflow,
    )
    assert from_flag.run_id == "flag-run"
    assert from_flag.recorded_by == "--baseline-run"
    assert dict(from_flag.versions.scorers) == {"correctness": 1, "safety": 1}

    from_config, _ = select_baseline(
        baseline_path=path, config_run_id="config-run", mlflow_module=fake_mlflow
    )
    assert from_config.run_id == "config-run"
    assert from_config.recorded_by == "baseline.run_id"

    from_file, _ = select_baseline(baseline_path=path)
    assert from_file.run_id == "run-123"


def test_unfetchable_run_is_a_baseline_error(tmp_path):
    def get_run(run_id):
        raise RuntimeError("no such run")

    with pytest.raises(BaselineMissingError) as excinfo:
        select_baseline(
            baseline_path=tmp_path / "baseline.json",
            flag_run_id="missing",
            mlflow_module=SimpleNamespace(get_run=get_run),
        )
    assert "missing" in str(excinfo.value)


def test_a_matching_baseline_is_comparable():
    record = _record()

    assert (
        comparability_failures(record, dataset=_dataset(), mode="full", rows=10) == []
    )


def test_a_changed_dataset_is_not_comparable():
    """A delta across different rows subtracts cleanly and means nothing."""

    failures = comparability_failures(
        _record(), dataset=_dataset(digest="other"), mode="full", rows=10
    )

    assert any("the dataset changed" in failure for failure in failures)


def test_changed_trace_ground_truth_is_not_comparable():
    """Trace assessments replace authored expectations in traces mode."""

    def _dataset_with(expected_response):
        rows = (
            {
                "inputs": {"question": "when can I retire?"},
                "trace": {
                    "info": {
                        "assessments": [
                            {
                                "assessment_name": "expected_response",
                                "expectation": {"value": expected_response},
                            }
                        ]
                    }
                },
            },
        )
        return LoadedDataset(
            ref="golden.json",
            source="local-json",
            rows=rows,
            digest=dataset_digest(rows),
            shape=DatasetShape(
                row_count=1,
                input_keys=("question",),
                has_outputs=False,
                expectation_keys=(),
                has_traces=True,
                strata_values={},
            ),
        )

    baseline_authored = _dataset_with("age 60")
    current_authored = _dataset_with("age 65")
    # Outside traces mode the recorded answer is deliberately not identity.
    assert baseline_authored.digest == current_authored.digest
    baseline = effective_dataset(baseline_authored, mode="traces")
    current = effective_dataset(current_authored, mode="traces")
    record = _record(
        dataset=BaselineDataset(ref="golden.json", digest=baseline.digest, rows=1),
        scope=BaselineScope(mode="full", rows=1, seed=None),
    )

    failures = comparability_failures(
        record,
        dataset=current,
        mode="full",
        rows=1,
    )

    assert baseline.digest != current.digest
    assert any("the dataset changed" in failure for failure in failures)


def test_a_changed_scope_is_not_comparable():
    failures = comparability_failures(
        _record(), dataset=_dataset(), mode="sample", rows=5
    )

    assert any("full/10 rows but this run scores sample/5" in f for f in failures)


def test_a_changed_scorer_version_is_not_comparable():
    """0.8 from v1 and 0.8 from v2 are not the same 0.8."""

    failures = comparability_failures(
        _record(),
        dataset=_dataset(),
        mode="full",
        rows=10,
        scorers={"keyword_coverage": 2},
    )

    assert any("keyword_coverage is v2" in failure for failure in failures)


def test_a_scorer_the_baseline_never_ran_is_not_comparable():
    """An added scorer has no baseline score to compare against."""

    failures = comparability_failures(
        _record(),
        dataset=_dataset(),
        mode="full",
        rows=10,
        scorers={"keyword_coverage": 1, "safety": 1},
    )

    assert any("this run scores safety" in failure for failure in failures)


def test_a_removed_scorer_is_not_comparable():
    """Removing a scorer also removes its threshold from the policy.

    That is a control disappearing because someone edited config, so the
    comparison must not quietly proceed without the evidence.
    """

    failures = comparability_failures(
        _record(),
        dataset=_dataset(),
        mode="full",
        rows=10,
        scorers={},
    )

    assert any(
        "the baseline scored keyword_coverage but this run does not" in failure
        for failure in failures
    )


def test_a_judge_free_run_is_not_punished_for_skipping_judges():
    """`smoke` runs code scorers only by design.

    A scorer missing because of the mode is not the same as one removed by
    configuration, and refusing here would break the fast loop entirely.
    """

    record = _record(
        versions=BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 1, "safety": 1, "correctness": 1},
            judge_model="endpoints:/judge",
            aai_core="0.4.0",
        )
    )

    assert (
        comparability_failures(
            record,
            dataset=_dataset(),
            mode="full",
            rows=10,
            scorers={"keyword_coverage": 1},
            judges_enabled=False,
        )
        == []
    )
    # With judges on, the same missing judges ARE a mismatch.
    assert comparability_failures(
        record,
        dataset=_dataset(),
        mode="full",
        rows=10,
        scorers={"keyword_coverage": 1},
        judges_enabled=True,
    )


def test_a_legacy_baseline_records_no_scorers_to_compare(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"metrics": {"m": 1.0}}))
    record, _ = load_baseline(path)

    assert (
        comparability_failures(
            record,
            dataset=_dataset(),
            mode="full",
            rows=0,
            scorers={"keyword_coverage": 1},
        )
        == []
    )


def test_a_moved_judge_prompt_is_not_comparable():
    """A moved alias is a different judge, so the delta is not evidence."""

    record = _record(
        versions=BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 1},
            judge_prompts={"pension_domain_policy": "prompts:/cat.sch.p/3"},
            aai_core="0.4.0",
        )
    )

    moved = comparability_failures(
        record,
        dataset=_dataset(),
        mode="full",
        rows=10,
        judge_prompts={"pension_domain_policy": "prompts:/cat.sch.p/4"},
    )
    assert any("judge prompt moved" in failure for failure in moved)

    unchanged = comparability_failures(
        record,
        dataset=_dataset(),
        mode="full",
        rows=10,
        judge_prompts={"pension_domain_policy": "prompts:/cat.sch.p/3"},
    )
    assert unchanged == []


def test_a_changed_judge_model_is_not_comparable():
    failures = comparability_failures(
        _record(),
        dataset=_dataset(),
        mode="full",
        rows=10,
        judge_model="endpoints:/other-judge",
    )

    assert any("the judge model changed" in failure for failure in failures)


def test_legacy_records_cannot_be_checked_and_say_so(tmp_path):
    """A baseline recorded before digests existed blocks nothing."""

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"metrics": {"m": 1.0}}))
    record, _ = load_baseline(path)

    assert comparability_failures(record, dataset=_dataset(), mode="full", rows=0) == []
    warnings = drift_warnings(record, dataset=_dataset(), mode="full", rows=0)
    assert any("predates dataset digests" in warning for warning in warnings)


def test_a_run_baseline_keeps_the_scope_it_was_scored_at(tmp_path):
    """A sampled baseline fetched by run id must stay a sampled baseline.

    Reconstructing it as `full` makes it incomparable with the very
    sampled run that produced it, so the comparability check would refuse
    a repeat of the same command.
    """

    run = SimpleNamespace(
        info=SimpleNamespace(experiment_id="42"),
        data=SimpleNamespace(
            tags={
                "aai.dataset": "golden.json",
                "aai.dataset_digest": "abc123",
                "aai.dataset_rows": "20",
                "aai.scope_mode": "sample",
                "aai.scope_rows": "20",
                "aai.scorer_versions": "keyword_coverage=1",
                "aai.agent_target": "src/app/example_agent.py:respond",
                "aai.recorded_at": "2026-08-02T10:00:00Z",
            },
            metrics={"keyword_coverage/mean": 0.8},
        ),
    )
    fake = SimpleNamespace(get_run=lambda run_id: run)

    record, _ = select_baseline(
        baseline_path=tmp_path / "missing.json",
        flag_run_id="run-9",
        mlflow_module=fake,
    )

    assert record.scope.mode == "sample"
    assert record.scope.rows == 20
    assert (
        comparability_failures(
            record, dataset=_dataset(digest="abc123", rows=20), mode="sample", rows=20
        )
        == []
    )


def test_a_run_baseline_without_scope_tags_reads_as_full(tmp_path):
    """Runs recorded before the scope tags existed still load."""

    run = SimpleNamespace(
        info=SimpleNamespace(experiment_id="42"),
        data=SimpleNamespace(
            tags={"aai.dataset_rows": "10", "aai.dataset_digest": "abc123"},
            metrics={},
        ),
    )
    fake = SimpleNamespace(get_run=lambda run_id: run)

    record, _ = select_baseline(
        baseline_path=tmp_path / "missing.json",
        flag_run_id="run-9",
        mlflow_module=fake,
    )

    assert record.scope.mode == "full"
    assert record.scope.rows == 10


def _prompt_record():
    return _record(
        versions=BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 1},
            judge_prompts={"pension_domain_policy": "prompts:/cat.sch.p/3"},
            aai_core="0.4.0",
        )
    )


def test_a_judge_prompt_that_stopped_resolving_is_not_comparable():
    """A deleted alias silently swaps in the bundled instructions.

    Nothing raises: the scorer keeps working, with different instructions.
    Comparing only the prompts this run resolved would never look at the
    entry that disappeared, so the judge changes and the delta is still
    accepted as evidence.
    """

    failures = comparability_failures(
        _prompt_record(),
        dataset=_dataset(),
        mode="full",
        rows=10,
        judge_prompts={},
    )

    assert any("no longer resolves" in failure for failure in failures)
    assert all("judge prompt" in failure for failure in failures)


def test_a_newly_registered_judge_prompt_is_not_comparable():
    """The same change in the other direction."""

    record = _record(
        versions=BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 1},
            judge_prompts={"other": "prompts:/cat.sch.other/1"},
            aai_core="0.4.0",
        )
    )

    failures = comparability_failures(
        record,
        dataset=_dataset(),
        mode="full",
        rows=10,
        judge_prompts={
            "other": "prompts:/cat.sch.other/1",
            "pension_domain_policy": "prompts:/cat.sch.p/3",
        },
    )

    assert any("bundled instructions" in failure for failure in failures)


def test_a_prompt_added_to_a_recorded_judge_is_not_comparable():
    """An empty prompt map can mean a governed judge used its fallback."""

    record = _record(
        versions=BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 1, "pension_domain_policy": 1},
            judge_prompts={},
            aai_core="0.4.0",
        )
    )

    failures = comparability_failures(
        record,
        dataset=_dataset(),
        mode="full",
        rows=10,
        judge_prompts={
            "pension_domain_policy": "prompts:/cat.sch.agentkit_judge_domain_policy/1"
        },
    )

    assert any("bundled instructions" in failure for failure in failures)


def test_a_baseline_with_no_recorded_prompts_still_compares():
    """A legacy or judge-free baseline says nothing about prompt membership."""

    failures = comparability_failures(
        _record(),
        dataset=_dataset(),
        mode="full",
        rows=10,
        judge_prompts={"pension_domain_policy": "prompts:/cat.sch.p/3"},
    )

    assert failures == []


def test_a_sample_of_the_recorded_dataset_is_not_a_changed_dataset():
    """Only the scope differs, and only the scope is reported."""

    sample = LoadedDataset(
        ref="golden.json",
        source="local-json+sample",
        rows=tuple({"inputs": {"q": str(i)}} for i in range(4)),
        digest="sampledigest",
        shape=DatasetShape(
            row_count=4,
            input_keys=("q",),
            has_outputs=False,
            expectation_keys=(),
            has_traces=False,
            strata_values={},
        ),
        sampled_from="abc123",
    )

    failures = comparability_failures(_record(), dataset=sample, mode="sample", rows=4)

    assert all("the dataset changed" not in failure for failure in failures)
    assert any("full/10 rows but this run scores sample/4" in f for f in failures)


def test_a_run_baseline_restores_its_judge_prompt_versions():
    """The prompt drift check is dead without this."""

    run = SimpleNamespace(
        info=SimpleNamespace(experiment_id="42"),
        data=SimpleNamespace(
            metrics={"keyword_coverage/mean": 0.7},
            tags={
                "aai.dataset": "golden.json",
                "aai.dataset_digest": "abc123",
                "aai.dataset_rows": "10",
                "aai.scorer_versions": "keyword_coverage=1",
                "aai.judge_prompt_versions": (
                    "pension_domain_policy=prompts:/cat.sch.p/3"
                ),
            },
        ),
    )
    mlflow = SimpleNamespace(get_run=lambda run_id: run)

    record, _ = select_baseline(
        baseline_path=None,
        flag_run_id="run-abc",
        config_run_id=None,
        mlflow_module=mlflow,
    )

    assert dict(record.versions.judge_prompts) == {
        "pension_domain_policy": "prompts:/cat.sch.p/3"
    }
    assert any(
        "judge prompt moved" in failure
        for failure in comparability_failures(
            record,
            dataset=_dataset(),
            mode="full",
            rows=10,
            judge_prompts={"pension_domain_policy": "prompts:/cat.sch.p/9"},
        )
    )


def test_a_repointed_judge_endpoint_is_not_comparable():
    """The endpoint name is stable; what it serves is not.

    A governed `endpoints:/judge` can be repointed at another model, or
    have a new version promoted behind it, without the URI changing — and
    two runs would then look comparable while being scored by different
    judges.
    """

    record = _record(
        versions=BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 1},
            judge_model="endpoints:/pension-judge",
            judge_model_identity="main.models.judge/3",
            aai_core="0.4.0",
        )
    )

    failures = comparability_failures(
        record,
        dataset=_dataset(),
        mode="full",
        rows=10,
        judge_model="endpoints:/pension-judge",
        judge_model_identity="main.models.judge/4",
    )

    assert any("not the same judge" in failure for failure in failures)


def test_the_same_served_entity_stays_comparable():
    record = _record(
        versions=BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 1},
            judge_model_identity="main.models.judge/3",
            aai_core="0.4.0",
        )
    )

    assert (
        comparability_failures(
            record,
            dataset=_dataset(),
            mode="full",
            rows=10,
            judge_model_identity="main.models.judge/3",
        )
        == []
    )


def test_an_unreadable_judge_identity_does_not_block():
    """A permission the CI principal may not hold must not fail the run.

    Section 4 of AGENTS.md forbids widening a grant to make a check pass,
    so an unreadable endpoint config is reported, not enforced.
    """

    record = _record(
        versions=BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 1},
            judge_model_identity="main.models.judge/3",
            aai_core="0.4.0",
        )
    )

    assert (
        comparability_failures(
            record,
            dataset=_dataset(),
            mode="full",
            rows=10,
            judge_model_identity=None,
        )
        == []
    )
