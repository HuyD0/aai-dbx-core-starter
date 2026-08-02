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
from aai_core.agentkit.datasets import DatasetShape, LoadedDataset
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
    # A scorer the baseline never ran is not a mismatch; it simply has no
    # baseline value, which the gate already handles.
    assert (
        comparability_failures(
            _record(),
            dataset=_dataset(),
            mode="full",
            rows=10,
            scorers={"keyword_coverage": 1, "safety": 1},
        )
        == []
    )


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
