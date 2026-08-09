"""Results persistence and MLflow artifact contract tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_core.agentkit.baseline import (
    BaselineDataset,
    BaselineScope,
    BaselineVersions,
)
from aai_core.agentkit.errors import ConfigError
from aai_core.agentkit.results import (
    RESULTS_ARTIFACT_PATH,
    ResultsRecord,
    fetch_results,
    publish_results,
    read_results,
    write_results,
)


def _record(run_id: str, **overrides) -> ResultsRecord:
    values = {
        "command": "compare",
        "recorded_at": "2026-08-09T18:35:00Z",
        "run_id": run_id,
        "experiment_id": "42",
        "experiment_name": "/Shared/agent-evaluation",
        "agent": "src/app/example_agent.py:respond",
        "dataset": BaselineDataset(ref="golden.json", digest="abc123", rows=10),
        "scope": BaselineScope(mode="full", rows=10),
        "mode": "live",
        "metrics": {"correctness/mean": 0.9},
        "versions": BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"correctness": 1},
            aai_core="0.4.0",
        ),
        "decision": "inconclusive",
        "change_id": "abc1234",
        "gate_passed": True,
    }
    values.update(overrides)
    return ResultsRecord(**values)


def test_real_mlflow_round_trip_uses_the_canonical_artifact_path(tmp_path):
    """The path published by pinned MLflow is exactly the path fetch reads."""

    mlflow = pytest.importorskip("mlflow")
    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking)
    try:
        experiment_id = mlflow.create_experiment(
            "agentkit-results-round-trip", artifact_location=artifacts.as_uri()
        )
        with mlflow.start_run(experiment_id=experiment_id) as active:
            run_id = active.info.run_id

        record = _record(run_id)
        assert publish_results(mlflow, run_id, record) is None

        artifacts = mlflow.MlflowClient().list_artifacts(run_id, "agentkit")
        assert [artifact.path for artifact in artifacts] == [RESULTS_ARTIFACT_PATH]
        assert fetch_results(run_id, mlflow_module=mlflow) == record
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_fetched_results_must_be_bound_to_the_requested_run(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(_record("different-run").model_dump_json(), encoding="utf-8")
    fake = SimpleNamespace(
        artifacts=SimpleNamespace(
            download_artifacts=lambda **kwargs: str(path),
        )
    )

    with pytest.raises(ConfigError) as excinfo:
        fetch_results("requested-run", mlflow_module=fake)

    assert "different-run" in str(excinfo.value)
    assert "requested-run" in str(excinfo.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["metrics", "baseline_metrics"])
def test_results_reject_non_finite_metric_evidence(field, value):
    with pytest.raises(ValueError, match="finite"):
        _record("run-1", **{field: {"correctness/mean": value}})


def test_written_results_are_strict_standard_json(tmp_path):
    path = write_results(tmp_path, _record("run-1"))

    document = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail(f"non-standard number: {value}"),
    )

    assert document["metrics"] == {"correctness/mean": 0.9}


def test_read_results_normalizes_malformed_and_unreadable_artifacts(tmp_path):
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b"\xff\xfe")
    directory = tmp_path / "directory.json"
    directory.mkdir()

    for path in (array, invalid_utf8, directory, tmp_path / "missing.json"):
        with pytest.raises(ConfigError):
            read_results(path)


@pytest.mark.parametrize("artifact_kind", ["array", "invalid-utf8", "directory"])
def test_fetch_results_normalizes_malformed_artifacts(tmp_path, artifact_kind):
    path = tmp_path / "downloaded"
    if artifact_kind == "array":
        path.write_text("[]", encoding="utf-8")
    elif artifact_kind == "invalid-utf8":
        path.write_bytes(b"\xff")
    else:
        path.mkdir()
    fake = SimpleNamespace(
        artifacts=SimpleNamespace(download_artifacts=lambda **kwargs: str(path))
    )

    with pytest.raises(ConfigError):
        fetch_results("run-1", mlflow_module=fake)


def test_fetch_results_normalizes_an_embedded_nul_artifact_path():
    fake = SimpleNamespace(
        artifacts=SimpleNamespace(download_artifacts=lambda **kwargs: "bad\0path")
    )

    with pytest.raises(ConfigError, match="could not read results record"):
        fetch_results("run-1", mlflow_module=fake)


def test_read_results_normalizes_io_errors(tmp_path, monkeypatch):
    path = tmp_path / "results.json"
    path.write_text(_record("run-1").model_dump_json(), encoding="utf-8")
    original = Path.read_bytes

    def fail_read(candidate):
        if candidate == path:
            raise PermissionError("denied")
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", fail_read)

    with pytest.raises(ConfigError, match="could not read"):
        read_results(path)
