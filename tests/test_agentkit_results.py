"""Results persistence and MLflow artifact contract tests."""

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
)


def _record(run_id: str) -> ResultsRecord:
    return ResultsRecord(
        command="compare",
        recorded_at="2026-08-09T18:35:00Z",
        run_id=run_id,
        experiment_id="42",
        experiment_name="/Shared/agent-evaluation",
        agent="src/app/example_agent.py:respond",
        dataset=BaselineDataset(ref="golden.json", digest="abc123", rows=10),
        scope=BaselineScope(mode="full", rows=10),
        mode="live",
        metrics={"correctness/mean": 0.9},
        versions=BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"correctness": 1},
            aai_core="0.4.0",
        ),
        decision="inconclusive",
        change_id="abc1234",
        gate_passed=True,
    )


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
