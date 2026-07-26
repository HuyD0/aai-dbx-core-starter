"""Unit tests for the experiment logging and reproducibility helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_core.evaluation import QualityThreshold, apply_thresholds
from aai_core.experiments import ExperimentManager, record_reproducibility
from aai_core.testing import dev_settings


class FakeMlflow:
    def __init__(self):
        self.params: dict = {}
        self.metrics: dict = {}
        self.tags: dict = {}
        self.inputs: list = []
        self.artifacts: list = []
        self.data = SimpleNamespace(
            from_pandas=lambda data, name, source: SimpleNamespace(
                digest="digest-123", name=name, data=data, source=source
            )
        )

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None, nested=False):
        class _Run:
            def __enter__(self):
                return SimpleNamespace(info=SimpleNamespace(run_name=run_name))

            def __exit__(self, *args):
                return False

        return _Run()

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics, step=None):
        self.metrics.update(metrics)

    def log_input(self, dataset, context=None):
        self.inputs.append((dataset, context))

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((Path(path).name, artifact_path))

    def log_artifacts(self, path, artifact_path=None):
        self.artifacts.append((Path(path).name, artifact_path))


def _manager(fake):
    return ExperimentManager(
        experiment_name="/Shared/test",
        context=dev_settings().resource,
        mlflow_module=fake,
    )


def test_log_dataset_records_input_and_digest_params():
    fake = FakeMlflow()

    digest = _manager(fake).log_dataset(
        [{"x": 1}], name="sample", context="training", source="data/sample.csv"
    )

    assert digest == "digest-123"
    assert fake.inputs[0][1] == "training"
    assert fake.params["dataset_digest"] == "digest-123"
    assert fake.params["dataset_name"] == "sample"


def test_log_metrics_rejects_non_numeric_values():
    fake = FakeMlflow()
    manager = _manager(fake)

    manager.log_metrics({"accuracy": 0.95})
    assert fake.metrics == {"accuracy": 0.95}
    with pytest.raises(ValueError, match="not numeric"):
        manager.log_metrics({"verdict": "good"})
    with pytest.raises(ValueError, match="not numeric"):
        manager.log_metrics({"flag": True})


def test_log_artifact_handles_files_and_directories(tmp_path):
    fake = FakeMlflow()
    manager = _manager(fake)
    file_path = tmp_path / "summary.json"
    file_path.write_text("{}")

    manager.log_artifact(file_path, artifact_path="reports")
    manager.log_artifact(tmp_path)

    assert ("summary.json", "reports") in fake.artifacts
    assert (tmp_path.name, None) in fake.artifacts


def test_record_reproducibility_logs_commit_seed_and_freeze(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    fake = FakeMlflow()

    record = record_reproducibility(seed=7, mlflow_module=fake)

    assert record["source_commit"] == "abc123"
    assert record["seed"] == "7"
    assert fake.params["seed"] == "7"
    assert ("requirements-frozen.txt", "reproducibility") in fake.artifacts
    assert fake.tags["aai.environment_digest"] == record["environment_digest"]


def test_record_reproducibility_refuses_sensitive_extras():
    with pytest.raises(ValueError, match="sensitive"):
        record_reproducibility(extra={"api_key": "value"}, mlflow_module=FakeMlflow())


def test_apply_thresholds_gates_precomputed_metrics():
    thresholds = [
        QualityThreshold(
            metric="accuracy", direction="higher", required=0.9, max_regression=0.02
        )
    ]

    passing = apply_thresholds({"accuracy": 0.95}, thresholds)
    assert passing.passed

    failing = apply_thresholds(
        {"accuracy": 0.95}, thresholds, baseline_metrics={"accuracy": 0.99}
    )
    assert not failing.passed
    assert "regressed" in failing.failures[0].reason


def test_publish_report_renders_and_appends_summary(tmp_path):
    from aai_core.evaluation import EvaluationReport, GateFailure, publish_report

    report = EvaluationReport(
        metrics={"accuracy": 0.95, "safety/mean": 0.5},
        failures=(GateFailure("safety/mean", "0.5 violates required 1.0"),),
    )
    summary = tmp_path / "summary.md"

    markdown = publish_report(
        report,
        title="Release gate",
        baseline={"accuracy": 0.97},
        run_link="https://example/run",
        summary_path=summary,
    )

    assert "| accuracy | 0.950 | 0.970 | -0.020 | ok |" in markdown
    assert "| safety/mean | 0.500 | — | — | FAIL |" in markdown
    assert "**Result: FAILED**" in markdown
    assert "https://example/run" in markdown
    assert markdown in summary.read_text()
