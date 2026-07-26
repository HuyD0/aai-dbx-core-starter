"""Hermetic unit tests: fakes only, no cloud, no pandas."""

from pathlib import Path
from types import SimpleNamespace

from aai_core.testing import dev_context
from app.experiment import DEFAULT_SEED, evaluate_rows, run_experiment

ROWS = [
    {"example_id": 1, "feature_signal": 12.5, "label": 0},
    {"example_id": 2, "feature_signal": 87.0, "label": 1},
    {"example_id": 3, "feature_signal": 45.3, "label": 0},
    {"example_id": 4, "feature_signal": 91.8, "label": 1},
]


class FakeMlflow:
    def __init__(self):
        self.params = {}
        self.metrics = {}
        self.tags = {}
        self.inputs = []
        self.artifacts = []
        self.data = SimpleNamespace(
            from_pandas=lambda data, name, source: SimpleNamespace(
                digest="digest-1", name=name
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
        self.inputs.append(dataset)

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((Path(path).name, artifact_path))


def test_evaluate_rows_is_deterministic_for_a_seed():
    first = evaluate_rows(ROWS, seed=DEFAULT_SEED)
    second = evaluate_rows(ROWS, seed=DEFAULT_SEED)

    assert first == second
    assert first["accuracy"] == 1.0
    assert first["example_count"] == 4.0


def test_run_experiment_records_everything_a_rerun_needs():
    fake = FakeMlflow()

    metrics = run_experiment(dev_context(), data=ROWS, mlflow_module=fake)

    assert fake.metrics["accuracy"] == metrics["accuracy"]
    assert fake.params["seed"] == "7"
    assert fake.params["dataset_digest"] == "digest-1"
    assert "source_commit" in fake.params
    assert ("requirements-frozen.txt", "reproducibility") in fake.artifacts
    assert ("summary.json", "reports") in fake.artifacts
    assert fake.experiment == dev_context().settings.experiment_name
