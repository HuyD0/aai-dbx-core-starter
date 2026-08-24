"""Hermetic unit tests: fakes only, no cloud, no pandas."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_core.evaluation import GatePolicy, MetricDirection, MetricRule
from aai_core.testing import dev_context
from app.config import DATASET_NAME
from app.experiment import (
    DEFAULT_SEED,
    evaluate_rows,
    evaluation_records,
    load_csv_rows,
    rows_from_evaluation_records,
    run_experiment,
)

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

    def start_run(self, run_name=None, nested=False, description=None):
        class _Run:
            def __enter__(self):
                return SimpleNamespace(
                    info=SimpleNamespace(
                        run_name=run_name,
                        description=description,
                    )
                )

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
    assert fake.params["dataset_name"] == DATASET_NAME
    assert "source_commit" in fake.params
    assert ("requirements-frozen.txt", "reproducibility") in fake.artifacts
    assert ("summary.json", "reports") in fake.artifacts
    assert fake.experiment == dev_context().settings.effective_experiment_name


def test_credentialed_gate_records_policy_and_baseline_digests():
    fake = FakeMlflow()
    policy = GatePolicy(
        rules=(
            MetricRule(
                metric="accuracy",
                direction=MetricDirection.HIGHER,
                required=0.9,
            ),
        )
    )

    run_experiment(
        dev_context(),
        data=ROWS,
        mlflow_module=fake,
        gate_policy=policy,
        baseline_metrics={"accuracy": 0.95},
    )

    assert fake.params["gate_policy_digest"] == policy.digest
    assert len(fake.params["gate_baseline_digest"]) == 64
    assert fake.tags["aai.gate_passed"] == "true"


def test_evaluation_dataset_record_conversion_is_lossless():
    records = evaluation_records(ROWS)

    assert rows_from_evaluation_records(records) == ROWS


def test_governed_dataset_is_logged_without_building_a_local_mlflow_dataset():
    fake = FakeMlflow()
    fake.data.from_pandas = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("the governed dataset must be used directly")
    )
    dataset = SimpleNamespace(
        dataset_id="dataset-1",
        digest="digest-governed",
        name="catalog.schema.evaluation_cases",
    )

    run_experiment(
        dev_context(),
        data=ROWS,
        mlflow_module=fake,
        evaluation_dataset=dataset,
        evaluation_dataset_name="catalog.schema.evaluation_cases",
    )

    assert fake.inputs == [dataset]
    assert fake.params["dataset_name"] == "catalog.schema.evaluation_cases"
    assert fake.params["dataset_id"] == "dataset-1"
    assert fake.params["dataset_digest"] == "digest-governed"


def test_dataset_contract_rejects_duplicates_extras_nonfinite_and_empty():
    duplicate = [*ROWS, dict(ROWS[0])]
    with pytest.raises(ValueError, match="duplicate example_id"):
        evaluate_rows(duplicate, seed=DEFAULT_SEED)

    extra = [dict(ROWS[0], unexpected="field")]
    with pytest.raises(ValidationError):
        evaluate_rows(extra, seed=DEFAULT_SEED)

    nonfinite = [dict(ROWS[0], feature_signal=float("nan"))]
    with pytest.raises(ValidationError):
        evaluate_rows(nonfinite, seed=DEFAULT_SEED)

    with pytest.raises(ValueError, match="at least one"):
        evaluate_rows([], seed=DEFAULT_SEED)


def test_evaluation_records_forbid_missing_and_extra_boundaries():
    with pytest.raises(ValidationError):
        rows_from_evaluation_records(
            [
                {
                    "inputs": {"example_id": 1, "feature_signal": 1.0},
                    "expectations": {"label": 0},
                    "unexpected": "field",
                }
            ]
        )


def test_csv_loader_explicitly_parses_scalars_before_strict_validation(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(
        "example_id,feature_signal,label\n1,12.5,0\n2,87.0,1\n",
        encoding="utf-8",
    )

    assert load_csv_rows(path) == [
        {"example_id": 1, "feature_signal": 12.5, "label": 0},
        {"example_id": 2, "feature_signal": 87.0, "label": 1},
    ]

    path.write_text(
        "example_id,feature_signal,label\n1,not-a-number,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CSV row 2"):
        load_csv_rows(path)
