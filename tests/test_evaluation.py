from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_core.evaluation import (
    EvaluationGateError,
    GatePolicy,
    GateResult,
    MetricDirection,
    MetricRule,
    apply_gate,
    get_or_create_evaluation_dataset,
    judge_model_uri,
)
from aai_core.providers.types import ProviderConfigurationError


def test_gate_accepts_native_mlflow_result_and_absolute_rules():
    native_result = SimpleNamespace(
        metrics={"groundedness/mean": 0.92, "latency_ms/mean": 240}
    )
    policy = GatePolicy(
        rules=(
            MetricRule(
                metric="groundedness/mean",
                direction=MetricDirection.HIGHER,
                required=0.8,
            ),
            MetricRule(
                metric="latency_ms/mean",
                direction=MetricDirection.LOWER,
                required=500,
            ),
        )
    )

    result = apply_gate(native_result, policy=policy)

    assert result.passed
    assert result.metrics["groundedness/mean"] == 0.92


def test_gate_reports_absolute_and_missing_metric_failures():
    policy = GatePolicy(
        rules=(
            MetricRule(
                metric="groundedness/mean",
                direction=MetricDirection.HIGHER,
                required=0.8,
            ),
            MetricRule(
                metric="safety/mean",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
        )
    )

    result = apply_gate({"groundedness/mean": 0.7}, policy=policy)

    assert [failure.metric for failure in result.failures] == [
        "groundedness/mean",
        "safety/mean",
    ]
    with pytest.raises(EvaluationGateError, match="groundedness/mean"):
        result.require_passed()


def test_regression_requires_baseline_unless_bootstrap_is_explicit():
    rule = MetricRule(
        metric="quality/mean",
        direction=MetricDirection.HIGHER,
        max_regression=0.02,
    )

    missing = apply_gate(
        {"quality/mean": 0.95},
        policy=GatePolicy(rules=(rule,)),
    )
    bootstrap = apply_gate(
        {"quality/mean": 0.95},
        policy=GatePolicy(
            rules=(rule,),
            allow_missing_regression_baseline=True,
        ),
    )

    assert not missing.passed
    assert missing.failures[0].reason == "regression baseline is missing"
    assert bootstrap.passed


@pytest.mark.parametrize(
    ("direction", "observed", "baseline"),
    [
        (MetricDirection.HIGHER, 0.94, 0.99),
        (MetricDirection.LOWER, 120.0, 80.0),
    ],
)
def test_gate_detects_regression_in_both_directions(direction, observed, baseline):
    policy = GatePolicy(
        rules=(
            MetricRule(
                metric="score",
                direction=direction,
                max_regression=0.02,
            ),
        )
    )

    result = apply_gate(
        {"score": observed},
        policy=policy,
        baseline_metrics={"score": baseline},
    )

    assert not result.passed
    assert "regressed" in result.failures[0].reason


def test_unknown_or_incomplete_cost_evidence_never_becomes_zero():
    policy = GatePolicy(minimum_cost_coverage=1.0)

    unknown = apply_gate({}, policy=policy)
    incomplete = apply_gate({"cost/coverage": 0.5}, policy=policy)
    complete_zero = apply_gate(
        {"cost/coverage": 1.0, "cost/total_usd": 0.0},
        policy=policy,
    )

    assert not unknown.passed
    assert unknown.failures[0].reason == "cost coverage is unknown"
    assert not incomplete.passed
    assert complete_zero.passed


def test_scorer_error_metrics_fail_without_exposing_error_text():
    result = apply_gate(
        {
            "correctness/mean": 1.0,
            "correctness/error_count": 2,
        },
        policy=GatePolicy(),
    )

    assert not result.passed
    assert result.failures[0].reason == "2 scorer invocation(s) failed"


def test_gate_fails_negative_scorer_error_counts_as_corrupt():
    result = apply_gate({"correctness/error_count": -1}, policy=GatePolicy())

    assert not result.passed
    assert "negative" in result.failures[0].reason
    # The check runs inside the recomputation, so a hand-built result
    # cannot claim a pass over a negative count either.
    with pytest.raises(ValidationError, match="apply_gate"):
        GateResult(
            metrics={"correctness/error_count": -1.0},
            failures=(),
            policy=GatePolicy(),
        )


def test_gate_refuses_non_finite_scorer_error_counts():
    # A NaN error count means scorer health is unknown; dropping it like
    # other malformed metrics would let the gate pass anyway.
    metrics = {"quality": 1.0, "correctness/error_count": float("nan")}

    with pytest.raises(ValueError, match="correctness/error_count"):
        apply_gate(metrics, policy=GatePolicy())

    # Opting out of scorer-error enforcement restores plain dropping.
    result = apply_gate(metrics, policy=GatePolicy(fail_on_scorer_errors=False))
    assert dict(result.metrics) == {"quality": 1.0}
    assert result.passed


def test_non_numeric_and_non_finite_native_metrics_are_not_gate_evidence():
    result = apply_gate(
        {
            "quality": float("nan"),
            "label": "good",
            "flag": True,
            "latency": 42,
        },
        policy=GatePolicy(
            rules=(
                MetricRule(
                    metric="quality",
                    direction=MetricDirection.HIGHER,
                    required=0.8,
                ),
            )
        ),
    )

    assert dict(result.metrics) == {"latency": 42.0}
    assert result.failures[0].reason == "metric is missing"


def test_gate_contracts_are_strict_frozen_and_serializable():
    with pytest.raises(ValidationError):
        GatePolicy(minimum_cost_coverage="1.0")
    with pytest.raises(ValidationError):
        GatePolicy(unrecognized=True)
    with pytest.raises(ValidationError):
        MetricRule(metric="quality", direction=MetricDirection.HIGHER)

    policy = GatePolicy(
        rules=(
            MetricRule(
                metric="quality",
                direction=MetricDirection.HIGHER,
                required=0.8,
            ),
        )
    )
    result = apply_gate({"quality": 0.9}, policy=policy)

    with pytest.raises(ValidationError):
        policy.minimum_cost_coverage = 0.5
    assert result.policy == policy
    assert result.model_dump(mode="json") == {
        "metrics": {"quality": 0.9},
        "failures": [],
        "policy": policy.model_dump(mode="json"),
        "baseline_metrics": None,
    }


def test_gate_result_refuses_failures_inconsistent_with_its_policy():
    policy = GatePolicy(
        rules=(
            MetricRule(
                metric="quality",
                direction=MetricDirection.HIGHER,
                required=0.8,
            ),
        )
    )

    with pytest.raises(ValidationError, match="recorded policy"):
        GateResult(metrics={"quality": 0.0}, failures=(), policy=policy)
    with pytest.raises(ValidationError, match="recorded policy"):
        GateResult(metrics={"irrelevant": 1.0}, failures=(), policy=policy)
    # Policy-less results remain constructible for fakes and refusal demos.
    assert GateResult(metrics={"quality": 0.0}).passed


def test_gate_result_refuses_non_finite_evidence_values():
    with pytest.raises(ValidationError, match="finite"):
        GateResult(metrics={"quality": float("nan")})
    with pytest.raises(ValidationError, match="finite"):
        GateResult(metrics={"quality": float("inf")})
    with pytest.raises(ValidationError, match="finite"):
        GateResult(
            metrics={"quality": 0.9},
            baseline_metrics={"quality": float("nan")},
        )


def test_gate_result_with_regression_rule_round_trips_with_its_baseline():
    policy = GatePolicy(
        rules=(
            MetricRule(
                metric="score",
                direction=MetricDirection.HIGHER,
                max_regression=0.02,
            ),
        )
    )
    result = apply_gate(
        {"score": 0.94},
        policy=policy,
        baseline_metrics={"score": 0.95},
    )
    assert result.passed
    assert dict(result.baseline_metrics) == {"score": 0.95}

    rebuilt = GateResult(**result.model_dump())

    assert rebuilt == result
    assert rebuilt.passed


def test_gate_rejects_values_without_a_native_metrics_mapping():
    with pytest.raises(TypeError, match="metrics mapping"):
        apply_gate(object(), policy=GatePolicy())


def test_judge_resolves_only_an_approved_databricks_deployment():
    settings = SimpleNamespace(
        models={"judge-model": {"provider": "databricks", "deployment": "judge-ep"}}
    )

    assert judge_model_uri(settings) == "endpoints:/judge-ep"


@pytest.mark.parametrize(
    ("models", "match"),
    [
        ({}, "no 'judge-model' model entry"),
        ({"judge-model": {"provider": "foundry", "deployment": "x"}}, "databricks"),
        ({"judge-model": {"provider": "databricks", "deployment": " "}}, "deployment"),
        (
            {
                "judge-model": {
                    "provider": "databricks",
                    "deployment": "replace-with-serving-endpoint",
                }
            },
            "placeholder",
        ),
        (
            {"judge-model": {"provider": "databricks", "deployment": "unset"}},
            "placeholder",
        ),
    ],
)
def test_judge_refuses_missing_or_ungoverned_configuration(models, match):
    with pytest.raises(ProviderConfigurationError, match=match):
        judge_model_uri(SimpleNamespace(models=models))


class FakeEvaluationDataset:
    def __init__(self, *, name, experiment_ids):
        self.name = name
        self.dataset_id = "dataset-1"
        self.experiment_ids = experiment_ids
        self.merged_records = None

    def merge_records(self, records):
        self.merged_records = records
        return self


class FakeDatasetApi:
    def __init__(self, *, existing=None, get_error=None):
        self.existing = existing
        self.get_error = get_error
        self.create_arguments = None

    def get_dataset(self, **kwargs):
        if self.get_error is not None:
            raise self.get_error
        return self.existing

    def create_dataset(self, **kwargs):
        self.create_arguments = kwargs
        self.existing = FakeEvaluationDataset(
            name=kwargs["name"],
            experiment_ids=[kwargs["experiment_id"]],
        )
        return self.existing


def _dataset_mlflow(datasets):
    return SimpleNamespace(genai=SimpleNamespace(datasets=datasets))


def test_dataset_helper_reuses_and_merges_the_existing_dataset():
    dataset = FakeEvaluationDataset(
        name="main.default.regression_v1",
        experiment_ids=["experiment-1"],
    )
    datasets = FakeDatasetApi(existing=dataset)
    records = [{"inputs": {"question": "Synthetic question"}}]

    result = get_or_create_evaluation_dataset(
        name="regression_v1",
        catalog="main",
        schema="default",
        experiment_id="experiment-1",
        records=records,
        mlflow_module=_dataset_mlflow(datasets),
    )

    assert result is dataset
    assert result.merged_records == records
    assert datasets.create_arguments is None


def test_dataset_helper_creates_without_unsupported_tags():
    error = RuntimeError("RESOURCE_DOES_NOT_EXIST: dataset not found")
    datasets = FakeDatasetApi(get_error=error)

    result = get_or_create_evaluation_dataset(
        name="regression_v1",
        catalog="main",
        schema="default",
        experiment_id="experiment-1",
        mlflow_module=_dataset_mlflow(datasets),
    )

    assert result.dataset_id == "dataset-1"
    assert datasets.create_arguments == {
        "name": "main.default.regression_v1",
        "experiment_id": "experiment-1",
    }
    assert "tags" not in datasets.create_arguments
    assert result.merged_records is None


def test_dataset_helper_rejects_wrong_experiment_association():
    dataset = FakeEvaluationDataset(
        name="main.default.regression_v1",
        experiment_ids=["some-other-experiment"],
    )

    with pytest.raises(RuntimeError, match="not associated"):
        get_or_create_evaluation_dataset(
            name="regression_v1",
            catalog="main",
            schema="default",
            experiment_id="experiment-1",
            mlflow_module=_dataset_mlflow(FakeDatasetApi(existing=dataset)),
        )


def test_dataset_helper_requires_a_logical_unqualified_name():
    with pytest.raises(ValueError, match="logical name"):
        get_or_create_evaluation_dataset(
            name="main.default.regression_v1",
            catalog="main",
            schema="default",
            experiment_id="experiment-1",
            mlflow_module=_dataset_mlflow(FakeDatasetApi()),
        )


@pytest.mark.parametrize(
    ("catalog", "schema"),
    [
        ("unset", "default"),
        ("main", "unset"),
        (" ", "default"),
        ("main", ""),
        ("ChangeMe", "default"),
        ("main", "todo"),
        ("replace-with-catalog", "default"),
    ],
)
def test_dataset_helper_fails_locally_on_placeholder_qualifiers(catalog, schema):
    class RecordingApi(FakeDatasetApi):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def get_dataset(self, **kwargs):
            self.calls += 1
            return super().get_dataset(**kwargs)

    registry = RecordingApi()

    with pytest.raises(ValueError, match="platform.catalog"):
        get_or_create_evaluation_dataset(
            name="regression_v1",
            catalog=catalog,
            schema=schema,
            experiment_id="experiment-1",
            mlflow_module=_dataset_mlflow(registry),
        )

    assert registry.calls == 0
