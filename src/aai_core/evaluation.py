"""MLflow GenAI evaluation suites and release gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from aai_core.tags import ResourceContext


@dataclass(frozen=True)
class QualityThreshold:
    metric: str
    direction: Literal["higher", "lower"]
    required: float | None = None
    max_regression: float | None = None


@dataclass(frozen=True)
class GateFailure:
    metric: str
    reason: str


@dataclass(frozen=True)
class EvaluationReport:
    metrics: Mapping[str, float]
    failures: tuple[GateFailure, ...] = ()
    raw: Any = field(default=None, repr=False, compare=False)

    @property
    def passed(self) -> bool:
        return not self.failures

    def require_passed(self) -> None:
        if self.failures:
            messages = "; ".join(
                f"{failure.metric}: {failure.reason}" for failure in self.failures
            )
            raise EvaluationGateError(messages)


class EvaluationGateError(RuntimeError):
    pass


class EvaluationDatasetManager:
    """Create and tag MLflow evaluation datasets from cases or traces."""

    def __init__(
        self,
        *,
        context: ResourceContext,
        experiment_id: str | None = None,
        mlflow_module: Any | None = None,
    ) -> None:
        self.context = context
        self.experiment_id = experiment_id
        self._mlflow = mlflow_module

    def create(
        self,
        name: str,
        *,
        records: Any | None = None,
        tags: Mapping[str, str] | None = None,
    ):
        metadata = self.context.merged(tags)
        dataset = self._client().genai.datasets.create_dataset(
            name=name,
            experiment_id=self.experiment_id,
            tags={f"aai.{key}": value for key, value in metadata.items()},
        )
        if records is not None:
            dataset = dataset.merge_records(records)
        return dataset

    def get(self, *, name: str | None = None, dataset_id: str | None = None):
        if bool(name) == bool(dataset_id):
            raise ValueError("Specify exactly one of name or dataset_id")
        return self._client().genai.datasets.get_dataset(
            name=name,
            dataset_id=dataset_id,
        )

    def _client(self):
        if self._mlflow is not None:
            return self._mlflow
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError(
                "Dataset support requires `pip install 'aai-core[genai]'`"
            ) from error
        return mlflow


class EvaluationSuite:
    def __init__(
        self,
        *,
        scorers: Sequence[Any],
        thresholds: Sequence[QualityThreshold],
        mlflow_module: Any | None = None,
    ) -> None:
        self.scorers = tuple(scorers)
        self.thresholds = tuple(thresholds)
        self._mlflow = mlflow_module

    def run(
        self,
        *,
        data: Any,
        predict_fn: Any | None = None,
        baseline_metrics: Mapping[str, float] | None = None,
    ) -> EvaluationReport:
        kwargs = {"data": data, "scorers": list(self.scorers)}
        if predict_fn is not None:
            kwargs["predict_fn"] = predict_fn
        raw = self._client().genai.evaluate(**kwargs)
        metrics = _extract_metrics(raw)
        failures = _evaluate_thresholds(
            metrics, self.thresholds, baseline_metrics or {}
        )
        return EvaluationReport(metrics=metrics, failures=tuple(failures), raw=raw)

    def log_feedback(
        self,
        *,
        trace_id: str,
        name: str,
        value: Any,
        rationale: str | None = None,
    ) -> None:
        self._client().log_feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            rationale=rationale,
        )

    def log_expectation(
        self,
        *,
        trace_id: str,
        name: str,
        value: Any,
    ) -> None:
        self._client().log_expectation(
            trace_id=trace_id,
            name=name,
            value=value,
        )

    def _client(self):
        if self._mlflow is not None:
            return self._mlflow
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError(
                "Evaluation support requires `pip install 'aai-core[genai]'`"
            ) from error
        return mlflow


def _extract_metrics(result: Any) -> dict[str, float]:
    source = getattr(result, "metrics", result if isinstance(result, Mapping) else {})
    return {
        str(key): float(value)
        for key, value in source.items()
        if isinstance(value, (int, float))
    }


def _evaluate_thresholds(
    metrics: Mapping[str, float],
    thresholds: Sequence[QualityThreshold],
    baseline: Mapping[str, float],
) -> list[GateFailure]:
    failures = []
    for threshold in thresholds:
        if threshold.metric not in metrics:
            failures.append(GateFailure(threshold.metric, "metric is missing"))
            continue
        candidate = metrics[threshold.metric]
        if threshold.required is not None:
            violates = (
                candidate < threshold.required
                if threshold.direction == "higher"
                else candidate > threshold.required
            )
            if violates:
                failures.append(
                    GateFailure(
                        threshold.metric,
                        f"{candidate} violates required {threshold.required}",
                    )
                )
        if threshold.max_regression is not None and threshold.metric in baseline:
            reference = baseline[threshold.metric]
            regression = (
                reference - candidate
                if threshold.direction == "higher"
                else candidate - reference
            )
            if regression > threshold.max_regression:
                failures.append(
                    GateFailure(
                        threshold.metric,
                        f"regressed by {regression} from baseline {reference}",
                    )
                )
    return failures
