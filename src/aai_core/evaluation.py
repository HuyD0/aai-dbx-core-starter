"""MLflow GenAI evaluation suites and release gates."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aai_core.exceptions import AaiCoreError
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


class EvaluationGateError(AaiCoreError):
    code = "aai_core.evaluation.gate_failed"


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
                "Dataset support requires the `genai` extra. From an aai-core "
                "checkout run `make examples-install` and use `.venv/bin/python`; "
                "in a consuming environment install `aai-core[genai]`."
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

    def run_tracked(
        self,
        *,
        experiments: Any,
        run_name: str,
        data: Any,
        predict_fn: Any | None = None,
        baseline_metrics: Mapping[str, float] | None = None,
        prompt_uri: str | None = None,
        dataset_name: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> tuple[EvaluationReport, str | None]:
        """Run the evaluation as a governed MLflow run and return its id.

        The run carries the platform ``aai.*`` tags (via the passed
        :class:`~aai_core.experiments.ExperimentManager`), the evaluated
        prompt URI and dataset identity as params, the gate metrics, and an
        ``aai.gate_passed`` tag. Traces produced by ``predict_fn`` during
        the evaluation attach to this run, and passing a Unity Catalog
        dataset object as ``data`` links the dataset natively — evaluation
        runs become fully connected records instead of floating metric bags.
        """

        linked: dict[str, Any] = {"case_count": _case_count(data)}
        if prompt_uri:
            linked["prompt_uri"] = prompt_uri
        if dataset_name:
            linked["evaluation_dataset"] = dataset_name
        linked.update(parameters or {})

        run_id: str | None = None
        with experiments.run(run_name=run_name, parameters=linked) as active_run:
            run_id = getattr(getattr(active_run, "info", None), "run_id", None)
            report = self.run(
                data=data,
                predict_fn=predict_fn,
                baseline_metrics=baseline_metrics,
            )
            client = self._client()
            numeric = {
                name: value
                for name, value in report.metrics.items()
                if isinstance(value, (int, float))
            }
            if numeric:
                client.log_metrics(numeric)
            client.set_tags({"aai.gate_passed": str(report.passed).lower()})
        return report, run_id

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
                "Evaluation support requires the `genai` extra. From an aai-core "
                "checkout run `make examples-install` and use `.venv/bin/python`; "
                "in a consuming environment install `aai-core[genai]`."
            ) from error
        return mlflow


def apply_thresholds(
    metrics: Mapping[str, float],
    thresholds: Sequence[QualityThreshold],
    *,
    baseline_metrics: Mapping[str, float] | None = None,
) -> EvaluationReport:
    """Gate already-computed metrics without running an evaluation.

    Lets LLM-free projects (for example the experiment starter) reuse the
    same threshold/regression engine and `require_passed()` contract that
    :class:`EvaluationSuite` applies to GenAI evaluations.
    """

    numeric = {str(key): float(value) for key, value in metrics.items()}
    failures = _evaluate_thresholds(numeric, tuple(thresholds), baseline_metrics or {})
    return EvaluationReport(metrics=numeric, failures=tuple(failures))


def publish_report(
    report: EvaluationReport,
    *,
    title: str,
    baseline: Mapping[str, float] | None = None,
    run_link: str | None = None,
    summary_path: str | Path | None = None,
) -> str:
    """Render an evaluation report as a markdown table and publish it.

    Appended to ``summary_path`` (or ``$GITHUB_STEP_SUMMARY`` when set, so CI
    runs surface the verdict on the workflow summary page) and returned, for
    logging or PR comments. Metric keys are taken from the report — never
    hardcode aggregate-key formats.
    """

    failed = {failure.metric for failure in report.failures}
    lines = [
        f"## {title}",
        "",
        "| Metric | Value | Baseline | Delta | Verdict |",
        "|---|---|---|---|---|",
    ]
    for name, value in sorted(report.metrics.items()):
        reference = (baseline or {}).get(name)
        shown_reference = f"{reference:.3f}" if reference is not None else "—"
        delta = f"{value - reference:+.3f}" if reference is not None else "—"
        verdict = "FAIL" if name in failed else "ok"
        lines.append(
            f"| {name} | {value:.3f} | {shown_reference} | {delta} | {verdict} |"
        )
    if report.failures:
        lines.append("")
        lines.append("**Gate failures:**")
        lines.extend(
            f"- `{failure.metric}`: {failure.reason}" for failure in report.failures
        )
    lines.append("")
    lines.append(f"**Result: {'PASSED' if report.passed else 'FAILED'}**")
    if run_link:
        lines.append("")
        lines.append(f"[Evaluation run]({run_link})")
    markdown = "\n".join(lines) + "\n"

    target = summary_path or os.getenv("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as stream:
            stream.write(markdown + "\n")
    return markdown


def workspace_run_url(
    run_id: str | None, experiment_id: str | None = None
) -> str | None:
    """A clickable workspace URL for an MLflow run, when derivable.

    Uses ``DATABRICKS_HOST`` from the environment — the same variable every
    credentialed path already sets — so gate reports can deep-link the
    evaluation run without new configuration.
    """

    host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    if not host or not run_id:
        return None
    if experiment_id:
        return f"{host}/ml/experiments/{experiment_id}/runs/{run_id}"
    return f"{host}/#mlflow/runs/{run_id}"


def _case_count(data: Any) -> int | str:
    try:
        return len(data)
    except TypeError:
        return "unknown"


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
