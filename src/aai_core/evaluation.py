"""Small release-gate contracts and thin helpers over native MLflow GenAI
evaluation results.

The contracts (:class:`GatePolicy`, :class:`GateResult`) apply deterministic
policy to a native result without wrapping or mutating it. The helpers stay
equally thin: they resolve the approved judge endpoint, compose the native
``mlflow.genai.evaluate()`` call with the gate, persist gate evidence on the
active run, and manage governed evaluation datasets. None of them owns an
MLflow run or mirrors native parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from math import isfinite
from numbers import Real
from re import fullmatch
from typing import Any, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.exceptions import AaiCoreError
from aai_core.providers.types import ProviderConfigurationError


class MetricDirection(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


class MetricRule(ContractModel):
    """One absolute and/or regression rule for a native MLflow metric."""

    metric: str = Field(min_length=1)
    direction: MetricDirection
    required: float | None = None
    max_regression: float | None = Field(default=None, ge=0.0)

    @field_validator("direction", mode="before")
    @classmethod
    def parse_direction(cls, value: Any) -> MetricDirection:
        if isinstance(value, MetricDirection):
            return value
        if not isinstance(value, str):
            raise TypeError("direction must be a string or MetricDirection")
        return MetricDirection(value.strip().lower())

    @field_validator("required", "max_regression")
    @classmethod
    def require_finite_number(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("metric rule values must be finite")
        return value

    @model_validator(mode="after")
    def require_a_constraint(self) -> Self:
        if self.required is None and self.max_regression is None:
            raise ValueError("A metric rule requires required or max_regression")
        return self


class GatePolicy(ContractModel):
    """Persistable policy applied after ``mlflow.genai.evaluate()``."""

    rules: tuple[MetricRule, ...] = ()
    minimum_cost_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    cost_coverage_metric: str = Field(default="cost/coverage", min_length=1)
    fail_on_scorer_errors: bool = True
    scorer_error_metric_suffix: str = Field(default="/error_count", min_length=1)
    allow_missing_regression_baseline: bool = False


class GateFailure(ContractModel):
    metric: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class GateResult(ContractModel):
    """Immutable release-gate evidence; native evaluation results stay native.

    ``policy`` and ``baseline_metrics`` record exactly which rules and which
    comparison baseline produced this result, and a policy-bearing result
    re-evaluates itself at construction: the stored failures must be exactly
    what the recorded policy yields for the recorded metrics, so a
    hand-built or deserialized result cannot claim ``passed`` while its own
    metrics violate its own policy.
    """

    metrics: Mapping[str, float]
    failures: tuple[GateFailure, ...] = ()
    policy: GatePolicy | None = None
    baseline_metrics: Mapping[str, float] | None = None

    @field_validator("metrics", "baseline_metrics", mode="after")
    @classmethod
    def freeze_metrics(
        cls, value: Mapping[str, float] | None
    ) -> Mapping[str, float] | None:
        if value is None:
            return None
        for name, metric in value.items():
            # NaN compares false against every threshold, so a non-finite
            # value would silently pass policy recomputation.
            if not isfinite(metric):
                raise ValueError(
                    f"gate evidence metric {name!r} must be a finite number"
                )
        return freeze_value(value)

    @field_serializer("metrics", "baseline_metrics")
    def serialize_metrics(
        self, value: Mapping[str, float] | None
    ) -> dict[str, float] | None:
        return None if value is None else thaw_value(value)

    @model_validator(mode="after")
    def failures_match_the_recorded_policy(self) -> Self:
        if self.policy is None:
            return self
        recomputed = _evaluate_policy(
            dict(self.metrics),
            self.policy,
            dict(self.baseline_metrics or {}),
        )
        if tuple(self.failures) != recomputed:
            raise ValueError(
                "failures do not match what the recorded policy yields for "
                "the recorded metrics; produce gate evidence with "
                "apply_gate()"
            )
        return self

    @property
    def passed(self) -> bool:
        return not self.failures

    def require_passed(self) -> None:
        if not self.failures:
            return
        raise EvaluationGateError(
            "; ".join(
                f"{failure.metric}: {failure.reason}" for failure in self.failures
            )
        )


class EvaluationGateError(AaiCoreError):
    code = "aai_core.evaluation.gate_failed"


def apply_gate(
    evaluation_result: Any,
    *,
    policy: GatePolicy,
    baseline_metrics: Mapping[str, float] | None = None,
) -> GateResult:
    """Apply deterministic policy to a native MLflow result or metric mapping.

    A scorer error-count metric that is not a finite number is corrupt
    scorer-health evidence: dropping it like other malformed metrics would
    let the gate pass with scorer health unknown, and gate evidence cannot
    carry non-finite values. While ``fail_on_scorer_errors`` is enforced,
    such a result is refused outright instead of becoming evidence.

    Native ``mlflow.genai.evaluate()`` results report scorer failures per
    row (``<scorer>/error_message`` columns), not as aggregated metrics, so
    those rows are counted here and persisted as ``<scorer>/error_count``
    gate evidence — a crashing scorer fails the gate even though the native
    metric mapping never mentions it.
    """

    metrics = _extract_metrics(evaluation_result)
    if policy.fail_on_scorer_errors:
        for name in _metric_source(evaluation_result):
            key = str(name)
            if key.endswith(policy.scorer_error_metric_suffix) and key not in metrics:
                raise ValueError(
                    f"scorer error metric {key!r} is not a finite number; "
                    "refusing to produce gate evidence with scorer health "
                    "unknown"
                )
        for name, count in _row_level_error_counts(
            evaluation_result, policy.scorer_error_metric_suffix
        ).items():
            # Observed failing rows are direct evidence. An aggregate that
            # contradicts them — 0 in the mapping while error_message rows
            # exist — must not erase the failure, so keep whichever count
            # is larger rather than always trusting the mapping.
            metrics[name] = max(metrics.get(name, 0.0), count)
    baseline = dict(baseline_metrics or {})
    return GateResult(
        metrics=metrics,
        failures=_evaluate_policy(metrics, policy, baseline),
        policy=policy,
        baseline_metrics=(
            dict(baseline_metrics) if baseline_metrics is not None else None
        ),
    )


def _evaluate_policy(
    metrics: dict[str, float],
    policy: GatePolicy,
    baseline: dict[str, float],
) -> tuple[GateFailure, ...]:
    failures: list[GateFailure] = []

    if policy.fail_on_scorer_errors:
        for metric, value in metrics.items():
            if not metric.endswith(policy.scorer_error_metric_suffix):
                continue
            if value > 0:
                failures.append(
                    GateFailure(
                        metric=metric,
                        reason=f"{value:g} scorer invocation(s) failed",
                    )
                )
            elif value < 0:
                # A count cannot be negative; this runs inside the
                # recomputation, so even a hand-built result cannot claim a
                # pass over corrupt scorer-health evidence.
                failures.append(
                    GateFailure(
                        metric=metric,
                        reason=(
                            f"error count {value:g} is negative; scorer "
                            "health evidence is corrupt"
                        ),
                    )
                )

    if policy.minimum_cost_coverage is not None:
        observed = metrics.get(policy.cost_coverage_metric)
        if observed is None:
            failures.append(
                GateFailure(
                    metric=policy.cost_coverage_metric,
                    reason="cost coverage is unknown",
                )
            )
        elif not 0.0 <= observed <= 1.0:
            # Coverage is a fraction by definition (the policy bounds its
            # threshold to [0, 1]); an impossible observed value would
            # otherwise satisfy any threshold. This runs inside the
            # recomputation, so a hand-built result cannot claim a pass
            # over corrupt coverage evidence.
            failures.append(
                GateFailure(
                    metric=policy.cost_coverage_metric,
                    reason=(
                        f"coverage {observed:g} is outside the unit "
                        "interval; cost-coverage evidence is corrupt"
                    ),
                )
            )
        elif observed < policy.minimum_cost_coverage:
            failures.append(
                GateFailure(
                    metric=policy.cost_coverage_metric,
                    reason=(
                        f"{observed:g} is below required "
                        f"{policy.minimum_cost_coverage:g}"
                    ),
                )
            )

    for rule in policy.rules:
        observed = metrics.get(rule.metric)
        if observed is None:
            failures.append(GateFailure(metric=rule.metric, reason="metric is missing"))
            continue
        if rule.required is not None:
            below = (
                rule.direction is MetricDirection.HIGHER and observed < rule.required
            )
            above = rule.direction is MetricDirection.LOWER and observed > rule.required
            if below or above:
                comparison = "below" if below else "above"
                failures.append(
                    GateFailure(
                        metric=rule.metric,
                        reason=(
                            f"{observed:g} is {comparison} required "
                            f"{rule.required:g}"
                        ),
                    )
                )
        if rule.max_regression is None:
            continue
        reference = baseline.get(rule.metric)
        if reference is None:
            if not policy.allow_missing_regression_baseline:
                failures.append(
                    GateFailure(
                        metric=rule.metric,
                        reason="regression baseline is missing",
                    )
                )
            continue
        regression = (
            reference - observed
            if rule.direction is MetricDirection.HIGHER
            else observed - reference
        )
        if regression > rule.max_regression:
            failures.append(
                GateFailure(
                    metric=rule.metric,
                    reason=(
                        f"regressed by {regression:g} from baseline {reference:g}; "
                        f"maximum allowed is {rule.max_regression:g}"
                    ),
                )
            )

    return tuple(failures)


def _row_level_error_counts(result: Any, suffix: str) -> dict[str, float]:
    """Count per-row failures a native result never aggregates.

    ``mlflow.genai.evaluate()`` records a failed scorer invocation as a
    non-null ``<scorer>/error_message`` cell in ``result_df``, and a failed
    ``predict_fn`` invocation in the bare ``error_message`` column; neither
    reaches the ``metrics`` mapping. Synthesizing ``*/error_count``
    evidence here is what lets the gate see scorer and application health
    at all.
    """

    frame = getattr(result, "result_df", None)
    columns = getattr(frame, "columns", None)
    if columns is None:
        return {}
    counts: dict[str, float] = {}
    for column in columns:
        name = str(column)
        if name == "error_message":
            errored = int(frame[column].notna().sum())
            if errored:
                counts[f"predict_fn{suffix}"] = float(errored)
            continue
        if not name.endswith("/error_message"):
            continue
        errored = int(frame[column].notna().sum())
        if errored:
            scorer = name.removesuffix("/error_message")
            counts[f"{scorer}{suffix}"] = float(errored)
    return counts


def _metric_source(result: Any) -> Mapping[str, Any]:
    source = result if isinstance(result, Mapping) else getattr(result, "metrics", None)
    if not isinstance(source, Mapping):
        raise TypeError(
            "evaluation_result must be a metric mapping or expose a metrics mapping"
        )
    return source


def _extract_metrics(result: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in _metric_source(result).items():
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        numeric = float(value)
        if isfinite(numeric):
            metrics[str(name)] = numeric
    return metrics


def judge_model_uri(settings: Any, logical_name: str = "judge-model") -> str:
    """Resolve an approved logical judge into MLflow's model URI.

    Judges run through a Databricks serving endpoint so authentication,
    gateway policy, and cost controls stay platform-owned. A Foundry model
    must first be exposed through a governed Databricks external-model
    endpoint.
    """

    models = getattr(settings, "models", {})
    config = models.get(logical_name) if isinstance(models, Mapping) else None
    if not isinstance(config, Mapping):
        raise ProviderConfigurationError(
            f"aai-platform.yml has no {logical_name!r} model entry",
            remediation=f"Add providers.models.{logical_name} with the "
            "gateway-fronted Databricks serving endpoint approved for judges.",
        )
    if config.get("provider") != "databricks":
        raise ProviderConfigurationError(
            f"LLM judge {logical_name!r} must use provider 'databricks'",
            remediation="Route the judge through a Databricks serving "
            "endpoint; for Foundry models, use an external-model endpoint.",
        )
    deployment = config.get("deployment")
    if not isinstance(deployment, str) or not deployment.strip():
        raise ProviderConfigurationError(
            f"LLM judge {logical_name!r} has no deployment",
            remediation=f"Set providers.models.{logical_name}.deployment to "
            "the approved serving endpoint name.",
        )
    if _is_placeholder(deployment):
        raise ProviderConfigurationError(
            f"LLM judge {logical_name!r} deployment {deployment.strip()!r} "
            "is a setup placeholder",
            remediation=f"Replace providers.models.{logical_name}.deployment "
            "with the approved serving endpoint before running evaluation.",
        )
    # Databricks serving endpoint names contain only alphanumerics, dashes,
    # and underscores; anything else builds an endpoints:/ URI that fails
    # only inside the later evaluation request, after the doctor has
    # already reported the judge ready.
    if not fullmatch(_NAME_COMPONENT, deployment.strip()):
        raise ProviderConfigurationError(
            f"LLM judge {logical_name!r} deployment {deployment.strip()!r} "
            "is not a valid serving endpoint name",
            remediation="Serving endpoint names contain only alphanumeric "
            f"characters, dashes, and underscores; set providers.models."
            f"{logical_name}.deployment to the endpoint's exact name, not "
            "a URI or display label.",
        )
    return f"endpoints:/{deployment.strip()}"


def log_gate_evidence(
    gate: GateResult,
    *,
    mlflow_module: Any | None = None,
) -> dict[str, str]:
    """Persist gate metrics and the ``aai.gate_passed`` tag on the active run.

    Call inside a governed run. Returns the tags it set.
    """

    mlflow = _mlflow(mlflow_module)
    if gate.metrics:
        mlflow.log_metrics(dict(gate.metrics))
    tags = {"aai.gate_passed": str(gate.passed).lower()}
    mlflow.set_tags(tags)
    return tags


def evaluate_with_gate(
    *,
    policy: GatePolicy,
    baseline_metrics: Mapping[str, float] | None = None,
    mlflow_module: Any | None = None,
    **evaluate_options: Any,
) -> tuple[Any, GateResult]:
    """Run native ``mlflow.genai.evaluate()`` and apply the gate policy.

    Every keyword in ``evaluate_options`` passes through to the native call
    untouched, and the native result is returned by identity, so new MLflow
    evaluation arguments never require an SDK change. Run governance stays
    with :class:`~aai_core.experiments.ExperimentManager`; persisting the
    evidence stays the explicit :func:`log_gate_evidence` call.
    """

    mlflow = _mlflow(mlflow_module)
    native_result = mlflow.genai.evaluate(**evaluate_options)
    gate = apply_gate(native_result, policy=policy, baseline_metrics=baseline_metrics)
    return native_result, gate


def get_or_create_evaluation_dataset(
    *,
    name: str,
    catalog: str,
    schema: str,
    experiment_id: str,
    records: Sequence[Mapping[str, Any]] | None = None,
    mlflow_module: Any | None = None,
) -> Any:
    """Return the governed evaluation dataset, creating and merging as needed.

    ``name`` is a logical dataset name; the catalog and schema qualify it.
    The native dataset object is returned unchanged.
    """

    # Checked before .strip(): a non-string would otherwise raise an
    # incidental AttributeError instead of the documented contract error,
    # and an object implementing strip() could return a valid-looking name
    # and reach the registry.
    if not isinstance(name, str):
        raise TypeError(
            f"name must be a string logical dataset name; got {type(name).__name__}"
        )
    logical_name = name.strip()
    if not fullmatch(_NAME_COMPONENT, logical_name) or _is_placeholder(logical_name):
        raise ValueError(
            "name must be a configured logical name without catalog or "
            "schema (letters, digits, underscores, and hyphens); got "
            f"{name!r}"
        )
    # Normalize before the first request: the raw value is both sent to
    # create_dataset() and compared against backend ids, which come back
    # normalized, so untrimmed input would fail in the cloud or falsely
    # report the dataset as associated with the wrong experiment.
    # str() would turn None into the plausible-looking id "None"; require
    # the real type before normalizing.
    if not isinstance(experiment_id, str):
        raise TypeError(
            "experiment_id must be a string MLflow experiment id; got "
            f"{type(experiment_id).__name__}"
        )
    experiment_id = experiment_id.strip()
    if not experiment_id or _is_placeholder(experiment_id):
        raise ValueError(
            "experiment_id must be the real MLflow experiment id the dataset "
            f"belongs to, not a setup placeholder; got {experiment_id!r}"
        )
    catalog = _dataset_qualifier("catalog", catalog)
    schema = _dataset_qualifier("schema", schema)

    mlflow = _mlflow(mlflow_module)
    qualified_name = f"{catalog}.{schema}.{logical_name}"
    try:
        dataset = mlflow.genai.datasets.get_dataset(name=qualified_name)
    except Exception as exc:
        if not _is_missing_dataset(exc):
            raise
        # Databricks-managed EvaluationDatasets reject MLflow dataset tags.
        # Governed context belongs on runs and UC securables instead.
        dataset = mlflow.genai.datasets.create_dataset(
            name=qualified_name,
            experiment_id=experiment_id,
        )

    experiment_ids = {
        str(associated).strip() for associated in (dataset.experiment_ids or [])
    }
    if experiment_id not in experiment_ids:
        raise RuntimeError(
            f"Unity Catalog dataset {qualified_name!r} is not associated with "
            f"MLflow experiment {experiment_id!r}. Databricks does not "
            "support adding experiment associations through this API; use a "
            "new approved dataset name or ask the platform owner to repair it."
        )
    if records:
        dataset.merge_records(list(records))
    return dataset


_QUALIFIER_PLACEHOLDERS = {"unset", "unknown", "todo", "changeme"}

# The identifier shape every registry component must satisfy — shared by
# the dataset helper, the prompt manager, and the doctor so no surface
# accepts a name another surface refuses.
_NAME_COMPONENT = r"[A-Za-z0-9_-]+"


def _is_placeholder(value: str) -> bool:
    """Recognize the setup-placeholder vocabulary shared across the repo:
    the unconfigured markers, the ``replace-with-*`` values that
    ``aai-platform.example.yml`` ships, and documentation-style
    ``<angle-bracket>`` markers (the same set the examples runner
    recognizes)."""

    lowered = str(value).strip().lower()
    return (
        lowered in _QUALIFIER_PLACEHOLDERS
        or lowered.startswith("replace-with-")
        or "<" in lowered
        or ">" in lowered
    )


def _is_placeholder_path(value: str) -> bool:
    """Placeholder test for slash-separated paths such as experiment names.

    ``_is_placeholder`` matches the bare markers exactly and anchors
    ``replace-with-`` at the start, so a placeholder sitting inside a path
    (``/Shared/replace-with-experiment``, ``/Shared/unset``) slips past it
    while looking configured. Experiment names are the one governed value
    that arrives as a path, so they are tested component by component.
    """

    return _is_placeholder(value) or any(
        _is_placeholder(component) for component in str(value).split("/") if component
    )


def _dataset_qualifier(role: str, value: str) -> str:
    """Fail locally on unconfigured qualifiers instead of querying the cloud."""

    # str() would make None the qualifier "None" and 123 the qualifier
    # "123", both of which satisfy _NAME_COMPONENT and would name a real
    # (wrong) securable in the registry.
    if not isinstance(value, str):
        raise TypeError(
            f"{role} must be a string Unity Catalog qualifier; got "
            f"{type(value).__name__}"
        )
    qualifier = value.strip()
    if not fullmatch(_NAME_COMPONENT, qualifier) or _is_placeholder(qualifier):
        raise ValueError(
            f"{role} must be a configured Unity Catalog qualifier; got "
            f"{value!r}. Set platform.catalog and platform.schema in "
            "aai-platform.yml before using governed evaluation datasets."
        )
    return qualifier


# Structured codes that are authoritatively NOT absence: registries often
# word a permission denial as "does not exist" to avoid disclosing
# inaccessible resources, so these codes override message markers.
_NON_MISSING_ERROR_CODES = {
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
    "UNAUTHORIZED",
    "TEMPORARILY_UNAVAILABLE",
    "REQUEST_LIMIT_EXCEEDED",
}


def _is_missing_registry_error(error: Exception) -> bool:
    """Shared absence test for registry errors (datasets and prompts alike).

    Message markers are consulted only when no authoritative structured
    code says otherwise; ``MlflowException`` defaults to ``INTERNAL_ERROR``
    on message-only raises, so markers must survive non-authoritative
    codes.
    """

    error_code = str(getattr(error, "error_code", "")).strip().upper()
    if error_code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}:
        return True
    if error_code in _NON_MISSING_ERROR_CODES:
        return False
    message = str(error).upper()
    if any(
        marker in message
        for marker in ("NOT_FOUND", "RESOURCE_DOES_NOT_EXIST", "DOES NOT EXIST")
    ):
        return True
    # The file and SQL registries report a missing alias as
    # INVALID_PARAMETER_VALUE with "Registered model alias ... not found."
    # — recognized narrowly so unrelated parameter errors stay errors.
    return "ALIAS" in message and "NOT FOUND" in message


def _is_missing_dataset(error: Exception) -> bool:
    return _is_missing_registry_error(error)


def _mlflow(module: Any | None) -> Any:
    if module is not None:
        return module
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "Evaluation support requires the `genai` extra. From an aai-core "
            "checkout run `make examples-install` and use `.venv/bin/python`; "
            "in a consuming environment install `aai-core[genai]`."
        ) from error
    return mlflow
