"""The scoring engine behind ``compare``, ``smoke``, and ``eval``.

Everything the developer would otherwise have to decide — which experiment,
what counts as a run, which scorer and prompt versions were used, what this
was compared against — is decided here and recorded as run tags. The
developer picks the change; the toolkit produces the evidence.

This is the only module that calls ``mlflow.genai.evaluate``.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from aai_core.agentkit import catalog as catalog_module
from aai_core.agentkit import integrity as integrity_module
from aai_core.agentkit._values import is_missing_scalar, numeric_score
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
from aai_core.agentkit.calibration import calibration_failures
from aai_core.agentkit.catalog import (
    ScorerKind,
    ScorerPlan,
    render_plan,
    select_scorers,
)
from aai_core.agentkit.config import ProjectContext
from aai_core.agentkit.cost import CostEstimate, enforce_budget, estimate
from aai_core.agentkit.cost import render as render_cost
from aai_core.agentkit.datasets import (
    LoadedDataset,
    _trace_response,
    attach_answer_sheet,
    effective_dataset,
    load_dataset,
    rows_missing_inputs,
    smoke_sample,
    validate_dataset,
)
from aai_core.agentkit.economics import EconomicsEvidence, build_economics_evidence
from aai_core.agentkit.errors import (
    BaselineIncomparableError,
    BaselineMissingError,
    ConfigError,
    EvidenceMissingError,
    missing_extra,
)
from aai_core.agentkit.gate import (
    EXIT_ERROR,
    EXIT_PASS,
    EXIT_THRESHOLD_FAILED,
    build_policy,
)
from aai_core.agentkit.integrity import IntegrityEvidence, JudgeAnchors
from aai_core.agentkit.results import (
    ResultsAttempt,
    ResultsRecord,
    begin_results_attempt,
    complete_results_attempt,
    publish_results,
    write_results,
)
from aai_core.agentkit.statistics import (
    StatisticalEvidence,
    build_statistical_evidence,
    is_statistics_metric,
)
from aai_core.agentkit.targets import (
    Target,
    TargetKind,
    build_predict_fn,
    preflight_target,
    resolve_target,
)
from aai_core.decisions import Decision
from aai_core.evaluation import (
    GatePolicy,
    GateResult,
    apply_gate,
    gate_enforces_release_rule,
)
from aai_core.experiments import ExperimentRunMetadata, RunPurpose
from aai_core.prompts import is_missing_prompt_error

WORKERS_ENV = "MLFLOW_GENAI_EVAL_MAX_WORKERS"
SCORER_WORKERS_ENV = "MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"
DEFAULT_ANSWER_SHEET = "evals/data/answer_sheet.json"
_SMOKE_JUDGE_NOTE = (
    "smoke runs deterministic code scorers only so it stays free and "
    "credential-free; use `agentkit smoke --live` or `agentkit compare` to "
    "run judges"
)


@dataclass(frozen=True)
class ComparisonRow:
    metric: str
    current: float
    baseline: float | None
    delta: float | None
    threshold: str | None
    verdict: str


@dataclass
class RunOutcome:
    plan: ScorerPlan
    cost: CostEstimate
    dataset: LoadedDataset
    results: ResultsRecord | None = None
    gate: GateResult | None = None
    comparison: tuple[ComparisonRow, ...] = ()
    established_baseline: bool = False
    results_path: Path | None = None
    warnings: tuple[str, ...] = ()
    plan_only: bool = False
    declined: bool = False
    messages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _PreparedRun:
    target: Target
    mode: str
    dataset: LoadedDataset
    sampled: bool
    plan: ScorerPlan
    cost: CostEstimate
    judge_model_uri: str | None
    mode_warnings: tuple[str, ...]
    outcome: RunOutcome
    # Judge-integrity inputs, resolved before spend so the budget covers
    # the re-scoring calls too.
    anchors: JudgeAnchors | None = None
    integrity_calls: int = 0


@dataclass(frozen=True)
class _EvaluationBackend:
    mlflow: Any | None
    prompt_loader: _PromptLoader | None
    judge_prompts: dict[str, str]
    baseline: BaselineRecord | None


@dataclass(frozen=True)
class _ScoredRun:
    gate: GateResult
    decision: str
    run_id: str | None
    experiment_id: str | None
    experiment_name: str | None
    recorded_at: str
    change_id: str
    baseline_metrics: dict[str, float]
    policy: GatePolicy
    metric_samples: Mapping[str, tuple[float | None, ...]]
    statistics: StatisticalEvidence | None
    integrity: IntegrityEvidence | None = None
    economics: EconomicsEvidence | None = None
    anchor_rows: tuple[Any, ...] = ()


def set_concurrency_env(
    concurrency: int, environ: MutableMapping[str, str] | None = None
) -> None:
    """Point MLflow's judge concurrency at the configured value.

    LLM evaluation is I/O-bound — the ceiling is the judge endpoint's rate
    limit, not local CPU. MLflow exposes this only through environment
    variables, so the toolkit sets them rather than making every developer
    discover them.
    """

    target = environ if environ is not None else os.environ
    target.setdefault(WORKERS_ENV, str(concurrency))
    target.setdefault(SCORER_WORKERS_ENV, str(max(1, min(concurrency, 4))))


def run_scoring(
    project: ProjectContext,
    *,
    command: str = "compare",
    mode: str | None = None,
    agent: str | None = None,
    rows_limit: int | None = None,
    judges_enabled: bool = True,
    require_baseline: bool = True,
    establish_baseline: bool = False,
    allow_baseline_drift: bool = False,
    decision: str | None = None,
    baseline_run_id: str | None = None,
    assume_yes: bool = False,
    plan_only: bool = False,
    mlflow_module: Any | None = None,
    transport: Callable[..., Any] | None = None,
    confirm: Callable[[str], bool] | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> tuple[RunOutcome, int]:
    """Score the current version and compare it against the baseline."""

    # Every real invocation supersedes earlier gate evidence immediately,
    # even when target, dataset, baseline, budget, or prompt validation later
    # refuses the run. A plan is inspection rather than an evaluation attempt
    # and therefore leaves the current gate evidence untouched.
    attempt = (
        None
        if plan_only
        else begin_results_attempt(project.results_dir, command=command)
    )
    prepared = _prepare_run(
        project,
        mode=mode,
        agent=agent,
        rows_limit=rows_limit,
        judges_enabled=judges_enabled,
        plan_only=plan_only,
        mlflow_module=mlflow_module,
    )
    outcome = prepared.outcome
    if plan_only:
        return outcome, EXIT_PASS
    assert attempt is not None

    _require_calibrated_judges(project, prepared, judges_enabled=judges_enabled)
    baseline, warnings = _select_run_baseline(
        project,
        prepared,
        establish_baseline=establish_baseline,
        baseline_run_id=baseline_run_id,
        require_baseline=require_baseline,
        mlflow_module=mlflow_module,
    )
    enforce_budget(
        prepared.cost,
        max_judge_calls=project.config.budget.max_judge_calls,
        extra_judge_calls=prepared.integrity_calls,
    )
    judge_model_identity = project.judge_model_identity() if judges_enabled else None
    baseline = _check_model_comparability(
        prepared,
        baseline,
        warnings,
        judge_model_identity=judge_model_identity,
        judges_enabled=judges_enabled,
        allow_drift=allow_baseline_drift,
        require_baseline=require_baseline,
    )
    backend = _prepare_backend(
        project,
        prepared,
        baseline,
        warnings,
        judges_enabled=judges_enabled,
        allow_drift=allow_baseline_drift,
        require_baseline=require_baseline,
        mlflow_module=mlflow_module,
    )
    baseline = backend.baseline

    if _confirmation_declined(
        prepared.cost,
        outcome,
        assume_yes=assume_yes,
        confirm=confirm,
    ):
        return outcome, EXIT_ERROR

    # Confirmation follows every refusal that can be determined without
    # scoring, including governed prompt drift. Only an accepted run tunes
    # process-wide evaluator concurrency.
    set_concurrency_env(project.config.concurrency, environ)
    scored = _score_prepared_run(
        project,
        prepared,
        backend,
        baseline=baseline,
        establish_baseline=establish_baseline,
        decision=decision,
        command=command,
        transport=transport,
        judge_model_identity=judge_model_identity,
        judges_enabled=judges_enabled,
        warnings=warnings,
    )
    return _finish_scoring(
        project,
        prepared,
        backend,
        scored,
        attempt,
        baseline=baseline,
        establish_baseline=establish_baseline,
        judges_enabled=judges_enabled,
        command=command,
        judge_model_identity=judge_model_identity,
        warnings=warnings,
    )


def _prepare_run(
    project: ProjectContext,
    *,
    mode: str | None,
    agent: str | None,
    rows_limit: int | None,
    judges_enabled: bool,
    plan_only: bool,
    mlflow_module: Any | None,
) -> _PreparedRun:
    config = project.config
    target = _prepare_target(project, agent or config.agent, mode=mode)
    dataset = _load_ready_dataset(project, mlflow_module=mlflow_module)
    resolved_mode = _resolve_scoring_mode(project, target, dataset, mode=mode)
    mode_warnings = tuple(
        _mode_warnings(resolved_mode, dataset, explicit=mode is not None)
    )
    if resolved_mode == "answer-sheet":
        dataset = attach_answer_sheet(dataset, _answer_sheet_path(project, target))
    dataset, sampled = _effective_sample(
        dataset,
        mode=resolved_mode,
        rows_limit=rows_limit,
        strata=config.strata,
    )
    _require_dataset_inputs(dataset, mode=resolved_mode)
    judge_model_uri = project.judge_model_uri() if judges_enabled else None
    plan = select_scorers(
        dataset.shape,
        config,
        mode=resolved_mode,
        judges_enabled=judges_enabled,
        judge_note=None if judges_enabled else _SMOKE_JUDGE_NOTE,
    )
    cost = estimate(
        dataset.rows,
        plan,
        price_per_1m_tokens=config.budget.judge_price_per_1m_tokens,
        chunks_per_row=config.budget.retrieved_chunks_per_row,
    )
    anchors: JudgeAnchors | None = None
    integrity_calls = 0
    if judges_enabled:
        anchors_path = project.root / config.integrity.anchors
        if anchors_path.is_file():
            anchors = integrity_module.load_anchors(anchors_path)
        row_judges = sum(
            1 for spec in plan.judge_specs if integrity_module.is_row_level_judge(spec)
        )
        integrity_calls = integrity_module.estimate_integrity_calls(
            config.integrity,
            row_judges=row_judges,
            dataset_rows=dataset.shape.row_count,
            anchor_rows=len(anchors.rows) if anchors is not None else 0,
        )
    outcome = RunOutcome(plan=plan, cost=cost, dataset=dataset, plan_only=plan_only)
    outcome.messages.extend(
        (
            f"Inferred evaluation plan  (dataset: {dataset.ref}, "
            f"{dataset.shape.row_count} rows, digest {dataset.digest})",
            render_plan(plan, judge_model_uri=judge_model_uri),
            render_cost(cost),
        )
    )
    if integrity_calls:
        outcome.messages.append(
            f"Judge integrity re-scoring adds ~{integrity_calls} judge "
            "call(s) (self-consistency sample and frozen anchors); the "
            "budget ceiling covers them too."
        )
    return _PreparedRun(
        target=target,
        mode=resolved_mode,
        dataset=dataset,
        sampled=sampled,
        plan=plan,
        cost=cost,
        judge_model_uri=judge_model_uri,
        mode_warnings=mode_warnings,
        outcome=outcome,
        anchors=anchors,
        integrity_calls=integrity_calls,
    )


def _prepare_target(
    project: ProjectContext, reference: str, *, mode: str | None
) -> Target:
    target = resolve_target(reference, root=project.root, settings=project.settings)
    if mode == "live" and target.kind is TargetKind.ANSWER_SHEET:
        raise ConfigError(
            "live mode cannot use an answer-sheet agent target because it "
            "has no invocable agent",
            remediation=(
                "Remove --mode live to select answer-sheet mode automatically, "
                "choose --mode answer-sheet, or select a callable, HTTP, "
                "serving-endpoint, or model target."
            ),
        )
    preflight_target(target, project=project, require_invocation=mode == "live")
    return target


def _load_ready_dataset(
    project: ProjectContext, *, mlflow_module: Any | None
) -> LoadedDataset:
    dataset = load_dataset(
        project.config.dataset,
        root=project.root,
        mlflow_module=mlflow_module,
    )
    structural = validate_dataset(dataset)
    if structural:
        raise ConfigError(
            "the evaluation dataset is not ready:\n"
            + "\n".join(f"  - {failure}" for failure in structural),
            remediation="Fix the dataset rows, then run the command again.",
        )
    return dataset


def _resolve_scoring_mode(
    project: ProjectContext,
    target: Target,
    dataset: LoadedDataset,
    *,
    mode: str | None,
) -> str:
    resolved = mode or _default_mode(target, dataset)
    if resolved == "live" and mode != "live":
        preflight_target(target, project=project, require_invocation=True)
    if resolved == "traces" and not dataset.shape.has_traces:
        detail = (
            "only some rows carry one"
            if dataset.shape.partial_traces
            else "none of the rows carry one"
        )
        raise ConfigError(
            f"--mode traces needs a trace on every row, but {detail}",
            remediation=(
                "Give every row a trace, or run in live mode so the agent "
                "produces one for each."
            ),
        )
    return resolved


def _effective_sample(
    dataset: LoadedDataset,
    *,
    mode: str,
    rows_limit: int | None,
    strata: tuple[str, ...],
) -> tuple[LoadedDataset, bool]:
    effective = effective_dataset(dataset, mode=mode)
    full_row_count = effective.shape.row_count
    if rows_limit:
        effective = smoke_sample(effective, rows_limit, strata=strata)
    return effective, effective.shape.row_count < full_row_count


def _require_dataset_inputs(dataset: LoadedDataset, *, mode: str) -> None:
    if mode == "traces":
        return
    missing_inputs = rows_missing_inputs(dataset)
    if not missing_inputs:
        return
    listed = ", ".join(str(index) for index in missing_inputs[:5])
    raise ConfigError(
        f"{len(missing_inputs)} row(s) have no inputs to send the "
        f"agent (rows {listed}); a trace-only row needs a request "
        "that can be recovered from its trace",
        remediation=(
            "Run `--mode traces` to score the recorded traces, or "
            "give every row an `inputs` object."
        ),
    )


def _select_run_baseline(
    project: ProjectContext,
    prepared: _PreparedRun,
    *,
    establish_baseline: bool,
    baseline_run_id: str | None,
    require_baseline: bool,
    mlflow_module: Any | None,
) -> tuple[BaselineRecord | None, list[str]]:
    warnings = list(prepared.mode_warnings)
    if establish_baseline:
        existing, _ = load_baseline(project.baseline_path)
        if existing is not None:
            warnings.append(
                f"replacing the baseline recorded at {existing.recorded_at}"
            )
        if _release_value() is not None:
            warnings.append(
                "establishing the baseline inside a release run records the "
                "reference before the deployment was verified live; prefer "
                "`agentkit baseline establish --from-run <run_id>` after the "
                "deploy and its post-deploy smoke pass"
            )
        return None, warnings
    try:
        baseline, baseline_warnings = select_baseline(
            baseline_path=project.baseline_path,
            flag_run_id=baseline_run_id,
            config_run_id=project.config.baseline.run_id,
            mlflow_module=mlflow_module,
        )
    except BaselineMissingError:
        if require_baseline:
            raise
        baseline = None
        baseline_warnings = [
            "no baseline recorded yet, so this run reports absolute scores "
            "only; run `agentkit compare --establish-baseline` to start comparing"
        ]
    warnings.extend(baseline_warnings)
    if baseline is not None:
        warnings.extend(
            drift_warnings(
                baseline,
                dataset=prepared.dataset,
                mode=_scope_mode(prepared),
                rows=prepared.dataset.shape.row_count,
            )
        )
    return baseline, warnings


def _check_model_comparability(
    prepared: _PreparedRun,
    baseline: BaselineRecord | None,
    warnings: list[str],
    *,
    judge_model_identity: str | None,
    judges_enabled: bool,
    allow_drift: bool,
    require_baseline: bool,
) -> BaselineRecord | None:
    if baseline is None:
        return None
    comparability, comparable = _enforce_comparability(
        baseline,
        dataset=prepared.dataset,
        mode=_scope_mode(prepared),
        rows=prepared.dataset.shape.row_count,
        plan=prepared.plan,
        judge_model=prepared.judge_model_uri,
        judge_model_identity=judge_model_identity,
        judges_enabled=judges_enabled,
        allow_drift=allow_drift,
        blocking=require_baseline,
    )
    warnings.extend(comparability)
    if baseline.versions.judge_model_identity and not judge_model_identity:
        warnings.append(
            "the baseline was judged by "
            f"{baseline.versions.judge_model_identity}, but what the endpoint "
            "serves now could not be read, so a change behind the same endpoint "
            "name is unverified"
        )
    return baseline if comparable else None


def _prepare_backend(
    project: ProjectContext,
    prepared: _PreparedRun,
    baseline: BaselineRecord | None,
    warnings: list[str],
    *,
    judges_enabled: bool,
    allow_drift: bool,
    require_baseline: bool,
    mlflow_module: Any | None,
) -> _EvaluationBackend:
    scored_locally = _is_locally_scorable(prepared.plan, prepared.mode)
    mlflow = None if scored_locally else _mlflow(mlflow_module)
    if scored_locally:
        warnings.append(
            "scored locally: a code-scorer-only run does not open an MLflow "
            "run. Use `agentkit compare` to record the comparison."
        )
    prompt_loader: _PromptLoader | None = None
    judge_prompts: dict[str, str] = {}
    if judges_enabled and mlflow is not None:
        prompt_loader = _prompt_loader(project, mlflow)
        judge_prompts = _resolved_prompt_versions(prepared.plan, prompt_loader)
        baseline = _check_prompt_comparability(
            prepared,
            baseline,
            warnings,
            judge_prompts=judge_prompts,
            judges_enabled=judges_enabled,
            allow_drift=allow_drift,
            require_baseline=require_baseline,
        )
    return _EvaluationBackend(
        mlflow=mlflow,
        prompt_loader=prompt_loader,
        judge_prompts=judge_prompts,
        baseline=baseline,
    )


def _check_prompt_comparability(
    prepared: _PreparedRun,
    baseline: BaselineRecord | None,
    warnings: list[str],
    *,
    judge_prompts: Mapping[str, str],
    judges_enabled: bool,
    allow_drift: bool,
    require_baseline: bool,
) -> BaselineRecord | None:
    if baseline is None:
        return None
    prompt_drift, comparable = _enforce_comparability(
        baseline,
        dataset=prepared.dataset,
        mode=_scope_mode(prepared),
        rows=prepared.dataset.shape.row_count,
        plan=prepared.plan,
        judge_model=prepared.judge_model_uri,
        judge_prompts=judge_prompts,
        judges_enabled=judges_enabled,
        allow_drift=allow_drift,
        blocking=require_baseline,
        only_prompts=True,
    )
    warnings.extend(prompt_drift)
    return baseline if comparable else None


def _confirmation_declined(
    cost: CostEstimate,
    outcome: RunOutcome,
    *,
    assume_yes: bool,
    confirm: Callable[[str], bool] | None,
) -> bool:
    if not cost.judge_calls or assume_yes:
        return False
    if confirm is not None and confirm("Proceed?"):
        return False
    outcome.declined = True
    outcome.messages.append(
        "Cancelled - nothing was scored. Pass --yes to run without the "
        "confirmation prompt."
    )
    return True


def _score_prepared_run(
    project: ProjectContext,
    prepared: _PreparedRun,
    backend: _EvaluationBackend,
    *,
    baseline: BaselineRecord | None,
    establish_baseline: bool,
    decision: str | None,
    command: str,
    transport: Callable[..., Any] | None,
    judge_model_identity: str | None,
    judges_enabled: bool,
    warnings: list[str],
) -> _ScoredRun:
    change_id = _change_id()
    summary = _run_summary(prepared.target, establish_baseline=establish_baseline)
    metadata = ExperimentRunMetadata(
        purpose=RunPurpose.BASELINE if establish_baseline else RunPurpose.RESULT,
        change_id=change_id,
        change_summary=summary,
        baseline_run_id=baseline.run_id if baseline else None,
    )
    recorded_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    baseline_metrics = dict(baseline.metrics) if baseline else {}
    policy = build_policy(
        project,
        plan=prepared.plan,
        allow_missing_regression_baseline=establish_baseline or not baseline_metrics,
        judges_enabled=judges_enabled,
    )
    baseline_samples = dict(baseline.metric_samples) if baseline else {}
    metric_samples: Mapping[str, tuple[float | None, ...]]
    integrity: IntegrityEvidence | None = None
    economics: EconomicsEvidence | None = None
    anchor_rows: tuple[Any, ...] = ()
    if backend.mlflow is None:
        local_metrics, metric_samples = _score_locally(prepared.dataset, prepared.plan)
        statistics, statistical_metrics = build_statistical_evidence(
            metric_samples,
            baseline_samples,
            policy.rules,
            project.config.statistics,
        )
        local_metrics.update(statistical_metrics)
        gate = apply_gate(
            local_metrics,
            policy=policy,
            baseline_metrics=baseline_metrics,
        )
        recorded_decision = _decision_value(decision, gate)
        run_id = experiment_id = experiment_name = None
    else:
        (
            gate,
            recorded_decision,
            run_id,
            experiment_id,
            experiment_name,
            metric_samples,
            statistics,
            integrity,
            economics,
            anchor_rows,
        ) = _score_with_mlflow(
            project,
            prepared,
            backend,
            metadata=metadata,
            summary=summary,
            command=command,
            decision=decision,
            transport=transport,
            judge_model_identity=judge_model_identity,
            recorded_at=recorded_at,
            baseline_metrics=baseline_metrics,
            baseline_samples=baseline_samples,
            policy=policy,
            warnings=warnings,
        )
    return _ScoredRun(
        gate=gate,
        decision=recorded_decision,
        run_id=run_id,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        recorded_at=recorded_at,
        change_id=change_id,
        baseline_metrics=baseline_metrics,
        policy=policy,
        metric_samples=metric_samples,
        statistics=statistics,
        integrity=integrity,
        economics=economics,
        anchor_rows=anchor_rows,
    )


def _score_with_mlflow(
    project: ProjectContext,
    prepared: _PreparedRun,
    backend: _EvaluationBackend,
    *,
    metadata: ExperimentRunMetadata,
    summary: str,
    command: str,
    decision: str | None,
    transport: Callable[..., Any] | None,
    judge_model_identity: str | None,
    recorded_at: str,
    baseline_metrics: Mapping[str, float],
    baseline_samples: Mapping[str, tuple[float | None, ...]],
    policy: GatePolicy,
    warnings: list[str],
) -> tuple[
    GateResult,
    str,
    str | None,
    str | None,
    str,
    Mapping[str, tuple[float | None, ...]],
    StatisticalEvidence | None,
    IntegrityEvidence | None,
    EconomicsEvidence | None,
    tuple[Any, ...],
]:
    mlflow = backend.mlflow
    assert mlflow is not None
    predict_fn = (
        build_predict_fn(
            prepared.target,
            project=project,
            transport=transport,
            mlflow_module=mlflow,
        )
        if prepared.mode == "live"
        else None
    )
    scorers = [
        catalog_module.build_scorer(
            entry.spec,
            judge_model_uri=prepared.judge_model_uri,
            guidelines=project.config.scorers.guidelines,
            prompt_loader=backend.prompt_loader,
            mlflow_module=mlflow,
        )
        for entry in prepared.plan.entries
    ]
    manager = project.experiment_manager(mlflow_module=mlflow)
    experiment_name = manager.experiment_name
    with manager.run(
        run_name=f"{command}-{prepared.mode}",
        description=summary,
        parameters={
            "mode": prepared.mode,
            "dataset": prepared.dataset.ref,
            "row_count": prepared.dataset.shape.row_count,
        },
        metadata=metadata,
    ) as active_run:
        run_id, experiment_id = _run_identity(active_run)
        native_result = mlflow.genai.evaluate(
            data=[dict(row) for row in prepared.dataset.rows],
            scorers=scorers,
            predict_fn=predict_fn,
        )
        warnings.extend(_coverage_warnings(native_result))
        metric_samples = _metric_samples(native_result)
        metrics = _metrics_with_scorer_errors(native_result)
        statistics, statistical_metrics = build_statistical_evidence(
            metric_samples,
            baseline_samples,
            policy.rules,
            project.config.statistics,
        )
        metrics.update(statistical_metrics)
        integrity, integrity_metrics, integrity_warnings, anchor_rows = _run_integrity(
            project,
            prepared,
            scorers=scorers,
            native_result=native_result,
            metric_samples=metric_samples,
            capture_anchors=metadata.purpose is RunPurpose.BASELINE,
        )
        metrics.update(integrity_metrics)
        warnings.extend(integrity_warnings)
        economics, economics_metrics, economics_warnings = _run_economics(
            project,
            prepared,
            native_result=native_result,
        )
        metrics.update(economics_metrics)
        warnings.extend(economics_warnings)
        gate = apply_gate(
            metrics,
            policy=policy,
            baseline_metrics=baseline_metrics,
        )
        recorded_decision = _decision_value(decision, gate)
        tags = _run_tags(
            prepared,
            backend,
            recorded_at=recorded_at,
            judge_model_identity=judge_model_identity,
            gate=gate,
            decision=recorded_decision,
        )
        mlflow.log_metrics(dict(gate.metrics))
        mlflow.set_tags(tags)
        _record_reproducibility(mlflow)
    return (
        gate,
        recorded_decision,
        run_id,
        experiment_id,
        experiment_name,
        metric_samples,
        statistics,
        integrity,
        economics,
        anchor_rows,
    )


def _run_economics(
    project: ProjectContext,
    prepared: _PreparedRun,
    *,
    native_result: Any,
) -> tuple[EconomicsEvidence | None, dict[str, float], list[str]]:
    """Harvest what this run spent from the traces it just produced.

    Live runs read the ``trace`` column of MLflow's result frame — the
    execution that was actually scored; traces runs fall back to the
    stored envelopes the rows carry. Answer-sheet replay is out of
    contract: the sheet holds no agent trace, and reading the harness's
    own evaluation traces would record what the judges cost, not what
    the agent did.
    """

    if prepared.mode not in ("live", "traces"):
        return None, {}, []
    config = project.config.economics
    if not config.enabled:
        return None, {}, []
    rows = prepared.dataset.rows
    traces = _row_traces(native_result, rows, mode=prepared.mode)
    if traces is None:
        return (
            None,
            {},
            [
                "run economics were not recorded: the evaluation result "
                "carried no per-row traces to read"
            ],
        )
    return build_economics_evidence(
        rows,
        traces,
        _row_error_flags(native_result, len(rows)),
        strata=project.config.strata,
        config=config,
    )


def _row_traces(
    native_result: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> list[Any] | None:
    """Per-row traces in dataset order, or ``None`` when none exist.

    The frame's ``trace`` column is what the run actually executed, so it
    wins; a traces-mode row keeps its stored envelope when the frame does
    not carry the column.
    """

    frame = _result_frame(native_result)
    columns = getattr(frame, "columns", None)
    frame_traces: list[Any] | None = None
    if columns is not None and "trace" in {str(column) for column in columns}:
        frame_traces = list(frame["trace"])
    collected: list[Any] = []
    found = False
    for index, row in enumerate(rows):
        value: Any = None
        if frame_traces is not None and index < len(frame_traces):
            candidate = frame_traces[index]
            if not is_missing_scalar(candidate):
                value = candidate
        if value is None and mode == "traces":
            candidate = row.get("trace")
            if candidate is not None and not is_missing_scalar(candidate):
                value = candidate
        if value is not None:
            found = True
        collected.append(value)
    return collected if found else None


def _row_error_flags(native_result: Any, row_count: int) -> tuple[bool, ...]:
    """Which rows produced no answer at all.

    MLflow records a failed ``predict_fn`` invocation in the result
    table's bare ``error_message`` column. The scorer error columns are
    deliberately not read here: a failed judge measures the instrument,
    and it already fails the gate through ``<scorer>/error_count``.
    """

    frame = _result_frame(native_result)
    columns = getattr(frame, "columns", None)
    if columns is None or "error_message" not in {str(c) for c in columns}:
        return tuple(False for _ in range(row_count))
    flags = [_is_reported_error(value) for value in frame["error_message"]]
    flags = flags[:row_count]
    flags.extend(False for _ in range(row_count - len(flags)))
    return tuple(flags)


def _run_integrity(
    project: ProjectContext,
    prepared: _PreparedRun,
    *,
    scorers: Sequence[Any],
    native_result: Any,
    metric_samples: Mapping[str, tuple[float | None, ...]],
    capture_anchors: bool,
) -> tuple[
    IntegrityEvidence | None,
    dict[str, float],
    list[str],
    tuple[Any, ...],
]:
    """Measure the judge inside the run it just scored.

    Runs between scoring and the gate so the integrity metrics are gate
    evidence like any other, and inside the active MLflow run so they land
    on the same run. Anchor rows are captured only for a baseline run —
    they are what ``--establish-baseline`` freezes.
    """

    if not prepared.plan.judges_enabled:
        return None, {}, [], ()
    config = project.config.integrity
    judges: list[integrity_module.RowJudge] = []
    for entry, scorer in zip(prepared.plan.entries, scorers, strict=True):
        spec = entry.spec
        if integrity_module.is_row_level_judge(spec):
            judges.append(
                integrity_module.RowJudge(
                    name=spec.name, metric=spec.metric, scorer=scorer
                )
            )
    wants_checks = (
        config.consistency_sample > 0
        or config.require_anchors
        or prepared.anchors is not None
    )
    if not wants_checks:
        # Anchors carry eval outputs, so nothing output-bearing is written
        # for a project that has not opted into the integrity checks.
        return None, {}, [], ()
    outputs_by_row = _integrity_outputs(native_result, prepared.dataset.rows)
    evidence, metrics, warnings = integrity_module.run_integrity_checks(
        config=config,
        rows=prepared.dataset.rows,
        outputs_by_row=outputs_by_row,
        metric_samples=metric_samples,
        judges=judges,
        anchors=prepared.anchors,
    )
    anchor_rows: tuple[Any, ...] = ()
    if capture_anchors and judges:
        anchor_rows = integrity_module.build_anchor_rows(
            rows=prepared.dataset.rows,
            outputs_by_row=outputs_by_row,
            metric_samples=metric_samples,
            judges=judges,
        )
    return evidence, metrics, warnings, anchor_rows


def _integrity_outputs(
    native_result: Any, rows: Sequence[Mapping[str, Any]]
) -> list[Any]:
    """The answer each row was judged on, recovered without a second run.

    Preference order: the native result frame's ``outputs`` column (live
    runs — MLflow records what ``predict_fn`` returned), the row's own
    recorded ``outputs`` (answer-sheet runs), then the response inside the
    row's trace (trace runs). ``None`` marks a row whose answer cannot be
    recovered; it is skipped rather than re-answered.
    """

    frame = _result_frame(native_result)
    columns = getattr(frame, "columns", None)
    frame_outputs: list[Any] | None = None
    if columns is not None and "outputs" in {str(column) for column in columns}:
        frame_outputs = list(frame["outputs"])
    recovered: list[Any] = []
    for index, row in enumerate(rows):
        value: Any = None
        if frame_outputs is not None and index < len(frame_outputs):
            candidate = frame_outputs[index]
            if not is_missing_scalar(candidate):
                value = candidate
        if value is None:
            candidate = row.get("outputs")
            if candidate is not None and not is_missing_scalar(candidate):
                value = candidate
        if value is None and row.get("trace") is not None:
            value = _trace_response(row.get("trace"))
        recovered.append(value)
    return recovered


def _run_tags(
    prepared: _PreparedRun,
    backend: _EvaluationBackend,
    *,
    recorded_at: str,
    judge_model_identity: str | None,
    gate: GateResult,
    decision: str,
) -> dict[str, str]:
    tags = {
        "aai.agentkit_version": _version(),
        "aai.scorer_versions": prepared.plan.scorer_versions_tag(),
        "aai.dataset": prepared.dataset.ref,
        "aai.dataset_digest": prepared.dataset.digest,
        "aai.dataset_rows": str(prepared.dataset.shape.row_count),
        "aai.scope_mode": _scope_mode(prepared),
        "aai.scope_rows": str(prepared.dataset.shape.row_count),
        "aai.agent_target": prepared.target.normalized,
        "aai.recorded_at": recorded_at,
        "aai.gate_passed": str(gate.passed).lower(),
        "aai.decision": decision,
    }
    if prepared.judge_model_uri:
        tags["aai.judge_model"] = prepared.judge_model_uri
    if judge_model_identity:
        tags["aai.judge_model_identity"] = judge_model_identity
    if backend.judge_prompts:
        tags["aai.judge_prompt_versions"] = ",".join(
            f"{name}={uri}" for name, uri in sorted(backend.judge_prompts.items())
        )
    return tags


def _finish_scoring(
    project: ProjectContext,
    prepared: _PreparedRun,
    backend: _EvaluationBackend,
    scored: _ScoredRun,
    attempt: ResultsAttempt,
    *,
    baseline: BaselineRecord | None,
    establish_baseline: bool,
    judges_enabled: bool,
    command: str,
    judge_model_identity: str | None,
    warnings: list[str],
) -> tuple[RunOutcome, int]:
    warnings.extend(_missing_judge_metric_warnings(scored.gate, prepared.plan))
    versions = _baseline_versions(
        prepared,
        backend,
        judge_model_identity=judge_model_identity,
    )
    scope = BaselineScope(
        mode=_scope_mode(prepared),
        rows=prepared.dataset.shape.row_count,
        seed=None,
    )
    results = _results_record(
        prepared,
        scored,
        attempt,
        baseline=baseline,
        versions=versions,
        scope=scope,
        establish_baseline=establish_baseline,
        judges_enabled=judges_enabled,
        command=command,
        warnings=warnings,
    )
    _publish_remote_results(prepared.outcome, backend.mlflow, results, scored)
    results_path = write_results(project.results_dir, results, attempt=attempt)
    if establish_baseline:
        _write_new_baseline(
            project,
            results,
            scored,
            versions=versions,
            scope=scope,
            command=command,
        )
        if judges_enabled and scored.anchor_rows:
            integrity_module.write_anchors(
                project.root / project.config.integrity.anchors,
                rows=scored.anchor_rows,
                recorded_at=scored.recorded_at,
                recorded_by=f"agentkit {command} --establish-baseline",
                change_id=scored.change_id,
                judge_model=prepared.judge_model_uri,
                judge_model_identity=judge_model_identity,
                judge_prompts=backend.judge_prompts,
                scorer_versions={
                    spec.name: spec.version
                    for spec in prepared.plan.specs
                    if spec.judge is not None
                },
            )
            prepared.outcome.messages.append(
                f"Judge anchors frozen at {project.config.integrity.anchors} "
                f"({len(scored.anchor_rows)} rows) - commit them together "
                "with the baseline so future runs can tell judge drift from "
                "agent change."
            )
    complete_results_attempt(project.results_dir, attempt, results_path)
    comparison = _comparison_rows(
        scored.gate,
        scored.baseline_metrics,
        scored.policy,
    )
    outcome = prepared.outcome
    outcome.results = results
    outcome.gate = scored.gate
    outcome.comparison = comparison
    outcome.established_baseline = establish_baseline
    outcome.results_path = results_path
    outcome.warnings = tuple(warnings)
    outcome.messages.extend(_render_outcome(results, comparison, warnings, scope))
    return outcome, (EXIT_PASS if scored.gate.passed else EXIT_THRESHOLD_FAILED)


def _baseline_versions(
    prepared: _PreparedRun,
    backend: _EvaluationBackend,
    *,
    judge_model_identity: str | None,
) -> BaselineVersions:
    return BaselineVersions(
        agent=prepared.target.normalized,
        scorers={spec.name: spec.version for spec in prepared.plan.specs},
        judge_model=prepared.judge_model_uri,
        judge_model_identity=judge_model_identity,
        judge_prompts=backend.judge_prompts,
        aai_core=_version(),
    )


def _results_record(
    prepared: _PreparedRun,
    scored: _ScoredRun,
    attempt: ResultsAttempt,
    *,
    baseline: BaselineRecord | None,
    versions: BaselineVersions,
    scope: BaselineScope,
    establish_baseline: bool,
    judges_enabled: bool,
    command: str,
    warnings: Sequence[str],
) -> ResultsRecord:
    return ResultsRecord(
        attempt_id=attempt.attempt_id,
        command=command,
        recorded_at=scored.recorded_at,
        run_id=scored.run_id,
        experiment_id=scored.experiment_id,
        experiment_name=scored.experiment_name,
        agent=prepared.target.normalized,
        dataset=BaselineDataset(
            ref=prepared.dataset.ref,
            digest=prepared.dataset.digest,
            rows=prepared.dataset.shape.row_count,
        ),
        scope=scope,
        mode=prepared.mode,
        metrics=dict(scored.gate.metrics),
        metric_samples=dict(scored.metric_samples),
        statistics=scored.statistics,
        integrity=scored.integrity,
        economics=scored.economics,
        versions=versions,
        baseline_run_id=baseline.run_id if baseline else None,
        baseline_metrics=scored.baseline_metrics,
        baseline_recorded_at=baseline.recorded_at if baseline else None,
        baseline_dataset_digest=baseline.dataset.digest if baseline else None,
        established_baseline=establish_baseline,
        policy_rules=scored.policy.rules,
        allow_missing_regression_baseline=(
            scored.policy.allow_missing_regression_baseline
        ),
        decision=scored.decision,
        change_id=scored.change_id,
        release=_release_value(),
        gate_passed=scored.gate.passed,
        gate_failures=tuple(
            {"metric": failure.metric, "reason": failure.reason}
            for failure in scored.gate.failures
        ),
        warnings=tuple(warnings),
        judges_enabled=judges_enabled,
    )


def _publish_remote_results(
    outcome: RunOutcome,
    mlflow: Any | None,
    results: ResultsRecord,
    scored: _ScoredRun,
) -> None:
    if mlflow is None or not scored.run_id:
        return
    failure = publish_results(mlflow, scored.run_id, results)
    if failure:
        raise EvidenceMissingError(
            f"{failure}\nThe run was scored (gate "
            f"{'passed' if scored.gate.passed else 'FAILED'}) but its results "
            "record could not be attached, so `agentkit evidence --run "
            f"{scored.run_id}` would find nothing.",
            remediation=(
                "Check MLflow artifact permissions for this experiment, then "
                "run the evaluation again."
            ),
        )
    outcome.messages.append(
        f"Evidence for this run: agentkit evidence --run {scored.run_id}"
    )


def _write_new_baseline(
    project: ProjectContext,
    results: ResultsRecord,
    scored: _ScoredRun,
    *,
    versions: BaselineVersions,
    scope: BaselineScope,
    command: str,
) -> None:
    write_baseline(
        project.baseline_path,
        BaselineRecord(
            schema_version=1,
            run_id=scored.run_id,
            experiment_id=scored.experiment_id,
            recorded_at=scored.recorded_at,
            dataset=results.dataset,
            scope=scope,
            metrics=dict(scored.gate.metrics),
            metric_samples=dict(scored.metric_samples),
            versions=versions,
            recorded_by=f"agentkit {command} --establish-baseline",
            change_id=scored.change_id,
        ),
    )


def _require_calibrated_judges(
    project: ProjectContext,
    prepared: _PreparedRun,
    *,
    judges_enabled: bool,
) -> None:
    """Refuse a judged run whose pinned judges lack calibration evidence.

    Opt-in (``integrity.require_calibration``) and checked BEFORE any
    judge spend: a score from an uncalibrated judge would be paid for and
    then unusable as promotion evidence.
    """

    if not judges_enabled or not project.config.integrity.require_calibration:
        return
    judge_scorers = {spec.name: spec.version for spec in prepared.plan.judge_specs}
    if not judge_scorers:
        return
    failures = calibration_failures(
        root=project.root,
        directory=project.config.integrity.calibration_dir,
        judge_scorers=judge_scorers,
    )
    if failures:
        raise ConfigError(
            "integrity.require_calibration is set, and the pinned judges "
            "are not covered:\n" + "\n".join(f"  - {failure}" for failure in failures),
            remediation=(
                "Calibrate each judge against SME labels (`agentkit judge "
                "calibrate --scorer <name> --labels <file>`) and commit the "
                "records, or unset integrity.require_calibration."
            ),
        )


def establish_baseline_from_run(
    project: ProjectContext,
    run_id: str,
    *,
    decided_by: str | None = None,
    mlflow_module: Any | None = None,
) -> tuple[list[str], int]:
    """Move the recorded baseline to an already-verified run's evidence.

    The v2.1 principle "the reference moves only after live verification",
    in this platform's vocabulary: after the deploy workflow's release
    gate and post-deploy smoke are green, a human points the baseline at
    that run. The run's own recorded policy is recomputed and must pass,
    adopt evidence is required (or recorded here with ``--decided-by``),
    and the result is a reviewable edit to the committed baseline file —
    CI never commits.
    """

    from aai_core.agentkit.gate import evaluate_gate
    from aai_core.agentkit.results import fetch_results
    from aai_core.decisions import DecisionRecord, record_decision

    messages: list[str] = []
    record = fetch_results(run_id, mlflow_module=mlflow_module)
    if record.command == "smoke":
        raise ConfigError(
            f"run {run_id} is a smoke run; a sampled, judge-free gate "
            "cannot become the judged reference",
            remediation="Point --from-run at an `agentkit compare` or "
            "`agentkit eval` run.",
        )
    if not record.judges_enabled:
        raise ConfigError(
            f"run {run_id} was scored without judges; a judge-free run "
            "must not become the judged reference",
            remediation="Point --from-run at a judged run.",
        )
    report, code = evaluate_gate(project, results=record, baseline=None)
    if code != EXIT_PASS:
        messages.append(
            f"run {run_id} does not pass its own recorded gate; the "
            "baseline stays where it is:"
        )
        for failure in report.result.failures:
            messages.append(f"  - {failure.metric}: {failure.reason}")
        return messages, EXIT_THRESHOLD_FAILED
    if record.decision != Decision.ADOPT.value:
        if not decided_by:
            messages.append(
                f"run {run_id} carries decision {record.decision!r}, not an "
                "adopt. Moving the baseline is adopting the change: pass "
                "--decided-by group:<owners> to record the governed adopt "
                "decision now, or record one first and re-run."
            )
            return messages, EXIT_THRESHOLD_FAILED
        try:
            decision = DecisionRecord(
                decision=Decision.ADOPT,
                change_id=record.change_id,
                change_summary=("Move the recorded baseline to the verified release"),
                rationale=(
                    "Adopted after the deployment and its post-deploy "
                    f"verification; the baseline moves to run {run_id}."
                ),
                baseline_run_id=record.baseline_run_id,
                change_run_id=run_id,
                gate=report.result,
                decided_by=decided_by,
            )
        except ValueError as error:
            raise ConfigError(
                f"run {run_id} cannot back an adopt decision: {error}"
            ) from error
        decision_run = record_decision(
            decision,
            experiments=project.experiment_manager(mlflow_module=mlflow_module),
        )
        messages.append(f"adopt decision recorded: run {decision_run}")
    with suppress(Exception):
        current = load_dataset(
            project.config.dataset, root=project.root, mlflow_module=mlflow_module
        )
        if current.digest != record.dataset.digest:
            messages.append(
                f"warning: run {run_id} scored dataset digest "
                f"{record.dataset.digest}, but the local dataset is "
                f"{current.digest}; the next `agentkit compare` will refuse "
                "the comparison until they agree"
            )
    write_baseline(
        project.baseline_path,
        BaselineRecord(
            schema_version=1,
            run_id=record.run_id,
            experiment_id=record.experiment_id,
            recorded_at=record.recorded_at,
            dataset=record.dataset,
            scope=record.scope,
            metrics=dict(record.metrics),
            metric_samples=dict(record.metric_samples),
            versions=record.versions,
            recorded_by="agentkit baseline establish --from-run",
            change_id=record.change_id,
        ),
    )
    messages.append(f"baseline -> run {run_id} ({project.config.baseline.file})")
    messages.append(
        "Commit the baseline file via a pull request - CI never commits. "
        "Judge anchors pin the judge, not the agent, and are not rebuilt "
        "here; after a judge release, refresh them with a judged "
        "`agentkit compare --establish-baseline` run."
    )
    return messages, EXIT_PASS


def _scope_mode(prepared: _PreparedRun) -> Literal["sample", "full"]:
    return "sample" if prepared.sampled else "full"


def _run_summary(target: Target, *, establish_baseline: bool) -> str:
    if establish_baseline:
        return "Recording the first scored version as the baseline"
    return f"Comparing {target.normalized} against the recorded baseline"


def submit_job(
    project: ProjectContext,
    *,
    target: str = "dev",
    runner: Callable[..., Any] | None = None,
) -> tuple[int, list[str]]:
    """Run the bundle's ``release_gate`` job — compute goes to the data."""

    runner = runner or subprocess.run
    messages = []
    for command in (
        ["databricks", "bundle", "validate", "-t", target],
        ["databricks", "bundle", "run", "release_gate", "-t", target],
    ):
        messages.append(f"$ {' '.join(command)}")
        completed = runner(command, cwd=str(project.root), check=False)
        code = getattr(completed, "returncode", 0)
        if code != 0:
            messages.append(f"command failed with exit code {code}")
            return code, messages
    return EXIT_PASS, messages


def _enforce_comparability(
    baseline: BaselineRecord,
    *,
    dataset: LoadedDataset,
    mode: str,
    rows: int,
    plan: ScorerPlan,
    judge_model: str | None,
    judge_model_identity: str | None = None,
    judge_prompts: Mapping[str, str] | None = None,
    judges_enabled: bool = True,
    allow_drift: bool = False,
    blocking: bool = True,
    only_prompts: bool = False,
) -> tuple[list[str], bool]:
    """Refuse a baseline that measured something else — or set it aside.

    Returns the warnings to record and whether the baseline survives.

    ``blocking`` is what separates the two speeds. `compare` and `eval`
    produce promotion evidence, so an incomparable baseline is a refusal:
    the delta is the deliverable, and a delta that measures nothing is
    worse than none. `smoke` is the fast threshold gate — it runs a
    deterministic sample of the dataset on every commit, which is by
    definition a narrower scope than the baseline's — so there the
    baseline is set aside and the run reports absolute scores, exactly as
    it does before any baseline exists. Refusing there would break the
    credential-free pull-request gate as soon as a suite outgrows
    ``smoke.rows``, and comparing anyway would fail pull requests on
    sampling noise. Neither is silent: every reason is printed and lands
    in the results record.

    ``only_prompts`` is the second pass. Judge prompt versions resolve
    only once MLflow is in hand, which is after the first pass has already
    cleared everything checkable offline; running the whole check again
    would report those same failures twice.
    """

    if only_prompts:
        failures = comparability_failures(
            baseline,
            dataset=dataset,
            mode=mode,
            rows=rows,
            judge_prompts=judge_prompts,
        )
        failures = [failure for failure in failures if "judge prompt" in failure]
    else:
        failures = comparability_failures(
            baseline,
            dataset=dataset,
            mode=mode,
            rows=rows,
            scorers={spec.name: spec.version for spec in plan.specs},
            judge_model=judge_model,
            judge_model_identity=judge_model_identity,
            judges_enabled=judges_enabled,
        )
    if not failures:
        return [], True
    listed = "\n".join(f"  - {failure}" for failure in failures)
    if allow_drift:
        # Overriding is a decision someone made; the evidence has to say so.
        return [
            "compared against a baseline that is not directly comparable "
            "(--allow-baseline-drift):\n" + listed
        ], True
    if not blocking:
        return [
            "the recorded baseline does not describe this run, so it is set "
            "aside and this run reports absolute scores only:\n"
            + listed
            + "\nRun `agentkit compare` to compare against the baseline."
        ], False
    raise BaselineIncomparableError(
        "the recorded baseline cannot be compared against this run:\n" + listed,
        remediation=(
            "Re-record the baseline on the current dataset and scorers with "
            "`agentkit compare --establish-baseline`, or pass "
            "--allow-baseline-drift to compare anyway (the reason is "
            "recorded in the results and the evidence)."
        ),
    )


def _metrics_with_scorer_errors(native_result: Any) -> dict[str, float]:
    """Native metrics plus a per-scorer count of failed invocations.

    MLflow reports a scorer that raised in the result table as a
    ``<scorer>/error_message`` cell and leaves it out of ``metrics``
    entirely, so a judge that failed on nine rows out of ten still
    produces a healthy-looking mean over the tenth. The gate engine
    already refuses a run with ``<scorer>/error_count`` above zero
    (``GatePolicy.fail_on_scorer_errors``) — it was never given the
    counts. This is where they come from.
    """

    metrics = {
        str(key): float(value)
        for key, value in dict(getattr(native_result, "metrics", {}) or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    frame = _result_frame(native_result)
    columns = getattr(frame, "columns", None)
    if columns is None:
        return metrics
    for column in list(columns):
        name = str(column)
        if not name.endswith("/error_message"):
            continue
        scorer = name.removesuffix("/error_message")
        # Duck-typed rather than imported: pandas is MLflow's dependency,
        # not the SDK's, and this path only runs with MLflow installed.
        failures = sum(1 for value in frame[column] if _is_reported_error(value))
        metrics[f"{scorer}/error_count"] = float(failures)
    return metrics


def _metric_samples(
    native_result: Any,
) -> dict[str, tuple[float | None, ...]]:
    """Extract aligned numeric per-row scores from MLflow's native table.

    Boolean and yes/no feedback are converted to 1/0, matching MLflow's mean
    aggregation. Unsupported categorical values remain absent rather than being
    assigned an invented ordering.
    """

    frame = _result_frame(native_result)
    columns = getattr(frame, "columns", None)
    if columns is None:
        return {}
    samples: dict[str, tuple[float | None, ...]] = {}
    for column in list(columns):
        name = str(column)
        if not name.endswith("/value"):
            continue
        scorer = name.removesuffix("/value")
        values = tuple(_numeric_score(value) for value in frame[column])
        if any(value is not None for value in values):
            samples[f"{scorer}/mean"] = values
    return samples


# The yes/no numeric mapping is shared with the integrity re-scoring path
# so a re-score is compared in the same units as the original.
_numeric_score = numeric_score


def _coverage_warnings(native_result: Any) -> list[str]:
    """Name the scorers that judged fewer rows than the run scored.

    A retrieval scorer skips a row whose trace retrieved nothing, so its
    mean is over the retrieving rows only. That is the right arithmetic —
    scoring the others zero would punish an agent for correctly not
    retrieving — but reporting it as a whole-dataset number would be the
    same quiet subset-averaging the expectation checks refuse. So the run
    says how many rows each scorer actually judged.
    """

    frame = _result_frame(native_result)
    columns = getattr(frame, "columns", None)
    if columns is None:
        return []
    names = {str(column) for column in columns}
    warnings = []
    for name in sorted(names):
        if not name.endswith("/value"):
            continue
        scorer = name.removesuffix("/value")
        errors = frame.get(f"{scorer}/error_message") if hasattr(frame, "get") else None
        rows = list(frame[name])
        scored = 0
        skipped = 0
        for index, value in enumerate(rows):
            if _is_populated_score(value):
                scored += 1
            elif errors is None or not _is_reported_error(list(errors)[index]):
                # Absent with no error recorded: the scorer declined the
                # row rather than failing on it.
                skipped += 1
        if skipped:
            warnings.append(
                f"{scorer} judged {scored} of {scored + skipped} rows; "
                f"{skipped} had nothing for it to score, so its mean covers "
                "the rest"
            )
    return warnings


def _result_frame(native_result: Any) -> Any:
    frame = getattr(native_result, "result_df", None)
    if frame is None:
        tables = getattr(native_result, "tables", None)
        if isinstance(tables, Mapping):
            frame = tables.get("eval_results")
    return frame


def _is_populated_score(value: Any) -> bool:
    """A recorded score, as opposed to a null cell.

    NaN is how a declined row arrives, and NaN != NaN is the only check
    that works without importing pandas.
    """

    return not is_missing_scalar(value)


def _is_reported_error(value: Any) -> bool:
    if is_missing_scalar(value):
        return False
    return bool(str(value).strip())


def _default_mode(target: Target, dataset: LoadedDataset) -> str:
    """Where the answers come from.

    A dataset that already carries traces has already been answered — by
    production, usually. Scoring it means scoring those traces, which is
    what MLflow does when ``predict_fn`` is omitted. Calling the agent
    instead would discard the recorded behaviour and score something else
    that happens to share the questions.
    """

    if target.kind is TargetKind.ANSWER_SHEET:
        return "answer-sheet"
    if dataset.shape.has_traces:
        return "traces"
    return "live"


def _mode_warnings(mode: str, dataset: LoadedDataset, *, explicit: bool) -> list[str]:
    warnings = []
    if mode == "live" and dataset.shape.has_traces and explicit:
        warnings.append(
            "--mode live on a dataset that carries traces: the agent is "
            "called again and the recorded traces are not what gets scored. "
            "Use --mode traces to score the traces the dataset holds."
        )
    if dataset.shape.partial_traces:
        # Scoring traces needs every row to have one: a traces run supplies
        # no predict_fn, so the untraced rows would have no answer at all.
        warnings.append(
            f"only some rows carry a trace, so this is a {mode} run. Give "
            "every row a trace to score the recorded behaviour instead."
        )
    return warnings


def _answer_sheet_path(project: ProjectContext, target: Target) -> Path:
    configured = project.config.smoke.answer_sheet
    if configured:
        return project.root / configured
    if target.kind is TargetKind.ANSWER_SHEET and target.path is not None:
        return target.path
    return project.root / DEFAULT_ANSWER_SHEET


def _decision_value(decision: str | None, gate: GateResult) -> str:
    if not decision:
        return Decision.INCONCLUSIVE.value
    parsed = Decision(decision)
    if parsed is Decision.ADOPT:
        if not gate.passed:
            raise ConfigError(
                "an adopt decision requires a passing release gate",
                remediation=(
                    "Fix the gate failures, or record reject/inconclusive for "
                    "this comparison."
                ),
            )
        if not gate_enforces_release_rule(gate):
            raise ConfigError(
                "an adopt decision requires a gate that enforced at least "
                "one substantive release rule",
                remediation=(
                    "Configure an absolute threshold, a positive cost-coverage "
                    "minimum, or a regression rule with its baseline metric."
                ),
            )
    return parsed.value


def _comparison_rows(
    gate: GateResult,
    baseline_metrics: Mapping[str, float],
    policy: Any,
) -> tuple[ComparisonRow, ...]:
    thresholds = {rule.metric: rule for rule in policy.rules}
    failed = {failure.metric for failure in gate.failures}
    rows = []
    for metric in sorted(
        metric
        for metric in gate.metrics
        if not is_statistics_metric(metric)
        and not integrity_module.is_integrity_metric(metric)
    ):
        current = gate.metrics[metric]
        reference = baseline_metrics.get(metric)
        rule = thresholds.get(metric)
        if metric in failed:
            verdict = "fail"
        elif rule is None:
            verdict = "report-only"
        else:
            verdict = "pass"
        rows.append(
            ComparisonRow(
                metric=metric,
                current=current,
                baseline=reference,
                delta=None if reference is None else current - reference,
                threshold=_threshold_text(rule),
                verdict=verdict,
            )
        )
    return tuple(rows)


def _missing_judge_metric_warnings(gate: GateResult, plan: ScorerPlan) -> list[str]:
    """Explain a judge that ran but produced nothing.

    MLflow reports a failed judge as an absent metric, so the gate says
    only "metric is missing". That reads like a configuration mistake when
    it is usually an unreachable judge endpoint.
    """

    judged_metrics = {spec.metric: spec.name for spec in plan.judge_specs}
    missing = sorted(
        judged_metrics[failure.metric]
        for failure in gate.failures
        if failure.reason == "metric is missing" and failure.metric in judged_metrics
    )
    if not missing:
        return []
    return [
        f"the judge scorer(s) {', '.join(missing)} produced no metric. The "
        "judge call most likely failed - check that the judge endpoint in "
        "aai-platform.yml exists and that you are authenticated to the "
        "workspace (`az login` with DATABRICKS_AUTH_TYPE=azure-cli)"
    ]


def _threshold_text(rule: Any) -> str | None:
    if rule is None:
        return None
    parts = []
    if rule.required is not None:
        comparison = ">=" if rule.direction.value == "higher" else "<="
        parts.append(f"{comparison}{rule.required:g}")
    if rule.max_regression is not None:
        parts.append(f"drop<={rule.max_regression:g}")
    return " ".join(parts) or None


def _render_outcome(
    results: ResultsRecord,
    comparison: Sequence[ComparisonRow],
    warnings: Sequence[str],
    scope: BaselineScope,
) -> list[str]:
    lines: list[str] = [""]
    if results.established_baseline:
        lines.append(
            f"Baseline established ({scope.mode} run over {scope.rows} rows). "
            "This run IS the baseline - the next `agentkit compare` scores "
            "against it."
        )
        if not results.gate_passed:
            lines.append(
                "Note: this baseline did NOT pass the gate. It records where "
                "you are today, not an approved release."
            )
    elif results.is_comparison:
        reference = results.baseline_run_id or "the recorded baseline"
        lines.append(f"Compared against {reference}.")
    else:
        lines.append(
            "No baseline yet: absolute scores only. Run "
            "`agentkit compare --establish-baseline` to start comparing."
        )
    for warning in warnings:
        lines.append(f"warning: {warning}")

    header = ("metric", "current", "baseline", "delta", "threshold", "verdict")
    table = [header]
    for row in comparison:
        table.append(
            (
                row.metric,
                f"{row.current:.4g}",
                "-" if row.baseline is None else f"{row.baseline:.4g}",
                "-" if row.delta is None else f"{row.delta:+.4g}",
                row.threshold or "-",
                row.verdict,
            )
        )
    widths = [max(len(item[index]) for item in table) for index in range(len(header))]
    lines.extend(
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(item)).rstrip()
        for item in table
    )
    lines.append("")
    if results.statistics is not None and results.statistics.estimates:
        level = results.statistics.confidence_level * 100
        lines.append(
            f"uncertainty: {level:g}% normal-mean intervals; minimum "
            f"enforceable sample {results.statistics.minimum_cases}"
        )
        for estimate in results.statistics.estimates:
            lines.append(
                f"  {estimate.metric}: n={estimate.sample_size}, "
                f"CI [{estimate.lower:.4g}, {estimate.upper:.4g}]"
            )
        for paired in results.statistics.paired:
            lines.append(
                f"  {paired.metric}: paired n={paired.pair_count}, "
                "improvement CI "
                f"[{paired.lower_improvement:+.4g}, "
                f"{paired.upper_improvement:+.4g}]"
            )
        lines.append("")
    lines.extend(_economics_outcome_lines(results))
    lines.extend(_integrity_outcome_lines(results))
    lines.append("gate: PASSED" if results.gate_passed else "gate: FAILED")
    for failure in results.gate_failures:
        lines.append(f"  FAIL {failure['metric']}: {failure['reason']}")
    lines.append(f"decision recorded: {results.decision}")
    if results.run_id:
        lines.append(f"MLflow run: {results.run_id} in {results.experiment_name}")
    return lines


def _economics_outcome_lines(results: ResultsRecord) -> list[str]:
    economics = results.economics
    if economics is None:
        return []
    lines = [
        "run economics:",
        (
            f"  {economics.successes}/{economics.rows} successful "
            f"completion(s); cost known for {economics.cost_known}/"
            f"{economics.rows} row(s), tokens for "
            f"{economics.tokens_known}/{economics.rows} "
            f"(cost source: {economics.cost_source})"
        ),
    ]
    for segment in economics.segments:
        parts = [
            f"  {segment.key}={segment.value or '(unset)'}: "
            f"{segment.successes}/{segment.rows} ok"
        ]
        if segment.cost_per_success_usd is not None:
            parts.append(f"cost/success ${segment.cost_per_success_usd:.4g}")
        if segment.cost_p95_usd is not None:
            parts.append(f"cost p95 ${segment.cost_p95_usd:.4g}")
        if segment.latency_p95_seconds is not None:
            parts.append(f"latency p95 {segment.latency_p95_seconds:.4g}s")
        if segment.llm_calls_p95 is not None:
            parts.append(f"llm calls p95 {segment.llm_calls_p95:.4g}")
        lines.append(", ".join(parts))
    lines.append("")
    return lines


def _integrity_outcome_lines(results: ResultsRecord) -> list[str]:
    if results.integrity is None:
        return []
    consistency = results.integrity.consistency
    drift = results.integrity.anchor_drift
    lines = ["judge integrity:"]
    if consistency is not None:
        lines.append(
            f"  self-inconsistency {consistency.overall:.3f} over "
            f"{consistency.sample_size} re-scored row(s)"
        )
    if drift is not None:
        lines.append(
            f"  anchor drift {drift.overall:.3f} over {drift.rows} "
            f"frozen row(s) ({drift.anchors_ref})"
        )
    lines.append("")
    return lines


class _PromptLoader:
    """Resolve each governed prompt once for provenance and construction."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._resolved: dict[tuple[str, str], Any | None] = {}

    def __call__(self, name: str, alias: str) -> Any:
        key = (name, alias)
        if key not in self._resolved:
            try:
                self._resolved[key] = self._manager.load(name, alias=alias)
            except Exception as error:
                if not is_missing_prompt_error(error):
                    raise
                # A judge prompt that is not registered yet falls back to the
                # catalog's bundled instructions. Cache that fallback too: a
                # retry must not make provenance and execution disagree.
                self._resolved[key] = None
        return self._resolved[key]

    def version_uri(self, name: str, alias: str) -> str | None:
        prompt = self(name, alias)
        uri = getattr(prompt, "uri", None)
        version = getattr(prompt, "version", None)
        if uri:
            return str(uri)
        if version is not None:
            return f"prompts:/{self._manager.qualify(name)}/{version}"
        return None


def _prompt_loader(project: ProjectContext, mlflow: Any) -> _PromptLoader:
    return _PromptLoader(project.prompt_manager(mlflow_module=mlflow))


def _resolved_prompt_versions(
    plan: ScorerPlan, prompt_loader: _PromptLoader
) -> dict[str, str]:
    versions: dict[str, str] = {}
    for spec in plan.specs:
        binding = spec.judge
        if binding is None or not binding.prompt_name:
            continue
        uri = prompt_loader.version_uri(binding.prompt_name, binding.prompt_alias)
        if uri:
            versions[spec.name] = uri
    return versions


def _run_identity(active_run: Any) -> tuple[str | None, str | None]:
    info = getattr(active_run, "info", None)
    run_id = getattr(info, "run_id", None)
    experiment_id = getattr(info, "experiment_id", None)
    return (
        str(run_id) if run_id is not None else None,
        str(experiment_id) if experiment_id is not None else None,
    )


def _record_reproducibility(mlflow: Any) -> None:
    from aai_core.experiments import record_reproducibility

    # Reproducibility metadata is evidence, not a gate: a run must not fail
    # because the package freeze could not be written.
    with suppress(Exception):
        record_reproducibility(mlflow_module=mlflow)


def _change_id() -> str:
    from_env = os.getenv("GIT_COMMIT")
    if from_env:
        return from_env[:12]
    # Job clusters carry the deployed commit as AAI_RELEASE (the bundle's
    # `deployment_release`, set to the CI SHA), and no GIT_COMMIT or .git
    # directory. Without this fallback every release-gate run records
    # "local-dev" — exactly the run whose commit identity matters most.
    release = _release_value()
    if release is not None:
        return release[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or "local-dev"
    except Exception:
        return "local-dev"


def _release_value() -> str | None:
    """The deployed-commit identity this process runs under, if any."""

    value = (os.getenv("AAI_RELEASE") or "").strip()
    if value and value != "local-dev":
        return value
    return None


def _version() -> str:
    from aai_core import __version__

    return __version__


def _mlflow(mlflow_module: Any | None) -> Any:
    if mlflow_module is not None:
        return mlflow_module
    try:
        import mlflow
    except ImportError as error:
        raise missing_extra("Recording an evaluation run", "genai") from error
    return mlflow


def _is_locally_scorable(plan: ScorerPlan, mode: str) -> bool:
    """True when every selected scorer is a pure function over recorded rows."""

    if mode != "answer-sheet":
        return False
    return all(
        entry.spec.kind is ScorerKind.CODE
        and entry.spec.name in catalog_module.CODE_SCORER_FUNCTIONS
        for entry in plan.entries
    )


def _score_locally(
    dataset: LoadedDataset, plan: ScorerPlan
) -> tuple[dict[str, float], dict[str, tuple[float | None, ...]]]:
    """Score recorded answers in-process — no MLflow, no cloud, no cost."""

    samples: dict[str, list[float | None]] = {
        entry.spec.metric: [] for entry in plan.entries
    }
    for row in dataset.rows:
        outputs = catalog_module._require_output_text(row.get("outputs"))
        expectations = dict(row.get("expectations") or {})
        for entry in plan.entries:
            function = catalog_module.CODE_SCORER_FUNCTIONS[entry.spec.name]
            samples[entry.spec.metric].append(float(function(outputs, expectations)))
    frozen = {metric: tuple(values) for metric, values in samples.items()}
    metrics = {
        metric: fmean(value for value in values if value is not None)
        for metric, values in frozen.items()
        if any(value is not None for value in values)
    }
    return metrics, frozen
