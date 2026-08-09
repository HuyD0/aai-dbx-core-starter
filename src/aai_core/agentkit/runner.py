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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from aai_core.agentkit import catalog as catalog_module
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
    STORED_TRACE_MODES,
    LoadedDataset,
    attach_answer_sheet,
    evaluation_rows,
    load_dataset,
    smoke_sample,
    trace_expectation_overrides,
    validate_dataset,
)
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
from aai_core.agentkit.results import ResultsRecord, publish_results, write_results
from aai_core.agentkit.targets import TargetKind, build_predict_fn, resolve_target
from aai_core.evaluation import GateResult, apply_gate
from aai_core.experiments import ExperimentRunMetadata, RunPurpose

WORKERS_ENV = "MLFLOW_GENAI_EVAL_MAX_WORKERS"
SCORER_WORKERS_ENV = "MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"
DEFAULT_ANSWER_SHEET = "evals/data/answer_sheet.json"
_SMOKE_JUDGE_NOTE = (
    "smoke runs deterministic code scorers only so it stays free and "
    "credential-free; use `agentkit smoke --live` or `agentkit compare` to "
    "run judges"
)


class Decision(StrEnum):
    """What the comparison concluded. Never 'candidate' (deprecated)."""

    ADOPT = "adopt"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


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

    config = project.config
    dataset = load_dataset(
        config.dataset, root=project.root, mlflow_module=mlflow_module
    )
    structural = validate_dataset(dataset)
    if structural:
        raise ConfigError(
            "the evaluation dataset is not ready:\n"
            + "\n".join(f"  - {failure}" for failure in structural),
            remediation="Fix the dataset rows, then run the command again.",
        )

    # An explicit target overrides the configured one. The deployment-job
    # gate needs this: it must score the model version that triggered it,
    # not whatever `agent:` happened to be committed.
    target = resolve_target(
        agent or config.agent, root=project.root, settings=project.settings
    )
    resolved_mode = mode or _default_mode(target, dataset)
    # Before the estimate and before any judge call: a traces run supplies
    # no predict_fn, so a row without a trace has no answer to score at
    # all. Choosing the mode by default already avoids this; asking for it
    # explicitly must fail rather than spend.
    if resolved_mode == "traces" and not dataset.shape.has_traces:
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
    mode_warnings = _mode_warnings(resolved_mode, dataset, explicit=mode is not None)
    full_row_count = dataset.shape.row_count
    if rows_limit:
        dataset = smoke_sample(dataset, rows_limit, strata=config.strata)
    # A "sample" that covered every row is a full run: saying otherwise
    # would raise a scope-drift warning on every later comparison.
    sampled = dataset.shape.row_count < full_row_count
    if resolved_mode == "answer-sheet":
        dataset = attach_answer_sheet(dataset, _answer_sheet_path(project, target))

    judge_model_uri = None
    judge_model_identity = None
    judge_note = None if judges_enabled else _SMOKE_JUDGE_NOTE
    if judges_enabled:
        judge_model_uri = project.judge_model_uri()
        # Best effort: an endpoint name is stable while the model behind
        # it is not, so pin what it currently serves when the workspace
        # will say. A least-privilege CI principal may hold CAN_QUERY
        # without CAN_VIEW, and widening that grant to make a check work
        # is exactly what section 4 of AGENTS.md forbids.
        judge_model_identity = project.judge_model_identity()
    plan = select_scorers(
        dataset.shape,
        config,
        mode=resolved_mode,
        judges_enabled=judges_enabled,
        judge_note=judge_note,
    )
    cost = estimate(
        dataset.rows,
        plan,
        price_per_1m_tokens=config.budget.judge_price_per_1m_tokens,
        chunks_per_row=config.budget.retrieved_chunks_per_row,
    )
    outcome = RunOutcome(plan=plan, cost=cost, dataset=dataset, plan_only=plan_only)
    outcome.messages.append(
        f"Inferred evaluation plan  (dataset: {dataset.ref}, "
        f"{dataset.shape.row_count} rows, digest {dataset.digest})"
    )
    outcome.messages.append(render_plan(plan, judge_model_uri=judge_model_uri))
    outcome.messages.append(render_cost(cost))
    if plan_only:
        return outcome, EXIT_PASS

    # The baseline is settled BEFORE the budget check and the confirmation
    # prompt. Discovering that the comparison was never valid after paying
    # for the judge calls would make the refusal worthless.
    baseline: BaselineRecord | None = None
    warnings: list[str] = list(mode_warnings)
    if establish_baseline:
        existing, _ = load_baseline(project.baseline_path)
        if existing is not None:
            warnings.append(
                f"replacing the baseline recorded at {existing.recorded_at}"
            )
    else:
        try:
            baseline, baseline_warnings = select_baseline(
                baseline_path=project.baseline_path,
                flag_run_id=baseline_run_id,
                config_run_id=config.baseline.run_id,
                mlflow_module=mlflow_module,
            )
        except BaselineMissingError:
            # `compare` and `eval` are promotion-grade: scoring into a
            # vacuum is an incomplete submission, so the refusal stands.
            # `smoke` is a threshold gate that has to work on a project
            # generated five minutes ago, before any baseline exists.
            if require_baseline:
                raise
            baseline, baseline_warnings = None, [
                "no baseline recorded yet, so this run reports absolute "
                "scores only; run `agentkit compare --establish-baseline` "
                "to start comparing"
            ]
        warnings.extend(baseline_warnings)
        if baseline is not None:
            warnings.extend(
                drift_warnings(
                    baseline,
                    dataset=dataset,
                    mode="sample" if sampled else "full",
                    rows=dataset.shape.row_count,
                )
            )
            comparability, comparable = _enforce_comparability(
                baseline,
                dataset=dataset,
                mode="sample" if sampled else "full",
                rows=dataset.shape.row_count,
                plan=plan,
                judge_model=judge_model_uri,
                judge_model_identity=judge_model_identity,
                judges_enabled=judges_enabled,
                allow_drift=allow_baseline_drift,
                blocking=require_baseline,
            )
            warnings.extend(comparability)
            if baseline.versions.judge_model_identity and not judge_model_identity:
                # Silence here would read as "the judge is unchanged".
                warnings.append(
                    "the baseline was judged by "
                    f"{baseline.versions.judge_model_identity}, but what the "
                    "endpoint serves now could not be read, so a change "
                    "behind the same endpoint name is unverified"
                )
            if not comparable:
                baseline = None

    enforce_budget(cost, max_judge_calls=config.budget.max_judge_calls)
    if cost.judge_calls and not assume_yes:
        if confirm is None or not confirm("Proceed?"):
            # Nothing was scored, so this cannot be a pass. The usual cause
            # is a CI job on a non-interactive stream with no --yes, and
            # exit 0 there would report success for an evaluation that
            # never happened.
            outcome.declined = True
            outcome.messages.append(
                "Cancelled - nothing was scored. Pass --yes to run without "
                "the confirmation prompt."
            )
            return outcome, EXIT_ERROR

    set_concurrency_env(config.concurrency, environ)
    # A code-scorer-only run over recorded answers needs nothing from
    # MLflow, so it does not open a run. That is deliberate on two counts:
    # it keeps `agentkit smoke` runnable on every commit with no
    # credentials and no tracking backend, and it keeps an afternoon of
    # throwaway smoke runs out of the experiment. Recorded comparisons are
    # what `compare` and `eval` are for.
    scored_locally = _is_locally_scorable(plan, resolved_mode)
    mlflow = None if scored_locally else _mlflow(mlflow_module)
    if scored_locally:
        warnings.append(
            "scored locally: a code-scorer-only run does not open an MLflow "
            "run. Use `agentkit compare` to record the comparison."
        )
    # Prompt versions resolve only once MLflow is in hand, so this is the
    # earliest the check can run — still before the run opens and before a
    # single judge call is paid for. A judge prompt whose alias has moved
    # is a different judge, and reporting that alongside the results would
    # be reporting it too late.
    judge_prompts: dict[str, str] = {}
    if judges_enabled and mlflow is not None:
        judge_prompts = _resolved_prompt_versions(plan, project, mlflow)
        if baseline is not None:
            prompt_drift, comparable = _enforce_comparability(
                baseline,
                dataset=dataset,
                mode="sample" if sampled else "full",
                rows=dataset.shape.row_count,
                plan=plan,
                judge_model=judge_model_uri,
                judge_prompts=judge_prompts,
                judges_enabled=judges_enabled,
                allow_drift=allow_baseline_drift,
                blocking=require_baseline,
                only_prompts=True,
            )
            warnings.extend(prompt_drift)
            if not comparable:
                baseline = None

    change_id = _change_id()
    purpose = RunPurpose.BASELINE if establish_baseline else RunPurpose.RESULT
    summary = (
        "Recording the first scored version as the baseline"
        if establish_baseline
        else f"Comparing {target.normalized} against the recorded baseline"
    )
    metadata = ExperimentRunMetadata(
        purpose=purpose,
        change_id=change_id,
        change_summary=summary,
        baseline_run_id=baseline.run_id if baseline else None,
    )
    recorded_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    baseline_metrics = dict(baseline.metrics) if baseline else {}
    policy = build_policy(
        project,
        plan=plan,
        allow_missing_regression_baseline=establish_baseline or not baseline_metrics,
    )

    if mlflow is None:
        gate = apply_gate(
            _score_locally(dataset, plan),
            policy=policy,
            baseline_metrics=baseline_metrics,
        )
        recorded_decision = _decision_value(decision, gate, establish_baseline)
    else:
        predict_fn = (
            build_predict_fn(
                target, project=project, transport=transport, mlflow_module=mlflow
            )
            if resolved_mode == "live"
            else None
        )
        prompt_loader = _prompt_loader(project, mlflow) if judges_enabled else None
        scorers = [
            catalog_module.build_scorer(
                entry.spec,
                judge_model_uri=judge_model_uri,
                guidelines=config.scorers.guidelines,
                prompt_loader=prompt_loader,
                mlflow_module=mlflow,
            )
            for entry in plan.entries
        ]
        manager = project.experiment_manager(mlflow_module=mlflow)
        experiment_name = manager.experiment_name
        with manager.run(
            run_name=f"{command}-{resolved_mode}",
            description=summary,
            parameters={
                "mode": resolved_mode,
                "dataset": dataset.ref,
                "row_count": dataset.shape.row_count,
            },
            metadata=metadata,
        ) as active_run:
            run_id, experiment_id = _run_identity(active_run)
            native_result = mlflow.genai.evaluate(
                data=evaluation_rows(dataset, mode=resolved_mode),
                scorers=scorers,
                predict_fn=predict_fn,
            )
            tags = {
                "aai.agentkit_version": _version(),
                "aai.scorer_versions": plan.scorer_versions_tag(),
                "aai.dataset": dataset.ref,
                "aai.dataset_digest": dataset.digest,
                "aai.dataset_rows": str(dataset.shape.row_count),
                # The scope travels with the run so a baseline fetched by
                # run id knows whether it scored a sample or everything.
                "aai.scope_mode": "sample" if sampled else "full",
                "aai.scope_rows": str(dataset.shape.row_count),
                "aai.agent_target": target.normalized,
                "aai.recorded_at": recorded_at,
            }
            if judge_model_uri:
                tags["aai.judge_model"] = judge_model_uri
            if judge_model_identity:
                tags["aai.judge_model_identity"] = judge_model_identity
            if judge_prompts:
                tags["aai.judge_prompt_versions"] = ",".join(
                    f"{name}={uri}" for name, uri in sorted(judge_prompts.items())
                )
            warnings.extend(_coverage_warnings(native_result))
            gate = apply_gate(
                _metrics_with_scorer_errors(native_result),
                policy=policy,
                baseline_metrics=baseline_metrics,
            )
            recorded_decision = _decision_value(decision, gate, establish_baseline)
            tags["aai.gate_passed"] = str(gate.passed).lower()
            tags["aai.decision"] = recorded_decision
            mlflow.log_metrics(dict(gate.metrics))
            mlflow.set_tags(tags)
            _record_reproducibility(mlflow)

    comparison = _comparison_rows(gate, baseline_metrics, policy)
    warnings.extend(_missing_judge_metric_warnings(gate, plan))
    versions = BaselineVersions(
        agent=target.normalized,
        scorers={spec.name: spec.version for spec in plan.specs},
        judge_model=judge_model_uri,
        judge_model_identity=judge_model_identity,
        judge_prompts=judge_prompts,
        aai_core=_version(),
    )
    scope = BaselineScope(
        mode="sample" if sampled else "full",
        rows=dataset.shape.row_count,
        seed=None,
    )
    results = ResultsRecord(
        command=command,
        recorded_at=recorded_at,
        run_id=run_id,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        agent=target.normalized,
        dataset=BaselineDataset(
            ref=dataset.ref, digest=dataset.digest, rows=dataset.shape.row_count
        ),
        scope=scope,
        mode=resolved_mode,
        metrics=dict(gate.metrics),
        versions=versions,
        baseline_run_id=baseline.run_id if baseline else None,
        baseline_metrics=baseline_metrics,
        baseline_recorded_at=baseline.recorded_at if baseline else None,
        baseline_dataset_digest=baseline.dataset.digest if baseline else None,
        established_baseline=establish_baseline,
        policy_rules=policy.rules,
        allow_missing_regression_baseline=policy.allow_missing_regression_baseline,
        decision=recorded_decision,
        change_id=change_id,
        gate_passed=gate.passed,
        gate_failures=tuple(
            {"metric": failure.metric, "reason": failure.reason}
            for failure in gate.failures
        ),
        warnings=tuple(warnings),
        judges_enabled=judges_enabled,
    )
    results_path = write_results(project.results_dir, results)
    if mlflow is not None and run_id:
        # `.aai/agentkit/results/` is the filesystem this run happened on.
        # For the deployment-job gate that is a job cluster the approver
        # cannot reach, so the record travels with the run.
        failure = publish_results(mlflow, run_id, results_path)
        if failure:
            # This used to be a warning, on the reasoning that the record
            # was already on disk and the gate already decided. That
            # reasoning does not survive the deployment-job gate: there the
            # disk is an ephemeral job cluster, the run is the only durable
            # copy, and the approval task would otherwise proceed with
            # evidence nobody can retrieve. A scored run whose evidence is
            # unreachable is not promotion evidence, so it fails closed.
            raise EvidenceMissingError(
                f"{failure}\nThe run was scored (gate "
                f"{'passed' if gate.passed else 'FAILED'}) but its results "
                "record could not be attached, so `agentkit evidence --run "
                f"{run_id}` would find nothing.",
                remediation=(
                    "Check MLflow artifact permissions for this experiment, "
                    "then run the evaluation again."
                ),
            )
        outcome.messages.append(
            f"Evidence for this run: agentkit evidence --run {run_id}"
        )

    if establish_baseline:
        write_baseline(
            project.baseline_path,
            BaselineRecord(
                schema_version=1,
                run_id=run_id,
                experiment_id=experiment_id,
                recorded_at=recorded_at,
                dataset=results.dataset,
                scope=scope,
                metrics=dict(gate.metrics),
                versions=versions,
                recorded_by=f"agentkit {command} --establish-baseline",
                change_id=change_id,
            ),
        )

    outcome.results = results
    outcome.gate = gate
    outcome.comparison = comparison
    outcome.established_baseline = establish_baseline
    outcome.results_path = results_path
    outcome.warnings = tuple(warnings)
    outcome.messages.extend(_render_outcome(results, comparison, warnings, scope))
    return outcome, (EXIT_PASS if gate.passed else EXIT_THRESHOLD_FAILED)


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

    if value is None:
        return False
    if isinstance(value, float) and value != value:
        return False
    return True


def _is_reported_error(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and value != value:  # NaN
        return False
    return bool(str(value).strip())


def _default_mode(target: Any, dataset: LoadedDataset) -> str:
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
    if mode in STORED_TRACE_MODES:
        overridden = trace_expectation_overrides(dataset)
        if overridden:
            # MLflow rewrites the whole expectations column from the traces'
            # assessments, despite documenting itself as filling it only when
            # absent. In this mode that is its behaviour and may be wanted —
            # but a silent change to what "correct" means is not something a
            # run should leave unsaid.
            named = ", ".join(overridden)
            warnings.append(
                f"the rows' traces carry expectation assessments ({named}) "
                "and the dataset also supplies expectations; MLflow scores "
                "the trace's assessments, not the dataset's."
            )
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


def _answer_sheet_path(project: ProjectContext, target: Any) -> Path:
    configured = project.config.smoke.answer_sheet
    if configured:
        return project.root / configured
    if target.kind is TargetKind.ANSWER_SHEET and target.path is not None:
        return target.path
    return project.root / DEFAULT_ANSWER_SHEET


def _decision_value(
    decision: str | None, gate: GateResult, establish_baseline: bool
) -> str:
    if decision:
        return Decision(decision).value
    return Decision.INCONCLUSIVE.value


def _comparison_rows(
    gate: GateResult,
    baseline_metrics: Mapping[str, float],
    policy: Any,
) -> tuple[ComparisonRow, ...]:
    thresholds = {rule.metric: rule for rule in policy.rules}
    failed = {failure.metric for failure in gate.failures}
    rows = []
    for metric in sorted(gate.metrics):
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
    lines.append("gate: PASSED" if results.gate_passed else "gate: FAILED")
    for failure in results.gate_failures:
        lines.append(f"  FAIL {failure['metric']}: {failure['reason']}")
    lines.append(f"decision recorded: {results.decision}")
    if results.run_id:
        lines.append(f"MLflow run: {results.run_id} in {results.experiment_name}")
    return lines


def _prompt_loader(project: ProjectContext, mlflow: Any):
    manager = project.prompt_manager(mlflow_module=mlflow)

    def load(name: str, alias: str) -> Any:
        try:
            return manager.load(name, alias=alias)
        except Exception:
            # A judge prompt that is not registered yet falls back to the
            # catalog's bundled instructions; the run records which was used.
            return None

    return load


def _resolved_prompt_versions(
    plan: ScorerPlan, project: ProjectContext, mlflow: Any
) -> dict[str, str]:
    versions: dict[str, str] = {}
    manager = None
    for spec in plan.specs:
        binding = spec.judge
        if binding is None or not binding.prompt_name:
            continue
        if manager is None:
            manager = project.prompt_manager(mlflow_module=mlflow)
        try:
            prompt = manager.load(binding.prompt_name, alias=binding.prompt_alias)
        except Exception:
            continue
        uri = getattr(prompt, "uri", None)
        version = getattr(prompt, "version", None)
        if uri:
            versions[spec.name] = str(uri)
        elif version is not None:
            versions[spec.name] = (
                f"prompts:/{manager.qualify(binding.prompt_name)}/{version}"
            )
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

    try:
        record_reproducibility(mlflow_module=mlflow)
    except Exception:
        # Reproducibility metadata is evidence, not a gate: a run must not
        # fail because the package freeze could not be written.
        pass


def _change_id() -> str:
    from_env = os.getenv("GIT_COMMIT")
    if from_env:
        return from_env[:12]
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


def _score_locally(dataset: LoadedDataset, plan: ScorerPlan) -> dict[str, float]:
    """Score recorded answers in-process — no MLflow, no cloud, no cost."""

    totals: dict[str, float] = {}
    for row in dataset.rows:
        outputs = str(row.get("outputs", ""))
        expectations = dict(row.get("expectations") or {})
        for entry in plan.entries:
            function = catalog_module.CODE_SCORER_FUNCTIONS[entry.spec.name]
            totals[entry.spec.metric] = totals.get(entry.spec.metric, 0.0) + function(
                outputs, expectations
            )
    row_count = len(dataset.rows) or 1
    return {metric: value / row_count for metric, value in totals.items()}
