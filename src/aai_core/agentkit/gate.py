"""The promotion gate and the CI exit-code contract.

Three exit codes, documented and stable:

    0  every threshold passed
    2  ran successfully, one or more thresholds failed  (CI hard fail)
    1  runtime or configuration error

The gate refuses an empty answer to "what did you compare against": a
single run with no named baseline is an incomplete submission, not a pass.
It also fails closed when a thresholded metric is missing from the results
— an evaluation that did not produce the evidence has not earned a pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from aai_core.agentkit.baseline import BaselineRecord, load_baseline
from aai_core.agentkit.catalog import (
    ScorerPlan,
    effective_threshold,
    get_spec,
    registry_direction,
)
from aai_core.agentkit.config import ProjectContext, parse_threshold
from aai_core.agentkit.errors import ConfigError, UnknownScorerError
from aai_core.agentkit.results import ResultsRecord, load_gate_results
from aai_core.agentkit.statistics import extend_rules_with_statistics
from aai_core.evaluation import (
    GateFailure,
    GatePolicy,
    GateResult,
    MetricDirection,
    MetricRule,
    apply_gate,
)

EXIT_PASS = 0
EXIT_ERROR = 1
EXIT_THRESHOLD_FAILED = 2

NO_RESULTS_MESSAGE = (
    "No evaluation results found. The gate reads the evidence a scoring run "
    "produced; it never scores by itself.\nRun:\n"
    "    agentkit compare\nthen run `agentkit gate` again."
)
NOT_A_COMPARISON_MESSAGE = (
    "These results are not a comparison - no baseline was named.\n"
    "Promotion evidence must say what the change was scored against. Run:\n"
    "    agentkit compare\n"
    "(or `agentkit compare --establish-baseline` if this really is the "
    "first version)."
)


@dataclass(frozen=True)
class GateReport:
    result: GateResult
    results: ResultsRecord
    baseline: BaselineRecord | None
    rules: tuple[MetricRule, ...]
    message: str | None = None
    # Set only when the rules did not come from the record itself.
    policy_note: str | None = None

    @property
    def passed(self) -> bool:
        return self.result.passed


def build_policy(
    project: ProjectContext,
    *,
    plan: ScorerPlan | None = None,
    scorer_names: tuple[str, ...] = (),
    allow_missing_regression_baseline: bool = False,
) -> GatePolicy:
    """Compose the gate policy from config thresholds and catalog defaults.

    Thresholds in ``agentkit.yaml`` win; otherwise each scorer that ran
    contributes its registry default. The scorers come from the live plan
    when scoring, or from the names the results record captured when
    gating an earlier run. ``regression_budget`` adds the maximum
    tolerated drop against the baseline.
    """

    config = project.config
    rules: dict[str, MetricRule] = {}
    specs = list(plan.specs) if plan is not None else []
    seen = {spec.name for spec in specs}
    for name in scorer_names:
        if name in seen:
            continue
        try:
            specs.append(get_spec(name))
        except UnknownScorerError:
            # A record written by a newer catalog: its thresholds still
            # apply if the project configured them explicitly.
            continue
    for spec in specs:
        expression = effective_threshold(spec, config)
        if expression is not None:
            rules[spec.metric] = parse_threshold(spec.metric, expression)
    for key, expression in config.thresholds.items():
        # Every configured threshold always applies, including one whose
        # metric never appeared: a run that did not produce the evidence
        # has not earned a pass.
        metric = _metric_for(key)
        rules[metric] = parse_threshold(metric, str(expression))
    for key, allowance in config.regression_budget.items():
        metric = _metric_for(key)
        existing = rules.get(metric)
        if existing is not None:
            rules[metric] = existing.model_copy(
                update={"max_regression": float(allowance)}
            )
        else:
            # With no absolute threshold to imply a direction, take it from
            # the registry: latency is lower-is-better, so "regression"
            # must mean slower, not faster.
            rules[metric] = MetricRule(
                metric=metric,
                direction=MetricDirection(registry_direction(metric)),
                max_regression=float(allowance),
            )
    minimum_effect = {
        _metric_for(name): value
        for name, value in config.statistics.minimum_effect.items()
    }
    unknown_effects = sorted(set(minimum_effect).difference(rules))
    if unknown_effects:
        raise ConfigError(
            "statistics.minimum_effect names metrics without a governed "
            "threshold or regression rule: " + ", ".join(unknown_effects),
            remediation=(
                "Add a threshold or regression_budget entry for each metric, "
                "or remove its minimum_effect requirement."
            ),
        )
    statistical_config = config.statistics.model_copy(
        update={"minimum_effect": minimum_effect}
    )
    governed_rules = extend_rules_with_statistics(
        tuple(rules[name] for name in sorted(rules)),
        statistical_config,
        allow_missing_regression_baseline=allow_missing_regression_baseline,
    )
    return GatePolicy(
        rules=governed_rules,
        allow_missing_regression_baseline=allow_missing_regression_baseline,
    )


def evaluate_gate(
    project: ProjectContext,
    *,
    results: ResultsRecord,
    baseline: BaselineRecord | None,
    plan: ScorerPlan | None = None,
    check_policy_drift: bool = False,
) -> tuple[GateReport, int]:
    """Apply the policy to an existing results record.

    ``check_policy_drift`` belongs to the promotion check: gating a record
    whose rules no longer match the project's configuration means the
    evidence is stale, and the answer is to re-score rather than to judge
    old numbers by new rules. Evidence rendering leaves it off — a run
    fetched from another machine is not expected to match this checkout.
    """

    if not results.is_comparison:
        # The refusal has to be a failure, not just an exit code: the JSON
        # output and the evidence pack both read the verdict off this
        # result, and a promotion record must never say PASSED for a run
        # that never named what it was compared against.
        refused = GateResult(
            metrics=dict(results.metrics),
            failures=(
                GateFailure(
                    metric="comparison",
                    reason=(
                        "no baseline was named, so these results are not a "
                        "comparison"
                    ),
                ),
            ),
        )
        report = GateReport(
            result=refused,
            results=results,
            baseline=baseline,
            rules=(),
            message=NOT_A_COMPARISON_MESSAGE,
        )
        return report, EXIT_THRESHOLD_FAILED
    baseline_metrics: Mapping[str, float] = (
        dict(results.baseline_metrics)
        if results.baseline_metrics
        else dict(baseline.metrics) if baseline is not None else {}
    )
    # The rules the run was judged by travel with the run. Re-deriving them
    # from the current agentkit.yaml would mean a relaxed threshold turns a
    # failed run into approved evidence with nothing re-scored, and a
    # record fetched with `--run` would be judged by whatever config the
    # reader happens to have checked out.
    note: str | None = None
    if results.policy_rules:
        policy = GatePolicy(
            rules=results.policy_rules,
            allow_missing_regression_baseline=(
                results.allow_missing_regression_baseline
            ),
        )
        drift = _policy_drift(project, results, plan) if check_policy_drift else None
        if drift is not None:
            # Neither applying the new rules to old numbers nor quietly
            # ignoring them is honest. Refuse, and name what changed.
            refused = GateResult(
                metrics=dict(results.metrics),
                failures=(GateFailure(metric="policy", reason=drift),),
            )
            return (
                GateReport(
                    result=refused,
                    results=results,
                    baseline=baseline,
                    rules=policy.rules,
                    message=(
                        f"The gate rules changed after this run was scored: "
                        f"{drift}.\nThese results were judged by the rules in "
                        "force when they were produced, so the new rules "
                        "cannot be applied to them. Run:\n"
                        "    agentkit compare\nthen run `agentkit gate` again."
                    ),
                ),
                EXIT_THRESHOLD_FAILED,
            )
    else:
        policy = build_policy(
            project,
            plan=plan,
            scorer_names=tuple(results.versions.scorers),
            allow_missing_regression_baseline=results.established_baseline
            or not baseline_metrics,
        )
        note = (
            "these results predate recorded gate rules, so the current "
            "agentkit.yaml was applied; re-run `agentkit compare` for a "
            "record that carries its own policy"
        )
    result = apply_gate(
        dict(results.metrics), policy=policy, baseline_metrics=baseline_metrics
    )
    report = GateReport(
        result=result,
        results=results,
        baseline=baseline,
        rules=policy.rules,
        policy_note=note,
    )
    return report, (EXIT_PASS if result.passed else EXIT_THRESHOLD_FAILED)


def run_gate(
    project: ProjectContext,
    *,
    results_path: Path | None = None,
) -> tuple[GateReport | None, int, str | None]:
    """Gate the newest (or named) results record.

    Returns ``(report, exit_code, message)``; ``report`` is None only when
    there is nothing to gate.
    """

    from aai_core.agentkit.results import read_results

    if results_path is not None:
        if not results_path.is_file():
            return None, EXIT_ERROR, f"{results_path} does not exist"
        results = read_results(results_path)
    else:
        try:
            found = load_gate_results(project.results_dir)
        except ConfigError as error:
            return None, EXIT_ERROR, str(error)
        if found is None:
            return None, EXIT_ERROR, NO_RESULTS_MESSAGE
        results, _ = found
    baseline, _ = load_baseline(project.baseline_path)
    report, code = evaluate_gate(
        project, results=results, baseline=baseline, check_policy_drift=True
    )
    return report, code, report.message


def render_report(report: GateReport) -> str:
    """The gate verdict plus the evidence it rests on."""

    if report.message:
        return report.message
    results = report.results
    lines = [
        "gate: PASSED" if report.passed else "gate: FAILED",
        f"  agent            {results.agent}",
        f"  dataset          {results.dataset.ref} "
        f"(digest {results.dataset.digest}, {results.dataset.rows} rows)",
        f"  scope            {results.scope.mode}/{results.scope.rows} rows",
        f"  scorer versions  {_versions_text(results)}",
    ]
    if results.versions.judge_model:
        lines.append(f"  judge model      {results.versions.judge_model}")
    if results.versions.judge_prompts:
        prompts = ", ".join(
            f"{name}={version}"
            for name, version in sorted(results.versions.judge_prompts.items())
        )
        lines.append(f"  judge prompts    {prompts}")
    if results.established_baseline:
        lines.append("  compared against this run IS the recorded baseline")
    else:
        reference = results.baseline_run_id or "the recorded baseline file"
        lines.append(f"  compared against {reference}")
    lines.append(f"  decision         {results.decision}")
    if report.rules:
        source = "as recorded by the run" if not report.policy_note else "from config"
        lines.append(f"  thresholds ({source}):")
        for rule in report.rules:
            observed = results.metrics.get(rule.metric)
            observed_text = "missing" if observed is None else f"{observed:g}"
            lines.append(f"    {rule.metric}: {observed_text} " f"({_rule_text(rule)})")
    for failure in report.result.failures:
        lines.append(f"  FAIL {failure.metric}: {failure.reason}")
    if report.policy_note:
        lines.append(f"  note: {report.policy_note}")
    return "\n".join(lines)


def _policy_drift(
    project: ProjectContext,
    results: ResultsRecord,
    plan: ScorerPlan | None,
) -> str | None:
    """How the project's current rules differ from the recorded ones."""

    # The recorded scorers say what the run measured; `scorers.add` and
    # `scorers.remove` say what the project asks for *now*. Deriving the
    # live policy from the recorded names alone hides exactly the change
    # this check exists to catch: adding a thresholded catalog scorer
    # leaves its registry-default rule out of both sides, so a stale
    # record with no metric for it passes. Auto-selection is not folded
    # in — that depends on the dataset, which the recorded run already
    # reflects, and reading it here would make `agentkit gate` load data.
    selection = set(results.versions.scorers) | set(project.config.scorers.add)
    selection -= set(project.config.scorers.remove)
    current = build_policy(
        project,
        plan=plan,
        scorer_names=tuple(sorted(selection)),
        allow_missing_regression_baseline=(results.allow_missing_regression_baseline),
    )
    recorded = {rule.metric: rule for rule in results.policy_rules}
    live = {rule.metric: rule for rule in current.rules}
    added = sorted(set(live) - set(recorded))
    removed = sorted(set(recorded) - set(live))
    changed = sorted(
        metric
        for metric in set(recorded) & set(live)
        if recorded[metric] != live[metric]
    )
    parts = []
    if added:
        parts.append(f"added {', '.join(added)}")
    if removed:
        parts.append(f"removed {', '.join(removed)}")
    if changed:
        parts.append(f"changed {', '.join(changed)}")
    return "; ".join(parts) if parts else None


def _rule_text(rule: MetricRule) -> str:
    parts = []
    if rule.required is not None:
        comparison = ">=" if rule.direction.value == "higher" else "<="
        parts.append(f"needs {comparison}{rule.required:g}")
    if rule.max_regression is not None:
        parts.append(f"max regression {rule.max_regression:g}")
    return ", ".join(parts)


def _versions_text(results: ResultsRecord) -> str:
    if not results.versions.scorers:
        return "none recorded"
    return ", ".join(
        f"{name}=v{version}"
        for name, version in sorted(results.versions.scorers.items())
    )


def _metric_for(key: str) -> str:
    try:
        return get_spec(key).metric
    except UnknownScorerError:
        return key
