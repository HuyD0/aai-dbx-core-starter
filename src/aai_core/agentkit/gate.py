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
from aai_core.agentkit.catalog import ScorerPlan, effective_threshold, get_spec
from aai_core.agentkit.config import ProjectContext, parse_threshold
from aai_core.agentkit.errors import UnknownScorerError
from aai_core.agentkit.results import ResultsRecord, load_latest_results
from aai_core.evaluation import GatePolicy, GateResult, MetricRule, apply_gate

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
            rules[metric] = MetricRule(
                metric=metric,
                direction="higher",
                max_regression=float(allowance),
            )
    return GatePolicy(
        rules=tuple(rules[name] for name in sorted(rules)),
        allow_missing_regression_baseline=allow_missing_regression_baseline,
    )


def evaluate_gate(
    project: ProjectContext,
    *,
    results: ResultsRecord,
    baseline: BaselineRecord | None,
    plan: ScorerPlan | None = None,
) -> tuple[GateReport, int]:
    """Apply the policy to an existing results record."""

    if not results.is_comparison:
        empty = GateResult(metrics=dict(results.metrics), failures=())
        report = GateReport(
            result=empty,
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
    policy = build_policy(
        project,
        plan=plan,
        scorer_names=tuple(results.versions.scorers),
        allow_missing_regression_baseline=results.established_baseline
        or not baseline_metrics,
    )
    result = apply_gate(
        dict(results.metrics), policy=policy, baseline_metrics=baseline_metrics
    )
    report = GateReport(
        result=result, results=results, baseline=baseline, rules=policy.rules
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
        found = load_latest_results(project.results_dir)
        if found is None:
            return None, EXIT_ERROR, NO_RESULTS_MESSAGE
        results, _ = found
    baseline, _ = load_baseline(project.baseline_path)
    report, code = evaluate_gate(project, results=results, baseline=baseline)
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
        lines.append("  thresholds:")
        for rule in report.rules:
            observed = results.metrics.get(rule.metric)
            observed_text = "missing" if observed is None else f"{observed:g}"
            lines.append(f"    {rule.metric}: {observed_text} " f"({_rule_text(rule)})")
    for failure in report.result.failures:
        lines.append(f"  FAIL {failure.metric}: {failure.reason}")
    return "\n".join(lines)


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
