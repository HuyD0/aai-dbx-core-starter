"""The ``agentkit`` command line.

    agentkit init       scaffold a working evaluation project
    agentkit compare    THE primary verb - score this version against the last
    agentkit smoke      fast gate: a sample, seconds, no cluster
    agentkit eval       the full suite, optionally as a Databricks job
    agentkit gate       promotion check against thresholds and the baseline
    agentkit evidence   the release readiness record
    agentkit scorers ls browse the shared enterprise scorer registry

Exit codes are the CI contract: 0 passed, 2 ran but a threshold failed,
1 runtime or configuration error.

Only the standard library is imported at module load; every command
imports what it needs so ``smoke``, ``gate``, ``evidence``, ``init`` and
``scorers ls`` work without MLflow installed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXIT_PASS = 0
EXIT_ERROR = 1
EXIT_THRESHOLD_FAILED = 2

DECISIONS = ("adopt", "reject", "inconclusive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    handler = arguments.handler
    try:
        return handler(arguments)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("cancelled", file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:  # noqa: BLE001 - single CLI boundary
        from aai_core.exceptions import AaiCoreError

        if isinstance(error, AaiCoreError):
            print(f"error: {error}", file=sys.stderr)
            return EXIT_ERROR
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentkit",
        description=(
            "Compare this version of your agent against the last one, and "
            "land the evidence in MLflow and Unity Catalog."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    init = subcommands.add_parser("init", help="scaffold a working evaluation project")
    init.add_argument("--name", required=True, help="project (and bundle) name")
    init.add_argument("--template", default="evaluation-project")
    init.add_argument(
        "--template-source",
        default=None,
        help="template repository (defaults to $AAI_TEMPLATE_REPO)",
    )
    init.add_argument("--output-dir", default=None)
    init.add_argument(
        "--print-only",
        action="store_true",
        help="print the command and the next steps without running it",
    )
    init.set_defaults(handler=_cmd_init)

    compare = subcommands.add_parser(
        "compare", help="score this version against the baseline"
    )
    _add_scoring_arguments(compare)
    compare.add_argument(
        "--establish-baseline",
        action="store_true",
        help="record this run as the baseline (use when there is no baseline yet)",
    )
    compare.add_argument("--full", action="store_true", help="score every row")
    compare.add_argument("--rows", type=int, default=None)
    compare.add_argument("--mode", choices=("live", "answer-sheet"), default=None)
    compare.add_argument("--decision", choices=DECISIONS, default=None)
    compare.add_argument("--baseline-run", default=None)
    compare.set_defaults(handler=_cmd_compare)

    smoke = subcommands.add_parser(
        "smoke", help="fast, free gate over a sample of rows"
    )
    _add_scoring_arguments(smoke)
    smoke.add_argument(
        "--live",
        action="store_true",
        help="call the agent instead of replaying the recorded answer sheet",
    )
    smoke.add_argument("--rows", type=int, default=None)
    smoke.add_argument(
        "--establish-baseline",
        action="store_true",
        help="record this sample run as the baseline",
    )
    smoke.set_defaults(handler=_cmd_smoke)

    evaluate = subcommands.add_parser(
        "eval", help="the full suite, locally or as a Databricks job"
    )
    _add_scoring_arguments(evaluate)
    evaluate.add_argument(
        "--submit",
        action="store_true",
        help="run the bundle's release_gate job instead of scoring locally",
    )
    evaluate.add_argument("--target", default="dev", help="bundle target")
    evaluate.add_argument("--mode", choices=("live", "answer-sheet"), default=None)
    evaluate.add_argument("--decision", choices=DECISIONS, default=None)
    evaluate.add_argument("--baseline-run", default=None)
    evaluate.add_argument("--establish-baseline", action="store_true")
    evaluate.set_defaults(handler=_cmd_eval)

    gate = subcommands.add_parser(
        "gate", help="promotion check against thresholds and the baseline"
    )
    gate.add_argument("--config", default=None)
    gate.add_argument("--results", default=None)
    gate.add_argument("--json", action="store_true", dest="as_json")
    gate.set_defaults(handler=_cmd_gate)

    evidence = subcommands.add_parser(
        "evidence", help="write the release readiness record"
    )
    evidence.add_argument("--config", default=None)
    evidence.add_argument("--output", default=None)
    evidence.add_argument("--json", action="store_true", dest="as_json")
    evidence.set_defaults(handler=_cmd_evidence)

    scorers = subcommands.add_parser(
        "scorers", help="browse the shared enterprise scorer registry"
    )
    scorer_commands = scorers.add_subparsers(dest="scorers_command", required=True)
    listing = scorer_commands.add_parser("ls", help="list registered scorers")
    listing.add_argument("--config", default=None)
    listing.add_argument("--json", action="store_true", dest="as_json")
    listing.add_argument(
        "--live",
        action="store_true",
        help="resolve judge endpoint and prompt versions from the workspace",
    )
    listing.set_defaults(handler=_cmd_scorers_ls)

    return parser


def _add_scoring_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--agent",
        default=None,
        help="score this target instead of the one in agentkit.yaml "
        "(same shapes: models:/..., endpoints:/..., a URL, module:function)",
    )
    parser.add_argument("--plan", action="store_true", help="print the plan and stop")
    parser.add_argument(
        "--yes",
        action="store_true",
        dest="assume_yes",
        help="skip the confirmation prompt (required for non-interactive runs)",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")


# --- commands --------------------------------------------------------------


def _cmd_init(arguments: argparse.Namespace) -> int:
    from aai_core.agentkit.init import run_init

    return run_init(
        project_name=arguments.name,
        template=arguments.template,
        template_source=arguments.template_source,
        output_dir=Path(arguments.output_dir) if arguments.output_dir else None,
        print_only=arguments.print_only,
    )


def _cmd_compare(arguments: argparse.Namespace) -> int:
    rows_limit = _rows_limit(arguments, default_to_sample=False)
    return _score(
        arguments,
        command="compare",
        rows_limit=rows_limit,
        judges_enabled=True,
        mode=arguments.mode,
    )


def _cmd_smoke(arguments: argparse.Namespace) -> int:
    project = _project(arguments)
    rows = arguments.rows or project.config.smoke.rows
    return _score(
        arguments,
        command="smoke",
        rows_limit=rows,
        # Smoke never runs judges: that is what keeps it seconds-long, free,
        # and runnable in credential-free pull-request CI. `--live` chooses
        # where the answers come from, not whether judges score them.
        judges_enabled=False,
        mode="live" if arguments.live else "answer-sheet",
        project=project,
        decision=None,
        baseline_run=None,
        # Smoke is the fast threshold gate: it must run on a project that
        # was generated minutes ago. `compare` owns the baseline demand.
        require_baseline=False,
    )


def _cmd_eval(arguments: argparse.Namespace) -> int:
    project = _project(arguments)
    if arguments.submit:
        from aai_core.agentkit.runner import submit_job

        code, messages = submit_job(project, target=arguments.target)
        for message in messages:
            print(message)
        return code
    return _score(
        arguments,
        command="eval",
        rows_limit=None,
        judges_enabled=True,
        mode=arguments.mode,
        project=project,
    )


def _cmd_gate(arguments: argparse.Namespace) -> int:
    from aai_core.agentkit.gate import render_report, run_gate

    project = _project(arguments)
    results_path = Path(arguments.results) if arguments.results else None
    report, code, message = run_gate(project, results_path=results_path)
    if arguments.as_json:
        document: dict[str, Any] = {"exit_code": code, "message": message}
        if report is not None:
            document["passed"] = report.passed
            document["metrics"] = dict(report.result.metrics)
            document["failures"] = [
                {"metric": failure.metric, "reason": failure.reason}
                for failure in report.result.failures
            ]
        print(json.dumps(document, indent=2, sort_keys=True, default=str))
        return code
    if report is None:
        print(message, file=sys.stderr)
        return code
    print(render_report(report))
    return code


def _cmd_evidence(arguments: argparse.Namespace) -> int:
    from aai_core.agentkit.baseline import load_baseline
    from aai_core.agentkit.evidence import (
        build_evidence,
        databricks_approver_lookup,
        write_evidence,
    )
    from aai_core.agentkit.gate import evaluate_gate
    from aai_core.agentkit.results import load_latest_results

    project = _project(arguments)
    found = load_latest_results(project.results_dir)
    if found is None:
        from aai_core.agentkit.errors import EvidenceMissingError

        raise EvidenceMissingError(
            "no evaluation results to report on",
            remediation="Run `agentkit compare` first.",
        )
    results, _ = found
    baseline, _ = load_baseline(project.baseline_path)
    report, _ = evaluate_gate(project, results=results, baseline=baseline)
    document, markdown = build_evidence(
        project,
        results=results,
        baseline=baseline,
        gate_report=report,
        approver_lookup=databricks_approver_lookup,
    )
    directory = Path(arguments.output) if arguments.output else project.evidence_dir
    path = write_evidence(directory, document, markdown)
    if arguments.as_json:
        print(json.dumps(document, indent=2, sort_keys=True, default=str))
        return EXIT_PASS
    print(markdown)
    print(f"\nwritten to {path.parent}")
    return EXIT_PASS


def _cmd_scorers_ls(arguments: argparse.Namespace) -> int:
    from aai_core.agentkit.catalog import CATALOG

    judge_uri = None
    prompt_versions: dict[str, str] = {}
    if arguments.live:
        project = _project(arguments)
        judge_uri = project.judge_model_uri()
        prompt_versions = _live_prompt_versions(project)

    if arguments.as_json:
        document = [
            {
                **spec.model_dump(),
                "resolved_judge": judge_uri if spec.judge else None,
                "resolved_prompt": prompt_versions.get(spec.name),
            }
            for spec in CATALOG
        ]
        print(json.dumps(document, indent=2, sort_keys=True, default=str))
        return EXIT_PASS

    header = ("scorer", "v", "kind", "requires", "judge", "threshold")
    rows = [header]
    for spec in CATALOG:
        if spec.judge is None:
            judge = "-"
        elif not spec.judge.overridable:
            judge = "platform default"
        else:
            judge = judge_uri or spec.judge.logical_model
        rows.append(
            (
                spec.name,
                str(spec.version),
                spec.kind.value,
                _requirements(spec),
                judge,
                spec.default_threshold or "report-only",
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(header))]
    for row in rows:
        print(
            "  ".join(
                cell.ljust(widths[index]) for index, cell in enumerate(row)
            ).rstrip()
        )
    print(
        "\nScorers are versioned platform assets: a project selects them and "
        "sets thresholds, but never redefines one. That is what makes 0.8 "
        "mean the same thing on two teams."
    )
    for name, uri in sorted(prompt_versions.items()):
        print(f"  {name} judge prompt: {uri}")
    return EXIT_PASS


# --- helpers ---------------------------------------------------------------


def _score(
    arguments: argparse.Namespace,
    *,
    command: str,
    rows_limit: int | None,
    judges_enabled: bool,
    mode: str | None,
    project: Any | None = None,
    decision: str | None = ...,  # type: ignore[assignment]
    baseline_run: str | None = ...,  # type: ignore[assignment]
    require_baseline: bool = True,
) -> int:
    from aai_core.agentkit.runner import run_scoring

    project = project if project is not None else _project(arguments)
    if decision is ...:
        decision = getattr(arguments, "decision", None)
    if baseline_run is ...:
        baseline_run = getattr(arguments, "baseline_run", None)
    assume_yes = bool(getattr(arguments, "assume_yes", False))
    as_json = bool(getattr(arguments, "as_json", False))
    if as_json and not assume_yes and judges_enabled:
        print(
            "error: --json runs non-interactively; pass --yes to confirm the "
            "judge spend up front",
            file=sys.stderr,
        )
        return EXIT_ERROR

    # MLflow's evaluation harness prints progress to stdout. With --json the
    # stream must carry exactly one document, so anything the run emits is
    # diverted to stderr where it stays visible in CI logs.
    stdout_guard = (
        contextlib.redirect_stdout(sys.stderr) if as_json else contextlib.nullcontext()
    )
    with stdout_guard:
        outcome, code = run_scoring(
            project,
            command=command,
            mode=mode,
            agent=getattr(arguments, "agent", None),
            rows_limit=rows_limit,
            judges_enabled=judges_enabled,
            require_baseline=require_baseline,
            establish_baseline=bool(getattr(arguments, "establish_baseline", False)),
            decision=decision,
            baseline_run_id=baseline_run,
            assume_yes=assume_yes,
            plan_only=bool(getattr(arguments, "plan", False)),
            confirm=_confirm,
        )
    if as_json:
        document: dict[str, Any] = {
            "exit_code": code,
            "plan": [
                {
                    "scorer": entry.spec.name,
                    "version": entry.spec.version,
                    "kind": entry.spec.kind.value,
                    "threshold": entry.threshold,
                    "reason": entry.reason,
                }
                for entry in outcome.plan.entries
            ],
            "excluded": [
                {"scorer": item.spec.name, "reason": item.reason}
                for item in outcome.plan.excluded
            ],
            "cost": outcome.cost.model_dump(),
            "warnings": list(outcome.warnings),
        }
        if outcome.results is not None:
            document["results"] = outcome.results.model_dump()
            document["comparison"] = [
                {
                    "metric": row.metric,
                    "current": row.current,
                    "baseline": row.baseline,
                    "delta": row.delta,
                    "threshold": row.threshold,
                    "verdict": row.verdict,
                }
                for row in outcome.comparison
            ]
        print(json.dumps(document, indent=2, sort_keys=True, default=str))
        return code
    for message in outcome.messages:
        print(message)
    return code


def _rows_limit(
    arguments: argparse.Namespace, *, default_to_sample: bool
) -> int | None:
    if getattr(arguments, "full", False):
        return None
    rows = getattr(arguments, "rows", None)
    if rows:
        return rows
    return None


def _project(arguments: argparse.Namespace) -> Any:
    from aai_core.agentkit.config import ProjectContext

    config = getattr(arguments, "config", None)
    return ProjectContext.load(config)


def _confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        print(
            f"{prompt} refusing to prompt on a non-interactive stream; pass "
            "--yes to proceed.",
            file=sys.stderr,
        )
        return False
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _live_prompt_versions(project: Any) -> dict[str, str]:
    from aai_core.agentkit.catalog import CATALOG

    versions: dict[str, str] = {}
    manager = project.prompt_manager()
    for spec in CATALOG:
        binding = spec.judge
        if binding is None or not binding.prompt_name:
            continue
        try:
            prompt = manager.load(binding.prompt_name, alias=binding.prompt_alias)
        except Exception:
            versions[spec.name] = "not registered (bundled instructions apply)"
            continue
        versions[spec.name] = str(
            getattr(prompt, "uri", None)
            or f"{binding.prompt_name}@{binding.prompt_alias}"
        )
    return versions


def _requirements(spec: Any) -> str:
    parts = []
    if spec.needs_expectations:
        parts.append(
            " or ".join(f"expectations.{key}" for key in spec.needs_expectations)
        )
    if spec.needs_trace.value == "retrieval":
        parts.append("RETRIEVER spans")
    elif spec.needs_trace.value == "tools":
        parts.append("tool-call spans")
    elif spec.needs_trace.value == "any":
        parts.append("a live trace")
    return "; ".join(parts) or "outputs"


if __name__ == "__main__":
    raise SystemExit(main())
