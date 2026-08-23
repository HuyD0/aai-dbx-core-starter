"""The release-readiness evidence pack.

Answers, in one document, what a reviewer needs to approve a promotion:
what ran, on which data version, scored by which scorer and prompt
versions, against which baseline, with what verdict and whose approval.
Written for someone who has never opened MLflow.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from aai_core.agentkit.baseline import BaselineRecord
from aai_core.agentkit.calibration import calibration_status
from aai_core.agentkit.catalog import get_spec
from aai_core.agentkit.config import ProjectContext
from aai_core.agentkit.errors import UnknownScorerError
from aai_core.agentkit.gate import GateReport
from aai_core.agentkit.integrity import is_integrity_metric
from aai_core.agentkit.results import ResultsRecord
from aai_core.agentkit.statistics import is_statistics_metric

EVIDENCE_JSON = "evidence.json"
EVIDENCE_MARKDOWN = "evidence.md"


def build_evidence(
    project: ProjectContext,
    *,
    results: ResultsRecord,
    baseline: BaselineRecord | None,
    gate_report: GateReport | None,
    approver_lookup: (
        Callable[[ProjectContext, ResultsRecord], Mapping[str, Any]] | None
    ) = None,
) -> tuple[dict[str, Any], str]:
    """Build the evidence document and its rendered markdown."""

    from aai_core import __version__

    approver: Mapping[str, Any]
    if approver_lookup is None:
        approver = {
            "status": "unknown",
            "reason": (
                "no deployment-job approval tag was read; approval is "
                "recorded on the Unity Catalog model version when the "
                "deployment job's approval task is approved"
            ),
        }
    else:
        approver = approver_lookup(project, results)

    passed = gate_report.passed if gate_report is not None else results.gate_passed
    document: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": f"agentkit (aai-core {__version__})",
        "identity": dict(project.settings.resource.as_dict()),
        "experiment": {
            "name": results.experiment_name,
            "id": results.experiment_id,
            "run_id": results.run_id,
        },
        "agent": results.agent,
        "dataset": {
            "ref": results.dataset.ref,
            "digest": results.dataset.digest,
            "rows": results.dataset.rows,
            "scope": {
                "mode": results.scope.mode,
                "rows": results.scope.rows,
            },
        },
        "versions": {
            "scorers": dict(results.versions.scorers),
            "judge_model": results.versions.judge_model,
            # The endpoint URI is mutable; this is what it actually served.
            # Without it an approver reading the pack later cannot tell
            # which model produced the scores.
            "judge_model_identity": results.versions.judge_model_identity,
            "judge_prompts": dict(results.versions.judge_prompts),
            "aai_core": results.versions.aai_core,
        },
        # The baseline the RUN was compared against, taken from the run's
        # own record. Reading the local `evals/baseline.json` instead would
        # pair this run's deltas with whatever baseline the reader's
        # checkout currently holds — different machine, or re-established
        # since. The local file is only a fallback for older records.
        "comparison": {
            "established_baseline": results.established_baseline,
            "baseline_run_id": results.baseline_run_id,
            "baseline_recorded_at": results.baseline_recorded_at
            or (baseline.recorded_at if baseline else None),
            "baseline_dataset_digest": results.baseline_dataset_digest
            or (baseline.dataset.digest if baseline else None),
            "metrics": _metric_rows(results),
        },
        "statistics": (
            results.statistics.model_dump(mode="json")
            if results.statistics is not None
            else None
        ),
        # What the run spent, coverage-first: unknown cost stays unknown
        # rather than reading as zero, and per-success ratios appear only
        # at complete coverage.
        "economics": (
            results.economics.model_dump(mode="json")
            if results.economics is not None
            else None
        ),
        # The judge measured as an instrument: self-consistency on this
        # run's outputs and drift on frozen anchors. A delta is a statement
        # about the agent only while the instrument held still.
        "judge_integrity": (
            results.integrity.model_dump(mode="json")
            if results.integrity is not None
            else None
        ),
        # Calibration coverage for the judges the run used — reported
        # always, enforced only when integrity.require_calibration is set.
        "judge_calibration": _calibration_rows(project, results),
        "gate": {
            "passed": passed,
            "failures": [
                {"metric": failure.metric, "reason": failure.reason}
                for failure in (
                    gate_report.result.failures if gate_report is not None else ()
                )
            ],
            "message": gate_report.message if gate_report is not None else None,
            "policy_source": (
                "the run's own recorded rules"
                if gate_report is not None and not gate_report.policy_note
                else (gate_report.policy_note if gate_report is not None else None)
            ),
        },
        "decision": results.decision,
        "change_id": results.change_id,
        "release": results.release,
        "recorded_at": results.recorded_at,
        "approver": dict(approver),
        "warnings": list(results.warnings),
    }
    return document, render_markdown(document)


def write_evidence(directory: Path, document: Mapping[str, Any], markdown: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / EVIDENCE_JSON).write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    path = directory / EVIDENCE_MARKDOWN
    path.write_text(markdown, encoding="utf-8")
    return path


def render_markdown(document: Mapping[str, Any]) -> str:
    identity = document["identity"]
    dataset = document["dataset"]
    versions = document["versions"]
    comparison = document["comparison"]
    gate = document["gate"]
    verdict = "PASSED" if gate["passed"] else "FAILED"

    lines = [
        f"# Release evidence: {identity.get('application', 'application')}",
        "",
        f"**Gate verdict: {verdict}**  ",
        f"Decision recorded: **{document['decision']}**  ",
        f"Generated by {document['generated_by']} at {document['recorded_at']}",
        "",
        "## What was evaluated",
        "",
        f"- Agent under test: `{document['agent']}`",
        f"- Dataset: `{dataset['ref']}` "
        f"(version digest `{dataset['digest']}`, {dataset['rows']} rows)",
        f"- Scored: {dataset['scope']['mode']} run over "
        f"{dataset['scope']['rows']} rows",
        f"- Environment: {identity.get('environment', 'unknown')} / "
        f"team {identity.get('team', 'unknown')} / "
        f"cost center {identity.get('cost_center', 'unknown')}",
        "",
        "## What it was compared against",
        "",
    ]
    _render_comparison(lines, comparison)
    _render_statistics(lines, document.get("statistics"))
    _render_economics(lines, document.get("economics"))
    _render_judge_integrity(lines, document.get("judge_integrity"))
    _render_judge_calibration(lines, document.get("judge_calibration"))
    _render_scoring(lines, versions)
    _render_gate(lines, gate)
    _render_approval(lines, document["approver"])
    _render_warnings(lines, document["warnings"])
    _render_provenance(lines, document, versions)
    return "\n".join(lines)


def _render_comparison(lines: list[str], comparison: Mapping[str, Any]) -> None:
    if comparison["established_baseline"]:
        lines.append(
            "This run **is** the recorded baseline - the first version of "
            "this agent scored on this dataset. Future runs are compared "
            "against it."
        )
    else:
        reference = comparison["baseline_run_id"] or "the committed baseline file"
        recorded = comparison["baseline_recorded_at"]
        lines.append(
            f"Compared against {reference}"
            + (f", recorded {recorded}" if recorded else "")
            + "."
        )
    lines.extend(["", "| metric | current | baseline | delta |", "|---|---|---|---|"])
    for row in comparison["metrics"]:
        lines.append(
            f"| {row['metric']} | {_format(row['current'])} | "
            f"{_format(row['baseline'])} | {_format(row['delta'])} |"
        )


def _render_scoring(lines: list[str], versions: Mapping[str, Any]) -> None:
    lines.extend(
        [
            "",
            "## How it was scored",
            "",
            "Scorers come from the shared enterprise registry, so a score "
            "means the same thing across teams.",
            "",
        ]
    )
    scorers = versions["scorers"]
    if scorers:
        for name, version in sorted(scorers.items()):
            lines.append(f"- `{name}` version {version}")
    else:
        lines.append("- no scorer versions recorded")
    if versions["judge_model"]:
        lines.append(f"- judge model: `{versions['judge_model']}`")
    if versions.get("judge_model_identity"):
        lines.append(f"- judge model served: `{versions['judge_model_identity']}`")
    for name, prompt in sorted(dict(versions["judge_prompts"]).items()):
        lines.append(f"- judge prompt `{name}`: `{prompt}`")


def _render_statistics(lines: list[str], statistics: Mapping[str, Any] | None) -> None:
    if not statistics or not statistics.get("estimates"):
        return
    level = float(statistics["confidence_level"]) * 100
    enforcement = "enabled" if statistics["enforced"] else "report-only"
    # Records from before the bootstrap option carry no method key; they
    # were computed with the normal approximation and render as such.
    method = str(statistics.get("method") or "normal")
    label = "bootstrap-percentile" if method == "bootstrap" else "normal-mean"
    reproduction = (
        f" ({statistics.get('bootstrap_resamples')} resamples, "
        f"seed {statistics.get('bootstrap_seed')})"
        if method == "bootstrap"
        else ""
    )
    lines.extend(
        [
            "",
            "## Statistical confidence",
            "",
            f"Intervals use the recorded {level:g}% {label} policy"
            f"{reproduction}. "
            f"The minimum enforceable sample is {statistics['minimum_cases']}; "
            f"confidence enforcement was {enforcement}.",
            "",
            "| metric | n | mean | lower | upper |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for estimate in statistics["estimates"]:
        lines.append(
            f"| {estimate['metric']} | {estimate['sample_size']} | "
            f"{_format(estimate['mean'])} | {_format(estimate['lower'])} | "
            f"{_format(estimate['upper'])} |"
        )
    paired = statistics.get("paired") or []
    if paired:
        lines.extend(
            [
                "",
                "Paired improvement is direction-normalized: positive is better.",
                "",
                "| metric | pairs | mean improvement | lower | upper |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for estimate in paired:
            lines.append(
                f"| {estimate['metric']} | {estimate['pair_count']} | "
                f"{_format(estimate['mean_improvement'])} | "
                f"{_format(estimate['lower_improvement'])} | "
                f"{_format(estimate['upper_improvement'])} |"
            )


def _render_economics(lines: list[str], economics: Mapping[str, Any] | None) -> None:
    if not economics:
        return
    rows = economics["rows"]
    cost_total = economics.get("cost_total_usd")
    lines.extend(
        [
            "",
            "## Run economics",
            "",
            (
                f"{economics['successes']} of {rows} rows completed "
                f"successfully. Cost is known for {economics['cost_known']} "
                f"of {rows} rows and token usage for "
                f"{economics['tokens_known']} of {rows} "
                f"(cost source: {economics['cost_source']}). Per-success "
                "ratios are reported only at complete coverage — unknown "
                "cost is never counted as zero."
            ),
        ]
    )
    if cost_total is not None:
        lines.extend(
            [
                "",
                f"Known spend across all rows, failed ones included: "
                f"${_format(cost_total)}.",
            ]
        )
    segments = economics.get("segments") or []
    if segments:
        lines.extend(
            [
                "",
                "Per-stratum economics — the evidence for routing an "
                "intent to a different model:",
                "",
                (
                    "| stratum | rows | success rate | cost/success | "
                    "cost p95 | latency p95 (s) |"
                ),
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for segment in segments:
            value = segment["value"] or "(unset)"
            lines.append(
                f"| {segment['key']}={value} | {segment['rows']} | "
                f"{_format(segment['success_rate'])} | "
                f"{_format(segment.get('cost_per_success_usd'))} | "
                f"{_format(segment.get('cost_p95_usd'))} | "
                f"{_format(segment.get('latency_p95_seconds'))} |"
            )


def _render_judge_integrity(
    lines: list[str], integrity: Mapping[str, Any] | None
) -> None:
    if not integrity:
        return
    preamble = (
        "A delta is a statement about the agent only while the judge "
        "held still; these checks measure the judge itself inside the run."
    )
    lines.extend(["", "## Judge integrity", "", preamble, ""])
    consistency = integrity.get("consistency")
    if consistency:
        lines.append(
            f"- Self-inconsistency: {_format(consistency['overall'])} over "
            f"{consistency['sample_size']} re-scored row(s)"
        )
        for name, rate in sorted(dict(consistency.get("flip_rates") or {}).items()):
            lines.append(f"  - `{name}`: {_format(rate)}")
    drift = integrity.get("anchor_drift")
    if drift:
        lines.append(
            f"- Anchor drift: {_format(drift['overall'])} over "
            f"{drift['rows']} frozen row(s) from `{drift['anchors_ref']}` "
            f"(digest `{str(drift['anchors_digest'])[:16]}`)"
        )
        for name, value in sorted(dict(drift.get("drift_by_scorer") or {}).items()):
            lines.append(f"  - `{name}`: {_format(value)}")
    failures = (consistency or {}).get("rescore_failures", 0) + (drift or {}).get(
        "rescore_failures", 0
    )
    if failures:
        lines.append(f"- Failed judge re-invocations: {failures}")


def _calibration_rows(
    project: ProjectContext, results: ResultsRecord
) -> list[dict[str, Any]]:
    judge_scorers: dict[str, int] = {}
    for name, version in results.versions.scorers.items():
        try:
            spec = get_spec(name)
        except UnknownScorerError:
            continue
        if spec.judge is not None:
            judge_scorers[name] = int(version)
    if not judge_scorers:
        return []
    return calibration_status(
        root=project.root,
        directory=project.config.integrity.calibration_dir,
        judge_scorers=judge_scorers,
        judge_prompts=dict(results.versions.judge_prompts),
        judge_model_identity=results.versions.judge_model_identity,
    )


def _render_judge_calibration(
    lines: list[str], calibration: list[Mapping[str, Any]] | None
) -> None:
    if not calibration:
        return
    preamble = (
        "A judged score is auditable only against a named human "
        "agreement measurement (chance-adjusted κ vs SME labels)."
    )
    lines.extend(["", "## Judge calibration", "", preamble, ""])
    for row in calibration:
        status = row.get("status", "unknown")
        if status in {"uncalibrated", "unreadable"}:
            lines.append(f"- `{row['scorer']}`: **{status}**")
            if row.get("reason"):
                lines.append(f"  - {row['reason']}")
            continue
        ceiling = row.get("human_ceiling_kappa")
        ceiling_text = (
            f", human ceiling κ {_format(ceiling)}" if ceiling is not None else ""
        )
        lines.append(
            f"- `{row['scorer']}`: **{status}** — κ {_format(row.get('kappa'))} "
            f"(minimum {_format(row.get('minimum_kappa'))}"
            f"{ceiling_text}) over {row.get('sample_size')} labels, "
            f"recorded {row.get('recorded_at')}"
        )
        for key in ("stale", "stale_prompt", "stale_judge"):
            if row.get(key):
                lines.append(f"  - **stale**: {row[key]}")


def _render_gate(lines: list[str], gate: Mapping[str, Any]) -> None:
    if gate["failures"]:
        lines.extend(["", "## Why the gate failed", ""])
        for failure in gate["failures"]:
            lines.append(f"- **{failure['metric']}**: {failure['reason']}")
    if gate.get("message"):
        lines.extend(["", "## Gate note", "", gate["message"]])
    if gate.get("policy_source"):
        lines.extend(["", f"Thresholds applied: {gate['policy_source']}."])


def _render_approval(lines: list[str], approver: Mapping[str, Any]) -> None:
    lines.extend(
        [
            "",
            "## Approval",
            "",
            f"- Status: **{approver.get('status', 'unknown')}**",
        ]
    )
    for key in ("model_version", "reason"):
        if approver.get(key):
            lines.append(f"- {key.replace('_', ' ').capitalize()}: {approver[key]}")
    if approver.get("required"):
        required = ", ".join(f"`{name}`" for name in approver["required"])
        lines.append(f"- Required approvals: {required}")
    for tag, value in sorted(dict(approver.get("tags") or {}).items()):
        lines.append(f"- Approval tag `{tag}`: {value}")
    if approver.get("caveat"):
        lines.append(f"- **Not verified**: {approver['caveat']}")


def _render_warnings(lines: list[str], warnings: list[str]) -> None:
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")


def _render_provenance(
    lines: list[str], document: Mapping[str, Any], versions: Mapping[str, Any]
) -> None:
    release = document.get("release")
    release_line = [f"- Release commit: `{release}`"] if release else []
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Change id: `{document['change_id']}`",
            *release_line,
            f"- MLflow experiment: {document['experiment']['name'] or 'unknown'}"
            + (
                f" (run `{document['experiment']['run_id']}`)"
                if document["experiment"]["run_id"]
                else ""
            ),
            f"- aai-core: {versions['aai_core']}",
            "",
        ]
    )


def databricks_approver_lookup(
    project: ProjectContext, results: ResultsRecord
) -> Mapping[str, Any]:
    """Read the deployment-job approval tag from the UC model version.

    Databricks records approval by tagging the model version with the
    approval task's name. This reads that tag; the approving principal
    itself lives in the job-run audit log, which the evidence links rather
    than restates.
    """

    model_name = project.config.registered_model
    if not model_name:
        return {
            "status": "unknown",
            "reason": (
                "no registered model is configured; set `registered_model` "
                "in agentkit.yaml to report the approval recorded on its "
                "model version"
            ),
        }
    try:
        from mlflow import MlflowClient
    except ImportError:
        return {
            "status": "unknown",
            "reason": "reading the approval tag requires the genai extra",
        }
    # Read the approval of the version that was actually evaluated. Taking
    # the newest version instead would let evidence for version N report
    # version N+1's approval the moment someone registers another one.
    evaluated = evaluated_model_version(results.agent, model_name)
    if evaluated is None:
        # No version, no verdict. Reading the newest version's tags instead
        # would let `"status": "approved"` describe a run that scored an
        # endpoint, a callable, or another model entirely — and a caveat in
        # the identity string does not stop a machine reading the status.
        # An alias is deliberately not resolved either: it may have moved
        # since the run, so resolving it now attributes an approval the run
        # never had.
        return {
            "status": "unknown",
            "reason": (
                f"this run evaluated {results.agent}, which does not name a "
                f"version of {model_name}, so no approval can be attributed "
                "to it"
            ),
        }
    try:
        client = MlflowClient(registry_uri="databricks-uc")
        version = client.get_model_version(model_name, evaluated)
        tags = dict(getattr(version, "tags", {}) or {})
        approvals = {
            key: value
            for key, value in tags.items()
            if key.lower().startswith("approval")
        }
        identity = f"{model_name} v{version.version}"
        recorded = {key: str(value) for key, value in sorted(approvals.items())}
        return _verdict(recorded, tuple(project.config.approvals), identity)
    except Exception as error:  # pragma: no cover - network/credential paths
        return {"status": "unknown", "reason": f"could not read approval tag: {error}"}


def _verdict(
    recorded: Mapping[str, str], required: tuple[str, ...], identity: str
) -> dict[str, Any]:
    """Approved only when every required approval tag says so.

    Discovering the required set from the tags that happen to exist cannot
    detect an absent one: a renamed approval task leaves `approval_old=
    Approved` behind while the current `approval_gate` tag never appears,
    and "every tag present says Approved" reads that as approved. So the
    required task names are configuration (`approvals:` in agentkit.yaml),
    and evidence generated without them says plainly that it could not
    verify completeness rather than implying it did.
    """

    approver: dict[str, Any] = {"model_version": identity}
    if recorded:
        approver["tags"] = dict(recorded)
    if required:
        approver["required"] = list(required)
        missing = [name for name in required if name not in recorded]
        unapproved = [
            name
            for name in required
            if name in recorded and recorded[name].lower() != "approved"
        ]
        if missing or unapproved:
            reasons = [f"{name} is not set" for name in missing]
            reasons += [f"{name}={recorded[name]}" for name in unapproved]
            approver["status"] = "pending" if not unapproved else "not approved"
            approver["reason"] = "outstanding approvals: " + ", ".join(reasons)
        else:
            approver["status"] = "approved"
        return approver

    if not recorded:
        approver["status"] = "pending"
        approver["reason"] = "no approval tag is set on the model version yet"
        return approver
    outstanding = [
        key for key, value in recorded.items() if value.lower() != "approved"
    ]
    approver["status"] = "not approved" if outstanding else "approved"
    if outstanding:
        approver["reason"] = "outstanding approvals: " + ", ".join(
            f"{key}={recorded[key]}" for key in outstanding
        )
    else:
        # Not a verified verdict: without the required set this cannot tell
        # an approved gate from a stale tag left by a renamed task.
        approver["caveat"] = (
            "the required approval set is not configured, so this reports "
            "only the tags that exist and cannot detect a required approval "
            "whose tag is absent; list the approval task names under "
            "`approvals:` in agentkit.yaml to verify completeness"
        )
    return approver


def evaluated_model_version(agent: str, model_name: str) -> str | None:
    """The model version an agent reference names, when it names one.

    ``models:/<catalog>.<schema>.<model>/<version>`` is what the
    deployment-job gate scores; anything else (an endpoint, a callable)
    identifies no version.
    """

    reference = str(agent or "")
    if not reference.startswith("models:/"):
        return None
    remainder = reference.removeprefix("models:/")
    name, separator, version = remainder.partition("/")
    if not separator or name != model_name:
        return None
    version = version.strip()
    return version if version.isdigit() else None


def _metric_rows(results: ResultsRecord) -> list[dict[str, Any]]:
    baseline = dict(results.baseline_metrics)
    rows = []
    for metric in sorted(
        metric
        for metric in results.metrics
        if not is_statistics_metric(metric) and not is_integrity_metric(metric)
    ):
        current = results.metrics[metric]
        reference = baseline.get(metric)
        rows.append(
            {
                "metric": metric,
                "current": current,
                "baseline": reference,
                "delta": None if reference is None else current - reference,
            }
        )
    return rows


def _format(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return (
            f"{value:+.4g}"
            if isinstance(value, float) and value < 0
            else f"{value:.4g}"
        )
    return str(value)
