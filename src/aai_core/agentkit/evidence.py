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
from aai_core.agentkit.config import ProjectContext
from aai_core.agentkit.gate import GateReport
from aai_core.agentkit.results import ResultsRecord

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
    for name, prompt in sorted(dict(versions["judge_prompts"]).items()):
        lines.append(f"- judge prompt `{name}`: `{prompt}`")

    if gate["failures"]:
        lines.extend(["", "## Why the gate failed", ""])
        for failure in gate["failures"]:
            lines.append(f"- **{failure['metric']}**: {failure['reason']}")
    if gate.get("message"):
        lines.extend(["", "## Gate note", "", gate["message"]])
    if gate.get("policy_source"):
        lines.extend(["", f"Thresholds applied: {gate['policy_source']}."])

    approver = document["approver"]
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

    if document["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in document["warnings"]:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Change id: `{document['change_id']}`",
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
    return "\n".join(lines)


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
    for metric in sorted(results.metrics):
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
