"""Unit tests for the release-readiness evidence pack."""

import json

from aai_core.agentkit.baseline import (
    BaselineDataset,
    BaselineRecord,
    BaselineScope,
    BaselineVersions,
)
from aai_core.agentkit.config import AgentkitConfig, ProjectContext
from aai_core.agentkit.evidence import build_evidence, write_evidence
from aai_core.agentkit.gate import evaluate_gate
from aai_core.agentkit.results import ResultsRecord
from aai_core.testing import dev_settings


def _project(tmp_path):
    return ProjectContext(
        config=AgentkitConfig(
            version=1,
            agent="src/app/example_agent.py:respond",
            dataset="evals/data/golden_cases.json",
        ),
        settings=dev_settings(),
        root=tmp_path,
    )


def _results(**overrides):
    values = {
        "command": "compare",
        "recorded_at": "2026-08-02T10:00:00Z",
        "run_id": "run-1",
        "experiment_id": "42",
        "experiment_name": "/Shared/test-team-test-project-test-app",
        "agent": "src/app/example_agent.py:respond",
        "dataset": BaselineDataset(ref="golden.json", digest="abc123", rows=10),
        "scope": BaselineScope(mode="full", rows=10),
        "mode": "answer-sheet",
        "metrics": {"keyword_coverage/mean": 0.8, "response_length_ok/mean": 1.0},
        "versions": BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 1, "response_length_ok": 1},
            judge_model="endpoints:/judge",
            judge_prompts={"pension_domain_policy": "prompts:/main.eval.p/4"},
            aai_core="0.4.0",
        ),
        "baseline_run_id": "run-0",
        "baseline_metrics": {
            "keyword_coverage/mean": 0.75,
            "response_length_ok/mean": 1.0,
        },
        "decision": "adopt",
        "change_id": "abc1234",
        "gate_passed": True,
    }
    values.update(overrides)
    return ResultsRecord(**values)


def _baseline():
    return BaselineRecord(
        schema_version=1,
        run_id="run-0",
        recorded_at="2026-08-01T10:00:00Z",
        dataset=BaselineDataset(ref="golden.json", digest="abc123", rows=10),
        scope=BaselineScope(mode="full", rows=10),
        metrics={"keyword_coverage/mean": 0.75},
        versions=BaselineVersions(agent="agent", aai_core="0.4.0"),
        recorded_by="agentkit compare --establish-baseline",
        change_id="0000000",
    )


def test_evidence_names_versions_baseline_and_decision(tmp_path):
    project = _project(tmp_path)
    results = _results()
    report, _ = evaluate_gate(project, results=results, baseline=_baseline())

    document, markdown = build_evidence(
        project, results=results, baseline=_baseline(), gate_report=report
    )

    assert document["agent"] == "src/app/example_agent.py:respond"
    assert document["dataset"]["digest"] == "abc123"
    assert document["versions"]["scorers"]["keyword_coverage"] == 1
    assert document["versions"]["judge_model"] == "endpoints:/judge"
    assert document["comparison"]["baseline_run_id"] == "run-0"
    assert document["decision"] == "adopt"
    assert document["identity"]["team"] == "test-team"

    assert "Gate verdict: PASSED" in markdown
    assert "keyword_coverage" in markdown
    assert "run-0" in markdown
    assert "prompts:/main.eval.p/4" in markdown
    assert "| metric | current | baseline | delta |" in markdown


def test_established_baseline_is_explicit_in_the_narrative(tmp_path):
    project = _project(tmp_path)
    results = _results(
        established_baseline=True, baseline_run_id=None, baseline_metrics={}
    )

    _, markdown = build_evidence(
        project, results=results, baseline=None, gate_report=None
    )

    assert "This run **is** the recorded baseline" in markdown


def test_gate_failures_are_explained(tmp_path):
    project = _project(tmp_path)
    results = _results(metrics={"keyword_coverage/mean": 0.1}, gate_passed=False)
    report, _ = evaluate_gate(project, results=results, baseline=_baseline())

    document, markdown = build_evidence(
        project, results=results, baseline=_baseline(), gate_report=report
    )

    assert document["gate"]["passed"] is False
    assert document["gate"]["failures"]
    assert "Why the gate failed" in markdown
    assert "Gate verdict: FAILED" in markdown


def test_approver_is_unknown_by_default_and_injectable(tmp_path):
    project = _project(tmp_path)
    results = _results()

    document, markdown = build_evidence(
        project, results=results, baseline=_baseline(), gate_report=None
    )
    assert document["approver"]["status"] == "unknown"
    assert "Status: **unknown**" in markdown

    def approver_lookup(project_context, record):
        return {
            "status": "approved",
            "tag": "approval_gate",
            "value": "Approved",
            "model_version": "main.eval.agent v3",
        }

    approved, approved_markdown = build_evidence(
        project,
        results=results,
        baseline=_baseline(),
        gate_report=None,
        approver_lookup=approver_lookup,
    )
    assert approved["approver"]["status"] == "approved"
    assert "main.eval.agent v3" in approved_markdown


def test_evidence_files_are_written(tmp_path):
    project = _project(tmp_path)
    results = _results()
    document, markdown = build_evidence(
        project, results=results, baseline=_baseline(), gate_report=None
    )

    path = write_evidence(project.evidence_dir, document, markdown)

    assert path.name == "evidence.md"
    written = json.loads(
        (project.evidence_dir / "evidence.json").read_text(encoding="utf-8")
    )
    assert written["change_id"] == "abc1234"
    assert path.read_text(encoding="utf-8").startswith("# Release evidence")


def test_evidence_carries_no_secret_shaped_content(tmp_path):
    project = _project(tmp_path)
    document, markdown = build_evidence(
        project, results=_results(), baseline=_baseline(), gate_report=None
    )

    serialized = json.dumps(document) + markdown
    for marker in ("dapi", "Bearer ", "client_secret", "AZURE_CLIENT_SECRET"):
        assert marker not in serialized


def test_non_comparison_evidence_never_reports_passed(tmp_path):
    """A record that named no baseline must not read as an approval."""

    project = _project(tmp_path)
    results = _results(baseline_run_id=None, baseline_metrics={}, gate_passed=True)
    report, code = evaluate_gate(project, results=results, baseline=None)

    document, markdown = build_evidence(
        project, results=results, baseline=None, gate_report=report
    )

    assert code == 2
    assert document["gate"]["passed"] is False
    assert "Gate verdict: FAILED" in markdown
    assert any(
        failure["metric"] == "comparison" for failure in document["gate"]["failures"]
    )


def test_approver_lookup_without_a_registered_model_says_why(tmp_path):
    from aai_core.agentkit.evidence import databricks_approver_lookup

    project = _project(tmp_path)

    approver = databricks_approver_lookup(project, _results())

    assert approver["status"] == "unknown"
    assert "registered_model" in approver["reason"]


def test_approval_is_read_for_the_evaluated_model_version():
    """Evidence for version N must not report version N+1's approval."""

    from aai_core.agentkit.evidence import evaluated_model_version

    name = "main.evaluation.agent"
    assert evaluated_model_version(f"models:/{name}/7", name) == "7"
    # Not a UC model reference, a different model, or no version: no claim.
    assert evaluated_model_version("endpoints:/serving", name) is None
    assert evaluated_model_version("models:/other.model/7", name) is None
    assert evaluated_model_version(f"models:/{name}", name) is None
    assert evaluated_model_version(f"models:/{name}@champion", name) is None
