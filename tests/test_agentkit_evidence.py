"""Unit tests for the release-readiness evidence pack."""

import json

import pytest

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
            scorers={"keyword_coverage": 2, "response_length_ok": 2},
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
    assert document["versions"]["scorers"]["keyword_coverage"] == 2
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


def _lookup_with_tags(
    tmp_path,
    monkeypatch,
    tags,
    agent="models:/main.eval.agent/7",
    approvals=(),
):
    """Run the real approver lookup against a fake UC registry client."""

    import sys
    from types import SimpleNamespace

    version = SimpleNamespace(version="7", tags=dict(tags))

    class _Client:
        def __init__(self, registry_uri=None):
            self.registry_uri = registry_uri

        def get_model_version(self, name, number):
            return version

    monkeypatch.setitem(
        sys.modules, "mlflow", SimpleNamespace(MlflowClient=_Client, __spec__=None)
    )
    from aai_core.agentkit.evidence import databricks_approver_lookup

    project = ProjectContext(
        config=AgentkitConfig(
            version=1,
            agent=agent,
            dataset="evals/data/golden_cases.json",
            registered_model="main.eval.agent",
            approvals=approvals,
        ),
        settings=dev_settings(),
        root=tmp_path,
    )
    return databricks_approver_lookup(project, _results(agent=agent))


def test_every_approval_tag_counts(tmp_path, monkeypatch):
    """One approved tag does not approve a second, still-open gate.

    A job with two approval tasks writes two tags, and a renamed task
    leaves its old one behind. Reading the alphabetically first tag would
    report a version as approved on the strength of a stale one.
    """

    approver = _lookup_with_tags(
        tmp_path,
        monkeypatch,
        {"approval_business": "Approved", "approval_risk": "Pending"},
    )

    assert approver["status"] == "not approved"
    assert approver["tags"] == {
        "approval_business": "Approved",
        "approval_risk": "Pending",
    }
    assert "approval_risk=Pending" in approver["reason"]


def test_all_approvals_present_reports_approved(tmp_path, monkeypatch):
    approver = _lookup_with_tags(
        tmp_path,
        monkeypatch,
        {"approval_business": "Approved", "approval_risk": "approved"},
    )

    assert approver["status"] == "approved"
    assert "reason" not in approver


def test_no_approval_tag_is_pending(tmp_path, monkeypatch):
    approver = _lookup_with_tags(tmp_path, monkeypatch, {"other": "value"})

    assert approver["status"] == "pending"


def test_markdown_lists_every_approval_tag(tmp_path):
    from aai_core.agentkit.evidence import build_evidence

    project = _project(tmp_path)
    document, markdown = build_evidence(
        project,
        results=_results(),
        baseline=None,
        gate_report=None,
        approver_lookup=lambda *_: {
            "status": "not approved",
            "tags": {"approval_business": "Approved", "approval_risk": "Pending"},
            "model_version": "main.eval.agent v7",
        },
    )

    assert "Approval tag `approval_business`: Approved" in markdown
    assert "Approval tag `approval_risk`: Pending" in markdown


def test_absent_required_approval_tag_is_not_approved(tmp_path, monkeypatch):
    """A stale tag cannot stand in for a required approval that is absent.

    A renamed approval task leaves `approval_old=Approved` behind while
    the current `approval_gate` tag never appears. Discovering the required
    set from the tags that exist cannot see the gap; the required names
    are configuration.
    """

    approver = _lookup_with_tags(
        tmp_path,
        monkeypatch,
        {"approval_old": "Approved"},
        approvals=("approval_gate",),
    )

    assert approver["status"] == "pending"
    assert "approval_gate is not set" in approver["reason"]
    assert approver["required"] == ["approval_gate"]


def test_every_required_approval_present_is_approved(tmp_path, monkeypatch):
    approver = _lookup_with_tags(
        tmp_path,
        monkeypatch,
        {"approval_gate": "Approved", "approval_old": "Superseded"},
        approvals=("approval_gate",),
    )

    assert approver["status"] == "approved"
    # The stale tag is still shown; it just no longer decides anything.
    assert approver["tags"]["approval_old"] == "Superseded"


def test_unconfigured_required_set_says_it_cannot_verify(tmp_path, monkeypatch):
    approver = _lookup_with_tags(tmp_path, monkeypatch, {"approval_gate": "Approved"})

    assert approver["status"] == "approved"
    assert "cannot detect a required approval whose tag is absent" in (
        approver["caveat"]
    )


def test_markdown_flags_an_unverified_approval(tmp_path):
    from aai_core.agentkit.evidence import build_evidence

    _, markdown = build_evidence(
        _project(tmp_path),
        results=_results(),
        baseline=None,
        gate_report=None,
        approver_lookup=lambda *_: {
            "status": "approved",
            "tags": {"approval_gate": "Approved"},
            "caveat": "the required approval set is not configured",
        },
    )

    assert "**Not verified**" in markdown


def test_evidence_reads_baseline_lineage_from_the_record(tmp_path):
    """`evidence --run` must not pair a run's deltas with a local baseline."""

    from aai_core.agentkit.evidence import build_evidence

    results = _results(
        baseline_run_id="run-0",
        baseline_recorded_at="2026-07-01T09:00:00Z",
        baseline_dataset_digest="digest-of-the-run",
    )
    local = BaselineRecord(
        schema_version=1,
        run_id="run-99",
        recorded_at="2026-08-02T23:59:00Z",
        dataset=BaselineDataset(ref="golden.json", digest="a-newer-digest", rows=10),
        scope=BaselineScope(mode="full", rows=10),
        metrics={"keyword_coverage/mean": 0.1},
        versions=BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"keyword_coverage": 2},
            aai_core="0.4.0",
        ),
        recorded_by="agentkit compare --establish-baseline",
        change_id="zzz9999",
    )

    document, _ = build_evidence(
        _project(tmp_path), results=results, baseline=local, gate_report=None
    )

    assert document["comparison"]["baseline_recorded_at"] == "2026-07-01T09:00:00Z"
    assert document["comparison"]["baseline_dataset_digest"] == "digest-of-the-run"


def _refusing_lookup(tmp_path, monkeypatch, agent):
    """The approver lookup against a registry that must not be consulted."""

    import sys
    from types import SimpleNamespace

    calls = []

    class _Client:
        def __init__(self, registry_uri=None):
            calls.append(registry_uri)

        def get_model_version(self, name, number):  # pragma: no cover - guard
            raise AssertionError("no version was evaluated; nothing to read")

        def search_model_versions(self, filter_string):  # pragma: no cover
            raise AssertionError("the newest version is not this run's version")

    monkeypatch.setitem(
        sys.modules, "mlflow", SimpleNamespace(MlflowClient=_Client, __spec__=None)
    )
    from aai_core.agentkit.evidence import databricks_approver_lookup

    project = ProjectContext(
        config=AgentkitConfig(
            version=1,
            agent=agent,
            dataset="evals/data/golden_cases.json",
            registered_model="main.eval.agent",
        ),
        settings=dev_settings(),
        root=tmp_path,
    )
    approver = databricks_approver_lookup(project, _results(agent=agent))
    return approver, calls


@pytest.mark.parametrize(
    "agent",
    [
        "endpoints:/agent-serving",
        "src/app/example_agent.py:respond",
        "models:/main.eval.agent@champion",
        "models:/other.catalog.model/3",
    ],
)
def test_approval_needs_the_exact_evaluated_version(tmp_path, monkeypatch, agent):
    """No version, no verdict — and no registry call either.

    An endpoint, a callable, an alias, and a different registered model all
    identify no version of the configured model. Reading the newest
    version's tags would let `status: approved` describe a run that scored
    something else entirely, and a caveat in the identity string does not
    stop a machine reading the status.
    """

    approver, calls = _refusing_lookup(tmp_path, monkeypatch, agent)

    assert approver["status"] == "unknown"
    assert agent in approver["reason"]
    assert "main.eval.agent" in approver["reason"]
    # The registry is never consulted: there is nothing to ask it for.
    assert calls == []


def test_an_evaluated_version_is_still_read(tmp_path, monkeypatch):
    """The exact-version path is untouched."""

    approver = _lookup_with_tags(
        tmp_path,
        monkeypatch,
        {"approval_gate": "Approved"},
        approvals=("approval_gate",),
    )

    assert approver["status"] == "approved"
    assert approver["model_version"] == "main.eval.agent v7"
    assert "latest" not in approver["model_version"]


def test_the_pack_names_the_model_behind_the_judge_endpoint(tmp_path):
    """`endpoints:/judge` is a mutable pointer, not an identity.

    The run resolves what the endpoint actually served; without carrying
    that into the pack, an approver reading it months later cannot tell
    which model produced the scores — and the comparability check that
    pins the identity has nothing to show for itself.
    """

    project = _project(tmp_path)
    results = _results(
        versions=BaselineVersions(
            agent="src/app/example_agent.py:respond",
            scorers={"correctness": 1},
            judge_model="endpoints:/judge",
            judge_model_identity="databricks-claude-sonnet-4-5",
            aai_core="0.4.0",
        )
    )

    document, markdown = build_evidence(
        project, results=results, baseline=_baseline(), gate_report=None
    )

    assert (
        document["versions"]["judge_model_identity"] == "databricks-claude-sonnet-4-5"
    )
    assert "judge model served: `databricks-claude-sonnet-4-5`" in markdown


def test_an_unresolved_judge_identity_adds_no_line(tmp_path):
    document, markdown = build_evidence(
        _project(tmp_path),
        results=_results(),
        baseline=_baseline(),
        gate_report=None,
    )

    assert document["versions"]["judge_model_identity"] is None
    assert "judge model served" not in markdown
