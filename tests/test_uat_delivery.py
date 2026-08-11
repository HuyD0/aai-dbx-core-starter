"""Focused contracts for immutable dev-to-UAT delivery."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def _release_artifact_module():
    path = ROOT / "scripts" / "release_artifact.py"
    spec = importlib.util.spec_from_file_location("aai_release_artifact", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_artifact_evidence_binds_one_wheel_to_one_commit(tmp_path):
    module = _release_artifact_module()
    wheel = tmp_path / "application-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"reviewed wheel")

    evidence_path = module.create_evidence(tmp_path, COMMIT)
    evidence = json.loads(evidence_path.read_text())

    assert evidence["source_commit"] == COMMIT
    assert evidence["artifact"]["filename"] == wheel.name
    assert len(evidence["artifact"]["sha256"]) == 64
    assert module.verify_evidence(tmp_path, COMMIT) == wheel

    wheel.write_bytes(b"changed after dev gate")
    with pytest.raises(ValueError, match="digest does not match"):
        module.verify_evidence(tmp_path, COMMIT)


def test_release_artifact_evidence_fails_closed_on_ambiguous_or_invalid_input(
    tmp_path,
):
    module = _release_artifact_module()
    (tmp_path / "one.whl").write_bytes(b"one")

    with pytest.raises(ValueError, match="full 40- or 64-character"):
        module.create_evidence(tmp_path, "main")

    (tmp_path / "two.whl").write_bytes(b"two")
    with pytest.raises(ValueError, match="exactly one wheel"):
        module.create_evidence(tmp_path, COMMIT)


def test_generated_and_root_evidence_implementations_are_identical():
    root_script = ROOT / "scripts" / "release_artifact.py"
    shared_script = ROOT / "templates/_shared/files/scripts/release_artifact.py"
    assert root_script.read_bytes() == shared_script.read_bytes()


def test_root_workflow_promotes_the_same_artifact_after_dev_and_manual_gate():
    path = ROOT / ".github/workflows/deploy.yml"
    text = path.read_text()
    workflow = yaml.safe_load(text)
    jobs = workflow["jobs"]

    assert workflow[True]["workflow_dispatch"]["inputs"]["target"]["options"] == [
        "dev",
        "uat",
    ]
    assert jobs["deploy-uat"]["needs"] == [
        "build",
        "deploy-dev",
        "uat-prerequisites",
    ]
    assert all("environment" not in job for job in jobs.values())
    assert "UAT_DEPLOYMENT_ENABLED" in text
    assert "DATABRICKS_UAT_HOST" in text
    assert "refs/heads/main" in text
    assert text.count("scripts/release_artifact.py verify") == 2
    assert "BUNDLE_VAR_deployment_release=$GITHUB_SHA" in text
    assert "databricks bundle deploy -t uat" in text
    assert "databricks bundle deploy -t prod" not in text


def test_uat_checks_existing_lakebase_references_before_requesting_credentials():
    workflow = yaml.safe_load((ROOT / ".github/workflows/deploy.yml").read_text())
    prerequisite_steps = workflow["jobs"]["uat-prerequisites"]["steps"]
    uat_steps = workflow["jobs"]["deploy-uat"]["steps"]

    assert "permissions" not in workflow["jobs"]["uat-prerequisites"]
    prerequisite = prerequisite_steps[0]
    assert "replace-with-*" in prerequisite["run"]
    assert "*placeholder*" in prerequisite["run"]
    assert "refs/heads/main" in prerequisite["run"]
    assert "azure/login" not in prerequisite["run"].lower()

    login_index = next(
        index
        for index, step in enumerate(uat_steps)
        if step.get("name", "").startswith("Azure login")
    )
    assert login_index > 0

    deploy_step = next(
        step
        for step in uat_steps
        if step.get("name") == "Validate & deploy the dev-verified bundle (UAT)"
    )
    assert {
        "BUNDLE_VAR_hub_lakebase_branch",
        "BUNDLE_VAR_hub_lakebase_database",
        "BUNDLE_VAR_hub_lakebase_schema",
    } <= deploy_step["env"].keys()


def test_uat_lakebase_defaults_do_not_enable_the_optional_console():
    bundle = yaml.safe_load((ROOT / "databricks.yml").read_text())

    assert bundle["targets"]["uat"]["variables"]["hub_state_mode"] == "lakebase"
    assert "default" not in bundle["variables"]["hub_lakebase_branch"]
    assert "default" not in bundle["variables"]["hub_lakebase_database"]
    assert bundle["include"] == ["resources/*.yml"]

    guide = (ROOT / "docs/uat-promotion.md").read_text()
    for name in (
        "HUB_LAKEBASE_BRANCH",
        "HUB_LAKEBASE_DATABASE",
        "HUB_LAKEBASE_SCHEMA",
    ):
        assert name in guide
    normalized_guide = " ".join(guide.split())
    assert "never guesses a Lakebase path" in normalized_guide
    assert "optional console resource remains" in normalized_guide


def test_shared_generated_workflow_runs_both_release_gates_before_uat_app():
    path = ROOT / "templates/_shared/files/.github/workflows/deploy.yml"
    text = path.read_text()
    workflow = yaml.safe_load(text)
    jobs = workflow["jobs"]

    assert all("environment" not in job for job in jobs.values())
    assert jobs["deploy-uat"]["needs"] == [
        "build",
        "deploy-dev",
        "uat-prerequisites",
    ]
    assert text.index("databricks bundle run release_gate -t dev") < text.index(
        "databricks bundle run release_gate -t uat"
    )
    assert text.index("databricks bundle run release_gate -t uat") < text.index(
        "databricks bundle run agent_app -t uat"
    )
    assert "databricks bundle deploy -t prod" not in text
