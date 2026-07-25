"""Credential-free regression tests for the platform's security boundaries."""

import re
import runpy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SHA_PIN = re.compile(r"^\s*uses:\s*[^@\s]+@([0-9a-f]{40})(?:\s+#.*)?$", re.MULTILINE)
USES = re.compile(r"^\s*uses:\s*[^@\s]+@([^\s]+)", re.MULTILINE)


def load_yaml(relative_path):
    with (ROOT / relative_path).open() as stream:
        return yaml.safe_load(stream)


def test_sample_notebook_runs(capsys):
    runpy.run_path(str(ROOT / "src" / "notebooks" / "sample_etl.py"))
    assert capsys.readouterr().out.strip().endswith("package import verified")


def test_dev_target_is_pinned_to_dev_workspace():
    bundle = load_yaml("databricks.yml")
    assert bundle["targets"]["dev"]["workspace"]["host"] == (
        "https://adb-7405609799238491.11.azuredatabricks.net"
    )


def test_sample_job_uses_constrained_job_compute_policy():
    resources = load_yaml("resources/sample_job.yml")
    cluster = resources["resources"]["jobs"]["aai_dbx_base_template_sample"][
        "job_clusters"
    ][0]["new_cluster"]
    assert cluster["policy_id"] == "0005F2031B6D2319"
    assert cluster["num_workers"] == 1
    assert cluster["spark_version"] == "18.0.x-scala2.13"
    assert "spark_conf" not in cluster


def test_bundle_and_compute_use_required_platform_tags():
    bundle = load_yaml("databricks.yml")
    resources = load_yaml("resources/sample_job.yml")
    bundle_tags = bundle["targets"]["dev"]["presets"]["tags"]
    compute_tags = resources["resources"]["jobs"]["aai_dbx_base_template_sample"][
        "job_clusters"
    ][0]["new_cluster"]["custom_tags"]
    required = {
        "application",
        "project",
        "environment",
        "team",
        "owner_group",
        "cost_center",
        "data_classification",
        "lifecycle",
        "tag_schema_version",
    }
    assert required.issubset(bundle_tags)
    assert required.issubset(compute_tags)


def test_all_github_actions_are_commit_pinned():
    for workflow in WORKFLOWS.glob("*.yml"):
        text = workflow.read_text()
        references = USES.findall(text)
        pins = SHA_PIN.findall(text)
        assert references, f"{workflow.name} has no action references"
        assert len(pins) == len(
            references
        ), f"{workflow.name} contains a mutable action reference"


def test_pr_ci_is_credential_free():
    text = (WORKFLOWS / "ci.yml").read_text()
    workflow = yaml.safe_load(text)
    assert workflow["permissions"] == {"contents": "read"}
    assert all(
        not reference.lower().startswith("azure/login@")
        for reference in USES.findall(text)
    )
    assert "${{ secrets." not in text


def test_credentialed_jobs_do_not_use_github_environments_or_secrets():
    for name in ("auth-smoke.yml", "deploy.yml", "publish-sdk.yml"):
        text = (WORKFLOWS / name).read_text()
        workflow = yaml.safe_load(text)
        workflow_permissions = workflow.get("permissions", {})
        credentialed_jobs = [
            job
            for job in workflow["jobs"].values()
            if (
                workflow_permissions.get("id-token") == "write"
                or job.get("permissions", {}).get("id-token") == "write"
            )
        ]
        assert credentialed_jobs
        assert "${{ secrets." not in text
        for job in workflow["jobs"].values():
            assert "environment" not in job
