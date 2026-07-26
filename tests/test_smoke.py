"""Credential-free regression tests for the platform's security boundaries."""

import json
import re
import runpy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SHA_PIN = re.compile(r"^\s*uses:\s*[^@\s]+@([0-9a-f]{40})(?:\s+#.*)?$", re.MULTILINE)
USES = re.compile(r"^\s*uses:\s*[^@\s]+@([^\s]+)", re.MULTILINE)
# Single source of truth for environment-specific identifiers. These tests
# cross-check every other occurrence against it so a clone that edits the
# fixture is pointed at each file that must agree.
IDENTIFIERS = json.loads((ROOT / "platform-identifiers.json").read_text())


def load_yaml(relative_path):
    with (ROOT / relative_path).open() as stream:
        return yaml.safe_load(stream)


def test_sample_notebook_runs(capsys):
    runpy.run_path(str(ROOT / "src" / "notebooks" / "sample_etl.py"))
    assert capsys.readouterr().out.strip().endswith("package import verified")


def test_first_llm_notebook_is_valid_safe_and_output_free():
    notebook = json.loads(
        (ROOT / "examples" / "first_llm_call.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    assert notebook["cells"]

    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'ctx.providers.model("general-chat")' in source
    assert "model.generate(" in source
    assert "DATABRICKS_TOKEN" not in source
    assert "AZURE_CLIENT_SECRET" not in source

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)


def test_dev_target_is_pinned_to_dev_workspace():
    bundle = load_yaml("databricks.yml")
    host = bundle["targets"]["dev"]["workspace"]["host"]
    # workspace.host must stay a literal (the Databricks CLI forbids variable
    # interpolation in authentication fields), so this cross-check keeps it in
    # sync with the identifiers fixture.
    assert host == IDENTIFIERS["databricks_host"]
    assert host.startswith("https://") and host.endswith(".azuredatabricks.net")


def test_sample_job_uses_constrained_job_compute_policy():
    bundle = load_yaml("databricks.yml")
    resources = load_yaml("resources/sample_job.yml")
    cluster = resources["resources"]["jobs"]["aai_dbx_base_template_sample"][
        "job_clusters"
    ][0]["new_cluster"]
    assert cluster["policy_id"] == "${var.job_compute_policy_id}"
    assert (
        bundle["variables"]["job_compute_policy_id"]["default"]
        == IDENTIFIERS["job_compute_policy_id"]
    )
    assert cluster["num_workers"] == 1
    assert cluster["spark_version"] == "18.0.x-scala2.13"
    assert "spark_conf" not in cluster


def test_identifier_fixture_is_the_single_source_of_truth():
    """Every other file holding an environment identifier must agree with
    platform-identifiers.json; a clone edits the fixture and this test lists
    each remaining literal that must follow."""
    schema = json.loads(
        (
            ROOT / "templates" / "agentic-rag" / "databricks_template_schema.json"
        ).read_text()
    )
    defaults = {name: prop["default"] for name, prop in schema["properties"].items()}
    assert defaults["workspace_host"] == IDENTIFIERS["databricks_host"]
    assert defaults["compute_policy_id"] == IDENTIFIERS["job_compute_policy_id"]
    assert defaults["aai_core_volume"] == IDENTIFIERS["sdk_artifact_volume"]

    verify = (ROOT / "scripts" / "cloud-verify.sh").read_text()
    assert "platform-identifiers.json" in verify
    for value in (
        IDENTIFIERS["azure_tenant_id"],
        IDENTIFIERS["azure_subscription_id"],
        IDENTIFIERS["databricks_host"],
    ):
        assert (
            value not in verify
        ), "cloud-verify.sh must read the fixture, not inline ids"


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


def test_cloud_environment_is_reproducible_and_credential_free():
    setup = (ROOT / "scripts" / "codex-cloud-setup.sh").read_text()
    maintenance = (ROOT / "scripts" / "codex-cloud-maintenance.sh").read_text()
    verify = (ROOT / "scripts" / "cloud-verify.sh").read_text()
    ci = (WORKFLOWS / "ci.yml").read_text()

    for version in ("0.8.23", "1.12.2", "1.6.0", "2.88.0"):
        assert version in setup

    assert "sha256sum --check" in setup
    assert "codex-cloud-setup.sh" in maintenance
    assert "./scripts/cloud-verify.sh" in ci
    assert "AZURE_CLIENT_SECRET" in verify
    assert "DATABRICKS_TOKEN" in verify
    assert "azure/login" not in verify.lower()
    assert "az login" not in verify.lower()
