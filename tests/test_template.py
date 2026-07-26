import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "agentic-rag"
IDENTIFIERS = json.loads((ROOT / "platform-identifiers.json").read_text())
ACTION_PIN = re.compile(r"^\s*uses:\s*[^@\s]+@([0-9a-f]{40})", re.MULTILINE)
ACTION_REF = re.compile(r"^\s*uses:\s*[^@\s]+@([^\s]+)", re.MULTILINE)


def test_template_schema_and_source_files_are_valid():
    schema = json.loads((TEMPLATE / "databricks_template_schema.json").read_text())
    assert schema["properties"]["model_provider"]["enum"] == [
        "databricks",
        "foundry",
    ]
    assert schema["properties"]["retrieval_provider"]["enum"] == [
        "azure_ai_search",
        "databricks_ai_search",
    ]
    assert not any(
        "${{ secrets." in path.read_text()
        for path in TEMPLATE.rglob("*")
        if path.is_file()
    )
    deploy = (
        TEMPLATE / "template" / ".github" / "workflows" / "deploy.yml"
    ).read_text()
    assert len(ACTION_REF.findall(deploy)) == len(ACTION_PIN.findall(deploy))
    assert "id-token: write" in deploy
    assert "environment:" not in deploy


@pytest.mark.skipif(shutil.which("databricks") is None, reason="Databricks CLI absent")
def test_template_renders_and_generated_unit_test_passes(tmp_path: Path):
    config = {
        "project_name": "test-agent",
        "application_name": "test-agent",
        "team": "platform",
        "owner_group": "group:platform-owners",
        "cost_center": "CC-1",
        "catalog": "main",
        "schema": "agents",
        "model_provider": "databricks",
        "model_deployment": "chat",
        "foundry_endpoint": "https://unused.services.ai.azure.com",
        "retrieval_provider": "azure_ai_search",
        "search_endpoint": "https://search.search.windows.net",
        "search_index": "knowledge",
        "embedding_deployment": "embedding",
        "aai_core_version": "0.1.0",
        "aai_core_volume": IDENTIFIERS["sdk_artifact_volume"],
        "workspace_host": IDENTIFIERS["databricks_host"],
        "compute_policy_id": IDENTIFIERS["job_compute_policy_id"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "generated"
    environment = dict(os.environ)
    # Local template rendering does not call the workspace, but the CLI still
    # resolves a configuration. Supply a deliberately non-working local config
    # so this test remains credential-free in pull requests.
    environment["DATABRICKS_HOST"] = "https://localhost.invalid"
    environment["DATABRICKS_TOKEN"] = "x"
    environment.pop("DATABRICKS_CONFIG_PROFILE", None)
    environment.pop("DATABRICKS_AUTH_TYPE", None)

    subprocess.run(
        [
            "databricks",
            "bundle",
            "init",
            str(TEMPLATE),
            "--output-dir",
            str(output),
            "--config-file",
            str(config_path),
        ],
        check=True,
        cwd=tmp_path,
        env=environment,
    )

    for path in output.rglob("*"):
        if path.is_file():
            assert "{{." not in path.read_text(), path
    yaml.safe_load((output / "databricks.yml").read_text())
    yaml.safe_load((output / "aai-platform.yml").read_text())
    generated_workflow = (output / ".github" / "workflows" / "deploy.yml").read_text()
    assert "${{ vars.AZURE_CLIENT_ID }}" in generated_workflow

    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(output / "src")]
    )
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        check=True,
        cwd=output,
        env=environment,
    )
    subprocess.run(
        [sys.executable, "-m", "black", "--check", "."],
        check=True,
        cwd=output,
        env=environment,
    )
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(output / "tests")],
        check=True,
        cwd=output,
        env=environment,
    )
