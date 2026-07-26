"""Tests for the explicit setup state shared by connected notebooks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "aai_notebook_setup", ROOT / "examples" / "notebook_setup.py"
)
notebook_setup = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = notebook_setup
_spec.loader.exec_module(notebook_setup)


class FakeMlflow:
    def __init__(self):
        self.tracking_uri = ""
        self.registry_uri = ""

    def set_tracking_uri(self, value):
        self.tracking_uri = value

    def get_tracking_uri(self):
        return self.tracking_uri

    def set_registry_uri(self, value):
        self.registry_uri = value

    def get_registry_uri(self):
        return self.registry_uri


class ApiRecord:
    def __init__(self, value):
        self.value = value

    def as_dict(self):
        return self.value


def make_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    (root / "examples").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='example'\n")
    (root / "examples" / "notebook_setup.py").write_text("# marker\n")
    config = root / "aai-platform.yml"
    config.write_text("platform: {}\n")
    (root / "platform-identifiers.json").write_text(
        json.dumps(
            {
                "azure_subscription_id": "subscription-1",
                "azure_tenant_id": "tenant-1",
                "databricks_host": "https://workspace.example",
            }
        )
    )
    return root, config


def make_environment(tmp_path: Path) -> notebook_setup.NotebookEnvironment:
    mlflow = FakeMlflow()
    return notebook_setup.NotebookEnvironment(
        repo_root=tmp_path,
        config_path=tmp_path / "aai-platform.yml",
        identifiers={
            "azure_subscription_id": "subscription-1",
            "azure_tenant_id": "tenant-1",
            "databricks_host": "https://workspace.example",
        },
        tracking_uri="sqlite:///local.db",
        registry_uri="sqlite:///local.db",
        artifact_root=tmp_path / "mlruns",
        evidence_destination=notebook_setup.EvidenceDestination.LOCAL,
        mlflow=mlflow,
    )


def azure_result(*, tenant="tenant-1", subscription="subscription-1"):
    return subprocess.CompletedProcess(
        args=["az", "account", "show"],
        returncode=0,
        stdout=json.dumps(
            {
                "id": subscription,
                "name": "Test subscription",
                "tenantId": tenant,
            }
        ),
        stderr="",
    )


def fake_context(*, deployment="ready-chat", task="llm/v1/chat", ready="READY"):
    endpoint = {
        "name": deployment,
        "task": task,
        "state": {"ready": ready},
    }
    endpoints = SimpleNamespace(
        list=lambda: [ApiRecord(endpoint)],
        get=lambda name: ApiRecord(endpoint),
    )
    workspace = SimpleNamespace(
        current_user=SimpleNamespace(
            me=lambda: SimpleNamespace(user_name="developer@example.invalid")
        ),
        serving_endpoints=endpoints,
        catalogs=SimpleNamespace(
            get=lambda name: SimpleNamespace(name=name),
            list=lambda: [SimpleNamespace(name="dbx_dev")],
        ),
        schemas=SimpleNamespace(
            get=lambda full_name: SimpleNamespace(full_name=full_name),
            list=lambda catalog_name: [SimpleNamespace(name="default")],
        ),
    )
    model = SimpleNamespace(model=deployment)
    return SimpleNamespace(
        workspace=workspace,
        settings=SimpleNamespace(
            catalog="dbx_dev",
            schema="default",
            models={
                "general-chat": {
                    "provider": "databricks",
                    "deployment": deployment,
                }
            },
        ),
        providers=SimpleNamespace(model=lambda logical_name: model),
        tags=SimpleNamespace(),
    )


def test_find_repo_root_searches_upward(tmp_path):
    root, _ = make_repo(tmp_path)
    nested = root / "examples" / "nested"
    nested.mkdir()

    assert notebook_setup.find_repo_root(nested) == root


def test_prepare_environment_routes_all_learning_evidence_locally(
    tmp_path, monkeypatch, capsys
):
    root, config = make_repo(tmp_path)
    mlflow = FakeMlflow()
    runtime = SimpleNamespace(find_platform_config=lambda start: config)

    monkeypatch.setattr(notebook_setup.sys, "path", list(notebook_setup.sys.path))
    monkeypatch.setattr(notebook_setup, "_missing_modules", lambda: [])

    def import_module(name):
        if name == "aai_core.runtime":
            return runtime
        if name == "mlflow":
            return mlflow
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(notebook_setup.importlib, "import_module", import_module)

    environment = notebook_setup.prepare_notebook_environment(root)

    expected_uri = f"sqlite:///{root / '.aai' / 'local' / 'mlflow.db'}"
    assert environment.tracking_uri == expected_uri
    assert environment.registry_uri == expected_uri
    assert environment.artifact_root == root / ".aai" / "local" / "mlruns"
    assert environment.evidence_destination is notebook_setup.EvidenceDestination.LOCAL
    assert environment.artifact_root.is_dir()
    assert notebook_setup.os.environ["MLFLOW_TRACKING_URI"] == expected_uri
    assert notebook_setup.os.environ["MLFLOW_REGISTRY_URI"] == expected_uri
    assert notebook_setup.os.environ["DATABRICKS_AUTH_TYPE"] == "azure-cli"
    assert str(root) in notebook_setup.sys.path
    assert "SETUP PASSED" in capsys.readouterr().out


def test_prepare_environment_routes_prompts_runs_and_traces_to_databricks(
    tmp_path, monkeypatch, capsys
):
    root, config = make_repo(tmp_path)
    mlflow = FakeMlflow()
    runtime = SimpleNamespace(find_platform_config=lambda start: config)

    monkeypatch.setattr(notebook_setup.sys, "path", list(notebook_setup.sys.path))
    monkeypatch.setattr(notebook_setup, "_missing_modules", lambda: [])

    def import_module(name):
        if name == "aai_core.runtime":
            return runtime
        if name == "mlflow":
            return mlflow
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(notebook_setup.importlib, "import_module", import_module)

    environment = notebook_setup.prepare_notebook_environment(
        root,
        evidence_destination="databricks",
    )

    assert environment.tracking_uri == "databricks"
    assert environment.registry_uri == "databricks-uc"
    assert (
        environment.evidence_destination
        is notebook_setup.EvidenceDestination.DATABRICKS
    )
    assert notebook_setup.os.environ["MLFLOW_TRACKING_URI"] == "databricks"
    assert notebook_setup.os.environ["MLFLOW_REGISTRY_URI"] == "databricks-uc"
    assert "Databricks-managed" in capsys.readouterr().out


def test_prepare_environment_reports_missing_kernel_modules(tmp_path, monkeypatch):
    root, _ = make_repo(tmp_path)
    monkeypatch.setattr(
        notebook_setup,
        "_missing_modules",
        lambda: ["databricks_openai"],
    )

    with pytest.raises(RuntimeError, match="make examples-install") as error:
        notebook_setup.prepare_notebook_environment(root)

    assert ".venv/bin/python" in str(error.value)


def test_preflight_resolves_model_only_after_access_checks(
    tmp_path, monkeypatch, capsys
):
    environment = make_environment(tmp_path)
    context = fake_context()
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setattr(
        notebook_setup.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            lifecycle_experiment_name=lambda ctx: "/Shared/earnings"
        ),
    )

    connected = notebook_setup.preflight_databricks(
        environment,
        which=lambda name: "/usr/bin/az",
        run_command=lambda *args, **kwargs: azure_result(),
        bootstrap_fn=lambda path: context,
    )

    assert connected.context is context
    assert connected.model.model == "ready-chat"
    assert connected.experiment_name == "/Shared/earnings"
    assert connected.endpoint["state"]["ready"] == "READY"
    assert "PREFLIGHT PASSED" in capsys.readouterr().out


def test_remote_preflight_verifies_prompt_namespace(tmp_path, monkeypatch):
    environment = replace(
        make_environment(tmp_path),
        tracking_uri="databricks",
        registry_uri="databricks-uc",
        evidence_destination=notebook_setup.EvidenceDestination.DATABRICKS,
    )
    context = fake_context()
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setattr(
        notebook_setup.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            lifecycle_experiment_name=lambda ctx: "/Shared/earnings"
        ),
    )

    connected = notebook_setup.preflight_databricks(
        environment,
        which=lambda name: "/usr/bin/az",
        run_command=lambda *args, **kwargs: azure_result(),
        bootstrap_fn=lambda path: context,
    )

    assert connected.context.settings.catalog == "dbx_dev"
    assert connected.context.settings.schema == "default"


@pytest.mark.parametrize(
    ("tenant", "subscription", "message"),
    (
        ("wrong-tenant", "subscription-1", "az login --tenant tenant-1"),
        ("tenant-1", "wrong-subscription", "az account set --subscription"),
    ),
)
def test_preflight_reports_wrong_azure_context(tmp_path, tenant, subscription, message):
    environment = make_environment(tmp_path)

    with pytest.raises(RuntimeError, match=message):
        notebook_setup.preflight_databricks(
            environment,
            which=lambda name: "/usr/bin/az",
            run_command=lambda *args, **kwargs: azure_result(
                tenant=tenant,
                subscription=subscription,
            ),
            bootstrap_fn=lambda path: fake_context(),
        )


def test_preflight_placeholder_lists_ready_chat_endpoints(tmp_path):
    environment = make_environment(tmp_path)
    context = fake_context(deployment="ready-chat")
    context.settings.models["general-chat"][
        "deployment"
    ] = "replace-with-serving-endpoint"

    with pytest.raises(RuntimeError, match="READY chat endpoints visible") as error:
        notebook_setup.preflight_databricks(
            environment,
            which=lambda name: "/usr/bin/az",
            run_command=lambda *args, **kwargs: azure_result(),
            bootstrap_fn=lambda path: context,
        )

    assert "ready-chat" in str(error.value)
