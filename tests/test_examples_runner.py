"""Tests for the clone-friendly learning-example runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "examples.py"


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("aai_examples_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_catalog_separates_offline_connected_and_interactive_examples(runner):
    assert runner.EXAMPLES["offline_hello_world"].connected is False
    assert runner.EXAMPLES["first_trace"].connected is True
    assert runner.EXAMPLES["first_llm_call"].interactive is True


def test_connected_environment_routes_mlflow_to_databricks(runner, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///wrong.db")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "sqlite:///wrong.db")
    monkeypatch.setenv("DATABRICKS_AUTH_TYPE", "pat")

    environment = runner._connected_environment()

    assert environment["MLFLOW_TRACKING_URI"] == "databricks"
    assert environment["MLFLOW_REGISTRY_URI"] == "databricks-uc"
    assert environment["DATABRICKS_AUTH_TYPE"] == "azure-cli"


def test_local_environment_uses_isolated_store(runner, tmp_path, monkeypatch):
    local_dir = tmp_path / ".aai" / "local"
    monkeypatch.setattr(runner, "LOCAL_DIR", local_dir)
    monkeypatch.setattr(runner, "LOCAL_DB", local_dir / "mlflow.db")
    monkeypatch.setattr(runner, "LOCAL_ARTIFACTS", local_dir / "mlruns")

    environment = runner._local_environment()

    assert environment["MLFLOW_TRACKING_URI"] == (
        f"sqlite:///{(local_dir / 'mlflow.db').resolve()}"
    )
    assert environment["MLFLOW_REGISTRY_URI"] == environment["MLFLOW_TRACKING_URI"]
    assert environment["AAI_PLATFORM_CONFIG"] == str(runner.CONFIG_EXAMPLE)


def test_config_preflight_checks_only_fields_used_by_the_example(
    runner, tmp_path, monkeypatch
):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        """
platform:
  experiment_name: /Shared/learning
providers:
  models:
    general-chat:
      deployment: replace-with-serving-endpoint
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "CONFIG", config)

    assert runner._config_issues(runner.EXAMPLES["first_trace"]) == []
    assert runner._config_issues(runner.EXAMPLES["first_llm_call"]) == [
        "Configure `providers.models.general-chat.deployment` in "
        "aai-platform.yml (current value: 'replace-with-serving-endpoint')."
    ]


def test_connect_creates_local_config_once(runner, tmp_path, monkeypatch, capsys):
    config = tmp_path / "aai-platform.yml"
    example = tmp_path / "aai-platform.example.yml"
    example.write_text(
        """
platform:
  experiment_name: /Shared/learning
providers:
  models:
    general-chat:
      deployment: ready-endpoint
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "CONFIG", config)
    monkeypatch.setattr(runner, "CONFIG_EXAMPLE", example)
    monkeypatch.setattr(runner, "_cloud_issues", lambda environment: [])

    assert runner.connect() == 0
    assert config.read_text(encoding="utf-8") == example.read_text(encoding="utf-8")
    assert "Created local configuration" in capsys.readouterr().out

    customized = config.read_text(encoding="utf-8").replace(
        "ready-endpoint", "my-endpoint"
    )
    config.write_text(customized, encoding="utf-8")
    assert runner.connect() == 0
    assert config.read_text(encoding="utf-8") == customized
    assert "Using existing local configuration" in capsys.readouterr().out


def test_connect_returns_nonzero_when_authentication_is_blocked(
    runner, tmp_path, monkeypatch, capsys
):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        """
platform:
  experiment_name: /Shared/learning
providers:
  models:
    general-chat:
      deployment: ready-endpoint
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "CONFIG", config)
    monkeypatch.setattr(
        runner,
        "_cloud_issues",
        lambda environment: ["Azure CLI is not authenticated; run `az login`."],
    )

    assert runner.connect() == 2
    output = capsys.readouterr().out
    assert "Authentication still needed" in output
    assert "az login" in output


def test_connected_run_stops_before_cloud_call_when_config_is_missing(
    runner, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(runner, "CONFIG", tmp_path / "missing.yml")
    monkeypatch.setattr(runner, "_module_issues", lambda example: [])

    def unexpected_cloud_check(environment):
        raise AssertionError("cloud check must not run before local preflight passes")

    monkeypatch.setattr(runner, "_cloud_issues", unexpected_cloud_check)

    assert runner.run_example("first_trace.py") == 2
    output = capsys.readouterr().out
    assert "aai-platform.yml is missing" in output
    assert "make workspace-connect" in output


def test_local_run_never_checks_cloud_and_reports_promotion_path(
    runner, tmp_path, monkeypatch, capsys
):
    local_dir = tmp_path / ".aai" / "local"
    monkeypatch.setattr(runner, "LOCAL_DIR", local_dir)
    monkeypatch.setattr(runner, "LOCAL_DB", local_dir / "mlflow.db")
    monkeypatch.setattr(runner, "LOCAL_ARTIFACTS", local_dir / "mlruns")
    monkeypatch.setattr(runner, "_module_issues", lambda example: [])

    def unexpected_cloud_check(environment):
        raise AssertionError("local execution must not check cloud access")

    monkeypatch.setattr(runner, "_cloud_issues", unexpected_cloud_check)
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        observed["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.run_example("first_trace", destination="local") == 0
    assert observed["command"][0] == sys.executable
    assert observed["cwd"] == local_dir
    assert observed["environment"]["AAI_PLATFORM_CONFIG"] == str(runner.CONFIG_EXAMPLE)
    assert observed["environment"]["MLFLOW_TRACKING_URI"].endswith(
        "/.aai/local/mlflow.db"
    )
    output = capsys.readouterr().out
    assert "make local-ui" in output
    assert "make workspace-connect" in output


def test_local_run_rejects_workspace_only_example(runner, capsys):
    assert runner.run_example("first_prompt", destination="local") == 2
    assert "requires workspace services" in capsys.readouterr().err


def test_interactive_workspace_example_prints_configured_exports(
    runner, tmp_path, monkeypatch, capsys
):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        """
providers:
  models:
    general-chat:
      deployment: ready-endpoint
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "CONFIG", config)
    monkeypatch.setattr(runner, "_module_issues", lambda example: [])
    monkeypatch.setattr(runner, "_cloud_issues", lambda environment: [])

    assert runner.run_example("first_llm_call") == 0
    output = capsys.readouterr().out
    assert (
        f"export DATABRICKS_HOST={runner._identifiers()['databricks_host']}" in output
    )
    assert "export DATABRICKS_AUTH_TYPE=azure-cli" in output
    assert "export MLFLOW_TRACKING_URI=databricks" in output
    assert f"export AAI_PLATFORM_CONFIG={config}" in output
    assert f"jupyter lab {runner.ROOT / 'examples/first_llm_call.ipynb'}" in output


def test_makefile_exposes_single_command_onboarding():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "check-uv:" in makefile
    assert "quickstart: install" in makefile
    assert "local-start: examples-install" in makefile
    assert "local-ui: examples-install" in makefile
    assert "workspace-connect: examples-install" in makefile
    assert "workspace-example: examples-install" in makefile
    assert "--extra databricks --extra genai --locked" in makefile
    assert "$(PYTHON) scripts/examples.py local" in makefile
    assert "$(PYTHON) scripts/examples.py workspace" in makefile
    assert "Example dependencies ready in" in makefile
