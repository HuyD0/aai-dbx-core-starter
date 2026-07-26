"""Tests for the clone-friendly learning-example runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
    assert "make examples-connect" in output


def test_makefile_exposes_single_command_onboarding():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "check-uv:" in makefile
    assert "quickstart: install" in makefile
    assert "examples-connect: examples-install" in makefile
    assert "example: examples-install" in makefile
    assert "--extra databricks --extra genai --locked" in makefile
    assert "$(PYTHON) scripts/examples.py run" in makefile
    assert "Example dependencies ready in" in makefile
