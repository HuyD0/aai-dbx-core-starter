"""Credential-free checks for the Microsoft Foundry notebook curriculum."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = ROOT / "examples" / "foundry-curriculum"
SETUP = CURRICULUM / "notebook_setup.py"

_spec = importlib.util.spec_from_file_location("foundry_notebook_setup", SETUP)
assert _spec is not None and _spec.loader is not None
setup = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = setup
_spec.loader.exec_module(setup)


def _fake_context(config_path: Path):
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return SimpleNamespace(
        settings=SimpleNamespace(
            models=document["providers"]["models"],
            azure_identity=document["platform"]["azure_identity"],
            raw=document,
        )
    )


def test_example_configuration_is_portable_and_project_scoped():
    path = CURRICULUM / "config" / "aai-platform.dev.example.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    platform = document["platform"]
    model = document["providers"]["models"]["foundry-chat"]

    assert platform["repository"] == "replace-with-owner/replace-with-repository"
    assert platform["catalog"] == "replace-with-catalog"
    assert model["provider"] == "foundry"
    assert "/api/projects/" in model["endpoint"]
    assert model["deployment"].startswith("replace-")
    assert document.get("secrets") == {}
    assert document["foundry"]["a2a"]["protocol_version"] == "1.0"
    assert document["foundry"]["agent"]["version"].startswith("replace-")
    assert document["foundry"]["evaluation"]["evaluator_model"].startswith("replace-")
    assert document["foundry"]["observability"][
        "application_insights_resource_id"
    ].startswith("replace-")


def test_readme_copies_the_portable_example_before_opening_notebooks():
    readme = (CURRICULUM / "README.md").read_text(encoding="utf-8")
    copy_source = "examples/foundry-curriculum/config/aai-platform.dev.example.yml"
    copy_target = "examples/foundry-curriculum/config/aai-platform.dev.yml"

    assert f"cp {copy_source} \\\n  {copy_target}" in readme
    assert f"Edit `{copy_target}`, not the tracked" in readme
    assert readme.index(copy_source) < readme.index("Open the repository")

    ignore = (CURRICULUM / "config" / ".gitignore").read_text(encoding="utf-8")
    assert "aai-platform.*.yml" in ignore
    assert "!aai-platform.*.example.yml" in ignore


def test_session_loads_endpoint_only_from_selected_configuration(tmp_path):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        """
platform:
  azure_identity: azure_cli
providers:
  models:
    foundry-chat:
      provider: foundry
      endpoint: https://account.services.ai.azure.com/api/projects/project-dev
      deployment: chat-model
""",
        encoding="utf-8",
    )

    session = setup.load_session(
        CURRICULUM,
        config_path=config,
        bootstrap_fn=_fake_context,
    )

    assert session.project_endpoint.endswith("/api/projects/project-dev")
    assert session.deployment == "chat-model"
    assert session.connected_ready
    assert session.safe_summary()["azure_identity"] == "azure_cli"
    assert not session.agent_ready
    assert not session.a2a_ready


def test_session_loads_advanced_identifiers_and_derives_a2a_urls(tmp_path):
    config = tmp_path / "aai-platform.yml"
    app_insights_resource_id = (
        "/subscriptions/00000000-0000-0000-0000-000000000000/"
        "resourceGroups/rg-foundry/providers/microsoft.insights/"
        "components/appi-foundry"
    )
    config.write_text(
        f"""
platform:
  azure_identity: azure_cli
providers:
  models:
    foundry-chat:
      provider: foundry
      endpoint: https://account.services.ai.azure.com/api/projects/project-dev
      deployment: chat-model
foundry:
  agent:
    name: curriculum-agent
    version: "2"
    id: curriculum-agent:2
  evaluation:
    evaluator_model: judge-model
  memory:
    store_name: learner-memory
  a2a:
    remote_agent_name: policy-specialist
    connection_name: policy-specialist-a2a
    protocol_version: "1.0"
  observability:
    application_insights_resource_id: {app_insights_resource_id}
""",
        encoding="utf-8",
    )

    session = setup.load_session(
        CURRICULUM,
        config_path=config,
        bootstrap_fn=_fake_context,
    )

    assert session.agent_ready
    assert session.trace_ready
    assert session.evaluation_ready
    assert session.a2a_ready
    assert session.observability_ready
    assert session.agent_reference() == {
        "type": "agent_reference",
        "name": "curriculum-agent",
        "version": "2",
    }
    assert session.a2a_base_url == (
        "https://account.services.ai.azure.com/api/projects/project-dev/agents/"
        "policy-specialist/endpoint/protocols/a2a"
    )
    assert session.a2a_agent_card_url.endswith("/agentCard/v1.0")


@pytest.mark.parametrize(
    "endpoint",
    (
        "http://account.services.ai.azure.com/api/projects/project-dev",
        "https://account.services.ai.azure.com",
        "https://example.com/api/projects/project-dev",
        "https://user:password@account.services.ai.azure.com/api/projects/project-dev",
    ),
)
def test_session_rejects_non_project_or_unsafe_endpoints(tmp_path, endpoint):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        f"""
platform:
  azure_identity: azure_cli
providers:
  models:
    foundry-chat:
      provider: foundry
      endpoint: {endpoint}
      deployment: chat-model
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="endpoint|HTTPS|host|project"):
        setup.load_session(
            CURRICULUM,
            config_path=config,
            bootstrap_fn=_fake_context,
        )


def test_connected_call_requires_an_explicit_network_opt_in():
    session = setup.FoundryNotebookSession(
        curriculum_root=CURRICULUM,
        config_path=Path("config.yml"),
        logical_model="foundry-chat",
        project_endpoint="https://account.services.ai.azure.com/api/projects/dev",
        deployment="chat-model",
        context=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="disabled"):
        setup.create_text_response(session, "hello")


@pytest.mark.parametrize(
    ("project_endpoint", "deployment"),
    (
        (
            "https://replace-with-foundry-account.services.ai.azure.com/"
            "api/projects/project-dev",
            "chat-model",
        ),
        (
            "https://account.services.ai.azure.com/api/projects/"
            "replace-with-project",
            "chat-model",
        ),
        (
            "https://account.services.ai.azure.com/api/projects/project-dev",
            "replace-with-model-deployment",
        ),
    ),
)
def test_connected_call_rejects_placeholder_configuration(project_endpoint, deployment):
    session = setup.FoundryNotebookSession(
        curriculum_root=CURRICULUM,
        config_path=Path("config.yml"),
        logical_model="foundry-chat",
        project_endpoint=project_endpoint,
        deployment=deployment,
        context=SimpleNamespace(),
    )

    assert not session.connected_ready
    with pytest.raises(RuntimeError, match="endpoint.*deployment"):
        setup.create_text_response(session, "hello", allow_network=True)


def test_curriculum_has_thirteen_clean_compilable_notebooks():
    notebooks = sorted((CURRICULUM / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebooks] == [
        "00_setup_and_architecture.ipynb",
        "01_models_and_prompting.ipynb",
        "02_responses_and_structured_outputs.ipynb",
        "03_rag_and_retrieval_security.ipynb",
        "04_agents_tools_and_mcp.ipynb",
        "05_evaluation_safety_and_red_team.ipynb",
        "06_observability_and_genaiops.ipynb",
        "07_capstone_release_gate.ipynb",
        "08_context_engineering_and_memory.ipynb",
        "09_foundry_a2a_and_handoffs.ipynb",
        "10_foundry_native_evaluation.ipynb",
        "11_mlflow_tracing_and_genai_evaluation.ipynb",
        "12_dual_otel_export_foundry_and_mlflow.ipynb",
    ]

    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(ids)
        assert len(ids) == len(set(ids))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert "load_session(" in source
        assert ".services.ai.azure.com/api/projects/" not in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile(
                "".join(cell.get("source", [])),
                f"{path.name}:code-cell-{index}",
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )


def test_advanced_notebooks_are_opt_in_and_do_not_provision_a2a():
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (CURRICULUM / "notebooks").glob("*.ipynb")
        if path.name[:2] in {"08", "09", "10", "11", "12"}
    }

    assert (
        '"RUN_CONNECTED = False' in sources["08_context_engineering_and_memory.ipynb"]
    )
    assert '"RUN_A2A_CONNECTED = False' in sources["09_foundry_a2a_and_handoffs.ipynb"]
    assert '"RUN_FOUNDRY_EVAL = False' in sources["10_foundry_native_evaluation.ipynb"]
    assert (
        '"RUN_MLFLOW_EVAL = False'
        in sources["11_mlflow_tracing_and_genai_evaluation.ipynb"]
    )
    assert (
        '"RUN_DUAL_EXPORT = False'
        in sources["12_dual_otel_export_foundry_and_mlflow.ipynb"]
    )
    assert "'canceled'" in sources["10_foundry_native_evaluation.ipynb"]
    assert (
        "AZURE_EXPERIMENTAL_ENABLE_GENAI_TRACING"
        in sources["12_dual_otel_export_foundry_and_mlflow.ipynb"]
    )
    assert all("update_details(" not in source for source in sources.values())
    assert all(
        "project.agents.create_version(" not in source for source in sources.values()
    )


def test_evaluation_starter_has_twenty_cases_and_four_attacks():
    records = [
        json.loads(line)
        for line in (CURRICULUM / "data" / "evaluation_cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert len(records) == 20
    assert len({record["case_id"] for record in records}) == 20
    assert sum(record["category"] == "adversarial" for record in records) == 4
    assert all("expectations" in record for record in records)


@pytest.mark.parametrize("dataset_name", ("context_cases.jsonl", "a2a_cases.jsonl"))
def test_advanced_datasets_use_mlflow_standard_shape(dataset_name):
    records = [
        json.loads(line)
        for line in (CURRICULUM / "data" / dataset_name)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert len(records) == 8
    assert len({record["case_id"] for record in records}) == 8
    assert {record["split"] for record in records} == {"regression", "validation"}
    assert all(
        set(record) >= {"inputs", "expectations", "critical"} for record in records
    )
    assert any(record["critical"] for record in records)


def test_current_practices_cites_primary_foundry_and_mlflow_sources():
    guide = (CURRICULUM / "CURRENT_PRACTICES.md").read_text(encoding="utf-8")

    assert "2026-08-01" in guide
    assert "learn.microsoft.com/en-us/azure/foundry" in guide
    assert "mlflow.org/docs/latest" in guide
    assert "docs.databricks.com" in guide
    assert "Application Insights" in guide
    assert "backend synchronization" in guide
