"""Credential-free checks for the Microsoft Foundry notebook curriculum."""

from __future__ import annotations

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
        )
    )


def test_example_configuration_is_portable_and_project_scoped():
    path = CURRICULUM / "config" / "aai-platform.dev.example.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = document["providers"]["models"]["foundry-chat"]

    assert model["provider"] == "foundry"
    assert "/api/projects/" in model["endpoint"]
    assert model["deployment"].startswith("replace-")
    assert document["platform"]["repository"] == (
        "replace-with-owner/replace-with-repository"
    )
    assert document.get("secrets") == {}


def test_readme_documents_the_required_local_config_copy():
    readme = (CURRICULUM / "README.md").read_text(encoding="utf-8")

    assert (
        "cp examples/foundry-curriculum/config/aai-platform.dev.example.yml "
        "examples/foundry-curriculum/config/aai-platform.dev.yml"
    ) in readme


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


def test_curriculum_has_nine_clean_compilable_notebooks():
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
        "08_agentops_release_gate.ipynb",
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
            )


def _notebook_source(name: str) -> str:
    notebook = json.loads((CURRICULUM / "notebooks" / name).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


@pytest.mark.parametrize(
    "name",
    (
        "04_agents_tools_and_mcp.ipynb",
        "05_evaluation_safety_and_red_team.ipynb",
        "06_observability_and_genaiops.ipynb",
        "08_agentops_release_gate.ipynb",
    ),
)
def test_taught_notebooks_narrate_every_result(name):
    """A bare dict literal teaches nothing. Each executable result is framed by
    prose that reads the output back and names one thing to change and re-run,
    so a learner is never left inferring why the cell was written that way."""
    notebook = json.loads((CURRICULUM / "notebooks" / name).read_text(encoding="utf-8"))
    cells = notebook["cells"]
    source = _notebook_source(name)

    # A notebook must never end on a result nobody explains.
    assert cells[-1]["cell_type"] == "markdown"

    # Every code cell other than the shared loader is followed by narration,
    # and each of those narrations reads the output back and names a mutation.
    taught = [
        cell
        for cell in cells
        if cell["cell_type"] == "code" and cell["id"] != "load-config"
    ]
    assert taught, f"{name}: nothing executable to narrate"
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code" or cell["id"] == "load-config":
            continue
        assert (
            cells[index + 1]["cell_type"] == "markdown"
        ), f"{name}: code cell {cell['id']!r} has no read-back cell"

    assert source.count("### What you just saw") >= len(taught)
    assert source.count("### Change this and re-run") >= len(taught)


def test_release_gate_notebook_names_the_right_package_and_the_platform_rules():
    """Two unrelated projects are called AgentOps, and `pip install agentops`
    silently fetches the wrong one. The generated PR workflow also breaks three
    of this repository's CI rules, so the notebook must teach the reconciliation
    rather than hand a learner a workflow they must not commit here."""
    source = _notebook_source("08_agentops_release_gate.ipynb")

    assert "agentops-accelerator[foundry,agent]" in source
    assert "docs.agentops.ai" in source

    for rule in (
        "credential-free-pull-requests",
        "no-environment-on-credentialed-jobs",
        "actions-pinned-to-commit-sha",
    ):
        assert rule in source

    # The exit-code contract is the reason the gate can be trusted at all.
    assert "do not merge" in source
    # Framework choice must stay orthogonal to how the agent is gated.
    assert "http-json" in source
    assert "LangGraph" in source


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
