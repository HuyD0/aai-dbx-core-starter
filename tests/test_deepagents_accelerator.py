"""Credential-free contract tests for the Deep Agents solution accelerator.

The accelerator is connected-only, so CI cannot execute it. What CI *can*
guarantee is that it stays discoverable, safe, and honest: every standalone
example is reachable from the examples index (this accelerator once was not),
the notebooks stay output-free and compilable with no secret material, the
workspace `%pip` stack stays exact-pinned, and the pins that overlap
`dependency-policy.toml` track the certified line instead of drifting.
"""

from __future__ import annotations

import json
import re
import shlex
import tomllib
from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
NOTEBOOKS = EXAMPLES / "deepagents-solution-accelerator" / "notebooks"
EXPECTED_NOTEBOOKS = (
    "01_agent_setup_and_definition.ipynb",
    "02_deployment_and_trace_logging.ipynb",
    "03_continuous_eval_and_feedback_loop.ipynb",
)
# Directories of curriculum-internal material rather than standalone examples.
_NOT_STANDALONE = {"support", "workshops"}
_REQUIREMENT = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[A-Za-z0-9,._ -]+\])?==([A-Za-z0-9.]+)$"
)


def _cell_source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def _load_notebooks() -> dict[str, dict]:
    return {
        name: json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
        for name in EXPECTED_NOTEBOOKS
    }


def test_standalone_examples_have_a_readme_and_an_index_link():
    index = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for directory in sorted(path for path in EXAMPLES.iterdir() if path.is_dir()):
        # Skip caches and other tooling directories, not just curriculum ones.
        if directory.name in _NOT_STANDALONE or directory.name.startswith((".", "_")):
            continue
        assert (directory / "README.md").is_file(), (
            f"examples/{directory.name} has no README.md; an example without one "
            "cannot explain its boundaries or prerequisites"
        )
        assert f"({directory.name}/README.md)" in index, (
            f"examples/README.md does not link examples/{directory.name}; an "
            "unlinked example is invisible to developers"
        )


def test_accelerator_notebooks_are_safe_clean_and_compilable():
    assert sorted(path.name for path in NOTEBOOKS.glob("*.ipynb")) == list(
        EXPECTED_NOTEBOOKS
    )
    for name, notebook in _load_notebooks().items():
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(cell_ids) and len(cell_ids) == len(set(cell_ids))
        source = "\n".join(_cell_source(cell) for cell in notebook["cells"])
        assert "DATABRICKS_TOKEN" not in source
        assert "AZURE_CLIENT_SECRET" not in source
        assert "OPENAI_API_KEY" not in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            # Databricks magics (%pip, %restart_python) are not Python; comment
            # them out line-preserving so the rest of the cell must compile.
            python_source = "\n".join(
                f"# {line}" if line.lstrip().startswith("%") else line
                for line in _cell_source(cell).splitlines()
            )
            compile(
                python_source,
                f"{name}:code-cell-{index}",
                "exec",
                flags=PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )


def _pip_requirements(notebook: dict, name: str) -> list[str]:
    lines = [
        stripped
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for line in _cell_source(cell).splitlines()
        if (stripped := line.strip()).startswith("%pip install")
    ]
    assert len(lines) == 1, f"{name} must install its runtime stack exactly once"
    return shlex.split(lines[0][len("%pip install") :])


def test_workspace_pip_stack_is_exact_pinned_to_the_certified_line():
    policy = tomllib.loads(
        (ROOT / "dependency-policy.toml").read_text(encoding="utf-8")
    )["packages"]
    for name, notebook in _load_notebooks().items():
        requirements = _pip_requirements(notebook, name)
        assert requirements
        for requirement in requirements:
            match = _REQUIREMENT.fullmatch(requirement)
            assert match, f"{name} requirement {requirement!r} is not `==`-pinned"
            package, _, version = match.groups()
            if package in policy:
                assert version == policy[package]["certified"], (
                    f"{name} pins {package}=={version} but the certified version "
                    f"is {policy[package]['certified']}; re-verify the "
                    "accelerator on the certified line or update both together"
                )


def test_accelerator_notebooks_stay_clone_portable():
    identifiers = json.loads(
        (ROOT / "platform-identifiers.json").read_text(encoding="utf-8")
    )
    forbidden = (
        "azure_tenant_id",
        "azure_subscription_id",
        "databricks_host",
        "databricks_uat_host",
        "job_compute_policy_id",
        "sdk_artifact_volume",
    )
    for name in EXPECTED_NOTEBOOKS:
        source = (NOTEBOOKS / name).read_text(encoding="utf-8")
        for key in forbidden:
            assert str(identifiers[key]) not in source, (
                f"{name} restates {key}; workspace resources arrive through "
                "widgets so a clone never edits the notebooks"
            )
