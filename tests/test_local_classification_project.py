"""Static repository contract for the standalone local classification course."""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "examples" / "local-classification"


def test_local_classification_project_has_complete_standalone_contract():
    expected = {
        "README.md",
        ".gitignore",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        "configs/project.yaml",
        "data/README.md",
        "docs/curriculum.md",
        "docs/resources.md",
        "docs/databricks-handoff.md",
        "docs/model-card.md",
        "notebooks/README.md",
        "scripts/render_notebooks.py",
        "scripts/check_notebooks.py",
        "src/aai_local_classification/data.py",
        "src/aai_local_classification/evaluation.py",
        "src/aai_local_classification/policy.py",
        "src/aai_local_classification/workflow.py",
        "tests/test_workflow.py",
    }
    assert not sorted(
        relative for relative in expected if not (PROJECT / relative).is_file()
    )


def test_course_uses_exact_isolated_dependencies():
    with (PROJECT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    dependencies = project["project"]["dependencies"]
    assert "mlflow==3.14.0" in dependencies
    assert "pandas==2.3.3" in dependencies
    assert "scikit-learn==1.9.0" in dependencies
    assert all("aai-core" not in dependency for dependency in dependencies)

    with (PROJECT / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    versions = {package["name"]: package["version"] for package in lock["package"]}
    assert versions["mlflow"] == "3.14.0"
    assert versions["pandas"] == "2.3.3"
    assert versions["scikit-learn"] == "1.9.0"


def test_notebook_course_is_clean_ordered_and_offline():
    notebooks = sorted((PROJECT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    assert [path.name[:2] for path in notebooks] == [
        f"{number:02d}" for number in range(10)
    ]
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["metadata"]["aai_course"]["network_required"] is False
        assert notebook["metadata"]["aai_course"]["cloud_credentials_required"] is False
        ids = [cell["id"] for cell in notebook["cells"]]
        assert len(ids) == len(set(ids))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


def test_packaged_python_and_notebook_sources_compile():
    for path in sorted((PROJECT / "src").rglob("*.py")) + sorted(
        (PROJECT / "scripts").glob("*.py")
    ):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for notebook_path in sorted((PROJECT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = cell["source"]
                if isinstance(source, list):
                    source = "".join(source)
                ast.parse(source, filename=f"{notebook_path}:{cell['id']}")


def test_generated_and_sensitive_assets_are_excluded():
    ignore = (PROJECT / ".gitignore").read_text(encoding="utf-8")
    for entry in (".venv/", "data/processed/", ".aai/", ".env", "*.env"):
        assert entry in ignore

    authored_paths = [PROJECT / "README.md", PROJECT / "pyproject.toml"]
    for directory in ("configs", "docs", "scripts", "src", "tests"):
        authored_paths.extend(
            path
            for path in (PROJECT / directory).rglob("*")
            if path.is_file() and path.suffix in {".md", ".toml", ".yaml", ".py"}
        )
    authored = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(authored_paths)
    )
    assert "DATABRICKS_TOKEN=" not in authored
    assert "MLFLOW_TRACKING_TOKEN=" not in authored
    assert "client_secret:" not in authored.lower()


def test_root_exposes_course_commands_and_excludes_it_from_sdk_sdist():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in (
        "classification-install:",
        "classification-prepare:",
        "classification-train:",
        "classification-check:",
        "classification-notebook:",
        "classification-ui:",
    ):
        assert target in makefile

    with (ROOT / "pyproject.toml").open("rb") as stream:
        root_project = tomllib.load(stream)
    excluded = root_project["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    assert "/examples/local-classification" in excluded
    assert "/tests/test_local_classification_project.py" in excluded
