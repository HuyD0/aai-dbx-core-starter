"""Static repository contract for the standalone fine-tuning course."""

from __future__ import annotations

import ast
import importlib.util
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "examples" / "fine-tuning"


def _lessons() -> dict:
    """Import the stdlib-only lesson sources without touching sys.path."""
    spec = importlib.util.spec_from_file_location(
        "finetuning_notebook_content", PROJECT / "scripts" / "notebook_content.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LESSONS


def test_finetuning_project_has_complete_standalone_contract():
    expected = {
        "README.md",
        ".gitignore",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        "scripts/doctor.py",
        "scripts/notebook_content.py",
        "scripts/render_notebooks.py",
        "scripts/check_notebooks.py",
        "src/aai_fine_tuning/__init__.py",
        "src/aai_fine_tuning/cli.py",
        "src/aai_fine_tuning/memory.py",
        "src/aai_fine_tuning/settings.py",
        "tests/test_memory.py",
        "tests/test_notebook_contract.py",
    }
    assert not sorted(
        relative for relative in expected if not (PROJECT / relative).is_file()
    )


def test_course_uses_exact_isolated_cpu_only_dependencies():
    with (PROJECT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    dependencies = project["project"]["dependencies"]
    assert "torch==2.13.0" in dependencies
    assert "transformers==5.15.1" in dependencies
    assert "peft==0.20.0" in dependencies
    assert "mlflow==3.14.0" in dependencies
    assert all("aai-core" not in dependency for dependency in dependencies)

    with (PROJECT / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])
    # Linux resolves the CPU-only torch build; macOS the PyPI build of the
    # same release. Nothing NVIDIA/CUDA may ever enter this lock.
    assert versions["torch"] == {"2.13.0", "2.13.0+cpu"}
    assert versions["transformers"] == {"5.15.1"}
    assert versions["peft"] == {"0.20.0"}
    assert versions["mlflow"] == {"3.14.0"}
    assert not [name for name in versions if name.startswith("nvidia-")]
    assert not [name for name in versions if "cuda" in name]


def test_notebook_course_is_clean_ordered_and_offline():
    lessons = _lessons()
    names = sorted(lessons)
    assert [name[:2] for name in names] == [
        f"{number:02d}" for number in range(len(names))
    ]
    notebooks = sorted((PROJECT / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebooks] == names
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        course = notebook["metadata"]["aai_course"]
        assert course["schema_version"] == 1
        assert course["network_required_after_install"] is False
        assert course["cloud_credentials_required"] is False
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
    for entry in (".venv/", ".aai/", ".env", "*.env"):
        assert entry in ignore

    authored_paths = [PROJECT / "README.md", PROJECT / "pyproject.toml"]
    for directory in ("scripts", "src", "tests"):
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


def test_course_makefile_enforces_isolation_and_the_offline_guard():
    course_makefile = (PROJECT / "Makefile").read_text(encoding="utf-8")
    assert 'AAI_FINETUNE_PROJECT_ROOT="$(COURSE_ROOT)"' in course_makefile
    assert "HF_HUB_OFFLINE=1" in course_makefile
    assert "--no-browser" not in course_makefile


def test_root_exposes_course_commands_and_excludes_it_from_sdk_sdist():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in (
        "finetune-install:",
        "finetune-doctor:",
        "finetune-check:",
        "finetune-notebook:",
        "finetune-ui:",
        "finetune-reset:",
    ):
        assert target in makefile

    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "finetuning-course:" in ci
    assert "make -C examples/fine-tuning check" in ci
    assert "make -C examples/fine-tuning notebook-check" in ci

    with (ROOT / "pyproject.toml").open("rb") as stream:
        root_project = tomllib.load(stream)
    excluded = root_project["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    assert "/examples/fine-tuning" in excluded
    assert "/tests/test_finetuning_project.py" in excluded
