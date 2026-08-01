"""Static contracts for the narrative, offline-first notebook course."""

from __future__ import annotations

import json
import re
from pathlib import Path

from nbconvert import HTMLExporter

from aai_local_finetuning.settings import PROJECT_ROOT

EXPECTED_NOTEBOOKS = (
    "00_start_here.ipynb",
    "01_dataset_provenance_and_license.ipynb",
    "02_dataset_exploration_and_validation.ipynb",
    "03_leakage_safe_splits.ipynb",
    "04_deterministic_baselines.ipynb",
    "05_prompt_baselines.ipynb",
    "06_lora_finetuning.ipynb",
    "07_frozen_evaluation.ipynb",
    "08_mlflow_and_promotion.ipynb",
    "09_capstone_policy_dataset.ipynb",
    "10_capstone_model_vs_hybrid.ipynb",
    "11_design_the_next_project.ipynb",
)


def _notebooks() -> list[tuple[Path, dict]]:
    directory = PROJECT_ROOT / "notebooks"
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("[0-9][0-9]_*.ipynb"))
    ]


def _source(cell: dict) -> str:
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else str(value)


def test_notebook_sequence_and_prerequisites_are_contiguous():
    notebooks = _notebooks()

    assert tuple(path.name for path, _ in notebooks) == EXPECTED_NOTEBOOKS
    seen: set[str] = set()
    stages = []
    for expected_order, (path, notebook) in enumerate(notebooks):
        metadata = notebook["metadata"]["aai_curriculum"]
        kernelspec = notebook["metadata"]["kernelspec"]
        assert metadata["order"] == expected_order
        assert int(path.name[:2]) == expected_order
        assert metadata["duration_minutes"] > 0
        assert metadata["learner_evidence"]
        assert set(metadata["prerequisites"]).issubset(seen)
        assert kernelspec == {
            "display_name": "AAI Local Fine-Tuning (offline)",
            "language": "python",
            "name": "aai-local-finetuning",
        }
        seen.add(path.name)
        stages.append(metadata["stage"])

    assert stages == [
        "orientation",
        "provenance",
        "data_audit",
        "data_preparation",
        "baselines",
        "prompt_baselines",
        "training",
        "evaluation",
        "decision",
        "capstone_data",
        "capstone_architecture",
        "extensions",
    ]


def test_all_notebooks_are_clean_unique_and_compilable():
    for path, notebook in _notebooks():
        assert notebook["nbformat"] == 4
        assert "widgets" not in notebook.get("metadata", {})
        cell_ids = [cell["id"] for cell in notebook["cells"]]
        assert len(cell_ids) == len(set(cell_ids))
        assert all(re.fullmatch(r"[0-9a-f]{12}", cell_id) for cell_id in cell_ids)
        assert all(not cell.get("attachments") for cell in notebook["cells"])
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells
        for cell in code_cells:
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile(_source(cell), f"{path.name}:{cell['id']}", "exec")


def test_every_notebook_has_narrative_exercises_and_checkpoints():
    for path, notebook in _notebooks():
        markdown = "\n".join(
            _source(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        tags = {
            tag
            for cell in notebook["cells"]
            for tag in cell.get("metadata", {}).get("tags", [])
        }
        assert "## Learning objectives" in markdown, path.name
        assert "**Estimated time:**" in markdown, path.name
        assert "**Prerequisites:**" in markdown, path.name
        assert "**Learner-produced evidence:**" in markdown, path.name
        assert {"exercise", "hint", "checkpoint", "what-to-notice"}.issubset(
            tags
        ), path.name
        assert "Next:" in markdown or "Final checkpoint" in markdown, path.name


def test_opening_markdown_renders_as_headings_not_indented_code():
    for path, notebook in _notebooks():
        opening = _source(notebook["cells"][0])
        lines = opening.splitlines()

        assert lines[0].startswith(f"# {path.name[:2]} —"), path.name
        assert "## Learning objectives" in lines, path.name
        assert not any(
            line.startswith(("    #", "    **Estimated", "    **Prerequisites"))
            for line in lines
        ), path.name


def test_opening_markdown_renders_as_real_html_headings():
    exporter = HTMLExporter()
    for path, _ in _notebooks():
        html, _ = exporter.from_filename(str(path))
        preformatted = "\n".join(re.findall(r"<pre.*?</pre>", html, re.DOTALL))

        assert re.search(r"<h1[^>]*>.*?\d{2} —.*?</h1>", html, re.DOTALL), path.name
        assert re.search(
            r"<h2[^>]*>.*?Learning objectives.*?</h2>", html, re.DOTALL
        ), path.name
        assert "Estimated time:" in html, path.name
        assert "Prerequisites:" in html, path.name
        assert "Estimated time:" not in preformatted, path.name
        assert "Learning objectives" not in preformatted, path.name


def test_notebook_code_is_local_only_and_enables_offline_controls_first():
    forbidden = (
        "%pip",
        "!pip",
        "pip install",
        "uv sync",
        "kaggle datasets download",
        "snapshot_download",
        "hf" + "_hub_download",
        "load_dataset(",
        "requests.",
        "httpx.",
        "urllib.request",
        "curl ",
        "wget ",
        "http://",
        "https://",
    )
    for path, notebook in _notebooks():
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        first_code = _source(code_cells[0])
        assert "enable_offline_environment" in first_code, path.name
        assert "AAI Local Fine-Tuning (offline)" in first_code, path.name
        assert 'project_root / ".venv" / "bin" / "python"' in first_code, path.name
        source = "\n".join(_source(cell) for cell in code_cells).lower()
        assert not any(fragment in source for fragment in forbidden), path.name


def test_frozen_test_content_is_not_loaded_before_evaluation_notebook():
    for path, notebook in _notebooks()[:7]:
        code_source = "\n".join(
            _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        assert "test.jsonl" not in code_source, path.name
        if "load_support_splits(" in code_source:
            assert "include_test=False" in code_source, path.name


def test_required_lifecycle_behaviors_are_taught_in_order():
    sources = {
        path.name: "\n".join(_source(cell) for cell in notebook["cells"])
        for path, notebook in _notebooks()
    }

    required = {
        "01_dataset_provenance_and_license.ipynb": ("license", "sha256"),
        "02_dataset_exploration_and_validation.ipynb": (
            "audit_dataset",
            "sensitive_pattern_counts",
            "near_duplicate_clusters",
        ),
        "03_leakage_safe_splits.ipynb": ("check_split_files", "frozen"),
        "04_deterministic_baselines.ipynb": (
            "MajorityBaseline",
            "KeywordRuleBaseline",
            "macro F1",
        ),
        "05_prompt_baselines.ipynb": ("basic", "strong", "few_shot"),
        "06_lora_finetuning.ipynb": ("run_lora", "training evidence"),
        "07_frozen_evaluation.ipynb": (
            "schema",
            "format_error_analysis",
            "memory",
        ),
        "08_mlflow_and_promotion.ipynb": ("mlflow", "decide_lora_promotion"),
        "09_capstone_policy_dataset.ipynb": ("rule_catalog", "400/100/150"),
        "10_capstone_model_vs_hybrid.ipynb": ("build_hybrid_review", "ceiling"),
    }
    for notebook, fragments in required.items():
        assert all(fragment in sources[notebook] for fragment in fragments), notebook


def test_renderer_and_index_cover_the_same_course():
    renderer = PROJECT_ROOT / "scripts" / "render_notebooks.py"
    index = (PROJECT_ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")

    assert renderer.is_file()
    for filename in EXPECTED_NOTEBOOKS:
        assert filename in index
