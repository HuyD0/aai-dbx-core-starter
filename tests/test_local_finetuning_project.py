"""Repository integration checks for the standalone offline study project."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from aai_local_finetuning import training
from aai_local_finetuning.capstone import (
    deterministic_capstone_predictions,
    evaluate_capstone_predictions,
    generate_capstone_dataset,
    load_capstone_records,
    render_capstone_mlx_dataset,
)
from aai_local_finetuning.evaluation import (
    DeterministicInferenceConfig,
    start_evaluation_session,
)
from aai_local_finetuning.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "examples" / "local-finetuning"


@pytest.fixture
def normalize_runtime_evidence_test_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclude pytest-only import state from strict application evidence."""

    pytest_only_roots = {
        ROOT,
        ROOT / "tests",
        ROOT / "src" / "platform_app",
        ROOT / "examples" / "agentic-ops-rag" / "src",
    }

    def resolved(entry: str) -> Path:
        candidate = Path(entry or os.getcwd())
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate.resolve(strict=False)

    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if resolved(entry) not in pytest_only_roots],
    )
    loaded_modules = training._runtime_loaded_modules

    def without_test_modules() -> tuple[tuple[str, object], ...]:
        selected: list[tuple[str, object]] = []
        for name, module in loaded_modules():
            if (
                training._runtime_audit.was_preexisting(name, module)
                and name != "_virtualenv"
            ):
                continue
            spec = getattr(module, "__spec__", None)
            loader = getattr(spec, "loader", None)
            if type(loader).__module__.startswith("_pytest."):
                continue
            origin = getattr(spec, "origin", None)
            if not isinstance(origin, str) or origin in {"built-in", "frozen"}:
                selected.append((name, module))
                continue
            if not any(
                resolved(origin).is_relative_to(root) for root in pytest_only_roots
            ):
                selected.append((name, module))
        return tuple(selected)

    monkeypatch.setattr(training, "_runtime_loaded_modules", without_test_modules)


def test_local_finetuning_project_has_isolated_locked_contract():
    expected = {
        "README.md",
        "OFFLINE_STUDY.md",
        "DATA_LICENSE.md",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        "configs/project.yaml",
        "configs/training/lora.yaml",
        "configs/training/capstone-lora.yaml",
        "dataset_cards/bitext-customer-support.md",
        "notebooks/00_start_here.ipynb",
        "notebooks/02_dataset_exploration_and_validation.ipynb",
        "notebooks/11_design_the_next_project.ipynb",
        "notebooks/README.md",
        "scripts/render_notebooks.py",
        "src/aai_local_finetuning/learning.py",
        "src/aai_local_finetuning/cli.py",
        "src/aai_local_finetuning/capstone/evaluation.py",
        "src/aai_local_finetuning/evaluation/session.py",
    }

    missing = [relative for relative in expected if not (PROJECT / relative).is_file()]
    assert not missing

    notebooks = sorted((PROJECT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    assert len(notebooks) == 12

    pyproject = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    assert "mlx-lm[train]==0.31.3" in pyproject
    assert "sys_platform == 'darwin'" in pyproject
    assert "mlflow==3.14.0" in pyproject
    assert "kaggle==2.2.4" in pyproject
    assert "jupyterlab==4.6.2" in pyproject


def test_pinned_assets_and_rights_are_specific_and_non_secret():
    settings = load_settings()

    assert settings.dataset.ref == (
        "bitext/bitext-gen-ai-chatbot-customer-support-dataset"
    )
    assert settings.dataset.license_spdx == "CDLA-Sharing-1.0"
    assert len(settings.dataset.archive_sha256) == 64
    assert len(settings.dataset.csv_sha256) == 64
    assert len(settings.model.revision) == 40
    assert len(settings.model.primary_weight_sha256) == 64

    content = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(PROJECT.rglob("*"))
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(PROJECT).parts)
        and path.suffix in {".md", ".toml", ".yaml", ".py"}
    )
    assert "KAGGLE_API_TOKEN=" not in content
    assert "hf_" not in content


def test_third_party_assets_are_not_tracked():
    completed = subprocess.run(
        ["git", "ls-files", "examples/local-finetuning"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = completed.stdout.splitlines()

    assert not any("/models/" in path for path in tracked)
    assert not any(path.endswith((".safetensors", ".zip", ".csv")) for path in tracked)
    assert not any("/artifacts/" in path for path in tracked)


def test_root_build_excludes_standalone_project_from_sdk_sdist():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        root_pyproject = tomllib.load(stream)

    excluded = root_pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    assert "/examples/local-finetuning" in excluded
    assert "/tests/test_local_finetuning_project.py" in excluded


def test_capstone_generation_is_portable_and_has_frozen_hashes(
    tmp_path,
    normalize_runtime_evidence_test_harness,
):
    source = tmp_path / "source"
    manifest = generate_capstone_dataset(source)

    counts = {
        artifact.split.value: artifact.record_count for artifact in manifest.artifacts
    }
    assert counts == {"train": 400, "validation": 100, "test": 150}
    assert manifest.frozen_test is True
    assert len(manifest.dataset_sha256) == 64
    persisted = json.loads((source / "split-manifest.json").read_text())
    assert persisted["dataset_sha256"] == manifest.dataset_sha256
    rendered = render_capstone_mlx_dataset(source, tmp_path / "mlx")
    assert rendered.splits["train"].record_count == 400
    records = load_capstone_records(source / "test.jsonl")
    evaluation_session = start_evaluation_session()
    predictions = deterministic_capstone_predictions(records)
    report = evaluate_capstone_predictions(
        records,
        predictions,
        evaluation_session=evaluation_session,
        inference_config=DeterministicInferenceConfig(method="deterministic-policy"),
    )
    assert report.aggregate.exact_review_rate == 1.0


def test_root_exposes_prepare_and_strict_offline_commands():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "study-prepare-flight:" in makefile
    assert "study-offline-check:" in makefile
    assert "study-lab:" in makefile
    assert "notebook:" in makefile
    notebook_rule = makefile.split("notebook:", 1)[1].split("\n\n", 1)[0]
    assert "examples/local-finetuning notebook" in notebook_rule
    offline_rule = makefile.split("study-offline-check:", 1)[1].split("\n\n", 1)[0]
    assert "prepare-flight" not in offline_rule
