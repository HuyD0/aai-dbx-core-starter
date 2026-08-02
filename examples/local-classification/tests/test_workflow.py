from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow
import pytest

from aai_local_classification.contracts import PromotionDecision, SplitName
from aai_local_classification.data import load_split
from aai_local_classification.inference import load_champion
from aai_local_classification.settings import PROJECT_ROOT
from aai_local_classification.tracking import local_paths
from aai_local_classification.workflow import (
    _locked_model_requirements,
    _model_conda_environment,
    get_or_run_candidate_selection,
    load_decision,
    load_selection,
    promote_if_approved,
    run_frozen_test_gate,
    run_full_workflow,
)


def test_logged_model_environment_uses_the_resolved_runtime_closure():
    environment = _model_conda_environment()
    dependencies = environment["dependencies"]
    pip_requirements = dependencies[-1]["pip"]

    assert dependencies[0].startswith("python=")
    assert dependencies[1].startswith("pip=")
    assert all("==" in requirement for requirement in pip_requirements)
    names = {requirement.split("==", 1)[0] for requirement in pip_requirements}
    assert len(pip_requirements) > 50
    assert {
        "joblib",
        "mlflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "skops",
        "threadpoolctl",
    } <= names


def test_packaged_model_requirements_match_the_certified_uv_lock():
    result = subprocess.run(
        [
            "uv",
            "export",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    exported = [
        line.strip()
        for line in result.stdout.splitlines()
        if line and not line.startswith(("#", " "))
    ]
    assert _locked_model_requirements() == exported


def test_environment_selects_an_isolated_course_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("AAI_CLASSIFICATION_PROJECT_ROOT", str(tmp_path))

    paths = local_paths()

    assert paths.project_root == tmp_path.resolve()
    assert paths.source_root == PROJECT_ROOT
    assert paths.data_root == tmp_path.resolve() / "data" / "processed"
    assert paths.state_root == tmp_path.resolve() / ".aai" / "state"


def test_complete_local_mlflow_release_and_reload(settings, tmp_path, caplog):
    result = run_full_workflow(settings, tmp_path)

    messages = [record.getMessage() for record in caplog.records]
    assert not any(
        "Failed to resolve installed pip version" in item for item in messages
    )
    assert not any("model type looks like" in item for item in messages)

    assert result["decision"]["decision"] == PromotionDecision.ADOPT.value
    assert all(result["decision"]["checks"].values())
    assert result["promotion"]["registered"] is True
    assert result["promotion"]["alias"] == "champion"
    assert result["selection"]["selected_candidate"] == "logistic-regression"

    model_info = mlflow.models.get_model_info(result["selection"]["selected_model_uri"])
    artifact_root = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=model_info.artifact_path)
    )
    logged_requirements = (artifact_root / "requirements.txt").read_text(
        encoding="utf-8"
    )
    assert logged_requirements.splitlines() == _locked_model_requirements()

    paths = local_paths(tmp_path)
    assert paths.mlflow_root.joinpath("mlflow.db").is_file()
    predictor = load_champion(settings, tmp_path)
    validation = load_split(settings, SplitName.VALIDATION, paths.data_root)
    predictions = predictor.predict(validation.head(5), settings)
    assert list(predictions.columns) == [
        "churn_probability",
        "churn_prediction",
        "model_name",
        "model_version",
    ]
    assert predictions.churn_probability.between(0, 1).all()
    assert set(predictions.churn_prediction).issubset({0, 1})

    client = mlflow.MlflowClient(
        tracking_uri=paths.tracking_uri,
        registry_uri=paths.tracking_uri,
    )
    version = client.get_model_version_by_alias(
        settings.registered_model_name,
        "champion",
    )
    assert version.tags["validation_status"] == "passed"
    assert version.tags["dataset_sha256"] == result["dataset_sha256"]
    assert float(version.tags["decision_threshold"]) == (
        result["decision"]["threshold"]
    )

    notebook_selection = get_or_run_candidate_selection(settings, tmp_path)
    notebook_decision = run_frozen_test_gate(
        settings,
        tmp_path,
        notebook_selection,
    )
    assert notebook_selection.selected_run_id == result["selection"]["selected_run_id"]
    assert notebook_selection.selected_model_id == (
        result["selection"]["selected_model_id"]
    )
    assert notebook_decision.test_run_id == result["decision"]["test_run_id"]

    repeated = run_full_workflow(settings, tmp_path)
    assert repeated["selection"]["selected_run_id"] == (
        result["selection"]["selected_run_id"]
    )
    assert repeated["decision"]["test_run_id"] == result["decision"]["test_run_id"]
    assert repeated["promotion"]["model_version"] == (
        result["promotion"]["model_version"]
    )


def test_consumed_test_and_promotion_linkage_fail_closed(settings, tmp_path):
    run_full_workflow(settings, tmp_path)
    selection = load_selection(tmp_path)
    decision = load_decision(tmp_path)

    selection_path = local_paths(tmp_path).state_root / "selection.json"
    original_selection = selection_path.read_text(encoding="utf-8")
    changed_selection_policy = settings.selection.model_copy(
        update={
            "simpler_model_tolerance": settings.selection.simpler_model_tolerance
            + 0.001
        }
    )
    changed_training_settings = settings.model_copy(
        update={"selection": changed_selection_policy}
    )
    with pytest.raises(ValueError, match="cannot be replaced"):
        get_or_run_candidate_selection(changed_training_settings, tmp_path)
    assert selection_path.read_text(encoding="utf-8") == original_selection

    changed_gate = settings.promotion_gate.model_copy(
        update={"minimum_test_recall": 0.99}
    )
    changed_settings = settings.model_copy(update={"promotion_gate": changed_gate})
    with pytest.raises(ValueError, match="frozen-test dataset version"):
        run_frozen_test_gate(changed_settings, tmp_path, selection)

    failed_checks = decision.checks.model_copy(update={"minimum_test_recall": False})
    forged_adopt = decision.model_copy(update={"checks": failed_checks})
    with pytest.raises(ValueError, match="gate outcome"):
        promote_if_approved(settings, forged_adopt, tmp_path, selection)

    other = next(
        candidate
        for candidate in selection.candidates
        if candidate.run_id != selection.selected_run_id
    )
    wrong_selection = selection.model_copy(
        update={
            "selected_candidate": other.candidate_name,
            "selected_run_id": other.run_id,
            "selected_model_id": other.model_id,
            "selected_model_uri": other.model_uri,
        }
    )
    with pytest.raises(ValueError, match="Release decision is not bound"):
        promote_if_approved(
            settings,
            decision,
            tmp_path,
            wrong_selection,
        )
