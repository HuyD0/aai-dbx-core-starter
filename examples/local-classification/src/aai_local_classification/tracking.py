"""Explicit local MLflow topology and reusable evidence logging helpers."""

from __future__ import annotations

import os
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.entities import Dataset

from aai_local_classification.contracts import ProjectSettings, SplitName
from aai_local_classification.settings import PROJECT_ROOT


@dataclass(frozen=True)
class LocalProjectPaths:
    source_root: Path
    project_root: Path
    data_root: Path
    mlflow_root: Path
    artifact_root: Path
    state_root: Path
    tracking_uri: str


def local_paths(project_root: Path | None = None) -> LocalProjectPaths:
    override = os.getenv("AAI_CLASSIFICATION_PROJECT_ROOT")
    root = (
        Path(override)
        if project_root is None and override
        else project_root or PROJECT_ROOT
    ).resolve()
    mlflow_root = root / ".aai" / "mlflow"
    database = mlflow_root / "mlflow.db"
    return LocalProjectPaths(
        source_root=PROJECT_ROOT,
        project_root=root,
        data_root=root / "data" / "processed",
        mlflow_root=mlflow_root,
        artifact_root=mlflow_root / "artifacts",
        state_root=root / ".aai" / "state",
        tracking_uri=f"sqlite:///{database}",
    )


def configure_mlflow(
    settings: ProjectSettings,
    project_root: Path | None = None,
) -> LocalProjectPaths:
    """Configure one SQLite-backed tracking and registry store for local study."""

    paths = local_paths(project_root)
    paths.artifact_root.mkdir(parents=True, exist_ok=True)
    paths.state_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(paths.tracking_uri)
    mlflow.set_registry_uri(paths.tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(settings.experiment_name)
    if experiment is None:
        client.create_experiment(
            settings.experiment_name,
            artifact_location=paths.artifact_root.resolve().as_uri(),
            tags={"purpose": "local-classification-learning"},
        )
    mlflow.set_experiment(settings.experiment_name)
    return paths


def source_state(project_root: Path) -> dict[str, str]:
    """Return bounded source metadata without assuming the sample stays in Git."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"source_commit": commit, "source_state": "dirty" if status else "clean"}
    except (OSError, subprocess.CalledProcessError):
        return {"source_commit": "unavailable", "source_state": "unavailable"}


def run_tags(
    settings: ProjectSettings,
    paths: LocalProjectPaths,
    *,
    lifecycle_role: str,
    dataset_sha256: str,
) -> dict[str, str]:
    return {
        "project": settings.project_name,
        "purpose": "learn-classification-mlops",
        "lifecycle_role": lifecycle_role,
        "dataset_sha256": dataset_sha256,
        "data_classification": "synthetic",
        "execution_environment": "local",
        **source_state(paths.source_root),
    }


def log_dataset(
    frame: pd.DataFrame,
    settings: ProjectSettings,
    paths: LocalProjectPaths,
    split: SplitName,
) -> Dataset:
    source = paths.data_root / f"{split.value}.csv"
    with warnings.catch_warnings():
        # These warnings describe model-signature inference. Here MLflow is only
        # fingerprinting a persisted dataset whose schema is validated separately.
        warnings.filterwarnings(
            "ignore",
            message=(
                "The specified dataset source can be interpreted in multiple " "ways.*"
            ),
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Hint: Inferred schema contains integer column.*",
            category=UserWarning,
        )
        dataset = mlflow.data.from_pandas(
            frame,
            source=source.resolve().as_uri(),
            targets=settings.data.target_column,
            name=f"{settings.data.dataset_name}-{split.value}",
        )
        mlflow.log_input(dataset, context=split.value)
    return dataset


def log_reproducibility_artifacts(paths: LocalProjectPaths) -> None:
    for artifact in (
        paths.source_root / "configs" / "project.yaml",
        paths.data_root / "manifest.json",
        paths.source_root / "uv.lock",
    ):
        if artifact.is_file():
            mlflow.log_artifact(artifact, artifact_path="reproducibility")


def log_linked_metrics(
    values: dict[str, float],
    *,
    model_id: str,
    dataset: Dataset,
) -> None:
    for key, value in values.items():
        mlflow.log_metric(key, value, model_id=model_id, dataset=dataset)
