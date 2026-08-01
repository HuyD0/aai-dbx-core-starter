"""Optional local MLflow evidence; imports remain lazy for portable tests."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .settings import PROJECT_ROOT, ProjectSettings, sha256_file


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _tracking_uri(settings: ProjectSettings) -> str:
    prefix = "sqlite:///"
    if not settings.tracking.uri.startswith(prefix):
        raise ValueError("the offline project only supports a local SQLite MLflow URI")
    database = _project_path(settings.tracking.uri.removeprefix(prefix)).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database}"


def configure_local_mlflow(settings: ProjectSettings) -> None:
    """Configure a repository-local SQLite backend and artifact directory."""

    import mlflow

    artifact_root = _project_path(settings.tracking.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = _tracking_uri(settings)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    existing = mlflow.get_experiment_by_name(settings.tracking.experiment)
    expected_artifact_location = artifact_root.as_uri()
    if existing is None:
        experiment_id = mlflow.create_experiment(
            settings.tracking.experiment,
            artifact_location=expected_artifact_location,
        )
    else:
        experiment_id = existing.experiment_id
        if existing.artifact_location.rstrip("/") != expected_artifact_location:
            raise RuntimeError(
                "the local MLflow experiment uses an unexpected artifact directory; "
                "remove .aai and rerun flight preparation"
            )
    mlflow.set_experiment(experiment_id=experiment_id)


def tracking_smoke(settings: ProjectSettings) -> str:
    """Prove local MLflow can write a run without any remote service."""

    import mlflow

    configure_local_mlflow(settings)
    with mlflow.start_run(run_name="offline-flight-readiness") as run:
        mlflow.log_param("execution_mode", "offline_local")
        mlflow.log_metric("local_store_write", 1.0)
        mlflow.set_tag("run_purpose", "readiness")
        return run.info.run_id


def log_evaluation(
    settings: ProjectSettings,
    *,
    run_name: str,
    role: str,
    method: str,
    metrics: Mapping[str, float],
    report: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    manifest_path: Path,
    prediction_path: Path,
    model_based: bool,
) -> str:
    """Log aggregate evidence, native dataset input, and detailed artifacts."""

    import mlflow
    import pandas as pd

    configure_local_mlflow(settings)
    rows = list(records)
    frame = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The specified dataset source can be interpreted in multiple ways",
            category=UserWarning,
            module="mlflow.data.dataset_source_registry",
        )
        dataset = mlflow.data.from_pandas(
            frame,
            source=manifest_path.resolve().as_uri(),
            name="bitext-customer-support-curated-v1",
        )
    output_dir = PROJECT_ROOT / "artifacts" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{run_name}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with mlflow.start_run(run_name=run_name) as run:
        tags = {
            "run_purpose": role,
            "evaluation_method": method,
            "execution_mode": "offline_local",
            "evidence_source": "measured_local",
        }
        if role == "change":
            tags["change_id"] = "bitext-structured-output-lora-v1"
        mlflow.set_tags(tags)
        if model_based:
            mlflow.log_params(
                {
                    "model_repo": settings.model.repo,
                    "model_revision": settings.model.revision,
                }
            )
        if role == "change":
            adapter = settings.adapter_dir / "adapters.safetensors"
            training_config = PROJECT_ROOT / "configs" / "training" / "lora.yaml"
            if not adapter.is_file():
                raise FileNotFoundError(f"missing evaluated LoRA adapter: {adapter}")
            mlflow.log_params(
                {
                    "adapter_sha256": sha256_file(adapter),
                    "training_config_sha256": sha256_file(training_config),
                }
            )
            mlflow.log_artifact(str(adapter), artifact_path="change")
            mlflow.log_artifact(str(training_config), artifact_path="change")
            training_evidence = PROJECT_ROOT / "artifacts" / "training" / "latest.json"
            if training_evidence.is_file():
                mlflow.log_artifact(str(training_evidence), artifact_path="change")
        mlflow.log_input(dataset, context="frozen_evaluation")
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.log_artifact(str(manifest_path), artifact_path="dataset")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        mlflow.log_artifact(str(prediction_path), artifact_path="evaluation")
        return run.info.run_id
