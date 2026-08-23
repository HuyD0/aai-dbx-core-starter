"""Optional local MLflow evidence; imports remain lazy for portable tests."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from .evaluation.models import InferenceConfig, LocalMLXInferenceConfig
from .evaluation.session import EvaluationSession, recheck_evaluation_session
from .settings import PROJECT_ROOT, ProjectSettings
from .training import (
    TRAINING_MANIFEST_NAME,
    ValidatedTrainingSnapshot,
    recheck_training_snapshot,
    shared_adapter_lock,
)


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
    evaluation_session: EvaluationSession,
    training_snapshot: ValidatedTrainingSnapshot | None = None,
) -> str:
    """Log aggregate evidence, native dataset input, and detailed artifacts."""

    import mlflow
    import pandas as pd

    recheck_evaluation_session(evaluation_session)
    evaluation_execution_contract = evaluation_session.execution_contract
    evaluation_execution_contract_sha256 = evaluation_session.execution_contract_sha256
    if (
        report.get("evaluation_execution_contract_sha256")
        != evaluation_execution_contract_sha256
    ):
        raise ValueError(
            "evaluation report does not match the current source/runtime contract"
        )
    inference_config = TypeAdapter(InferenceConfig).validate_json(
        json.dumps(report.get("inference_config")),
        strict=True,
    )
    if inference_config.method != method:
        raise ValueError("tracking method does not match report inference evidence")
    config_is_model_based = isinstance(
        inference_config,
        LocalMLXInferenceConfig,
    )
    if model_based != config_is_model_based:
        raise ValueError("model_based does not match report inference evidence")
    if config_is_model_based:
        if (
            evaluation_session.base_model_execution_contract
            != inference_config.base_model
        ):
            raise ValueError(
                "report inference evidence does not match the model-aware session"
            )
    if role == "change":
        if training_snapshot is None:
            raise ValueError("change tracking requires a validated training snapshot")
        if report.get("training_manifest_sha256") != (
            training_snapshot.manifest_sha256
        ):
            raise ValueError(
                "change report and validated training snapshot lineage differ"
            )
        if report.get("training_execution_contract_sha256") != (
            training_snapshot.manifest.execution_contract_sha256
        ):
            raise ValueError("change report and training source/runtime lineage differ")
        if not isinstance(inference_config, LocalMLXInferenceConfig) or (
            inference_config.adapter_manifest_sha256
            != training_snapshot.manifest_sha256
        ):
            raise ValueError(
                "change inference configuration and training lineage differ"
            )
        recheck_training_snapshot(training_snapshot)
    elif training_snapshot is not None:
        raise ValueError("only change tracking accepts training evidence")
    elif isinstance(inference_config, LocalMLXInferenceConfig) and (
        inference_config.adapter_manifest_sha256 is not None
    ):
        raise ValueError("baseline inference evidence must not carry an adapter")

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
        mlflow.log_param(
            "evaluation_execution_contract_sha256",
            evaluation_execution_contract_sha256,
        )
        mlflow.log_dict(
            evaluation_execution_contract.model_dump(mode="json"),
            "runtime/evaluation-execution-contract.json",
        )
        mlflow.log_dict(
            inference_config.model_dump(mode="json"),
            "evaluation/inference-config.json",
        )
        mlflow.log_params(
            {
                "inference_method": inference_config.method,
                "inference_mode": inference_config.mode,
            }
        )
        if isinstance(inference_config, LocalMLXInferenceConfig):
            mlflow.log_params(
                {
                    "model_repo": inference_config.base_model.repository,
                    "model_revision": inference_config.base_model.model_revision,
                    "model_files_sha256": (
                        inference_config.base_model.model_files_sha256
                    ),
                    "prompt_recipe": inference_config.prompt_recipe,
                    "max_tokens": inference_config.generation.max_tokens,
                    "sampler": inference_config.generation.sampler,
                    "few_shot_examples": inference_config.few_shot_examples,
                }
            )
        if role == "change":
            assert training_snapshot is not None
            with shared_adapter_lock(training_snapshot.adapter_path):
                recheck_training_snapshot(training_snapshot)
                adapter = training_snapshot.adapter_path / "adapters.safetensors"
                adapter_config = training_snapshot.adapter_path / "adapter_config.json"
                training_config = training_snapshot.config_path
                training_manifest = (
                    training_snapshot.adapter_path / TRAINING_MANIFEST_NAME
                )
                validated_manifest = training_snapshot.manifest
                mlflow.log_params(
                    {
                        "adapter_sha256": validated_manifest.adapter_sha256,
                        "adapter_config_sha256": (
                            validated_manifest.adapter_config_sha256
                        ),
                        "training_config_sha256": (
                            validated_manifest.source_config_sha256
                        ),
                        "effective_training_config_sha256": (
                            validated_manifest.effective_config_sha256
                        ),
                        "training_manifest_sha256": (training_snapshot.manifest_sha256),
                        "training_source_files_sha256": (
                            validated_manifest.execution_contract.source_files_sha256
                        ),
                        "training_runtime_packages_sha256": (
                            validated_manifest.execution_contract.runtime_packages_sha256
                        ),
                        "training_execution_contract_sha256": (
                            validated_manifest.execution_contract_sha256
                        ),
                    }
                )
                mlflow.log_dict(
                    validated_manifest.execution_contract.model_dump(mode="json"),
                    "change/training-execution-contract.json",
                )
                mlflow.log_artifact(str(adapter), artifact_path="change")
                mlflow.log_artifact(str(adapter_config), artifact_path="change")
                mlflow.log_artifact(str(training_config), artifact_path="change")
                mlflow.log_artifact(str(training_manifest), artifact_path="change")
                recheck_training_snapshot(training_snapshot)
        mlflow.log_input(dataset, context="frozen_evaluation")
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.log_artifact(str(manifest_path), artifact_path="dataset")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        mlflow.log_artifact(str(prediction_path), artifact_path="evaluation")
        if training_snapshot is not None:
            recheck_training_snapshot(training_snapshot)
        recheck_evaluation_session(evaluation_session)
        return run.info.run_id
