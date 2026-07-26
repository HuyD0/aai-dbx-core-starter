"""Opinionated MLflow experiment and run management."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field

from aai_core.contracts import ContractModel
from aai_core.secrets import SecretRef, SecretValue
from aai_core.tags import ResourceContext

_SENSITIVE_NAMES = {
    "api_key",
    "client_secret",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}


class RunPurpose(StrEnum):
    """Closed vocabulary for the evidence a run is intended to produce."""

    BASELINE = "baseline"
    CHANGE = "change"
    RESULT = "result"
    DECISION = "decision"
    MONITORING = "monitoring"
    EXPLORATION = "exploration"


class ExperimentRunMetadata(ContractModel):
    """Searchable intent and comparison lineage for a governed run."""

    purpose: RunPurpose
    change_id: str = Field(min_length=1)
    change_summary: str = Field(min_length=1)
    hypothesis: str | None = Field(default=None, min_length=1)
    baseline_run_id: str | None = None
    application_model_id: str | None = None

    def as_tags(self) -> dict[str, str]:
        values = {
            "aai.run_purpose": self.purpose.value,
            "aai.change_id": self.change_id,
            "aai.change_summary": self.change_summary,
        }
        if self.hypothesis:
            values["aai.hypothesis"] = self.hypothesis
        if self.baseline_run_id:
            values["aai.baseline_run_id"] = self.baseline_run_id
        if self.application_model_id:
            values["aai.application_model_id"] = self.application_model_id
        return values


class ExperimentManager:
    def __init__(
        self,
        *,
        experiment_name: str,
        context: ResourceContext,
        mlflow_module: Any | None = None,
    ) -> None:
        if not experiment_name.strip():
            raise ValueError("experiment_name must not be blank")
        self.experiment_name = experiment_name
        self.context = context
        self._mlflow = mlflow_module

    @contextmanager
    def run(
        self,
        *,
        run_name: str,
        parameters: Mapping[str, Any] | None = None,
        tags: Mapping[str, str] | None = None,
        nested: bool = False,
        metadata: ExperimentRunMetadata | None = None,
    ) -> Iterator[Any]:
        if not run_name.strip():
            raise ValueError("run_name must not be blank")
        mlflow = self._client()
        mlflow.set_experiment(self.experiment_name)
        active_model = None
        if metadata is not None and metadata.application_model_id:
            setter = getattr(mlflow, "set_active_model", None)
            if setter is None:
                raise RuntimeError(
                    "Application-version lineage requires MLflow 3 "
                    "`set_active_model()` support."
                )
            active_model = setter(model_id=metadata.application_model_id)

        @contextmanager
        def governed_run() -> Iterator[Any]:
            with mlflow.start_run(run_name=run_name, nested=nested) as active_run:
                merged_tags = self.context.merged(tags)
                run_tags = {f"aai.{key}": value for key, value in merged_tags.items()}
                run_tags["aai.experiment_name"] = self.experiment_name
                if metadata is not None:
                    conflicts = set(run_tags).intersection(metadata.as_tags())
                    if conflicts:
                        raise ValueError(
                            "Run metadata conflicts with governed tags: "
                            + ", ".join(sorted(conflicts))
                        )
                    run_tags.update(metadata.as_tags())
                mlflow.set_tags(run_tags)
                if parameters:
                    mlflow.log_params(_safe_parameters(parameters))
                yield active_run

        if active_model is None:
            with governed_run() as active_run:
                yield active_run
        else:
            # MLflow's native ActiveModel context restores the previous model.
            # Returning it unchanged avoids an SDK mirror of LoggedModel.
            with active_model:
                with governed_run() as active_run:
                    yield active_run

    @property
    def native_client(self) -> Any:
        """Return the native MLflow module for unsupported fluent APIs."""

        return self._client()

    def _client(self):
        if self._mlflow is not None:
            return self._mlflow
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError(
                "Experiment support requires the `genai` extra. From an aai-core "
                "checkout run `make examples-install` and use `.venv/bin/python`; "
                "in a consuming environment install `aai-core[genai]`."
            ) from error
        return mlflow


def record_reproducibility(
    *,
    seed: int | None = None,
    extra: Mapping[str, str] | None = None,
    mlflow_module: Any | None = None,
) -> dict[str, str]:
    """Record what a run needs to be repeated: seed, source commit, aai-core
    version, and the installed-package freeze (as an artifact).

    Call inside an active run. Returns the logged key/value summary. Commit
    resolution order: ``GIT_COMMIT`` env var (CI), ``git rev-parse HEAD``,
    else ``"local-dev"``.
    """

    if mlflow_module is None:
        import mlflow as mlflow_module  # type: ignore[no-redef]

    from aai_core import __version__

    record: dict[str, str] = {
        "source_commit": _source_commit(),
        "source_state": _source_state(),
        "aai_core_version": __version__,
    }
    if seed is not None:
        record["seed"] = str(seed)
    for key, value in (extra or {}).items():
        record[str(key)] = str(value)
    mlflow_module.log_params(_safe_parameters(record))

    freeze = _package_freeze()
    with tempfile.TemporaryDirectory() as scratch:
        freeze_file = Path(scratch) / "requirements-frozen.txt"
        freeze_file.write_text(freeze, encoding="utf-8")
        mlflow_module.log_artifact(str(freeze_file), artifact_path="reproducibility")
    record["environment_digest"] = hashlib.sha256(freeze.encode()).hexdigest()[:16]
    mlflow_module.set_tags({"aai.environment_digest": record["environment_digest"]})
    return record


def _source_commit() -> str:
    from_env = os.getenv("GIT_COMMIT")
    if from_env:
        return from_env
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "local-dev"


def _source_state() -> str:
    """Return a non-sensitive clean/dirty/unknown source state."""

    from_environment = os.getenv("GIT_DIRTY")
    if from_environment is not None:
        return (
            "dirty"
            if from_environment.strip().lower() in {"1", "true", "yes"}
            else "clean"
        )
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return "dirty" if result.stdout.strip() else "clean"
    except Exception:
        return "unknown"


def _package_freeze() -> str:
    from importlib.metadata import distributions

    lines = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in distributions()
        if distribution.metadata["Name"]
    )
    return "\n".join(lines) + "\n"


def _safe_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in parameters.items():
        normalized = key.lower().replace("-", "_").replace(".", "_").replace("/", "_")
        sensitive_name = any(
            normalized == name or normalized.endswith(f"_{name}")
            for name in _SENSITIVE_NAMES
        )
        if sensitive_name or isinstance(value, (SecretValue, SecretRef)):
            raise ValueError(f"Refusing to log sensitive MLflow parameter: {key}")
        safe[str(key)] = value
    return safe
