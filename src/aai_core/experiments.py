"""Opinionated MLflow experiment and run management."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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


class ExperimentManager:
    def __init__(
        self,
        *,
        experiment_name: str,
        context: ResourceContext,
        mlflow_module: Any | None = None,
    ) -> None:
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
    ) -> Iterator[Any]:
        mlflow = self._client()
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=run_name, nested=nested) as active_run:
            merged_tags = self.context.merged(tags)
            mlflow.set_tags({f"aai.{key}": value for key, value in merged_tags.items()})
            if parameters:
                mlflow.log_params(_safe_parameters(parameters))
            yield active_run

    def log_dataset(
        self,
        data: Any,
        *,
        name: str,
        context: str = "training",
        source: str | None = None,
    ) -> str:
        """Log a pandas DataFrame as an MLflow input dataset inside the active
        run and return its digest (also logged as ``dataset_digest`` so the
        exact data version is queryable from run params)."""

        mlflow = self._client()
        dataset = mlflow.data.from_pandas(data, name=name, source=source)
        mlflow.log_input(dataset, context=context)
        digest = str(dataset.digest)
        mlflow.log_params({"dataset_name": name, "dataset_digest": digest})
        return digest

    def log_metrics(
        self, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None:
        """Log numeric metrics to the active run (non-numeric values are a
        caller bug and raise instead of being dropped silently)."""

        numeric: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Metric {key!r} is not numeric: {value!r}")
            numeric[str(key)] = float(value)
        self._client().log_metrics(numeric, step=step)

    def log_artifact(
        self, path: str | Path, *, artifact_path: str | None = None
    ) -> None:
        """Log a file or directory as run artifacts."""

        target = Path(path)
        mlflow = self._client()
        if target.is_dir():
            mlflow.log_artifacts(str(target), artifact_path=artifact_path)
        else:
            mlflow.log_artifact(str(target), artifact_path=artifact_path)

    def _client(self):
        if self._mlflow is not None:
            return self._mlflow
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError(
                "Experiment support requires `pip install 'aai-core[genai]'`"
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
        normalized = key.lower().replace("-", "_")
        sensitive_name = any(
            normalized == name or normalized.endswith(f"_{name}")
            for name in _SENSITIVE_NAMES
        )
        if sensitive_name or isinstance(value, (SecretValue, SecretRef)):
            raise ValueError(f"Refusing to log sensitive MLflow parameter: {key}")
        safe[str(key)] = value
    return safe
