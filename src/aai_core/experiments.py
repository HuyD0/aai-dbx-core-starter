"""Opinionated MLflow experiment and run management."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
