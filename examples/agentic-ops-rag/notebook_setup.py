"""Safe setup shared by every agentic operations RAG notebook."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aai_core import PlatformContext, bootstrap

_LOGICAL_MODEL = "operations-chat"
_LOGICAL_EMBEDDING = "operations-embedding"
_LOGICAL_RETRIEVER = "operations-knowledge"
_PLACEHOLDERS = ("replace-with", "replace_with")


@dataclass(frozen=True)
class WorkshopSession:
    course_root: Path
    repository_root: Path
    config_path: Path
    context: PlatformContext
    using_example_config: bool
    connected_ready: bool

    def safe_summary(self) -> dict[str, Any]:
        """Return identifiers and readiness only; never serialize raw settings."""

        return {
            "config": str(self.config_path),
            "using_example_config": self.using_example_config,
            "connected_ready": self.connected_ready,
            "experiment_name": self.context.settings.effective_experiment_name,
            "logical_model": _LOGICAL_MODEL,
            "logical_embedding": _LOGICAL_EMBEDDING,
            "logical_retriever": _LOGICAL_RETRIEVER,
            "azure_identity": self.context.settings.azure_identity,
        }

    def offline_pipeline(self):
        from agentic_ops_rag import (
            OfflineOperationsRetriever,
            OperationsRAGPipeline,
            load_documents,
        )

        documents = load_documents(
            self.course_root / "data" / "operations_documents.jsonl"
        )
        return OperationsRAGPipeline(OfflineOperationsRetriever(documents))

    def connected_components(self, *, allow_network: bool = False):
        """Resolve real providers only after a deliberate connected opt-in."""

        if not allow_network:
            raise RuntimeError(
                "Connected provider resolution is disabled. Set the notebook's "
                "RUN_CONNECTED switch only after reviewing the call and data policy."
            )
        if not self.connected_ready:
            raise RuntimeError(
                "Connected configuration still contains placeholders. Copy an "
                "example config to config/aai-platform.yml and replace every "
                "placeholder with externally provisioned logical resources."
            )
        return {
            "model": self.context.providers.model(_LOGICAL_MODEL),
            "embedding": self.context.providers.embedding(_LOGICAL_EMBEDDING),
            "retriever": self.context.providers.retriever(_LOGICAL_RETRIEVER),
        }

    def judge_model_uri(self) -> str:
        """Resolve the optional MLflow judge URI from its logical resource."""

        config = self.context.settings.models.get("judge-model")
        if config is None:
            raise RuntimeError("Workshop configuration is missing logical judge-model")
        if _contains_placeholder(config):
            raise RuntimeError(
                "Logical judge-model still contains a placeholder deployment"
            )
        if config.get("provider") != "databricks":
            raise RuntimeError("MLflow judges require a governed Databricks endpoint")
        deployment = str(config.get("deployment", "")).strip()
        if not deployment:
            raise RuntimeError("Logical judge-model has no deployment")
        return f"endpoints:/{deployment}"


def find_course_root(start: str | Path | None = None) -> Path:
    base = Path(start or Path.cwd()).resolve()
    for directory in (base, *base.parents):
        candidate = directory / "examples" / "agentic-ops-rag"
        if (candidate / "data" / "operations_documents.jsonl").is_file():
            return candidate
        if (directory / "data" / "operations_documents.jsonl").is_file():
            return directory
    raise FileNotFoundError("Could not locate examples/agentic-ops-rag")


def prepare_notebook_environment(
    course_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> WorkshopSession:
    course = find_course_root(course_root)
    repository = course.parents[1]
    for source_root in (repository / "src", course / "src"):
        value = str(source_root)
        if value not in sys.path:
            sys.path.insert(0, value)

    local_config = course / "config" / "aai-platform.yml"
    example_config = course / "config" / "aai-platform.azure-search.example.yml"
    selected = Path(config_path) if config_path else local_config
    using_example = not selected.is_file()
    if using_example:
        selected = example_config
    context = bootstrap(selected)
    _require_logical_resources(context)
    ready = not _contains_placeholder(
        {
            "model": context.settings.models[_LOGICAL_MODEL],
            "embedding": context.settings.embeddings[_LOGICAL_EMBEDDING],
            "retriever": context.settings.retrievers[_LOGICAL_RETRIEVER],
        }
    )
    return WorkshopSession(
        course_root=course,
        repository_root=repository,
        config_path=selected,
        context=context,
        using_example_config=using_example,
        connected_ready=ready,
    )


def _require_logical_resources(context: PlatformContext) -> None:
    missing = []
    if _LOGICAL_MODEL not in context.settings.models:
        missing.append(_LOGICAL_MODEL)
    if _LOGICAL_EMBEDDING not in context.settings.embeddings:
        missing.append(_LOGICAL_EMBEDDING)
    if _LOGICAL_RETRIEVER not in context.settings.retrievers:
        missing.append(_LOGICAL_RETRIEVER)
    if missing:
        raise ValueError("Workshop configuration is missing: " + ", ".join(missing))


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_placeholder(item) for item in value)
    return isinstance(value, str) and any(
        placeholder in value.lower() for placeholder in _PLACEHOLDERS
    )
