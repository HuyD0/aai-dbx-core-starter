"""Governed MLflow Prompt Registry operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aai_core.tags import ResourceContext


@dataclass(frozen=True)
class PromptReference:
    name: str
    version: int | None = None
    alias: str | None = None

    @property
    def uri(self) -> str:
        if self.version is not None:
            return f"prompts:/{self.name}/{self.version}"
        if self.alias:
            return f"prompts:/{self.name}@{self.alias}"
        raise ValueError("PromptReference requires a version or alias")


class PromptManager:
    def __init__(
        self,
        *,
        context: ResourceContext,
        catalog: str,
        schema: str,
        mlflow_module: Any | None = None,
    ) -> None:
        self.context = context
        self.catalog = catalog
        self.schema = schema
        self._mlflow = mlflow_module

    def register(
        self,
        name: str,
        template: str | list[dict[str, str]],
        *,
        commit_message: str,
        tags: dict[str, str] | None = None,
    ):
        qualified = self.qualify(name)
        metadata = self.context.merged(tags)
        return self._client().genai.register_prompt(
            name=qualified,
            template=template,
            commit_message=commit_message,
            tags={f"aai.{key}": value for key, value in metadata.items()},
        )

    def load(
        self,
        name: str,
        *,
        version: int | None = None,
        alias: str | None = None,
        cache_ttl_seconds: float | None = None,
    ):
        reference = PromptReference(self.qualify(name), version=version, alias=alias)
        kwargs = (
            {"cache_ttl_seconds": cache_ttl_seconds}
            if cache_ttl_seconds is not None
            else {}
        )
        return self._client().genai.load_prompt(reference.uri, **kwargs)

    def set_alias(self, name: str, *, alias: str, version: int) -> None:
        if alias not in {"development", "candidate", "production"}:
            raise ValueError(f"Unsupported governed prompt alias: {alias}")
        self._client().genai.set_prompt_alias(
            name=self.qualify(name),
            alias=alias,
            version=version,
        )

    def qualify(self, name: str) -> str:
        parts = name.split(".")
        if len(parts) == 1:
            return f"{self.catalog}.{self.schema}.{name}"
        if len(parts) == 3:
            return name
        raise ValueError("Prompt names must be unqualified or catalog.schema.name")

    def _client(self):
        if self._mlflow is not None:
            return self._mlflow
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError(
                "Prompt support requires `pip install 'aai-core[genai]'`"
            ) from error
        return mlflow
