"""Resolve logical platform resources into native provider adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from aai_core.providers.openai_compatible import (
    OpenAICompatibleChatModel,
    OpenAICompatibleEmbeddingProvider,
)
from aai_core.providers.search import (
    AzureAISearchRetriever,
    DatabricksAISearchRetriever,
)
from aai_core.providers.types import ModelCapabilities, ProviderConfigurationError

if TYPE_CHECKING:
    from aai_core.context import PlatformContext


class ProviderResolver:
    def __init__(self, context: PlatformContext) -> None:
        self.context = context
        self._models: dict[str, Any] = {}
        self._embeddings: dict[str, Any] = {}
        self._retrievers: dict[str, Any] = {}
        self._model_factories: dict[str, Callable[[str, dict[str, Any]], Any]] = {
            "databricks": self._databricks_model,
            "foundry": self._foundry_model,
        }
        self._embedding_factories = dict(self._model_factories)
        self._retriever_factories: dict[str, Callable[[str, dict[str, Any]], Any]] = {
            "azure_ai_search": self._azure_search,
            "databricks_ai_search": self._databricks_search,
        }

    def register_model(self, logical_name: str, model: Any) -> None:
        self._models[logical_name] = model

    def register_embedding(self, logical_name: str, embedding: Any) -> None:
        self._embeddings[logical_name] = embedding

    def register_retriever(self, logical_name: str, retriever: Any) -> None:
        self._retrievers[logical_name] = retriever

    def model(self, logical_name: str):
        return self._resolve(
            logical_name,
            self.context.settings.models,
            self._models,
            self._model_factories,
        )

    def embedding(self, logical_name: str):
        if logical_name in self._embeddings:
            return self._embeddings[logical_name]
        config = self._config(logical_name, self.context.settings.embeddings)
        provider = self._provider(config)
        client, model = self._openai_client(provider, config)
        adapter = OpenAICompatibleEmbeddingProvider(
            logical_name=logical_name,
            provider=provider,
            model=model,
            client=client,
            dimensions=_optional_int(config.get("dimensions")),
        )
        self._embeddings[logical_name] = adapter
        return adapter

    def retriever(self, logical_name: str):
        return self._resolve(
            logical_name,
            self.context.settings.retrievers,
            self._retrievers,
            self._retriever_factories,
        )

    def _resolve(
        self,
        logical_name: str,
        configurations: dict[str, dict[str, Any]],
        cache: dict[str, Any],
        factories: dict[str, Callable[[str, dict[str, Any]], Any]],
    ):
        if logical_name in cache:
            return cache[logical_name]
        config = self._config(logical_name, configurations)
        provider = self._provider(config)
        try:
            factory = factories[provider]
        except KeyError as error:
            raise ProviderConfigurationError(
                f"Unsupported provider {provider!r} for {logical_name!r}"
            ) from error
        cache[logical_name] = factory(logical_name, config)
        return cache[logical_name]

    def _databricks_model(self, logical_name: str, config: dict[str, Any]):
        client, model = self._openai_client("databricks", config)
        return OpenAICompatibleChatModel(
            logical_name=logical_name,
            provider="databricks",
            model=model,
            client=client,
            capabilities=_capabilities(config),
        )

    def _foundry_model(self, logical_name: str, config: dict[str, Any]):
        client, model = self._openai_client("foundry", config)
        return OpenAICompatibleChatModel(
            logical_name=logical_name,
            provider="foundry",
            model=model,
            client=client,
            capabilities=_capabilities(config),
        )

    def _openai_client(self, provider: str, config: dict[str, Any]) -> tuple[Any, str]:
        model = _required(config, "deployment")
        if provider == "databricks":
            from databricks_openai import DatabricksOpenAI

            return DatabricksOpenAI(), model
        if provider == "foundry":
            from azure.identity import get_bearer_token_provider
            from openai import OpenAI

            from aai_core.identity import azure_credential

            endpoint = _required(config, "endpoint").rstrip("/")
            credential = azure_credential(self.context.settings.azure_identity)
            token_provider = get_bearer_token_provider(
                credential, config.get("scope", "https://ai.azure.com/.default")
            )
            return (
                OpenAI(
                    base_url=f"{endpoint}/openai/v1/",
                    api_key=token_provider,
                ),
                model,
            )
        raise ProviderConfigurationError(f"Unsupported model provider: {provider}")

    def _azure_search(self, logical_name: str, config: dict[str, Any]):
        from azure.search.documents import SearchClient

        from aai_core.identity import azure_credential

        client = SearchClient(
            endpoint=_required(config, "endpoint"),
            index_name=_required(config, "index"),
            credential=azure_credential(self.context.settings.azure_identity),
        )
        return AzureAISearchRetriever(
            logical_name=logical_name,
            client=client,
            content_field=config.get("content_field", "content"),
            id_field=config.get("id_field", "id"),
            source_uri_field=config.get("source_uri_field", "source_uri"),
            chunk_id_field=config.get("chunk_id_field", "chunk_id"),
            vector_fields=config.get("vector_fields", ()),
        )

    def _databricks_search(self, logical_name: str, config: dict[str, Any]):
        from databricks.ai_search.client import AISearchClient

        client = AISearchClient()
        index = client.get_index(
            endpoint_name=_required(config, "endpoint"),
            index_name=_required(config, "index"),
        )
        return DatabricksAISearchRetriever(
            logical_name=logical_name,
            index=index,
            columns=config.get("columns", ["id", "content", "source_uri", "chunk_id"]),
            content_field=config.get("content_field", "content"),
            id_field=config.get("id_field", "id"),
            source_uri_field=config.get("source_uri_field", "source_uri"),
            chunk_id_field=config.get("chunk_id_field", "chunk_id"),
        )

    @staticmethod
    def _config(
        logical_name: str, configurations: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        try:
            return dict(configurations[logical_name])
        except KeyError as error:
            raise ProviderConfigurationError(
                f"Unknown logical resource: {logical_name!r}"
            ) from error

    @staticmethod
    def _provider(config: dict[str, Any]) -> str:
        return str(_required(config, "provider")).lower()


def _required(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if value in {None, ""}:
        raise ProviderConfigurationError(f"Provider configuration requires {key!r}")
    return str(value)


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _capabilities(config: dict[str, Any]) -> ModelCapabilities:
    supplied = dict(config.get("capabilities", {}))
    valid = set(ModelCapabilities.__dataclass_fields__)
    unknown = set(supplied).difference(valid)
    if unknown:
        raise ProviderConfigurationError(
            "Unknown model capabilities: " + ", ".join(sorted(unknown))
        )
    return ModelCapabilities(**supplied)
