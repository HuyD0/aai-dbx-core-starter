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
from aai_core.tags import databricks_ai_gateway_request_headers

if TYPE_CHECKING:
    from aai_core.context import PlatformContext


class ProviderResolver:
    def __init__(self, context: PlatformContext) -> None:
        self.context = context
        self._models: dict[str, Any] = {}
        self._embeddings: dict[str, Any] = {}
        self._retrievers: dict[str, Any] = {}
        self._model_factories: dict[str, Callable[[str, dict[str, Any]], Any]] = {
            "databricks": self._openai_compatible_model,
            "foundry": self._openai_compatible_model,
            "azure_apim": self._openai_compatible_model,
        }
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
        client, model, _ = self._openai_clients(provider, config)
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

    def _openai_compatible_model(self, logical_name: str, config: dict[str, Any]):
        provider = self._provider(config)
        client, model, async_client_factory = self._openai_clients(provider, config)
        return OpenAICompatibleChatModel(
            logical_name=logical_name,
            provider=provider,
            model=model,
            client=client,
            async_client_factory=async_client_factory,
            capabilities=_capabilities(config),
        )

    def _openai_clients(
        self, provider: str, config: dict[str, Any]
    ) -> tuple[Any, str, Callable[[], Any]]:
        """Build native sync and async clients from one governed configuration."""

        model = _required(config, "deployment")
        native_options = _native_client_options(config)
        if provider == "databricks":
            try:
                from databricks_openai import AsyncDatabricksOpenAI, DatabricksOpenAI
            except ModuleNotFoundError as error:
                if error.name != "databricks_openai":
                    raise
                raise ProviderConfigurationError(
                    "The Databricks model provider requires the optional "
                    "'databricks-openai' package.",
                    remediation="Install the SDK's Databricks dependencies with "
                    "`python -m pip install 'aai-core[databricks]'`, then restart "
                    "the Python process. In this source repository, run "
                    "`make examples-install` and select `.venv/bin/python` as "
                    "the notebook kernel.",
                ) from error

            # These headers are fixed when the governed native clients are
            # constructed. The adapters reject per-call headers, so application
            # code cannot remove or replace attribution on individual requests.
            native_options["default_headers"] = databricks_ai_gateway_request_headers(
                self.context.tags
            )
            return (
                DatabricksOpenAI(**native_options),
                model,
                lambda: AsyncDatabricksOpenAI(**native_options),
            )
        if provider == "foundry":
            from openai import AsyncOpenAI, OpenAI

            endpoint = _required(config, "endpoint").rstrip("/")
            token_provider = self._bearer_token_provider(
                config.get("token_scope")
                or config.get("scope")
                or "https://ai.azure.com/.default"
            )
            client_options = {
                **native_options,
                "base_url": f"{endpoint}/openai/v1/",
                "api_key": token_provider,
            }
            return (
                OpenAI(**client_options),
                model,
                lambda: AsyncOpenAI(**client_options),
            )
        if provider == "azure_apim":
            from openai import AsyncOpenAI, OpenAI

            # The enterprise AI-gateway path: the APIM (or any OpenAI-
            # compatible gateway) base URL is taken verbatim, auth is a
            # keyless Entra bearer token for the gateway's own audience, and
            # an optional per-team subscription key is resolved through the
            # secret machinery — never inlined in configuration.
            base_url = _required(config, "base_url").rstrip("/")
            token_provider = self._bearer_token_provider(
                _required(config, "token_scope")
            )
            client_options: dict[str, Any] = {
                **native_options,
                "base_url": base_url,
                "api_key": token_provider,
            }
            headers = self._subscription_key_headers(config)
            if headers:
                client_options["default_headers"] = headers
            api_version = config.get("api_version")
            if api_version:
                client_options["default_query"] = {"api-version": str(api_version)}
            return (
                OpenAI(**client_options),
                model,
                lambda: AsyncOpenAI(**client_options),
            )
        raise ProviderConfigurationError(f"Unsupported model provider: {provider}")

    def _bearer_token_provider(self, scope: str):
        from azure.identity import get_bearer_token_provider

        from aai_core.identity import azure_credential

        credential = azure_credential(self.context.settings.azure_identity)
        return get_bearer_token_provider(credential, scope)

    def _subscription_key_headers(
        self, config: dict[str, Any]
    ) -> dict[str, str] | None:
        reference = config.get("subscription_key")
        if not reference:
            return None
        try:
            secret = self.context.secrets.resolve(str(reference))
        except ValueError as error:
            # Deliberately do not echo the configured value: if someone pasted
            # a raw key instead of a reference, it must not reach logs.
            raise ProviderConfigurationError(
                "subscription_key is not a valid secret reference",
                remediation="Use a keyvault://<vault>/<name> or "
                "databricks-secret://<scope>/<key> reference in "
                "aai-platform.yml; never place the key value in "
                "configuration.",
            ) from error
        header = str(config.get("subscription_key_header", "api-key"))
        return {header: secret.reveal()}

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
            embedding_provider=self._retriever_embedding(config),
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
            embedding_provider=self._retriever_embedding(config),
        )

    def _retriever_embedding(self, config: dict[str, Any]):
        """Resolve the optional `embedding` logical name a retriever uses to
        embed queries when the caller supplies no vector."""

        logical_name = config.get("embedding")
        return self.embedding(str(logical_name)) if logical_name else None

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


def _native_client_options(config: dict[str, Any]) -> dict[str, Any]:
    from aai_core.providers.openai_compatible import (
        DEFAULT_MAX_RETRIES,
        DEFAULT_TIMEOUT_SECONDS,
    )

    options: dict[str, Any] = {
        "max_retries": int(config.get("max_retries", DEFAULT_MAX_RETRIES)),
    }
    if "timeout_seconds" in config:
        timeout = config["timeout_seconds"]
        options["timeout"] = None if timeout is None else float(timeout)
    else:
        options["timeout"] = DEFAULT_TIMEOUT_SECONDS
    return options


def _capabilities(config: dict[str, Any]) -> ModelCapabilities:
    supplied = dict(config.get("capabilities", {}))
    valid = set(ModelCapabilities.__dataclass_fields__)
    unknown = set(supplied).difference(valid)
    if unknown:
        raise ProviderConfigurationError(
            "Unknown model capabilities: " + ", ".join(sorted(unknown))
        )
    return ModelCapabilities(**supplied)
