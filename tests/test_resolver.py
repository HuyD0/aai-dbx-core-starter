"""ProviderResolver wiring tests — the code that turns aai-platform.yml into
clients, exercised hermetically with fake provider modules."""

import builtins
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace

import pytest
from conftest import install_fake_module

from aai_core.providers.openai_compatible import OpenAICompatibleChatModel
from aai_core.providers.resolver import ProviderResolver, _capabilities
from aai_core.providers.types import ProviderConfigurationError
from aai_core.secrets import SecretResolver
from aai_core.tags import DATABRICKS_AI_GATEWAY_REQUEST_TAGS_HEADER, ResourceContext


class FakeCompletions:
    def create(self, **request):
        return SimpleNamespace(
            model="resolved-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", tool_calls=None)
                )
            ],
            usage=None,
        )


class FakeOpenAIClient:
    def __init__(self, **options):
        self.options = options
        self.chat = SimpleNamespace(completions=FakeCompletions())
        self.embeddings = SimpleNamespace()


class FakeAsyncOpenAIClient:
    def __init__(self, **options):
        self.options = options


class FakeSecretProvider:
    def resolve(self, reference):
        assert str(reference) == "keyvault://gw-vault/apim-subscription-key"
        return "subscription-key-value"


def _resource_context(**overrides):
    values = {
        "application": "claims-agent",
        "project": "claims",
        "environment": "dev",
        "team": "claims-ai",
        "owner_group": "group:claims-ai-owners",
        "cost_center": "CC-1042",
        "data_classification": "internal",
        "lifecycle": "experimental",
        "repository": "org/claims",
        "release": "1.0.0",
    }
    values.update(overrides)
    return ResourceContext(**values)


def _context(models=None, embeddings=None, retrievers=None, resource=None):
    secrets = SecretResolver()
    secrets.register("keyvault", FakeSecretProvider())
    tags = resource or _resource_context()
    return SimpleNamespace(
        settings=SimpleNamespace(
            azure_identity="azure_cli",
            models=models or {},
            embeddings=embeddings or {},
            retrievers=retrievers or {},
            resource=tags,
        ),
        secrets=secrets,
        tags=tags,
    )


def _install_identity_fakes(monkeypatch, recorded):
    monkeypatch.setattr(
        "aai_core.identity.azure_credential",
        lambda mode, **kwargs: recorded.setdefault("identity_mode", mode)
        or "credential",
    )
    install_fake_module(
        monkeypatch,
        "azure.identity",
        get_bearer_token_provider=lambda credential, scope: recorded.update(
            {"scope": scope}
        )
        or "token-provider",
    )


def test_unknown_logical_name_and_provider_fail_with_clear_errors():
    resolver = ProviderResolver(_context(models={"chat": {"provider": "watson"}}))

    with pytest.raises(ProviderConfigurationError):
        resolver.model("missing")
    with pytest.raises(ProviderConfigurationError):
        resolver.model("chat")


def test_azure_apim_model_wires_gateway_options(monkeypatch):
    recorded = {}
    _install_identity_fakes(monkeypatch, recorded)
    install_fake_module(
        monkeypatch,
        "openai",
        OpenAI=FakeOpenAIClient,
        AsyncOpenAI=FakeAsyncOpenAIClient,
    )

    resolver = ProviderResolver(
        _context(
            models={
                "general-chat": {
                    "provider": "azure_apim",
                    "base_url": "https://gw.azure-api.net/llm/",
                    "deployment": "gpt-chat",
                    "token_scope": "api://apim-gateway/.default",
                    "api_version": "2025-04-01",
                    "subscription_key": ("keyvault://gw-vault/apim-subscription-key"),
                }
            }
        )
    )

    model = resolver.model("general-chat")

    assert recorded["identity_mode"] == "azure_cli"
    assert recorded["scope"] == "api://apim-gateway/.default"
    assert model.native_client.options == {
        "base_url": "https://gw.azure-api.net/llm",
        "api_key": "token-provider",
        "max_retries": 2,
        "timeout": 60.0,
        "default_headers": {"api-key": "subscription-key-value"},
        "default_query": {"api-version": "2025-04-01"},
    }
    assert model.create_native_async_client().options == model.native_client.options
    assert model.generate([{"role": "user", "content": "q"}]).content == "answer"


def test_azure_apim_requires_token_scope_and_secret_reference(monkeypatch):
    recorded = {}
    _install_identity_fakes(monkeypatch, recorded)
    install_fake_module(
        monkeypatch,
        "openai",
        OpenAI=FakeOpenAIClient,
        AsyncOpenAI=FakeAsyncOpenAIClient,
    )

    resolver = ProviderResolver(
        _context(
            models={
                "no-scope": {
                    "provider": "azure_apim",
                    "base_url": "https://gw.azure-api.net/llm",
                    "deployment": "gpt-chat",
                },
                "raw-key": {
                    "provider": "azure_apim",
                    "base_url": "https://gw.azure-api.net/llm",
                    "deployment": "gpt-chat",
                    "token_scope": "api://apim-gateway/.default",
                    "subscription_key": "raw-key-value-pasted-by-mistake",
                },
            }
        )
    )

    with pytest.raises(ProviderConfigurationError):
        resolver.model("no-scope")
    with pytest.raises(ProviderConfigurationError) as excinfo:
        resolver.model("raw-key")
    # The raw value must never be echoed into the error/logs.
    assert "raw-key-value" not in str(excinfo.value)
    assert "keyvault://" in str(excinfo.value.remediation)


def test_same_logical_name_resolves_via_both_providers(monkeypatch):
    recorded = {}
    _install_identity_fakes(monkeypatch, recorded)
    install_fake_module(
        monkeypatch,
        "openai",
        OpenAI=FakeOpenAIClient,
        AsyncOpenAI=FakeAsyncOpenAIClient,
    )
    install_fake_module(
        monkeypatch,
        "databricks_openai",
        DatabricksOpenAI=FakeOpenAIClient,
        AsyncDatabricksOpenAI=FakeAsyncOpenAIClient,
    )

    configs = {
        "databricks": {"provider": "databricks", "deployment": "chat-endpoint"},
        "azure_apim": {
            "provider": "azure_apim",
            "base_url": "https://gw.azure-api.net/llm",
            "deployment": "gpt-chat",
            "token_scope": "api://apim-gateway/.default",
        },
    }
    for provider, config in configs.items():
        resolver = ProviderResolver(_context(models={"general-chat": config}))
        model = resolver.model("general-chat")
        assert model.provider == provider
        assert model.logical_name == "general-chat"
        headers = model.native_client.options.get("default_headers", {})
        if provider == "databricks":
            assert json.loads(headers[DATABRICKS_AI_GATEWAY_REQUEST_TAGS_HEADER]) == {
                "application_id": "claims_agent",
                "application_version": "1.0.0",
                "cost_center": "CC-1042",
                "environment": "dev",
                "team": "claims-ai",
            }
            assert (
                model.create_native_async_client().options["default_headers"] == headers
            )
        else:
            assert DATABRICKS_AI_GATEWAY_REQUEST_TAGS_HEADER not in headers


def test_databricks_embedding_client_carries_governed_request_tags(monkeypatch):
    install_fake_module(
        monkeypatch,
        "databricks_openai",
        DatabricksOpenAI=FakeOpenAIClient,
        AsyncDatabricksOpenAI=FakeAsyncOpenAIClient,
    )
    resolver = ProviderResolver(
        _context(
            embeddings={
                "knowledge-embedding": {
                    "provider": "databricks",
                    "deployment": "embedding-endpoint",
                    "dimensions": 1536,
                }
            },
            resource=_resource_context(
                application="claims agent",
                release="git-abc123",
            ),
        )
    )

    embedding = resolver.embedding("knowledge-embedding")
    header = embedding.native_client.options["default_headers"][
        DATABRICKS_AI_GATEWAY_REQUEST_TAGS_HEADER
    ]

    assert json.loads(header) == {
        "application_id": "claims_agent",
        "application_version": "git-abc123",
        "cost_center": "CC-1042",
        "environment": "dev",
        "team": "claims-ai",
    }


def test_missing_databricks_extra_has_actionable_remediation(monkeypatch):
    native_import = builtins.__import__

    def import_without_databricks_openai(name, *args, **kwargs):
        if name == "databricks_openai":
            raise ModuleNotFoundError(
                "No module named 'databricks_openai'",
                name="databricks_openai",
            )
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_databricks_openai)
    resolver = ProviderResolver(
        _context(
            models={
                "general-chat": {
                    "provider": "databricks",
                    "deployment": "chat-endpoint",
                }
            }
        )
    )

    with pytest.raises(ProviderConfigurationError) as excinfo:
        resolver.model("general-chat")

    assert "databricks-openai" in str(excinfo.value)
    assert "make examples-install" in str(excinfo.value.remediation)
    assert ".venv/bin/python" in str(excinfo.value.remediation)


@pytest.mark.parametrize("name", ["streaming", "responses_api"])
def test_native_only_features_are_not_stable_capability_flags(name):
    with pytest.raises(ProviderConfigurationError, match="Unknown model capabilities"):
        _capabilities({"capabilities": {name: True}})


def test_resilience_options_come_from_configuration(monkeypatch):
    recorded = {}
    _install_identity_fakes(monkeypatch, recorded)
    install_fake_module(
        monkeypatch,
        "openai",
        OpenAI=FakeOpenAIClient,
        AsyncOpenAI=FakeAsyncOpenAIClient,
    )

    resolver = ProviderResolver(
        _context(
            models={
                "general-chat": {
                    "provider": "azure_apim",
                    "base_url": "https://gw.azure-api.net/llm",
                    "token_scope": "api://apim-gateway/.default",
                    "deployment": "gpt-chat",
                    "timeout_seconds": 15,
                    "max_retries": 0,
                }
            }
        )
    )

    model = resolver.model("general-chat")

    assert model.native_client.options["timeout"] == 15.0
    assert model.native_client.options["max_retries"] == 0
    assert model.create_native_async_client().options == model.native_client.options


def test_resolver_constructs_one_owned_model_under_concurrency():
    calls = 0
    call_lock = Lock()

    class ClosableClient(FakeOpenAIClient):
        def close(self):
            pass

    def factory(logical_name, config):
        nonlocal calls
        with call_lock:
            calls += 1
        time.sleep(0.02)
        return OpenAICompatibleChatModel(
            logical_name=logical_name,
            provider="fake",
            model=config["deployment"],
            client=ClosableClient(),
        )

    resolver = ProviderResolver(
        _context(models={"general-chat": {"provider": "fake", "deployment": "m"}})
    )
    resolver._model_factories["fake"] = factory

    with ThreadPoolExecutor(max_workers=8) as pool:
        models = list(pool.map(lambda _: resolver.model("general-chat"), range(16)))

    assert calls == 1
    assert all(model is models[0] for model in models)


def test_resolver_closes_owned_native_clients_not_registered_models():
    class ClosableClient(FakeOpenAIClient):
        def __init__(self):
            super().__init__()
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    owned_client = ClosableClient()
    caller_client = ClosableClient()
    resolver = ProviderResolver(
        _context(models={"owned": {"provider": "fake", "deployment": "m"}})
    )
    resolver._model_factories["fake"] = lambda logical_name, config: (
        OpenAICompatibleChatModel(
            logical_name=logical_name,
            provider="fake",
            model=config["deployment"],
            client=owned_client,
        )
    )
    resolver.register_model(
        "registered",
        OpenAICompatibleChatModel(
            logical_name="registered",
            provider="fake",
            model="m",
            client=caller_client,
        ),
    )

    resolver.model("owned")
    resolver.close()
    resolver.close()

    assert owned_client.close_calls == 1
    assert caller_client.close_calls == 0
    with pytest.raises(RuntimeError, match="ProviderResolver is closed"):
        resolver.model("owned")
