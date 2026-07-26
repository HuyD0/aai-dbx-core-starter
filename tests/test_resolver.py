"""ProviderResolver wiring tests — the code that turns aai-platform.yml into
clients, exercised hermetically with fake provider modules."""

import builtins
from types import SimpleNamespace

import pytest
from conftest import install_fake_module

from aai_core.providers.resolver import ProviderResolver, _capabilities
from aai_core.providers.types import ProviderConfigurationError
from aai_core.secrets import SecretResolver


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


def _context(models=None, embeddings=None, retrievers=None):
    secrets = SecretResolver()
    secrets.register("keyvault", FakeSecretProvider())
    return SimpleNamespace(
        settings=SimpleNamespace(
            azure_identity="azure_cli",
            models=models or {},
            embeddings=embeddings or {},
            retrievers=retrievers or {},
        ),
        secrets=secrets,
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


def test_same_logical_name_resolves_via_all_three_providers(monkeypatch):
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
        "foundry": {
            "provider": "foundry",
            "endpoint": "https://foundry.services.ai.azure.com",
            "deployment": "gpt-chat",
        },
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
                    "provider": "foundry",
                    "endpoint": "https://foundry.services.ai.azure.com",
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
