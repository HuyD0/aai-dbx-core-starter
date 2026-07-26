import logging
import pickle

import pytest

from aai_core.logging import RedactingFilter, Redactor
from aai_core.secrets import (
    AzureKeyVaultSecretProvider,
    DatabricksSecretProvider,
    SecretRef,
    SecretResolver,
    SecretValue,
)


class StaticProvider:
    def __init__(self, value):
        self.value = value

    def resolve(self, reference):
        return self.value


def test_secret_reference_and_value_never_render_raw():
    reference = SecretRef.parse("keyvault://team-vault/vendor-key")
    value = SecretValue("super-secret")

    assert reference.authority == "team-vault"
    assert reference.name == "vendor-key"
    assert str(value) == "[REDACTED]"
    assert "super-secret" not in repr(value)
    assert value.reveal() == "super-secret"
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(value)


def test_resolved_secret_is_registered_for_log_redaction():
    redactor = Redactor()
    resolver = SecretResolver(redactor=redactor)
    resolver.register("test", StaticProvider("super-secret"))

    resolver.resolve("test://scope/name")

    record = logging.LogRecord(
        "test",
        logging.INFO,
        __file__,
        1,
        "token=%s",
        ("super-secret",),
        None,
    )
    assert RedactingFilter(redactor).filter(record)
    assert record.getMessage() == "token=[REDACTED]"


def test_unknown_secret_scheme_fails_closed():
    with pytest.raises(ValueError, match="No secret provider"):
        SecretResolver().resolve("unknown://scope/name")


def test_key_vault_provider_uses_vault_url_and_memory_cache():
    calls = []

    class Client:
        def get_secret(self, name):
            calls.append(name)
            return type("Secret", (), {"value": "resolved"})()

    provider = AzureKeyVaultSecretProvider(
        credential=object(),
        client_factory=lambda url, credential: (calls.append(url) or Client()),
    )
    reference = SecretRef.parse("keyvault://team-vault/vendor-key")

    assert provider.resolve(reference) == "resolved"
    assert provider.resolve(reference) == "resolved"
    assert calls == ["https://team-vault.vault.azure.net", "vendor-key"]


def test_databricks_provider_accepts_injected_notebook_getter():
    provider = DatabricksSecretProvider(
        getter=lambda scope, key: f"{scope}:{key}:resolved"
    )

    assert provider.resolve(
        SecretRef.parse("databricks-secret://application/vendor-key")
    ) == ("application:vendor-key:resolved")


def test_key_vault_provider_honors_configured_identity_mode(monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        "aai_core.identity.azure_credential",
        lambda mode, **kwargs: recorded.setdefault("mode", mode) or "credential",
    )
    provider = AzureKeyVaultSecretProvider(
        azure_identity="managed_identity",
        client_factory=lambda url, credential: type(
            "Client",
            (),
            {"get_secret": staticmethod(lambda name: type("S", (), {"value": "v"}))},
        )(),
    )

    provider.resolve(SecretRef.parse("keyvault://team-vault/vendor-key"))

    assert recorded["mode"] == "managed_identity"


def test_default_resolver_threads_identity_mode_into_key_vault():
    from aai_core.secrets import default_secret_resolver

    resolver = default_secret_resolver(azure_identity="workload_identity")

    provider = resolver._providers["keyvault"]
    assert provider._azure_identity == "workload_identity"
