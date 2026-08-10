import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

import pytest

from aai_core.logging import RedactingFilter, Redactor
from aai_core.secrets import (
    AzureKeyVaultSecretProvider,
    DatabricksSecretProvider,
    EnvironmentSecretProvider,
    SecretRef,
    SecretResolver,
    SecretValue,
    _databricks_workspace_close_resource,
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


def test_databricks_fallback_decodes_and_closes_one_workspace(
    monkeypatch,
):
    import builtins

    from conftest import install_fake_module

    workspaces = []

    class Secrets:
        @staticmethod
        def get_secret(*, scope, key):
            return type("Response", (), {"value": "cmVzb2x2ZWQ="})()

    class Workspace:
        def __init__(self):
            self.close_calls = 0
            self.secrets = Secrets()
            workspaces.append(self)

        def close(self):
            self.close_calls += 1

    monkeypatch.delattr(builtins, "dbutils", raising=False)
    install_fake_module(monkeypatch, "databricks.sdk", WorkspaceClient=Workspace)
    provider = DatabricksSecretProvider()

    assert provider.resolve(
        SecretRef.parse("databricks-secret://application/vendor-key")
    ) == ("resolved")
    assert provider.resolve(
        SecretRef.parse("databricks-secret://application/other-key")
    ) == ("resolved")

    provider.close()

    assert len(workspaces) == 1
    assert workspaces[0].close_calls == 1


def test_databricks_fallback_closes_certified_sdk_transport_session():
    class Session:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    session = Session()
    workspace = type(
        "Workspace",
        (),
        {
            "api_client": type(
                "ApiClient",
                (),
                {"_api_client": type("Transport", (), {"_session": session})()},
            )()
        },
    )()

    resource = _databricks_workspace_close_resource(workspace)
    resource.close()

    assert session.close_calls == 1


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


def test_secret_cache_performs_one_cold_load_under_concurrency():
    calls = 0
    call_lock = Lock()

    def get_secret(scope, key):
        nonlocal calls
        with call_lock:
            calls += 1
        time.sleep(0.02)
        return f"{scope}:{key}:resolved"

    provider = DatabricksSecretProvider(getter=get_secret)
    reference = SecretRef.parse("databricks-secret://application/vendor-key")

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: provider.resolve(reference), range(16)))

    assert values == ["application:vendor-key:resolved"] * 16
    assert calls == 1


def test_databricks_fallback_getter_is_lazy_singleton_and_owned(monkeypatch):
    factory_calls = 0

    class OwnedWorkspace:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    workspace = OwnedWorkspace()

    def fallback_factory():
        nonlocal factory_calls
        factory_calls += 1
        time.sleep(0.01)
        return (lambda scope, key: f"{scope}:{key}"), workspace

    monkeypatch.setattr("aai_core.secrets._databricks_secret_getter", fallback_factory)
    provider = DatabricksSecretProvider()
    references = [
        SecretRef.parse(f"databricks-secret://application/key-{index}")
        for index in range(8)
    ]

    assert factory_calls == 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(provider.resolve, references))

    assert values == [f"application:key-{index}" for index in range(8)]
    assert factory_calls == 1

    provider.close()
    provider.close()

    assert workspace.close_calls == 1


def test_databricks_provider_keeps_injected_getter_caller_owned():
    class InjectedGetter:
        def __init__(self):
            self.close_calls = 0

        def __call__(self, scope, key):
            return f"{scope}:{key}"

        def close(self):
            self.close_calls += 1

    getter = InjectedGetter()
    provider = DatabricksSecretProvider(getter=getter)

    provider.resolve(SecretRef.parse("databricks-secret://application/key"))
    provider.close()

    assert getter.close_calls == 0


def test_secret_cache_reloads_only_after_ttl(monkeypatch):
    now = 100.0
    calls = 0

    def monotonic():
        return now

    def getter(scope, key):
        nonlocal calls
        calls += 1
        return f"value-{calls}"

    monkeypatch.setattr("aai_core.secrets.time.monotonic", monotonic)
    provider = DatabricksSecretProvider(getter=getter, ttl_seconds=10)
    reference = SecretRef.parse("databricks-secret://application/key")

    assert provider.resolve(reference) == "value-1"
    now = 109.0
    assert provider.resolve(reference) == "value-1"
    now = 111.0
    assert provider.resolve(reference) == "value-2"
    assert calls == 2


@pytest.mark.parametrize(
    "provider",
    [
        DatabricksSecretProvider(getter=lambda scope, key: "value"),
        EnvironmentSecretProvider(environ={"KEY": "value"}),
    ],
)
def test_caching_providers_reject_resolve_after_close(provider):
    reference = SecretRef.parse(
        "databricks-secret://application/key"
        if isinstance(provider, DatabricksSecretProvider)
        else "env://local/KEY"
    )
    provider.close()

    with pytest.raises(RuntimeError, match="is closed"):
        provider.resolve(reference)


def test_close_race_prevents_inflight_secret_from_being_returned_or_cached():
    load_started = Event()
    release_load = Event()

    def getter(scope, key):
        load_started.set()
        assert release_load.wait(timeout=2)
        return "must-not-escape"

    provider = DatabricksSecretProvider(getter=getter)
    reference = SecretRef.parse("databricks-secret://application/key")

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(provider.resolve, reference)
        assert load_started.wait(timeout=2)
        follower = pool.submit(provider.resolve, reference)
        time.sleep(0.01)
        provider.close()
        with pytest.raises(RuntimeError, match="is closed"):
            follower.result(timeout=0.2)
        release_load.set()
        with pytest.raises(RuntimeError, match="is closed"):
            leader.result()

    assert provider._cache == {}


def test_secret_cache_shares_a_failure_but_does_not_cache_it():
    calls = 0
    call_lock = Lock()
    release = Event()
    start_together = Barrier(9)

    def get_secret(scope, key):
        nonlocal calls
        with call_lock:
            calls += 1
        assert release.wait(timeout=2)
        raise RuntimeError("provider unavailable")

    provider = DatabricksSecretProvider(getter=get_secret)
    reference = SecretRef.parse("databricks-secret://application/vendor-key")

    def resolve():
        start_together.wait(timeout=2)
        return provider.resolve(reference)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(resolve) for _ in range(8)]
        start_together.wait(timeout=2)
        time.sleep(0.02)
        release.set()
        for future in futures:
            with pytest.raises(RuntimeError, match="provider unavailable"):
                future.result()

    assert calls == 1
    with pytest.raises(RuntimeError, match="provider unavailable"):
        provider.resolve(reference)
    assert calls == 2


def test_key_vault_client_construction_is_singleton_under_concurrency():
    factory_calls = 0
    call_lock = Lock()

    class Client:
        def get_secret(self, name):
            return type("Secret", (), {"value": name})()

    def client_factory(url, credential):
        nonlocal factory_calls
        with call_lock:
            factory_calls += 1
        time.sleep(0.01)
        return Client()

    provider = AzureKeyVaultSecretProvider(
        credential=object(),
        client_factory=client_factory,
    )
    references = [
        SecretRef.parse(f"keyvault://team-vault/secret-{index}") for index in range(8)
    ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(provider.resolve, references))

    assert values == [f"secret-{index}" for index in range(8)]
    assert factory_calls == 1


def test_key_vault_close_orders_clients_before_owned_credential_once():
    events = []

    class Credential:
        def close(self):
            events.append("credential")

    class Client:
        def get_secret(self, name):
            return type("Secret", (), {"value": name})()

        def close(self):
            events.append("client")

    provider = AzureKeyVaultSecretProvider(
        client_factory=lambda url, credential: Client(),
    )
    provider._credential = Credential()
    provider.resolve(SecretRef.parse("keyvault://team-vault/key"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: provider.close(), range(16)))

    assert events == ["client", "credential"]


def test_secret_resolver_closes_owned_but_not_registered_providers():
    class ClosableProvider:
        def __init__(self):
            self.close_calls = 0

        def resolve(self, reference):
            return "secret"

        def close(self):
            self.close_calls += 1

    owned = ClosableProvider()
    caller_owned = ClosableProvider()
    resolver = SecretResolver()
    resolver.register("owned", owned, owned=True)
    resolver.register("external", caller_owned)

    resolver.close()
    resolver.close()

    assert owned.close_calls == 1
    assert caller_owned.close_calls == 0
    with pytest.raises(RuntimeError, match="SecretResolver is closed"):
        resolver.resolve("external://scope/name")


def test_secret_resolver_rejects_duplicate_scheme_without_displacing_owner():
    class ClosableProvider(StaticProvider):
        def __init__(self, value):
            super().__init__(value)
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    original = ClosableProvider("original")
    replacement = ClosableProvider("replacement")
    resolver = SecretResolver()
    resolver.register("test", original, owned=True)

    with pytest.raises(ValueError, match="already registered"):
        resolver.register("test", replacement, owned=True)

    assert resolver.resolve("test://scope/name").reveal() == "original"
    resolver.close()
    assert original.close_calls == 1
    assert replacement.close_calls == 0


def test_secret_resolver_rechecks_close_after_provider_io():
    load_started = Event()
    release_load = Event()

    class BlockingProvider:
        def resolve(self, reference):
            load_started.set()
            assert release_load.wait(timeout=2)
            return "must-not-be-registered"

    registered = []
    redactor = Redactor()
    redactor.register = registered.append
    resolver = SecretResolver(redactor=redactor)
    resolver.register("blocked", BlockingProvider())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(resolver.resolve, "blocked://scope/name")
        assert load_started.wait(timeout=2)
        resolver.close()
        release_load.set()
        with pytest.raises(RuntimeError, match="SecretResolver is closed"):
            future.result()

    assert registered == []
