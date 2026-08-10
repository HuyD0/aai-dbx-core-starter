"""Composition-root wiring tests for bootstrap()/PlatformContext."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from aai_core import PlatformContext, bootstrap


@pytest.fixture
def config_file(tmp_path):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        """
platform:
  application: wiring-test
  project: wiring
  environment: dev
  team: test-team
  owner_group: group:test-owners
  cost_center: CC-0000
  azure_identity: azure_cli

providers:
  models:
    general-chat:
      provider: databricks
      deployment: chat-endpoint
""",
        encoding="utf-8",
    )
    return config


def test_bootstrap_loads_settings_and_wires_lazy_services(config_file):
    context = bootstrap(config_file)

    assert isinstance(context, PlatformContext)
    assert context.tags.application == "wiring-test"
    assert context.settings.models["general-chat"]["provider"] == "databricks"
    # Lazy services are memoized single instances.
    assert context.providers is context.providers
    assert context.secrets is context.secrets


def test_context_secrets_honor_environment_and_identity_mode(config_file):
    context = bootstrap(config_file)

    resolver = context.secrets
    # Dev is non-strict, so the explicit local-only env provider is available…
    assert "env" in resolver._providers
    # …and the Key Vault provider inherits the configured identity mode.
    assert resolver._providers["keyvault"]._azure_identity == "azure_cli"


def test_bootstrap_overrides_reach_settings(config_file):
    context = bootstrap(config_file, team="override-team")

    assert context.tags.team == "override-team"


def test_lazy_services_are_singletons_under_concurrent_cold_access(
    config_file,
    monkeypatch,
):
    calls = 0
    call_lock = Lock()
    resolver = object()

    def build_resolver(**kwargs):
        nonlocal calls
        with call_lock:
            calls += 1
        time.sleep(0.02)
        return resolver

    monkeypatch.setattr("aai_core.context.default_secret_resolver", build_resolver)
    context = bootstrap(config_file)

    with ThreadPoolExecutor(max_workers=8) as pool:
        resolved = list(pool.map(lambda _: context.secrets, range(16)))

    assert calls == 1
    assert all(item is resolver for item in resolved)


def test_context_manager_closes_only_sdk_created_resources_once(
    config_file,
    monkeypatch,
):
    class Closable:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    workspace = Closable()
    secrets = Closable()
    providers = Closable()
    monkeypatch.setattr(
        "aai_core.context.databricks_workspace_client",
        lambda: workspace,
    )
    monkeypatch.setattr(
        "aai_core.context.default_secret_resolver",
        lambda **kwargs: secrets,
    )
    monkeypatch.setattr(
        "aai_core.providers.ProviderResolver",
        lambda context: providers,
    )

    with bootstrap(config_file) as context:
        assert context.workspace is workspace
        assert context.secrets is secrets
        assert context.providers is providers

    context.close()
    assert workspace.close_calls == 1
    assert secrets.close_calls == 1
    assert providers.close_calls == 1
    with pytest.raises(RuntimeError, match="PlatformContext is closed"):
        _ = context.workspace
