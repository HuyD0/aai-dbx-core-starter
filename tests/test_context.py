"""Composition-root wiring tests for bootstrap()/PlatformContext."""

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
