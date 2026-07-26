from pathlib import Path

import pytest

from aai_core.runtime import PlatformSettings


def test_settings_precedence(tmp_path: Path):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        """
platform:
  application: yaml-app
  project: yaml-project
  environment: dev
  team: yaml-team
  owner_group: group:owners
  cost_center: CC-1
  data_classification: internal
  lifecycle: experimental
  repository: org/repo
  release: dev
  catalog: yaml_catalog
  schema: app
  experiment_name: /Shared/app
providers:
  models:
    chat:
      provider: databricks
      deployment: endpoint
""",
        encoding="utf-8",
    )

    settings = PlatformSettings.load(
        config,
        environ={"AAI_APPLICATION": "env-app"},
        application="explicit-app",
    )

    assert settings.resource.application == "explicit-app"
    assert settings.resource.team == "yaml-team"
    assert settings.models["chat"]["deployment"] == "endpoint"


def test_production_rejects_placeholder_values(tmp_path: Path):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        """
platform:
  application: app
  project: project
  environment: production
  team: team
  owner_group: group:owners
  cost_center: unset
  data_classification: internal
  lifecycle: production
  repository: org/repo
  release: "1.0.0"
  catalog: main
  schema: app
  experiment_name: /Shared/app
  azure_identity: workload_identity
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="placeholders"):
        PlatformSettings.load(config, environ={})


def test_production_rejects_auto_identity(tmp_path: Path):
    config = tmp_path / "aai-platform.yml"
    config.write_text(
        """
platform:
  application: app
  project: project
  environment: production
  team: team
  owner_group: group:owners
  cost_center: CC-1
  data_classification: confidential
  lifecycle: production
  repository: org/repo
  release: "1.0.0"
  catalog: main
  schema: app
  experiment_name: /Shared/app
  azure_identity: auto
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must select"):
        PlatformSettings.load(config, environ={})


def test_staging_is_a_strict_environment():
    from aai_core.runtime import PlatformSettings
    from aai_core.tags import ResourceContext

    settings = PlatformSettings(
        resource=ResourceContext(
            application="app",
            project="proj",
            environment="staging",
            team="team",
            owner_group="group:owners",
            cost_center="CC-1",
            data_classification="internal",
            lifecycle="candidate",
            repository="org/repo",
            release="1.0.0",
        )
    )

    assert settings.strict


def test_default_experiment_is_one_application_comparison_space():
    from aai_core.runtime import PlatformSettings
    from aai_core.tags import ResourceContext

    settings = PlatformSettings(
        resource=ResourceContext(
            application="claims-agent",
            project="claims",
            environment="dev",
            team="claims-ai",
            owner_group="group:claims-ai-owners",
            cost_center="CC-1",
            data_classification="internal",
            lifecycle="experimental",
            repository="org/repo",
            release="dev",
        )
    )

    assert settings.effective_experiment_name == "/Shared/claims-ai-claims-claims-agent"


def test_settings_dump_round_trips_public_aliases_and_freezes_nested_config():
    from aai_core.tags import ResourceContext

    settings = PlatformSettings(
        resource=ResourceContext(
            application="claims-agent",
            project="claims",
            environment="dev",
            team="claims-ai",
            owner_group="group:claims-ai-owners",
            cost_center="CC-1",
            data_classification="internal",
            lifecycle="experimental",
            repository="org/repo",
            release="dev",
        ),
        schema="claims",
        models={
            "general-chat": {
                "provider": "databricks",
                "deployment": "claims-chat",
            }
        },
        secrets={"vendor": "keyvault://example/vendor"},
    )

    dumped = settings.model_dump()
    restored = PlatformSettings.model_validate(dumped)

    assert "schema" in dumped
    assert "schema_name" not in dumped
    assert restored.schema == "claims"
    assert restored.models == settings.models
    assert settings.model_copy(deep=True).models == settings.models
    with pytest.raises(TypeError):
        settings.models["other"] = {}
    with pytest.raises(TypeError):
        settings.models["general-chat"]["deployment"] = "other"
    with pytest.raises(TypeError):
        settings.secrets["vendor"] = "raw-secret"
