import pytest

from aai_core.tags import ResourceContext


def context(**overrides):
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


def test_context_projects_tags_for_each_destination():
    resource = context(application="claims agent")

    assert resource.for_mlflow()["aai.cost_center"] == "CC-1042"
    assert resource.for_databricks()["owner_group"] == "group:claims-ai-owners"
    assert resource.for_azure()["application"] == "claims agent"


def test_application_cannot_override_controlled_tags():
    with pytest.raises(ValueError, match="controlled tags"):
        context().merged({"cost_center": "another"})


def test_production_owner_must_be_group_not_email():
    with pytest.raises(ValueError, match="group identifier"):
        context(owner_group="person@example.com").validate(strict=True)
