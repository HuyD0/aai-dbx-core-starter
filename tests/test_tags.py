import pytest
from pydantic import ValidationError

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


def test_resource_context_is_strict_and_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        context(application=123)

    values = context().model_dump()
    values["invented_term"] = "value"
    with pytest.raises(ValidationError):
        ResourceContext(**values)


def test_lifecycle_uses_a_small_descriptive_vocabulary():
    assert context(lifecycle="CANDIDATE").lifecycle.value == "candidate"

    with pytest.raises(ValidationError, match="lifecycle must be one of"):
        context(lifecycle="validation")
