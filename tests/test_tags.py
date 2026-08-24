import pytest
from pydantic import ValidationError

from aai_core.tags import (
    DATABRICKS_AI_GATEWAY_REQUEST_TAGS_HEADER,
    DatabricksAIRequestTags,
    DataClassification,
    ResourceContext,
    databricks_ai_gateway_request_headers,
)


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
    assert context(lifecycle="VALIDATION").lifecycle.value == "validation"
    assert context().tag_schema_version == "2"

    with pytest.raises(ValidationError, match="not 'candidate'"):
        context(lifecycle="candidate")


def test_schema_v1_candidate_remains_readable_without_normalization():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        historical = context(lifecycle="candidate", tag_schema_version="1")

    assert historical.lifecycle.value == "candidate"
    assert historical.tag_schema_version == "1"
    assert historical.model_dump()["lifecycle"] == "candidate"

    with pytest.raises(ValidationError, match="schema version 1"):
        context(lifecycle="validation", tag_schema_version="1")


def test_data_classification_uses_a_closed_information_handling_vocabulary():
    assert (
        context(data_classification="CONFIDENTIAL").data_classification
        is DataClassification.CONFIDENTIAL
    )

    with pytest.raises(ValidationError, match="data_classification must be one of"):
        context(data_classification="customer-data")


def test_ai_gateway_request_tags_are_exact_immutable_context_projection():
    tags = DatabricksAIRequestTags.from_resource_context(context())

    assert tags.model_dump() == {
        "application_id": "claims_agent",
        "environment": "dev",
        "team": "claims-ai",
        "cost_center": "CC-1042",
        "application_version": "1.0.0",
    }
    assert "end_user_id" not in type(tags).model_fields

    with pytest.raises(ValidationError, match="frozen"):
        tags.team = "other-team"

    with pytest.raises(ValidationError):
        DatabricksAIRequestTags(
            **tags.model_dump(),
            end_user_id="opaque-but-not-approved",
        )

    invalid_types = tags.model_dump()
    invalid_types["team"] = 123
    with pytest.raises(ValidationError):
        DatabricksAIRequestTags(**invalid_types)


@pytest.mark.parametrize(
    "application",
    [" Claims   Agent ", "claims agent", "claims-agent", "claims_agent"],
)
def test_ai_gateway_request_tags_use_the_manifest_application_id(application):
    resource = context(application=application)

    tags = DatabricksAIRequestTags.from_resource_context(resource)

    assert tags.application_id == "claims_agent"
    assert (
        '"application_id":"claims_agent"'
        in databricks_ai_gateway_request_headers(resource)[
            DATABRICKS_AI_GATEWAY_REQUEST_TAGS_HEADER
        ]
    )


def test_ai_gateway_request_tag_header_is_deterministic_compact_json():
    headers = databricks_ai_gateway_request_headers(context())

    assert headers == {
        DATABRICKS_AI_GATEWAY_REQUEST_TAGS_HEADER: (
            '{"application_id":"claims_agent",'
            '"application_version":"1.0.0",'
            '"cost_center":"CC-1042",'
            '"environment":"dev",'
            '"team":"claims-ai"}'
        )
    }
    value = headers[DATABRICKS_AI_GATEWAY_REQUEST_TAGS_HEADER]
    assert " " not in value
    assert "end_user_id" not in value


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("application", "person@example.com", "personal identifier"),
        ("environment", " dev", "surrounding whitespace"),
        ("team", "token=not-safe", "credential material"),
        ("cost_center", "CC 1042", "ASCII letters"),
        ("release", "dapi0123456789abcdefghijkl", "credential material"),
        ("release", "https://example.test/release", "credential material"),
        ("application", "Bearer not-a-tag-value", "credential material"),
        ("team", "unknown", "placeholder"),
    ],
)
def test_ai_gateway_request_tags_reject_pii_credentials_and_invalid_values(
    field, value, message
):
    with pytest.raises(ValidationError, match=message) as excinfo:
        DatabricksAIRequestTags.from_resource_context(context(**{field: value}))
    assert value not in str(excinfo.value)
