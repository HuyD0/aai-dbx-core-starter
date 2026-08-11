"""Versioned AI Platform Hub manifest contract tests."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import aai_console.hub.manifest as compatibility_manifest
from aai_core.manifest import (
    AIApplicationManifest,
    ManifestEnvelope,
    build_manifest_envelope,
    canonical_manifest_json,
    load_manifest,
    manifest_json_schema,
)

ROOT = Path(__file__).resolve().parents[1]
SHARED_TEMPLATE = ROOT / "templates" / "_shared" / "files" / "ai-app.yaml.tmpl"
SHARED_MANIFEST = ROOT / "templates" / "_shared" / "manifest.json"
TEMPLATE_VARIABLE = re.compile(r"\{\{\.(?P<name>[a-z_]+)\}\}")
COMMON_TEMPLATE_VARIABLES = {
    "application_name",
    "catalog",
    "cost_center",
    "owner_group",
    "project_name",
    "repository_url",
    "schema",
    "team",
}


def test_console_manifest_module_is_a_compatibility_export_of_the_sdk_contract():
    assert compatibility_manifest.AIApplicationManifest is AIApplicationManifest
    assert compatibility_manifest.load_manifest is load_manifest
    assert compatibility_manifest.manifest_json_schema is manifest_json_schema


def valid_document() -> dict:
    return {
        "apiVersion": "ai-platform/v1",
        "kind": "AIApplication",
        "metadata": {
            "id": "TFM-Analyst",
            "name": "TFM Analyst",
            "description": "Analyst assistant for TFM workflows.",
            "owner": "investment-ai@example.com",
            "supportGroup": "group:investment-ai-support",
            "businessDomain": "Investments-Technology",
            "costCenter": "CC-1234",
            "riskTier": "MEDIUM",
            "tags": {
                "Team": "investment-ai",
                "Domain": "investments-technology",
                "cost-center": "CC-1234",
                "application-id": "tfm-analyst",
            },
        },
        "spec": {
            "repository": {
                "url": "https://github.com/aai-test/tfm-analyst",
            },
            "authorization": {"mode": "USER"},
            "environments": {
                "Dev-West": {
                    "workspaceId": "123",
                    "databricksAppName": "tfm-analyst-dev",
                    "mlflowExperimentId": "456",
                    "aiGatewayService": "ai_platform.models.enterprise_chat",
                    "tags": {"Environment": "dev-west"},
                },
            },
            "resources": {
                "evaluationJobKey": "release-gate",
                "sqlWarehouseId": "abc123",
                "aiSearchIndexes": ["catalog.schema.index"],
                "unityCatalogFunctions": ["catalog.schema.safe_tool"],
                "mcpServices": ["catalog.schema.mcp_service"],
            },
            "evaluation": {
                "profile": "Grounded-Agent-v1",
                "dataset": "ai_platform.evaluations.tfm_analyst",
                "minimumCases": 30,
                "maximumAgeHours": 168,
                "thresholds": {
                    "Safety-Pass-Rate": 1,
                    "groundedness": 0.85,
                    "p95-latency-ms": 8000,
                    "token_count": 1000,
                },
            },
            "readiness": {"profile": "Medium-Risk-Production-v1"},
            "costControls": {"budgetPolicy": "Platform-Standard-v1"},
            "serviceLevels": {
                "maximumErrorRate": 0.02,
                "p95LatencyMs": 8000,
            },
        },
    }


def test_valid_manifest_normalizes_platform_owned_identifiers_and_aliases():
    manifest = load_manifest(valid_document())

    assert manifest.api_version == "ai-platform/v1"
    assert manifest.kind == "AIApplication"
    assert manifest.metadata.id == "tfm_analyst"
    assert manifest.metadata.business_domain == "investments_technology"
    assert manifest.metadata.risk_tier == "medium"
    assert dict(manifest.metadata.tags) == {
        "team": "investment-ai",
        "domain": "investments_technology",
        "cost_center": "CC-1234",
        "application_id": "tfm_analyst",
    }
    assert set(manifest.spec.environments) == {"dev_west"}
    environment = manifest.spec.environments["dev_west"]
    assert environment.workspace_id == "123"
    assert dict(environment.tags) == {"environment": "dev_west"}
    assert manifest.spec.authorization.mode == "user"
    assert manifest.spec.resources.evaluation_job_key == "release_gate"
    assert manifest.spec.resources.promotion_job_id is None
    assert manifest.spec.resources.promotion_job_key is None
    assert manifest.spec.evaluation.profile == "grounded_agent_v1"
    assert dict(manifest.spec.evaluation.thresholds) == {
        "safety_pass_rate": 1.0,
        "groundedness": 0.85,
        "p95_latency_ms": 8000.0,
        "token_count": 1000.0,
    }
    assert manifest.spec.readiness.profile == "medium_risk_production_v1"
    assert manifest.spec.cost_controls.budget_policy == "platform_standard_v1"

    serialized = manifest.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert serialized["apiVersion"] == "ai-platform/v1"
    assert serialized["metadata"]["supportGroup"] == "group:investment-ai-support"
    assert serialized["spec"]["resources"]["evaluationJobKey"] == "release_gate"
    assert serialized["spec"]["costControls"]["budgetPolicy"] == (
        "platform_standard_v1"
    )
    assert serialized["spec"]["serviceLevels"]["p95LatencyMs"] == 8000


def test_models_and_nested_collections_are_immutable():
    manifest = load_manifest(valid_document())

    with pytest.raises(ValidationError, match="frozen"):
        manifest.kind = "Other"
    with pytest.raises(TypeError):
        manifest.metadata.tags["team"] = "attacker"
    with pytest.raises(TypeError):
        manifest.spec.environments["prod"] = manifest.spec.environments["dev_west"]
    with pytest.raises(TypeError):
        manifest.spec.evaluation.thresholds["groundedness"] = 0.0
    assert isinstance(manifest.spec.resources.ai_search_indexes, tuple)


def test_candidate_lifecycle_matches_the_sdk_vocabulary():
    document = valid_document()
    document["metadata"]["tags"]["lifecycle"] = "candidate"

    with pytest.warns(DeprecationWarning, match="historical"):
        manifest = load_manifest(document)

    assert manifest.metadata.tags["lifecycle"] == "candidate"


def test_data_classification_tag_uses_the_sdk_vocabulary():
    document = valid_document()
    document["metadata"]["tags"]["data_classification"] = "customer_data"

    with pytest.raises(ValidationError, match="data_classification tag must be"):
        load_manifest(document)


def test_canonical_json_and_hash_are_stable_for_semantically_equivalent_input():
    first = valid_document()
    second = copy.deepcopy(first)
    second["metadata"]["id"] = "tfm_analyst"
    second["metadata"]["businessDomain"] = "investments_technology"
    second["metadata"]["riskTier"] = "medium"
    second["metadata"]["tags"] = {
        "application_id": "TFM Analyst",
        "cost_center": "CC-1234",
        "domain": "Investments Technology",
        "team": "investment-ai",
    }
    second["spec"]["authorization"]["mode"] = "user"
    second["spec"]["environments"] = {
        "dev_west": {
            **first["spec"]["environments"]["Dev-West"],
            "tags": {"environment": "Dev West"},
        }
    }
    second["spec"]["resources"]["evaluationJobKey"] = "release_gate"
    second["spec"]["evaluation"]["profile"] = "grounded_agent_v1"
    second["spec"]["evaluation"]["thresholds"] = {
        "token_count": 1000.0,
        "p95_latency_ms": 8000.0,
        "groundedness": 0.85,
        "safety_pass_rate": 1.0,
    }
    second["spec"]["readiness"]["profile"] = "medium_risk_production_v1"

    first_envelope = build_manifest_envelope(first)
    second_envelope = build_manifest_envelope(second)

    assert first_envelope.canonical_json == second_envelope.canonical_json
    assert first_envelope.manifest_hash == second_envelope.manifest_hash
    assert (
        first_envelope.manifest_hash
        == hashlib.sha256(first_envelope.canonical_json.encode("utf-8")).hexdigest()
    )
    assert ": " not in first_envelope.canonical_json
    assert ", " not in first_envelope.canonical_json
    assert json.loads(first_envelope.canonical_json)["metadata"]["id"] == "tfm_analyst"


def test_canonical_hash_changes_when_policy_changes():
    original = valid_document()
    changed = copy.deepcopy(original)
    changed["spec"]["evaluation"]["thresholds"]["groundedness"] = 0.9

    assert (
        build_manifest_envelope(original).manifest_hash
        != build_manifest_envelope(changed).manifest_hash
    )


@pytest.mark.parametrize(
    ("path", "key"),
    [
        (("metadata", "tags"), "client_secret"),
        (("spec", "environments", "Dev-West", "tags"), "authorizationHeader"),
        (("spec", "evaluation", "thresholds"), "vendorApiKey"),
        (("spec", "resources"), "accessToken"),
    ],
)
def test_secret_like_keys_are_rejected_recursively_before_extra_handling(path, key):
    document = valid_document()
    target = document
    for component in path:
        target = target[component]
    target[key] = "must-not-enter-the-manifest"

    with pytest.raises(ValidationError) as error:
        load_manifest(document)

    message = str(error.value)
    assert key in message
    assert "secret-like" in message


def test_usage_metric_with_token_in_its_name_is_not_misclassified_as_a_secret():
    manifest = load_manifest(valid_document())
    assert manifest.spec.evaluation.thresholds["token_count"] == 1000.0


def test_unknown_fields_and_non_strict_scalar_types_are_rejected():
    unknown = valid_document()
    unknown["metadata"]["permissions"] = ["admin"]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_manifest(unknown)

    coerced = valid_document()
    coerced["spec"]["evaluation"]["minimumCases"] = "30"
    with pytest.raises(ValidationError):
        load_manifest(coerced)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("apiVersion", "ai-platform/v2"),
        ("kind", "Application"),
    ],
)
def test_version_and_kind_are_closed_literals(field, value):
    document = valid_document()
    document[field] = value
    with pytest.raises(ValidationError):
        load_manifest(document)


@pytest.mark.parametrize(
    ("tag", "location"),
    [
        ("Team", "metadata"),
        ("Domain", "metadata"),
        ("cost-center", "metadata"),
        ("application-id", "metadata"),
        ("Environment", "environment"),
    ],
)
def test_each_controlled_tag_is_required_for_every_environment(tag, location):
    document = valid_document()
    if location == "metadata":
        del document["metadata"]["tags"][tag]
    else:
        del document["spec"]["environments"]["Dev-West"]["tags"][tag]

    with pytest.raises(ValidationError, match="missing controlled tags"):
        load_manifest(document)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("metadata", "tags", "application-id"),
            "another-app",
            "application_id tag",
        ),
        (
            ("metadata", "tags", "Domain"),
            "another-domain",
            "domain tag",
        ),
        (
            ("metadata", "tags", "cost-center"),
            "CC-9999",
            "cost_center tag",
        ),
        (
            ("spec", "environments", "Dev-West", "tags", "Environment"),
            "prod",
            "environment tag",
        ),
    ],
)
def test_controlled_tag_values_must_match_manifest_context(path, value, message):
    document = valid_document()
    target = document
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ValidationError, match=message):
        load_manifest(document)


def test_normalization_collisions_are_rejected_in_tags_environments_and_thresholds():
    tag_collision = valid_document()
    tag_collision["metadata"]["tags"]["cost_center"] = "CC-1234"
    with pytest.raises(ValidationError, match="both normalize"):
        load_manifest(tag_collision)

    environment_collision = valid_document()
    environment_collision["spec"]["environments"]["dev_west"] = copy.deepcopy(
        environment_collision["spec"]["environments"]["Dev-West"]
    )
    with pytest.raises(ValidationError, match="both normalize"):
        load_manifest(environment_collision)

    threshold_collision = valid_document()
    threshold_collision["spec"]["evaluation"]["thresholds"]["grounded-ness"] = 0.85
    threshold_collision["spec"]["evaluation"]["thresholds"]["grounded_ness"] = 0.85
    with pytest.raises(ValidationError, match="both normalize"):
        load_manifest(threshold_collision)


@pytest.mark.parametrize("application_id", ["123-app", "---", "a" * 129])
def test_application_id_must_normalize_to_a_bounded_snake_identifier(application_id):
    document = valid_document()
    document["metadata"]["id"] = application_id
    with pytest.raises(ValidationError, match="snake-case identifier"):
        load_manifest(document)


def test_resource_bindings_require_one_evaluation_reference_and_clear_promotion():
    missing = valid_document()
    del missing["spec"]["resources"]["evaluationJobKey"]
    with pytest.raises(ValidationError, match="exactly one"):
        load_manifest(missing)

    duplicate = valid_document()
    duplicate["spec"]["resources"]["evaluationJobId"] = "111"
    with pytest.raises(ValidationError, match="exactly one"):
        load_manifest(duplicate)

    promotion = valid_document()
    promotion["spec"]["resources"].update(
        {"promotionJobId": "222", "promotionJobKey": "promote"}
    )
    with pytest.raises(ValidationError, match="not both"):
        load_manifest(promotion)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluationJobId", "bad id"),
        ("sqlWarehouseId", "bad id"),
        ("aiSearchIndexes", ["valid.name", "bad name"]),
        ("unityCatalogFunctions", ["same.name", "same.name"]),
    ],
)
def test_resource_ids_and_reference_lists_are_validated(field, value):
    document = valid_document()
    resources = document["spec"]["resources"]
    if field == "evaluationJobId":
        del resources["evaluationJobKey"]
    resources[field] = value
    with pytest.raises(ValidationError):
        load_manifest(document)


@pytest.mark.parametrize("field", ["evaluationJobId", "promotionJobId"])
@pytest.mark.parametrize("value", ["0", "-1", "job-123", "9223372036854775808"])
def test_job_resource_ids_are_positive_numeric_databricks_ids(field, value):
    document = valid_document()
    resources = document["spec"]["resources"]
    resources.pop("evaluationJobKey", None)
    resources["evaluationJobId"] = "111"
    resources[field] = value

    with pytest.raises(ValidationError):
        load_manifest(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", "only.two"),
        ("minimumCases", 0),
        ("minimumCases", "30"),
        ("maximumAgeHours", 0),
        ("thresholds", {}),
        ("thresholds", {"groundedness": 1.1}),
        ("thresholds", {"p95LatencyMs": 0}),
        ("thresholds", {"safety_pass_rate": float("nan")}),
        ("thresholds", {"safety_pass_rate": -0.1}),
        ("thresholds", {"safety_pass_rate": True}),
    ],
)
def test_evaluation_policy_rejects_invalid_dataset_counts_ages_and_thresholds(
    field, value
):
    document = valid_document()
    document["spec"]["evaluation"][field] = value
    with pytest.raises(ValidationError):
        load_manifest(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximumErrorRate", 1.1),
        ("maximumErrorRate", "0.02"),
        ("p95LatencyMs", 0),
        ("p95LatencyMs", "8000"),
    ],
)
def test_service_levels_are_strict_and_bounded(field, value):
    document = valid_document()
    document["spec"]["serviceLevels"][field] = value
    with pytest.raises(ValidationError):
        load_manifest(document)


def test_readiness_profile_must_normalize_to_a_real_identifier():
    document = valid_document()
    document["spec"]["readiness"]["profile"] = "---"
    with pytest.raises(ValidationError, match="snake-case identifier"):
        load_manifest(document)


@pytest.mark.parametrize("value", ["", "---", 123])
def test_budget_policy_must_normalize_to_a_real_identifier(value):
    document = valid_document()
    document["spec"]["costControls"]["budgetPolicy"] = value
    with pytest.raises(ValidationError, match="budgetPolicy"):
        load_manifest(document)


def test_v1_manifest_without_cost_controls_preserves_legacy_canonical_form():
    legacy_document = valid_document()
    del legacy_document["spec"]["costControls"]

    manifest = load_manifest(legacy_document)
    canonical = canonical_manifest_json(manifest)

    assert manifest.spec.cost_controls is None
    assert "costControls" not in canonical
    # Regression value from the pre-costControls ai-platform/v1 contract.
    assert build_manifest_envelope(manifest).manifest_hash == (
        "653eb87bccf71d25341f2221b760db28c1b1b1cdf4c9d855b1f51614ee5fa102"
    )


def test_budget_policy_is_required_when_cost_controls_are_declared():
    missing_policy = valid_document()
    del missing_policy["spec"]["costControls"]["budgetPolicy"]
    with pytest.raises(ValidationError):
        load_manifest(missing_policy)


@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "http://github.example.com/ai/app",
        "https://user:password@github.example.com/ai/app",
        "https://github.example.com/ai/app",
        "https://github.com/replace-with-owner/ai-app",
    ],
)
def test_repository_url_must_identify_a_real_https_repository(url):
    document = valid_document()
    document["spec"]["repository"]["url"] = url
    with pytest.raises(ValidationError):
        load_manifest(document)


def test_validation_is_the_canonical_lifecycle_tag():
    document = valid_document()
    document["metadata"]["tags"]["lifecycle"] = "validation"

    manifest = load_manifest(document)

    assert manifest.metadata.tags["lifecycle"] == "validation"

    document["metadata"]["tags"]["lifecycle"] = "preproduction"
    with pytest.raises(ValidationError, match="lifecycle tag must be"):
        load_manifest(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workspaceId", "workspace-123"),
        ("databricksAppName", "Bad_App"),
        ("mlflowExperimentId", "bad id"),
        ("aiGatewayService", "bad service"),
    ],
)
def test_environment_resource_references_are_validated_without_rewriting(field, value):
    document = valid_document()
    document["spec"]["environments"]["Dev-West"][field] = value
    with pytest.raises(ValidationError):
        load_manifest(document)


def test_envelope_rejects_tampered_canonical_json_or_hash():
    envelope = build_manifest_envelope(valid_document())
    payload = envelope.model_dump(mode="json", by_alias=True)

    wrong_hash = copy.deepcopy(payload)
    wrong_hash["manifestHash"] = "0" * 64
    with pytest.raises(ValidationError, match="does not match"):
        ManifestEnvelope.model_validate(wrong_hash)

    wrong_json = copy.deepcopy(payload)
    wrong_json["canonicalJson"] += " "
    with pytest.raises(ValidationError, match="canonicalJson"):
        ManifestEnvelope.model_validate(wrong_json)


def test_canonical_function_accepts_an_already_validated_manifest():
    manifest = AIApplicationManifest.model_validate(valid_document())
    assert (
        canonical_manifest_json(manifest)
        == build_manifest_envelope(manifest).canonical_json
    )


def test_json_schema_uses_external_aliases_and_forbids_unknown_fields():
    schema = manifest_json_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "urn:aai:schema:ai-platform:v1"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["apiVersion"]["const"] == "ai-platform/v1"
    assert schema["properties"]["kind"]["const"] == "AIApplication"
    assert "supportGroup" in schema["$defs"]["ManifestMetadata"]["properties"]
    resource_properties = schema["$defs"]["ResourceBindings"]["properties"]
    assert "evaluationJobKey" in resource_properties
    assert "promotionJobKey" in resource_properties
    cost_properties = schema["$defs"]["CostControlsSpec"]["properties"]
    assert "budgetPolicy" in cost_properties
    assert "costControls" in schema["$defs"]["ApplicationSpec"]["properties"]
    assert "costControls" not in schema["$defs"]["ApplicationSpec"]["required"]
    assert "serviceLevels" in schema["$defs"]["ApplicationSpec"]["properties"]
    json.dumps(schema, allow_nan=False)


def test_shared_template_uses_only_cross_template_variables_and_disables_promotion():
    text = SHARED_TEMPLATE.read_text(encoding="utf-8")
    assert set(TEMPLATE_VARIABLE.findall(text)) == COMMON_TEMPLATE_VARIABLES
    assert text.count('{{ template "project_name_underscored" . }}') == 1
    assert "evaluationJobKey: release_gate" in text
    assert "promotionJobId:" not in text
    assert "promotionJobKey:" not in text
    assert "budgetPolicy: platform_standard_v1" in text


def test_shared_template_renders_to_a_valid_manifest_and_is_synced_everywhere():
    values = {
        "application_name": "example-assistant",
        "catalog": "main",
        "cost_center": "CC-1234",
        "owner_group": "group:data-platform-owners",
        "project_name": "example-ai",
        "repository_url": "https://github.com/aai-test/example-ai",
        "schema": "example_ai",
        "team": "data-platform",
    }
    canonical = SHARED_TEMPLATE.read_text(encoding="utf-8")
    rendered = canonical
    for name, value in values.items():
        rendered = rendered.replace(f"{{{{.{name}}}}}", value)
    rendered = rendered.replace(
        '{{ template "project_name_underscored" . }}',
        values["project_name"].replace("-", "_"),
    )
    assert "{{." not in rendered

    manifest = load_manifest(yaml.safe_load(rendered))
    assert manifest.metadata.id == "example_assistant"
    assert manifest.spec.resources.evaluation_job_key == "release_gate"
    assert manifest.spec.resources.promotion_job_id is None
    assert manifest.spec.resources.promotion_job_key is None

    template_roots = sorted(
        path
        for path in (ROOT / "templates").iterdir()
        if (path / "databricks_template_schema.json").is_file()
    )
    assert len(template_roots) == 6
    shared_manifest = json.loads(SHARED_MANIFEST.read_text(encoding="utf-8"))
    opted_out = set(shared_manifest["opt_out"].get("ai-app.yaml.tmpl", []))
    for template in template_roots:
        copied = template / "template" / "ai-app.yaml.tmpl"
        if template.name not in opted_out:
            assert copied.read_text(encoding="utf-8") == canonical
