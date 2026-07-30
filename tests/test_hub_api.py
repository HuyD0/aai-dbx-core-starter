"""HTTP and server-rendered UI contracts for the AI Platform Hub."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pytest
from asgi_client import ASGIClient

from aai_console.config import ConsoleConfig, HubStateMode
from aai_console.hub.identity import (
    RoleAssignment,
    StaticRoleResolver,
)
from aai_console.hub.jobs import RecordingJobRunner
from aai_console.hub.models import (
    ApplicationPrincipalRecord,
    PrincipalType,
    Role,
)
from aai_console.hub.repository import InMemoryHubRepository
from aai_console.server import create_app

CI_PRINCIPAL = "github-actions@example.com"
OWNER_PRINCIPAL = "owner@example.com"
FLEET_VIEWER = "fleet-viewer@example.com"
STRANGER = "stranger@example.com"
FORWARDED_USER = "X-Forwarded-User"

IDENTIFIERS = {
    "databricks_host": "https://workspace.example.invalid",
    "sdk_artifact_volume": "/Volumes/platform/artifacts/aai_core",
    "job_compute_policy_id": "constrained-jobs-policy",
}


def _config(
    *,
    hosted: bool,
    registration_principals: frozenset[str] = frozenset(),
) -> ConsoleConfig:
    return ConsoleConfig(
        identifiers=IDENTIFIERS,
        hosted=hosted,
        app_name="aai-platform-hub-test" if hosted else None,
        template_repo="https://github.com/aai-test/platform-templates",
        hub_state_mode=(HubStateMode.UNAVAILABLE if hosted else HubStateMode.MEMORY),
        hub_registration_principals=registration_principals,
        hub_local_actor="local-developer",
    )


def _client(
    *,
    hosted: bool = True,
    registration_principals: frozenset[str] = frozenset({CI_PRINCIPAL}),
    role_resolver=None,
    job_runner=None,
) -> ASGIClient:
    return ASGIClient(
        create_app(
            _config(
                hosted=hosted,
                registration_principals=registration_principals,
            ),
            hub_repository=InMemoryHubRepository(),
            role_resolver=role_resolver,
            job_runner=job_runner,
        )
    )


def _headers(principal: str) -> dict[str, str]:
    return {FORWARDED_USER: principal}


def _manifest(
    application_id: str = "claims_assistant",
    *,
    name: str | None = None,
    owner: str = OWNER_PRINCIPAL,
    team: str = "claims",
    domain: str = "insurance",
    evaluation_job_key: str = "release_gate",
) -> dict:
    return {
        "apiVersion": "ai-platform/v1",
        "kind": "AIApplication",
        "metadata": {
            "id": application_id,
            "name": name or application_id.replace("_", " ").title(),
            "description": f"Governed {application_id} application.",
            "owner": owner,
            "supportGroup": f"group:{application_id}-support",
            "businessDomain": domain,
            "costCenter": "CC-100",
            "riskTier": "medium",
            "tags": {
                "application_id": application_id,
                "team": team,
                "domain": domain,
                "cost_center": "CC-100",
                "data_classification": "internal",
                "lifecycle": "experimental",
            },
        },
        "spec": {
            "repository": {"url": f"https://github.com/aai-test/{application_id}"},
            "authorization": {"mode": "application"},
            "environments": {
                "dev": {
                    "workspaceId": "123",
                    "databricksAppName": f"{application_id.replace('_', '-')}-dev",
                    "mlflowExperimentId": "456",
                    "aiGatewayService": "ai_platform.models.enterprise_chat",
                    "tags": {"environment": "dev"},
                }
            },
            "resources": {"evaluationJobKey": evaluation_job_key},
            "evaluation": {
                "profile": "grounded_agent_v1",
                "dataset": f"ai_platform.evaluations.{application_id}",
                "minimumCases": 30,
                "maximumAgeHours": 168,
                "thresholds": {
                    "groundedness": 0.85,
                    "safety_pass_rate": 1.0,
                },
            },
            "readiness": {"profile": "medium_risk_production_v1"},
            "serviceLevels": {
                "maximumErrorRate": 0.02,
                "p95LatencyMs": 8000,
            },
        },
    }


def _registration_payload(
    manifest: dict,
    *,
    git_sha: str = "a" * 40,
    environment: str = "dev",
) -> dict:
    return {
        "manifest": manifest,
        "environment": environment,
        "gitCommitSha": git_sha,
        "deploymentTarget": f"{environment}-bundle",
    }


def _register(
    client: ASGIClient,
    manifest: dict,
    *,
    principal: str = CI_PRINCIPAL,
    git_sha: str = "a" * 40,
    environment: str = "dev",
    authorize: bool = True,
):
    response = client.post(
        "/api/v1/registrations",
        json=_registration_payload(
            manifest,
            git_sha=git_sha,
            environment=environment,
        ),
        headers=_headers(principal),
    )
    if authorize and response.status_code in {200, 201}:
        repository = client.app.state.hub_repository
        application = repository.get_application(
            response.json()["application"]["applicationId"]
        )
        for application_principal in (
            ApplicationPrincipalRecord(
                application_id=application.application_id,
                principal_type=PrincipalType.USER,
                principal_name=application.owner_principal,
                application_role=Role.OWNER,
            ),
            ApplicationPrincipalRecord(
                application_id=application.application_id,
                principal_type=PrincipalType.GROUP,
                principal_name=application.support_group,
                application_role=Role.CONTRIBUTOR,
            ),
        ):
            repository.upsert_application_principal(application_principal)
    return response


def test_local_portfolio_renders_an_explicit_empty_preview_state() -> None:
    response = _client(hosted=False).get("/")

    assert response.status_code == 200
    assert "Application portfolio" in response.text
    assert "Local preview registry" in response.text
    assert "No registered applications match this view" in response.text
    assert "0 results" in response.text


def test_hosted_routes_require_the_forwarded_databricks_identity() -> None:
    client = _client()
    response = client.get("/")
    checks = client.post("/api/checks/run")

    assert response.status_code == 401
    assert checks.status_code == 401
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["title"] == "Authentication required"
    assert "local-developer" not in response.text


def test_registration_is_allowlisted_and_idempotent_over_http() -> None:
    client = _client()
    manifest = _manifest()

    first = _register(client, manifest)
    replay = _register(client, manifest)
    denied = _register(
        client,
        _manifest("unapproved_app"),
        principal="unapproved-ci@example.com",
    )

    assert first.status_code == 201
    assert first.json()["created"] is True
    assert first.json()["application"]["applicationId"] == "claims_assistant"
    assert replay.status_code == 200
    assert replay.json()["created"] is False
    assert replay.json()["version"]["versionId"] == first.json()["version"]["versionId"]
    assert denied.status_code == 403
    assert denied.json()["title"] == "Forbidden"


def test_list_and_direct_reads_enforce_the_same_fail_closed_visibility() -> None:
    client = _client()
    assert _register(client, _manifest("private_app")).status_code == 201

    stranger_list = client.get(
        "/api/v1/applications",
        headers=_headers(STRANGER),
    )
    hidden = client.get(
        "/api/v1/applications/private_app",
        headers=_headers(STRANGER),
    )
    absent = client.get(
        "/api/v1/applications/does_not_exist",
        headers=_headers(STRANGER),
    )
    owner = client.get(
        "/api/v1/applications/private_app",
        headers=_headers(OWNER_PRINCIPAL),
    )

    assert stranger_list.status_code == 200
    assert stranger_list.json()["items"] == []
    assert hidden.status_code == absent.status_code == 404
    assert hidden.json()["title"] == absent.json()["title"] == "Resource not found"
    assert hidden.json()["detail"] == absent.json()["detail"]
    assert owner.status_code == 200
    assert owner.json()["application"]["applicationId"] == "private_app"


def test_portfolio_and_detail_expose_each_current_environment() -> None:
    client = _client()
    manifest = _manifest()
    manifest["spec"]["environments"]["prod"] = {
        "workspaceId": "123",
        "databricksAppName": "claims-assistant-prod",
        "mlflowExperimentId": "789",
        "aiGatewayService": "ai_platform.models.enterprise_chat",
        "tags": {"environment": "prod"},
    }
    assert _register(client, manifest, git_sha="a" * 40).status_code == 201
    assert (
        _register(
            client,
            manifest,
            git_sha="b" * 40,
            environment="prod",
        ).status_code
        == 201
    )

    portfolio = client.get(
        "/api/v1/applications",
        headers=_headers(OWNER_PRINCIPAL),
    )
    detail = client.get(
        "/api/v1/applications/claims_assistant",
        params={"environment": "dev"},
        headers=_headers(OWNER_PRINCIPAL),
    )

    assert portfolio.status_code == detail.status_code == 200
    assert {
        item["environment"] for item in portfolio.json()["items"][0]["deployments"]
    } == {"dev", "prod"}
    assert {item["environment"] for item in detail.json()["deployments"]} == {
        "dev",
        "prod",
    }
    assert detail.json()["currentVersion"]["environment"] == "dev"


def test_tag_filters_are_or_within_a_key_and_and_across_keys() -> None:
    resolver = StaticRoleResolver(
        {
            FLEET_VIEWER: RoleAssignment(
                platform_roles=(Role.PLATFORM_VIEWER,),
            )
        }
    )
    client = _client(role_resolver=resolver)
    for manifest in (
        _manifest("alpha", team="blue", domain="investments"),
        _manifest("beta", team="green", domain="investments"),
        _manifest("gamma", team="blue", domain="research"),
    ):
        assert _register(client, manifest).status_code == 201

    response = client.get(
        "/api/v1/applications",
        params={"tags": "team=blue|green,domain=investments"},
        headers=_headers(FLEET_VIEWER),
    )

    assert response.status_code == 200
    assert [item["applicationId"] for item in response.json()["items"]] == [
        "alpha",
        "beta",
    ]


@pytest.mark.parametrize(
    ("params", "expected_title"),
    [
        ({"unexpected": "ignored-looking-value"}, "Unsupported filter"),
        ({"sort": "name; DROP TABLE applications"}, "Invalid portfolio filter"),
    ],
)
def test_application_queries_reject_unknown_keys_and_sort_injection(
    params: dict[str, str],
    expected_title: str,
) -> None:
    response = _client().get(
        "/api/v1/applications",
        params=params,
        headers=_headers(STRANGER),
    )

    assert response.status_code == 400
    assert response.json()["title"] == expected_title


def test_request_id_is_preserved_on_the_response() -> None:
    response = _client().get(
        "/api/v1/capabilities",
        headers={"X-Request-Id": "hub-contract-test-123"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "hub-contract-test-123"


def test_credential_shaped_request_ids_and_paths_never_enter_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    credential = "github_pat_this_value_must_never_enter_a_log"
    caplog.set_level(logging.INFO, logger="aai_console")

    response = _client().get(
        f"/missing/{credential}",
        headers={"X-Request-Id": credential},
    )

    assert response.status_code == 404
    assert response.headers["x-request-id"] != credential
    assert credential not in response.text
    assert credential not in caplog.text
    assert '"route":"<unmatched>"' in caplog.text


def test_structured_request_log_records_the_real_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="aai_console")

    response = _client().get("/api/v1/capabilities")

    assert response.status_code == 200
    assert '"status":200' in caplog.text
    assert '"result":"success"' in caplog.text
    assert '"action":"hub_capabilities"' in caplog.text


def test_job_adapter_does_not_claim_workflows_are_ready_without_durable_state() -> None:
    client = ASGIClient(
        create_app(
            _config(hosted=True),
            job_runner=RecordingJobRunner(),
        )
    )

    response = client.get("/api/v1/capabilities")

    assert response.status_code == 200
    assert response.json()["state_store"] == "unavailable"
    assert response.json()["evaluation"] == "gated"
    assert response.json()["promotion"] == "gated"
    assert "durable operational store" in response.json()["evaluation_detail"]


def test_validation_problem_does_not_reflect_secret_like_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_like_value = "github_pat_this_value_must_never_be_reflected"
    payload = _registration_payload(_manifest())
    payload["gitCommitSha"] = secret_like_value
    caplog.set_level(logging.INFO, logger="aai_console")

    response = _client().post(
        "/api/v1/registrations",
        json=payload,
        headers={
            **_headers(CI_PRINCIPAL),
            "X-Forwarded-Access-Token": secret_like_value,
        },
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["title"] == "Request validation failed"
    assert secret_like_value not in response.text
    assert secret_like_value not in caplog.text


def test_openapi_and_manifest_schema_are_published_without_authentication() -> None:
    client = _client()

    openapi = client.get("/api/openapi.json")
    schema = client.get("/api/v1/manifest-schemas/ai-platform-v1")

    assert openapi.status_code == 200
    assert openapi.json()["info"]["title"] == "AAI AI Platform Hub"
    assert "/api/v1/registrations" in openapi.json()["paths"]
    assert (
        openapi.json()["components"]["securitySchemes"]["DatabricksAppIdentity"]["name"]
        == "X-Forwarded-User"
    )
    registration = openapi.json()["paths"]["/api/v1/registrations"]["post"]
    assert registration["security"] == [{"DatabricksAppIdentity": []}]
    assert (
        registration["responses"]["409"]["content"]["application/problem+json"][
            "schema"
        ]["$ref"]
        == "#/components/schemas/Problem"
    )
    assert (
        registration["responses"]["422"]["content"]["application/problem+json"][
            "schema"
        ]["$ref"]
        == "#/components/schemas/Problem"
    )
    assert schema.status_code == 200
    assert schema.json()["title"] == "AIApplicationManifest"
    assert "$defs" in schema.json()


def test_portfolio_rejects_future_incomplete_cost_dates() -> None:
    tomorrow = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()

    response = _client().get(
        "/api/v1/applications",
        params={"end_date": tomorrow},
        headers=_headers(STRANGER),
    )

    assert response.status_code == 400
    assert "last complete UTC day" in response.json()["detail"]


@pytest.mark.parametrize(
    ("path", "capability"),
    [
        ("/api/v1/applications/claims_assistant/traces", "Traces unavailable"),
        ("/api/v1/applications/claims_assistant/costs", "Costs unavailable"),
        ("/api/v1/optimization/summary", "Optimization unavailable"),
    ],
)
def test_unconfigured_observability_is_explicitly_unavailable(
    path: str,
    capability: str,
) -> None:
    resolver = StaticRoleResolver(
        {
            FLEET_VIEWER: RoleAssignment(
                platform_roles=(Role.PLATFORM_VIEWER,),
            )
        }
    )
    client = _client(role_resolver=resolver)
    assert _register(client, _manifest()).status_code == 201

    response = client.get(path, headers=_headers(FLEET_VIEWER))

    assert response.status_code == 503
    assert response.json()["title"] == capability
    assert response.headers["content-type"].startswith("application/problem+json")


def test_non_administrator_cannot_read_the_admin_action_queue() -> None:
    response = _client().get(
        "/api/v1/admin/actions",
        headers=_headers(STRANGER),
    )

    assert response.status_code == 403
    assert response.json()["title"] == "Forbidden"


def test_evaluation_is_gated_until_the_logical_job_key_is_resolved() -> None:
    client = _client()
    assert _register(client, _manifest()).status_code == 201

    response = client.post(
        "/api/v1/applications/claims_assistant/evaluations",
        json={"environment": "dev", "datasetVersion": "2026-07-29"},
        headers=_headers(OWNER_PRINCIPAL),
    )
    evaluations = client.get(
        "/api/v1/applications/claims_assistant/evaluations",
        headers=_headers(OWNER_PRINCIPAL),
    )

    assert response.status_code == 503
    assert response.json()["title"] == "Capability unavailable"
    assert "resolved to an approved job ID" in response.json()["detail"]
    assert evaluations.status_code == 200
    assert evaluations.json()["items"] == []


def test_resolved_evaluation_job_launches_once_and_returns_typed_state() -> None:
    runner = RecordingJobRunner()
    client = _client(job_runner=runner)
    manifest = _manifest()
    manifest["spec"]["resources"] = {"evaluationJobId": "123"}
    assert _register(client, manifest).status_code == 201
    payload = {"environment": "dev", "datasetVersion": "dataset-v1"}

    accepted = client.post(
        "/api/v1/applications/claims_assistant/evaluations",
        json=payload,
        headers=_headers(OWNER_PRINCIPAL),
    )
    duplicate = client.post(
        "/api/v1/applications/claims_assistant/evaluations",
        json=payload,
        headers=_headers(OWNER_PRINCIPAL),
    )

    assert accepted.status_code == 202
    assert accepted.json()["status"] == "QUEUED"
    assert accepted.json()["jobRunId"] == "preview-run-000001"
    assert duplicate.status_code == 409
    assert len(runner.requests) == 1
