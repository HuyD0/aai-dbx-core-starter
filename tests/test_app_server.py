"""The console must start and answer with no cloud identity at all.

That is the credential-free premise the whole repository rests on: pull-request CI has
no Azure or Databricks credential, so anything that only works when authenticated is
untestable there.
"""

import json
from pathlib import Path

import pytest
from asgi_client import ASGIClient

from aai_console.checks import PLATFORM_STATE_HEADING, WorkspaceProbe, run_checks
from aai_console.config import IDENTIFIER_KEYS, ConsoleConfig, load_config
from aai_console.server import create_app

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIERS = json.loads((ROOT / "platform-identifiers.json").read_text())

CREDENTIAL_VARS = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_CLIENT_ID",
    "DATABRICKS_CLIENT_SECRET",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
)


class _FakeWorkspace:
    """Stands in for WorkspaceClient. The SDK ships no fake, and the dev environment
    deliberately excludes databricks-sdk entirely."""

    class current_user:
        @staticmethod
        def me():
            return type("User", (), {"user_name": "console-sp@example"})()

    class cluster_policies:
        @staticmethod
        def get(policy_id):
            return type("Policy", (), {"name": "Constrained Jobs"})()

    class files:
        @staticmethod
        def list_directory_contents(path):
            return [object(), object()]


@pytest.fixture
def config():
    return ConsoleConfig(
        identifiers={key: IDENTIFIERS[key] for key in IDENTIFIER_KEYS},
        hosted=False,
        app_name=None,
    )


@pytest.fixture
def client(config, monkeypatch):
    for name in CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)
    return ASGIClient(create_app(config, probe=WorkspaceProbe(_FakeWorkspace())))


def test_healthz_and_session_respond_without_any_cloud_identity(client):
    assert client.get("/healthz").json()["status"] == "ok"
    session = client.get("/api/session").json()
    assert session["hosted"] is False
    assert session["capability"] == "guide-and-generate"


def test_index_and_every_track_page_render(client):
    assert client.get("/").status_code == 200
    for track in client.get("/api/content").json()["tracks"]:
        response = client.get(f"/track/{track['id']}")
        assert response.status_code == 200, track["id"]
        assert track["title"] in response.text


def test_unknown_track_is_a_404_not_a_crash(client):
    assert client.get("/track/does-not-exist").status_code == 404


def test_checks_render_as_platform_state(client):
    response = client.post("/api/checks/run")
    assert response.status_code == 200
    assert PLATFORM_STATE_HEADING in response.text
    # Table rows must survive fragment parsing; this is why the client uses
    # <template>.content rather than DOMParser.
    assert response.text.count("<tr") == 3


def test_generate_emits_the_chosen_template_and_the_configured_host(client):
    response = client.post("/api/generate", json={"template": "rag-app"})
    assert response.status_code == 200
    assert "--template-dir templates/rag-app" in response.text or (
        "./templates/rag-app" in response.text
    )
    assert IDENTIFIERS["databricks_host"] in response.text


def test_generate_rejects_an_unknown_template(client):
    response = client.post("/api/generate", json={"template": "../../etc"})
    assert response.status_code == 400


def test_generate_rejects_an_unsafe_project_name(client):
    response = client.post(
        "/api/generate", json={"template": "rag-app", "project_name": "a; rm -rf /"}
    )
    assert response.status_code == 400


def test_palette_search_finds_a_step(client):
    results = client.get("/api/palette", params={"q": "trace"}).json()["results"]
    assert results and all("title" in hit for hit in results)


def test_checks_skip_cleanly_when_the_workspace_is_unreachable(config):
    """No SDK and no credentials is the normal state in credential-free CI."""
    checks = run_checks(config, WorkspaceProbe(None))
    assert {check.status for check in checks} == {"skip"}
    assert all(check.identity == "app_sp" for check in checks)


def test_config_falls_back_to_the_repository_fixture_for_local_runs():
    loaded = load_config({}, start=ROOT / "src" / "platform_app")
    assert loaded.hosted is False
    assert loaded.identifiers["databricks_host"] == IDENTIFIERS["databricks_host"]


def test_hosted_config_never_reads_the_repository_fixture():
    """A hosted app has no checkout; a missing value there must stay a loud failure."""
    loaded = load_config(
        {"DATABRICKS_APP_NAME": "aai-platform-console-dev"},
        start=ROOT / "src" / "platform_app",
    )
    assert loaded.hosted is True
    assert "job_compute_policy_id" not in loaded.identifiers
