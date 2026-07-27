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
    # Three workspace rows plus the workspace-independent template_source row.
    assert response.text.count("<tr") == 4


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


def test_workspace_probes_do_not_run_on_the_event_loop(config):
    """A blocking SDK call on an async route would stall a single-worker server.

    app.yaml starts one uvicorn worker, so a checks route running its blocking
    Databricks calls on the event loop would take health, navigation and generation
    down for the whole SDK timeout. Asserting the route is a plain `def` — which
    Starlette runs in its worker threadpool — is the property that prevents that.
    """
    import asyncio
    import inspect

    app = create_app(config, probe=WorkspaceProbe(_FakeWorkspace()))
    route = next(r for r in app.routes if getattr(r, "path", "") == "/api/checks/run")
    assert not asyncio.iscoroutinefunction(
        route.endpoint
    ), "the checks route must be a sync def so blocking SDK calls run in a threadpool"
    assert not inspect.isasyncgenfunction(route.endpoint)


def test_content_is_parsed_once_rather_than_per_request(config):
    """Re-reading the content YAML per request is blocking disk I/O on the loop."""
    app = create_app(config, probe=WorkspaceProbe(_FakeWorkspace()))
    client = ASGIClient(app)
    before = app.state.tracks
    client.get("/")
    client.get("/api/content")
    assert app.state.tracks is before, "tracks must not be re-rendered per request"


def test_local_runs_never_probe_the_workspace_as_the_developer(config, monkeypatch):
    """Outside the hosted app, ambient auth is the developer's own identity.

    Probing it would report personal permissions under a heading that claims they are
    platform state — the conflation this module exists to prevent. A local run must skip
    instead, even when databricks-sdk is installed and `az login` has been run.
    """
    called = False

    class _ShouldNotBeUsed:
        def __init__(self, *a, **k):
            nonlocal called
            called = True

    monkeypatch.setattr("aai_console.checks.WorkspaceProbe", _ShouldNotBeUsed)
    assert config.hosted is False
    checks = run_checks(config)  # no probe injected: the ambient path

    assert not called, "a local run must not construct a workspace client"
    assert {c.status for c in checks} == {"skip"}
    workspace_rows = [c for c in checks if c.id != "template_source"]
    assert workspace_rows, "the workspace rows must still be reported"
    assert all("hosted app" in c.detail for c in workspace_rows)


def test_hosted_runs_do_probe(monkeypatch):
    """The gate is on `hosted`, not a blanket disable — the app must still report."""
    hosted = ConsoleConfig(
        identifiers={key: IDENTIFIERS[key] for key in IDENTIFIER_KEYS},
        hosted=True,
        app_name="aai-platform-console-dev",
        template_repo=IDENTIFIERS["template_repo"],
    )
    monkeypatch.setattr(
        "aai_console.checks.WorkspaceProbe", lambda: WorkspaceProbe(_FakeWorkspace())
    )
    checks = run_checks(hosted)
    assert [c.status for c in checks] == ["pass", "pass", "pass", "pass"]


def _hosted(template_repo=None):
    return ConsoleConfig(
        identifiers={key: IDENTIFIERS[key] for key in IDENTIFIER_KEYS},
        hosted=True,
        app_name="aai-platform-console-dev",
        template_repo=template_repo,
    )


def test_hosted_console_without_a_template_repo_refuses_to_generate():
    """A hosted viewer has no checkout, so the `./templates/...` fallback would be a
    command that cannot work. A clone whose bundle never wired `template_repo` must
    fail loudly here rather than hand every developer a broken `bundle init`."""
    from aai_console.generate import GenerateError, GenerateRequest, bundle_init

    with pytest.raises(GenerateError) as error:
        bundle_init(GenerateRequest(template="rag-app"), _hosted())
    assert "AAI_CONSOLE_TEMPLATE_REPO" in str(error.value)


def test_hosted_console_reports_a_missing_template_repo_as_a_failed_check(monkeypatch):
    """The generate-time refusal is a last line of defence; the panel must surface it
    before a developer picks a template."""
    monkeypatch.setattr(
        "aai_console.checks.WorkspaceProbe", lambda: WorkspaceProbe(_FakeWorkspace())
    )
    row = next(c for c in run_checks(_hosted()) if c.id == "template_source")
    assert row.status == "fail"
    assert "AAI_CONSOLE_TEMPLATE_REPO" in row.detail

    configured = next(
        c
        for c in run_checks(_hosted(IDENTIFIERS["template_repo"]))
        if c.id == "template_source"
    )
    assert configured.status == "pass"


def test_local_console_still_generates_from_the_checkout(config):
    """`make app-run` has no bundle to supply the variable and does have a checkout,
    so the relative form stays correct there."""
    from aai_console.generate import GenerateRequest, bundle_init

    codes = [block.code for block in bundle_init(GenerateRequest("rag-app"), config)]
    assert any("./templates/rag-app" in code for code in codes)
