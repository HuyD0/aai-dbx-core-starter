"""Nothing credential-shaped may reach a response body or a log line.

The Databricks Apps runtime injects a live OAuth secret (`DATABRICKS_CLIENT_SECRET`)
into the app process environment, and app logs are readable by anyone with CAN MANAGE.
AGENTS.md rule 9 requires that secret material never leak through str, repr, logs,
exceptions, traces, tags or parameters; a rendered body and a log line are both in
scope.

The primary guard is an explicit key allowlist rather than a sentinel substring scan.
`dataclasses.asdict()` recurses into nested dataclasses and dicts — so one careless
`asdict(settings)` would serialise `PlatformSettings.raw`, and `repr=False` does not
prevent it. A substring scan passes whenever the leaked structure happens not to contain
the sentinel; a key allowlist fails structurally, whatever the content.
"""

import json
import logging
from pathlib import Path

import pytest
from asgi_client import ASGIClient

from aai_console.config import IDENTIFIER_KEYS, ConsoleConfig
from aai_console.server import create_app

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIERS = json.loads((ROOT / "platform-identifiers.json").read_text())

SENTINEL = "s3nt1nel-never-render-this"

# Exact top-level keys each JSON endpoint may return. Adding a key here must be a
# deliberate act, reviewed for what it exposes.
ALLOWED_KEYS = {
    "/healthz": {"status", "version"},
    "/api/session": {"hosted", "app_name", "version", "capability"},
    "/api/content": {"tracks"},
    "/api/palette": {"results"},
}


@pytest.fixture
def client(monkeypatch):
    # Every credential-shaped variable the runtime could inject, holding a sentinel.
    for name in (
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_TOKEN",
        "AZURE_CLIENT_SECRET",
    ):
        monkeypatch.setenv(name, SENTINEL)
    config = ConsoleConfig(
        identifiers={key: IDENTIFIERS[key] for key in IDENTIFIER_KEYS},
        hosted=False,
        app_name=None,
    )
    return ASGIClient(create_app(config))


@pytest.mark.parametrize("path", sorted(ALLOWED_KEYS))
def test_json_endpoints_return_only_allowlisted_keys(client, path):
    body = client.get(path).json()
    assert set(body) == ALLOWED_KEYS[path], f"{path} response shape changed"


def test_no_credential_material_reaches_any_response_body(client):
    paths = ["/", "/healthz", "/api/session", "/api/content", "/api/palette?q=a"]
    for path in paths:
        assert SENTINEL not in client.get(path).text, path
    for path in ["/api/checks/run"]:
        assert SENTINEL not in client.post(path).text, path
    generated = client.post("/api/generate", json={"template": "rag-app"})
    assert SENTINEL not in generated.text


def test_no_credential_material_reaches_a_log_line(client, caplog):
    with caplog.at_level(logging.DEBUG):
        client.get("/")
        client.post("/api/checks/run")
        client.get("/api/session")
    assert SENTINEL not in caplog.text


def test_unhandled_errors_return_a_generic_body_and_log_no_detail(client, caplog):
    """A traceback reaches the app's Logs tab, which shares the secret's environment."""
    app = client.app

    @app.get("/api/boom")
    async def boom():
        raise RuntimeError(f"leaky message containing {SENTINEL}")

    with caplog.at_level(logging.DEBUG):
        response = client.get("/api/boom")

    assert response.status_code == 500
    assert response.json() == {"error": "internal error"}
    assert SENTINEL not in response.text
    assert SENTINEL not in caplog.text


def test_safe_detail_scrubs_credential_values_out_of_provider_errors(monkeypatch):
    """Regression: the Databricks SDK's auth error interpolates client_id verbatim.

    It redacts client_secret but not client_id, so a raw provider message rendered into
    the page leaked an environment value. Scrubbing at our own boundary stays correct
    no matter what a future SDK version chooses to include.
    """
    from aai_console.checks import _safe_detail

    monkeypatch.setenv("DATABRICKS_CLIENT_ID", SENTINEL)
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", SENTINEL + "-secret")
    detail = _safe_detail(
        RuntimeError(
            f"default auth: client_id={SENTINEL}, client_secret={SENTINEL}-secret"
        )
    )
    assert SENTINEL not in detail
    assert "***" in detail


def test_short_environment_values_are_not_used_for_scrubbing(monkeypatch):
    """A one- or two-character value would redact unrelated prose into nonsense."""
    from aai_console.checks import _safe_detail

    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "ab")
    assert _safe_detail(RuntimeError("a cabbage problem")) == "a cabbage problem"


def test_check_details_are_length_capped(client):
    """Provider errors can be enormous; an unbounded detail is a denial-of-readability
    problem and a good way to smuggle unexpected content into the page."""
    from aai_console.checks import _safe_detail

    assert len(_safe_detail(RuntimeError("x" * 5000))) <= 200


def test_inline_code_renderer_escapes_before_marking_safe():
    """Content is trusted, but the renderer stays injection-proof by construction."""
    from aai_console.content import inline_code

    rendered = str(inline_code("<script>alert(1)</script> and `ok`"))
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<code>ok</code>" in rendered


def test_nothing_propagates_out_of_the_asgi_app(client):
    """The property that keeps exception messages out of the server's log.

    Starlette's ServerErrorMiddleware re-raises after sending the error response so the
    server can log it — and uvicorn then logs the message, not just the type. The app is
    wrapped so nothing escapes; tests/asgi_client.py does not swallow, so if the wrapper
    were removed this would fail rather than silently pass.
    """
    app = client.app

    @app.get("/api/leaky")
    async def leaky():
        raise RuntimeError(f"provider payload {SENTINEL}")

    response = client.get("/api/leaky")
    assert response.status_code == 500
    assert response.json() == {"error": "internal error"}
    assert SENTINEL not in response.text


def test_uvicorn_does_not_log_the_exception_message(tmp_path):
    """End-to-end against a real uvicorn, because the unit tests above cannot see it.

    The leak this guards lived entirely in the server's log: the client response was
    always clean. Only running the actual server and reading what it printed can prove
    the fix, which is why this spawns one rather than asserting on a mock.
    """
    uvicorn = pytest.importorskip("uvicorn")  # noqa: F841
    import socket
    import subprocess
    import sys
    import time
    import urllib.error
    import urllib.request

    module = tmp_path / "leak_probe_app.py"
    module.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'src' / 'platform_app')!r})\n"
        "from aai_console.server import create_app\n"
        "from aai_console.config import ConsoleConfig\n"
        f"ids = {({k: IDENTIFIERS[k] for k in IDENTIFIER_KEYS})!r}\n"
        "app = create_app(\n"
        "    ConsoleConfig(identifiers=ids, hosted=False, app_name=None)\n"
        ")\n"
        "@app.get('/leak')\n"
        "async def leak():\n"
        f"    raise RuntimeError('creds {SENTINEL}')\n",
        encoding="utf-8",
    )

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "leak_probe_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-access-log",
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(80):
            if process.poll() is not None:
                pytest.skip("uvicorn could not start in this environment")
            try:
                urllib.request.urlopen(f"{base}/healthz", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        else:
            pytest.skip("uvicorn did not become ready in time")

        try:
            urllib.request.urlopen(f"{base}/leak", timeout=10)
        except urllib.error.HTTPError as error:
            assert error.code == 500
            assert SENTINEL not in error.read().decode()
        time.sleep(0.5)
    finally:
        process.terminate()
        output = process.communicate(timeout=30)[0] or ""

    assert (
        SENTINEL not in output
    ), f"exception message reached the server log:\n{output}"
    assert "Traceback" not in output, f"traceback reached the server log:\n{output}"
    assert "RuntimeError" in output, "the error should still be recorded, by type"
