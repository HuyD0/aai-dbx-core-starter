"""Governance guards for the console's deployment surface.

The container installs from `src/platform_app/requirements.txt`, which `uv lock --check`
never sees. That is a second dependency channel, and the only thing stopping it drifting
from the versions this repository actually tests is a test.
"""

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "src" / "platform_app"
WORKFLOWS = ROOT / ".github" / "workflows"

REQUIREMENTS = APP_DIR / "requirements.txt"
APP_YAML = APP_DIR / "app.yaml"
RESOURCE = ROOT / "resources" / "optional" / "platform_console.yml"
IDENTIFIERS = json.loads((ROOT / "platform-identifiers.json").read_text())


def _app_resource() -> dict:
    document = yaml.safe_load(RESOURCE.read_text(encoding="utf-8"))
    return document["resources"]["apps"]["platform_console"]


def _pins() -> dict[str, str]:
    pins = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        assert version, f"{line!r} must be == pinned so the container is reproducible"
        pins[name.strip().lower()] = version.strip()
    return pins


def _locked_versions() -> dict[str, str]:
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    return {
        match.group(1).lower(): match.group(2)
        for match in re.finditer(
            r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', lock
        )
    }


def test_container_requirements_are_all_exactly_pinned():
    assert _pins(), "the container needs an explicit install list"


@pytest.mark.parametrize("package", sorted(_pins()))
def test_container_pin_matches_the_locked_version(package):
    """Tested versions and deployed versions must not diverge silently."""
    locked = _locked_versions()
    assert package in locked, f"{package} is not resolved in uv.lock"
    assert _pins()[package] == locked[package], (
        f"{package} is pinned to {_pins()[package]} for the container but uv.lock "
        f"resolves {locked[package]}"
    )


def test_container_requirements_declare_no_extras():
    """uvloop/httptools/watchfiles are absent from uv.lock, so `uvicorn[standard]` would
    be an unpinned, unaudited addition. No requirement may carry an extras bracket,
    because an extra drags in packages this repository never locked."""
    # Checks parsed requirement names, not the raw file, so a comment mentioning an
    # extra does not trip the guard.
    offenders = [name for name in _pins() if "[" in name]
    assert not offenders, f"container requirements must declare no extras: {offenders}"


def test_app_yaml_declares_only_the_two_supported_keys():
    document = yaml.safe_load(APP_YAML.read_text(encoding="utf-8"))
    assert set(document) <= {"command", "env"}, "app.yaml supports only command and env"
    assert isinstance(document["command"], list)


def test_app_yaml_starts_the_console_and_disables_the_access_log():
    """Request logs reach the app's Logs tab, and the process environment holds a live
    OAuth client secret."""
    command = yaml.safe_load(APP_YAML.read_text(encoding="utf-8"))["command"]
    assert command[0] == "uvicorn"
    assert "aai_console.server:app" in command
    assert "--no-access-log" in command


def test_app_yaml_hardcodes_no_environment_values():
    """Environment-specific values arrive from the bundle so a clone stays portable."""
    document = yaml.safe_load(APP_YAML.read_text(encoding="utf-8"))
    assert (
        "env" not in document or not document["env"]
    ), "AAI_CONSOLE_* values belong in the bundle's config.env, not in app.yaml"


def test_the_console_is_not_packaged_into_the_published_sdk_wheel():
    """`pip install aai-core` must not hand an SDK consumer a web server."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/aai_core"
    ]
    extras = pyproject["project"]["optional-dependencies"]
    assert "app" not in extras, (
        "a public `app` extra would misrepresent the SDK; the console's web "
        "dependencies belong in `dev`"
    )


def test_only_the_wheel_is_published_to_the_artifact_volume():
    """The sdist is a source archive and does contain `src/platform_app` — as it
    already contained `src/notebooks`. That is harmless only while the release workflow
    publishes the wheel alone, so pin that rather than leave it implicit: publishing the
    sdist would put a web server into the SDK artifact volume.
    """
    workflow = (WORKFLOWS / "publish-sdk.yml").read_text(encoding="utf-8")
    assert "python -m build --wheel" in workflow, "the release build must be wheel-only"
    assert "dist/*.tar.gz" not in workflow, "the sdist must not be published"


def test_console_dependencies_are_reachable_from_the_dev_extra():
    """cloud-verify.sh syncs only `--extra dev`, so anything the tests import must be
    there or pull-request CI fails on an ImportError."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = " ".join(pyproject["project"]["optional-dependencies"]["dev"])
    for package in ("fastapi", "jinja2", "uvicorn"):
        assert package in dev, f"{package} must be in the dev extra"


def test_databricks_sdk_stays_out_of_the_dev_extra():
    """tests/conftest.py records that provider extras are deliberately absent, and
    test_diagnostics relies on `dependency:databricks.sdk` still exercising its skip
    path. The console imports the SDK lazily instead."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = " ".join(pyproject["project"]["optional-dependencies"]["dev"])
    assert "databricks-sdk" not in dev


def test_console_tests_do_not_depend_on_starlettes_test_client():
    """Starlette's TestClient needs an HTTP client this repository does not lock.

    Starlette 0.x wants httpx, 1.x wants httpx2, and neither is in uv.lock, so a test
    built on it passes or fails depending on which starlette the resolver picked — it
    passed locally on 0.52 and broke CI on 1.3. tests/asgi_client.py drives the ASGI
    callable directly instead.
    """
    # Matches import statements only, so this test's own prose does not trip it.
    banned = re.compile(
        r"^\s*(from\s+(starlette|fastapi)\.testclient\s+import|import\s+httpx)", re.M
    )
    offenders = [
        path.name
        for path in sorted(Path(__file__).parent.glob("test_app_*.py"))
        if banned.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"use tests/asgi_client.py instead: {offenders}"


def test_workflow_directory_contains_no_yaml_extension_files():
    """GitHub honours `.yaml`, but every guard in this repository globs `*.yml`.

    A file named `.github/workflows/x.yaml` would run with a live OIDC identity while
    being invisible to the commit-pinning, CLI-lockstep, credential-free and
    no-environment checks simultaneously.
    """
    assert not list(
        WORKFLOWS.glob("*.yaml")
    ), "workflows must use .yml so the pinning and credential guards see them"


def test_app_resource_is_not_swept_in_by_the_wildcard_include():
    """`resources/*.yml` is one level deep, so a file in resources/optional/ is only
    deployed by an explicit, deliberate include. That is the safety property: the CI
    principal has no app permission until the external grants land."""
    bundle = yaml.safe_load((ROOT / "databricks.yml").read_text(encoding="utf-8"))
    assert "resources/*.yml" in bundle["include"]
    assert RESOURCE.parent.name == "optional"
    assert RESOURCE not in set((ROOT / "resources").glob("*.yml"))


def test_opting_the_console_in_requires_a_real_usage_policy():
    """Opting in must stay *possible*. This gate previously forbade the include
    outright, making the documented deployment path unmergeable on protected main.

    So the include is allowed; what is enforced is that enabling it comes with the
    provisioned inputs. An app has no tags field, so deploying with the placeholder
    policy id would put an always-on billable app in the workspace, unattributed.
    """
    bundle = yaml.safe_load((ROOT / "databricks.yml").read_text(encoding="utf-8"))
    included = any(
        "resources/optional/platform_console.yml" in str(entry)
        for entry in bundle["include"]
    )
    if not included:
        return  # default-off, nothing to validate

    policy = bundle["variables"]["app_usage_policy_id"]["default"]
    assert not policy.startswith("replace-with"), (
        "set app_usage_policy_id to the real serverless usage policy id before "
        "including the console — see docs/platform-console.md"
    )
    suffix = bundle["variables"]["app_suffix"]["default"]
    assert suffix, "app_suffix must be set; app names are workspace-global"


def test_app_resource_carries_cost_attribution_and_no_tags():
    """An app has no tags field, so a usage policy is the only attribution surface."""
    app = _app_resource()
    assert "tags" not in app and "custom_tags" not in app
    assert app["usage_policy_id"] == "${var.app_usage_policy_id}"


def test_app_name_is_legal_and_not_workspace_global():
    """App names allow only lowercase alphanumerics and hyphens, and get no
    development-mode prefix, so they must be disambiguated explicitly."""
    name = _app_resource()["name"]
    assert "${var.app_suffix}" in name
    literal = name.replace("${var.app_suffix}", "x")
    assert re.fullmatch(r"[a-z0-9-]+", literal), f"illegal app name: {literal}"


def test_app_source_code_path_points_at_the_console():
    assert (RESOURCE.parent / _app_resource()["source_code_path"]).resolve() == APP_DIR


def test_app_resource_command_matches_app_yaml():
    """Two sources of the start command would drift silently."""
    from_yaml = yaml.safe_load(APP_YAML.read_text(encoding="utf-8"))["command"]
    assert _app_resource()["config"]["command"] == from_yaml


def test_app_resource_supplies_every_identifier_the_console_reads():
    """The container cannot read platform-identifiers.json, so each AAI_CONSOLE_* value
    must arrive through the bundle rather than be baked into the source."""
    env = {e["name"]: e["value"] for e in _app_resource()["config"]["env"]}
    assert set(env) == {
        "AAI_CONSOLE_DATABRICKS_HOST",
        "AAI_CONSOLE_SDK_ARTIFACT_VOLUME",
        "AAI_CONSOLE_JOB_COMPUTE_POLICY_ID",
        "AAI_CONSOLE_TEMPLATE_REPO",
    }
    for value in env.values():
        assert value.startswith("${"), f"{value!r} must come from a bundle variable"


def test_dotted_volume_name_is_derived_from_the_identifier_fixture():
    """Two spellings of one volume is the drift the fixture exists to prevent."""
    bundle = yaml.safe_load((ROOT / "databricks.yml").read_text(encoding="utf-8"))
    variables = bundle["variables"]
    path = IDENTIFIERS["sdk_artifact_volume"]
    assert variables["sdk_artifact_volume"]["default"] == path
    assert variables["sdk_artifact_volume_full_name"]["default"] == ".".join(
        path.strip("/").split("/")[1:]
    )


def test_volume_binding_is_read_only():
    """The console reads no application data; anything beyond READ_VOLUME is excess."""
    binding = _app_resource()["resources"][0]["uc_securable"]
    assert binding["securable_type"] == "VOLUME"
    assert binding["permission"] == "READ_VOLUME"
