"""Governance guards for the console's deployment surface.

The container installs from `src/platform_app/requirements.txt`, which `uv lock --check`
never sees. That is a second dependency channel, and the only thing stopping it drifting
from the versions this repository actually tests is a test.
"""

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


def test_workflow_directory_contains_no_yaml_extension_files():
    """GitHub honours `.yaml`, but every guard in this repository globs `*.yml`.

    A file named `.github/workflows/x.yaml` would run with a live OIDC identity while
    being invisible to the commit-pinning, CLI-lockstep, credential-free and
    no-environment checks simultaneously.
    """
    assert not list(
        WORKFLOWS.glob("*.yaml")
    ), "workflows must use .yml so the pinning and credential guards see them"
