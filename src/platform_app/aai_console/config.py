"""Environment-supplied configuration for the platform console.

The console is deployed with `source_code_path` pointing at `src/platform_app`, so the
container cannot read repository files such as `platform-identifiers.json`. Every
environment-specific value therefore arrives as an `AAI_CONSOLE_*` environment variable
that the bundle supplies from its own variables.

`tests/test_app_content.py` asserts that no identifier literal appears anywhere under
`src/platform_app`, which is what keeps this repository clonable into another tenant.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Identifier keys the content may reference as ${identifier:<key>}. Keeping this list
# closed means a typo in the content YAML fails a test instead of rendering an empty
# string into a command a developer would then paste.
IDENTIFIER_KEYS = frozenset(
    {
        "databricks_host",
        "sdk_artifact_volume",
        "job_compute_policy_id",
    }
)

_ENV_PREFIX = "AAI_CONSOLE_"

# Injected by the Databricks Apps runtime for every app. Its presence is how we know we
# are hosted rather than running on a developer's machine.
_HOSTED_MARKER = "DATABRICKS_APP_NAME"


class ConfigError(RuntimeError):
    """Raised when the console cannot assemble a usable configuration."""


@dataclass(frozen=True)
class ConsoleConfig:
    identifiers: dict[str, str]
    hosted: bool
    app_name: str | None
    #: Git URL of the template hub. Unset means "generate the in-checkout relative
    #: form", which keeps a clone working before its own repository URL is configured.
    template_repo: str | None = None

    def identifier(self, key: str) -> str:
        if key not in IDENTIFIER_KEYS:
            raise ConfigError(f"unknown identifier {key!r}")
        value = self.identifiers.get(key, "")
        if not value:
            raise ConfigError(
                f"identifier {key!r} is not configured; the bundle must set "
                f"{_ENV_PREFIX}{key.upper()}"
            )
        return value


def _from_environment(environ: dict[str, str]) -> dict[str, str]:
    resolved = {}
    for key in IDENTIFIER_KEYS:
        value = environ.get(f"{_ENV_PREFIX}{key.upper()}", "").strip()
        if value:
            resolved[key] = value
    # The runtime always injects the workspace URL, so accept it as a fallback rather
    # than making the bundle repeat a value the platform already knows.
    if "databricks_host" not in resolved:
        host = environ.get("DATABRICKS_HOST", "").strip()
        if host:
            resolved["databricks_host"] = host
    return resolved


def _from_repository(start: Path) -> dict[str, str]:
    """Read the identifier fixture when running from a checkout.

    Local development only. A hosted app has no repository, so this never runs there —
    which is exactly why it is safe for it to know the fixture's filename.
    """
    for candidate in (start, *start.parents):
        fixture = candidate / "platform-identifiers.json"
        if fixture.is_file():
            raw = json.loads(fixture.read_text(encoding="utf-8"))
            return {key: raw[key] for key in IDENTIFIER_KEYS if raw.get(key)}
    return {}


def load_config(
    environ: dict[str, str] | None = None,
    *,
    start: Path | None = None,
) -> ConsoleConfig:
    environ = dict(os.environ if environ is None else environ)
    app_name = environ.get(_HOSTED_MARKER) or None
    hosted = app_name is not None

    identifiers = _from_environment(environ)
    if not hosted:
        # Fall back to the checkout so `make app-run` works with no setup. Hosted apps
        # never reach this branch, so a missing value there stays a loud failure.
        for key, value in _from_repository(start or Path(__file__).resolve()).items():
            identifiers.setdefault(key, value)

    return ConsoleConfig(
        identifiers=identifiers,
        hosted=hosted,
        app_name=app_name,
        template_repo=environ.get(f"{_ENV_PREFIX}TEMPLATE_REPO", "").strip() or None,
    )
