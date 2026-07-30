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
from enum import StrEnum
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


class HubStateMode(StrEnum):
    """Operational-store modes the current application can select safely.

    ``memory`` exists only for credential-free local development and tests. A hosted
    app must never accept it: Databricks App filesystems are ephemeral, and presenting
    process memory as a durable registry would make workflow state disappear on every
    restart. A Lakebase or SQL implementation can be added behind the repository
    interface once the corresponding resource and least-privilege grants are approved.
    """

    UNAVAILABLE = "unavailable"
    MEMORY = "memory"


class HubJobMode(StrEnum):
    """Explicit execution adapters; preview can never be selected when hosted."""

    UNAVAILABLE = "unavailable"
    DATABRICKS = "databricks"
    PREVIEW = "preview"


@dataclass(frozen=True)
class ConsoleConfig:
    identifiers: dict[str, str]
    hosted: bool
    app_name: str | None
    #: Git URL of the template hub. Unset means "generate the in-checkout relative
    #: form", which keeps a clone working before its own repository URL is configured.
    template_repo: str | None = None
    hub_state_mode: HubStateMode = HubStateMode.UNAVAILABLE
    hub_job_mode: HubJobMode = HubJobMode.UNAVAILABLE
    hub_registration_principals: frozenset[str] = frozenset()
    hub_platform_viewer_group: str | None = None
    hub_platform_admin_group: str | None = None
    hub_platform_auditor_group: str | None = None
    hub_local_actor: str = "local-developer"

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


def _csv_set(value: str) -> frozenset[str]:
    return frozenset(part.strip() for part in value.split(",") if part.strip())


def _optional(environ: dict[str, str], name: str) -> str | None:
    return environ.get(name, "").strip() or None


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

    requested_mode = environ.get("AAI_HUB_STATE_MODE", "").strip().lower()
    if requested_mode:
        try:
            state_mode = HubStateMode(requested_mode)
        except ValueError as error:
            choices = ", ".join(mode.value for mode in HubStateMode)
            raise ConfigError(
                f"AAI_HUB_STATE_MODE must be one of: {choices}"
            ) from error
    else:
        state_mode = HubStateMode.UNAVAILABLE if hosted else HubStateMode.MEMORY
    if hosted and state_mode is HubStateMode.MEMORY:
        raise ConfigError(
            "AAI_HUB_STATE_MODE=memory is local-preview only; bind an approved "
            "durable store before enabling hosted Hub writes"
        )

    requested_job_mode = environ.get("AAI_HUB_JOB_MODE", "").strip().lower()
    if requested_job_mode:
        try:
            job_mode = HubJobMode(requested_job_mode)
        except ValueError as error:
            choices = ", ".join(mode.value for mode in HubJobMode)
            raise ConfigError(f"AAI_HUB_JOB_MODE must be one of: {choices}") from error
    else:
        job_mode = HubJobMode.UNAVAILABLE
    if hosted and job_mode is HubJobMode.PREVIEW:
        raise ConfigError(
            "AAI_HUB_JOB_MODE=preview is local-preview only; hosted workflow "
            "actions must use approved Databricks Jobs"
        )

    return ConsoleConfig(
        identifiers=identifiers,
        hosted=hosted,
        app_name=app_name,
        template_repo=environ.get(f"{_ENV_PREFIX}TEMPLATE_REPO", "").strip() or None,
        hub_state_mode=state_mode,
        hub_job_mode=job_mode,
        hub_registration_principals=_csv_set(
            environ.get("AAI_HUB_REGISTRATION_PRINCIPALS", "")
        ),
        hub_platform_viewer_group=_optional(environ, "AAI_HUB_PLATFORM_VIEWER_GROUP"),
        hub_platform_admin_group=_optional(environ, "AAI_HUB_PLATFORM_ADMIN_GROUP"),
        hub_platform_auditor_group=_optional(environ, "AAI_HUB_PLATFORM_AUDITOR_GROUP"),
        hub_local_actor=(
            environ.get("AAI_HUB_LOCAL_ACTOR", "").strip() or "local-developer"
        ),
    )
