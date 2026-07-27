"""Workspace checks, executed as the app's own service principal.

Everything here answers "can the *platform* reach this?", never "can *you* reach this?".
On-behalf-of-user authorization is deliberately not used: its consent is
irrevocable, and its documented scopes do not cover compute policies, Unity Catalog
volumes or catalog grants — precisely the rungs onboarding cares about.

That distinction is enforced rather than documented. Every check carries
`identity="app_sp"`, and `assert_platform_state` raises if a caller tries to present
these rows as the viewer's own access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from .config import ConsoleConfig

Status = Literal["pass", "fail", "skip"]
Identity = Literal["app_sp"]

#: Rendered above every check list. The console has no way to see a viewer's grants, so
#: saying so plainly is the honest design, not a caveat bolted on afterwards.
PLATFORM_STATE_HEADING = "App runtime — not your laptop, and not your personal access"


@dataclass(frozen=True)
class PlatformCheck:
    id: str
    label: str
    status: Status
    detail: str
    identity: Identity = "app_sp"


def assert_platform_state(checks: list[PlatformCheck], heading: str) -> None:
    """Guard against a future edit presenting app-SP rows as the viewer's own access."""
    if heading != PLATFORM_STATE_HEADING:
        raise ValueError(
            "app-service-principal checks may only be rendered under "
            f"{PLATFORM_STATE_HEADING!r}; got {heading!r}"
        )
    for check in checks:
        if check.identity != "app_sp":
            raise ValueError(f"check {check.id!r} claims a non-app_sp identity")


def _workspace_client() -> tuple[object | None, str]:
    """Import the Databricks SDK lazily and build a client from ambient auth.

    Returns `(client, reason)`; `client` is None when the console cannot reach the
    workspace, and `reason` says which of the two causes applies. The SDK is
    deliberately absent from the dev environment (see tests/conftest.py), so
    must degrade rather than fail to import — and a developer running `make app-run`
    locally has no app credentials either.
    """
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        return None, "the Databricks SDK is not installed in this environment"
    try:
        # Let unified auth read DATABRICKS_CLIENT_ID/SECRET from the environment. Never
        # pass the secret explicitly: it must never become an argument that a traceback
        # or a log line could capture.
        return WorkspaceClient(), ""
    except Exception as error:
        return (
            None,
            "no workspace credentials are configured for the console "
            f"({type(error).__name__})",
        )


class WorkspaceProbe:
    """Thin seam over the Databricks SDK so tests can run with no cloud identity."""

    def __init__(self, client=None) -> None:
        if client is not None:
            self._client, self._reason = client, ""
        else:
            self._client, self._reason = _workspace_client()

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    def current_principal(self) -> str:
        me = self._client.current_user.me()
        return getattr(me, "user_name", None) or getattr(me, "display_name", "unknown")

    def compute_policy_name(self, policy_id: str) -> str:
        policy = self._client.cluster_policies.get(policy_id=policy_id)
        return getattr(policy, "name", policy_id)

    def volume_entries(self, volume_path: str) -> int:
        return len(list(self._client.files.list_directory_contents(volume_path)))


def _skip_all(reason: str) -> list[PlatformCheck]:
    return [
        PlatformCheck(id=check_id, label=label, status="skip", detail=reason)
        for check_id, label in (
            ("identity", "App service principal"),
            ("compute_policy", "Constrained job compute policy"),
            ("sdk_volume", "SDK artifact volume"),
        )
    ]


def template_source_check(config: ConsoleConfig) -> PlatformCheck:
    """Report whether this console can generate a usable `bundle init`.

    Needs no workspace call, so it runs even when the workspace is unreachable —
    a clone whose bundle never wired `template_repo` would otherwise discover the
    problem only when a developer pasted a broken command.
    """

    label = "Template repository"
    if config.template_repo:
        return PlatformCheck(
            id="template_source",
            label=label,
            status="pass",
            detail=f"projects generate from {config.template_repo}",
        )
    if config.hosted:
        return PlatformCheck(
            id="template_source",
            label=label,
            status="fail",
            detail=(
                "no template repository configured; set AAI_CONSOLE_TEMPLATE_REPO "
                "from the bundle's `template_repo` variable "
                "(value: platform-identifiers.json)"
            ),
        )
    return PlatformCheck(
        id="template_source",
        label=label,
        status="skip",
        detail="running from a checkout; projects generate from ./templates",
    )


def run_checks(
    config: ConsoleConfig, probe: WorkspaceProbe | None = None
) -> list[PlatformCheck]:
    # Independent of the workspace, so it is reported on every path below.
    template_source = template_source_check(config)

    if probe is None:
        if not config.hosted:
            # Outside the hosted app there is no app service principal. Ambient auth
            # here is the *developer's* own identity, so probing would report personal
            # permissions under a heading claiming they are platform state — the exact
            # conflation this module exists to prevent. Refuse rather than mislabel.
            return [
                template_source,
                *_skip_all(
                    "platform state is only reported by the hosted app; locally, "
                    "run `python3.12 scripts/setup_dev.py --check-only` to check "
                    "your access"
                ),
            ]
        probe = WorkspaceProbe()

    if not probe.available:
        return [
            template_source,
            *_skip_all(probe.unavailable_reason or "the workspace is not reachable"),
        ]

    checks: list[PlatformCheck] = [template_source]

    try:
        principal = probe.current_principal()
        checks.append(
            PlatformCheck(
                id="identity",
                label="App service principal",
                status="pass",
                detail=f"the console runs as {principal}",
            )
        )
    except Exception as error:
        checks.append(
            PlatformCheck(
                id="identity",
                label="App service principal",
                status="fail",
                detail=_safe_detail(error),
            )
        )

    policy_id = config.identifier("job_compute_policy_id")
    volume = config.identifier("sdk_artifact_volume")
    checks.append(
        _guarded(
            "compute_policy",
            "Constrained job compute policy",
            lambda: f"visible: {probe.compute_policy_name(policy_id)}",
        )
    )
    checks.append(
        _guarded(
            "sdk_volume",
            "SDK artifact volume",
            lambda: f"readable: {probe.volume_entries(volume)} entries",
        )
    )
    return checks


def _guarded(check_id: str, label: str, action) -> PlatformCheck:
    try:
        return PlatformCheck(id=check_id, label=label, status="pass", detail=action())
    except Exception as error:
        return PlatformCheck(
            id=check_id, label=label, status="fail", detail=_safe_detail(error)
        )


#: Environment variables whose *values* must never appear in a rendered detail. The
#: Databricks SDK's own auth error redacts client_secret but interpolates client_id
#: verbatim, so scrubbing here rather than trusting the provider is the load-bearing
#: control — it stays correct whatever a future SDK version decides to include.
_SENSITIVE_ENV = (
    "DATABRICKS_CLIENT_SECRET",
    "DATABRICKS_CLIENT_ID",
    "DATABRICKS_TOKEN",
    "AZURE_CLIENT_SECRET",
    "AZURE_CLIENT_ID",
    "ARM_CLIENT_SECRET",
)


def _safe_detail(error: Exception) -> str:
    """Render an exception without letting a credential or a long payload reach the
    page."""
    text = " ".join(str(error).split())
    for name in _SENSITIVE_ENV:
        value = os.environ.get(name, "")
        # Short values would match far too much unrelated text to redact safely.
        if len(value) >= 8:
            text = text.replace(value, "***")
    return text[:200] if text else error.__class__.__name__
