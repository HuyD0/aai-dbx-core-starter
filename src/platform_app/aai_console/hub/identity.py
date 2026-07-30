"""Trusted identity boundary for AI Platform Hub requests.

Databricks Apps terminates user authentication and supplies the authenticated user in
``X-Forwarded-User``.  This module deliberately reads only that identity assertion.
Forwarded access tokens and request-supplied role/group claims are outside this
boundary and must never influence Hub authorization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import field_validator

from .models import AuthorizationContext, HubModel, NonEmptyStr, Role

_FORWARDED_USER_HEADER = "x-forwarded-user"
_MAX_PRINCIPAL_LENGTH = 512


class HubAuthenticationError(RuntimeError):
    """Raised when a request has no usable trusted identity assertion."""


class HubRoleResolutionError(RuntimeError):
    """Raised when the configured role resolver cannot produce trusted roles."""


class RoleAssignment(HubModel):
    """Trusted group and platform-role membership returned by a resolver."""

    groups: tuple[NonEmptyStr, ...] = ()
    platform_roles: tuple[Role, ...] = ()

    @field_validator("groups", "platform_roles")
    @classmethod
    def unique_values(cls, value: tuple) -> tuple:
        if len(value) != len(set(value)):
            raise ValueError("values must be unique")
        return value


@runtime_checkable
class RoleResolver(Protocol):
    """Resolve trusted memberships for one already-authenticated principal."""

    def resolve(self, principal: str) -> RoleAssignment:
        """Return memberships from a trusted source."""


class FailClosedRoleResolver:
    """Default resolver: authenticate the user but grant no inferred privileges."""

    def resolve(self, principal: str) -> RoleAssignment:
        del principal
        return RoleAssignment()


class StaticRoleResolver:
    """Explicit, immutable assignments for tests and credential-free local preview."""

    def __init__(self, assignments: Mapping[str, RoleAssignment]) -> None:
        self._assignments = dict(assignments)

    def resolve(self, principal: str) -> RoleAssignment:
        return self._assignments.get(principal, RoleAssignment())


class ActorView(HubModel):
    """Minimal actor data safe to project into the server-rendered UI."""

    display_name: NonEmptyStr
    role_label: NonEmptyStr
    is_platform_admin: bool


def authorization_context_from_headers(
    headers: Mapping[str, str],
    *,
    hosted: bool,
    local_actor: str,
    role_resolver: RoleResolver | None = None,
) -> AuthorizationContext:
    """Build request authorization only from the trusted delivery boundary.

    Hosted requests require Databricks Apps' ``X-Forwarded-User`` assertion. Local
    preview has no trusted proxy, so it uses the explicitly configured local actor and
    ignores forwarded identity claims.
    """

    if hosted:
        principal = _header_value(headers, _FORWARDED_USER_HEADER)
        if principal is None:
            raise HubAuthenticationError("authenticated Databricks user is required")
    else:
        principal = local_actor

    principal = _validated_principal(principal)
    resolver = role_resolver or FailClosedRoleResolver()
    try:
        assignment = resolver.resolve(principal)
    except Exception:
        raise HubRoleResolutionError("trusted role resolution failed") from None
    if not isinstance(assignment, RoleAssignment):
        raise HubRoleResolutionError(
            "trusted role resolver returned an invalid assignment"
        )

    return AuthorizationContext(
        principal=principal,
        groups=assignment.groups,
        platform_roles=assignment.platform_roles,
    )


def actor_view(actor: AuthorizationContext) -> ActorView:
    """Return the smallest useful UI projection, excluding group memberships."""

    roles = set(actor.platform_roles)
    if Role.PLATFORM_ADMINISTRATOR in roles:
        role_label = "Platform administrator"
    elif Role.AUDITOR in roles:
        role_label = "Auditor"
    elif Role.PLATFORM_VIEWER in roles:
        role_label = "Platform viewer"
    elif Role.OWNER in roles:
        role_label = "Application owner"
    elif Role.CONTRIBUTOR in roles:
        role_label = "Contributor"
    else:
        role_label = "Application-scoped access"
    return ActorView(
        display_name=actor.principal,
        role_label=role_label,
        is_platform_admin=Role.PLATFORM_ADMINISTRATOR in roles,
    )


def _header_value(headers: Mapping[str, str], expected_name: str) -> str | None:
    """Read exactly one non-secret header without materializing all header values."""

    direct = headers.get(expected_name)
    if direct is not None:
        return direct
    for name in headers:
        if name.casefold() == expected_name:
            return headers[name]
    return None


def _validated_principal(raw: str) -> str:
    if not isinstance(raw, str):
        raise HubAuthenticationError("authenticated Databricks user is invalid")
    principal = raw.strip()
    if (
        not principal
        or len(principal) > _MAX_PRINCIPAL_LENGTH
        or any(character in "\r\n\x00" for character in principal)
    ):
        raise HubAuthenticationError("authenticated Databricks user is invalid")
    return principal
