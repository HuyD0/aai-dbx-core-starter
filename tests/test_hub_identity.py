from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest
from pydantic import ValidationError

from aai_console.hub.identity import (
    FailClosedRoleResolver,
    HubAuthenticationError,
    HubRoleResolutionError,
    RoleAssignment,
    StaticRoleResolver,
    actor_view,
    authorization_context_from_headers,
)
from aai_console.hub.models import Role


class TokenGuardHeaders(Mapping[str, str]):
    """Mapping that fails if production code attempts to read the access token."""

    def __init__(self) -> None:
        self._values = {
            "X-Forwarded-User": "alice@example.com",
            "X-Forwarded-Access-Token": "secret-access-token",
        }

    def __getitem__(self, key: str) -> str:
        if key.casefold() == "x-forwarded-access-token":
            raise AssertionError("forwarded access token must not be read")
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def test_hosted_identity_requires_forwarded_user() -> None:
    with pytest.raises(
        HubAuthenticationError, match="authenticated Databricks user is required"
    ):
        authorization_context_from_headers(
            {},
            hosted=True,
            local_actor="must-not-be-used",
        )


def test_hosted_identity_accepts_case_insensitive_forwarded_user_header() -> None:
    actor = authorization_context_from_headers(
        {"X-Forwarded-User": " alice@example.com "},
        hosted=True,
        local_actor="must-not-be-used",
    )

    assert actor.principal == "alice@example.com"
    assert actor.groups == ()
    assert actor.platform_roles == ()


def test_local_preview_uses_configured_actor_and_ignores_forwarded_claims() -> None:
    actor = authorization_context_from_headers(
        {
            "X-Forwarded-User": "spoofed@example.com",
            "X-Forwarded-Groups": "admins",
            "X-Forwarded-Role": "platform_administrator",
        },
        hosted=False,
        local_actor="local-developer",
    )

    assert actor.principal == "local-developer"
    assert actor.groups == ()
    assert actor.platform_roles == ()


def test_default_resolver_is_fail_closed() -> None:
    actor = authorization_context_from_headers(
        {
            "x-forwarded-user": "alice@example.com",
            "x-forwarded-groups": "admins",
            "x-forwarded-role": "platform_administrator",
        },
        hosted=True,
        local_actor="unused",
        role_resolver=FailClosedRoleResolver(),
    )

    assert actor.groups == ()
    assert actor.platform_roles == ()


def test_static_resolver_supplies_only_explicit_trusted_assignments() -> None:
    resolver = StaticRoleResolver(
        {
            "alice@example.com": RoleAssignment(
                groups=("ai-platform-admins",),
                platform_roles=(Role.PLATFORM_ADMINISTRATOR, Role.AUDITOR),
            )
        }
    )

    actor = authorization_context_from_headers(
        {"x-forwarded-user": "alice@example.com"},
        hosted=True,
        local_actor="unused",
        role_resolver=resolver,
    )

    assert actor.groups == ("ai-platform-admins",)
    assert actor.platform_roles == (
        Role.PLATFORM_ADMINISTRATOR,
        Role.AUDITOR,
    )


def test_forwarded_access_token_is_never_read_or_projected() -> None:
    actor = authorization_context_from_headers(
        TokenGuardHeaders(),
        hosted=True,
        local_actor="unused",
    )
    projection = actor_view(actor)

    rendered = repr((actor, projection, projection.model_dump()))
    assert actor.principal == "alice@example.com"
    assert "secret-access-token" not in rendered
    assert set(projection.model_dump()) == {
        "display_name",
        "role_label",
        "is_platform_admin",
    }


def test_actor_view_excludes_groups_and_uses_highest_platform_role() -> None:
    resolver = StaticRoleResolver(
        {
            "alice@example.com": RoleAssignment(
                groups=("sensitive-group-name",),
                platform_roles=(Role.PLATFORM_VIEWER, Role.PLATFORM_ADMINISTRATOR),
            )
        }
    )
    actor = authorization_context_from_headers(
        {"x-forwarded-user": "alice@example.com"},
        hosted=True,
        local_actor="unused",
        role_resolver=resolver,
    )

    projection = actor_view(actor)

    assert projection.display_name == "alice@example.com"
    assert projection.role_label == "Platform administrator"
    assert projection.is_platform_admin is True
    assert "sensitive-group-name" not in repr(projection)


@pytest.mark.parametrize(
    "principal",
    ["", "   ", "alice@example.com\r\nx-forged: true", "a" * 513],
)
def test_invalid_forwarded_principal_fails_without_echoing_value(
    principal: str,
) -> None:
    with pytest.raises(HubAuthenticationError) as exc_info:
        authorization_context_from_headers(
            {"x-forwarded-user": principal},
            hosted=True,
            local_actor="unused",
        )

    if principal:
        assert principal not in str(exc_info.value)


def test_malformed_or_failed_role_resolver_is_reported_generically() -> None:
    class BrokenResolver:
        def resolve(self, principal: str) -> RoleAssignment:
            raise RuntimeError(f"sensitive resolver detail for {principal}")

    with pytest.raises(
        HubRoleResolutionError, match="trusted role resolution failed"
    ) as exc_info:
        authorization_context_from_headers(
            {"x-forwarded-user": "alice@example.com"},
            hosted=True,
            local_actor="unused",
            role_resolver=BrokenResolver(),
        )

    assert "alice@example.com" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_role_assignment_is_strict_frozen_and_rejects_duplicates() -> None:
    with pytest.raises(ValidationError):
        RoleAssignment(
            groups=("admins", "admins"),
            platform_roles=(),
        )
    with pytest.raises(ValidationError):
        RoleAssignment.model_validate(
            {
                "groups": (),
                "platform_roles": (),
                "untrustedRole": "platform_administrator",
            }
        )

    assignment = RoleAssignment()
    with pytest.raises(ValidationError):
        assignment.groups = ("admins",)
