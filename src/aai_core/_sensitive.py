"""Internal canonical normalization for credential-bearing field names."""

from __future__ import annotations

import re
from collections.abc import Collection

__all__: list[str] = []

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
_SENSITIVE_COMPONENTS = frozenset({"password", "passwd", "secret", "secrets"})
_SENSITIVE_NAMES = frozenset(
    {
        "access_token",
        "accesstoken",
        "api_key",
        "apikey",
        "auth_header",
        "authorization",
        "authorization_header",
        "bearer_token",
        "client_secret",
        "clientsecret",
        "connection_string",
        "connectionstring",
        "credential",
        "credentials",
        "encryption_key",
        "oauth_token",
        "pat",
        "private_key",
        "privatekey",
        "refresh_token",
        "refreshtoken",
        "sas_token",
        "sastoken",
        "signing_key",
        "token",
    }
)


def normalize_sensitive_name(value: str) -> str:
    """Normalize camelCase and punctuation without changing field semantics."""

    separated = _CAMEL_BOUNDARY.sub("_", str(value).strip())
    return _NON_ALNUM.sub("_", separated).strip("_").casefold()


def is_sensitive_name(value: str) -> bool:
    """Return whether a field name denotes credential material."""

    normalized = normalize_sensitive_name(value)
    if not normalized:
        return False
    parts = tuple(part for part in normalized.split("_") if part)
    if set(parts).intersection(_SENSITIVE_COMPONENTS):
        return True
    return any(
        normalized == protected or normalized.endswith(f"_{protected}")
        for protected in _SENSITIVE_NAMES
    )


def matches_protected_name(value: str, protected_names: Collection[str]) -> bool:
    """Match a field against a boundary-specific protected-name vocabulary."""

    normalized = normalize_sensitive_name(value)
    return any(
        normalized == protected or normalized.endswith(f"_{protected}")
        for candidate in protected_names
        if (protected := normalize_sensitive_name(candidate))
    )
