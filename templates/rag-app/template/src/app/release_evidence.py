"""Canonical, non-secret RAG configuration evidence for release joins."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

_RESOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$")
_WORLD_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")
_SECRET_PREFIXES = (
    "bearer",
    "dapi",
    "eyj",
    "github_pat_",
    "ghp_",
    "gho_",
    "ghr_",
    "ghs_",
    "ghu_",
    "pat-",
    "sk-",
)
_SENSITIVE_CONFIG_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "client_secret",
        "credential",
        "credentials",
        "header",
        "headers",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
)


def release_configuration(settings: Any) -> dict[str, dict[str, Any]]:
    """Return reviewed model, embedding, retrieval, and index evidence."""

    model_config = _configured(settings, "models", "general-chat")
    embedding_config = _configured(settings, "embeddings", "knowledge-embedding")
    retrieval_config = _configured(settings, "retrievers", "product-knowledge")
    _required_string(model_config, "deployment", "general-chat")
    _required_string(embedding_config, "deployment", "knowledge-embedding")
    _required_string(retrieval_config, "index", "product-knowledge")
    _required_string(retrieval_config, "embedding", "product-knowledge")
    model = _reviewed_configuration(
        model_config,
        logical_name="general-chat",
        allowed=("provider", "deployment", "model", "capabilities"),
        include_endpoint_for=("foundry",),
    )
    embedding = _reviewed_configuration(
        embedding_config,
        logical_name="knowledge-embedding",
        allowed=(
            "provider",
            "deployment",
            "model",
            "dimensions",
            "capabilities",
        ),
        include_endpoint_for=("foundry",),
    )
    retrieval = _reviewed_configuration(
        retrieval_config,
        logical_name="product-knowledge",
        allowed=(
            "provider",
            "index",
            "id_field",
            "content_field",
            "source_uri_field",
            "chunk_id_field",
            "vector_fields",
            "columns",
            "embedding",
        ),
        include_endpoint_for=("azure_ai_search", "databricks_ai_search"),
    )
    index = {
        key: retrieval[key]
        for key in (
            "provider",
            "endpoint_sha256",
            "index",
            "id_field",
            "content_field",
            "source_uri_field",
            "chunk_id_field",
            "vector_fields",
            "columns",
        )
        if key in retrieval
    }
    return {
        "model": model,
        "embedding": embedding,
        "retrieval": retrieval,
        "index": index,
    }


def configuration_digests(
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Digest every independently changeable RAG configuration boundary."""

    required = ("model", "embedding", "retrieval", "index")
    missing = [name for name in required if name not in evidence]
    if missing:
        raise ValueError("RAG configuration evidence is incomplete")
    digests = {
        f"{name}_configuration_digest": canonical_digest(evidence[name])
        for name in required
    }
    digests["rag_configuration_digest"] = canonical_digest(
        {name: evidence[name] for name in required}
    )
    return digests


def model_identity(settings: Any, logical_name: str) -> str:
    """Return the release identity used by both evaluation and manifesting."""

    config = _configured(settings, "models", logical_name)
    provider = _required_string(config, "provider", logical_name)
    deployment = _required_string(config, "deployment", logical_name)
    identity = f"{provider}:{deployment}"
    if provider.casefold() == "foundry":
        identity += "@endpoint-sha256:" + endpoint_sha256(config.get("endpoint"))
    return identity


def knowledge_version(value: str) -> str:
    """Validate a bounded, non-secret external knowledge snapshot identifier."""

    if not isinstance(value, str):
        raise TypeError("knowledge version must be a bounded non-secret identifier")
    normalized = value.strip()
    lowered = normalized.casefold()
    if (
        not _WORLD_VERSION.fullmatch(normalized)
        or lowered.startswith(_SECRET_PREFIXES)
        or any(marker in lowered for marker in ("replace-with", "changeme", "todo"))
    ):
        raise ValueError("knowledge version must be a bounded non-secret identifier")
    return normalized


def endpoint_sha256(value: Any) -> str:
    """Hash a validated endpoint without persisting its routing value."""

    if not isinstance(value, str) or not value.strip():
        raise TypeError("provider endpoint must be a non-empty string")
    endpoint = value.strip()
    if any(character.isspace() or ord(character) < 32 for character in endpoint):
        raise ValueError("provider endpoint contains invalid characters")
    if "://" not in endpoint:
        if not _RESOURCE_NAME.fullmatch(endpoint):
            raise ValueError(
                "provider endpoint must be HTTPS or a bounded resource name"
            )
        normalized = endpoint
    else:
        normalized = _normalized_https_endpoint(endpoint)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonical_digest(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _configured(settings: Any, collection: str, logical_name: str) -> Mapping[str, Any]:
    configured = getattr(settings, collection, None)
    if not isinstance(configured, Mapping):
        raise TypeError(f"settings.{collection} must be a mapping")
    config = configured.get(logical_name)
    if not isinstance(config, Mapping):
        raise TypeError(f"{logical_name} must be configured as a mapping")
    _reject_secret_material(config)
    return config


def _reviewed_configuration(
    config: Mapping[str, Any],
    *,
    logical_name: str,
    allowed: tuple[str, ...],
    include_endpoint_for: tuple[str, ...],
) -> dict[str, Any]:
    provider = _required_string(config, "provider", logical_name)
    reviewed = {key: config[key] for key in allowed if key in config}
    reviewed["logical_name"] = logical_name
    if provider.casefold() in include_endpoint_for:
        reviewed["endpoint_sha256"] = endpoint_sha256(config.get("endpoint"))
    return reviewed


def _required_string(config: Mapping[str, Any], key: str, logical_name: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{logical_name} {key} must be a non-empty string")
    return value.strip()


def _normalized_https_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "provider endpoint must be HTTPS without userinfo, query, or fragment"
        )
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("provider endpoint contains an invalid port") from error
    raw_hostname = parsed.hostname.rstrip(".")
    try:
        hostname = raw_hostname.encode("idna").decode("ascii").casefold()
    except UnicodeError as error:
        raise ValueError("provider endpoint contains an invalid hostname") from error
    if ":" in hostname:
        hostname = f"[{hostname}]"
    path = re.sub(r"/+", "/", parsed.path or "/")
    if any(segment in {".", ".."} for segment in path.split("/")):
        raise ValueError("provider endpoint path must not contain dot segments")
    if path != "/":
        path = path.rstrip("/")
    return f"https://{hostname}:{port or 443}{path}"


def _reject_secret_material(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _SENSITIVE_CONFIG_NAMES or any(
                normalized.endswith(f"_{name}") for name in _SENSITIVE_CONFIG_NAMES
            ):
                raise RuntimeError(
                    "Release configuration contains secret-bearing material; "
                    "use a governed secret reference outside release evidence"
                )
            _reject_secret_material(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _reject_secret_material(nested)
        return
    if isinstance(value, str) and value.strip().casefold().startswith(_SECRET_PREFIXES):
        raise RuntimeError(
            "Release configuration contains credential-shaped material; use a "
            "governed secret reference outside release evidence"
        )
