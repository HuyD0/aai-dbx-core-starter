"""Governed MLflow Prompt Registry operations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from re import sub
from typing import Any

from aai_core.tags import ResourceContext

# Structured codes that are authoritatively NOT absence. Registry services
# often word permission denials as "does not exist" to avoid disclosing an
# inaccessible prompt, so these must override any message marker below.
_NON_MISSING_ERROR_CODES = {
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
    "UNAUTHORIZED",
    "TEMPORARILY_UNAVAILABLE",
    "REQUEST_LIMIT_EXCEEDED",
}


@dataclass(frozen=True)
class PromptReference:
    name: str
    version: int | None = None
    alias: str | None = None

    @property
    def uri(self) -> str:
        if self.version is not None:
            return f"prompts:/{self.name}/{self.version}"
        if self.alias:
            return f"prompts:/{self.name}@{self.alias}"
        raise ValueError("PromptReference requires a version or alias")


class PromptManager:
    def __init__(
        self,
        *,
        context: ResourceContext,
        catalog: str,
        schema: str,
        mlflow_module: Any | None = None,
    ) -> None:
        self.context = context
        self.catalog = catalog
        self.schema = schema
        self._mlflow = mlflow_module

    def register(
        self,
        name: str,
        template: str | list[dict[str, str]],
        *,
        commit_message: str,
        tags: dict[str, str] | None = None,
    ):
        qualified = self.qualify(name)
        metadata = self.context.merged(tags)
        return self._client().genai.register_prompt(
            name=qualified,
            template=template,
            commit_message=commit_message,
            # Unity Catalog prompt tags reject punctuation accepted by the
            # local MLflow registry. Use one portable projection for both.
            tags={_prompt_tag_key(key): value for key, value in metadata.items()},
        )

    def load(
        self,
        name: str,
        *,
        version: int | None = None,
        alias: str | None = None,
        cache_ttl_seconds: float | None = None,
    ):
        reference = PromptReference(self.qualify(name), version=version, alias=alias)
        kwargs = (
            {"cache_ttl_seconds": cache_ttl_seconds}
            if cache_ttl_seconds is not None
            else {}
        )
        return self._client().genai.load_prompt(reference.uri, **kwargs)

    def set_alias(self, name: str, *, alias: str, version: int) -> None:
        if alias == "candidate":
            warnings.warn(
                "The 'candidate' prompt alias is deprecated; use the more "
                "descriptive 'validation' alias. It will be removed in "
                "aai-core 0.5.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if alias not in {"development", "validation", "candidate", "production"}:
            raise ValueError(f"Unsupported governed prompt alias: {alias}")
        self._client().genai.set_prompt_alias(
            name=self.qualify(name),
            alias=alias,
            version=version,
        )

    def qualify(self, name: str) -> str:
        parts = name.split(".")
        if len(parts) == 1:
            return f"{self.catalog}.{self.schema}.{name}"
        if len(parts) == 3:
            return name
        raise ValueError("Prompt names must be unqualified or catalog.schema.name")

    def _client(self):
        if self._mlflow is not None:
            return self._mlflow
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError(
                "Prompt support requires the `genai` extra. From an aai-core "
                "checkout run `make examples-install` and use `.venv/bin/python`; "
                "in a consuming environment install `aai-core[genai]`."
            ) from error
        return mlflow

    @property
    def native_client(self) -> Any:
        """Expose native MLflow prompt APIs without wrapping new features."""

        return self._client()


def _prompt_tag_key(key: str) -> str:
    normalized = sub(r"[.,\-=/ :]+", "_", str(key)).strip("_")
    return f"aai_{normalized}"


def is_missing_prompt_error(error: Exception) -> bool:
    """True only when a registry error authoritatively means absence.

    Authentication, permission, rate-limit, and transient registry failures
    return ``False`` even when their message uses non-disclosure wording such
    as "does not exist". Callers may therefore fall back to bundled content
    only for a genuinely missing prompt or alias.
    """

    error_code = str(getattr(error, "error_code", "")).strip().upper()
    if error_code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}:
        return True
    if error_code in _NON_MISSING_ERROR_CODES:
        return False
    message = str(error).upper()
    if any(
        marker in message
        for marker in ("NOT_FOUND", "RESOURCE_DOES_NOT_EXIST", "DOES NOT EXIST")
    ):
        return True
    # The file and SQL registries report a missing alias as
    # INVALID_PARAMETER_VALUE with "Registered model alias ... not found."
    # Recognize that narrow shape without treating arbitrary invalid input as
    # absence.
    return "ALIAS" in message and "NOT FOUND" in message
