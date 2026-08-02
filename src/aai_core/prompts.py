"""Governed MLflow Prompt Registry operations."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from re import sub
from typing import Any

from aai_core.exceptions import AaiCoreError
from aai_core.tags import ResourceContext


class PromptPromotionError(AaiCoreError):
    """A prompt alias move was refused for lack of release evidence."""

    code = "aai_core.prompts.promotion_blocked"


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

    def ensure_version(
        self,
        name: str,
        template: str | list[dict[str, str]],
        *,
        commit_message: str,
        tags: dict[str, str] | None = None,
    ):
        """Reuse an identical immutable prompt version or register it once.

        Idempotency is by content digest: an existing version with the same
        template and ``prompt_digest`` tag is returned unchanged, so repeated
        deployments never mint duplicate versions.
        """

        qualified = self.qualify(name)
        digest = prompt_digest(template)
        client = self._client().MlflowClient()
        try:
            prompt_exists = client.get_prompt(qualified) is not None
        except Exception as exc:
            if not _is_missing_prompt(exc):
                raise
            # OSS returns None for a missing prompt; Unity Catalog raises
            # NOT_FOUND.
            prompt_exists = False
        if prompt_exists:
            for version in client.search_prompt_versions(qualified):
                version_tags = getattr(version, "tags", None) or {}
                if getattr(version, "template", None) == template and (
                    version_tags.get("aai_prompt_digest") == digest
                    or version_tags.get("aai.prompt_digest") == digest
                ):
                    return version
        merged_tags = dict(tags or {})
        merged_tags["prompt_digest"] = digest
        return self.register(
            name,
            template,
            commit_message=commit_message,
            tags=merged_tags,
        )

    def promote(
        self,
        name: str,
        *,
        version: int,
        evidence: Any,
        alias: str = "production",
    ) -> None:
        """Move a governed alias only on adopt-grade release evidence.

        ``evidence`` is a passing :class:`~aai_core.evaluation.GateResult` or
        an adopt :class:`~aai_core.decisions.DecisionRecord`. Anything less
        raises :class:`PromptPromotionError` and leaves the alias untouched.
        """

        from aai_core.decisions import Decision, DecisionRecord
        from aai_core.evaluation import GateResult

        if isinstance(evidence, GateResult):
            if not evidence.passed:
                failing = ", ".join(
                    failure.metric for failure in evidence.failures
                )
                raise PromptPromotionError(
                    f"Refusing to move alias {alias!r} for prompt {name!r}: "
                    f"the cited gate failed on {failing}",
                    remediation="Record an adopt decision backed by a passing "
                    "gate before moving the production alias.",
                )
        elif isinstance(evidence, DecisionRecord):
            if evidence.decision is not Decision.ADOPT:
                raise PromptPromotionError(
                    f"Refusing to move alias {alias!r} for prompt {name!r}: "
                    f"the cited decision is {evidence.decision.value!r}",
                    remediation="Record an adopt decision backed by a passing "
                    "gate before moving the production alias.",
                )
        else:
            raise TypeError(
                "evidence must be a GateResult or DecisionRecord"
            )
        self.set_alias(name, alias=alias, version=version)

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


def prompt_digest(template: str | list[dict[str, str]]) -> str:
    """Canonical content digest binding evidence to an exact prompt template."""

    canonical = json.dumps(
        {"template": template},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _is_missing_prompt(error: Exception) -> bool:
    error_code = str(getattr(error, "error_code", "")).upper()
    message = str(error).upper()
    return error_code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"} or any(
        marker in message
        for marker in ("NOT_FOUND", "RESOURCE_DOES_NOT_EXIST", "DOES NOT EXIST")
    )


def _prompt_tag_key(key: str) -> str:
    normalized = sub(r"[.,\-=/ :]+", "_", str(key)).strip("_")
    return f"aai_{normalized}"
