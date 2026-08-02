"""Governed MLflow Prompt Registry operations."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from re import fullmatch, sub
from typing import Any

from aai_core.exceptions import AaiCoreError
from aai_core.tags import ResourceContext


class PromptPromotionError(AaiCoreError):
    """A prompt alias move was refused for lack of release evidence."""

    code = "aai_core.prompts.promotion_blocked"


# 'candidate' remains accepted only as the deprecated alias name that
# set_alias() warns about; it is not lifecycle vocabulary.
_GOVERNED_ALIASES = {"development", "validation", "candidate", "production"}

# The same component shape DecisionRecord.prompt_name accepts: a name the
# registry would take but the evidence contract refuses could never be
# promoted, so the mismatch is refused at registration time instead.
_NAME_COMPONENT = r"[A-Za-z0-9_-]+"


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
            if not is_missing_prompt_error(exc):
                raise
            # OSS returns None for a missing prompt; Unity Catalog raises
            # NOT_FOUND.
            prompt_exists = False
        if prompt_exists:
            # The registry paginates version search; a long-lived prompt's
            # matching version may sit past the first page.
            page_token = None
            while True:
                if page_token is None:
                    page = client.search_prompt_versions(qualified)
                else:
                    page = client.search_prompt_versions(
                        qualified, page_token=page_token
                    )
                for version in page:
                    version_tags = getattr(version, "tags", None) or {}
                    if getattr(version, "template", None) == template and (
                        version_tags.get("aai_prompt_digest") == digest
                        or version_tags.get("aai.prompt_digest") == digest
                    ):
                        return version
                page_token = getattr(page, "token", None)
                if not page_token:
                    break
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
        """Move a governed alias only on an adopt decision bound to the
        exact version being promoted.

        ``evidence`` is an adopt :class:`~aai_core.decisions.DecisionRecord`
        whose ``prompt_digest``, qualified ``prompt_name``, and immutable
        ``prompt_version`` were recorded at decision time; the registry
        version's actual template must match the digest, and the name and
        version must match the prompt being promoted, so evidence gathered
        for one template, one prompt, or one version can never move
        another's alias.
        A bare :class:`~aai_core.evaluation.GateResult` is refused: gate
        evidence alone carries no template identity, and a digest supplied
        at promotion time would prove only what is being promoted, not what
        was evaluated. Anything less raises :class:`PromptPromotionError`
        and leaves the alias untouched.

        This is a process guard against mistakes, not an authorization
        mechanism: metric provenance is not attestable at the client, so
        authorization for alias moves remains the registry's access
        controls and the protected-main release path. Persisting the
        decision first with ``record_decision()`` is the documented
        convention the labs and templates follow.
        """

        from aai_core.decisions import Decision, DecisionRecord
        from aai_core.evaluation import GateResult

        # The cheapest, fully local check comes first: an alias typo must
        # fail deterministically, never as a network error from the
        # verification fetch below.
        if alias not in _GOVERNED_ALIASES:
            raise ValueError(f"Unsupported governed prompt alias: {alias}")
        if isinstance(evidence, GateResult):
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: "
                "gate evidence alone carries no template identity",
                remediation="Record an adopt DecisionRecord citing this gate "
                "with prompt_digest set at decision time, and promote with "
                "that record.",
            )
        if not isinstance(evidence, DecisionRecord):
            raise TypeError("evidence must be a DecisionRecord")
        if evidence.decision is not Decision.ADOPT:
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: "
                f"the cited decision is {evidence.decision.value!r}",
                remediation="Record an adopt decision backed by a passing "
                "gate before moving the production alias.",
            )
        bound_digest = evidence.prompt_digest
        if not bound_digest:
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: the "
                "adopt decision is not bound to any template content",
                remediation="Record the decision with "
                "prompt_digest=prompt_digest(template) for the evaluated "
                "template so promotion can verify the registry version it "
                "moves.",
            )
        # Content identity is not registry identity: two prompts can share a
        # template, so the decision must also name the exact prompt (and,
        # when recorded, the immutable version) it was made for.
        qualified = self.qualify(name)
        if not evidence.prompt_name:
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: the "
                "adopt decision names no prompt",
                remediation="Record the decision with "
                "prompt_name=manager.qualify(name) for the evaluated prompt "
                "so evidence for one prompt can never promote another.",
            )
        if evidence.prompt_name != qualified:
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: the "
                f"decision is bound to {evidence.prompt_name!r}, not "
                f"{qualified!r}",
                remediation="Promote the prompt the decision was recorded "
                "for, or record a new decision for this prompt.",
            )
        if evidence.prompt_version is None:
            # The digest hashes only the template: two immutable versions
            # can share it while differing in native configuration, so
            # version-unbound evidence could promote an unevaluated sibling.
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: the "
                "adopt decision is not bound to a registry version",
                remediation="Record the decision with prompt_version set to "
                "the evaluated registry version so evidence for one "
                "immutable version can never promote another that shares "
                "its template.",
            )
        if evidence.prompt_version != version:
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: the "
                f"decision is bound to version {evidence.prompt_version}, "
                f"not {version}",
                remediation="Promote the exact registry version the "
                "decision evaluated.",
            )
        # Every load_prompt flavor links the version to active lineage —
        # even the client-level one attaches it to the active experiment.
        # get_prompt_version is the only fetch with no lineage side
        # effects, so a rejected change never becomes associated evidence.
        registered = (
            self._client()
            .MlflowClient()
            .get_prompt_version(self.qualify(name), version)
        )
        template = getattr(registered, "template", None)
        if template is None:
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: "
                f"registry version {version} exposes no template to verify",
                remediation="Promote a registered prompt version whose "
                "template content the registry returns.",
            )
        observed_digest = prompt_digest(template)
        if observed_digest != bound_digest:
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: "
                f"version {version} has content digest "
                f"{observed_digest[:16]} but the evidence is bound to "
                f"{bound_digest[:16]}",
                remediation="Evaluate the exact registry version being "
                "promoted and bind its content digest to the decision "
                "evidence.",
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
        if alias not in _GOVERNED_ALIASES:
            raise ValueError(f"Unsupported governed prompt alias: {alias}")
        self._client().genai.set_prompt_alias(
            name=self.qualify(name),
            alias=alias,
            version=version,
        )

    def qualify(self, name: str) -> str:
        cleaned = str(name).strip()
        parts = cleaned.split(".")
        if len(parts) == 1:
            # Fail locally: a blank name would otherwise reach the registry
            # as the malformed 'catalog.schema.'.
            if not cleaned:
                raise ValueError("Prompt names must not be blank")
            if not fullmatch(_NAME_COMPONENT, cleaned):
                raise ValueError(
                    "Prompt names may contain only letters, digits, "
                    f"underscores, and hyphens; got {cleaned!r}"
                )
            catalog = _registry_qualifier("catalog", self.catalog)
            schema = _registry_qualifier("schema", self.schema)
            return f"{catalog}.{schema}.{cleaned}"
        if len(parts) == 3 and all(fullmatch(_NAME_COMPONENT, part) for part in parts):
            return cleaned
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


def _registry_qualifier(role: str, value: str) -> str:
    """Fail locally on unconfigured qualifiers instead of querying the
    registry for names like ``unset.unset.<name>``."""

    from aai_core.evaluation import _is_placeholder

    qualifier = str(value).strip()
    if not fullmatch(_NAME_COMPONENT, qualifier) or _is_placeholder(qualifier):
        raise ValueError(
            f"{role} must be a configured Unity Catalog qualifier; got "
            f"{value!r}. Set platform.catalog and platform.schema in "
            "aai-platform.yml before using the governed prompt registry."
        )
    return qualifier


def is_missing_prompt_error(error: Exception) -> bool:
    """True only when a registry error means the prompt or alias is absent.

    Authentication, permission, and transient registry failures return
    False — even when their message says "does not exist", the common
    non-disclosure wording — so callers seeding a first version or first
    promotion can fall back on absence without swallowing real failures.
    One shared predicate serves prompts and evaluation datasets alike.
    """

    from aai_core.evaluation import _is_missing_registry_error

    return _is_missing_registry_error(error)


def _prompt_tag_key(key: str) -> str:
    normalized = sub(r"[.,\-=/ :]+", "_", str(key)).strip("_")
    return f"aai_{normalized}"
