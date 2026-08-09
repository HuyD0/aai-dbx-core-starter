"""Governed MLflow Prompt Registry operations."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from re import fullmatch, sub
from typing import Any

from aai_core.evaluation import _NAME_COMPONENT
from aai_core.exceptions import AaiCoreError
from aai_core.tags import ResourceContext


class PromptPromotionError(AaiCoreError):
    """A prompt alias move was refused for lack of release evidence."""

    code = "aai_core.prompts.promotion_blocked"


# 'candidate' remains accepted only as the deprecated alias name that
# set_alias() warns about; it is not lifecycle vocabulary.
_GOVERNED_ALIASES = {"development", "validation", "candidate", "production"}


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
        decision_run_id: str | None = None,
        evidence: Any | None = None,
        alias: str = "production",
    ) -> None:
        """Move an alias only on a verified, persisted adopt decision.

        ``decision_run_id`` names the finished MLflow decision run written by
        :func:`aai_core.decisions.record_decision`. Promotion downloads its
        strict ``decision.json`` artifact and verifies the canonical digest,
        lifecycle tags, gate metrics, run purpose, status, and identity before
        inspecting the target prompt. The persisted record must be an adopt
        decision bound to the exact qualified prompt, immutable version, and
        template digest being promoted.

        ``evidence`` is optional in-memory evidence for callers that already
        hold the record. It is validated locally first for useful refusal
        messages, then must match the persisted artifact exactly; it can never
        replace ``decision_run_id``. A bare gate is refused because it carries
        no prompt identity. Anything less raises :class:`PromptPromotionError`
        and leaves the alias untouched. Registry access controls and protected
        main remain the authorization boundary; this method enforces the
        persisted release process before that authorized alias move.
        """

        from aai_core.decisions import (
            DecisionEvidenceError,
            decision_digest,
            load_decision,
        )

        # The cheapest, fully local check comes first: an alias typo must
        # fail deterministically, never as a network error from the
        # evidence or prompt verification fetches below.
        if alias not in _GOVERNED_ALIASES:
            raise ValueError(f"Unsupported governed prompt alias: {alias}")
        qualified = self.qualify(name)
        local_record = None
        if evidence is not None:
            local_record, _ = _bound_prompt_decision(
                evidence,
                name=name,
                qualified=qualified,
                version=version,
                alias=alias,
            )
        if decision_run_id is None:
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: the "
                "decision is not backed by a persisted decision run",
                remediation="Persist the decision with record_decision(), then "
                "pass the returned decision_run_id to promote().",
            )
        try:
            persisted = load_decision(
                decision_run_id,
                mlflow_module=self._client(),
            )
        except DecisionEvidenceError as error:
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: {error}",
                remediation="Use the finished decision run produced by "
                "record_decision() for this exact evaluated prompt version.",
            ) from error
        if local_record is not None and decision_digest(
            local_record
        ) != decision_digest(persisted):
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: the "
                "supplied in-memory evidence differs from decision.json",
                remediation="Use the DecisionRecord persisted in the cited "
                "decision run, or cite the run that persisted this record.",
            )
        _, bound_digest = _bound_prompt_decision(
            persisted,
            name=name,
            qualified=qualified,
            version=version,
            alias=alias,
        )
        # Every load_prompt flavor links the version to active lineage —
        # even the client-level one attaches it to the active experiment.
        # get_prompt_version is the only fetch with no lineage side
        # effects, so a rejected change never becomes associated evidence.
        try:
            registered = (
                self._client().MlflowClient().get_prompt_version(qualified, version)
            )
        except Exception as error:
            # A version that does not exist is invalid promotion input, not
            # a provider outage, so it joins the same guarded refusal as the
            # evidence checks above rather than escaping as a raw registry
            # error. Permission and transport failures still propagate.
            if not is_missing_prompt_error(error):
                raise
            raise PromptPromotionError(
                f"Refusing to move alias {alias!r} for prompt {name!r}: "
                f"registry version {version} does not exist",
                remediation="Promote a registered version of this prompt; "
                "register the evaluated template with ensure_version() first.",
            ) from error
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
        from aai_core.evaluation import _is_placeholder

        # str() would make None the component "None" and 123 the component
        # "123", both valid-looking, so register()/load()/ensure_version()/
        # set_alias() would address a real (wrong) prompt in the registry.
        if not isinstance(name, str):
            raise TypeError(f"Prompt names must be strings; got {type(name).__name__}")
        cleaned = name.strip()
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
            if _is_placeholder(cleaned):
                raise ValueError(f"Prompt names must not be placeholders; got {name!r}")
            catalog = _registry_qualifier("catalog", self.catalog)
            schema = _registry_qualifier("schema", self.schema)
            return f"{catalog}.{schema}.{cleaned}"
        if len(parts) == 3 and all(fullmatch(_NAME_COMPONENT, part) for part in parts):
            # Explicit qualification is not an escape hatch from the
            # placeholder vocabulary: unset.app.prompt is as unconfigured
            # as the derived form.
            if any(_is_placeholder(part) for part in parts):
                raise ValueError(
                    "Qualified prompt names must not contain placeholder "
                    f"components; got {name!r}"
                )
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


def _bound_prompt_decision(
    evidence: Any,
    *,
    name: str,
    qualified: str,
    version: int,
    alias: str,
) -> tuple[Any, str]:
    """Validate prompt bindings shared by local and persisted evidence."""

    from aai_core.decisions import Decision, DecisionRecord
    from aai_core.evaluation import GateResult

    if isinstance(evidence, GateResult):
        raise PromptPromotionError(
            f"Refusing to move alias {alias!r} for prompt {name!r}: "
            "gate evidence alone carries no template identity",
            remediation="Record an adopt DecisionRecord citing this gate with "
            "prompt identity, persist it with record_decision(), and promote "
            "with that decision run.",
        )
    if not isinstance(evidence, DecisionRecord):
        raise TypeError("evidence must be a DecisionRecord")
    if evidence.decision is not Decision.ADOPT:
        raise PromptPromotionError(
            f"Refusing to move alias {alias!r} for prompt {name!r}: "
            f"the cited decision is {evidence.decision.value!r}",
            remediation="Record an adopt decision backed by a passing gate "
            "before moving the production alias.",
        )
    bound_digest = evidence.prompt_digest
    if not bound_digest:
        raise PromptPromotionError(
            f"Refusing to move alias {alias!r} for prompt {name!r}: the "
            "adopt decision is not bound to any template content",
            remediation="Record the decision with "
            "prompt_digest=prompt_digest(template) for the evaluated template "
            "so promotion can verify the registry version it moves.",
        )
    # Content identity is not registry identity: two prompts can share a
    # template, so the decision must also name the exact prompt and immutable
    # version it was made for.
    if not evidence.prompt_name:
        raise PromptPromotionError(
            f"Refusing to move alias {alias!r} for prompt {name!r}: the "
            "adopt decision names no prompt",
            remediation="Record the decision with "
            "prompt_name=manager.qualify(name) for the evaluated prompt so "
            "evidence for one prompt can never promote another.",
        )
    if evidence.prompt_name != qualified:
        raise PromptPromotionError(
            f"Refusing to move alias {alias!r} for prompt {name!r}: the "
            f"decision is bound to {evidence.prompt_name!r}, not {qualified!r}",
            remediation="Promote the prompt the decision was recorded for, or "
            "record a new decision for this prompt.",
        )
    if evidence.prompt_version is None:
        # The digest hashes only the template: two immutable versions can
        # share it while differing in native configuration.
        raise PromptPromotionError(
            f"Refusing to move alias {alias!r} for prompt {name!r}: the "
            "adopt decision is not bound to a registry version",
            remediation="Record the decision with prompt_version set to the "
            "evaluated registry version so evidence for one immutable version "
            "can never promote another that shares its template.",
        )
    if evidence.prompt_version != version:
        raise PromptPromotionError(
            f"Refusing to move alias {alias!r} for prompt {name!r}: the "
            f"decision is bound to version {evidence.prompt_version}, not {version}",
            remediation="Promote the exact registry version the decision evaluated.",
        )
    return evidence, bound_digest


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

    # str() would make None the qualifier "None" and 123 the qualifier
    # "123", both of which satisfy _NAME_COMPONENT and would address a
    # real (wrong) registry namespace.
    if not isinstance(value, str):
        raise TypeError(
            f"{role} must be a string Unity Catalog qualifier; got "
            f"{type(value).__name__}"
        )
    qualifier = value.strip()
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
