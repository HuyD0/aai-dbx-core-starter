"""Shared, config-driven setup for the Microsoft Foundry curriculum notebooks."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, ConfigDict, Field

_PROJECT_PATH = re.compile(r"^/api/projects/[^/]+/?$")
_PLACEHOLDER_PREFIXES = ("replace-", "<", "unset", "todo")
_RESOURCE_NAME = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$"
_APP_INSIGHTS_RESOURCE_ID = (
    r"^(?:replace-with-application-insights-resource-id|"
    r"/subscriptions/[0-9A-Fa-f-]{36}/resourceGroups/[A-Za-z0-9._()-]+/"
    r"providers/(?:Microsoft\.Insights|microsoft\.insights)/"
    r"components/[A-Za-z0-9._()-]+)$"
)


class FoundryAgentSettings(BaseModel):
    """Immutable identifiers for a pre-provisioned Foundry agent version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        default="replace-with-agent-name",
        min_length=1,
        max_length=256,
        pattern=_RESOURCE_NAME,
    )
    version: str = Field(
        default="replace-with-agent-version",
        min_length=1,
        max_length=256,
        pattern=_RESOURCE_NAME,
    )
    id: str = Field(
        default="replace-with-agent-id",
        min_length=1,
        max_length=256,
        pattern=_RESOURCE_NAME,
    )


class FoundryEvaluationSettings(BaseModel):
    """Immutable identifiers for cloud evaluation resources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator_model: str = Field(
        default="replace-with-evaluator-model-deployment",
        min_length=1,
        max_length=256,
        pattern=_RESOURCE_NAME,
    )


class FoundryMemorySettings(BaseModel):
    """Reference to a memory store provisioned by the platform owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    store_name: str = Field(
        default="replace-with-memory-store-name",
        min_length=1,
        max_length=256,
        pattern=_RESOURCE_NAME,
    )


class FoundryA2ASettings(BaseModel):
    """References for a pre-enabled remote A2A agent and project connection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    remote_agent_name: str = Field(
        default="replace-with-remote-agent-name",
        min_length=1,
        max_length=256,
        pattern=_RESOURCE_NAME,
    )
    connection_name: str = Field(
        default="replace-with-a2a-connection-name",
        min_length=1,
        max_length=256,
        pattern=_RESOURCE_NAME,
    )
    protocol_version: Literal["1.0"] = "1.0"


class FoundryObservabilitySettings(BaseModel):
    """Non-secret resource reference used to resolve legacy telemetry links."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_insights_resource_id: str = Field(
        default="replace-with-application-insights-resource-id",
        min_length=1,
        max_length=1024,
        pattern=_APP_INSIGHTS_RESOURCE_ID,
    )


class FoundryLabSettings(BaseModel):
    """Strict boundary for optional advanced-lab identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: FoundryAgentSettings = Field(default_factory=FoundryAgentSettings)
    evaluation: FoundryEvaluationSettings = Field(
        default_factory=FoundryEvaluationSettings
    )
    memory: FoundryMemorySettings = Field(default_factory=FoundryMemorySettings)
    a2a: FoundryA2ASettings = Field(default_factory=FoundryA2ASettings)
    observability: FoundryObservabilitySettings = Field(
        default_factory=FoundryObservabilitySettings
    )


@dataclass(frozen=True)
class FoundryNotebookSession:
    """Validated notebook context without credentials or mutable global state."""

    curriculum_root: Path
    config_path: Path
    logical_model: str
    project_endpoint: str
    deployment: str
    context: Any
    labs: FoundryLabSettings = field(default_factory=FoundryLabSettings)

    @property
    def connected_ready(self) -> bool:
        deployment_ready = not _is_placeholder(self.deployment)
        endpoint_ready = not _is_project_endpoint_placeholder(self.project_endpoint)
        return deployment_ready and endpoint_ready

    @property
    def agent_ready(self) -> bool:
        return self.connected_ready and all(
            not _is_placeholder(value)
            for value in (self.labs.agent.name, self.labs.agent.version)
        )

    @property
    def trace_ready(self) -> bool:
        return self.agent_ready and not _is_placeholder(self.labs.agent.id)

    @property
    def evaluation_ready(self) -> bool:
        return self.agent_ready and not _is_placeholder(
            self.labs.evaluation.evaluator_model
        )

    @property
    def a2a_ready(self) -> bool:
        return self.connected_ready and not _is_placeholder(
            self.labs.a2a.remote_agent_name
        )

    @property
    def observability_ready(self) -> bool:
        return not _is_placeholder(
            self.labs.observability.application_insights_resource_id
        )

    @property
    def a2a_base_url(self) -> str:
        agent_name = quote(self.labs.a2a.remote_agent_name, safe="-._:")
        return f"{self.project_endpoint}/agents/{agent_name}/endpoint/protocols/a2a"

    @property
    def a2a_agent_card_url(self) -> str:
        return f"{self.a2a_base_url}/agentCard/v{self.labs.a2a.protocol_version}"

    def agent_reference(self) -> dict[str, str]:
        """Return an immutable agent reference without inventing a latest version."""

        if not self.agent_ready:
            raise RuntimeError(
                "Configure foundry.agent.name and foundry.agent.version before "
                "invoking an agent."
            )
        reference = {
            "type": "agent_reference",
            "name": self.labs.agent.name,
            "version": self.labs.agent.version,
        }
        return reference

    def safe_summary(self) -> dict[str, str | bool]:
        """Return non-secret configuration fields safe to display in a notebook."""

        return {
            "config": str(self.config_path),
            "logical_model": self.logical_model,
            "provider": "foundry",
            "project_endpoint": self.project_endpoint,
            "deployment": self.deployment,
            "agent_name": self.labs.agent.name,
            "agent_version": self.labs.agent.version,
            "a2a_protocol": self.labs.a2a.protocol_version,
            "azure_identity": str(self.context.settings.azure_identity),
            "connected_ready": self.connected_ready,
            "agent_ready": self.agent_ready,
            "trace_ready": self.trace_ready,
            "evaluation_ready": self.evaluation_ready,
            "a2a_ready": self.a2a_ready,
            "observability_ready": self.observability_ready,
        }


def find_curriculum_root(start: str | Path | None = None) -> Path:
    """Locate this curriculum from a repository root or notebook directory."""

    base = Path(start).expanduser().resolve() if start is not None else Path.cwd()
    for directory in (base, *base.parents):
        candidates = (
            directory,
            directory / "examples" / "foundry-curriculum",
        )
        for candidate in candidates:
            if (candidate / "notebook_setup.py").is_file() and (
                candidate / "notebooks"
            ).is_dir():
                return candidate
    raise FileNotFoundError(
        "Could not find examples/foundry-curriculum. Open the repository as the "
        "notebook workspace and restart the kernel."
    )


def load_offline_labs(curriculum_root: str | Path | None = None) -> ModuleType:
    """Load the typed offline lab support without making a provider request."""

    root = find_curriculum_root(curriculum_root)
    module_name = "foundry_curriculum_offline_labs"
    spec = importlib.util.spec_from_file_location(module_name, root / "offline_labs.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the Foundry offline lab support module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_config_path(
    curriculum_root: Path,
    config_path: str | Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """Resolve explicit path, standard SDK override, then local dev config."""

    environment = environ if environ is not None else os.environ
    configured = config_path or environment.get("AAI_PLATFORM_CONFIG")
    path = (
        Path(configured).expanduser()
        if configured
        else curriculum_root / "config" / "aai-platform.dev.yml"
    )
    resolved = path.resolve()
    if not resolved.is_file():
        example = curriculum_root / "config" / "aai-platform.dev.example.yml"
        raise FileNotFoundError(
            f"No curriculum configuration found at {resolved}. Copy {example} "
            "to config/aai-platform.dev.yml and set the project endpoint and "
            "model deployment."
        )
    return resolved


def load_session(
    curriculum_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    logical_model: str = "foundry-chat",
    environ: dict[str, str] | None = None,
    bootstrap_fn: Any | None = None,
) -> FoundryNotebookSession:
    """Load and validate the Foundry project endpoint without making a request."""

    root = find_curriculum_root(curriculum_root)
    resolved_config = resolve_config_path(
        root,
        config_path,
        environ=environ,
    )
    if bootstrap_fn is None:
        from aai_core import bootstrap

        bootstrap_fn = bootstrap
    context = bootstrap_fn(resolved_config)
    model_config = dict(context.settings.models.get(logical_model, {}))
    if not model_config:
        raise ValueError(
            f"Configuration has no providers.models.{logical_model} entry."
        )
    provider = str(model_config.get("provider", ""))
    if provider != "foundry":
        raise ValueError(
            f"providers.models.{logical_model}.provider must be 'foundry'; "
            f"found {provider!r}."
        )
    endpoint = _validate_project_endpoint(str(model_config.get("endpoint", "")))
    deployment = str(model_config.get("deployment", "")).strip()
    if not deployment:
        raise ValueError(
            f"providers.models.{logical_model}.deployment must be configured."
        )
    raw_settings = getattr(context.settings, "raw", {})
    foundry_document = dict(raw_settings.get("foundry", {}))
    labs = FoundryLabSettings.model_validate(foundry_document)
    return FoundryNotebookSession(
        curriculum_root=root,
        config_path=resolved_config,
        logical_model=logical_model,
        project_endpoint=endpoint,
        deployment=deployment,
        context=context,
        labs=labs,
    )


def create_text_response(
    session: FoundryNotebookSession,
    prompt: str,
    *,
    allow_network: bool = False,
) -> Any:
    """Make one explicit Responses API call through the configured project."""

    if not allow_network:
        raise RuntimeError(
            "Connected calls are disabled. Pass allow_network=True only after "
            "reviewing the prompt, deployment, expected cost, and data policy."
        )
    if not session.connected_ready:
        raise RuntimeError(
            f"Set providers.models.{session.logical_model}.endpoint to a real "
            "Foundry project endpoint and its deployment to a real model "
            "deployment before making a connected call."
        )
    if not prompt.strip():
        raise ValueError("prompt must not be blank")
    model = session.context.providers.model(session.logical_model)
    return model.native_client.responses.create(
        model=session.deployment,
        input=prompt,
    )


def create_agent_response(
    session: FoundryNotebookSession,
    prompt: str,
    *,
    allow_network: bool = False,
) -> tuple[Any, Any]:
    """Create one conversation and invoke one immutable Foundry agent version."""

    if not allow_network:
        raise RuntimeError(
            "Connected calls are disabled. Pass allow_network=True only after "
            "reviewing the prompt, agent version, expected cost, and data policy."
        )
    if not session.agent_ready:
        raise RuntimeError(
            "Set a real project endpoint plus foundry.agent.name and "
            "foundry.agent.version before making an agent call."
        )
    if not prompt.strip():
        raise ValueError("prompt must not be blank")
    model = session.context.providers.model(session.logical_model)
    client = model.native_client
    conversation = client.conversations.create()
    response = client.responses.create(
        conversation=conversation.id,
        input=prompt,
        extra_body={"agent_reference": session.agent_reference()},
    )
    return conversation, response


def _validate_project_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    if not endpoint:
        raise ValueError("Foundry project endpoint must not be blank.")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Foundry project endpoint must be an absolute HTTPS URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "Foundry project endpoint must not contain credentials, query text, "
            "or a fragment."
        )
    if not parsed.hostname.endswith(".services.ai.azure.com"):
        raise ValueError(
            "Foundry project endpoint host must end with .services.ai.azure.com."
        )
    if not _PROJECT_PATH.fullmatch(parsed.path):
        raise ValueError(
            "Use the project endpoint ending in /api/projects/<project-name>, "
            "not an account-level endpoint."
        )
    return endpoint


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return not normalized or normalized.startswith(_PLACEHOLDER_PREFIXES)


def _is_project_endpoint_placeholder(value: str) -> bool:
    parsed = urlsplit(value.strip())
    account_name = (parsed.hostname or "").partition(".")[0]
    project_name = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return _is_placeholder(account_name) or _is_placeholder(project_name)
