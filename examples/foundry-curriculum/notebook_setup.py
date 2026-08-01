"""Shared, config-driven setup for the Microsoft Foundry curriculum notebooks."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_PROJECT_PATH = re.compile(r"^/api/projects/[^/]+/?$")
_PLACEHOLDER_PREFIXES = ("replace-", "<", "unset", "todo")


@dataclass(frozen=True)
class FoundryNotebookSession:
    """Validated notebook context without credentials or mutable global state."""

    curriculum_root: Path
    config_path: Path
    logical_model: str
    project_endpoint: str
    deployment: str
    context: Any

    @property
    def connected_ready(self) -> bool:
        return not _is_placeholder(self.deployment)

    def safe_summary(self) -> dict[str, str | bool]:
        """Return non-secret configuration fields safe to display in a notebook."""

        return {
            "config": str(self.config_path),
            "logical_model": self.logical_model,
            "provider": "foundry",
            "project_endpoint": self.project_endpoint,
            "deployment": self.deployment,
            "azure_identity": str(self.context.settings.azure_identity),
            "connected_ready": self.connected_ready,
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
    return FoundryNotebookSession(
        curriculum_root=root,
        config_path=resolved_config,
        logical_model=logical_model,
        project_endpoint=endpoint,
        deployment=deployment,
        context=context,
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
            "Set providers.models.foundry-chat.deployment in the selected config "
            "before making a connected call."
        )
    if not prompt.strip():
        raise ValueError("prompt must not be blank")
    model = session.context.providers.model(session.logical_model)
    return model.native_client.responses.create(
        model=session.deployment,
        input=prompt,
    )


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
