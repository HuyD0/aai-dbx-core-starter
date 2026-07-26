"""Models-from-code serving entry point (MLflow ResponsesAgent).

Logged by scripts/deploy_serving.py with the app package as code_paths, the
project's aai-platform.yml as model_config, and the pinned runtime
requirements — so the model loads identically in Model Serving and in a
developer checkout. Keep this file thin: the real agent lives in src/app and
is unit-tested; this wrapper only adapts the Responses API shape.
"""

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.models import ModelConfig
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from aai_core import PlatformContext, bootstrap
from aai_core.agents import AgentRequest
from app.agent import ToolAgent
from app.messages import response_message_text

_PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "aai-platform.yml"


def _load_platform_context() -> PlatformContext:
    """Bootstrap from the logged model_config in serving, or the project's
    aai-platform.yml in a local checkout (config holds references only —
    never secret values)."""

    development = str(_PROJECT_CONFIG) if _PROJECT_CONFIG.is_file() else None
    config = ModelConfig(development_config=development)
    document = {}
    for section in ("platform", "providers", "secrets"):
        try:
            document[section] = config.get(section)
        except KeyError:
            continue
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yml", delete=False, encoding="utf-8"
    ) as stream:
        yaml.safe_dump(document, stream)
        rendered = stream.name
    return bootstrap(rendered)


class ServedToolAgent(ResponsesAgent):
    def __init__(self):
        self._agent = ToolAgent(_load_platform_context())

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # ResponsesAgent auto-traces predict() before this body runs, so its
        # initial span inputs contain the complete request. Replace them
        # immediately with a safe placeholder: if text normalization rejects
        # the request, context.user_id and unsupported content still cannot be
        # persisted on the failed trace.
        span = mlflow.get_current_active_span()
        if span is not None:
            span.set_inputs({"input": []})

        messages = [
            {"role": item.role, "content": response_message_text(item)}
            for item in request.input
            if getattr(item, "role", None) in {"user", "assistant"}
        ]
        context = request.context
        conversation_id = _context_value(context, "conversation_id")
        if span is not None:
            trace_inputs: dict[str, Any] = {"input": messages}
            if conversation_id:
                trace_inputs["context"] = {"conversation_id": conversation_id}
            span.set_inputs(trace_inputs)

        response = self._agent.invoke(
            AgentRequest(
                messages=messages,
                # Group turns with an opaque conversation id. This template
                # intentionally does not propagate context.user_id into the
                # traced application request.
                session_id=conversation_id,
            )
        )
        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(text=response.content, id="agent-answer")
            ],
            custom_outputs=dict(response.metadata),
        )


def _context_value(context: Any, field: str) -> str | None:
    if context is None:
        return None
    value = (
        context.get(field)
        if isinstance(context, Mapping)
        else getattr(context, field, None)
    )
    return value if isinstance(value, str) and value else None


mlflow.models.set_model(ServedToolAgent())
