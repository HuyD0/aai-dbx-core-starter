"""Models-from-code serving entry point (MLflow ResponsesAgent).

Logged by scripts/deploy_serving.py with the app package as code_paths, the
project's aai-platform.yml as model_config, and the pinned runtime
requirements — so the model loads identically in Model Serving and in a
developer checkout. Keep this file thin: the real agent lives in src/app and
is unit-tested; this wrapper only adapts the Responses API shape.
"""

import tempfile
from pathlib import Path

import mlflow
import yaml
from mlflow.models import ModelConfig
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from aai_core import PlatformContext, bootstrap
from aai_core.agents import AgentRequest
from app.agent import ToolAgent

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
        messages = [
            {"role": item.role, "content": item.content}
            for item in request.input
            if getattr(item, "role", None)
        ]
        response = self._agent.invoke(AgentRequest(messages=messages))
        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(text=response.content, id="agent-answer")
            ],
            custom_outputs=dict(response.metadata),
        )


mlflow.models.set_model(ServedToolAgent())
