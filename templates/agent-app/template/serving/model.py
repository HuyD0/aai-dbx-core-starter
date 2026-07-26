"""Models-from-code serving entry point (MLflow ResponsesAgent).

Logged by scripts/deploy_serving.py via mlflow.pyfunc.log_model(
python_model="serving/model.py", resources=...). Keep this file thin: the
real agent lives in src/app and is unit-tested; this wrapper only adapts the
Responses API shape.
"""

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from aai_core.agents import AgentRequest
from app.agent import ToolAgent


class ServedToolAgent(ResponsesAgent):
    def __init__(self):
        self._agent = ToolAgent()

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
