"""Tool-using agent: governed prompt, SDK tool loop, structured final answer."""

from __future__ import annotations

from aai_core import PlatformContext, bootstrap
from aai_core.agents import AgentRequest, AgentResponse, run_tool_loop
from aai_core.structured import generate_structured
from aai_core.tracing import traced
from app.config import PROMPT_NAME
from app.schemas import FINAL_ANSWER_SCHEMA
from app.tools import build_registry


class ToolAgent:
    def __init__(
        self,
        context: PlatformContext | None = None,
        *,
        prompt_version: int | None = None,
    ) -> None:
        self.context = context or bootstrap()
        self.model = self.context.providers.model("general-chat")
        self.registry = build_registry()
        if prompt_version is not None:
            self.prompt = self.context.prompts.load(PROMPT_NAME, version=prompt_version)
        else:
            alias = (
                "production"
                if self.context.settings.resource.environment in {"prod", "production"}
                else "development"
            )
            self.prompt = self.context.prompts.load(PROMPT_NAME, alias=alias)

    @traced(name="agent.invoke", span_type="AGENT")
    def invoke(self, request: AgentRequest) -> AgentResponse:
        question = _latest_user_message(request)
        messages = self.prompt.format(question=question)
        loop = run_tool_loop(self.model, messages, self.registry)
        structured = generate_structured(
            self.model,
            [
                *loop.transcript,
                {
                    "role": "user",
                    "content": "Provide the final answer as the requested "
                    "JSON object.",
                },
            ],
            json_schema=FINAL_ANSWER_SCHEMA,
        )
        return AgentResponse(
            content=structured["answer"],
            metadata={
                "confidence": structured.get("confidence"),
                "tools_used": list(loop.tool_names),
                "model": loop.response.model,
            },
        )


def _latest_user_message(request: AgentRequest) -> str:
    for message in reversed(list(request.messages)):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise ValueError("AgentRequest requires a non-empty user message")
