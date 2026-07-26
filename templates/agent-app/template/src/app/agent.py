"""Tool-using agent: governed prompt, SDK tool loop, structured final answer."""

from __future__ import annotations

from aai_core import PlatformContext, bootstrap
from aai_core.agents import AgentRequest, AgentResponse, run_tool_loop
from aai_core.structured import generate_structured
from aai_core.tracing import set_trace_session, traced
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

    def invoke(self, request: AgentRequest) -> AgentResponse:
        # MLflow serializes traced function arguments. Deliberately omit
        # user_id and arbitrary metadata from the traced request; the caller
        # may pass them for application-owned use, but this template does not
        # have governance approval to record them.
        trace_request = AgentRequest(
            messages=request.messages,
            session_id=request.session_id,
        )
        return self._invoke_traced(trace_request)

    @traced(name="agent.invoke", span_type="AGENT")
    def _invoke_traced(self, request: AgentRequest) -> AgentResponse:
        if request.session_id:
            set_trace_session(request.session_id)
        messages = _conversation_messages(self.prompt, request)
        loop = run_tool_loop(self.model, messages, self.registry)
        structured = generate_structured(
            self.model,
            [
                *loop.transcript,
                {
                    "role": "user",
                    "content": "Provide the final answer as the requested JSON object.",
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


def _conversation_messages(prompt, request: AgentRequest) -> list[dict]:
    """Combine the governed system prompt with request conversation history.

    Only user and assistant messages cross the serving boundary. A caller
    cannot replace the registered prompt by inserting a system message into
    the request. Governed few-shot messages remain intact; prior history is
    inserted before the prompt's governed, formatted final user turn.
    """

    history: list[dict] = []
    for message in request.messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("AgentRequest user/assistant messages must contain text")
        history.append({"role": role, "content": content})
    if (
        not history
        or history[-1]["role"] != "user"
        or not history[-1]["content"].strip()
    ):
        raise ValueError("AgentRequest must end with a non-empty user message")
    question = history[-1]["content"]

    formatted = [dict(message) for message in prompt.format(question=question)]
    if not formatted or formatted[-1].get("role") != "user":
        raise ValueError("The registered prompt must end with a user message")
    if not isinstance(formatted[-1].get("content"), str):
        raise TypeError("The registered prompt's final user message must contain text")
    governed = formatted[:-1]
    if not any(message.get("role") == "system" for message in governed):
        raise ValueError("The registered prompt requires a system message")
    return [*governed, *history[:-1], formatted[-1]]
