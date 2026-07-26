"""Prompt-driven assistant.

Aliases are deployment pointers: production loads the `production` alias,
everything else loads `development`. Evaluation NEVER uses an alias — it pins
an exact version (see evals/evaluate.py) so results stay reproducible.
"""

from __future__ import annotations

from aai_core import PlatformContext, bootstrap
from aai_core.tracing import traced
from app.config import PROMPT_NAME


class Assistant:
    def __init__(
        self,
        context: PlatformContext | None = None,
        *,
        prompt_version: int | None = None,
    ) -> None:
        self.context = context or bootstrap()
        self.model = self.context.providers.model("general-chat")
        if prompt_version is not None:
            self.prompt = self.context.prompts.load(PROMPT_NAME, version=prompt_version)
        else:
            alias = (
                "production"
                if self.context.settings.resource.environment in {"prod", "production"}
                else "development"
            )
            self.prompt = self.context.prompts.load(PROMPT_NAME, alias=alias)

    @traced(name="assistant.ask", span_type="CHAIN")
    def ask(self, question: str) -> str:
        if not question.strip():
            raise ValueError("question must be a non-empty string")
        messages = self.prompt.format(question=question)
        response = self.model.generate(messages, temperature=0.2)
        return response.content
