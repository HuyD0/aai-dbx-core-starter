"""Prompt-driven assistant.

Aliases are deployment pointers: production loads the `production` alias,
everything else loads `development`. Evaluation NEVER uses an alias — it pins
an exact version (see evals/evaluate.py) so results stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from aai_core import PlatformContext, bootstrap
from aai_core.tracing import traced
from app.config import PROMPT_NAME


@dataclass(frozen=True)
class PromptLimits:
    max_question_chars: int = 8_000
    max_output_tokens: int = 1_024
    max_output_chars: int = 8_192

    def __post_init__(self) -> None:
        if any(value <= 0 for value in vars(self).values()):
            raise ValueError("prompt limits must all be positive")


DEFAULT_PROMPT_LIMITS = PromptLimits()


class Assistant:
    def __init__(
        self,
        context: PlatformContext | None = None,
        *,
        prompt_version: int | None = None,
        limits: PromptLimits = DEFAULT_PROMPT_LIMITS,
    ) -> None:
        self.context = context or bootstrap()
        self.model = self.context.providers.model("general-chat")
        self.limits = limits
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
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if len(question) > self.limits.max_question_chars:
            raise ValueError(
                f"question exceeds the {self.limits.max_question_chars}-character bound"
            )
        messages = self.prompt.format(question=question)
        response = self.model.generate(
            messages,
            temperature=0.2,
            max_tokens=self.limits.max_output_tokens,
        )
        if not response.content.strip():
            raise RuntimeError("model returned an empty response")
        if len(response.content) > self.limits.max_output_chars:
            raise RuntimeError(
                "model response exceeded the centrally configured "
                f"{self.limits.max_output_chars}-character bound"
            )
        return response.content
