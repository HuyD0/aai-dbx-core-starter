"""Structured-output contracts for the agent's final answers."""

from pydantic import BaseModel, ConfigDict, Field


class FinalAnswer(BaseModel):
    """Validated boundary shared by the provider schema and application."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    answer: str = Field(description="The final answer for the user.")
    confidence: float = Field(
        ge=0,
        le=1,
        description="Self-assessed confidence in the answer.",
    )
    tools_used: tuple[str, ...] = Field(
        default=(),
        description="Names of the tools consulted for this answer.",
    )
