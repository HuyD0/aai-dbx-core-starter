"""Central runtime limits for the generated analytics application."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalyticsLimits:
    max_question_chars: int = 8_000
    max_agent_turns: int = 8
    max_tool_calls_per_turn: int = 4
    max_total_tool_calls: int = 12
    request_deadline_seconds: float = 180.0
    tool_timeout_seconds: float = 60.0
    max_result_rows: int = 100
    max_tool_output_chars: int = 32 * 1024
    max_output_tokens: int = 2_048

    def __post_init__(self) -> None:
        numeric = {
            name: value
            for name, value in vars(self).items()
            if isinstance(value, int | float)
        }
        if any(value <= 0 for value in numeric.values()):
            raise ValueError("analytics limits must all be positive")
        if self.max_tool_calls_per_turn > self.max_total_tool_calls:
            raise ValueError(
                "per-turn tool calls cannot exceed the total tool-call limit"
            )
        if self.max_result_rows > 100:
            raise ValueError("max_result_rows cannot exceed the hard safety cap of 100")


DEFAULT_ANALYTICS_LIMITS = AnalyticsLimits()
