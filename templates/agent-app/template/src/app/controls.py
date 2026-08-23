"""Central, conservative execution bounds for the starter agent."""

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


class AgentLimits(BaseModel):
    """One place to review the agent's cost and latency envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_tool_turns: int = Field(default=6, ge=1, le=20)
    max_tool_calls_per_turn: int = Field(default=4, ge=1, le=20)
    max_total_tool_calls: int = Field(default=12, ge=1, le=100)
    tool_timeout_seconds: float = Field(default=10.0, gt=0, le=60.0)
    max_tool_output_chars: int = Field(default=8_000, ge=1, le=100_000)
    max_input_messages: int = Field(default=32, ge=1, le=200)
    max_message_chars: int = Field(default=8_000, ge=1, le=100_000)
    max_total_input_chars: int = Field(default=32_000, ge=1, le=500_000)
    max_output_tokens: int = Field(default=1024, ge=1, le=8192)
    max_stream_output_chars: int = Field(default=32_000, ge=1, le=500_000)
    request_deadline_seconds: float = Field(default=60.0, gt=0, le=300.0)

    @property
    def digest(self) -> str:
        serialized = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()


DEFAULT_AGENT_LIMITS = AgentLimits()
