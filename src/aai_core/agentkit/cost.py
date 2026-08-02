"""Judge-call cost estimation — shown BEFORE the run, never after.

Deterministic arithmetic over the actual dataset: rows x judge scorers,
with a character-count token heuristic. Smoke is nearly free by
construction (code scorers only). Dollar figures appear only when the
project configures its negotiated rate — the SDK ships no price table to
go stale.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field, field_serializer, field_validator

from aai_core.agentkit.catalog import ScorerPlan
from aai_core.agentkit.errors import BudgetExceededError
from aai_core.contracts import ContractModel, freeze_value, thaw_value

_ASSUMED_JUDGE_OUTPUT_TOKENS = 150
_CHARS_PER_TOKEN = 4


class CostEstimate(ContractModel):
    rows: int = Field(ge=0)
    judge_scorers: tuple[str, ...] = ()
    judge_calls: int = Field(ge=0)
    mean_row_tokens: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
    estimated_usd: float | None = None
    calls_by_scorer: Mapping[str, int] = Field(default_factory=dict)

    @field_validator("calls_by_scorer", mode="after")
    @classmethod
    def freeze_calls(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        return freeze_value(value)

    @field_serializer("calls_by_scorer")
    def serialize_calls(self, value: Mapping[str, int]) -> dict[str, int]:
        return thaw_value(value)


def estimate(
    rows: Sequence[Mapping[str, Any]],
    plan: ScorerPlan,
    *,
    price_per_1m_tokens: float | None = None,
) -> CostEstimate:
    """Estimate judge calls and tokens for scoring ``rows`` under ``plan``."""

    judge_specs = plan.judge_specs
    row_count = len(rows)
    mean_row_tokens = _mean_row_tokens(rows)
    total_tokens = 0
    calls_by_scorer: dict[str, int] = {}
    for spec in judge_specs:
        calls_by_scorer[spec.name] = row_count
        total_tokens += row_count * (
            spec.judge_overhead_tokens + mean_row_tokens + _ASSUMED_JUDGE_OUTPUT_TOKENS
        )
    judge_calls = row_count * len(judge_specs)
    estimated_usd = None
    if price_per_1m_tokens is not None and judge_calls:
        estimated_usd = round(total_tokens / 1_000_000 * price_per_1m_tokens, 4)
    return CostEstimate(
        rows=row_count,
        judge_scorers=tuple(spec.name for spec in judge_specs),
        judge_calls=judge_calls,
        mean_row_tokens=mean_row_tokens,
        estimated_tokens=total_tokens,
        estimated_usd=estimated_usd,
        calls_by_scorer=calls_by_scorer,
    )


def enforce_budget(cost: CostEstimate, *, max_judge_calls: int | None) -> None:
    """Abort BEFORE any judge call when the estimate exceeds the budget."""

    if max_judge_calls is not None and cost.judge_calls > max_judge_calls:
        raise BudgetExceededError(
            f"this run would make {cost.judge_calls} judge calls; "
            f"budget.max_judge_calls is {max_judge_calls}",
            remediation="Reduce rows (--rows), remove judge scorers, or "
            "raise budget.max_judge_calls in agentkit.yaml.",
        )


def render(cost: CostEstimate) -> str:
    """One line, printed with the plan before any spend."""

    if cost.judge_calls == 0:
        return (
            "Judge cost estimate: 0 judge calls (code scorers only; judged "
            "runs estimate here before spending)"
        )
    line = (
        f"Judge cost estimate: {cost.judge_calls} judge calls across "
        f"{len(cost.judge_scorers)} judge(s), ~{cost.estimated_tokens:,} "
        "tokens"
    )
    if cost.estimated_usd is not None:
        line += f" (~${cost.estimated_usd:,.4f} at the configured rate)"
    else:
        line += " (set budget.judge_price_per_1m_tokens for a dollar figure)"
    return line


def _mean_row_tokens(rows: Sequence[Mapping[str, Any]]) -> int:
    if not rows:
        return 0
    total_characters = 0
    for row in rows:
        payload = {
            "inputs": row.get("inputs"),
            "expectations": row.get("expectations"),
            "outputs": row.get("outputs"),
        }
        total_characters += len(json.dumps(payload, default=str))
    return round(total_characters / len(rows) / _CHARS_PER_TOKEN)
