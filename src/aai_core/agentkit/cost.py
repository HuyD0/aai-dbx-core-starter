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
from typing import Any, cast

from pydantic import Field, field_serializer, field_validator

from aai_core.agentkit.catalog import JudgeFanout, ScorerPlan
from aai_core.agentkit.datasets import retrieval_fanout, trace_judge_text
from aai_core.agentkit.errors import BudgetExceededError
from aai_core.contracts import ContractModel, freeze_value, thaw_value

_ASSUMED_JUDGE_OUTPUT_TOKENS = 150
_CHARS_PER_TOKEN = 4
# One retriever span per row is the ordinary shape; the chunk count is the
# retriever's `k`, which only the project knows. It is configurable
# (`budget.retrieved_chunks_per_row`) because a wrong guess here is the
# difference between a budget that holds and one that is exceeded 4x.
_ASSUMED_RETRIEVER_SPANS_PER_ROW = 1
DEFAULT_CHUNKS_PER_ROW = 5
_RETRIEVAL_SUFFICIENCY_SCORER = "retrieval_sufficiency"


class CostEstimate(ContractModel):
    rows: int = Field(ge=0)
    judge_scorers: tuple[str, ...] = ()
    judge_calls: int = Field(ge=0)
    mean_row_tokens: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
    estimated_usd: float | None = None
    calls_by_scorer: Mapping[str, int] = Field(default_factory=dict)
    # True when every fan-out multiplier was counted from the rows' own
    # traces; False when a retrieval scorer's chunk count had to be assumed
    # because the traces do not exist yet (a live run).
    fanout_counted: bool = True
    assumed_chunks_per_row: int = Field(default=DEFAULT_CHUNKS_PER_ROW, ge=1)

    @field_validator("calls_by_scorer", mode="after")
    @classmethod
    def freeze_calls(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        return cast(Mapping[str, int], freeze_value(value))

    @field_serializer("calls_by_scorer")
    def serialize_calls(self, value: Mapping[str, int]) -> dict[str, int]:
        return cast(dict[str, int], thaw_value(value))


def estimate(
    rows: Sequence[Mapping[str, Any]],
    plan: ScorerPlan,
    *,
    price_per_1m_tokens: float | None = None,
    chunks_per_row: int = DEFAULT_CHUNKS_PER_ROW,
) -> CostEstimate:
    """Estimate judge calls and tokens for scoring ``rows`` under ``plan``.

    Not one call per row: the registry records each scorer's fan-out, and
    the retrieval scorers fan out per retriever span or per retrieved
    chunk. Where the rows carry traces the multipliers are counted; where
    they do not (a live run has no traces until it runs) they are assumed
    and the estimate says so.
    """

    judge_specs = plan.judge_specs
    row_count = len(rows)
    mean_row_tokens = _mean_row_tokens(rows)
    needs_fanout = any(spec.fanout is not JudgeFanout.ROW for spec in judge_specs)
    counted = retrieval_fanout(rows) if needs_fanout else None
    # Only rows with no readable trace need an assumption. A traced row
    # that retrieved nothing is a counted zero: the retrieval scorers skip
    # it, so charging it an assumed span and a page of chunks would let
    # `max_judge_calls` refuse a conditionally retrieving agent whose real
    # run is far inside its budget.
    uncounted_rows = row_count - (counted.rows_with_traces if counted else 0)
    fanout_counted = not needs_fanout or uncounted_rows == 0
    spans = chunks = 0
    span_input_tokens = sufficiency_input_tokens = chunk_input_tokens = 0
    if counted is not None:
        spans = counted.retriever_spans + uncounted_rows * (
            _ASSUMED_RETRIEVER_SPANS_PER_ROW
        )
        chunks = counted.retrieved_chunks + uncounted_rows * chunks_per_row
        # Traced retrieval calls are priced from the spans/chunks actually
        # judged. Only unreadable/unavailable traces need an assumption, and
        # their own row payloads supply that assumption's token size. Using
        # the mean across every row lets many small non-retrieving rows dilute
        # one large retrieved context before approval.
        uncounted_row_tokens = sum(
            _mean_row_tokens((row,))
            for row in rows
            if retrieval_fanout((row,)).rows_with_traces == 0
        )
        span_input_tokens = (
            round(counted.retriever_span_input_characters / _CHARS_PER_TOKEN)
            + uncounted_row_tokens * _ASSUMED_RETRIEVER_SPANS_PER_ROW
        )
        sufficiency_input_tokens = (
            round(counted.retrieval_sufficiency_input_characters / _CHARS_PER_TOKEN)
            + uncounted_row_tokens * _ASSUMED_RETRIEVER_SPANS_PER_ROW
        )
        chunk_input_tokens = (
            round(counted.retrieved_chunk_input_characters / _CHARS_PER_TOKEN)
            + uncounted_row_tokens * chunks_per_row
        )

    total_tokens = 0
    calls_by_scorer: dict[str, int] = {}
    for spec in judge_specs:
        if spec.fanout is JudgeFanout.RETRIEVER_SPAN:
            calls = spans
            input_tokens = (
                sufficiency_input_tokens
                if spec.name == _RETRIEVAL_SUFFICIENCY_SCORER
                else span_input_tokens
            )
        elif spec.fanout is JudgeFanout.RETRIEVED_CHUNK:
            calls = chunks
            input_tokens = chunk_input_tokens
        else:
            calls = row_count
            input_tokens = calls * mean_row_tokens
        calls_by_scorer[spec.name] = calls
        total_tokens += input_tokens + calls * (
            spec.judge_overhead_tokens + _ASSUMED_JUDGE_OUTPUT_TOKENS
        )
    judge_calls = sum(calls_by_scorer.values())
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
        fanout_counted=fanout_counted,
        assumed_chunks_per_row=chunks_per_row,
    )


def enforce_budget(
    cost: CostEstimate,
    *,
    max_judge_calls: int | None,
    extra_judge_calls: int = 0,
) -> None:
    """Abort BEFORE any judge call when the estimate exceeds the budget.

    ``extra_judge_calls`` covers spend the plan itself does not model —
    the judge-integrity re-scoring calls — so the configured ceiling is a
    ceiling on the whole run, not just its first pass.
    """

    total = cost.judge_calls + max(0, extra_judge_calls)
    if max_judge_calls is not None and total > max_judge_calls:
        detail = (
            f" (including {extra_judge_calls} integrity re-scoring calls)"
            if extra_judge_calls
            else ""
        )
        raise BudgetExceededError(
            f"this run would make {total} judge calls{detail}; "
            f"budget.max_judge_calls is {max_judge_calls}",
            remediation="Reduce rows (--rows), remove judge scorers, lower "
            "integrity.consistency_sample, or raise budget.max_judge_calls "
            "in agentkit.yaml.",
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
    if not cost.fanout_counted:
        line += (
            "\n  Retrieval scorers are judged per retrieved chunk, and a live "
            f"run has no traces to count yet: assuming "
            f"{cost.assumed_chunks_per_row} chunks per row "
            "(budget.retrieved_chunks_per_row)."
        )
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
        # A trace-backed row carries its answer and its retrieved context
        # in the trace, and those are what the judge is shown. Counting
        # only the row's own fields reports a near-zero estimate for the
        # runs that cost the most.
        total_characters += len(trace_judge_text(row.get("trace")))
    return round(total_characters / len(rows) / _CHARS_PER_TOKEN)
