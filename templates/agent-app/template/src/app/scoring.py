"""Deterministic trajectory scoring (pure functions, both gate tiers)."""

from __future__ import annotations

from collections.abc import Sequence


def tool_call_accuracy(actual: Sequence[str], expected: Sequence[str]) -> float:
    """1.0 when the agent used exactly the expected tool set (order-free).

    Extra tool calls are failures too — they are latency and cost the case
    says the answer does not need.
    """

    return 1.0 if sorted(actual) == sorted(expected) else 0.0
