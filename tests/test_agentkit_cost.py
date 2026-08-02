"""Unit tests for the pre-run judge cost estimate and budget enforcement."""

import pytest

from aai_core.agentkit.catalog import select_scorers
from aai_core.agentkit.config import AgentkitConfig
from aai_core.agentkit.cost import enforce_budget, estimate, render
from aai_core.agentkit.datasets import DatasetShape
from aai_core.agentkit.errors import BudgetExceededError


def _shape():
    return DatasetShape(
        row_count=4,
        input_keys=("question",),
        has_outputs=True,
        expectation_keys=("expected_response",),
        has_traces=False,
        strata_values={},
    )


def _config():
    return AgentkitConfig(version=1, agent="agent.py:respond", dataset="golden.json")


def _rows(count=4):
    return [
        {
            "inputs": {"question": f"question {index}"},
            "expectations": {"expected_response": f"answer {index}"},
        }
        for index in range(count)
    ]


def _judged_plan():
    return select_scorers(_shape(), _config(), mode="live", judges_enabled=True)


def _code_only_plan():
    return select_scorers(
        _shape(), _config(), mode="answer-sheet", judges_enabled=False
    )


def test_estimate_arithmetic_is_deterministic():
    plan = _judged_plan()
    rows = _rows(4)

    cost = estimate(rows, plan)

    judge_names = {spec.name for spec in plan.judge_specs}
    assert judge_names == {"correctness", "safety"}
    assert cost.judge_calls == 4 * 2
    assert dict(cost.calls_by_scorer) == {"correctness": 4, "safety": 4}
    # both judges carry the default 350-token overhead + 150 output tokens
    expected_tokens = sum(4 * (350 + cost.mean_row_tokens + 150) for _ in judge_names)
    assert cost.estimated_tokens == expected_tokens
    assert cost.estimated_usd is None

    again = estimate(rows, plan)
    assert again == cost


def test_zero_judges_is_zero_cost_by_construction():
    cost = estimate(_rows(4), _code_only_plan())

    assert cost.judge_calls == 0
    assert cost.estimated_tokens == 0
    assert "0 judge calls" in render(cost)


def test_dollar_line_only_with_a_configured_rate():
    plan = _judged_plan()
    rows = _rows(4)

    without_rate = estimate(rows, plan)
    assert "$" not in render(without_rate).split("(")[0]
    assert "dollar figure" in render(without_rate)

    with_rate = estimate(rows, plan, price_per_1m_tokens=5.0)
    assert with_rate.estimated_usd == round(
        with_rate.estimated_tokens / 1_000_000 * 5.0, 4
    )
    assert "$" in render(with_rate)


def test_budget_enforced_before_any_call():
    cost = estimate(_rows(4), _judged_plan())

    enforce_budget(cost, max_judge_calls=None)
    enforce_budget(cost, max_judge_calls=8)
    with pytest.raises(BudgetExceededError) as excinfo:
        enforce_budget(cost, max_judge_calls=7)
    assert "max_judge_calls" in str(excinfo.value)
