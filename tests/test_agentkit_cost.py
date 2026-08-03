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


def _rag_rows(count=3, chunks=4):
    """Rows whose traces carry a retriever span with ``chunks`` documents."""

    return [
        {
            "inputs": {"question": f"question {index}"},
            "expectations": {"expected_response": f"answer {index}"},
            "trace": {
                "data": {
                    "spans": [
                        {
                            "type": "RETRIEVER",
                            "name": "search",
                            "outputs": [
                                {"page_content": f"chunk {position}"}
                                for position in range(chunks)
                            ],
                        }
                    ]
                }
            },
        }
        for index in range(count)
    ]


def _rag_plan(mode="traces", **shape_kwargs):
    shape = DatasetShape(
        row_count=3,
        input_keys=("question",),
        has_outputs=False,
        expectation_keys=("expected_response",),
        has_traces=shape_kwargs.get("has_traces", True),
        strata_values={},
        has_retrieval_spans=shape_kwargs.get("has_retrieval_spans", True),
    )
    config = AgentkitConfig(
        version=1,
        agent="agent.py:respond",
        dataset="golden.json",
        scorers={"add": ["retrieval_groundedness", "retrieval_relevance"]},
    )
    return select_scorers(shape, config, mode=mode, judges_enabled=True)


def test_retrieval_relevance_costs_one_call_per_chunk():
    """MLflow judges retrieval relevance per chunk, not per row.

    ``RetrievalRelevance._compute_span_relevance`` loops the chunks of each
    RETRIEVER span. Counting one call per row would make
    ``budget.max_judge_calls`` a number rather than a ceiling.
    """

    plan = _rag_plan()
    rows = _rag_rows(count=3, chunks=4)

    cost = estimate(rows, plan)

    calls = dict(cost.calls_by_scorer)
    assert calls["retrieval_relevance"] == 12  # 3 rows x 4 chunks
    assert calls["retrieval_groundedness"] == 3  # 3 rows x 1 retriever span
    assert calls["correctness"] == 3  # per row
    assert cost.fanout_counted is True


def test_multiple_retriever_spans_multiply_span_scorers():
    plan = _rag_plan()
    rows = _rag_rows(count=1, chunks=2)
    spans = rows[0]["trace"]["data"]["spans"]
    rows[0]["trace"]["data"]["spans"] = spans + [dict(spans[0])]

    cost = estimate(rows, plan)

    assert dict(cost.calls_by_scorer)["retrieval_groundedness"] == 2
    assert dict(cost.calls_by_scorer)["retrieval_relevance"] == 4


def test_live_run_assumes_chunk_fanout_and_says_so():
    """A live run has no traces to count, so the estimate is disclosed."""

    plan = _rag_plan(mode="live", has_traces=False, has_retrieval_spans=False)
    rows = _rows(3)

    cost = estimate(rows, plan, chunks_per_row=10)

    assert cost.fanout_counted is False
    assert dict(cost.calls_by_scorer)["retrieval_relevance"] == 30
    assert "assuming 10 chunks per row" in render(cost)


def test_budget_ceiling_counts_the_fanout():
    plan = _rag_plan()
    rows = _rag_rows(count=3, chunks=4)

    cost = estimate(rows, plan)

    # 12 relevance (per chunk) + 3 groundedness + 3 sufficiency (per
    # retriever span) + 3 correctness + 3 safety (per row) = 24
    with pytest.raises(BudgetExceededError) as excinfo:
        enforce_budget(cost, max_judge_calls=15)
    assert "24 judge calls" in str(excinfo.value)
    # the pre-fanout count would have been 5 scorers x 3 rows = 15,
    # which is exactly the budget it now correctly refuses
    assert cost.judge_calls > len(plan.judge_specs) * cost.rows


def test_serialized_traces_are_counted_like_mappings():
    """MLflow serialises a dataframe's trace column as a JSON string.

    Inspecting only mappings would leave the real chunk count uncounted and
    silently fall back to the assumption, so `max_judge_calls` would
    authorize a run well past its stated ceiling.
    """

    import json as json_module

    plan = _rag_plan()
    mapping_rows = _rag_rows(count=3, chunks=4)
    serialized_rows = [
        {**row, "trace": json_module.dumps(row["trace"])} for row in mapping_rows
    ]

    from_mapping = estimate(mapping_rows, plan)
    from_string = estimate(serialized_rows, plan)

    assert from_string.fanout_counted is True
    assert dict(from_string.calls_by_scorer) == dict(from_mapping.calls_by_scorer)
    assert dict(from_string.calls_by_scorer)["retrieval_relevance"] == 12


def test_trace_backed_rows_count_the_context_the_judge_sees():
    """A trace-only row's tokens live in the trace, not in the row."""

    from aai_core.agentkit.datasets import trace_judge_text

    chunk = "Contributions vest after two years of continuous service. " * 40
    trace = {
        "info": {
            "request_preview": "when do contributions vest?",
            "response_preview": "After two years.",
        },
        "data": {
            "spans": [
                {
                    "span_id": "1",
                    "type": "RETRIEVER",
                    "outputs": [{"page_content": chunk}],
                }
            ]
        },
    }
    bare = [{"inputs": {}, "trace": trace}]
    plan = _rag_plan(mode="traces")

    counted = estimate(bare, plan)

    assert len(trace_judge_text(trace)) > len(chunk)
    # The retrieved context dominates, so the estimate is not near-zero.
    assert counted.mean_row_tokens > len(chunk) // 4


def test_a_row_without_a_trace_is_unchanged():
    from aai_core.agentkit.datasets import trace_judge_text

    assert trace_judge_text(None) == ""
    assert trace_judge_text({"data": {"spans": []}}) == ""
