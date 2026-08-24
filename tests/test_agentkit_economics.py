"""Unit tests for the run-economics evidence: coverage-first spend accounting.

The module under test never invents a number: unknown cost stays unknown,
per-success ratios appear only at complete coverage, and success means the
agent completed — not that the judge stayed healthy.
"""

import json
from types import SimpleNamespace

import pytest

from aai_core.agentkit.economics import (
    EconomicsConfig,
    EconomicsEvidence,
    _percentile,
    build_economics_evidence,
    economics_direction,
    is_economics_metric,
)


def _envelope(
    *,
    input_tokens=None,
    output_tokens=None,
    total_tokens=None,
    cost=None,
    duration_ms=None,
    state="OK",
    spans=(),
):
    """A minimal v3-shaped trace document, the way a stored row carries one."""

    metadata = {}
    usage = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    if usage:
        metadata["mlflow.trace.tokenUsage"] = json.dumps(usage)
    if cost is not None:
        metadata["mlflow.trace.cost"] = json.dumps(cost)
    info = {"trace_id": "tr-1", "state": state, "trace_metadata": metadata}
    if duration_ms is not None:
        info["execution_duration_ms"] = duration_ms
    return {"info": info, "data": {"spans": list(spans)}}


def _rows(count, intents=None):
    return [
        {"inputs": {"question": f"q{index}", "intent": (intents or {}).get(index, "")}}
        for index in range(count)
    ]


def _build(traces, *, error_flags=None, strata=(), config=None, rows=None):
    rows = rows if rows is not None else _rows(len(traces))
    return build_economics_evidence(
        rows,
        traces,
        error_flags if error_flags is not None else [False] * len(rows),
        strata=strata,
        config=config or EconomicsConfig(),
    )


def test_percentile_matches_hand_computed_values():
    assert _percentile([5.0], 0.95) == 5.0
    assert _percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    # position (3-1)*0.95 = 1.9 -> 2*0.1 + 3*0.9
    assert _percentile([1.0, 2.0, 3.0], 0.95) == pytest.approx(2.9)
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)
    # position (4-1)*0.95 = 2.85 -> 3*0.15 + 4*0.85
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_direction_and_membership_tables():
    assert economics_direction("cost/coverage") == "higher"
    assert economics_direction("tokens/coverage") == "higher"
    assert economics_direction("economics/success_rate") == "higher"
    assert economics_direction("economics/cost_p95_usd") == "lower"
    assert economics_direction("economics/cost_per_success_usd") == "lower"
    assert economics_direction("economics/latency_p95_seconds") == "lower"
    assert economics_direction("correctness/mean") is None
    assert is_economics_metric("cost/coverage")
    assert is_economics_metric("economics/tokens_p50")
    assert not is_economics_metric("keyword_coverage/mean")


def test_unknown_cost_is_never_zero():
    """Partial coverage reports the known subtotal and refuses the ratio."""

    traces = [
        _envelope(input_tokens=10, output_tokens=5, cost={"total_cost": 1.0}),
        _envelope(input_tokens=10, output_tokens=5, cost={"total_cost": 3.0}),
        _envelope(input_tokens=10, output_tokens=5),
        _envelope(input_tokens=10, output_tokens=5),
    ]

    evidence, metrics, warnings = _build(traces)

    assert evidence is not None
    assert metrics["cost/coverage"] == pytest.approx(0.5)
    assert metrics["tokens/coverage"] == 1.0
    assert evidence.cost_known == 2
    assert evidence.cost_total_usd == pytest.approx(4.0)
    assert "economics/cost_per_success_usd" not in metrics
    assert any(
        "economics/cost_per_success_usd" in warning and "2 of 4" in warning
        for warning in warnings
    )
    # Tokens have complete coverage, so their ratio is reported.
    assert metrics["economics/tokens_per_success"] == pytest.approx(15.0)


def test_cost_per_success_counts_failed_rows_spend():
    """The failed row's spend lands on the successes — that is the point."""

    traces = [
        _envelope(cost={"total_cost": 1.0}),
        _envelope(cost={"total_cost": 1.0}),
        _envelope(cost={"total_cost": 1.0}),
        _envelope(cost={"total_cost": 5.0}),
    ]

    evidence, metrics, _ = _build(traces, error_flags=[False, False, False, True])

    assert evidence is not None
    assert evidence.successes == 3
    assert metrics["economics/success_rate"] == pytest.approx(0.75)
    assert metrics["economics/cost_per_success_usd"] == pytest.approx(8.0 / 3.0)


def test_success_excludes_error_rows_and_error_traces():
    traces = [
        _envelope(state="OK"),
        _envelope(state="ERROR"),
        _envelope(state="OK"),
    ]

    evidence, metrics, _ = _build(traces, error_flags=[True, False, False])

    assert evidence is not None
    assert evidence.success == (False, False, True)
    assert metrics["economics/success_rate"] == pytest.approx(1.0 / 3.0)


def test_zero_successes_report_no_per_success_ratio():
    traces = [_envelope(cost={"total_cost": 2.0}, state="ERROR")]

    evidence, metrics, warnings = _build(traces)

    assert evidence is not None
    assert evidence.successes == 0
    assert "economics/cost_per_success_usd" not in metrics
    assert any("no successful completions" in warning for warning in warnings)


def test_usage_read_from_live_trace_info_attributes():
    """The pinned TraceInfo properties are the first-choice reading."""

    trace = SimpleNamespace(
        info=SimpleNamespace(
            token_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            cost={"input_cost": 0.2, "output_cost": 0.3, "total_cost": 0.5},
            execution_duration=250,
            state="TraceState.OK",
        ),
        to_dict=lambda: {"info": {}, "data": {"spans": []}},
    )

    evidence, metrics, _ = _build([trace])

    assert evidence is not None
    assert evidence.total_tokens == (15,)
    assert evidence.cost_usd == (0.5,)
    assert evidence.duration_ms == (250,)
    assert evidence.cost_source == "trace"
    assert metrics["economics/latency_p95_seconds"] == pytest.approx(0.25)


def test_usage_read_from_envelope_trace_metadata():
    traces = [
        _envelope(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            duration_ms=1000,
        )
    ]

    evidence, metrics, _ = _build(traces)

    assert evidence is not None
    assert evidence.input_tokens == (100,)
    assert evidence.output_tokens == (50,)
    assert evidence.total_tokens == (150,)
    assert metrics["economics/tokens_p50"] == 150.0


def test_usage_summed_from_llm_spans_when_metadata_is_absent():
    """A retry loop is two LLM spans; the row's spend is their sum."""

    spans = [
        {
            "attributes": {
                "mlflow.spanType": json.dumps("LLM"),
                "mlflow.chat.tokenUsage": json.dumps(
                    {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
                ),
            }
        },
        {
            "attributes": {
                "mlflow.spanType": json.dumps("CHAT_MODEL"),
                "gen_ai.usage.input_tokens": 200,
                "gen_ai.usage.output_tokens": 20,
            }
        },
        {"attributes": {"mlflow.spanType": json.dumps("TOOL")}},
    ]

    evidence, metrics, _ = _build([_envelope(spans=spans)])

    assert evidence is not None
    assert evidence.input_tokens == (300,)
    assert evidence.output_tokens == (30,)
    # Each span contributes its effective total: 110 explicit + 220 derived.
    assert evidence.total_tokens == (330,)
    assert evidence.llm_calls == (2,)
    assert metrics["economics/tokens_p50"] == pytest.approx(330.0)


def test_trace_recorded_cost_beats_the_configured_price():
    config = EconomicsConfig(
        price_per_1m_input_tokens=100.0, price_per_1m_output_tokens=100.0
    )
    traces = [
        _envelope(input_tokens=1000, output_tokens=1000, cost={"total_cost": 0.5})
    ]

    evidence, metrics, _ = _build(traces, config=config)

    assert evidence is not None
    assert evidence.cost_usd == (0.5,)
    assert evidence.cost_source == "trace"


def test_configured_price_pair_prices_known_tokens():
    config = EconomicsConfig(
        price_per_1m_input_tokens=2.0, price_per_1m_output_tokens=6.0
    )
    traces = [
        _envelope(input_tokens=1000, output_tokens=2000),
        _envelope(),
    ]

    evidence, metrics, _ = _build(traces, config=config)

    assert evidence is not None
    expected = (1000 * 2.0 + 2000 * 6.0) / 1_000_000.0
    assert evidence.cost_usd == (pytest.approx(expected), None)
    assert evidence.cost_source == "configured-price"
    assert metrics["cost/coverage"] == pytest.approx(0.5)


def test_price_pair_is_both_or_neither():
    with pytest.raises(ValueError, match="pair"):
        EconomicsConfig(price_per_1m_input_tokens=1.0)
    with pytest.raises(ValueError, match="pair"):
        EconomicsConfig(price_per_1m_output_tokens=1.0)


def test_segments_group_by_stratum_value():
    intents = {0: "billing", 1: "billing", 2: "smalltalk"}
    traces = [
        _envelope(cost={"total_cost": 1.0}, duration_ms=1000),
        _envelope(cost={"total_cost": 3.0}, duration_ms=3000),
        _envelope(cost={"total_cost": 0.1}, duration_ms=100),
    ]

    evidence, _, warnings = _build(
        traces,
        rows=_rows(3, intents),
        error_flags=[False, True, False],
        strata=("intent",),
    )

    assert evidence is not None
    assert not [warning for warning in warnings if "strata" in warning]
    by_value = {segment.value: segment for segment in evidence.segments}
    assert set(by_value) == {"billing", "smalltalk"}
    billing = by_value["billing"]
    assert billing.rows == 2
    assert billing.successes == 1
    # Both rows' spend lands on the one success, failed row included.
    assert billing.cost_per_success_usd == pytest.approx(4.0)
    assert billing.latency_p95_seconds == pytest.approx(2.9)
    assert by_value["smalltalk"].cost_per_success_usd == pytest.approx(0.1)


def test_segment_cardinality_is_capped():
    count = 25
    intents = {index: f"intent-{index:02d}" for index in range(count)}
    traces = [_envelope() for _ in range(count)]

    evidence, _, warnings = _build(
        traces, rows=_rows(count, intents), strata=("intent",)
    )

    assert evidence is not None
    assert len(evidence.segments) == 20
    assert any("25 distinct values" in warning for warning in warnings)


def test_disabled_or_empty_builds_nothing():
    assert build_economics_evidence(
        [], [], [], strata=(), config=EconomicsConfig()
    ) == (None, {}, [])
    assert _build([_envelope()], config=EconomicsConfig(enabled=False)) == (
        None,
        {},
        [],
    )


def test_missing_traces_leave_rows_unknown_not_zero():
    evidence, metrics, _ = _build([None, _envelope(cost={"total_cost": 1.0})])

    assert evidence is not None
    assert evidence.cost_usd == (None, 1.0)
    assert evidence.cost_total_usd == pytest.approx(1.0)
    assert metrics["cost/coverage"] == pytest.approx(0.5)
    # An absent trace is not evidence of failure.
    assert evidence.success == (True, True)


def test_shorter_trace_sequence_marks_the_tail_unknown():
    rows = _rows(3)
    evidence, metrics, _ = _build(
        [_envelope(cost={"total_cost": 1.0})], rows=rows, error_flags=[False] * 3
    )

    assert evidence is not None
    assert evidence.rows == 3
    assert evidence.cost_usd == (1.0, None, None)


def test_evidence_round_trips_through_strict_json():
    traces = [
        _envelope(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost={"total_cost": 0.25},
            duration_ms=500,
        ),
        _envelope(state="ERROR"),
    ]
    evidence, _, _ = _build(traces, strata=("intent",))
    assert evidence is not None

    document = json.loads(evidence.model_dump_json())
    restored = EconomicsEvidence.model_validate(document)

    assert restored == evidence
