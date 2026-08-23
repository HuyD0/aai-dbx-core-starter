"""Behavioral tests for the typed mechanics behind examples 07-13."""

import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.support.agent_assurance import (  # noqa: E402
    build_judge_reports,
    build_session_report,
    build_tool_trajectory_reports,
    judge_authorization,
    multi_turn_sessions,
    score_tool_trajectory_case,
    session_gate,
    tool_trajectory_cases,
    tool_trajectory_gate,
)
from examples.support.connected_llm import response_text  # noqa: E402
from examples.support.cost_quality import (  # noqa: E402
    build_cost_comparison,
    comparison_decision,
)
from examples.support.model_selection import (  # noqa: E402
    GOLDEN_CASES,
    PAIRWISE_CASES,
    SIMULATED_APPROVED_PRICE_CARD,
    SIMULATED_SESSION_OBSERVATIONS,
    compare_session_economics,
    golden_fixture_context,
    pairwise_judge_fixture,
    run_balanced_pairwise_judge,
    run_golden_ab,
)
from examples.support.optimization import (  # noqa: E402
    SPLIT_MANIFEST,
    SPLIT_RECORDS,
    split_contract_summary,
)


def test_stream_text_normalization_handles_supported_provider_shapes():
    block = type("TextBlock", (), {"type": "text", "text": "second"})()
    assert response_text("first") == "first"
    assert response_text([{"type": "text", "text": "first"}, block]) == (
        "first\nsecond"
    )
    assert response_text(None) == ""


def test_tool_assurance_helpers_preserve_layered_gate_behavior():
    cases = tool_trajectory_cases()
    report, assurance = build_tool_trajectory_reports(cases)

    assert assurance["outcome_assessment"].tolist() == ["PASS", "PASS", "PASS"]
    assert assurance["behavior_assessment"].tolist() == ["PASS", "FAIL", "PASS"]
    assert tool_trajectory_gate(report)["decision"] == "reject"

    reordered = deepcopy(cases[2])
    events = reordered["observed"]["trajectory_events"]
    events[2], events[3] = events[3], events[2]
    assert score_tool_trajectory_case(reordered)["safe_fallback_observed"] is False


def test_session_and_judge_helpers_fail_closed_on_incomplete_evidence():
    session_report = build_session_report(multi_turn_sessions())
    metrics, _policy, gate = session_gate(session_report)
    assert metrics["conversation_completion_rate"] == 1 / 3
    assert metrics["minimum_critical_session_pass"] == 0.0
    assert gate.passed is False

    deterministic, labels, agreements = build_judge_reports()
    authorization = judge_authorization(labels)
    assert deterministic["critical_case_pass"].tolist() == [True, False, False]
    assert agreements.set_index("split").loc["validation", "judge_v2_agreement"] == 0.75
    assert authorization["judge_status"] == "report_only"


def test_cost_helpers_compare_only_quality_eligible_covered_models():
    comparison = build_cost_comparison()
    decision = comparison_decision(comparison)

    assert decision["preferred_under_demonstration_budget"] == "economy-chat"
    assert decision["unknown_cost_models"] == ["quality-chat"]
    assert decision["release"] == "blocked_until_connected_evaluation"


def test_optimization_splits_are_complete_disjoint_and_repeatable():
    first = split_contract_summary()
    second = split_contract_summary()

    assert first == second
    assert first["case_counts"] == {name: 6 for name in SPLIT_MANIFEST}
    assert set(SPLIT_RECORDS) == set(SPLIT_MANIFEST)
    assert all(len(records) == 6 for records in SPLIT_RECORDS.values())


def test_model_selection_helpers_keep_measurement_sources_and_order_stable():
    golden = run_golden_ab(
        golden_fixture_context(),
        ("baseline-chat", "change-chat"),
        GOLDEN_CASES,
    )
    assert golden["measurement_source"] == "simulated_offline_fixture"
    assert golden["summary"]["baseline-chat"]["critical_pass_rate"] == 1.0
    assert golden["summary"]["change-chat"]["critical_pass_rate"] == 0.5

    pairwise = run_balanced_pairwise_judge(
        pairwise_judge_fixture(),
        PAIRWISE_CASES,
        ("baseline-chat", "change-chat"),
    )
    assert pairwise["measurement_source"] == "simulated_offline_fixture"
    assert pairwise["position_A_win_rate"] == pairwise["position_B_win_rate"]

    economics = compare_session_economics(
        SIMULATED_SESSION_OBSERVATIONS,
        SIMULATED_APPROVED_PRICE_CARD,
        sessions_per_month=50_000,
    )
    assert all(row["cost_comparable"] for row in economics)
    assert all(
        row["measurement_source"] == "simulated_offline_fixture" for row in economics
    )
