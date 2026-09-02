"""Behavioral tests for the typed mechanics behind examples 07-13."""

import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.support.agent_assurance import (  # noqa: E402
    build_judge_reports,
    build_session_report,
    build_tool_trajectory_reports,
    judge_authorization,
    judge_disagreements,
    judge_v1,
    judge_v2,
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
    quality_gate_report,
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
    demonstrate_split_leakage,
    evaluate_winner_on_holdout,
    optimization_plan,
    run_toy_prompt_optimization,
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

    assert assurance["outcome_assessment"].tolist() == ["PASS"] * 4
    assert assurance["behavior_assessment"].tolist() == ["PASS", "FAIL", "PASS", "FAIL"]
    assert report["tool_order_policy"].tolist() == [True, True, True, False]
    assert tool_trajectory_gate(report)["decision"] == "reject"

    # The ordering policy alone can block: swap the two calls and the fourth
    # case passes, so a fixture where only the order was wrong is exactly the
    # case the multiset cannot distinguish from a clean one.
    reordered_calls = deepcopy(cases[3])
    observed_calls = reordered_calls["observed"]["tool_calls"]
    observed_calls[0], observed_calls[1] = observed_calls[1], observed_calls[0]
    assert score_tool_trajectory_case(reordered_calls)["tool_order_policy"] is True

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

    # Judge verdicts are computed by running the rules over the reviewed
    # response texts, never stored alongside the human labels.
    assert labels["judge_v1"].tolist() == [
        judge_v1(text) for text in labels["response"]
    ]
    assert labels["judge_v2"].tolist() == [
        judge_v2(text) for text in labels["response"]
    ]
    assert judge_v1("Results may improve.") is True
    assert judge_v2("Results may improve.") is False
    assert judge_v2("Based on the excerpt, results may vary.") is True
    assert judge_v2("Based on the excerpt, growth is guaranteed.") is False

    by_split = agreements.set_index("split")
    assert by_split.loc["calibration", "judge_v1_agreement"] == 0.5
    assert by_split.loc["calibration", "judge_v2_agreement"] == 1.0
    assert by_split.loc["validation", "judge_v1_agreement"] == 0.5
    assert by_split.loc["validation", "judge_v2_agreement"] == 0.75
    assert judge_disagreements(labels, "judge_v2")["case_id"].tolist() == ["val-04"]
    assert authorization["judge_status"] == "report_only"


def test_cost_helpers_compare_only_quality_eligible_covered_models():
    comparison = build_cost_comparison()
    decision = comparison_decision(comparison)
    by_model = comparison.set_index("logical_model")

    # The gate and coverage flags are consequences of executed fixture runs.
    assert not bool(by_model.loc["draft-chat", "quality_eligible"])
    assert bool(by_model.loc["economy-chat", "cost_comparable"])
    assert bool(by_model.loc["general-chat", "cost_comparable"])
    assert by_model.loc["quality-chat", "quality_score"] == 1.0
    assert by_model.loc["quality-chat", "cost_coverage"] == 0.0
    assert pd.isna(by_model.loc["quality-chat", "target_inference_cost_usd"])

    # Verbosity and cost are measured from the actual answer strings.
    assert (
        by_model.loc["general-chat", "output_tokens"]
        > by_model.loc["economy-chat", "output_tokens"]
    )
    assert (
        by_model.loc["general-chat", "target_inference_cost_usd"]
        > by_model.loc["economy-chat", "target_inference_cost_usd"]
    )

    gate = quality_gate_report(comparison).set_index("logical_model")
    assert "investment recommendation" in gate.loc["draft-chat", "reason"]
    assert gate.loc["economy-chat", "reason"] == "passed every quality gate"

    assert decision["preferred_under_demonstration_budget"] == "economy-chat"
    assert decision["quality_ineligible_models"] == ["draft-chat"]
    assert decision["unknown_cost_models"] == ["quality-chat"]
    assert decision["top_quality_without_cost_evidence"] == ["quality-chat"]
    assert decision["release"] == "blocked_until_connected_evaluation"


def test_optimization_splits_are_complete_disjoint_and_repeatable():
    first = split_contract_summary()
    second = split_contract_summary()

    assert first == second
    assert first["case_counts"] == {name: 6 for name in SPLIT_MANIFEST}
    assert set(SPLIT_RECORDS) == set(SPLIT_MANIFEST)
    assert all(len(records) == 6 for records in SPLIT_RECORDS.values())

    # Splits are disjoint in content, not only in case IDs: held-out cases
    # use genuinely different excerpt wording than optimizer training.
    train_texts = {
        record["inputs"]["earnings_excerpt"]
        for record in SPLIT_RECORDS["optimizer_training"]
    }
    holdout_texts = {
        record["inputs"]["earnings_excerpt"]
        for record in SPLIT_RECORDS["held_out_release"]
    }
    assert train_texts.isdisjoint(holdout_texts)


def test_toy_optimizer_wins_on_train_and_is_exposed_by_disjoint_holdout():
    optimization = run_toy_prompt_optimization()
    assert optimization["winner"] == "seed+exemplars"
    assert optimization["train_scores"]["seed+exemplars"] == 1.0
    assert optimization["train_scores"]["seed+cite"] == 0.944
    assert optimization["train_scores"]["seed"] == 0.611
    assert optimization["train_scores"]["seed+brevity"] == 0.611

    heldout = evaluate_winner_on_holdout(optimization)
    assert heldout["winner"] == "seed+exemplars"
    assert heldout["train_score"] == 1.0
    assert heldout["holdout_score"] == 0.611
    assert heldout["generalization_gap"] == 0.389
    # The honest holdout even ranks the memorizing winner below the simple
    # citation edit it beat on training data.
    assert heldout["holdout_score"] < optimization["train_scores"]["seed+cite"]

    leakage = demonstrate_split_leakage(optimization)
    assert leakage["leaked_overlapping_holdout"] == 1.0
    assert leakage["honest_disjoint_holdout"] == heldout["holdout_score"]
    assert leakage["leak_inflation"] == 0.389

    plan = optimization_plan()
    assert plan["offline_toy_winner"] == "seed+exemplars"
    assert plan["decision"] == "inconclusive"
    assert plan["release"] == "blocked"


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
