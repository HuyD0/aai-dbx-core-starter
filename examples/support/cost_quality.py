"""Cost/quality comparison mechanics for the advanced earnings lesson."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

JsonRecord = dict[str, Any]

PROMPT_URI = "fixture:earnings_summary/v2"
PROMPT_TEMPLATE_FIXTURE = (
    "Summarize only the supplied fictional facts. "
    "Include {{source_id}} exactly once and provide no investment advice."
)
PROMPT_DIGEST = hashlib.sha256(PROMPT_TEMPLATE_FIXTURE.encode()).hexdigest()
INFERENCE_PARAMETERS = {"temperature": 0.0, "max_tokens": 400}
SCORER_SET = "earnings-release-scorers-v1"
DEMONSTRATION_COST_BUDGET_USD = 0.005

EVALUATION_RECORDS: list[JsonRecord] = [
    {
        "inputs": {
            "case_id": "quarterly-revenue-and-margin",
            "question": "What were fictional revenue and margin results?",
            "earnings_excerpt": (
                "Revenue was $128.4 million, up 12%; operating margin was "
                "18.6%, up from 16.9%."
            ),
            "source_id": "ARS-FY25-Q2-RESULTS",
        },
        "expectations": {
            "required_facts": ["$128.4 million", "12%", "18.6%", "16.9%"],
            "source_id": "ARS-FY25-Q2-RESULTS",
            "no_investment_recommendation": True,
        },
    },
    {
        "inputs": {
            "case_id": "forward-revenue-and-margin-guidance",
            "question": "What fictional guidance was provided?",
            "earnings_excerpt": (
                "Revenue guidance was $132 million to $136 million; operating "
                "margin guidance was 19% to 20%."
            ),
            "source_id": "ARS-FY25-Q2-GUIDANCE",
        },
        "expectations": {
            "required_facts": ["$132 million", "$136 million", "19%", "20%"],
            "source_id": "ARS-FY25-Q2-GUIDANCE",
            "no_investment_recommendation": True,
        },
    },
    {
        "inputs": {
            "case_id": "cash-flow-inventory-and-supplier-risk",
            "question": "What fictional cash-flow result and risk were disclosed?",
            "earnings_excerpt": (
                "Free cash flow was $21.7 million; inventory rose 28%; "
                "single-source supplier concentration is a risk."
            ),
            "source_id": "ARS-FY25-Q2-CASH-RISK",
        },
        "expectations": {
            "required_facts": [
                "$21.7 million",
                "28%",
                "single-source supplier concentration",
            ],
            "source_id": "ARS-FY25-Q2-CASH-RISK",
            "no_investment_recommendation": True,
        },
    },
]
CASE_IDS = [record["inputs"]["case_id"] for record in EVALUATION_RECORDS]
DATASET_DIGEST = hashlib.sha256(
    json.dumps(CASE_IDS, separators=(",", ":")).encode()
).hexdigest()

MEASUREMENTS: list[JsonRecord] = [
    {
        "logical_model": "economy-chat",
        "quality_score": 0.92,
        "minimum_row_quality": 0.90,
        "critical_case_pass_rate": 1.0,
        "recommendation_policy_compliance": 1.0,
        "latency_ms_mean": 340.0,
        "input_tokens": 300,
        "output_tokens": 180,
        "target_inference_cost_usd": 0.0034,
        "evaluation_judge_cost_usd": 0.0018,
        "cost_coverage": 1.0,
        "measurement_source": "simulated_offline_fixture",
    },
    {
        "logical_model": "general-chat",
        "quality_score": 0.97,
        "minimum_row_quality": 0.95,
        "critical_case_pass_rate": 1.0,
        "recommendation_policy_compliance": 1.0,
        "latency_ms_mean": 520.0,
        "input_tokens": 302,
        "output_tokens": 210,
        "target_inference_cost_usd": 0.0068,
        "evaluation_judge_cost_usd": 0.0018,
        "cost_coverage": 1.0,
        "measurement_source": "simulated_offline_fixture",
    },
    {
        "logical_model": "quality-chat",
        "quality_score": 0.99,
        "minimum_row_quality": 0.97,
        "critical_case_pass_rate": 1.0,
        "recommendation_policy_compliance": 1.0,
        "latency_ms_mean": 710.0,
        "input_tokens": 301,
        "output_tokens": 235,
        "target_inference_cost_usd": None,
        "evaluation_judge_cost_usd": 0.0018,
        "cost_coverage": 0.67,
        "measurement_source": "simulated_offline_fixture",
    },
]


def comparison_contract() -> JsonRecord:
    return {
        "prompt_uri": PROMPT_URI,
        "prompt_digest": PROMPT_DIGEST,
        "dataset_digest": DATASET_DIGEST,
        "inference_parameters": INFERENCE_PARAMETERS,
        "scorer_set": SCORER_SET,
    }


def build_cost_comparison() -> pd.DataFrame:
    comparison = pd.DataFrame(MEASUREMENTS)
    comparison["total_tokens"] = (
        comparison["input_tokens"] + comparison["output_tokens"]
    )
    comparison["quality_eligible"] = (
        (comparison["quality_score"] >= 0.90)
        & (comparison["minimum_row_quality"] >= 0.90)
        & (comparison["critical_case_pass_rate"] == 1.0)
        & (comparison["recommendation_policy_compliance"] == 1.0)
    )
    comparison["cost_comparable"] = (
        comparison["quality_eligible"]
        & (comparison["cost_coverage"] == 1.0)
        & comparison["target_inference_cost_usd"].notna()
    )
    comparison["quality_per_cost"] = comparison["quality_score"].div(
        comparison["target_inference_cost_usd"]
    )
    return comparison


def preferred_model_under_budget(
    comparison: pd.DataFrame,
    budget_usd: float = DEMONSTRATION_COST_BUDGET_USD,
) -> str | None:
    eligible = comparison.loc[
        comparison["cost_comparable"]
        & (comparison["target_inference_cost_usd"] <= budget_usd)
    ]
    if eligible.empty:
        return None
    return str(
        eligible.sort_values(
            ["quality_score", "target_inference_cost_usd"],
            ascending=[False, True],
        ).iloc[0]["logical_model"]
    )


def comparison_decision(comparison: pd.DataFrame) -> JsonRecord:
    unknown_cost_models = comparison.loc[
        comparison["quality_eligible"] & ~comparison["cost_comparable"],
        "logical_model",
    ].tolist()
    return {
        "preferred_under_demonstration_budget": preferred_model_under_budget(
            comparison
        ),
        "unknown_cost_models": unknown_cost_models,
        "decision": "inconclusive",
        "release": "blocked_until_connected_evaluation",
        "reason": (
            "the table is a simulated fixture; rerun the exact contract against "
            "configured logical models and authoritative cost evidence"
        ),
    }


def persist_cost_quality_evidence(comparison: pd.DataFrame) -> JsonRecord:
    import mlflow

    from aai_core.experiments import (
        ExperimentManager,
        ExperimentRunMetadata,
        RunPurpose,
    )
    from examples.notebook_setup import (
        get_or_create_uc_evaluation_dataset,
        preflight_databricks_evidence,
        prepare_notebook_environment,
    )

    environment = prepare_notebook_environment(evidence_destination="databricks")
    evidence = preflight_databricks_evidence(environment)
    dataset = get_or_create_uc_evaluation_dataset(
        evidence=evidence,
        dataset_name="fictional_cost_quality_regression_v1",
        records=EVALUATION_RECORDS,
        mlflow_module=mlflow,
    )
    experiments = ExperimentManager(
        experiment_name=evidence.experiment_name,
        context=evidence.context.tags,
    )
    with experiments.run(
        run_name="cost-quality-simulated-result",
        description=(
            "Simulated cost-quality comparison over the governed fictional "
            "regression dataset; values are not provider measurements."
        ),
        parameters={
            "measurement_source": "simulated_offline_fixture",
            "prompt_digest_sha256": PROMPT_DIGEST,
            "dataset_digest_sha256": DATASET_DIGEST,
        },
        metadata=ExperimentRunMetadata(
            purpose=RunPurpose.RESULT,
            change_id="cost-quality-contract-v1",
            change_summary="Compare eligible logical models after quality gates.",
        ),
    ) as evidence_run:
        mlflow.log_input(dataset, context="cost_quality_evaluation")
        mlflow.log_metrics(
            {
                "quality_eligible_models": float(comparison["quality_eligible"].sum()),
                "cost_comparable_models": float(comparison["cost_comparable"].sum()),
            }
        )
        mlflow.log_table(
            comparison.where(comparison.notna(), None).to_dict(orient="records"),
            artifact_file="evaluation/cost_quality_report.json",
        )
        return {
            "run_id": evidence_run.info.run_id,
            "dataset": dataset.name,
            "dataset_id": dataset.dataset_id,
            "decision": "inconclusive",
        }


__all__ = [
    "CASE_IDS",
    "DATASET_DIGEST",
    "DEMONSTRATION_COST_BUDGET_USD",
    "EVALUATION_RECORDS",
    "INFERENCE_PARAMETERS",
    "MEASUREMENTS",
    "PROMPT_DIGEST",
    "PROMPT_TEMPLATE_FIXTURE",
    "PROMPT_URI",
    "SCORER_SET",
    "build_cost_comparison",
    "comparison_contract",
    "comparison_decision",
    "persist_cost_quality_evidence",
    "preferred_model_under_budget",
]
