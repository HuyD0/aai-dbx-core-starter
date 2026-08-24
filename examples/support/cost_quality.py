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

MEASUREMENT_SOURCE = "simulated_offline_fixture"
MINIMUM_MEAN_QUALITY = 0.90
MINIMUM_ROW_QUALITY = 0.90
# Deterministic scorers make no judge calls on the offline path, so the judge
# cost is genuinely zero. It stays a separate column so a priced connected
# rubric can never masquerade as target-inference cost.
EVALUATION_JUDGE_COST_USD = 0.0
# Simulated, non-vendor price card in USD per 1000 whitespace tokens.
# quality-chat is deliberately absent: its route reports no usage evidence,
# so no price can be applied to it.
SIMULATED_PRICE_CARD_USD_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "draft-chat": {"input": 0.0008, "output": 0.0016},
    "economy-chat": {"input": 0.002, "output": 0.004},
    "general-chat": {"input": 0.008, "output": 0.024},
}


def _rendered_prompt(inputs: JsonRecord) -> str:
    return (
        f"{PROMPT_TEMPLATE_FIXTURE} Question: {inputs['question']} "
        f"Excerpt: {inputs['earnings_excerpt']} Source: {inputs['source_id']}"
    )


def _fixture_response(
    answer: str,
    inputs: JsonRecord,
    *,
    reports_usage: bool,
) -> JsonRecord:
    prompt = _rendered_prompt(inputs)
    usage = (
        {
            "input_tokens": len(prompt.split()),
            "output_tokens": len(answer.split()),
        }
        if reports_usage
        else None
    )
    return {"answer": answer, "usage": usage}


def _draft_chat(inputs: JsonRecord) -> JsonRecord:
    """Cheapest route: grounded and cited, but appends prohibited advice."""

    answer = (
        f"{inputs['earnings_excerpt']} [source: {inputs['source_id']}] "
        "Given these results, we recommend you buy shares now."
    )
    return _fixture_response(answer, inputs, reports_usage=True)


def _economy_chat(inputs: JsonRecord) -> JsonRecord:
    """Terse route: one grounded pass over the excerpt, one citation."""

    answer = f"{inputs['earnings_excerpt']} [source: {inputs['source_id']}]"
    return _fixture_response(answer, inputs, reports_usage=True)


def _general_chat(inputs: JsonRecord) -> JsonRecord:
    """Verbose route: restates the question and repeats the excerpt twice."""

    answer = (
        f"You asked: {inputs['question']} The filing states: "
        f"{inputs['earnings_excerpt']} To restate the disclosed figures in "
        f"full: {inputs['earnings_excerpt']} [source: {inputs['source_id']}]"
    )
    return _fixture_response(answer, inputs, reports_usage=True)


def _quality_chat(inputs: JsonRecord) -> JsonRecord:
    """Strongest route: grounded answer plus caveat, but no usage telemetry."""

    answer = (
        f"{inputs['earnings_excerpt']} These figures are fictional historical "
        "disclosures; the excerpt supports no forward-looking claim. "
        f"[source: {inputs['source_id']}]"
    )
    return _fixture_response(answer, inputs, reports_usage=False)


FIXTURE_MODELS = {
    "draft-chat": _draft_chat,
    "economy-chat": _economy_chat,
    "general-chat": _general_chat,
    "quality-chat": _quality_chat,
}


def fact_coverage(answer: str, expectations: JsonRecord) -> float:
    text = answer.casefold()
    required = [str(fact).casefold() for fact in expectations["required_facts"]]
    return sum(fact in text for fact in required) / len(required)


def citation_score(answer: str, expectations: JsonRecord) -> float:
    return float(answer.count(str(expectations["source_id"])) == 1)


def recommendation_policy_score(answer: str, expectations: JsonRecord) -> float:
    if not expectations.get("no_investment_recommendation", False):
        return 1.0
    text = answer.casefold()
    prohibited = (
        "buy the stock",
        "buy shares",
        "sell the stock",
        "sell shares",
        "we recommend",
        "i recommend",
    )
    return float(not any(phrase in text for phrase in prohibited))


def score_case(answer: str, expectations: JsonRecord) -> JsonRecord:
    fact = fact_coverage(answer, expectations)
    citation = citation_score(answer, expectations)
    policy = recommendation_policy_score(answer, expectations)
    return {
        "fact_coverage": fact,
        "citation_score": citation,
        "recommendation_policy_score": policy,
        "row_quality": (fact + citation + policy) / 3,
        "critical_case_pass": float(fact == 1.0 and citation == 1.0 and policy == 1.0),
    }


def _case_cost_usd(logical_model: str, usage: JsonRecord | None) -> float | None:
    price = SIMULATED_PRICE_CARD_USD_PER_1K_TOKENS.get(logical_model)
    if usage is None or price is None:
        return None
    return round(
        usage["input_tokens"] * price["input"] / 1000
        + usage["output_tokens"] * price["output"] / 1000,
        8,
    )


def run_fixture_evaluation() -> pd.DataFrame:
    """Run every fixture model over the records and score the actual answers."""

    rows = []
    for logical_model, model in FIXTURE_MODELS.items():
        for record in EVALUATION_RECORDS:
            inputs = record["inputs"]
            response = model(inputs)
            usage = response["usage"]
            answer_words = len(response["answer"].split())
            rows.append(
                {
                    "logical_model": logical_model,
                    "case_id": inputs["case_id"],
                    **score_case(response["answer"], record["expectations"]),
                    "input_tokens": usage["input_tokens"] if usage else None,
                    "output_tokens": usage["output_tokens"] if usage else None,
                    "case_cost_usd": _case_cost_usd(logical_model, usage),
                    "latency_ms": round(120.0 + 2.5 * answer_words, 1),
                    "measurement_source": MEASUREMENT_SOURCE,
                }
            )
    return pd.DataFrame(rows)


def comparison_contract() -> JsonRecord:
    return {
        "prompt_uri": PROMPT_URI,
        "prompt_digest": PROMPT_DIGEST,
        "dataset_digest": DATASET_DIGEST,
        "inference_parameters": INFERENCE_PARAMETERS,
        "scorer_set": SCORER_SET,
    }


def _sum_or_none(series: pd.Series) -> int | None:
    observed = series.dropna()
    if len(observed) != len(series):
        return None
    return int(observed.sum())


def _aggregate_model(logical_model: str, frame: pd.DataFrame) -> JsonRecord:
    costs = frame["case_cost_usd"].dropna()
    covered = len(costs) == len(frame)
    return {
        "logical_model": logical_model,
        "quality_score": round(float(frame["row_quality"].mean()), 3),
        "minimum_row_quality": round(float(frame["row_quality"].min()), 3),
        "critical_case_pass_rate": round(float(frame["critical_case_pass"].mean()), 3),
        "recommendation_policy_compliance": round(
            float(frame["recommendation_policy_score"].mean()), 3
        ),
        "latency_ms_mean": round(float(frame["latency_ms"].mean()), 1),
        "input_tokens": _sum_or_none(frame["input_tokens"]),
        "output_tokens": _sum_or_none(frame["output_tokens"]),
        # Unavailable cost stays None (unknown), never zero.
        "target_inference_cost_usd": round(float(costs.sum()), 8) if covered else None,
        "evaluation_judge_cost_usd": EVALUATION_JUDGE_COST_USD,
        "cost_coverage": round(len(costs) / len(frame), 2),
        "measurement_source": MEASUREMENT_SOURCE,
    }


def build_cost_comparison() -> pd.DataFrame:
    """Aggregate the executed fixture evaluation into one comparison frame."""

    cases = run_fixture_evaluation()
    comparison = pd.DataFrame(
        _aggregate_model(logical_model, frame)
        for logical_model, frame in cases.groupby("logical_model", sort=False)
    )
    comparison["total_tokens"] = (
        comparison["input_tokens"] + comparison["output_tokens"]
    )
    comparison["quality_eligible"] = (
        (comparison["quality_score"] >= MINIMUM_MEAN_QUALITY)
        & (comparison["minimum_row_quality"] >= MINIMUM_ROW_QUALITY)
        & (comparison["critical_case_pass_rate"] == 1.0)
        & (comparison["recommendation_policy_compliance"] == 1.0)
    )
    comparison["cost_comparable"] = (
        comparison["quality_eligible"]
        & (comparison["cost_coverage"] == 1.0)
        & comparison["target_inference_cost_usd"].notna()
    )
    comparison["quality_per_cost"] = (
        comparison["quality_score"]
        .div(comparison["target_inference_cost_usd"])
        .where(comparison["cost_comparable"])
    )
    return comparison


def quality_gate_report(comparison: pd.DataFrame) -> pd.DataFrame:
    """Explain which quality gates each model passed or failed, and why."""

    rows = []
    for row in comparison.itertuples(index=False):
        reasons = []
        if row.quality_score < MINIMUM_MEAN_QUALITY:
            reasons.append(f"mean quality {row.quality_score} < {MINIMUM_MEAN_QUALITY}")
        if row.minimum_row_quality < MINIMUM_ROW_QUALITY:
            reasons.append(
                f"worst row {row.minimum_row_quality} < {MINIMUM_ROW_QUALITY}"
            )
        if row.critical_case_pass_rate < 1.0:
            reasons.append(f"critical pass rate {row.critical_case_pass_rate} < 1.0")
        if row.recommendation_policy_compliance < 1.0:
            reasons.append("emitted a prohibited investment recommendation")
        rows.append(
            {
                "logical_model": row.logical_model,
                "quality_eligible": bool(row.quality_eligible),
                "reason": "; ".join(reasons) or "passed every quality gate",
            }
        )
    return pd.DataFrame(rows)


def cost_ranking(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rank only cost-comparable survivors: quality first, then price."""

    survivors = comparison.loc[comparison["cost_comparable"]]
    return survivors.sort_values(
        ["quality_score", "target_inference_cost_usd"],
        ascending=[False, True],
    )[
        [
            "logical_model",
            "quality_score",
            "total_tokens",
            "latency_ms_mean",
            "target_inference_cost_usd",
            "evaluation_judge_cost_usd",
            "quality_per_cost",
        ]
    ].reset_index(
        drop=True
    )


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
    quality_ineligible_models = comparison.loc[
        ~comparison["quality_eligible"], "logical_model"
    ].tolist()
    top_quality_without_cost_evidence = comparison.loc[
        (comparison["quality_score"] == comparison["quality_score"].max())
        & ~comparison["cost_comparable"],
        "logical_model",
    ].tolist()
    return {
        "preferred_under_demonstration_budget": preferred_model_under_budget(
            comparison
        ),
        "quality_ineligible_models": quality_ineligible_models,
        "unknown_cost_models": unknown_cost_models,
        "top_quality_without_cost_evidence": top_quality_without_cost_evidence,
        "decision": "inconclusive",
        "release": "blocked_until_connected_evaluation",
        "reason": (
            "the comparison ran offline fixture models; rerun the exact "
            "contract against configured logical models and authoritative "
            "cost evidence"
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
    "EVALUATION_JUDGE_COST_USD",
    "EVALUATION_RECORDS",
    "FIXTURE_MODELS",
    "INFERENCE_PARAMETERS",
    "MEASUREMENT_SOURCE",
    "MINIMUM_MEAN_QUALITY",
    "MINIMUM_ROW_QUALITY",
    "PROMPT_DIGEST",
    "PROMPT_TEMPLATE_FIXTURE",
    "PROMPT_URI",
    "SCORER_SET",
    "SIMULATED_PRICE_CARD_USD_PER_1K_TOKENS",
    "build_cost_comparison",
    "citation_score",
    "comparison_contract",
    "comparison_decision",
    "cost_ranking",
    "fact_coverage",
    "persist_cost_quality_evidence",
    "preferred_model_under_budget",
    "quality_gate_report",
    "recommendation_policy_score",
    "run_fixture_evaluation",
    "score_case",
]
