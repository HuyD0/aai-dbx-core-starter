"""Reusable mechanics for the model-selection reference lesson.

All values in the default path are deterministic teaching fixtures.  They are
not provider benchmarks and cannot authorize a production release.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import Field

from aai_core.contracts import ContractModel
from aai_core.evaluation import GatePolicy, MetricDirection, MetricRule, apply_gate
from aai_core.providers import ModelCapabilities, ModelResponse
from aai_core.structured import generate_typed
from aai_core.testing import dev_context

JsonRecord = dict[str, Any]


class GoldenFixtureModel:
    """Deterministic SDK-compatible model used only by the offline lesson."""

    provider = "fixture"
    model = "simulated-offline-fixture"
    capabilities = ModelCapabilities(structured_output=True)
    native_client = None

    def __init__(
        self,
        logical_name: str,
        replies: Mapping[str, str],
        latency_ms: float,
        usage: Mapping[str, int],
    ) -> None:
        self.logical_name = logical_name
        self.replies = dict(replies)
        self.latency_ms = float(latency_ms)
        self.usage = dict(usage)
        self.requests: list[dict[str, Any]] = []

    def create_native_async_client(self) -> None:
        raise RuntimeError("Offline fixture provides no native async client")

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> ModelResponse:
        prompt = str(messages[-1]["content"])
        case_line = next(
            line for line in prompt.splitlines() if line.startswith("Case ID: ")
        )
        case_id = case_line.removeprefix("Case ID: ")
        self.requests.append({"messages": list(messages), **options})
        return ModelResponse(
            content=self.replies[case_id],
            provider=self.provider,
            logical_name=self.logical_name,
            model=self.model,
            latency_ms=self.latency_ms,
            usage=self.usage,
        )


GOLDEN_CASES: tuple[JsonRecord, ...] = (
    {
        "inputs": {
            "case_id": "missing-purchase-order",
            "request": "Invoice INV-104 has no purchase order.",
        },
        "expectations": {
            "required_terms": ("ESCALATE", "missing purchase order"),
            "critical": True,
        },
    },
    {
        "inputs": {
            "case_id": "duplicate-invoice",
            "request": "INV-205 duplicates paid invoice INV-199.",
        },
        "expectations": {"required_terms": ("REJECT", "duplicate"), "critical": True},
    },
    {
        "inputs": {
            "case_id": "tax-mismatch",
            "request": "Invoice tax is 15%; the purchase order says 13%.",
        },
        "expectations": {
            "required_terms": ("ESCALATE", "tax mismatch"),
            "critical": False,
        },
    },
)

BASELINE_REPLIES = {
    "missing-purchase-order": "ESCALATE: missing purchase order.",
    "duplicate-invoice": "REJECT: duplicate invoice.",
    "tax-mismatch": "ESCALATE: tax mismatch requires review.",
}
CHANGE_REPLIES = {
    "missing-purchase-order": "APPROVE: vendor history looks normal.",
    "duplicate-invoice": "REJECT: duplicate invoice.",
    "tax-mismatch": "ESCALATE: tax mismatch requires review.",
}


def golden_fixture_context() -> Any:
    """Return a fresh context with both deterministic comparison models."""

    context = dev_context()
    context.providers.register_model(
        "baseline-chat",
        GoldenFixtureModel(
            "baseline-chat",
            BASELINE_REPLIES,
            latency_ms=420,
            usage={"prompt_tokens": 65, "completion_tokens": 18},
        ),
    )
    context.providers.register_model(
        "change-chat",
        GoldenFixtureModel(
            "change-chat",
            CHANGE_REPLIES,
            latency_ms=310,
            usage={"prompt_tokens": 65, "completion_tokens": 15},
        ),
    )
    return context


def _required_terms_pass(response: str, required_terms: Iterable[str]) -> bool:
    normalized = response.casefold()
    return all(term.casefold() in normalized for term in required_terms)


def run_golden_ab(
    context: Any,
    logical_models: Iterable[str],
    cases: Iterable[Mapping[str, Any]],
) -> JsonRecord:
    """Run the detailed, same-case comparison shown by the reference lesson."""

    model_names = tuple(logical_models)
    case_records = tuple(cases)
    parameters = {"temperature": 0.0, "max_tokens": 80}
    dataset_payload = [
        {
            "inputs": case["inputs"],
            "expectations": {
                **case["expectations"],
                "required_terms": list(case["expectations"]["required_terms"]),
            },
        }
        for case in case_records
    ]
    dataset_digest = hashlib.sha256(
        json.dumps(dataset_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rows: list[JsonRecord] = []
    for logical_model in model_names:
        model = context.providers.model(logical_model)
        for case in case_records:
            inputs = case["inputs"]
            prompt = (
                f"Case ID: {inputs['case_id']}\n"
                f"Route this invoice: {inputs['request']}"
            )
            response = model.generate(
                [
                    {
                        "role": "system",
                        "content": "Return a routing action and short reason.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **parameters,
            )
            usage = dict(response.usage)
            input_tokens = int(
                usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
            )
            output_tokens = int(
                usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
            )
            rows.append(
                {
                    "logical_model": logical_model,
                    "case_id": inputs["case_id"],
                    "passed": _required_terms_pass(
                        response.content,
                        case["expectations"]["required_terms"],
                    ),
                    "critical": case["expectations"]["critical"],
                    "latency_ms": response.latency_ms,
                    "tokens": usage.get("total_tokens", input_tokens + output_tokens),
                }
            )
    summary: dict[str, JsonRecord] = {}
    for logical_model in model_names:
        model_rows = [row for row in rows if row["logical_model"] == logical_model]
        critical_rows = [row for row in model_rows if row["critical"]]
        summary[logical_model] = {
            "accuracy": sum(row["passed"] for row in model_rows) / len(model_rows),
            "critical_pass_rate": sum(row["passed"] for row in critical_rows)
            / len(critical_rows),
            "mean_latency_ms": sum(row["latency_ms"] for row in model_rows)
            / len(model_rows),
            "total_tokens": sum(row["tokens"] for row in model_rows),
        }
    return {
        "measurement_source": "simulated_offline_fixture",
        "dataset_digest_sha256": dataset_digest,
        "parameters": parameters,
        "rows": rows,
        "summary": summary,
    }


def run_golden_comparison(
    context: Any,
    logical_models: Iterable[str],
    cases: Iterable[Mapping[str, Any]],
) -> JsonRecord:
    """Exercise solution: compare declared models on one ordered case set."""

    model_names = tuple(logical_models)
    case_records = tuple(cases)
    if not model_names or not case_records:
        raise ValueError("At least one logical model and case are required.")
    case_ids = [case["inputs"]["case_id"] for case in case_records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Golden case IDs must be unique.")

    rows: list[JsonRecord] = []
    for logical_model in model_names:
        model = context.providers.model(logical_model)
        for case in case_records:
            inputs = case["inputs"]
            response = model.generate(
                [{"role": "user", "content": inputs["request"]}],
                temperature=0.0,
                max_tokens=80,
            )
            rows.append(
                {
                    "logical_model": logical_model,
                    "case_id": inputs["case_id"],
                    "passed": _required_terms_pass(
                        response.content,
                        case["expectations"]["required_terms"],
                    ),
                }
            )
    summary = {
        name: {
            "accuracy": sum(
                row["passed"] for row in rows if row["logical_model"] == name
            )
            / sum(row["logical_model"] == name for row in rows)
        }
        for name in model_names
    }
    return {"rows": rows, "summary": summary}


class PairwiseVerdict(ContractModel):
    winner: Literal["A", "B", "tie"]
    rationale: str = Field(min_length=1)


class PairwiseJudgeFixture:
    """Deterministic structured-output judge for position-bias teaching."""

    logical_name = "judge-model"
    provider = "fixture"
    model = "simulated-offline-fixture"
    capabilities = ModelCapabilities(structured_output=True)
    native_client = None

    def create_native_async_client(self) -> None:
        raise RuntimeError("Offline fixture provides no native async client")

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> ModelResponse:
        del options
        payload = json.loads(str(messages[-1]["content"]))
        required_terms = payload["required_terms"]

        def coverage(label: str) -> int:
            response = payload[f"response_{label}"].casefold()
            return sum(term.casefold() in response for term in required_terms)

        score_a, score_b = coverage("A"), coverage("B")
        winner = "tie" if score_a == score_b else ("A" if score_a > score_b else "B")
        content = json.dumps(
            {
                "winner": winner,
                "rationale": "Selected by required-fact coverage fixture.",
            }
        )
        return ModelResponse(
            content=content,
            provider=self.provider,
            logical_name=self.logical_name,
            model=self.model,
            latency_ms=95.0,
            usage={"prompt_tokens": 110, "completion_tokens": 22},
        )


PAIRWISE_CASES: tuple[JsonRecord, ...] = (
    {
        "case_id": "policy-clause",
        "required_terms": ["policy P-14", "appeal"],
        "responses": {
            "baseline-chat": "Policy P-14 applies; you may appeal.",
            "change-chat": "You may appeal this decision.",
        },
    },
    {
        "case_id": "next-steps",
        "required_terms": ["30 days", "case portal"],
        "responses": {
            "baseline-chat": "Respond within 30 days.",
            "change-chat": "Respond within 30 days in the case portal.",
        },
    },
    {
        "case_id": "contact-channel",
        "required_terms": ["claims desk"],
        "responses": {
            "baseline-chat": "Contact the claims desk.",
            "change-chat": "Contact the claims desk.",
        },
    },
)

SAMPLE_VERDICTS: tuple[JsonRecord, ...] = (
    {
        "case_id": "case-1",
        "model_A": "baseline-chat",
        "model_B": "change-chat",
        "winner_label": "B",
    },
    {
        "case_id": "case-1",
        "model_A": "change-chat",
        "model_B": "baseline-chat",
        "winner_label": "A",
    },
    {
        "case_id": "case-2",
        "model_A": "baseline-chat",
        "model_B": "change-chat",
        "winner_label": "tie",
    },
    {
        "case_id": "case-2",
        "model_A": "change-chat",
        "model_B": "baseline-chat",
        "winner_label": "tie",
    },
)


def pairwise_judge_fixture() -> Any:
    context = dev_context()
    context.providers.register_model("judge-model", PairwiseJudgeFixture())
    return context.providers.model("judge-model")


def run_balanced_pairwise_judge(
    judge: Any,
    cases: Iterable[Mapping[str, Any]],
    logical_models: Sequence[str],
) -> JsonRecord:
    baseline, change = logical_models
    records: list[JsonRecord] = []
    for case in cases:
        for model_a, model_b in ((baseline, change), (change, baseline)):
            payload = {
                "rubric": "Prefer complete, grounded required facts.",
                "required_terms": case["required_terms"],
                "response_A": case["responses"][model_a],
                "response_B": case["responses"][model_b],
            }
            verdict = generate_typed(
                judge,
                [{"role": "user", "content": json.dumps(payload)}],
                response_model=PairwiseVerdict,
                temperature=0.0,
                max_tokens=120,
            )
            winner_model = (
                None
                if verdict.winner == "tie"
                else (model_a if verdict.winner == "A" else model_b)
            )
            records.append(
                {
                    "case_id": case["case_id"],
                    "model_A": model_a,
                    "model_B": model_b,
                    "winner_label": verdict.winner,
                    "winner_model": winner_model,
                }
            )
    summary = summarize_pairwise_verdicts(records, logical_models)
    return {
        "measurement_source": "simulated_offline_fixture",
        "verdicts": records,
        "summary": {
            name: {
                "wins": values["wins"],
                "decisive_win_rate": values["decisive_win_rate"],
                "tie_rate": summary["tie_rate"],
            }
            for name, values in summary["models"].items()
        },
        "position_A_win_rate": summary["position_A_win_rate"],
        "position_B_win_rate": summary["position_B_win_rate"],
    }


def summarize_pairwise_verdicts(
    verdicts: Iterable[Mapping[str, Any]],
    logical_models: Iterable[str],
) -> JsonRecord:
    """Exercise solution: report logical-model wins independent of position."""

    records = tuple(verdicts)
    model_names = tuple(logical_models)
    if not records or len(model_names) != len(set(model_names)):
        raise ValueError("Verdicts and unique logical model names are required.")
    counts: dict[str, JsonRecord] = {
        name: {"wins": 0, "losses": 0, "ties": 0} for name in model_names
    }
    position_wins = {"A": 0, "B": 0}
    tie_count = 0
    for record in records:
        label = record["winner_label"]
        model_a, model_b = record["model_A"], record["model_B"]
        if model_a not in counts or model_b not in counts or model_a == model_b:
            raise ValueError("Each verdict must compare two declared models.")
        if label == "tie":
            counts[model_a]["ties"] += 1
            counts[model_b]["ties"] += 1
            tie_count += 1
            continue
        if label not in position_wins:
            raise ValueError(f"Unsupported winner label: {label!r}")
        winner, loser = (model_a, model_b) if label == "A" else (model_b, model_a)
        counts[winner]["wins"] += 1
        counts[loser]["losses"] += 1
        position_wins[label] += 1
    for count in counts.values():
        decisive = count["wins"] + count["losses"]
        count["decisive_win_rate"] = count["wins"] / decisive if decisive else None
    total = len(records)
    return {
        "models": counts,
        "tie_rate": tie_count / total,
        "position_A_win_rate": position_wins["A"] / total,
        "position_B_win_rate": position_wins["B"] / total,
    }


SIMULATED_SESSION_OBSERVATIONS: dict[str, list[JsonRecord]] = {
    "baseline-chat": [
        {
            "session_id": "b-01",
            "turns": 4,
            "input_tokens": 3200,
            "output_tokens": 620,
            "latency_ms": 5100,
            "quality_passed": True,
        },
        {
            "session_id": "b-02",
            "turns": 5,
            "input_tokens": 3900,
            "output_tokens": 760,
            "latency_ms": 6200,
            "quality_passed": True,
        },
    ],
    "change-chat": [
        {
            "session_id": "c-01",
            "turns": 4,
            "input_tokens": 3500,
            "output_tokens": 900,
            "latency_ms": 4300,
            "quality_passed": True,
        },
        {
            "session_id": "c-02",
            "turns": 5,
            "input_tokens": 4400,
            "output_tokens": 1120,
            "latency_ms": 5000,
            "quality_passed": True,
        },
    ],
}
SIMULATED_APPROVED_PRICE_CARD: dict[str, JsonRecord] = {
    "baseline-chat": {"input_usd_per_million": 2.0, "output_usd_per_million": 8.0},
    "change-chat": {"input_usd_per_million": 0.8, "output_usd_per_million": 3.2},
}


def estimate_session_economics(
    usage: Mapping[str, float],
    price: Mapping[str, float | None],
    sessions_per_month: int,
) -> JsonRecord:
    """Exercise solution: calculate throughput and fail-closed cost evidence."""

    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    latency_ms = usage["latency_ms"]
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")
    if latency_ms <= 0 or sessions_per_month < 0:
        raise ValueError("Latency must be positive and session volume non-negative.")
    input_rate = price.get("input_usd_per_million")
    output_rate = price.get("output_usd_per_million")
    session_cost = (
        None
        if input_rate is None or output_rate is None
        else input_tokens / 1_000_000 * input_rate
        + output_tokens / 1_000_000 * output_rate
    )
    return {
        "output_tokens_per_second": output_tokens / (latency_ms / 1000),
        "session_cost_usd": session_cost,
        "monthly_tco_usd": (
            None if session_cost is None else session_cost * sessions_per_month
        ),
        "cost_coverage": float(session_cost is not None),
    }


def compare_session_economics(
    observations: Mapping[str, Sequence[Mapping[str, Any]]],
    price_card: Mapping[str, Mapping[str, float | None]],
    sessions_per_month: int,
) -> list[JsonRecord]:
    """Aggregate complete-session economics only after quality passes."""

    report: list[JsonRecord] = []
    for logical_model, sessions in observations.items():
        price = price_card.get(logical_model, {})
        estimates = [
            estimate_session_economics(session, price, sessions_per_month)
            for session in sessions
        ]
        known_costs = [
            row["session_cost_usd"]
            for row in estimates
            if row["session_cost_usd"] is not None
        ]
        coverage = len(known_costs) / len(sessions)
        mean_cost = sum(known_costs) / len(known_costs) if coverage == 1.0 else None
        total_output = sum(session["output_tokens"] for session in sessions)
        total_latency_seconds = (
            sum(session["latency_ms"] for session in sessions) / 1000
        )
        quality_pass_rate = sum(
            session["quality_passed"] for session in sessions
        ) / len(sessions)
        report.append(
            {
                "logical_model": logical_model,
                "mean_turns_per_session": sum(session["turns"] for session in sessions)
                / len(sessions),
                "mean_input_tokens_per_session": sum(
                    session["input_tokens"] for session in sessions
                )
                / len(sessions),
                "mean_output_tokens_per_session": total_output / len(sessions),
                "output_tokens_per_second": total_output / total_latency_seconds,
                "mean_session_cost_usd": mean_cost,
                "monthly_tco_usd": (
                    None if mean_cost is None else mean_cost * sessions_per_month
                ),
                "cost_coverage": coverage,
                "quality_pass_rate": quality_pass_rate,
                "cost_comparable": coverage == 1.0 and quality_pass_rate == 1.0,
                "measurement_source": "simulated_offline_fixture",
            }
        )
    return report


class ModelGovernanceEvidence(ContractModel):
    logical_model: str
    evidence_version: str
    evidence_source: str
    zero_data_retention: bool | None
    retention_days: int | None
    customer_data_training_disabled: bool | None
    data_residencies: tuple[str, ...]
    keyless_authentication: bool | None
    private_network_or_controlled_egress: bool | None
    audit_logging: bool | None
    enterprise_terms_approved: bool | None
    liability_terms_approved: bool | None


REQUIRED_BOOLEAN_CONTROLS = (
    "zero_data_retention",
    "customer_data_training_disabled",
    "keyless_authentication",
    "private_network_or_controlled_egress",
    "audit_logging",
    "enterprise_terms_approved",
    "liability_terms_approved",
)


def run_governance_preflight(
    evidence: ModelGovernanceEvidence,
    required_residency: str,
) -> JsonRecord:
    metrics = {
        f"governance/{name}": float(getattr(evidence, name))
        for name in REQUIRED_BOOLEAN_CONTROLS
        if getattr(evidence, name) is not None
    }
    if evidence.retention_days is not None:
        metrics["governance/retention_days"] = float(evidence.retention_days)
    metrics["governance/required_residency"] = float(
        required_residency in evidence.data_residencies
    )
    rules = [
        MetricRule(
            metric=f"governance/{name}",
            direction=MetricDirection.HIGHER,
            required=1.0,
        )
        for name in REQUIRED_BOOLEAN_CONTROLS
    ]
    rules.extend(
        [
            MetricRule(
                metric="governance/retention_days",
                direction=MetricDirection.LOWER,
                required=0.0,
            ),
            MetricRule(
                metric="governance/required_residency",
                direction=MetricDirection.HIGHER,
                required=1.0,
            ),
        ]
    )
    gate = apply_gate(metrics, policy=GatePolicy(rules=tuple(rules)))
    return {
        "logical_model": evidence.logical_model,
        "evidence_version": evidence.evidence_version,
        "evidence_source": evidence.evidence_source,
        "approved": gate.passed,
        "metrics": dict(gate.metrics),
        "failures": [
            {"metric": failure.metric, "reason": failure.reason}
            for failure in gate.failures
        ],
        "measurement_source": "simulated_offline_fixture",
    }


def gate_model_governance(
    evidence: Mapping[str, Any],
    required_residency: str,
) -> JsonRecord:
    """Exercise solution: fail closed when any governance evidence is absent."""

    controls = {
        "zero_data_retention": evidence.get("zero_data_retention") is True,
        "retention_days": evidence.get("retention_days") == 0,
        "customer_data_training_disabled": evidence.get(
            "customer_data_training_disabled"
        )
        is True,
        "required_residency": required_residency
        in (evidence.get("data_residencies") or ()),
        "keyless_authentication": evidence.get("keyless_authentication") is True,
        "private_network_or_controlled_egress": evidence.get(
            "private_network_or_controlled_egress"
        )
        is True,
        "audit_logging": evidence.get("audit_logging") is True,
        "enterprise_terms_approved": evidence.get("enterprise_terms_approved") is True,
        "liability_terms_approved": evidence.get("liability_terms_approved") is True,
    }
    failures = [name for name, compliant in controls.items() if not compliant]
    return {"approved": not failures, "failures": failures}


def governance_fixtures() -> tuple[ModelGovernanceEvidence, ModelGovernanceEvidence]:
    """Return one eligible baseline and one intentionally blocked change."""

    common = {
        "evidence_version": "provider-catalog-fixture-v1",
        "evidence_source": "approved-external-provider-catalog-fixture",
        "customer_data_training_disabled": True,
        "keyless_authentication": True,
        "audit_logging": True,
        "enterprise_terms_approved": True,
    }
    return (
        ModelGovernanceEvidence(
            logical_model="baseline-chat",
            zero_data_retention=True,
            retention_days=0,
            data_residencies=("canada",),
            private_network_or_controlled_egress=True,
            liability_terms_approved=True,
            **common,
        ),
        ModelGovernanceEvidence(
            logical_model="change-chat",
            zero_data_retention=False,
            retention_days=30,
            data_residencies=("canada", "united-states"),
            private_network_or_controlled_egress=False,
            liability_terms_approved=False,
            **common,
        ),
    )


UNSAFE_GOVERNANCE_EVIDENCE: JsonRecord = {
    "zero_data_retention": False,
    "retention_days": 30,
    "customer_data_training_disabled": True,
    "data_residencies": ("canada",),
    "keyless_authentication": True,
    "private_network_or_controlled_egress": False,
    "audit_logging": True,
    "enterprise_terms_approved": True,
    "liability_terms_approved": False,
}


__all__ = [
    "GOLDEN_CASES",
    "PAIRWISE_CASES",
    "SAMPLE_VERDICTS",
    "SIMULATED_APPROVED_PRICE_CARD",
    "SIMULATED_SESSION_OBSERVATIONS",
    "UNSAFE_GOVERNANCE_EVIDENCE",
    "ModelGovernanceEvidence",
    "compare_session_economics",
    "estimate_session_economics",
    "gate_model_governance",
    "golden_fixture_context",
    "governance_fixtures",
    "pairwise_judge_fixture",
    "run_balanced_pairwise_judge",
    "run_golden_ab",
    "run_golden_comparison",
    "run_governance_preflight",
    "summarize_pairwise_verdicts",
]
