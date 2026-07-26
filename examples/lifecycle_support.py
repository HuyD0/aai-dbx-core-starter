"""Shared, deterministic fixtures for the progressive learning examples.

The examples deliberately use a synthetic application so tracing, tracking,
prompt registration, and evaluation can run without a model endpoint or cloud
credentials. The fixed latency, token, and cost values are teaching fixtures,
not a pricing model. Connected applications must record values observed from
their provider, gateway, and billing systems.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "aai-platform.example.yml"
DEFAULT_LOCAL_DIR = ROOT / ".aai" / "local"

EXPERIMENT_PURPOSE = "earnings-summary-quality-cost"
PROMPT_NAME = "earnings_summary"
BASELINE_NAME = "baseline-earnings-summary-prompt-v1"
CHANGE_NAME = "change-cited-earnings-summary-prompt-v2"
CHANGE_ID = "require-exact-earnings-source-id-v2"
CHANGE_SUMMARY = (
    "Require every fictional earnings summary to include its exact source identifier."
)
RELEASE_NAME = "earnings-summary-prompt-v2"
DATASET_NAME = "fictional-earnings-summary-regression-v1"

HYPOTHESIS = (
    "Requiring an exact source identifier will make fictional earnings summaries "
    "easier to verify without reducing fact coverage or recommendation-policy "
    "compliance, while keeping latency, token use, and cost within the declared "
    "budgets."
)
DECISION_RULE = (
    "Release the change only when every critical case passes, mean quality is "
    "at least 0.90, cost coverage is complete, and latency, tokens, and cost "
    "remain within their budgets relative to the baseline."
)

BASELINE_PROMPT = """Summarize the fictional earnings excerpt using only the
supplied facts.
Aster Ridge Systems and every figure in this exercise are fictional.
Do not provide investment advice or recommend buying, selling, or holding securities.

Question: {{question}}
Fictional earnings excerpt: {{earnings_excerpt}}
"""

_CITATION_REQUIREMENT = "Include the supplied source identifier exactly once."
CHANGE_PROMPT = (
    BASELINE_PROMPT.replace(
        "\n\nQuestion:",
        f"\n{_CITATION_REQUIREMENT}\n\nQuestion:",
    )
    + "Source identifier: {{source_id}}\n"
)


Role = Literal["baseline", "change"]


@dataclass(frozen=True)
class EarningsCase:
    case_id: str
    question: str
    earnings_excerpt: str
    source_id: str
    required_facts: tuple[str, ...]

    def evaluation_record(self) -> dict[str, Any]:
        return {
            "inputs": {
                "case_id": self.case_id,
                "question": self.question,
                "earnings_excerpt": self.earnings_excerpt,
                "source_id": self.source_id,
            },
            "expectations": {
                "source_id": self.source_id,
                "required_facts": list(self.required_facts),
                "investment_recommendation_prohibited": True,
            },
        }


CASES = (
    EarningsCase(
        case_id="quarterly-revenue-and-margin",
        question=(
            "What were Aster Ridge Systems' quarterly revenue and operating margin?"
        ),
        earnings_excerpt=(
            "Aster Ridge Systems reported fictional quarterly revenue of "
            "$128.4 million, up 12% year over year, and an operating margin of "
            "18.6%, up from 16.9%."
        ),
        source_id="ARS-FY25-Q2-RESULTS",
        required_facts=("$128.4 million", "12%", "18.6%", "16.9%"),
    ),
    EarningsCase(
        case_id="forward-revenue-and-margin-guidance",
        question=(
            "What fictional revenue and operating-margin guidance did "
            "Aster Ridge Systems provide for next quarter?"
        ),
        earnings_excerpt=(
            "Aster Ridge Systems' fictional next-quarter guidance calls for "
            "revenue of $132 million to $136 million and an operating margin "
            "of 19% to 20%."
        ),
        source_id="ARS-FY25-Q2-GUIDANCE",
        required_facts=("$132 million", "$136 million", "19%", "20%"),
    ),
    EarningsCase(
        case_id="cash-flow-inventory-and-supplier-risk",
        question=(
            "Summarize Aster Ridge Systems' fictional free cash flow, inventory "
            "growth, and supplier risk."
        ),
        earnings_excerpt=(
            "Aster Ridge Systems reported fictional free cash flow of "
            "$21.7 million. Inventory grew 28% year over year, and management "
            "identified single-source supplier concentration as a risk."
        ),
        source_id="ARS-FY25-Q2-CASH-RISK",
        required_facts=("$21.7 million", "28%", "single-source supplier"),
    ),
)


@dataclass(frozen=True)
class OfflineResponse:
    answer: str
    role: Role
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    cost_source: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_output(self) -> dict[str, Any]:
        value = asdict(self)
        value["total_tokens"] = self.total_tokens
        return value


def load_context():
    """Load normal platform context without requiring a local config copy."""

    from aai_core import bootstrap

    configured = os.getenv("AAI_PLATFORM_CONFIG")
    return bootstrap(Path(configured) if configured else DEFAULT_CONFIG)


def lifecycle_experiment_name(context: Any) -> str:
    """Return a stable experiment scope named for the decision it supports."""

    base = context.settings.effective_experiment_name.rstrip("/")
    suffix = f"-{EXPERIMENT_PURPOSE}"
    return base if base.endswith(suffix) else f"{base}{suffix}"


def prepare_mlflow(context: Any) -> str:
    """Configure MLflow locally by default and return the experiment name.

    The Make runner supplies explicit local or Databricks URIs. Direct
    execution falls back to the same ignored local store so a fresh clone
    remains credential-free.
    """

    import mlflow

    local_dir = Path(os.getenv("AAI_EXAMPLE_LOCAL_DIR", DEFAULT_LOCAL_DIR)).resolve()
    default_uri = f"sqlite:///{local_dir / 'mlflow.db'}"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_uri)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    experiment_name = lifecycle_experiment_name(context)
    if not tracking_uri.startswith("databricks"):
        artifact_root = Path(
            os.getenv("AAI_EXAMPLE_ARTIFACT_ROOT", local_dir / "mlruns")
        ).resolve()
        artifact_path = artifact_root / stable_digest(experiment_name)[:16]
        artifact_path.mkdir(parents=True, exist_ok=True)
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_path.as_uri(),
            )
    return experiment_name


def evaluation_data() -> list[dict[str, Any]]:
    """Return fresh records in MLflow GenAI evaluation format."""

    return [case.evaluation_record() for case in CASES]


def dataset_digest() -> str:
    """Hash the exact ordered cases with canonical JSON serialization."""

    return stable_digest(evaluation_data())


def prompt_digest(template: str) -> str:
    return stable_digest({"template": template})


def stable_digest(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def generate_response(
    role: Role,
    *,
    question: str,
    earnings_excerpt: str,
    source_id: str,
    case_id: str,
) -> dict[str, Any]:
    """Return a deterministic model-shaped response for offline learning."""

    del case_id  # The stable ID belongs in lineage, not generated answer text.
    template = BASELINE_PROMPT if role == "baseline" else CHANGE_PROMPT
    rendered_prompt = render_prompt(
        template,
        question=question,
        earnings_excerpt=earnings_excerpt,
        source_id=source_id,
    )
    answer = (
        earnings_excerpt
        if role == "baseline"
        else f"{earnings_excerpt} [source: {source_id}]"
    )
    input_tokens = len(rendered_prompt.split())
    output_tokens = len(answer.split())
    latency_ms = float(18 + input_tokens / 4 + output_tokens / 2)
    if role == "change":
        latency_ms += 1.5
    cost_usd = round(input_tokens * 0.0000002 + output_tokens * 0.0000006, 8)
    return OfflineResponse(
        answer=answer,
        role=role,
        latency_ms=round(latency_ms, 3),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        cost_source="simulated_offline_fixture",
    ).as_output()


def metrics_for(role: Role) -> dict[str, float]:
    outputs = [
        generate_response(role, **record["inputs"]) for record in evaluation_data()
    ]
    qualities = [
        quality_score(output, record["expectations"])
        for output, record in zip(outputs, evaluation_data(), strict=True)
    ]
    critical_passes = [
        critical_case_pass(output, record["expectations"])
        for output, record in zip(outputs, evaluation_data(), strict=True)
    ]
    costs = [
        float(output["cost_usd"])
        for output in outputs
        if output["cost_usd"] is not None
    ]
    return {
        "quality_score": fmean(qualities),
        "critical_case_pass_rate": fmean(critical_passes),
        "citation_rate": fmean(
            citation_score(output, record["expectations"])
            for output, record in zip(outputs, evaluation_data(), strict=True)
        ),
        "recommendation_policy_compliance": fmean(
            recommendation_policy_score(output, record["expectations"])
            for output, record in zip(outputs, evaluation_data(), strict=True)
        ),
        "latency_ms_mean": fmean(float(output["latency_ms"]) for output in outputs),
        "latency_ms_max": max(float(output["latency_ms"]) for output in outputs),
        "input_tokens_total": float(sum(output["input_tokens"] for output in outputs)),
        "output_tokens_total": float(
            sum(output["output_tokens"] for output in outputs)
        ),
        "total_tokens": float(sum(output["total_tokens"] for output in outputs)),
        "cost_usd_total": sum(costs),
        "cost_coverage": len(costs) / len(outputs),
    }


def fact_coverage(output: dict[str, Any], expectations: dict[str, Any]) -> float:
    answer = _normalize_fact_text(output["answer"])
    required = [_normalize_fact_text(value) for value in expectations["required_facts"]]
    return sum(value in answer for value in required) / len(required)


def _normalize_fact_text(value: Any) -> str:
    """Ignore presentation-only formatting when matching required facts.

    LLMs commonly add Markdown, non-breaking spaces, and typographic hyphens.
    Those changes do not alter a financial fact. Citation scoring deliberately
    does not use this normalization because the source-ID contract is exact.
    """

    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.translate(
        str.maketrans(
            {
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
            }
        )
    )
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+(?=%)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def citation_score(output: dict[str, Any], expectations: dict[str, Any]) -> float:
    source_id = str(expectations["source_id"])
    standalone_source = re.compile(
        rf"(?<![A-Za-z0-9_-]){re.escape(source_id)}(?![A-Za-z0-9_-])"
    )
    return float(len(standalone_source.findall(str(output["answer"]))) == 1)


def recommendation_policy_score(
    output: dict[str, Any],
    expectations: dict[str, Any],
) -> float:
    if not expectations.get("investment_recommendation_prohibited", False):
        return 1.0
    answer = str(output["answer"]).casefold()
    recommendation_phrases = (
        "buy the stock",
        "buy shares",
        "sell the stock",
        "sell shares",
        "hold the stock",
        "hold shares",
        "we recommend",
        "i recommend",
    )
    recommendation_patterns = (
        (
            r"\b(?:buy|sell|hold|purchase)\s+(?:the\s+)?"
            r"(?:stock|shares?|securities|aster ridge systems)\b"
        ),
        (
            r"\b(?:you|investors?|we|i)\s+"
            r"(?:should|must|ought to|need to)\s+"
            r"(?:buy|sell|hold|purchase)\b"
        ),
        (
            r"\b(?:recommend|advise)\w*(?:\s+\w+){0,5}\s+"
            r"(?:buy|sell|hold|purchase)\b"
        ),
        r"\b(?:overweight|underweight)\b",
    )
    contains_recommendation = any(
        phrase in answer for phrase in recommendation_phrases
    ) or any(re.search(pattern, answer) for pattern in recommendation_patterns)
    return float(not contains_recommendation)


def quality_score(output: dict[str, Any], expectations: dict[str, Any]) -> float:
    score = fact_coverage(output, expectations) + citation_score(output, expectations)
    return score / 2


def critical_case_pass(output: dict[str, Any], expectations: dict[str, Any]) -> float:
    return float(
        fact_coverage(output, expectations) == 1.0
        and citation_score(output, expectations) == 1.0
        and recommendation_policy_score(output, expectations) == 1.0
    )


def render_prompt(
    template: str,
    *,
    question: str,
    earnings_excerpt: str,
    source_id: str,
) -> str:
    """Render the exact prompt text used for deterministic token estimation."""

    return (
        template.replace("{{question}}", question)
        .replace("{{earnings_excerpt}}", earnings_excerpt)
        .replace("{{source_id}}", source_id)
    )


def release_decision(
    baseline: dict[str, float],
    change: dict[str, float],
) -> dict[str, Any]:
    checks = {
        "critical_cases": change["critical_case_pass_rate"] == 1.0,
        "quality_floor": change["quality_score"] >= 0.90,
        "quality_not_regressed": change["quality_score"] >= baseline["quality_score"],
        "cost_coverage": change["cost_coverage"] == 1.0,
        "latency_budget": (
            change["latency_ms_mean"] <= baseline["latency_ms_mean"] * 1.25
        ),
        "token_budget": change["total_tokens"] <= baseline["total_tokens"] * 1.30,
        "cost_budget": change["cost_usd_total"] <= baseline["cost_usd_total"] * 1.30,
    }
    passed = all(checks.values())
    return {
        "decision": "release_change" if passed else "keep_baseline",
        "release": RELEASE_NAME if passed else "blocked",
        "checks": checks,
    }


def ensure_prompt_version(
    prompts: Any,
    *,
    role: Role,
    template: str,
):
    """Reuse an identical immutable prompt version or register it once.

    ``PromptManager`` supplies governed naming and tags. Native ``MlflowClient``
    search is intentionally visible here because version discovery is a
    provider capability rather than a new SDK abstraction.
    """

    from mlflow import MlflowClient

    qualified_name = prompts.qualify(PROMPT_NAME)
    digest = prompt_digest(template)
    client = MlflowClient()
    if client.get_prompt(qualified_name) is not None:
        for version in client.search_prompt_versions(qualified_name):
            if (
                getattr(version, "template", None) == template
                and getattr(version, "tags", {}).get("aai.prompt_digest") == digest
            ):
                return version
    return prompts.register(
        PROMPT_NAME,
        template,
        commit_message=(
            "Establish the fictional earnings-summary baseline"
            if role == "baseline"
            else "Require an exact source identifier"
        ),
        tags={
            "experiment_role": role,
            "prompt_digest": digest,
        },
    )


def emit_result(payload: dict[str, Any]) -> None:
    print(f"LIFECYCLE_RESULT={json.dumps(payload, sort_keys=True)}")
