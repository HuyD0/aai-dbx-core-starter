"""Release gate: outcome judges plus trajectory and decision scoring.

Runs the real agent per case (pinned prompt version), scores its traced TOOL
spans against expected names and arguments, compares decision claims with
observed execution and reviewed expectations, checks root/TOOL status as
operations evidence, applies Correctness/Safety judges, and enforces every
threshold plus baseline regression. Publishes the report to the CI summary.
Credentialed path only; pull-request CI runs ``evals/offline_checks.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from contextvars import copy_context
from pathlib import Path
from urllib.parse import urlsplit

import mlflow
from mlflow.genai.scorers import Correctness, Safety

from aai_core import bootstrap
from aai_core.agents import AgentRequest
from aai_core.evaluation import (
    GatePolicy,
    MetricRule,
    apply_gate,
)
from aai_core.experiments import record_reproducibility
from aai_core.providers.types import ProviderConfigurationError
from aai_core.tracing import TraceIntegration, traced
from app.agent import ToolAgent
from app.config import DATASET_NAME, PROMPT_NAME
from app.controls import DEFAULT_AGENT_LIMITS
from app.tool_scoring import (
    decision_action_consistency_scorer,
    decision_tool_appropriateness_scorer,
    exact_tool_call_scorer,
    trace_execution_success_scorer,
)
from app.tools import build_registry

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evals" / "baseline.json"


def load_thresholds() -> list[MetricRule]:
    config = json.loads((ROOT / "evals" / "gate_config.json").read_text("utf-8"))
    return [MetricRule(**threshold) for threshold in config["thresholds"]]


def load_baseline() -> dict[str, float]:
    if not BASELINE.exists():
        print(
            "No evals/baseline.json yet; regression checks activate after the "
            "first release records one with --update-baseline."
        )
        return {}
    metrics = json.loads(BASELINE.read_text(encoding="utf-8"))["metrics"]
    return {name: float(value) for name, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-version",
        type=int,
        required=True,
        help="Exact immutable Prompt Registry version to evaluate.",
    )
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    if args.prompt_version < 1:
        parser.error("--prompt-version must be a positive integer")

    context = bootstrap(ROOT / "aai-platform.yml")
    context.configure_tracing(integration=TraceIntegration.SDK)
    version = args.prompt_version
    cases = json.loads(
        (ROOT / "evals" / "data" / "release_cases.json").read_text(encoding="utf-8")
    )

    thresholds = load_thresholds()
    baseline = load_baseline()
    judge_model, target_identity, judge_identity = _evaluation_models(context.settings)
    policy = GatePolicy(
        rules=tuple(thresholds),
        allow_missing_regression_baseline=args.update_baseline and not baseline,
    )
    prompt_uri = f"prompts:/{context.prompts.qualify(PROMPT_NAME)}/{version}"
    dataset_name = (
        f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
    )
    dataset = _load_release_dataset(dataset_name, cases)
    tool_schema_digest = _tool_schema_digest()
    limit_parameters = _agent_limit_parameters()

    # One governed MLflow run holds outcome judges, trajectory, decision, and
    # operational scores, pinned prompt/dataset identity, and the verdict.
    manager = context.experiments
    with asyncio.Runner() as runner:
        agent = ToolAgent(context, prompt_version=version)
        predict_fn = _build_predict_fn(agent, runner)

        try:
            with manager.run(
                run_name=f"agent-prompt-v{version}-validation-gate",
                parameters={
                    "prompt_version": version,
                    "prompt_uri": prompt_uri,
                    "evaluation_dataset": dataset_name,
                    "evaluation_dataset_id": dataset.dataset_id,
                    "evaluation_dataset_digest": dataset.digest,
                    "case_count": len(cases),
                    "target_model": target_identity,
                    "judge_model": judge_identity,
                    "tool_schema_digest": tool_schema_digest,
                    "agent_limits_digest": DEFAULT_AGENT_LIMITS.digest,
                    **limit_parameters,
                },
            ) as active_run:
                record_reproducibility()
                native_result = mlflow.genai.evaluate(
                    data=dataset,
                    predict_fn=predict_fn,
                    scorers=[
                        exact_tool_call_scorer(),
                        decision_action_consistency_scorer(),
                        decision_tool_appropriateness_scorer(),
                        trace_execution_success_scorer(),
                        Correctness(model=judge_model),
                        Safety(model=judge_model),
                    ],
                )
                report = apply_gate(
                    native_result,
                    policy=policy,
                    baseline_metrics=baseline,
                )
                mlflow.log_metrics(dict(report.metrics))
                mlflow.log_params(
                    {
                        "gate_policy_digest": report.policy_digest,
                        "gate_baseline_digest": report.baseline_digest or "none",
                    }
                )
                mlflow.set_tags(
                    {
                        "aai.gate_passed": str(report.passed).lower(),
                        "aai.target_model": target_identity,
                        "aai.judge_model": judge_identity,
                    }
                )
        finally:
            runner.run(agent.aclose())

    report.require_passed()
    if args.update_baseline:
        BASELINE.write_text(
            json.dumps({"metrics": dict(report.metrics)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(
        {
            "prompt_version": version,
            "evaluation_run": active_run.info.run_id,
            "metrics": report.metrics,
            "baseline_updated": args.update_baseline,
        }
    )


def _build_predict_fn(agent: ToolAgent, runner: asyncio.Runner):
    """Keep the complete multi-step agent trajectory in one MLflow trace."""

    @traced(name="agent.evaluate", span_type="AGENT")
    def predict_fn(question: str) -> str:
        response = runner.run(
            agent.ainvoke(
                AgentRequest(messages=[{"role": "user", "content": question}])
            ),
            context=copy_context(),
        )
        return response.content

    return predict_fn


def _evaluation_models(settings) -> tuple[str, str, str]:
    target = _model_config(settings, "general-chat")
    judge = _model_config(settings, "judge-model")
    if judge["provider"] != "databricks":
        raise ProviderConfigurationError(
            "judge-model must resolve to a governed Databricks serving endpoint"
        )
    if (
        judge["provider"].casefold() == target["provider"].casefold()
        and judge["deployment"].casefold() == target["deployment"].casefold()
    ):
        raise ProviderConfigurationError(
            "judge-model must use a deployment distinct from general-chat; "
            "a release gate cannot rely on the target judging itself"
        )
    target_identity = _evaluation_model_identity(target)
    judge_identity = f"{judge['provider']}:{judge['deployment']}"
    return f"endpoints:/{judge['deployment']}", target_identity, judge_identity


def _model_config(settings, logical_name: str) -> dict[str, str]:
    config = settings.models.get(logical_name)
    if not isinstance(config, Mapping):
        raise ProviderConfigurationError(f"{logical_name} must be configured")
    provider = config.get("provider")
    deployment = config.get("deployment")
    if not isinstance(provider, str) or not provider.strip():
        raise ProviderConfigurationError(f"{logical_name} requires a provider")
    if not isinstance(deployment, str) or not deployment.strip():
        raise ProviderConfigurationError(f"{logical_name} requires a deployment")
    normalized = {
        "provider": provider.strip(),
        "deployment": deployment.strip(),
    }
    if "endpoint" in config:
        normalized["endpoint"] = config["endpoint"]
    return normalized


def _evaluation_model_identity(config: Mapping) -> str:
    identity = f"{config['provider']}:{config['deployment']}"
    if str(config["provider"]).casefold() == "foundry":
        identity += "@endpoint-sha256:" + _endpoint_sha256(config.get("endpoint"))
    return identity


def _endpoint_sha256(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderConfigurationError(
            "Foundry model configuration requires an HTTPS endpoint"
        )
    endpoint = value.strip()
    parsed = urlsplit(endpoint)
    if (
        any(character.isspace() or ord(character) < 32 for character in endpoint)
        or parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderConfigurationError(
            "Foundry endpoint must be HTTPS without userinfo, query, or fragment"
        )
    try:
        port = parsed.port
    except ValueError as error:
        raise ProviderConfigurationError(
            "Foundry endpoint contains an invalid port"
        ) from error
    raw_hostname = parsed.hostname.rstrip(".")
    try:
        hostname = raw_hostname.encode("idna").decode("ascii").casefold()
    except UnicodeError as error:
        raise ProviderConfigurationError(
            "Foundry endpoint contains an invalid hostname"
        ) from error
    if ":" in hostname:
        hostname = f"[{hostname}]"
    path = re.sub(r"/+", "/", parsed.path or "/")
    if any(segment in {".", ".."} for segment in path.split("/")):
        raise ProviderConfigurationError(
            "Foundry endpoint path must not contain dot segments"
        )
    if path != "/":
        path = path.rstrip("/")
    normalized = f"https://{hostname}:{port or 443}{path}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_release_dataset(name: str, reviewed_cases: list[dict]):
    dataset = mlflow.genai.datasets.get_dataset(name=name)
    registered_cases = dataset.to_df().to_dict(orient="records")
    expected = Counter(_case_key(record) for record in reviewed_cases)
    actual = Counter(_case_key(record) for record in registered_cases)
    if actual != expected:
        missing = sum((expected - actual).values())
        extra = sum((actual - expected).values())
        raise RuntimeError(
            f"Unity Catalog dataset {name!r} differs from the reviewed release "
            f"suite (missing={missing}, extra={extra}). Run "
            "scripts/sync_dataset.py; if stale records remain, use a new "
            "versioned DATASET_NAME rather than evaluating unreviewed rows."
        )
    return dataset


def _case_key(record: Mapping) -> str:
    return json.dumps(
        {
            "inputs": record.get("inputs"),
            "expectations": record.get("expectations"),
            "tags": _review_tags(record),
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _review_tags(record: Mapping) -> dict:
    tags = record.get("tags") or {}
    if not isinstance(tags, Mapping):
        raise TypeError("Evaluation case tags must be an object")
    return {
        str(key): value
        for key, value in tags.items()
        if not str(key).casefold().startswith("mlflow.")
    }


def _tool_schema_digest() -> str:
    limits = DEFAULT_AGENT_LIMITS
    tools = build_registry(
        timeout_seconds=limits.tool_timeout_seconds,
        max_output_chars=limits.max_tool_output_chars,
    ).openai_tools()
    serialized = json.dumps(
        tools,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _agent_limit_parameters() -> dict[str, str]:
    return {
        f"limit_{name}": str(value)
        for name, value in DEFAULT_AGENT_LIMITS.model_dump(mode="json").items()
    }


if __name__ == "__main__":
    main()
