"""Release gate: LLM-judge evaluation against configured quality thresholds.

This run needs model access (the judges call an LLM), so it executes on the
credentialed path — locally with keyless auth, or in the workspace job — and
must pass BEFORE a release is promoted. Pull-request CI runs only the
deterministic checks in evals/offline_checks.py.

Thresholds live in evals/gate_config.json. Regression checks activate once
evals/baseline.json exists; refresh it from a passing release run by adding
``--update-baseline`` to the exact prompt/knowledge evaluation command.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, Safety

from aai_core import bootstrap
from aai_core.agents import AgentRequest
from aai_core.decisions import Decision, DecisionRecord, record_decision
from aai_core.evaluation import (
    GatePolicy,
    MetricRule,
    apply_gate,
    judge_model_uri,
)
from aai_core.experiments import record_reproducibility
from aai_core.prompts import prompt_digest
from app.config import DATASET_NAME, PROMPT_NAME
from app.rag import DEFAULT_RAG_LIMITS, RAGAgent, rag_limit_parameters
from app.release_evidence import (
    configuration_digests,
    knowledge_version,
    model_identity,
    release_configuration,
)

ROOT = Path(__file__).resolve().parents[1]
GATE_CONFIG = ROOT / "evals" / "gate_config.json"
BASELINE = ROOT / "evals" / "baseline.json"


def load_thresholds() -> list[MetricRule]:
    config = json.loads(GATE_CONFIG.read_text(encoding="utf-8"))
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


def resolve_version(context, requested: int) -> int:
    """Resolve and verify the exact immutable registry version requested."""

    registered = context.prompts.load(
        PROMPT_NAME, version=requested, cache_ttl_seconds=0
    )
    resolved = int(registered.version)
    if resolved != requested:
        raise RuntimeError("Prompt Registry returned a different immutable version")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-version",
        type=int,
        required=True,
        help="Exact immutable Prompt Registry version to evaluate.",
    )
    parser.add_argument(
        "--knowledge-version",
        required=True,
        help="Immutable knowledge/chunk/index snapshot identifier.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="After a passing run, write the metrics to evals/baseline.json "
        "so future runs regression-check against this release.",
    )
    args = parser.parse_args()
    if args.prompt_version < 1:
        parser.error("--prompt-version must be a positive integer")
    try:
        world_version = knowledge_version(args.knowledge_version)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    context = bootstrap(ROOT / "aai-platform.yml")
    judge_model = judge_model_uri(context.settings)
    version = resolve_version(context, args.prompt_version)
    target_identity, judge_identity = _evaluation_model_identities(context.settings)

    cases = json.loads(
        (ROOT / "evals" / "data" / "release_cases.json").read_text(encoding="utf-8")
    )
    baseline = load_baseline()
    policy = GatePolicy(
        rules=tuple(load_thresholds()),
        allow_missing_regression_baseline=args.update_baseline and not baseline,
    )
    prompt_uri = f"prompts:/{context.prompts.qualify(PROMPT_NAME)}/{version}"
    dataset_name = (
        f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
    )
    dataset = _load_release_dataset(dataset_name, cases)
    configuration = release_configuration(context.settings)
    config_digests = configuration_digests(configuration)
    limit_parameters = rag_limit_parameters(DEFAULT_RAG_LIMITS)
    agent = RAGAgent(context, prompt_version=version)

    def predict_fn(question: str) -> str:
        response = agent.invoke(
            AgentRequest(messages=[{"role": "user", "content": question}])
        )
        return response.content

    # A governed MLflow run: aai.* tags, pinned prompt URI + dataset as
    # params, exact code/config/world joins, gate metrics, verdict tag, and
    # the evaluation traces attached.
    with context.experiments.run(
        run_name=f"rag-prompt-v{version}-validation-gate",
        parameters={
            "prompt_version": version,
            "prompt_uri": prompt_uri,
            "evaluation_dataset": dataset_name,
            "evaluation_dataset_id": dataset.dataset_id,
            "evaluation_dataset_digest": dataset.digest,
            "case_count": len(cases),
            "target_model": target_identity,
            "judge_model": judge_identity,
            "knowledge_version": world_version,
            "rag_limits_digest": DEFAULT_RAG_LIMITS.digest,
            **config_digests,
            **limit_parameters,
        },
    ) as evaluation_run:
        registered = context.prompts.load(
            PROMPT_NAME, version=version, cache_ttl_seconds=0
        )
        _validate_dataset_association(dataset, evaluation_run.info.experiment_id)
        record_reproducibility()
        native_result = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=predict_fn,
            scorers=[
                RetrievalGroundedness(model=judge_model),
                RelevanceToQuery(model=judge_model),
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
        evaluation_run_id = str(evaluation_run.info.run_id)
    template = getattr(registered, "template", None)
    if not isinstance(template, (str, list)):
        raise TypeError(
            "The evaluated prompt version exposes no template for decision evidence."
        )
    decision = Decision.ADOPT if report.passed else Decision.REJECT
    decision_run_id = record_decision(
        DecisionRecord(
            decision=decision,
            change_id=f"prompt-v{version}",
            change_summary=f"Evaluate pinned prompt version {version} for release.",
            rationale=(
                "The release gate passed for the exact registered prompt version."
                if report.passed
                else "The release gate failed for the exact registered prompt version."
            ),
            change_run_id=evaluation_run_id,
            gate=report,
            prompt_name=context.prompts.qualify(PROMPT_NAME),
            prompt_version=version,
            prompt_digest=prompt_digest(template),
            decided_by="code:release-gate",
        ),
        experiments=context.experiments,
    )
    if report.passed and args.update_baseline:
        BASELINE.write_text(
            json.dumps({"metrics": dict(report.metrics)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(
        {
            "application": context.tags.application,
            "release": context.tags.release,
            "prompt_version": version,
            "knowledge_version": world_version,
            "evaluation_run": evaluation_run_id,
            "metrics": report.metrics,
            "baseline_updated": args.update_baseline,
            "decision": decision.value,
            "decision_run_id": decision_run_id,
        }
    )
    report.require_passed()


def _evaluation_model_identities(settings) -> tuple[str, str]:
    target = _model_config(settings, "general-chat")
    judge = _model_config(settings, "judge-model")
    if (
        judge["provider"].casefold() == target["provider"].casefold()
        and judge["deployment"].casefold() == target["deployment"].casefold()
    ):
        raise ValueError(
            "judge-model must use a deployment distinct from general-chat; "
            "a release gate cannot rely on the target judging itself"
        )
    try:
        target_identity = model_identity(settings, "general-chat")
        judge_identity = model_identity(settings, "judge-model")
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "Evaluation model identity configuration is invalid"
        ) from error
    return target_identity, judge_identity


def _model_config(settings, logical_name: str) -> dict[str, str]:
    config = settings.models.get(logical_name)
    if not isinstance(config, Mapping):
        raise ValueError(f"{logical_name} must be configured")
    provider = config.get("provider")
    deployment = config.get("deployment")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(f"{logical_name} requires a provider")
    if not isinstance(deployment, str) or not deployment.strip():
        raise ValueError(f"{logical_name} requires a deployment")
    return {
        "provider": provider.strip(),
        "deployment": deployment.strip(),
    }


def _load_release_dataset(name: str, reviewed_cases: list[dict]):
    dataset = mlflow.genai.datasets.get_dataset(name=name)
    if not isinstance(dataset.dataset_id, str) or not dataset.dataset_id.strip():
        raise RuntimeError(f"Unity Catalog dataset {name!r} has no stable dataset ID")
    if not isinstance(dataset.digest, str) or not dataset.digest.strip():
        raise RuntimeError(f"Unity Catalog dataset {name!r} has no stable digest")
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


def _validate_dataset_association(dataset, experiment_id: str) -> None:
    associated = {str(value) for value in (dataset.experiment_ids or [])}
    if str(experiment_id) not in associated:
        raise RuntimeError(
            "The release dataset is not associated with this application's "
            "configured experiment"
        )


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


if __name__ == "__main__":
    main()
