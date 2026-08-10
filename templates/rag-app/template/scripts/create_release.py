"""Write immutable v2 RAG release evidence from one passing gate run."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import mlflow

from aai_core import __version__, bootstrap
from aai_core.deployment import ApplicationRelease
from app.config import DATASET_NAME, PROMPT_NAME
from app.rag import DEFAULT_RAG_LIMITS, rag_limit_parameters
from app.release_evidence import (
    configuration_digests,
    knowledge_version,
    model_identity,
    release_configuration,
)

ROOT = Path(__file__).resolve().parents[1]
_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def source_commit() -> str:
    configured = os.environ.get("GIT_COMMIT")
    if configured is not None:
        return _normalized_source_commit(configured)
    return _normalized_source_commit(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        )
    )


def source_state() -> str:
    configured = os.environ.get("GIT_DIRTY")
    if configured is not None:
        normalized = configured.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return "dirty"
        if normalized in {"0", "false", "no"}:
            return "clean"
        return "unknown"
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    return "dirty" if result.stdout.strip() else "clean"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-version", required=True, type=int)
    parser.add_argument("--evaluation-run", required=True)
    parser.add_argument(
        "--knowledge-version",
        required=True,
        help="Immutable knowledge/chunk/index snapshot identifier used by the gate.",
    )
    parser.add_argument("--output", default="release.json")
    arguments = parser.parse_args()
    if arguments.prompt_version < 1:
        parser.error("--prompt-version must be a positive integer")
    try:
        world_version = knowledge_version(arguments.knowledge_version)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    current_source_commit = source_commit()
    current_source_state = source_state()
    if current_source_state != "clean":
        raise RuntimeError(
            "Release evidence can be created only from a clean source tree"
        )

    context = bootstrap(ROOT / "aai-platform.yml")
    configuration = release_configuration(context.settings)
    config_digests = configuration_digests(configuration)
    dataset_name = (
        f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
    )
    prompt_name = context.prompts.qualify(PROMPT_NAME)
    prompt_uri = f"prompts:/{prompt_name}/{arguments.prompt_version}"
    experiment = mlflow.get_experiment_by_name(
        context.settings.effective_experiment_name
    )
    if experiment is None:
        raise RuntimeError(
            "The application's configured MLflow experiment does not exist"
        )
    dataset = mlflow.genai.datasets.get_dataset(name=dataset_name)
    run = mlflow.get_run(arguments.evaluation_run)
    evidence = _validated_gate_evidence(
        run=run,
        dataset=dataset,
        dataset_name=dataset_name,
        expected_experiment_id=experiment.experiment_id,
        prompt_name=prompt_name,
        prompt_version=arguments.prompt_version,
        settings=context.settings,
        current_source_commit=current_source_commit,
        current_source_state=current_source_state,
        current_knowledge_version=world_version,
        current_configuration_digests=config_digests,
    )

    model = {
        **configuration["model"],
        "configuration_sha256": config_digests["model_configuration_digest"],
    }
    retrieval = {
        **configuration["retrieval"],
        "embedding_configuration": configuration["embedding"],
        "retrieval_configuration_sha256": config_digests[
            "retrieval_configuration_digest"
        ],
        "embedding_configuration_sha256": config_digests[
            "embedding_configuration_digest"
        ],
        "index_configuration_sha256": config_digests["index_configuration_digest"],
        "rag_configuration_sha256": config_digests["rag_configuration_digest"],
    }
    limits = DEFAULT_RAG_LIMITS.as_dict()
    release = ApplicationRelease(
        application=context.tags.application,
        release=context.tags.release,
        source_commit=current_source_commit,
        core_sdk_version=__version__,
        model=model,
        prompt={
            "name": prompt_name,
            "version": arguments.prompt_version,
            "uri": prompt_uri,
        },
        retrieval=retrieval,
        evaluation={
            "dataset": dataset_name,
            "dataset_id": dataset.dataset_id,
            "dataset_digest": dataset.digest,
            "run_id": arguments.evaluation_run,
            "passed": True,
            "policy_digest": evidence["gate_policy_digest"],
            "baseline_digest": evidence["gate_baseline_digest"],
            "target_model": evidence["target_model"],
            "judge_model": evidence["judge_model"],
        },
        world={
            "knowledge_version": world_version,
            "retrieval_index": configuration["index"]["index"],
            "index_configuration_sha256": config_digests["index_configuration_digest"],
        },
        control={
            "gate_policy_digest": evidence["gate_policy_digest"],
            "gate_baseline_digest": evidence["gate_baseline_digest"],
            "judge_model": evidence["judge_model"],
            "rag_limits_digest": DEFAULT_RAG_LIMITS.digest,
            **limits,
        },
        environment=context.tags.environment,
        schema_version="2",
    )
    release.write(ROOT / arguments.output)
    print(
        {
            "release": release.release,
            "digest": release.digest,
            "clock_digests": release.clock_digests,
        }
    )


def _validated_gate_evidence(
    *,
    run: object,
    dataset: object,
    dataset_name: str,
    expected_experiment_id: str,
    prompt_name: str,
    prompt_version: int,
    settings: object,
    current_source_commit: str,
    current_source_state: str,
    current_knowledge_version: str,
    current_configuration_digests: dict[str, str],
) -> dict[str, str | None]:
    if current_source_state != "clean":
        raise RuntimeError(
            "Release evidence can be created only from a clean source tree"
        )
    info = getattr(run, "info", None)
    data = getattr(run, "data", None)
    if info is None or data is None:
        raise RuntimeError("The evaluation run does not expose MLflow run evidence")
    if getattr(info, "status", None) != "FINISHED":
        raise RuntimeError("The evaluation run must be FINISHED")
    tags = getattr(data, "tags", {})
    parameters = getattr(data, "params", {})
    if not isinstance(tags, dict) or not isinstance(parameters, dict):
        raise RuntimeError("The evaluation run has malformed tags or parameters")
    if tags.get("aai.gate_passed") != "true":
        raise RuntimeError("The evaluation run did not record a passing release gate")
    if str(getattr(info, "experiment_id", "")) != str(expected_experiment_id):
        raise RuntimeError(
            "The evaluation run is not from this application's configured experiment"
        )

    dataset_id = getattr(dataset, "dataset_id", None)
    dataset_digest = getattr(dataset, "digest", None)
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise RuntimeError("The release dataset does not expose a stable dataset ID")
    if not isinstance(dataset_digest, str) or not dataset_digest.strip():
        raise RuntimeError("The release dataset does not expose a stable digest")

    target_identity = model_identity(settings, "general-chat")
    judge_identity = model_identity(settings, "judge-model")
    prompt_uri = f"prompts:/{prompt_name}/{prompt_version}"
    expected = {
        "prompt_version": str(prompt_version),
        "prompt_uri": prompt_uri,
        "evaluation_dataset": dataset_name,
        "evaluation_dataset_id": dataset_id,
        "evaluation_dataset_digest": dataset_digest,
        "target_model": target_identity,
        "judge_model": judge_identity,
        "source_commit": current_source_commit,
        "source_state": "clean",
        "aai_core_version": __version__,
        "knowledge_version": current_knowledge_version,
        "rag_limits_digest": DEFAULT_RAG_LIMITS.digest,
        **current_configuration_digests,
        **rag_limit_parameters(DEFAULT_RAG_LIMITS),
    }
    mismatches = {
        key: (parameters.get(key), value)
        for key, value in expected.items()
        if parameters.get(key) != value
    }
    expected_tags = {
        "aai.target_model": target_identity,
        "aai.judge_model": judge_identity,
    }
    mismatches.update(
        {
            key: (tags.get(key), value)
            for key, value in expected_tags.items()
            if tags.get(key) != value
        }
    )
    if mismatches:
        raise RuntimeError(
            "Evaluation evidence no longer matches this release configuration: "
            + ", ".join(
                f"{key}={actual!r} (expected {wanted!r})"
                for key, (actual, wanted) in sorted(mismatches.items())
            )
        )
    associated = {
        str(value) for value in (getattr(dataset, "experiment_ids", None) or [])
    }
    if str(expected_experiment_id) not in associated:
        raise RuntimeError(
            "The release dataset is not associated with this application's experiment"
        )

    policy_digest = _require_digest(
        "gate_policy_digest", parameters.get("gate_policy_digest")
    )
    baseline = parameters.get("gate_baseline_digest")
    baseline_digest = (
        None
        if baseline == "none"
        else _require_digest("gate_baseline_digest", baseline)
    )
    return {
        "gate_policy_digest": policy_digest,
        "gate_baseline_digest": baseline_digest,
        "target_model": target_identity,
        "judge_model": judge_identity,
    }


def _normalized_source_commit(value: str) -> str:
    normalized = value.strip().lower()
    if not _GIT_OBJECT_ID.fullmatch(normalized):
        raise RuntimeError(
            "GIT_COMMIT must be a full 40- or 64-character hexadecimal object id"
        )
    return normalized


def _require_digest(name: str, value: str | None) -> str:
    if (
        value is None
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{name} must be a lowercase SHA-256 digest")
    return value


if __name__ == "__main__":
    main()
