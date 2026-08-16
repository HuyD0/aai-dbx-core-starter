"""Write immutable v2 release evidence from a passing MLflow gate run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

import mlflow
import yaml

from aai_core import __version__, bootstrap
from aai_core.deployment import ApplicationRelease
from aai_core.manifest import build_manifest_envelope
from app.config import DATASET_NAME, PROMPT_NAME
from app.controls import DEFAULT_AGENT_LIMITS
from app.tools import build_agent_registry

ROOT = Path(__file__).resolve().parents[1]
_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_WORLD_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")
_SECRET_PREFIXES = (
    "bearer",
    "dapi",
    "eyj",
    "github_pat_",
    "ghp_",
    "gho_",
    "ghr_",
    "ghs_",
    "ghu_",
    "pat-",
    "sk-",
)
_SENSITIVE_CONFIG_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "client_secret",
        "credential",
        "credentials",
        "header",
        "headers",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
)


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
        "--world-version",
        required=True,
        type=_world_version,
        help="Immutable operational data, tool-source, or business-rules version.",
    )
    parser.add_argument("--output", default="release.json")
    arguments = parser.parse_args()
    if arguments.prompt_version < 1:
        parser.error("--prompt-version must be a positive integer")
    world_version = arguments.world_version

    context = bootstrap(ROOT / "aai-platform.yml")
    manifest_document = yaml.safe_load(
        (ROOT / "ai-app.yaml").read_text(encoding="utf-8")
    )
    manifest = build_manifest_envelope(manifest_document)
    cost_controls = manifest.manifest.spec.cost_controls
    if cost_controls is None:
        raise RuntimeError("ai-app.yaml requires spec.costControls")
    dataset_name = (
        f"{context.settings.catalog}.{context.settings.schema}.{DATASET_NAME}"
    )
    experiment = mlflow.get_experiment_by_name(
        context.settings.effective_experiment_name
    )
    if experiment is None:
        raise RuntimeError(
            "The application's configured MLflow experiment does not exist"
        )
    dataset = mlflow.genai.datasets.get_dataset(name=dataset_name)
    run = mlflow.get_run(arguments.evaluation_run)
    limits = DEFAULT_AGENT_LIMITS.model_dump(mode="json")
    tools = build_agent_registry(
        timeout_seconds=limits["tool_timeout_seconds"],
        max_output_chars=limits["max_tool_output_chars"],
    ).openai_tools()
    tool_schema_digest = _canonical_digest(tools)
    current_source_commit = source_commit()
    current_source_state = source_state()
    evidence = _validated_gate_evidence(
        run=run,
        dataset=dataset,
        dataset_name=dataset_name,
        expected_experiment_id=experiment.experiment_id,
        prompt_version=arguments.prompt_version,
        settings=context.settings,
        current_source_commit=current_source_commit,
        current_source_state=current_source_state,
        current_tool_schema_digest=tool_schema_digest,
        current_limits=DEFAULT_AGENT_LIMITS,
    )
    release = ApplicationRelease(
        application=context.tags.application,
        release=context.tags.release,
        source_commit=current_source_commit,
        core_sdk_version=__version__,
        model=_release_model_config(context.settings, "general-chat"),
        prompt={
            "name": context.prompts.qualify(PROMPT_NAME),
            "version": arguments.prompt_version,
        },
        retrieval={},
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
        world={"operational_source_version": world_version},
        tools={
            "schema_sha256": tool_schema_digest,
            "max_tool_turns": limits["max_tool_turns"],
            "max_tool_calls_per_turn": limits["max_tool_calls_per_turn"],
            "max_total_tool_calls": limits["max_total_tool_calls"],
            "tool_timeout_seconds": limits["tool_timeout_seconds"],
            "max_tool_output_chars": limits["max_tool_output_chars"],
        },
        control={
            "gate_policy_digest": evidence["gate_policy_digest"],
            "gate_baseline_digest": evidence["gate_baseline_digest"],
            "judge_model": evidence["judge_model"],
            "max_tool_calls_per_turn": limits["max_tool_calls_per_turn"],
            "max_total_tool_calls": limits["max_total_tool_calls"],
            "tool_timeout_seconds": limits["tool_timeout_seconds"],
            "max_tool_output_chars": limits["max_tool_output_chars"],
            "max_input_messages": limits["max_input_messages"],
            "max_message_chars": limits["max_message_chars"],
            "max_total_input_chars": limits["max_total_input_chars"],
            "max_output_tokens": limits["max_output_tokens"],
            "max_stream_output_chars": limits["max_stream_output_chars"],
            "request_deadline_seconds": limits["request_deadline_seconds"],
            "owner_group": context.tags.owner_group,
            "cost_center": context.tags.cost_center,
            "manifest_hash": manifest.manifest_hash,
            "readiness_profile": manifest.manifest.spec.readiness.profile,
            "budget_policy": cost_controls.budget_policy,
            "service_levels": manifest.manifest.spec.service_levels.model_dump(
                mode="json",
                by_alias=True,
            ),
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
    run,
    dataset,
    dataset_name: str,
    expected_experiment_id: str,
    prompt_version: int,
    settings,
    current_source_commit: str,
    current_source_state: str,
    current_tool_schema_digest: str,
    current_limits,
) -> dict[str, str | None]:
    if current_source_state != "clean":
        raise RuntimeError(
            "Release evidence can be created only from a clean source tree"
        )
    if run.info.status != "FINISHED":
        raise RuntimeError("The evaluation run must be FINISHED")
    if run.data.tags.get("aai.gate_passed") != "true":
        raise RuntimeError("The evaluation run did not record a passing release gate")
    if str(run.info.experiment_id) != str(expected_experiment_id):
        raise RuntimeError(
            "The evaluation run is not from this application's configured experiment"
        )

    parameters = run.data.params
    expected = {
        "prompt_version": str(prompt_version),
        "evaluation_dataset": dataset_name,
        "evaluation_dataset_id": str(dataset.dataset_id),
        "evaluation_dataset_digest": str(dataset.digest),
        "target_model": _model_identity(settings, "general-chat"),
        "judge_model": _model_identity(settings, "judge-model"),
        "source_commit": current_source_commit,
        "source_state": "clean",
        "aai_core_version": __version__,
        "tool_schema_digest": current_tool_schema_digest,
        "agent_limits_digest": current_limits.digest,
        **_agent_limit_parameters(current_limits),
    }
    mismatches = {
        key: (parameters.get(key), value)
        for key, value in expected.items()
        if parameters.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Evaluation evidence no longer matches this release configuration: "
            + ", ".join(
                f"{key}={actual!r} (expected {wanted!r})"
                for key, (actual, wanted) in sorted(mismatches.items())
            )
        )
    if str(expected_experiment_id) not in {
        str(value) for value in (dataset.experiment_ids or [])
    }:
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
        "target_model": expected["target_model"],
        "judge_model": expected["judge_model"],
    }


def _model_identity(settings, logical_name: str) -> str:
    config = settings.models.get(logical_name)
    if not isinstance(config, Mapping):
        raise TypeError(f"{logical_name} must be configured as a mapping")
    provider = config.get("provider")
    deployment = config.get("deployment")
    if not isinstance(provider, str) or not provider.strip():
        raise TypeError(f"{logical_name} provider must be a non-empty string")
    if not isinstance(deployment, str) or not deployment.strip():
        raise TypeError(f"{logical_name} deployment must be a non-empty string")
    return f"{provider.strip()}:{deployment.strip()}"


def _release_model_config(settings, logical_name: str) -> dict:
    config = settings.models.get(logical_name)
    if not isinstance(config, Mapping):
        raise TypeError(f"{logical_name} must be configured as a mapping")
    _reject_secret_material(config)
    # Endpoint URLs and token scopes are runtime routing/authentication details.
    # The immutable release contains only reviewed, non-secret model identity
    # and declared capabilities.
    reviewed = {
        key: config[key]
        for key in ("provider", "deployment", "model", "capabilities")
        if key in config
    }
    _model_identity(settings, logical_name)
    return {"logical_name": logical_name, **reviewed}


def _reject_secret_material(value) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _SENSITIVE_CONFIG_NAMES or any(
                normalized.endswith(f"_{name}") for name in _SENSITIVE_CONFIG_NAMES
            ):
                raise RuntimeError(
                    "Model configuration contains secret-bearing material; "
                    "use a governed secret reference outside release evidence"
                )
            _reject_secret_material(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _reject_secret_material(nested)
        return
    if isinstance(value, str) and value.strip().casefold().startswith(_SECRET_PREFIXES):
        raise RuntimeError(
            "Model configuration contains credential-shaped material; use a "
            "governed secret reference outside release evidence"
        )


def _world_version(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("--world-version must be a bounded non-secret identifier")
    normalized = value.strip()
    if not _WORLD_VERSION.fullmatch(normalized) or normalized.casefold().startswith(
        _SECRET_PREFIXES
    ):
        raise ValueError("--world-version must be a bounded non-secret identifier")
    return normalized


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


def _canonical_digest(value) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _agent_limit_parameters(limits) -> dict[str, str]:
    return {
        f"limit_{name}": str(value)
        for name, value in limits.model_dump(mode="json").items()
    }


if __name__ == "__main__":
    main()
