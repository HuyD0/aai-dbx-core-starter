"""Validate the generated project's cross-file platform contracts.

This check is credential-free.  It validates declarations only; platform
infrastructure remains responsible for resolving resources and enforcing the
referenced policies.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from aai_core.manifest import load_manifest
from aai_core.runtime import PlatformSettings
from aai_core.tags import DatabricksAIRequestTags

ROOT = Path(__file__).resolve().parents[1]
_VARIABLE = re.compile(r"\$\{var\.([A-Za-z_][A-Za-z0-9_]*)\}")
_GOVERNED_BUNDLE_VARIABLES = frozenset(
    {"team", "owner_group", "cost_center", "compute_policy_id"}
)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise TypeError(f"{path.name} must contain a YAML mapping")
    return document


def _bundle_documents(
    root: Path,
    bundle: Mapping[str, Any],
) -> tuple[tuple[tuple[Path, Mapping[str, Any]], ...], list[str]]:
    failures: list[str] = []
    documents: list[tuple[Path, Mapping[str, Any]]] = [
        (root / "databricks.yml", bundle)
    ]
    includes = bundle.get("include", [])
    if not isinstance(includes, list) or not all(
        isinstance(pattern, str) for pattern in includes
    ):
        failures.append("databricks.yml include must be a list of path patterns")
        return tuple(documents), failures

    for pattern in includes:
        relative = Path(pattern)
        if relative.is_absolute() or ".." in relative.parts:
            failures.append("databricks.yml include paths must stay inside the project")
            continue
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            failures.append(
                f"databricks.yml include pattern {pattern!r} matched no files"
            )
            continue
        for path in matches:
            try:
                documents.append((path, _load_yaml(path)))
            except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
                failures.append(
                    f"{path.name} could not be loaded: {type(error).__name__}"
                )

    return tuple(documents), failures


def _bundle_job_keys(
    documents: tuple[tuple[Path, Mapping[str, Any]], ...],
) -> tuple[set[str], list[str]]:
    failures: list[str] = []
    job_keys: set[str] = set()
    for source, document in documents:
        resources = document.get("resources", {})
        if not isinstance(resources, Mapping):
            failures.append(f"{source.name} resources must be a mapping")
            continue
        jobs = resources.get("jobs", {})
        if not isinstance(jobs, Mapping):
            failures.append(f"{source.name} resources.jobs must be a mapping")
            continue
        job_keys.update(str(key) for key in jobs)
    return job_keys, failures


def _bundle_tag_failures(
    root: Path,
    bundle: Mapping[str, Any],
    documents: tuple[tuple[Path, Mapping[str, Any]], ...],
    *,
    approved_compute_policy_id: str | None,
    resource,
) -> list[str]:
    """Verify generated bundle tags resolve to the runtime ownership contract."""

    variables = bundle.get("variables", {})
    if not isinstance(variables, Mapping):
        return ["databricks.yml variables must be a mapping"]
    defaults = {
        str(name): declaration["default"]
        for name, declaration in variables.items()
        if isinstance(declaration, Mapping)
        and isinstance(declaration.get("default"), str)
    }
    expected_tags = _expected_tags(resource)
    target_failures = _target_contract_failures(
        bundle,
        resource=resource,
        defaults=defaults,
        expected_tags=expected_tags,
    )
    failures = _governed_override_failures(root, variables)
    failures.extend(target_failures)
    failures.extend(
        _bundle_document_failures(
            root,
            documents,
            approved_compute_policy_id=approved_compute_policy_id,
            defaults=defaults,
            expected_tags=expected_tags,
            target_name=resource.environment,
        )
    )
    return failures


def _expected_tags(resource: Any) -> dict[str, str]:
    return {
        "application": resource.application,
        "project": resource.project,
        "environment": resource.environment,
        "team": resource.team,
        "owner_group": resource.owner_group,
        "cost_center": resource.cost_center,
        "data_classification": resource.data_classification.value,
        "lifecycle": resource.lifecycle.value,
        "tag_schema_version": resource.tag_schema_version,
    }


def _governed_override_failures(
    root: Path,
    variables: Mapping[str, Any],
) -> list[str]:
    """Reject local and environment overrides of platform-owned values."""

    failures: list[str] = []
    for name in sorted(_GOVERNED_BUNDLE_VARIABLES.intersection(variables)):
        failures.append(
            f"databricks.yml governed value {name!r} must not be a runtime variable"
        )
    for name in sorted(_GOVERNED_BUNDLE_VARIABLES):
        if f"BUNDLE_VAR_{name}" in os.environ:
            failures.append(
                f"BUNDLE_VAR_{name} cannot override a governed ownership tag"
            )

    override_files = sorted(root.glob(".databricks/bundle/*/variable-overrides.json"))
    for path in override_files:
        failures.append(
            f"{path.relative_to(root)} is not permitted for a governed deployment"
        )

    try:
        ignored = {
            line.strip()
            for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
        }
    except OSError:
        failures.append(".gitignore could not be loaded")
    else:
        if ".databricks/" not in ignored:
            failures.append(".gitignore must exclude .databricks/")
    return failures


def _target_contract_failures(
    bundle: Mapping[str, Any],
    *,
    resource: Any,
    defaults: Mapping[str, str],
    expected_tags: Mapping[str, str],
) -> list[str]:
    """Validate the one declared target and its non-overridable policy."""

    failures: list[str] = []
    targets = bundle.get("targets", {})
    if isinstance(targets, Mapping):
        extra_targets = sorted(set(map(str, targets)) - {resource.environment})
        if extra_targets:
            failures.append(
                "databricks.yml contains undeclared deployment target(s): "
                + ", ".join(extra_targets)
            )
    target = targets.get(resource.environment) if isinstance(targets, Mapping) else None
    if not isinstance(target, Mapping):
        failures.append(
            f"databricks.yml target {resource.environment!r} must be a mapping"
        )
        target = {}
    failures.extend(
        _target_override_failures(
            bundle,
            target,
            environment=resource.environment,
        )
    )
    presets = target.get("presets", {})
    preset_tags = presets.get("tags", {}) if isinstance(presets, Mapping) else {}
    failures.extend(
        _resolved_tag_failures(
            preset_tags,
            expected=expected_tags,
            variables=defaults,
            target=resource.environment,
            source=f"databricks.yml target {resource.environment!r} preset",
        )
    )
    return failures


def _target_override_failures(
    bundle: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    environment: str,
) -> list[str]:
    """Reject target and bundle escape hatches around governed compute."""

    failures: list[str] = []
    target_variables = target.get("variables", {})
    if isinstance(target_variables, Mapping):
        for name in sorted(_GOVERNED_BUNDLE_VARIABLES.intersection(target_variables)):
            failures.append(
                f"databricks.yml target {environment!r} cannot override "
                f"governed tag variable {name!r}"
            )
    if target.get("cluster_id") is not None or target.get("compute_id") is not None:
        failures.append(
            f"databricks.yml target {environment!r} cannot use "
            "cluster_id or compute_id; "
            "governed jobs must use the declared policy-constrained clusters"
        )
    target_bundle = target.get("bundle", {})
    if isinstance(target_bundle, Mapping) and any(
        target_bundle.get(name) is not None for name in ("cluster_id", "compute_id")
    ):
        failures.append(
            f"databricks.yml target {environment!r} bundle cannot use "
            "cluster_id or compute_id"
        )
    target_resources = target.get("resources", {})
    if target_resources not in ({}, None):
        failures.append(
            f"databricks.yml target {environment!r} cannot override "
            "resources in a governed starter"
        )
    if bundle.get("cluster_id") is not None or bundle.get("compute_id") is not None:
        failures.append("databricks.yml cannot set cluster_id or compute_id")
    if bundle.get("python") not in ({}, None):
        failures.append(
            "databricks.yml cannot configure Python mutators in a governed starter"
        )
    experimental = bundle.get("experimental", {})
    if isinstance(experimental, Mapping) and experimental.get("python") not in (
        {},
        None,
    ):
        failures.append(
            "databricks.yml cannot configure experimental Python mutators in a "
            "governed starter"
        )
    bundle_header = bundle.get("bundle", {})
    if isinstance(bundle_header, Mapping) and any(
        bundle_header.get(name) is not None for name in ("cluster_id", "compute_id")
    ):
        failures.append("databricks.yml bundle cannot set cluster_id or compute_id")
    return failures


def _bundle_document_failures(
    root: Path,
    documents: tuple[tuple[Path, Mapping[str, Any]], ...],
    *,
    approved_compute_policy_id: str | None,
    defaults: Mapping[str, str],
    expected_tags: Mapping[str, str],
    target_name: str,
) -> list[str]:
    """Validate that included resource documents contain governed jobs only."""

    failures: list[str] = []
    main_path = root / "databricks.yml"
    for path, document in documents:
        if path != main_path:
            unexpected_sections = sorted(set(map(str, document)) - {"resources"})
            if unexpected_sections:
                failures.append(
                    f"{path.name} may declare only resources; found: "
                    + ", ".join(unexpected_sections)
                )
        resources = document.get("resources", {})
        jobs = resources.get("jobs", {}) if isinstance(resources, Mapping) else {}
        if not isinstance(jobs, Mapping):
            continue
        for job_key, job in jobs.items():
            failures.extend(
                _job_contract_failures(
                    path,
                    job_key,
                    job,
                    approved_compute_policy_id=approved_compute_policy_id,
                    defaults=defaults,
                    expected_tags=expected_tags,
                    target_name=target_name,
                )
            )
    return failures


def _job_contract_failures(
    path: Path,
    job_key: Any,
    job: Any,
    *,
    approved_compute_policy_id: str | None,
    defaults: Mapping[str, str],
    expected_tags: Mapping[str, str],
    target_name: str,
) -> list[str]:
    if not isinstance(job, Mapping):
        return [f"{path.name} job {job_key!r} must be a mapping"]
    source = f"{path.name} job {job_key!r}"
    failures: list[str] = []
    if "tags" in job:
        failures.extend(
            _resolved_tag_failures(
                job.get("tags"),
                expected=expected_tags,
                variables=defaults,
                target=target_name,
                source=source,
            )
        )
    cluster_keys, cluster_failures = _job_cluster_failures(
        job.get("job_clusters", []),
        source=source,
        approved_compute_policy_id=approved_compute_policy_id,
        defaults=defaults,
        expected_tags=expected_tags,
        target_name=target_name,
    )
    failures.extend(cluster_failures)
    failures.extend(
        _job_task_failures(
            job.get("tasks", []),
            source=source,
            cluster_keys=cluster_keys,
            approved_compute_policy_id=approved_compute_policy_id,
            defaults=defaults,
            expected_tags=expected_tags,
            target_name=target_name,
        )
    )
    return failures


def _job_cluster_failures(
    clusters: Any,
    *,
    source: str,
    approved_compute_policy_id: str | None,
    defaults: Mapping[str, str],
    expected_tags: Mapping[str, str],
    target_name: str,
) -> tuple[set[str], list[str]]:
    if not isinstance(clusters, list):
        return set(), [f"{source} job_clusters is invalid"]
    cluster_keys: set[str] = set()
    failures: list[str] = []
    for index, cluster in enumerate(clusters):
        cluster_key = (
            cluster.get("job_cluster_key") if isinstance(cluster, Mapping) else None
        )
        if isinstance(cluster_key, str) and cluster_key.strip():
            cluster_keys.add(cluster_key)
        else:
            failures.append(f"{source} cluster {index} requires job_cluster_key")
        new_cluster = (
            cluster.get("new_cluster", {}) if isinstance(cluster, Mapping) else {}
        )
        failures.extend(
            _cluster_contract_failures(
                new_cluster,
                expected_tags=expected_tags,
                expected_policy_id=approved_compute_policy_id,
                variables=defaults,
                target=target_name,
                source=f"{source} cluster {index}",
            )
        )
    return cluster_keys, failures


def _job_task_failures(
    tasks: Any,
    *,
    source: str,
    cluster_keys: set[str],
    approved_compute_policy_id: str | None,
    defaults: Mapping[str, str],
    expected_tags: Mapping[str, str],
    target_name: str,
) -> list[str]:
    if not isinstance(tasks, list):
        return [f"{source} tasks is invalid"]
    failures = [] if tasks else [f"{source} requires at least one task"]
    for index, task in enumerate(tasks):
        task_source = f"{source} task {index}"
        if not isinstance(task, Mapping):
            failures.append(f"{task_source} must be a mapping")
            continue
        if task.get("existing_cluster_id") is not None:
            failures.append(f"{task_source} cannot use existing_cluster_id")
        if task.get("environment_key") is not None:
            failures.append(f"{task_source} cannot use a serverless environment_key")
        if "new_cluster" in task:
            failures.extend(
                _cluster_contract_failures(
                    task.get("new_cluster"),
                    expected_tags=expected_tags,
                    expected_policy_id=approved_compute_policy_id,
                    variables=defaults,
                    target=target_name,
                    source=f"{task_source} cluster",
                )
            )
        elif task.get("job_cluster_key") not in cluster_keys:
            failures.append(
                f"{task_source} must reference a declared policy-constrained "
                "job_cluster_key"
            )
    return failures


def _cluster_contract_failures(
    new_cluster: Any,
    *,
    expected_tags: Mapping[str, str],
    expected_policy_id: str | None,
    variables: Mapping[str, str],
    target: str,
    source: str,
) -> list[str]:
    if not isinstance(new_cluster, Mapping):
        return [f"{source} new_cluster must be a mapping"]
    failures: list[str] = []
    raw_policy_id = new_cluster.get("policy_id")
    policy_id = _resolve_bundle_value(raw_policy_id, variables=variables, target=target)
    if (
        policy_id is None
        or not policy_id.strip()
        or not isinstance(raw_policy_id, str)
        or "${" in raw_policy_id
    ):
        failures.append(f"{source} must declare a literal constrained policy_id")
    elif expected_policy_id is None:
        failures.append(f"{source} approved compute policy is unknown")
    elif policy_id != expected_policy_id:
        failures.append(f"{source} policy_id does not match the generation contract")
    failures.extend(
        _resolved_tag_failures(
            new_cluster.get("custom_tags"),
            expected=expected_tags,
            variables=variables,
            target=target,
            source=source,
        )
    )
    return failures


def _resolved_tag_failures(
    tags: Any,
    *,
    expected: Mapping[str, str],
    variables: Mapping[str, str],
    target: str,
    source: str,
) -> list[str]:
    if not isinstance(tags, Mapping):
        return [f"{source} custom tags must be a mapping"]
    failures: list[str] = []
    for key, expected_value in expected.items():
        raw_value = tags.get(key)
        if (
            isinstance(raw_value, str)
            and "${" in raw_value
            and not (key == "environment" and raw_value == "${bundle.target}")
        ):
            failures.append(f"{source} tag {key!r} must be a literal governed value")
            continue
        actual = _resolve_bundle_value(raw_value, variables=variables, target=target)
        if actual != expected_value:
            failures.append(f"{source} tag {key!r} does not match the runtime contract")
    return failures


def _resolve_bundle_value(
    value: Any,
    *,
    variables: Mapping[str, str],
    target: str,
) -> str | None:
    if not isinstance(value, str):
        return None
    unresolved = False

    def variable(match: re.Match[str]) -> str:
        nonlocal unresolved
        name = match.group(1)
        resolved = variables.get(name)
        if resolved is None:
            unresolved = True
            return ""
        return resolved

    resolved = _VARIABLE.sub(variable, value).replace("${bundle.target}", target)
    if unresolved or "${" in resolved:
        return None
    return resolved


def _approved_compute_policy_id(root: Path) -> tuple[str | None, list[str]]:
    """Read the immutable generation-time compute-policy identifier."""

    failures: list[str] = []
    try:
        stamp = json.loads((root / ".aai-template.json").read_text(encoding="utf-8"))
        generated_with = stamp.get("generated_with", {})
        candidate_policy_id = (
            generated_with.get("compute_policy_id")
            if isinstance(generated_with, Mapping)
            else None
        )
        if not isinstance(candidate_policy_id, str) or not candidate_policy_id.strip():
            raise ValueError("missing compute_policy_id")
    except (OSError, TypeError, ValueError):
        failures.append(
            ".aai-template.json must record the approved generated compute_policy_id"
        )
        return None, failures
    return candidate_policy_id, failures


def _manifest_resource_failures(manifest: Any, resource: Any) -> list[str]:
    """Compare application manifest ownership with the runtime resource."""

    failures: list[str] = []
    if manifest.spec.cost_controls is None:
        failures.append(
            "ai-app.yaml spec.costControls.budgetPolicy is required for "
            "generated projects"
        )
    try:
        request_tags = DatabricksAIRequestTags.from_resource_context(resource)
    except (TypeError, ValueError) as error:
        failures.append(f"aai-platform.yml tags are invalid: {type(error).__name__}")
    else:
        if manifest.metadata.id != request_tags.application_id:
            failures.append(
                "ai-app.yaml metadata.id does not match aai-platform.yml application"
            )

    if manifest.metadata.owner != resource.owner_group:
        failures.append(
            "ai-app.yaml metadata.owner does not match aai-platform.yml owner_group"
        )
    if manifest.metadata.support_group != resource.owner_group:
        failures.append(
            "ai-app.yaml metadata.supportGroup does not match "
            "aai-platform.yml owner_group"
        )
    if manifest.metadata.cost_center != resource.cost_center:
        failures.append(
            "ai-app.yaml metadata.costCenter does not match "
            "aai-platform.yml cost_center"
        )
    if manifest.spec.repository.url != resource.repository:
        failures.append(
            "ai-app.yaml spec.repository.url does not match "
            "aai-platform.yml repository"
        )
    failures.extend(_manifest_environment_failures(manifest, resource))
    return failures


def _manifest_environment_failures(manifest: Any, resource: Any) -> list[str]:
    failures: list[str] = []

    environment = manifest.spec.environments.get(resource.environment)
    if environment is None:
        failures.append(
            "ai-app.yaml spec.environments does not declare the "
            "canonical aai-platform.yml environment"
        )
    else:
        effective_tags = {**dict(manifest.metadata.tags), **dict(environment.tags)}
        if effective_tags.get("team") != resource.team:
            failures.append(
                "ai-app.yaml effective tags.team does not match "
                "aai-platform.yml team"
            )
        expected_classification = resource.data_classification.value
        if effective_tags.get("data_classification") != expected_classification:
            failures.append(
                "ai-app.yaml effective data_classification does not match "
                "aai-platform.yml data_classification"
            )
        if effective_tags.get("lifecycle", "experimental") != resource.lifecycle.value:
            failures.append(
                "ai-app.yaml effective lifecycle does not match "
                "aai-platform.yml lifecycle"
            )
    return failures


def validate_project(root: Path = ROOT) -> tuple[str, ...]:
    """Return bounded validation failures for one generated project."""

    try:
        manifest_document = _load_yaml(root / "ai-app.yaml")
        manifest = load_manifest(manifest_document)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        return (f"ai-app.yaml could not be loaded: {type(error).__name__}",)

    try:
        settings = PlatformSettings.load(root / "aai-platform.yml", environ={})
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        return (f"aai-platform.yml could not be loaded: {type(error).__name__}",)

    try:
        bundle = _load_yaml(root / "databricks.yml")
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        return (f"databricks.yml could not be loaded: {type(error).__name__}",)

    approved_compute_policy_id, failures = _approved_compute_policy_id(root)
    resource = settings.resource
    failures.extend(_manifest_resource_failures(manifest, resource))

    documents, include_failures = _bundle_documents(root, bundle)
    failures.extend(include_failures)
    job_keys, bundle_failures = _bundle_job_keys(documents)
    failures.extend(bundle_failures)
    failures.extend(
        _bundle_tag_failures(
            root,
            bundle,
            documents,
            approved_compute_policy_id=approved_compute_policy_id,
            resource=resource,
        )
    )
    evaluation_job_key = manifest.spec.resources.evaluation_job_key
    if evaluation_job_key is not None and evaluation_job_key not in job_keys:
        failures.append(
            "ai-app.yaml evaluationJobKey is not declared under bundle resources.jobs"
        )

    return tuple(failures)


def main() -> int:
    failures = validate_project()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("project platform contracts are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
