"""Credential-free cross-file validation for generated projects."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = runpy.run_path(
    str(ROOT / "templates" / "_shared" / "files" / "scripts" / "validate_project.py")
)["validate_project"]


@pytest.fixture
def valid_project(tmp_path: Path) -> Path:
    manifest = {
        "apiVersion": "ai-platform/v1",
        "kind": "AIApplication",
        "metadata": {
            "id": "claims-agent",
            "name": "Claims Agent",
            "description": "A governed claims assistant.",
            "owner": "group:claims-owners",
            "supportGroup": "group:claims-owners",
            "businessDomain": "claims",
            "costCenter": "CC-123",
            "riskTier": "medium",
            "tags": {
                "application_id": "claims-agent",
                "team": "claims-ai",
                "domain": "claims",
                "cost_center": "CC-123",
                "data_classification": "internal",
                "lifecycle": "experimental",
            },
        },
        "spec": {
            "repository": {"url": "https://github.com/aai-test/claims-agent"},
            "authorization": {"mode": "user"},
            "environments": {
                "dev": {"tags": {"environment": "dev"}},
                "uat": {
                    "tags": {
                        "environment": "uat",
                        "lifecycle": "validation",
                    }
                },
            },
            "resources": {
                "evaluationJobKey": "release_gate",
                "aiSearchIndexes": [],
                "unityCatalogFunctions": [],
                "mcpServices": [],
            },
            "evaluation": {
                "profile": "golden_path_release_gate_v1",
                "dataset": "main.claims.evaluation_cases",
                "minimumCases": 30,
                "maximumAgeHours": 168,
                "thresholds": {"gate_pass_rate": 1.0},
            },
            "readiness": {"profile": "medium_risk_production_v1"},
            "costControls": {"budgetPolicy": "platform_standard_v1"},
            "serviceLevels": {"maximumErrorRate": 0.02, "p95LatencyMs": 8000},
        },
    }
    platform = {
        "platform": {
            "application": "claims-agent",
            "project": "claims",
            "environment": "dev",
            "team": "claims-ai",
            "owner_group": "group:claims-owners",
            "cost_center": "CC-123",
            "data_classification": "internal",
            "lifecycle": "experimental",
            "repository": "https://github.com/aai-test/claims-agent",
            "release": "1.0.0",
            "catalog": "main",
            "schema": "claims",
            "experiment_name": "/Shared/claims-ai-claims-claims-agent",
            "azure_identity": "workload_identity",
        }
    }
    bundle = {
        "bundle": {"name": "claims-agent"},
        "variables": {
            "deployment_environment": {
                "description": "Platform-owned runtime environment."
            },
            "deployment_lifecycle": {
                "description": "Platform-owned runtime lifecycle."
            },
            "deployment_release": {"description": "CI-attested source commit."},
        },
        "include": ["resources/*.yml"],
        "targets": {
            "dev": {
                "mode": "development",
                "variables": {
                    "deployment_environment": "dev",
                    "deployment_lifecycle": "experimental",
                    "deployment_release": "local-dev",
                },
                "presets": {
                    "tags": {
                        "application": "claims-agent",
                        "project": "claims",
                        "environment": "${bundle.target}",
                        "team": "claims-ai",
                        "owner_group": "group:claims-owners",
                        "cost_center": "CC-123",
                        "data_classification": "internal",
                        "lifecycle": "${var.deployment_lifecycle}",
                        "tag_schema_version": "2",
                    }
                },
            },
            "uat": {
                "mode": "production",
                "variables": {
                    "deployment_environment": "uat",
                    "deployment_lifecycle": "validation",
                },
                "presets": {
                    "tags": {
                        "application": "claims-agent",
                        "project": "claims",
                        "environment": "${bundle.target}",
                        "team": "claims-ai",
                        "owner_group": "group:claims-owners",
                        "cost_center": "CC-123",
                        "data_classification": "internal",
                        "lifecycle": "${var.deployment_lifecycle}",
                        "tag_schema_version": "2",
                    }
                },
            },
        },
    }
    jobs = {
        "resources": {
            "jobs": {
                "release_gate": {
                    "name": "release-gate",
                    "tasks": [{"task_key": "evaluate", "job_cluster_key": "default"}],
                    "job_clusters": [
                        {
                            "job_cluster_key": "default",
                            "new_cluster": {
                                "policy_id": "policy-123",
                                "spark_env_vars": {
                                    "AAI_ENVIRONMENT": "${var.deployment_environment}",
                                    "AAI_LIFECYCLE": "${var.deployment_lifecycle}",
                                    "AAI_RELEASE": "${var.deployment_release}",
                                },
                                "custom_tags": dict(
                                    bundle["targets"]["dev"]["presets"]["tags"]
                                ),
                            },
                        }
                    ],
                }
            }
        }
    }

    resources = tmp_path / "resources"
    resources.mkdir()
    for path, document in (
        (tmp_path / "ai-app.yaml", manifest),
        (tmp_path / "aai-platform.yml", platform),
        (tmp_path / "databricks.yml", bundle),
        (resources / "release.yml", jobs),
    ):
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    (tmp_path / ".aai-template.json").write_text(
        '{"generated_with":{"compute_policy_id":"policy-123"}}\n',
        encoding="utf-8",
    )
    (tmp_path / ".gitignore").write_text(".databricks/\n", encoding="utf-8")
    return tmp_path


def _replace(
    project: Path,
    filename: str,
    path: tuple[str | int, ...],
    value: Any,
) -> None:
    document = yaml.safe_load((project / filename).read_text(encoding="utf-8"))
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    (project / filename).write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def test_valid_project_contracts_pass_without_credentials(
    valid_project: Path, monkeypatch: pytest.MonkeyPatch
):
    # Ambient runtime overrides must not change validation of source declarations.
    monkeypatch.setenv("AAI_APPLICATION", "ambient-override")
    monkeypatch.setenv("DATABRICKS_TOKEN", "not-used")

    assert VALIDATOR(valid_project) == ()


def test_generated_project_requires_a_budget_policy_reference(valid_project: Path):
    document = yaml.safe_load(
        (valid_project / "ai-app.yaml").read_text(encoding="utf-8")
    )
    del document["spec"]["costControls"]
    (valid_project / "ai-app.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )

    assert any(
        "spec.costControls.budgetPolicy is required" in failure
        for failure in VALIDATOR(valid_project)
    )


@pytest.mark.parametrize(
    ("filename", "path", "value", "expected"),
    [
        (
            "aai-platform.yml",
            ("platform", "application"),
            "other-agent",
            "metadata.id does not match",
        ),
        (
            "aai-platform.yml",
            ("platform", "owner_group"),
            "group:other-owners",
            "metadata.owner does not match",
        ),
        (
            "aai-platform.yml",
            ("platform", "team"),
            "other-ai",
            "effective tags.team does not match",
        ),
        (
            "ai-app.yaml",
            ("spec", "environments", "dev", "tags", "team"),
            "environment-override",
            "effective tags.team does not match",
        ),
        (
            "databricks.yml",
            ("variables", "cost_center"),
            {"default": "CC-999"},
            "governed value 'cost_center' must not be a runtime variable",
        ),
        (
            "databricks.yml",
            ("targets", "dev", "variables", "cost_center"),
            "CC-999",
            "cannot override governed tag variable 'cost_center'",
        ),
        (
            "databricks.yml",
            ("targets", "dev", "presets", "tags", "data_classification"),
            "restricted",
            "preset tag 'data_classification' does not match",
        ),
        (
            "databricks.yml",
            ("targets", "dev", "cluster_id"),
            "bypass-cluster",
            "cannot use cluster_id",
        ),
        (
            "resources/release.yml",
            (
                "resources",
                "jobs",
                "release_gate",
                "job_clusters",
                0,
                "new_cluster",
                "custom_tags",
                "owner_group",
            ),
            "group:other-owners",
            "cluster 0 tag 'owner_group' does not match",
        ),
        (
            "aai-platform.yml",
            ("platform", "cost_center"),
            "CC-999",
            "metadata.costCenter does not match",
        ),
        (
            "aai-platform.yml",
            ("platform", "environment"),
            "staging",
            "spec.environments does not declare",
        ),
        (
            "aai-platform.yml",
            ("platform", "data_classification"),
            "restricted",
            "effective data_classification does not match",
        ),
        (
            "aai-platform.yml",
            ("platform", "lifecycle"),
            "validation",
            "effective lifecycle does not match",
        ),
        (
            "ai-app.yaml",
            ("spec", "resources", "evaluationJobKey"),
            "missing_gate",
            "evaluationJobKey is not declared",
        ),
    ],
)
def test_validator_rejects_cross_file_contract_drift(
    valid_project: Path,
    filename: str,
    path: tuple[str | int, ...],
    value: Any,
    expected: str,
):
    _replace(valid_project, filename, path, value)

    assert any(expected in failure for failure in VALIDATOR(valid_project))


def test_validator_rejects_ambient_governed_bundle_override(
    valid_project: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("BUNDLE_VAR_cost_center", "CC-OVERRIDE")

    assert any(
        "BUNDLE_VAR_cost_center cannot override" in failure
        for failure in VALIDATOR(valid_project)
    )


def test_validator_rejects_persisted_bundle_overrides(valid_project: Path):
    overrides = valid_project / ".databricks" / "bundle" / "dev"
    overrides.mkdir(parents=True)
    (overrides / "variable-overrides.json").write_text(
        '{"cost_center":"CC-OVERRIDE"}\n', encoding="utf-8"
    )

    assert any(
        "variable-overrides.json is not permitted" in failure
        for failure in VALIDATOR(valid_project)
    )


def test_validator_rejects_target_resource_overrides(valid_project: Path):
    _replace(
        valid_project,
        "databricks.yml",
        ("targets", "dev", "resources"),
        {"jobs": {"release_gate": {"tags": {"cost_center": "CC-OVERRIDE"}}}},
    )

    assert any(
        "cannot override resources" in failure for failure in VALIDATOR(valid_project)
    )


def test_validator_scans_job_tags_in_main_bundle(valid_project: Path):
    bundle_path = valid_project / "databricks.yml"
    bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    bundle["resources"] = {
        "jobs": {
            "inline_job": {
                "tags": {"cost_center": "CC-OVERRIDE"},
                "tasks": [],
                "job_clusters": [],
            }
        }
    }
    bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")

    assert any(
        "inline_job" in failure and "cost_center" in failure
        for failure in VALIDATOR(valid_project)
    )


def test_validator_scans_task_level_clusters(valid_project: Path):
    jobs_path = valid_project / "resources" / "release.yml"
    jobs = yaml.safe_load(jobs_path.read_text(encoding="utf-8"))
    job = jobs["resources"]["jobs"]["release_gate"]
    job["tasks"] = [
        {
            "task_key": "inline",
            "new_cluster": {
                "policy_id": "policy-123",
                "custom_tags": {
                    **job["job_clusters"][0]["new_cluster"]["custom_tags"],
                    "owner_group": "group:other-owners",
                },
            },
        }
    ]
    jobs_path.write_text(yaml.safe_dump(jobs, sort_keys=False), encoding="utf-8")

    assert any(
        "task 0 cluster" in failure and "owner_group" in failure
        for failure in VALIDATOR(valid_project)
    )


def test_validator_rejects_existing_task_cluster(valid_project: Path):
    jobs_path = valid_project / "resources" / "release.yml"
    jobs = yaml.safe_load(jobs_path.read_text(encoding="utf-8"))
    jobs["resources"]["jobs"]["release_gate"]["tasks"] = [
        {"task_key": "unsafe", "existing_cluster_id": "cluster-123"}
    ]
    jobs_path.write_text(yaml.safe_dump(jobs, sort_keys=False), encoding="utf-8")

    assert any(
        "cannot use existing_cluster_id" in failure
        for failure in VALIDATOR(valid_project)
    )


def test_validator_rejects_unapproved_serverless_task_compute(valid_project: Path):
    jobs_path = valid_project / "resources" / "release.yml"
    jobs = yaml.safe_load(jobs_path.read_text(encoding="utf-8"))
    jobs["resources"]["jobs"]["release_gate"]["tasks"] = [
        {"task_key": "unsafe", "environment_key": "serverless-default"}
    ]
    jobs_path.write_text(yaml.safe_dump(jobs, sort_keys=False), encoding="utf-8")

    failures = VALIDATOR(valid_project)
    assert any("cannot use a serverless environment_key" in item for item in failures)
    assert any("policy-constrained job_cluster_key" in item for item in failures)


def test_validator_rejects_unapproved_compute_policy(valid_project: Path):
    _replace(
        valid_project,
        "resources/release.yml",
        (
            "resources",
            "jobs",
            "release_gate",
            "job_clusters",
            0,
            "new_cluster",
            "policy_id",
        ),
        "other-policy",
    )

    assert any(
        "policy_id does not match the generation contract" in failure
        for failure in VALIDATOR(valid_project)
    )


def test_validator_rejects_aliases_for_governed_tags(valid_project: Path):
    bundle_path = valid_project / "databricks.yml"
    bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    bundle["variables"]["billing_alias"] = {"default": "CC-123"}
    bundle["targets"]["dev"]["variables"]["billing_alias"] = "CC-OVERRIDE"
    bundle["targets"]["dev"]["presets"]["tags"]["cost_center"] = "${var.billing_alias}"
    bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")

    assert any(
        "tag 'cost_center' must be a literal governed value" in failure
        for failure in VALIDATOR(valid_project)
    )


def test_validator_rejects_undeclared_targets(valid_project: Path):
    bundle_path = valid_project / "databricks.yml"
    bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    bundle["targets"]["prod"] = {
        "cluster_id": "bypass-cluster",
        "presets": {"tags": {}},
    }
    bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")

    assert any(
        "undeclared deployment target(s): prod" in failure
        for failure in VALIDATOR(valid_project)
    )


def test_validator_rejects_override_sections_in_included_resources(valid_project: Path):
    jobs_path = valid_project / "resources" / "release.yml"
    jobs = yaml.safe_load(jobs_path.read_text(encoding="utf-8"))
    jobs["bundle"] = {"cluster_id": "bypass-cluster"}
    jobs_path.write_text(yaml.safe_dump(jobs, sort_keys=False), encoding="utf-8")

    assert any(
        "release.yml may declare only resources; found: bundle" in failure
        for failure in VALIDATOR(valid_project)
    )


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        (
            {"python": {"mutators": ["unsafe:rewrite"]}},
            "cannot configure Python mutators",
        ),
        (
            {"experimental": {"python": {"mutators": ["unsafe:rewrite"]}}},
            "cannot configure experimental Python mutators",
        ),
    ],
)
def test_validator_rejects_post_validation_bundle_mutators(
    valid_project: Path,
    section: dict,
    expected: str,
):
    bundle_path = valid_project / "databricks.yml"
    bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    bundle.update(section)
    bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")

    assert any(expected in failure for failure in VALIDATOR(valid_project))
