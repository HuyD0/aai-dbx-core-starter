"""Reusable setup and preflight for the connected example notebooks.

The functions in this module keep environment and cloud-access plumbing out of
the teaching flow. They return explicit, named state so notebooks do not depend
on ``%run`` side effects or variables left behind by another notebook.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any

_REQUIRED_MODULES = (
    "aai_core",
    "databricks.sdk",
    "databricks_openai",
    "mlflow",
    "pandas",
)
_REQUIRED_IDENTIFIERS = (
    "azure_subscription_id",
    "azure_tenant_id",
    "databricks_host",
)


class EvidenceDestination(StrEnum):
    """Supported stores for notebook prompt, run, and trace evidence."""

    LOCAL = "local"
    DATABRICKS = "databricks"


@dataclass(frozen=True)
class NotebookEnvironment:
    """Local evidence destinations and non-secret platform configuration."""

    repo_root: Path
    config_path: Path
    identifiers: Mapping[str, str]
    tracking_uri: str
    registry_uri: str
    artifact_root: Path
    evidence_destination: EvidenceDestination
    mlflow: ModuleType


@dataclass(frozen=True)
class DatabricksEvidencePreflight:
    """Identity and governed evidence resources resolved without a model request."""

    context: Any
    workspace: Any
    experiment_name: str
    experiment_id: str
    catalog: str
    schema: str
    evidence_destination: EvidenceDestination
    azure_account: Mapping[str, Any]
    current_user: Any


@dataclass(frozen=True)
class DatabricksPreflight(DatabricksEvidencePreflight):
    """Evidence resources and model endpoint proven before a billable request."""

    model: Any
    deployment: str
    endpoint: Mapping[str, Any]


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the source checkout from a VS Code or terminal working directory."""

    base = Path(start).expanduser().resolve() if start is not None else Path.cwd()
    for directory in (base, *base.parents):
        if (directory / "pyproject.toml").is_file() and (
            directory / "examples" / "notebook_setup.py"
        ).is_file():
            return directory
    raise FileNotFoundError(
        "Could not find the repository root. Open the cloned repository as your "
        "VS Code workspace, then restart the notebook kernel."
    )


def _missing_modules() -> list[str]:
    missing = []
    for module_name in _REQUIRED_MODULES:
        try:
            available = importlib.util.find_spec(module_name) is not None
        except ModuleNotFoundError:
            available = False
        if not available:
            missing.append(module_name)
    return missing


def _load_identifiers(repo_root: Path) -> dict[str, str]:
    identifiers_path = repo_root / "platform-identifiers.json"
    if not identifiers_path.is_file():
        raise FileNotFoundError(
            f"Missing {identifiers_path}; use a complete repository clone."
        )
    try:
        identifiers = json.loads(identifiers_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{identifiers_path} is not valid JSON.") from exc
    missing = [name for name in _REQUIRED_IDENTIFIERS if not identifiers.get(name)]
    if missing:
        raise RuntimeError(
            f"{identifiers_path} is missing required identifiers: {missing}."
        )
    return {str(name): str(value) for name, value in identifiers.items()}


def prepare_notebook_environment(
    repo_root: str | Path | None = None,
    *,
    evidence_destination: EvidenceDestination | str = EvidenceDestination.LOCAL,
) -> NotebookEnvironment:
    """Configure one compatible MLflow evidence backend and Databricks auth."""

    root = find_repo_root(repo_root)
    try:
        destination = EvidenceDestination(evidence_destination)
    except ValueError as exc:
        choices = ", ".join(item.value for item in EvidenceDestination)
        raise ValueError(
            f"Unsupported evidence destination {evidence_destination!r}; "
            f"choose one of: {choices}."
        ) from exc
    # Make imports deterministic even when VS Code starts the kernel in examples/.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    missing_modules = _missing_modules()
    if missing_modules:
        raise RuntimeError(
            f"Missing modules in {sys.executable}: {missing_modules}. Run `make "
            "examples-install`, select `.venv/bin/python` as the VS Code kernel, "
            "and restart the kernel."
        )

    runtime = importlib.import_module("aai_core.runtime")
    config_path = Path(runtime.find_platform_config(start=root)).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No platform configuration found at {config_path}. From the repository "
            "root run `cp aai-platform.example.yml aai-platform.yml`, then configure "
            "`providers.models.general-chat.deployment`."
        )

    identifiers = _load_identifiers(root)
    local_mlflow_dir = root / ".aai" / "local"
    artifact_root = local_mlflow_dir / "mlruns"
    artifact_root.mkdir(parents=True, exist_ok=True)
    if destination is EvidenceDestination.DATABRICKS:
        tracking_uri = "databricks"
        registry_uri = "databricks-uc"
    else:
        tracking_uri = f"sqlite:///{local_mlflow_dir / 'mlflow.db'}"
        registry_uri = tracking_uri

    # Keep tracking and registry compatible so exact run/trace/prompt links are valid.
    os.environ["AAI_PLATFORM_CONFIG"] = str(config_path)
    os.environ["AAI_EXAMPLE_LOCAL_DIR"] = str(local_mlflow_dir)
    os.environ["AAI_EXAMPLE_ARTIFACT_ROOT"] = str(artifact_root)
    os.environ["DATABRICKS_HOST"] = identifiers["databricks_host"]
    os.environ["DATABRICKS_AUTH_TYPE"] = "azure-cli"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_REGISTRY_URI"] = registry_uri

    mlflow = importlib.import_module("mlflow")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    environment = NotebookEnvironment(
        repo_root=root,
        config_path=config_path,
        identifiers=identifiers,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        artifact_root=artifact_root,
        evidence_destination=destination,
        mlflow=mlflow,
    )
    print("SETUP PASSED")
    print(
        {
            "python_kernel": sys.executable,
            "config": str(environment.config_path),
            "tracking_uri": mlflow.get_tracking_uri(),
            "registry_uri": mlflow.get_registry_uri(),
            "evidence_destination": destination.value,
            "artifact_root": (
                str(environment.artifact_root)
                if destination is EvidenceDestination.LOCAL
                else "Databricks-managed"
            ),
        }
    )
    return environment


def _azure_account(
    *,
    run_command: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    try:
        result = run_command(
            ["az", "account", "show", "--output", "json"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Azure CLI is not installed or not on PATH. Install it through your "
            "approved workstation process, then restart VS Code."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Azure CLI did not respond within 30 seconds. Verify the installation "
            "and retry `az account show` in a terminal."
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            "Azure CLI is not authenticated. Run `az login`, then rerun this "
            "checkpoint."
        )
    try:
        account = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Azure CLI returned an unreadable account response. Run "
            "`az account show --output json` in a terminal."
        ) from exc
    if not isinstance(account, dict):
        raise RuntimeError("Azure CLI account response must be a JSON object.")
    return account


def _ready_chat_endpoints(workspace: Any) -> list[str]:
    names = []
    for item in workspace.serving_endpoints.list():
        endpoint = item.as_dict()
        if (
            endpoint.get("task") == "llm/v1/chat"
            and endpoint.get("state", {}).get("ready") == "READY"
        ):
            names.append(str(endpoint["name"]))
    return sorted(names)


def preflight_databricks_evidence(
    environment: NotebookEnvironment,
    *,
    which: Callable[[str], str | None] | None = None,
    run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    bootstrap_fn: Callable[[Path], Any] | None = None,
) -> DatabricksEvidencePreflight:
    """Verify identity, workspace membership, UC namespace, and experiment access."""

    which_command = which or shutil.which
    if which_command("az") is None:
        raise RuntimeError(
            "Azure CLI is not installed or not on PATH. Install it through your "
            "approved workstation process, then restart VS Code."
        )
    account = _azure_account(run_command=run_command or subprocess.run)
    expected_tenant = environment.identifiers["azure_tenant_id"]
    if account.get("tenantId") != expected_tenant:
        raise RuntimeError(
            f"Azure CLI tenant is {account.get('tenantId')!r}; expected "
            f"{expected_tenant!r}. Run `az login --tenant {expected_tenant}`."
        )
    expected_subscription = environment.identifiers["azure_subscription_id"]
    if account.get("id") != expected_subscription:
        raise RuntimeError(
            f"Azure CLI subscription is {account.get('id')!r}; expected "
            f"{expected_subscription!r}. Run `az account set --subscription "
            f"{expected_subscription}`."
        )

    if bootstrap_fn is None:
        bootstrap_fn = importlib.import_module("aai_core").bootstrap
    context = bootstrap_fn(environment.config_path)
    workspace = context.workspace
    try:
        current_user = workspace.current_user.me()
    except Exception as exc:
        raise RuntimeError(
            "Azure login succeeded, but workspace access failed for "
            f"{os.environ['DATABRICKS_HOST']}. Ask the platform team to verify "
            "your workspace registration."
        ) from exc

    catalog = str(context.settings.catalog)
    schema = str(context.settings.schema)
    if environment.evidence_destination is EvidenceDestination.DATABRICKS:
        try:
            workspace.catalogs.get(catalog)
        except Exception as exc:
            visible_catalogs = ", ".join(
                sorted(
                    str(item.name)
                    for item in workspace.catalogs.list()
                    if getattr(item, "name", None)
                )
            )
            raise RuntimeError(
                f"Unity Catalog catalog {catalog!r} does not exist or is not "
                "visible. Set `platform.catalog` in aai-platform.yml to an "
                f"approved catalog. Visible catalogs: {visible_catalogs or 'none'}."
            ) from exc
        prompt_namespace = f"{catalog}.{schema}"
        try:
            workspace.schemas.get(prompt_namespace)
        except Exception as exc:
            visible_schemas = ", ".join(
                sorted(
                    str(item.name)
                    for item in workspace.schemas.list(catalog_name=catalog)
                    if getattr(item, "name", None)
                )
            )
            raise RuntimeError(
                f"Unity Catalog schema {prompt_namespace!r} does not exist or is "
                "not visible. Set `platform.schema` in aai-platform.yml to an "
                f"approved schema. Visible schemas: {visible_schemas or 'none'}."
            ) from exc

    lifecycle = importlib.import_module("examples.lifecycle_support")
    experiment_name = lifecycle.lifecycle_experiment_name(context)
    try:
        experiment = environment.mlflow.set_experiment(experiment_name)
        experiment_id = str(experiment.experiment_id)
    except Exception as exc:
        raise RuntimeError(
            f"MLflow experiment {experiment_name!r} could not be resolved or "
            "created with the current identity."
        ) from exc

    preflight = DatabricksEvidencePreflight(
        context=context,
        workspace=workspace,
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        catalog=catalog,
        schema=schema,
        evidence_destination=environment.evidence_destination,
        azure_account=account,
        current_user=current_user,
    )
    print("EVIDENCE PREFLIGHT PASSED")
    print(
        {
            "azure_account": account.get("name"),
            "databricks_user": getattr(current_user, "user_name", None),
            "workspace": os.environ["DATABRICKS_HOST"],
            "experiment": experiment_name,
            "experiment_id": experiment_id,
            "evidence_destination": environment.evidence_destination.value,
            "dataset_namespace": (
                f"{catalog}.{schema}"
                if environment.evidence_destination is EvidenceDestination.DATABRICKS
                else "local SQLite"
            ),
        }
    )
    return preflight


def preflight_databricks(
    environment: NotebookEnvironment,
    *,
    which: Callable[[str], str | None] | None = None,
    run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    bootstrap_fn: Callable[[Path], Any] | None = None,
) -> DatabricksPreflight:
    """Verify governed evidence access and model endpoint readiness."""

    evidence = preflight_databricks_evidence(
        environment,
        which=which,
        run_command=run_command,
        bootstrap_fn=bootstrap_fn,
    )
    context = evidence.context
    workspace = evidence.workspace
    model_config = dict(context.settings.models.get("general-chat", {}))
    provider = str(model_config.get("provider", ""))
    if provider != "databricks":
        raise RuntimeError(
            "This connected tutorial expects `general-chat` to use provider "
            f"'databricks'; found {provider!r}."
        )

    deployment = str(model_config.get("deployment", ""))
    if not deployment or deployment.startswith("replace-"):
        choices = ", ".join(_ready_chat_endpoints(workspace)) or "none visible"
        raise RuntimeError(
            "Set `providers.models.general-chat.deployment` in aai-platform.yml. "
            f"READY chat endpoints visible to you: {choices}."
        )
    try:
        endpoint = workspace.serving_endpoints.get(deployment).as_dict()
    except Exception as exc:
        raise RuntimeError(
            f"Endpoint {deployment!r} was not found or is not visible to your "
            "identity."
        ) from exc
    if endpoint.get("task") != "llm/v1/chat":
        raise RuntimeError(
            f"Endpoint {deployment!r} has task {endpoint.get('task')!r}; choose a "
            "chat endpoint."
        )
    if endpoint.get("state", {}).get("ready") != "READY":
        raise RuntimeError(
            f"Endpoint {deployment!r} is not READY: {endpoint.get('state')!r}."
        )

    # Resolve the logical name only after every non-billable access check passes.
    model = context.providers.model("general-chat")
    preflight = DatabricksPreflight(
        context=context,
        workspace=workspace,
        experiment_name=evidence.experiment_name,
        experiment_id=evidence.experiment_id,
        catalog=evidence.catalog,
        schema=evidence.schema,
        evidence_destination=evidence.evidence_destination,
        azure_account=evidence.azure_account,
        current_user=evidence.current_user,
        model=model,
        deployment=deployment,
        endpoint=endpoint,
    )
    print("PREFLIGHT PASSED")
    print(
        {
            "azure_account": evidence.azure_account.get("name"),
            "databricks_user": getattr(evidence.current_user, "user_name", None),
            "workspace": os.environ["DATABRICKS_HOST"],
            "deployment": model.model,
            "endpoint_state": endpoint.get("state", {}).get("ready"),
            "experiment": evidence.experiment_name,
            "evidence_destination": environment.evidence_destination.value,
            "prompt_namespace": (
                f"{context.settings.catalog}.{context.settings.schema}"
                if environment.evidence_destination is EvidenceDestination.DATABRICKS
                else "local SQLite"
            ),
        }
    )
    return preflight


def get_or_create_uc_evaluation_dataset(
    *,
    evidence: DatabricksEvidencePreflight,
    dataset_name: str,
    records: list[dict[str, Any]],
    mlflow_module: Any | None = None,
) -> Any:
    """Return an idempotently populated UC EvaluationDataset for a notebook lab."""

    if evidence.evidence_destination is not EvidenceDestination.DATABRICKS:
        raise ValueError(
            "Unity Catalog dataset registration requires Databricks evidence mode."
        )
    logical_name = dataset_name.strip()
    if not logical_name or "." in logical_name:
        raise ValueError(
            "dataset_name must be a non-blank logical name without catalog or schema"
        )
    if not records:
        raise ValueError("records must not be empty")

    mlflow = mlflow_module or importlib.import_module("mlflow")
    qualified_name = f"{evidence.catalog}.{evidence.schema}.{logical_name}"
    try:
        dataset = mlflow.genai.datasets.get_dataset(name=qualified_name)
    except Exception as exc:
        if not _is_missing_dataset(exc):
            raise
        # Databricks-managed EvaluationDatasets reject MLflow dataset tags.
        # Governed context belongs on runs and UC securables instead.
        dataset = mlflow.genai.datasets.create_dataset(
            name=qualified_name,
            experiment_id=evidence.experiment_id,
        )

    experiment_ids = {
        str(experiment_id) for experiment_id in (dataset.experiment_ids or [])
    }
    if evidence.experiment_id not in experiment_ids:
        raise RuntimeError(
            f"Unity Catalog dataset {qualified_name!r} is not associated with "
            f"MLflow experiment {evidence.experiment_id!r}. Databricks does not "
            "support adding experiment associations through this API; use a new "
            "approved dataset name or ask the platform owner to repair it."
        )
    dataset.merge_records(records)
    return dataset


def _is_missing_dataset(error: Exception) -> bool:
    error_code = str(getattr(error, "error_code", "")).upper()
    message = str(error).upper()
    return error_code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"} or any(
        marker in message
        for marker in ("NOT_FOUND", "RESOURCE_DOES_NOT_EXIST", "DOES NOT EXIST")
    )
