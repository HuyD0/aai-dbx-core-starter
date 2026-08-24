"""Safe platform preflight diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from re import fullmatch
from typing import Any, cast

from aai_core.evaluation import (
    _NAME_COMPONENT,
    _is_placeholder,
    _is_placeholder_path,
    judge_model_uri,
)
from aai_core.identity import identity_summary
from aai_core.providers.types import ProviderConfigurationError
from aai_core.runtime import PlatformSettings

__all__ = ["DoctorCheck", "main", "run_doctor"]


@dataclass(frozen=True)
class DoctorCheck:
    """One safe diagnostic result with pass, fail, skip, or info status."""

    name: str
    status: str
    detail: str


def run_doctor(
    *,
    config_path: str | Path = "aai-platform.yml",
    check_cloud: bool = False,
) -> list[DoctorCheck]:
    """Run safe local preflight checks without accepting missing config."""

    checks: list[DoctorCheck] = []
    resolved_config = Path(config_path)
    if not resolved_config.is_file():
        return [
            DoctorCheck(
                "configuration",
                "fail",
                f"configuration file does not exist: {resolved_config}",
            )
        ]
    try:
        settings = PlatformSettings.load(resolved_config)
        checks.append(DoctorCheck("configuration", "pass", "configuration is valid"))
    except Exception as error:
        return [DoctorCheck("configuration", "fail", str(error))]

    summary = identity_summary(settings.azure_identity)
    checks.extend(
        DoctorCheck(key, "info", value) for key, value in sorted(summary.items())
    )
    for module, extra in (
        ("databricks.sdk", "databricks"),
        ("azure.identity", "azure-apim or azure-search"),
        ("mlflow", "genai"),
    ):
        status = "pass" if _module_available(module) else "skip"
        checks.append(
            DoctorCheck(
                f"dependency:{module}",
                status,
                "installed" if status == "pass" else f"install aai-core[{extra}]",
            )
        )

    checks.extend(_lifecycle_checks(settings))

    if check_cloud:
        try:
            from aai_core.identity import databricks_workspace_client

            workspace = cast(Any, databricks_workspace_client())
            current = workspace.current_user.me()
            identity = getattr(current, "user_name", None) or getattr(
                current, "display_name", "authenticated"
            )
            checks.append(DoctorCheck("databricks", "pass", str(identity)))
        except Exception as error:
            checks.append(DoctorCheck("databricks", "fail", str(error)))
    return checks


def _lifecycle_checks(settings: PlatformSettings) -> list[DoctorCheck]:
    """Report lifecycle readiness; optional configuration skips, never fails."""

    # An explicit placeholder name would pass straight through
    # effective_experiment_name to get_experiment_by_name(), and a derived
    # /Shared/<team>-<project>-<application> name built from placeholder
    # components (/Shared/unset-unset-unset) is just as unconfigured.
    experiment = settings.effective_experiment_name
    resource = settings.resource
    derived = settings.experiment_name in {"", "unset"}
    unconfigured = _is_placeholder_path(experiment) or (
        derived
        and any(
            _is_placeholder(component)
            for component in (resource.team, resource.project, resource.application)
        )
    )
    if unconfigured:
        checks = [
            DoctorCheck(
                "lifecycle:experiment",
                "skip",
                "set platform.experiment_name to a real experiment path, or "
                "leave it 'unset' and configure platform.team, "
                "platform.project, and platform.application so the derived "
                "/Shared/<team>-<project>-<application> name is real",
            )
        ]
    else:
        checks = [DoctorCheck("lifecycle:experiment", "pass", experiment)]

    # Same placeholder vocabulary the dataset helper rejects, so the doctor
    # never reports ready what the connected workflow will refuse.
    catalog = str(settings.catalog).strip()
    schema = str(settings.schema_name).strip()
    # The identifier shape covers blank, dotted, and invalid-character
    # values in one check — the same shape the SDK helpers enforce.
    if (
        not fullmatch(_NAME_COMPONENT, catalog)
        or not fullmatch(_NAME_COMPONENT, schema)
        or _is_placeholder(catalog)
        or _is_placeholder(schema)
    ):
        checks.append(
            DoctorCheck(
                "lifecycle:prompt-registry",
                "skip",
                "set platform.catalog and platform.schema to enable the "
                "governed prompt registry and evaluation datasets",
            )
        )
    else:
        checks.append(
            DoctorCheck("lifecycle:prompt-registry", "pass", f"{catalog}.{schema}")
        )

    try:
        checks.append(
            DoctorCheck("lifecycle:judge-model", "pass", judge_model_uri(settings))
        )
    except ProviderConfigurationError as error:
        checks.append(DoctorCheck("lifecycle:judge-model", "skip", str(error)))
    return checks


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line doctor and return a process exit code."""

    parser = argparse.ArgumentParser(prog="aai-core")
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser("doctor")
    doctor.add_argument("--config", default="aai-platform.yml")
    doctor.add_argument("--cloud", action="store_true")
    arguments = parser.parse_args(argv)

    checks = run_doctor(config_path=arguments.config, check_cloud=arguments.cloud)
    print(json.dumps([asdict(check) for check in checks], indent=2))
    return 1 if any(check.status == "fail" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
