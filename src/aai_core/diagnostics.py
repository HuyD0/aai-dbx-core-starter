"""Safe platform preflight diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from aai_core.evaluation import _is_placeholder, judge_model_uri
from aai_core.identity import identity_summary
from aai_core.providers.types import ProviderConfigurationError
from aai_core.runtime import PlatformSettings


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


def run_doctor(
    *,
    config_path: str | Path = "aai-platform.yml",
    check_cloud: bool = False,
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    try:
        settings = PlatformSettings.load(config_path)
        checks.append(DoctorCheck("configuration", "pass", "configuration is valid"))
    except Exception as error:
        return [DoctorCheck("configuration", "fail", str(error))]

    summary = identity_summary(settings.azure_identity)
    checks.extend(
        DoctorCheck(key, "info", value) for key, value in sorted(summary.items())
    )
    for module, extra in (
        ("databricks.sdk", "databricks"),
        ("azure.identity", "foundry or azure-search"),
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

            current = databricks_workspace_client().current_user.me()
            identity = getattr(current, "user_name", None) or getattr(
                current, "display_name", "authenticated"
            )
            checks.append(DoctorCheck("databricks", "pass", str(identity)))
        except Exception as error:
            checks.append(DoctorCheck("databricks", "fail", str(error)))
    return checks


def _lifecycle_checks(settings: PlatformSettings) -> list[DoctorCheck]:
    """Report lifecycle readiness; optional configuration skips, never fails."""

    checks = [
        DoctorCheck(
            "lifecycle:experiment",
            "pass",
            settings.effective_experiment_name,
        )
    ]

    # Same placeholder vocabulary the dataset helper rejects, so the doctor
    # never reports ready what the connected workflow will refuse.
    catalog = str(settings.catalog).strip()
    schema = str(settings.schema_name).strip()
    if (
        not catalog
        or not schema
        or "." in catalog
        or "." in schema
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
