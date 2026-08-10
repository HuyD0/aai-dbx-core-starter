"""Safe platform preflight diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from aai_core.identity import identity_summary
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
