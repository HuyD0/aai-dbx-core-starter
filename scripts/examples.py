#!/usr/bin/env python3
"""Clone-friendly setup, preflight, and runner for the learning examples."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "aai-platform.yml"
CONFIG_EXAMPLE = ROOT / "aai-platform.example.yml"
IDENTIFIERS_FILE = ROOT / "platform-identifiers.json"
LOCAL_DIR = ROOT / ".aai" / "local"
LOCAL_DB = LOCAL_DIR / "mlflow.db"
LOCAL_ARTIFACTS = LOCAL_DIR / "mlruns"


@dataclass(frozen=True)
class Example:
    name: str
    path: str
    description: str
    connected: bool
    local: bool = False
    modules: tuple[str, ...] = ()
    config_fields: tuple[str, ...] = ()
    interactive: bool = False


EXAMPLES = {
    example.name: example
    for example in (
        Example(
            name="offline_hello_world",
            path="examples/offline_hello_world.py",
            description="SDK contracts with in-memory fakes",
            connected=False,
        ),
        Example(
            name="first_trace",
            path="examples/first_trace.py",
            description="MLflow trace, first local and then in Databricks",
            connected=True,
            local=True,
            modules=("mlflow", "databricks.sdk"),
            config_fields=("platform.experiment_name",),
        ),
        Example(
            name="first_experiment",
            path="examples/first_experiment.py",
            description="Tagged MLflow run, first local and then in Databricks",
            connected=True,
            local=True,
            modules=("mlflow", "databricks.sdk"),
            config_fields=("platform.experiment_name",),
        ),
        Example(
            name="first_prompt",
            path="examples/first_prompt.py",
            description="Unity Catalog prompt registration and loading",
            connected=True,
            modules=("mlflow", "databricks.sdk"),
            config_fields=("platform.catalog", "platform.schema"),
        ),
        Example(
            name="first_evaluation",
            path="examples/first_evaluation.py",
            description="MLflow GenAI evaluation with an LLM judge",
            connected=True,
            modules=("mlflow", "databricks.sdk"),
            config_fields=("platform.experiment_name",),
        ),
        Example(
            name="first_llm_call",
            path="examples/first_llm_call.ipynb",
            description="Interactive first call to a configured serving endpoint",
            connected=True,
            modules=("databricks.sdk",),
            config_fields=("providers.models.general-chat.deployment",),
            interactive=True,
        ),
    )
}


def _identifiers() -> dict[str, str]:
    return json.loads(IDENTIFIERS_FILE.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _connected_environment() -> dict[str, str]:
    identifiers = _identifiers()
    environment = dict(os.environ)
    environment["DATABRICKS_HOST"] = identifiers["databricks_host"]
    environment["DATABRICKS_AUTH_TYPE"] = "azure-cli"
    environment["MLFLOW_TRACKING_URI"] = "databricks"
    environment["MLFLOW_REGISTRY_URI"] = "databricks-uc"
    environment["AAI_PLATFORM_CONFIG"] = str(CONFIG)
    return environment


def _local_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["MLFLOW_TRACKING_URI"] = f"sqlite:///{LOCAL_DB.resolve()}"
    environment["MLFLOW_REGISTRY_URI"] = environment["MLFLOW_TRACKING_URI"]
    environment["AAI_PLATFORM_CONFIG"] = str(CONFIG_EXAMPLE)
    return environment


def _normalize_example_name(value: str) -> str:
    name = Path(value).stem.replace("-", "_")
    if name not in EXAMPLES:
        choices = ", ".join(EXAMPLES)
        raise ValueError(f"Unknown example {value!r}. Choose one of: {choices}")
    return name


def _module_issues(example: Example) -> list[str]:
    issues = []
    for module in example.modules:
        try:
            available = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            available = False
        if not available:
            issues.append(
                f"Python module {module!r} is missing from {sys.executable}; "
                "run `make examples-install`, then use a `make local-*` or "
                "`make workspace-*` target."
            )
    return issues


def _load_config() -> dict[str, Any]:
    with CONFIG.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError(f"{CONFIG} must contain a YAML mapping")
    return document


def _nested_value(document: dict[str, Any], dotted_path: str) -> Any:
    value: Any = document
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    normalized = str(value).strip().lower()
    return (
        normalized in {"", "unset"}
        or normalized.startswith("replace-with-")
        or "<" in normalized
        or ">" in normalized
    )


def _config_issues(example: Example) -> list[str]:
    if not example.connected:
        return []
    if not CONFIG.is_file():
        return [
            "aai-platform.yml is missing; run `make workspace-connect` to create it."
        ]
    try:
        document = _load_config()
    except (OSError, ValueError, yaml.YAMLError) as error:
        return [f"Cannot load aai-platform.yml: {error}"]
    issues = []
    for dotted_path in example.config_fields:
        value = _nested_value(document, dotted_path)
        if _is_placeholder(value):
            issues.append(
                f"Configure `{dotted_path}` in aai-platform.yml "
                f"(current value: {value!r})."
            )
    return issues


def _run_check(
    command: list[str],
    *,
    environment: dict[str, str],
) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return False, f"{command[0]} is not installed or not on PATH"
    except subprocess.TimeoutExpired:
        return False, f"{command[0]} did not respond within 30 seconds"
    detail = (result.stderr or result.stdout).strip().splitlines()
    return result.returncode == 0, detail[-1] if detail else "command failed"


def _cloud_issues(environment: dict[str, str]) -> list[str]:
    issues = []
    identifiers = _identifiers()
    if shutil.which("az") is None:
        issues.append("Azure CLI is missing; install `az`, then run `az login`.")
        return issues
    try:
        account = subprocess.run(
            [
                "az",
                "account",
                "show",
                "--query",
                "{tenant:tenantId,subscription:id}",
                "--output",
                "json",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        issues.append("Azure CLI did not respond; verify the `az` installation.")
        return issues
    if account.returncode != 0:
        issues.append(
            "Azure CLI is not authenticated; run "
            f"`az login --tenant {identifiers['azure_tenant_id']}`."
        )
        return issues
    try:
        current = json.loads(account.stdout)
    except json.JSONDecodeError:
        issues.append("Azure CLI returned an unreadable account response.")
        return issues
    if current.get("tenant") != identifiers["azure_tenant_id"]:
        issues.append(
            "Azure CLI is using the wrong tenant; run "
            f"`az login --tenant {identifiers['azure_tenant_id']}`."
        )
    if current.get("subscription") != identifiers["azure_subscription_id"]:
        issues.append(
            "Azure CLI is using the wrong subscription; run "
            "`az account set --subscription "
            f"{identifiers['azure_subscription_id']}`."
        )
    if issues:
        return issues
    if shutil.which("databricks") is None:
        issues.append("Databricks CLI is missing or not on PATH.")
        return issues
    connected, detail = _run_check(
        ["databricks", "current-user", "me"],
        environment=environment,
    )
    if not connected:
        issues.append(f"Databricks authentication failed: {detail}")
    return issues


def _print_issues(issues: list[str]) -> None:
    print("Example preflight did not pass:")
    for issue in issues:
        print(f"  - {issue}")


def _print_interactive_instructions(
    example: Example,
    environment: dict[str, str],
) -> None:
    print(f"{example.path} is interactive. Export the configured environment:")
    for name in (
        "DATABRICKS_HOST",
        "DATABRICKS_AUTH_TYPE",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_REGISTRY_URI",
        "AAI_PLATFORM_CONFIG",
    ):
        print(f"export {name}={shlex.quote(environment[name])}")
    print(f"jupyter lab {shlex.quote(str(ROOT / example.path))}")


def quickstart() -> int:
    print("Running the credential-free example...", flush=True)
    result = subprocess.run(
        [sys.executable, str(ROOT / EXAMPLES["offline_hello_world"].path)],
        cwd=ROOT,
        check=False,
    )
    if result.returncode:
        return result.returncode
    print("\nQuickstart passed. No credentials or cloud configuration were used.")
    print("Next: run `make local-start` to create a local MLflow trace.")
    return 0


def connect() -> int:
    created = False
    if not CONFIG.exists():
        shutil.copyfile(CONFIG_EXAMPLE, CONFIG)
        created = True
    if created:
        print(f"Created local configuration: {_display_path(CONFIG)}")
    else:
        print(f"Using existing local configuration: {_display_path(CONFIG)}")

    environment = _connected_environment()
    core_issues = _config_issues(EXAMPLES["first_trace"])
    cloud_issues = _cloud_issues(environment)
    model_issues = _config_issues(EXAMPLES["first_llm_call"])
    print(
        "\nConnected examples use keyless Azure CLI authentication and "
        f"{environment['DATABRICKS_HOST']}."
    )
    if core_issues:
        print("\nConfiguration still needed for connected examples:")
        for issue in core_issues:
            print(f"  - {issue}")
    if cloud_issues:
        print("\nAuthentication still needed:")
        for issue in cloud_issues:
            print(f"  - {issue}")
    if model_issues:
        print("\nAdditional setup needed before model-calling examples:")
        for issue in model_issues:
            print(f"  - {issue}")
    if core_issues or cloud_issues:
        print(
            "\nAfter addressing the items above, rerun `make workspace-connect`, "
            "then use `make workspace-example EXAMPLE=first_trace`."
        )
    else:
        print("\nConnected identity and experiment preflight passed.")
        print("Run an example with `make workspace-example EXAMPLE=first_trace`.")
    return 2 if core_issues or cloud_issues else 0


def run_example(value: str, *, destination: str = "workspace") -> int:
    try:
        name = _normalize_example_name(value)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    example = EXAMPLES[name]
    if destination == "local":
        if not example.local:
            print(
                f"{example.name} requires workspace services and cannot run locally. "
                "Choose first_trace or first_experiment.",
                file=sys.stderr,
            )
            return 2
        LOCAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        environment = _local_environment()
        issues = _module_issues(example)
    else:
        environment = (
            _connected_environment() if example.connected else dict(os.environ)
        )
        issues = [*_module_issues(example), *_config_issues(example)]
        if example.connected and not issues:
            issues.extend(_cloud_issues(environment))
    if issues:
        _print_issues(issues)
        return 2
    if example.interactive:
        _print_interactive_instructions(example, environment)
        return 0
    print(f"Running {example.path} against {destination}...", flush=True)
    result = subprocess.run(
        [sys.executable, str(ROOT / example.path)],
        cwd=LOCAL_DIR if destination == "local" else ROOT,
        env=environment,
        check=False,
    )
    if result.returncode == 0 and destination == "local":
        print(f"\nLocal MLflow data: {LOCAL_DB}")
        print("View it with `make local-ui` (Ctrl-C stops the server).")
        print(
            "Next: run `make workspace-connect`, then "
            f"`make workspace-example EXAMPLE={example.name}`."
        )
    elif result.returncode == 0 and example.connected:
        print(
            "\nWorkspace run complete. View the configured experiment in "
            f"{environment['DATABRICKS_HOST']}."
        )
    return result.returncode


def list_examples() -> int:
    print("Available learning examples:")
    for example in EXAMPLES.values():
        mode = "workspace"
        if not example.connected:
            mode = "offline"
        elif example.local:
            mode = "local → workspace"
        if example.interactive:
            mode += ", interactive"
        print(f"  {example.name:<22} [{mode}] {example.description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("quickstart")
    subcommands.add_parser("connect")
    subcommands.add_parser("list")
    local_parser = subcommands.add_parser("local")
    local_parser.add_argument("example")
    workspace_parser = subcommands.add_parser("workspace")
    workspace_parser.add_argument("example")
    # Backward-compatible alias for the original connected runner command.
    run_parser = subcommands.add_parser("run")
    run_parser.add_argument("example")
    arguments = parser.parse_args(argv)
    if arguments.command == "quickstart":
        return quickstart()
    if arguments.command == "connect":
        return connect()
    if arguments.command == "list":
        return list_examples()
    destination = "local" if arguments.command == "local" else "workspace"
    return run_example(arguments.example, destination=destination)


if __name__ == "__main__":
    raise SystemExit(main())
