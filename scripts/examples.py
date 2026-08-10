#!/usr/bin/env python3
"""Clone-friendly setup, preflight, and runner for the learning examples."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
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
            path="examples/00_offline_hello_world.py",
            description="SDK contracts with in-memory fakes",
            connected=False,
        ),
        Example(
            name="first_trace",
            path="examples/01_first_trace.py",
            description="Governed baseline trace with bounded cost/usage evidence",
            connected=True,
            local=True,
            modules=("mlflow",),
            config_fields=("platform.experiment_name",),
        ),
        Example(
            name="first_experiment",
            path="examples/02_first_experiment.py",
            description="Reproducible baseline/change MLflow comparison",
            connected=True,
            local=True,
            modules=("mlflow",),
            config_fields=("platform.experiment_name",),
        ),
        Example(
            name="first_prompt",
            path="examples/03_first_prompt.py",
            description="Exact prompt versions, digests, links, and safe render trace",
            connected=True,
            local=True,
            modules=("mlflow",),
            config_fields=("platform.catalog", "platform.schema"),
        ),
        Example(
            name="first_evaluation",
            path="examples/04_first_evaluation.py",
            description="Deterministic MLflow GenAI gate and release decision",
            connected=True,
            local=True,
            modules=("mlflow",),
            config_fields=(
                "platform.experiment_name",
                "platform.catalog",
                "platform.schema",
            ),
        ),
        Example(
            name="connected_setup",
            path="examples/05_connected_setup.ipynb",
            description="Kernel, keyless identity, workspace, and endpoint preflight",
            connected=True,
            modules=("databricks.sdk", "databricks_openai", "mlflow", "pandas"),
            config_fields=(
                "providers.models.general-chat.deployment",
                "platform.experiment_name",
            ),
            interactive=True,
        ),
        Example(
            name="connected_first_call",
            path="examples/06_connected_first_call.py",
            description="Real governed LLM call through stable model.generate()",
            connected=True,
            modules=("databricks_openai", "mlflow"),
            config_fields=(
                "providers.models.general-chat.deployment",
                "platform.experiment_name",
            ),
        ),
        Example(
            name="first_llm_call",
            path="examples/07_first_llm_call.ipynb",
            description=(
                "Advanced native async streaming comparison with MLflow autologging"
            ),
            connected=True,
            modules=("databricks.sdk", "mlflow"),
            config_fields=(
                "providers.models.general-chat.deployment",
                "platform.catalog",
                "platform.schema",
            ),
            interactive=True,
        ),
        Example(
            name="tool_trajectory_evaluation",
            path="examples/08_tool_trajectory_evaluation.ipynb",
            description="Exact tool-call trajectories and critical-case gates",
            connected=False,
            local=True,
            modules=("pandas",),
            interactive=True,
        ),
        Example(
            name="multi_turn_session_evaluation",
            path="examples/09_multi_turn_session_evaluation.ipynb",
            description="Session-scoped conversation metrics and judge handoff",
            connected=False,
            local=True,
            modules=("pandas",),
            interactive=True,
        ),
        Example(
            name="layered_judges",
            path="examples/10_layered_judges.ipynb",
            description="Deterministic rules, nuanced judges, and held-out agreement",
            connected=False,
            local=True,
            modules=("pandas",),
            interactive=True,
        ),
        Example(
            name="cost_quality_tradeoff",
            path="examples/11_cost_quality_tradeoff.ipynb",
            description="Quality-first logical-model cost comparison",
            connected=False,
            local=True,
            modules=("pandas",),
            interactive=True,
        ),
        Example(
            name="agent_alignment_optimization",
            path="examples/12_agent_alignment_optimization.ipynb",
            description="Disabled-by-default aligned-judge and optimizer workflow",
            connected=False,
            local=True,
            interactive=True,
        ),
        Example(
            name="decision_promotion_lifecycle",
            path="examples/13_decision_and_promotion_lifecycle.ipynb",
            description="Recorded adopt/reject decisions gate prompt promotion",
            connected=False,
            local=True,
            interactive=True,
        ),
        Example(
            name="platform_llm_operations",
            path="examples/14_platform_llm_operations.ipynb",
            description="Platform-team judge, gateway-tag, cost, and fleet loop",
            connected=True,
            modules=("databricks.sdk", "mlflow"),
            config_fields=("platform.catalog", "platform.schema"),
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
    environment["AAI_EXAMPLE_LOCAL_DIR"] = str(LOCAL_DIR.resolve())
    environment["AAI_EXAMPLE_ARTIFACT_ROOT"] = str(LOCAL_ARTIFACTS.resolve())
    return environment


def _normalize_example_name(value: str) -> str:
    name = Path(value).stem.replace("-", "_")
    prefix, separator, unnumbered = name.partition("_")
    if separator and len(prefix) == 2 and prefix.isdigit():
        name = unnumbered
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
    # The shared platform placeholder vocabulary; kept literal because this
    # script stays stdlib-only (SDK homes: aai_core.tags._PLACEHOLDERS and
    # aai_core.evaluation._QUALIFIER_PLACEHOLDERS).
    return (
        normalized in {"", "unset", "unknown", "todo", "changeme"}
        or normalized.startswith("replace-with-")
        or "<" in normalized
        or ">" in normalized
    )


# Unity Catalog qualifier fields: exactly one level each, so a dotted value
# is a configuration error the connected SDK helpers will reject.
def _is_placeholder_path(value: Any) -> bool:
    """Component-aware placeholder test for slash-separated paths.

    ``_is_placeholder`` matches the bare markers exactly and anchors
    ``replace-with-`` at the start, so a placeholder inside a path
    (``/Shared/unset``) looks configured. Mirrors
    ``aai_core.evaluation._is_placeholder_path``.
    """

    return _is_placeholder(value) or any(
        _is_placeholder(part) for part in str(value).split("/") if part
    )


_QUALIFIER_FIELDS = {"platform.catalog", "platform.schema"}

# Governed values that arrive as slash-separated paths, so the placeholder
# vocabulary has to be applied per component rather than to the whole string.
_PATH_FIELDS = {"platform.experiment_name"}


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
        if value is not None and not isinstance(value, str):
            # Every configured identifier is a string. The checks below
            # stringify, so 123 would read as a configured name and pass
            # the qualifier regex as "123", leaving strict PlatformSettings
            # to reject it at bootstrap — after the cloud preflight ran.
            # A missing key stays None so it still reports as unconfigured.
            issues.append(
                f"`{dotted_path}` must be a string in aai-platform.yml "
                f"(current value: {value!r})."
            )
        elif (
            _is_placeholder_path(value)
            if dotted_path in _PATH_FIELDS
            else _is_placeholder(value)
        ):
            issues.append(
                f"Configure `{dotted_path}` in aai-platform.yml "
                f"(current value: {value!r})."
            )
        elif dotted_path in _QUALIFIER_FIELDS and not re.fullmatch(
            r"[A-Za-z0-9_-]+", str(value).strip()
        ):
            # The SDK's qualifier validation rejects dots and any character
            # outside the identifier set; fail the preflight instead of
            # opening cloud checks that will refuse.
            issues.append(
                f"`{dotted_path}` must be a single Unity Catalog qualifier "
                "(letters, digits, underscores, and hyphens; no dots) "
                f"(current value: {value!r})."
            )
    if "platform.experiment_name" not in example.config_fields:
        issue = _effective_experiment_issue(document)
        if issue:
            issues.append(issue)
    return issues


def _effective_experiment_issue(document: dict[str, Any]) -> str | None:
    """Validate the experiment the SDK will actually use.

    An explicit name wins unless it is a placeholder; the 'unset' sentinel
    derives /Shared/<team>-<project>-<application>, which is only as
    configured as its components.
    """

    platform = document.get("platform")
    platform = platform if isinstance(platform, dict) else {}
    explicit = platform.get("experiment_name")
    if explicit not in (None, "", "unset"):
        # _is_placeholder stringifies, so a number or list would read as a
        # configured name here and only fail later inside strict
        # PlatformSettings — after the cloud preflight has already run.
        if not isinstance(explicit, str):
            return (
                "`platform.experiment_name` must be a string experiment path "
                f"(current value: {explicit!r})."
            )
        if _is_placeholder_path(explicit):
            return (
                "Configure `platform.experiment_name` in aai-platform.yml "
                f"(current value: {explicit!r})."
            )
        return None
    # _is_placeholder stringifies, so a numeric component would read as
    # configured and only fail inside strict PlatformSettings at bootstrap.
    non_string = [
        f"platform.{field}"
        for field in ("team", "project", "application")
        if platform.get(field) is not None and not isinstance(platform.get(field), str)
    ]
    if non_string:
        return (
            "`platform.experiment_name` derives from "
            + ", ".join(non_string)
            + ", which must be strings in aai-platform.yml."
        )
    unset_components = [
        f"platform.{field}"
        for field in ("team", "project", "application")
        if _is_placeholder(platform.get(field))
    ]
    if unset_components:
        return (
            "`platform.experiment_name` derives from "
            + ", ".join(unset_components)
            + "; configure them or set an explicit experiment path."
        )
    return None


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
    print(f"{example.path} is interactive.")
    names: list[str] = []
    if example.connected:
        print("Export the configured environment:")
        names = [
            "DATABRICKS_HOST",
            "DATABRICKS_AUTH_TYPE",
            "AAI_PLATFORM_CONFIG",
        ]
        if example.name not in {"connected_setup", "first_llm_call"}:
            names[2:2] = ["MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI"]
    elif "MLFLOW_TRACKING_URI" in environment:
        print("The local runner selected this evidence environment:")
        names = [
            "MLFLOW_TRACKING_URI",
            "MLFLOW_REGISTRY_URI",
            "AAI_PLATFORM_CONFIG",
        ]
    for name in names:
        print(f"export {name}={shlex.quote(environment[name])}")
    print(f"Open {ROOT / example.path} in your preferred notebook editor.")
    print(f"Select this Python kernel: {sys.executable}")
    if example.connected:
        print(
            "A Databricks CLI profile is not required: the notebook uses the "
            "keyless Azure CLI authentication configured above."
        )
    else:
        print("The default path is credential-free and makes no model request.")
    if example.name == "first_llm_call":
        print(
            "Keep SEND_EVIDENCE_TO_DATABRICKS = False for local MLflow, or set "
            "it to True and restart the kernel to store prompts, runs, and "
            "traces in Databricks."
        )


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
            choices = ", ".join(item.name for item in EXAMPLES.values() if item.local)
            print(
                f"{example.name} requires workspace services and cannot run locally. "
                f"Choose one of: {choices}.",
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
            mode = "local" if example.local else "offline"
        elif example.local:
            mode = "local → workspace"
        if example.interactive:
            mode += ", interactive"
        print(f"  {example.name:<30} [{mode}] {example.description}")
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
