"""Run OSV-backed pip-audit with narrow, expiring applicability exceptions."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

IGNORED_DIRECTORIES = {".git", ".mypy_cache", ".pytest_cache", ".venv"}


def _source_uses(symbol: str, root: Path) -> bool:
    for path in root.rglob("*.py"):
        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue
        if symbol in path.read_text(encoding="utf-8", errors="ignore"):
            return True
    return False


def _active_exceptions(policy_path: Path, root: Path) -> list[str]:
    with policy_path.open("rb") as stream:
        policy = tomllib.load(stream)
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported security-audit policy schema")

    today = dt.date.today()
    active: list[str] = []
    for exception in policy.get("exceptions", []):
        identifier = str(exception["id"])
        expiry = dt.date.fromisoformat(str(exception["expires"]))
        if expiry < today:
            raise RuntimeError(
                f"Security audit exception {identifier} expired on {expiry}"
            )
        used = [
            symbol
            for symbol in exception.get("forbidden_symbols", [])
            if _source_uses(str(symbol), root)
        ]
        if used:
            raise RuntimeError(
                f"Security audit exception {identifier} is not applicable; "
                f"source uses affected symbol(s): {', '.join(used)}"
            )
        print(
            f"active audit exception: {identifier} until {expiry}: "
            f"{exception['reason']}",
            file=sys.stderr,
        )
        active.append(identifier)
    return active


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--uv-project",
        type=Path,
        help="Export and audit every locked optional dependency from a uv project.",
    )
    source.add_argument(
        "--installed-python",
        type=Path,
        help="Freeze and audit a resolved environment, excluding editable projects.",
    )
    parser.add_argument("--uv", default="uv")
    arguments, pip_audit_arguments = parser.parse_known_args(argv)

    ignored = _active_exceptions(arguments.policy, arguments.source_root)
    with tempfile.TemporaryDirectory(prefix="aai-dependency-audit-") as scratch:
        if arguments.uv_project is not None:
            lock = Path(scratch) / "requirements.lock"
            subprocess.run(
                [
                    arguments.uv,
                    "export",
                    "--locked",
                    "--all-extras",
                    "--no-emit-project",
                    "--no-hashes",
                    "--output-file",
                    str(lock),
                ],
                cwd=arguments.uv_project,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            pip_audit_arguments.extend(["--requirement", str(lock)])
        elif arguments.installed_python is not None:
            lock = Path(scratch) / "requirements.lock"
            frozen = subprocess.run(
                [
                    arguments.uv,
                    "pip",
                    "freeze",
                    "--strict",
                    "--exclude-editable",
                    "--python",
                    str(arguments.installed_python),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            lock.write_text(frozen.stdout, encoding="utf-8")
            pip_audit_arguments.extend(["--requirement", str(lock)])

        command = [
            sys.executable,
            "-m",
            "pip_audit",
            "--vulnerability-service",
            "osv",
            "--progress-spinner",
            "off",
            "--strict",
        ]
        if any(
            argument in {"-r", "--requirement"} or argument.startswith("--requirement=")
            for argument in pip_audit_arguments
        ):
            command.extend(["--no-deps", "--disable-pip"])
        for identifier in ignored:
            command.extend(["--ignore-vuln", identifier])
        command.extend(pip_audit_arguments)
        return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
