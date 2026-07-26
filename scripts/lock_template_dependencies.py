"""Generate exact universal transitive locks for every project template.

The template ``pyproject.toml.tmpl`` files remain the human-edited declaration
of supported ranges. This script selects their certified direct versions from
``dependency-policy.toml`` and asks the pinned uv toolchain to resolve the full
Python 3.11/3.12 graph. Databricks jobs and Apps both consume the resulting
plain requirements files, so they intentionally use exact pins without pip's
global hash mode (the private aai-core wheel is verified separately against
its release manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?P<extras>\[[^\]]+\])?")
NAME_NORMALIZER = re.compile(r"[-_.]+")


@dataclass(frozen=True)
class LockTarget:
    template: str
    name: str
    output: Path
    requirements: tuple[str, ...]


def normalized_name(value: str) -> str:
    return NAME_NORMALIZER.sub("-", value).lower()


def certified_requirements(
    requirements: list[str],
    *,
    certified: dict[str, str],
) -> tuple[str, ...]:
    result = []
    for requirement in requirements:
        match = REQUIREMENT.match(requirement)
        if match is None:
            raise ValueError(f"Cannot parse requirement {requirement!r}")
        name = match.group("name")
        version = certified.get(normalized_name(name))
        if version is None:
            raise ValueError(f"{name} has no certified dependency-policy version")
        result.append(f"{name}{match.group('extras') or ''}=={version}")
    return tuple(sorted(result, key=str.casefold))


def discover_targets() -> list[LockTarget]:
    with (ROOT / "dependency-policy.toml").open("rb") as stream:
        policy = tomllib.load(stream)
    certified = {
        normalized_name(name): str(details["certified"])
        for name, details in policy["packages"].items()
    }
    targets = []
    for template in sorted(TEMPLATES.iterdir()):
        project_file = template / "template" / "pyproject.toml.tmpl"
        if not project_file.is_file():
            continue
        project = tomllib.loads(project_file.read_text(encoding="utf-8"))["project"]
        targets.append(
            LockTarget(
                template=template.name,
                name="runtime",
                output=template / "template" / "requirements.lock",
                requirements=certified_requirements(
                    list(project["dependencies"]),
                    certified=certified,
                ),
            )
        )
        optional = project.get("optional-dependencies", {})
        recipes = template / "template" / "recipes"
        if recipes.is_dir():
            for recipe in sorted(path for path in recipes.iterdir() if path.is_dir()):
                if recipe.name not in optional:
                    raise ValueError(
                        f"{template.name}/{recipe.name} has no matching optional "
                        "dependency group"
                    )
                targets.append(
                    LockTarget(
                        template=template.name,
                        name=f"recipe:{recipe.name}",
                        output=recipe / "requirements.lock",
                        requirements=certified_requirements(
                            list(optional[recipe.name]),
                            certified=certified,
                        ),
                    )
                )
    return targets


def render_lock(
    target: LockTarget,
    *,
    uv: str,
    offline: bool,
    existing_lock: Path | None = None,
) -> str:
    direct = "\n".join(target.requirements) + "\n"
    digest = hashlib.sha256(direct.encode("utf-8")).hexdigest()
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        inputs = scratch_path / "requirements.in"
        output = scratch_path / "requirements.lock"
        inputs.write_text(direct, encoding="utf-8")
        command = [
            uv,
            "pip",
            "compile",
            str(inputs),
            "--universal",
            "--python-version",
            "3.11",
            "--no-annotate",
            "--no-header",
            "--quiet",
            "--output-file",
            str(output),
        ]
        if existing_lock is None:
            command.append("--upgrade")
        else:
            # A lock does not become stale merely because a newer compatible
            # transitive release appears. Check the existing exact graph as a
            # constraint; intentional regeneration is the only upgrade path.
            command.extend(["--constraints", str(existing_lock)])
        if offline:
            command.append("--offline")
        subprocess.run(command, cwd=ROOT, check=True)
        resolved = output.read_text(encoding="utf-8")
    direct_display = ", ".join(target.requirements)
    return (
        "# Certified universal transitive runtime lock for Python >=3.11,<3.13.\n"
        "# Exact versions prevent dependency drift; the private aai-core wheel is\n"
        "# checksum-verified separately by its immutable release manifest.\n"
        "# Regenerate: python scripts/lock_template_dependencies.py\n"
        f"# Direct-input-sha256: {digest}\n"
        f"# Direct inputs: {direct_display}\n"
        f"{resolved}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--uv", default="uv")
    arguments = parser.parse_args()

    changed = []
    for target in discover_targets():
        rendered = render_lock(
            target,
            uv=arguments.uv,
            offline=arguments.offline,
            existing_lock=(
                target.output if arguments.check and target.output.is_file() else None
            ),
        )
        current = (
            target.output.read_text(encoding="utf-8")
            if target.output.is_file()
            else None
        )
        if current == rendered:
            continue
        changed.append(f"{target.template}/{target.name}")
        if not arguments.check:
            target.output.write_text(rendered, encoding="utf-8")

    if arguments.check and changed:
        print("Template dependency locks need regeneration:")
        for target in changed:
            print(f"- {target}")
        return 1
    if changed:
        print("Updated template dependency locks: " + ", ".join(changed))
    else:
        print("Template dependency locks are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
