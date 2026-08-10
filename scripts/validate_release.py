"""Validate SDK, template, dependency, wheel, and release-version invariants.

The default mode is credential-free and safe for pull requests. Release mode
adds changelog and immutable-tag checks. When a wheel is supplied, the exact
artifact metadata is validated and an optional release manifest can be written
for publication beside the wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
NAME_NORMALIZER = re.compile(r"[-_.]+")
PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==(?P<version>[^;\s]+)")
REQUIREMENT_NAME = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)")
REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]+\])?(?P<specifier>[^;]*)"
)
DIRECT_REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?P<extras>\[[^\]]+\])?")


def normalized_name(value: str) -> str:
    base = value.split("[", 1)[0]
    return NAME_NORMALIZER.sub("-", base).lower()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def discover_templates() -> list[Path]:
    return sorted(
        path
        for path in TEMPLATES.iterdir()
        if path.is_dir() and (path / "databricks_template_schema.json").is_file()
    )


def certified_pins(policy: dict[str, Any]) -> dict[str, str]:
    return {
        normalized_name(name): str(config["certified"])
        for name, config in policy["packages"].items()
    }


def requirement_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PIN.match(line.strip())
        if match:
            pins[normalized_name(match.group("name"))] = match.group("version")
    return pins


def unpinned_requirement_lines(path: Path) -> list[str]:
    """Return installable lock lines that are not exact ``==`` pins."""

    failures = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if PIN.match(line) is None:
            failures.append(line)
    return failures


def certified_input_requirements(
    requirements: list[str],
    *,
    policy_pins: dict[str, str],
) -> tuple[str, ...]:
    inputs = []
    for requirement in requirements:
        match = DIRECT_REQUIREMENT.match(requirement)
        if match is None:
            raise ValueError(f"Cannot parse requirement {requirement!r}")
        name = match.group("name")
        version = policy_pins.get(normalized_name(name))
        if version is None:
            raise ValueError(f"{name} has no certified dependency-policy version")
        inputs.append(f"{name}{match.group('extras') or ''}=={version}")
    return tuple(sorted(inputs, key=str.casefold))


def lock_source_issues(
    path: Path,
    *,
    expected_inputs: tuple[str, ...],
) -> list[str]:
    text = path.read_text(encoding="utf-8")
    input_match = re.search(r"^# Direct inputs: (.+)$", text, re.MULTILINE)
    digest_match = re.search(
        r"^# Direct-input-sha256: ([0-9a-f]{64})$",
        text,
        re.MULTILINE,
    )
    if input_match is None or digest_match is None:
        return ["lock is missing generated direct-input provenance"]
    actual_inputs = tuple(input_match.group(1).split(", "))
    issues = []
    if actual_inputs != expected_inputs:
        issues.append(
            f"lock direct inputs {actual_inputs} != certified inputs {expected_inputs}"
        )
    expected_digest = hashlib.sha256(
        ("\n".join(expected_inputs) + "\n").encode("utf-8")
    ).hexdigest()
    if digest_match.group(1) != expected_digest:
        issues.append("lock direct-input digest is stale")
    return issues


def requirement_names(requirements: list[str]) -> set[str]:
    names: set[str] = set()
    for requirement in requirements:
        match = REQUIREMENT_NAME.match(requirement)
        if match:
            names.add(normalized_name(match.group("name")))
    return names


def requirement_specs(requirements: list[str]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for requirement in requirements:
        match = REQUIREMENT.match(requirement.replace(" ", ""))
        if match:
            specs[normalized_name(match.group("name"))] = match.group("specifier")
    return specs


def validate_repository() -> (  # noqa: C901 - linear, independent release assertions
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
):
    failures: list[str] = []
    project = load_toml(ROOT / "pyproject.toml")["project"]
    policy = load_toml(ROOT / "dependency-policy.toml")
    compatibility = load_json(ROOT / "compatibility.json")
    toolchain = load_json(ROOT / "toolchain.json")
    sdk_version = str(project["version"])

    if sdk_version != compatibility["sdk"]["version"]:
        failures.append(
            "pyproject SDK version "
            f"{sdk_version} != compatibility SDK {compatibility['sdk']['version']}"
        )
    if sorted(compatibility["sdk"]["python"]) != sorted(policy["python"]["supported"]):
        failures.append("Python support differs between compatibility and policy")
    if sorted(compatibility["sdk"]["python"]) != sorted(
        toolchain["python"]["supported"]
    ):
        failures.append("Python support differs between compatibility and toolchain")

    setup = (ROOT / "scripts" / "codex-cloud-setup.sh").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for variable, key in (
        ("PYTHON_VERSION", ("python", "default")),
        ("UV_VERSION", ("uv",)),
        ("DATABRICKS_CLI_VERSION", ("databricks_cli",)),
        ("AZURE_CLI_VERSION", ("azure_cli",)),
    ):
        match = re.search(rf'{variable}="([^"]+)"', setup)
        expected: Any = toolchain
        for part in key:
            expected = expected[part]
        if not match or match.group(1) != expected:
            failures.append(
                f"{variable} in codex setup does not match toolchain value {expected}"
            )
    if f"UV_VERSION ?= {toolchain['uv']}" not in makefile:
        failures.append("Makefile uv version does not match toolchain.json")
    if f'pipx=={toolchain["pipx"]}' not in setup:
        failures.append("Codex setup pipx version does not match toolchain.json")

    workflow_files = list((ROOT / ".github" / "workflows").glob("*.yml"))
    workflow_files.extend(
        path
        for template in discover_templates()
        for path in (template / "template" / ".github" / "workflows").glob("*.yml")
    )
    if any(
        "runs-on: ubuntu-latest" in path.read_text(encoding="utf-8")
        for path in workflow_files
    ):
        failures.append("workflow runner must use the toolchain OS, not ubuntu-latest")
    action_cli_versions = {
        version
        for path in workflow_files
        for version in re.findall(
            r"databricks/setup-cli@[0-9a-f]{40}\s+#\s+v([0-9.]+)",
            path.read_text(encoding="utf-8"),
        )
    }
    if action_cli_versions != {toolchain["databricks_cli"]}:
        failures.append(
            "Databricks setup action comments do not match toolchain version: "
            f"{sorted(action_cli_versions)}"
        )

    fallback_text = (ROOT / "src" / "aai_core" / "__init__.py").read_text(
        encoding="utf-8"
    )
    if f'__version__ = "{sdk_version}"' not in fallback_text:
        failures.append("source-checkout __version__ fallback differs from pyproject")

    expected_templates = compatibility["templates"]
    discovered = {path.name: path for path in discover_templates()}
    if set(discovered) != set(expected_templates):
        failures.append(
            "template catalog differs from compatibility manifest: "
            f"catalog={sorted(discovered)}, manifest={sorted(expected_templates)}"
        )

    policy_pins = certified_pins(policy)
    policy_ranges = {
        normalized_name(name): str(config["supported"])
        for name, config in policy["packages"].items()
    }
    root_runtime_requirements = list(project["dependencies"])
    for extra, requirements in project.get("optional-dependencies", {}).items():
        if extra not in {"all", "dev", "examples"}:
            root_runtime_requirements.extend(requirements)
    for package, specifier in requirement_specs(root_runtime_requirements).items():
        if package == "aai-core":
            continue
        expected_range = policy_ranges.get(package)
        if expected_range is None:
            failures.append(f"root runtime dependency {package} has no policy entry")
        elif specifier != expected_range:
            failures.append(
                f"root {package}{specifier} != policy range {expected_range}"
            )

    for name, template in discovered.items():
        expected = expected_templates[name]
        lock_format = policy["resolution"]["template_lock_format"]
        if expected.get("dependency_lock_format") != lock_format:
            failures.append(
                f"{name}: compatibility lock format "
                f"{expected.get('dependency_lock_format')} != policy {lock_format}"
            )
        schema = load_json(template / "databricks_template_schema.json")
        configured_sdk = schema["properties"]["aai_core_version"]["default"]
        if configured_sdk != expected["aai_core"]:
            failures.append(
                f"{name}: schema SDK {configured_sdk} != {expected['aai_core']}"
            )
        if configured_sdk != sdk_version:
            failures.append(
                f"{name}: default SDK {configured_sdk} != repository SDK {sdk_version}"
            )
        pip_source = schema["properties"]["aai_core_pip_source"]["default"]
        if "@v{{.aai_core_version}}" not in pip_source:
            failures.append(
                f"{name}: credential-free SDK source is not tied to the SDK version"
            )
        stamp = load_json(template / "template" / ".aai-template.json.tmpl")
        if stamp.get("template") != name:
            failures.append(f"{name}: provenance stamp has wrong template name")
        if stamp.get("template_version") != expected["version"]:
            failures.append(
                f"{name}: template version {stamp.get('template_version')} "
                f"!= {expected['version']}"
            )

        template_project = load_toml(template / "template" / "pyproject.toml.tmpl")[
            "project"
        ]
        runtime_groups = {"default": list(template_project["dependencies"])}
        runtime_groups.update(
            {
                group: list(requirements)
                for group, requirements in template_project.get(
                    "optional-dependencies", {}
                ).items()
                if group != "dev"
            }
        )
        for group, requirements in runtime_groups.items():
            for package, specifier in requirement_specs(requirements).items():
                if package == "aai-core":
                    continue
                expected_range = policy_ranges.get(package)
                if expected_range is None:
                    failures.append(
                        f"{name}: {group} runtime dependency {package} "
                        "has no policy entry"
                    )
                elif specifier != expected_range:
                    failures.append(
                        f"{name}: {group} {package}{specifier} "
                        f"!= policy range {expected_range}"
                    )

        direct = requirement_names(runtime_groups["default"])
        runtime_lock = template / "template" / "requirements.lock"
        pins = requirement_pins(runtime_lock)
        unpinned = unpinned_requirement_lines(runtime_lock)
        if unpinned:
            failures.append(
                f"{name}: requirements.lock contains non-exact lines {unpinned}"
            )
        expected_lock_inputs = certified_input_requirements(
            runtime_groups["default"],
            policy_pins=policy_pins,
        )
        failures.extend(
            f"{name}: {issue}"
            for issue in lock_source_issues(
                runtime_lock,
                expected_inputs=expected_lock_inputs,
            )
        )
        missing = direct - set(pins)
        if missing:
            failures.append(
                f"{name}: requirements.lock missing direct pins {sorted(missing)}"
            )
        for package in direct:
            version = pins.get(package)
            certified = policy_pins.get(package)
            if version is not None and certified is not None and version != certified:
                failures.append(
                    f"{name}: {package}=={version} != certified {certified}"
                )

        lock_header = (template / "template" / "requirements.lock").read_text(
            encoding="utf-8"
        )
        if "certified universal transitive runtime lock" not in lock_header.lower():
            failures.append(
                f"{name}: requirements.lock must state its transitive-lock semantics"
            )

        recipes_root = template / "template" / "recipes"
        if recipes_root.is_dir():
            for recipe_lock in sorted(recipes_root.glob("*/requirements.lock")):
                recipe = recipe_lock.parent.name
                declared = runtime_groups.get(recipe)
                if declared is None:
                    failures.append(
                        f"{name}: recipe {recipe} has no matching optional "
                        "dependency group"
                    )
                    continue
                declared_names = requirement_names(declared)
                recipe_pins = requirement_pins(recipe_lock)
                recipe_unpinned = unpinned_requirement_lines(recipe_lock)
                if recipe_unpinned:
                    failures.append(
                        f"{name}: recipe {recipe} lock contains non-exact lines "
                        f"{recipe_unpinned}"
                    )
                if not recipe_pins:
                    failures.append(f"{name}: recipe {recipe} has no exact pins")
                    continue
                expected_recipe_inputs = certified_input_requirements(
                    declared,
                    policy_pins=policy_pins,
                )
                failures.extend(
                    f"{name}: recipe {recipe} {issue}"
                    for issue in lock_source_issues(
                        recipe_lock,
                        expected_inputs=expected_recipe_inputs,
                    )
                )
                missing_recipe_pins = declared_names - set(recipe_pins)
                if missing_recipe_pins:
                    failures.append(
                        f"{name}: recipe {recipe} lock missing direct pins "
                        f"{sorted(missing_recipe_pins)}"
                    )
                for package in declared_names:
                    version = recipe_pins.get(package)
                    certified = policy_pins.get(package)
                    if (
                        version is not None
                        and certified is not None
                        and version != certified
                    ):
                        failures.append(
                            f"{name}: recipe {recipe} {package}=={version} "
                            f"!= certified {certified}"
                        )
                recipe_header = recipe_lock.read_text(encoding="utf-8").lower()
                if "certified universal transitive runtime lock" not in recipe_header:
                    failures.append(
                        f"{name}: recipe {recipe} lock must state its "
                        "transitive-lock semantics"
                    )

        bundle = (template / "template" / "databricks.yml.tmpl").read_text(
            encoding="utf-8"
        )
        certified_runtime = compatibility["certified_runtime"]["databricks_runtime"]
        if f"default: {certified_runtime}" not in bundle:
            failures.append(
                f"{name}: Databricks Runtime differs from compatibility manifest"
            )

    versions = load_json(TEMPLATES / "_shared" / "versions.json")["pins"]
    for package, version in versions.items():
        certified = policy_pins.get(normalized_name(package))
        if certified != version:
            failures.append(
                f"shared pin {package}=={version} != policy value {certified}"
            )

    if failures:
        raise ValueError("\n".join(f"- {failure}" for failure in failures))
    return policy, compatibility, toolchain


def validate_wheel(wheel: Path, expected_version: str) -> dict[str, str]:
    if not wheel.is_file():
        raise ValueError(f"wheel does not exist: {wheel}")
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        wheel_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise ValueError("wheel must contain exactly one METADATA and WHEEL file")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_names[0]))

    if normalized_name(str(metadata["Name"])) != "aai-core":
        raise ValueError(f"unexpected wheel project name: {metadata['Name']}")
    if metadata["Version"] != expected_version:
        raise ValueError(
            f"wheel version {metadata['Version']} != expected {expected_version}"
        )
    python_specifiers = {
        part.strip() for part in str(metadata["Requires-Python"]).split(",")
    }
    if python_specifiers != {">=3.11", "<3.13"}:
        raise ValueError(
            "wheel Requires-Python must remain synchronized with compatibility policy"
        )
    if wheel_metadata["Root-Is-Purelib"] != "true":
        raise ValueError("aai-core wheel must remain pure Python")
    tags = wheel_metadata.get_all("Tag", [])
    if "py3-none-any" not in tags:
        raise ValueError(f"wheel is missing py3-none-any tag: {tags}")
    return {
        "filename": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_release_version(version: str, *, require_tag: bool) -> str:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if not re.search(rf"^## {re.escape(version)}(?:\s|$)", changelog, re.MULTILINE):
        raise ValueError(f"CHANGELOG.md has no release section for {version}")
    commit = git_output("rev-parse", "HEAD")
    if require_tag:
        tag_ref = f"refs/tags/v{version}"
        if git_output("cat-file", "-t", tag_ref) != "tag":
            raise ValueError(f"v{version} must be an annotated tag")
        tagged_commit = git_output("rev-parse", "--verify", f"{tag_ref}^{{}}")
        if tagged_commit != commit:
            raise ValueError(
                f"v{version} points to {tagged_commit}, not release commit {commit}"
            )
    return commit


def write_manifest(
    destination: Path,
    *,
    version: str,
    commit: str,
    wheel: dict[str, str],
    compatibility: dict[str, Any],
) -> None:
    identifiers = load_json(ROOT / "platform-identifiers.json")
    manifest = {
        "schema_version": 2,
        "package": "aai-core",
        "version": version,
        "source_commit": commit,
        "sdk_artifact_volume": identifiers["sdk_artifact_volume"],
        "wheel": wheel,
        "compatibility_sha256": hashlib.sha256(
            (ROOT / "compatibility.json").read_bytes()
        ).hexdigest(),
        "dependency_policy_sha256": hashlib.sha256(
            (ROOT / "dependency-policy.toml").read_bytes()
        ).hexdigest(),
        "toolchain_sha256": hashlib.sha256(
            (ROOT / "toolchain.json").read_bytes()
        ).hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(
            (ROOT / "uv.lock").read_bytes()
        ).hexdigest(),
        "templates": {
            name: details["version"]
            for name, details in sorted(compatibility["templates"].items())
        },
    }
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--release-version")
    parser.add_argument("--require-tag", action="store_true")
    parser.add_argument("--write-manifest", type=Path)
    arguments = parser.parse_args(argv)

    try:
        _, compatibility, _ = validate_repository()
        expected_version = compatibility["sdk"]["version"]
        if arguments.release_version and arguments.release_version != expected_version:
            raise ValueError(
                f"requested release {arguments.release_version} "
                f"!= repository SDK {expected_version}"
            )
        if arguments.wheel and arguments.wheel.is_dir():
            candidates = sorted(
                arguments.wheel.glob(f"aai_core-{expected_version}-*.whl")
            )
            if len(candidates) != 1:
                raise ValueError(
                    f"expected one aai-core {expected_version} wheel in "
                    f"{arguments.wheel}, found {candidates}"
                )
            arguments.wheel = candidates[0]
        wheel_details = (
            validate_wheel(arguments.wheel, expected_version)
            if arguments.wheel
            else None
        )
        commit = git_output("rev-parse", "HEAD")
        if arguments.release_version:
            commit = validate_release_version(
                arguments.release_version, require_tag=arguments.require_tag
            )
        if arguments.write_manifest:
            if not arguments.release_version or not wheel_details:
                raise ValueError(
                    "--write-manifest requires --release-version and --wheel"
                )
            write_manifest(
                arguments.write_manifest,
                version=arguments.release_version,
                commit=commit,
                wheel=wheel_details,
                compatibility=compatibility,
            )
    except (KeyError, OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"release validation failed:\n{error}", file=sys.stderr)
        return 1

    print("SDK, template, dependency, and release invariants are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
