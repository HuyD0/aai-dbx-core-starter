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
from collections.abc import Mapping
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
SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.-]+)?$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def normalized_name(value: str) -> str:
    base = value.split("[", 1)[0]
    return NAME_NORMALIZER.sub("-", base).lower()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _sdk_content_sha256(
    project_document: Mapping[str, Any],
    sources: list[tuple[str, bytes]],
) -> str:
    project = project_document["project"]
    material = {
        "build-system": project_document["build-system"],
        "project": {
            key: project[key]
            for key in (
                "name",
                "version",
                "requires-python",
                "dependencies",
                "optional-dependencies",
                "scripts",
            )
        },
        "wheel": project_document["tool"]["hatch"]["build"]["targets"]["wheel"],
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\0")
    for relative, content in sorted(sources):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def sdk_content_sha256(root: Path = ROOT) -> str:
    """Hash the wheel-bearing SDK source and package metadata."""

    project_document = load_toml(root / "pyproject.toml")
    source_root = root / "src" / "aai_core"
    sources: list[tuple[str, bytes]] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        sources.append((path.relative_to(root).as_posix(), path.read_bytes()))
    return _sdk_content_sha256(project_document, sources)


def sdk_content_sha256_at_commit(commit: str) -> str | None:
    """Hash a locally available commit, or return None for a shallow checkout."""

    try:
        project_text = subprocess.run(
            ["git", "show", f"{commit}:pyproject.toml"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        listing = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit, "--", "src/aai_core"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        sources = [
            (
                relative,
                subprocess.run(
                    ["git", "show", f"{commit}:{relative}"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                ).stdout,
            )
            for relative in listing
            if "__pycache__" not in Path(relative).parts
            and Path(relative).suffix != ".pyc"
        ]
    except subprocess.CalledProcessError:
        return None
    return _sdk_content_sha256(tomllib.loads(project_text), sources)


def sdk_version_at_commit(commit: str) -> str | None:
    """Read the package version from a locally available commit."""

    try:
        project_text = subprocess.run(
            ["git", "show", f"{commit}:pyproject.toml"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    return str(tomllib.loads(project_text)["project"]["version"])


def _candidate_source_issues(source: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if source.get("kind") != "git-commit":
        failures.append("a release-candidate SDK source must be a git-commit")
    source_ref = source.get("ref")
    if not isinstance(source_ref, str) or GIT_COMMIT.fullmatch(source_ref) is None:
        failures.append("release-candidate SDK ref must be a full commit SHA")
    if source.get("annotated") is not None:
        failures.append("release-candidate SDK source cannot claim an annotated tag")
    return failures


def _published_source_issues(source: Mapping[str, Any], version: object) -> list[str]:
    failures: list[str] = []
    if source.get("kind") != "git-tag":
        failures.append("a published SDK source must be a git-tag")
    if isinstance(version, str) and source.get("ref") != f"v{version}":
        failures.append("published SDK ref must be the exact v<version> tag")
    if source.get("annotated") is not True:
        failures.append("published SDK source must declare an annotated tag")
    return failures


def generated_sdk_default_issues(  # noqa: C901 - linear metadata assertions
    compatibility: Mapping[str, Any],
    *,
    current_sdk_version: str,
    pinned_content_sha256: str | None,
    pinned_sdk_version: str | None,
) -> list[str]:
    """Validate the offline generated-project SDK release-channel contract."""

    failures: list[str] = []
    if compatibility.get("schema_version") != 2:
        failures.append("compatibility schema_version must be 2")
    sdk = compatibility.get("sdk")
    if isinstance(sdk, Mapping) and sdk.get("development_status") != "unreleased":
        failures.append("the checkout SDK must be marked as unreleased development")
    generated = (
        sdk.get("generated_project_default") if isinstance(sdk, Mapping) else None
    )
    if not isinstance(generated, Mapping):
        return failures + ["sdk.generated_project_default must be a mapping"]

    version = generated.get("version")
    status = generated.get("status")
    source = generated.get("source")
    if not isinstance(version, str) or SEMANTIC_VERSION.fullmatch(version) is None:
        failures.append("generated-project SDK version must be semantic")
    if status not in {"release-candidate", "published"}:
        failures.append(
            "generated-project SDK status must be release-candidate or published"
        )
    if not isinstance(source, Mapping):
        return failures + ["generated-project SDK source must be a mapping"]

    content_digest = source.get("content_sha256")
    if not isinstance(content_digest, str) or SHA256.fullmatch(content_digest) is None:
        failures.append("generated-project SDK content_sha256 must be a SHA-256")
    if status == "release-candidate":
        failures.extend(_candidate_source_issues(source))
    elif status == "published":
        failures.extend(_published_source_issues(source, version))

    if (
        isinstance(content_digest, str)
        and pinned_content_sha256 is not None
        and content_digest != pinned_content_sha256
    ):
        failures.append(
            "generated-project SDK content digest does not describe its pinned commit"
        )
    if isinstance(version, str) and pinned_sdk_version not in {None, version}:
        failures.append(
            "generated-project SDK version does not match its pinned commit metadata"
        )
    if (
        isinstance(version, str)
        and version == current_sdk_version
        and status == "published"
    ):
        failures.append(
            "the checkout cannot mark its own SDK version both unreleased and published"
        )
    return failures


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


def workflow_pip_requirements(path: Path) -> dict[str, str]:
    """Requirement specifiers the dependency canary installs, as {name: range}.

    The canary pins the *bounds* it proves, so this is a fourth copy of the
    supported ranges in dependency-policy.toml. Nothing used to compare them:
    moving a bound in the policy left the canary proving the old one, green.
    """

    pattern = re.compile(
        r'^\s*"(?P<name>[A-Za-z0-9._-]+)(?:\[[^\]]*\])?'
        r'(?P<spec>(?:[<>=!~]=?[^,"]+)(?:,[<>=!~]=?[^,"]+)*)"\s*\\?\s*$'
    )
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            found[match.group("name").lower()] = match.group("spec").strip()
    return found


def workflow_matrix_values(path: Path, key: str) -> set[str]:
    """Read one simple GitHub Actions matrix sequence without a YAML dependency."""

    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^        {re.escape(key)}:\s*\n(?P<items>(?:          - .+\n)+)",
        text,
        re.MULTILINE,
    )
    if match is None:
        return set()
    return {
        line.removeprefix("- ").strip().strip("\"'")
        for line in (item.strip() for item in match.group("items").splitlines())
    }


def validate_repository() -> (  # noqa: C901 - linear, independent release assertions
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
):
    failures: list[str] = []
    project = load_toml(ROOT / "pyproject.toml")["project"]
    policy = load_toml(ROOT / "dependency-policy.toml")
    compatibility = load_json(ROOT / "compatibility.json")
    toolchain = load_json(ROOT / "toolchain.json")
    sdk_version = str(project["version"])
    generated_default = compatibility["sdk"]["generated_project_default"]
    generated_sdk_version = str(generated_default["version"])
    generated_source_ref = str(generated_default["source"]["ref"])

    if sdk_version != compatibility["sdk"]["version"]:
        failures.append(
            "pyproject SDK version "
            f"{sdk_version} != compatibility SDK {compatibility['sdk']['version']}"
        )
    failures.extend(
        generated_sdk_default_issues(
            compatibility,
            current_sdk_version=sdk_version,
            pinned_content_sha256=sdk_content_sha256_at_commit(generated_source_ref),
            pinned_sdk_version=sdk_version_at_commit(generated_source_ref),
        )
    )
    if sorted(compatibility["sdk"]["python"]) != sorted(policy["python"]["supported"]):
        failures.append("Python support differs between compatibility and policy")
    if sorted(compatibility["sdk"]["python"]) != sorted(
        toolchain["python"]["supported"]
    ):
        failures.append("Python support differs between compatibility and toolchain")

    canary = compatibility.get("release_acceptance", {}).get("dependency_canary")
    if not isinstance(canary, Mapping) or canary.get("required") is not True:
        failures.append("release acceptance must require the dependency canary")
    else:
        green_runs = canary.get("minimum_green_runs")
        if (
            not isinstance(green_runs, int)
            or isinstance(green_runs, bool)
            or green_runs < 1
        ):
            failures.append("dependency canary minimum_green_runs must be at least 1")
        if sorted(canary.get("python", [])) != sorted(policy["python"]["supported"]):
            failures.append("dependency canary Python matrix differs from policy")
        expected_resolutions = {
            policy["resolution"]["minimum_resolution"],
            policy["resolution"]["latest_resolution"],
        }
        if set(canary.get("resolutions", [])) != expected_resolutions:
            failures.append("dependency canary resolutions differ from policy")
        canary_workflow = ROOT / ".github" / "workflows" / "dependency-canary.yml"
        if workflow_matrix_values(canary_workflow, "python") != set(
            canary.get("python", [])
        ):
            failures.append("dependency canary workflow Python matrix differs")
        if workflow_matrix_values(canary_workflow, "resolution") != set(
            canary.get("resolutions", [])
        ):
            failures.append("dependency canary workflow resolutions differ")
        installed = workflow_pip_requirements(canary_workflow)
        if not installed:
            failures.append("dependency canary installs no explicit supported bounds")
        for name, specifier in sorted(installed.items()):
            supported = policy["packages"].get(name, {}).get("supported")
            if supported is None:
                failures.append(
                    f"dependency canary installs {name}, which dependency-policy.toml "
                    "does not declare"
                )
            elif specifier != supported:
                failures.append(
                    f"dependency canary installs {name}{specifier} but "
                    f"dependency-policy.toml supports {supported}"
                )

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
        if configured_sdk != generated_sdk_version:
            failures.append(
                f"{name}: default SDK {configured_sdk} != generated-project SDK "
                f"{generated_sdk_version}"
            )
        configured_source_ref = schema["properties"]["aai_core_source_ref"]["default"]
        if configured_source_ref != generated_source_ref:
            failures.append(
                f"{name}: SDK source ref {configured_source_ref} != release metadata "
                f"{generated_source_ref}"
            )
        pip_source = schema["properties"]["aai_core_pip_source"]["default"]
        if (
            pip_source.startswith("git+")
            and "@{{.aai_core_source_ref}}" not in pip_source
        ):
            failures.append(
                f"{name}: git SDK source is not tied to the immutable source ref"
            )
        if "@v{{.aai_core_version}}" in pip_source:
            failures.append(
                f"{name}: SDK source still assumes the version tag already exists"
            )
        stamp = load_json(template / "template" / ".aai-template.json.tmpl")
        if stamp.get("template") != name:
            failures.append(f"{name}: provenance stamp has wrong template name")
        if stamp.get("template_version") != expected["version"]:
            failures.append(
                f"{name}: template version {stamp.get('template_version')} "
                f"!= {expected['version']}"
            )
        if (
            stamp.get("generated_with", {}).get("aai_core_source_ref")
            != "{{.aai_core_source_ref}}"
        ):
            failures.append(f"{name}: provenance stamp omits the SDK source ref")

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
    parser.add_argument(
        "--print-sdk-content-sha256",
        metavar="COMMIT",
        help="Print the SDK content digest for a locally available commit and exit.",
    )
    arguments = parser.parse_args(argv)

    try:
        if arguments.print_sdk_content_sha256:
            digest = sdk_content_sha256_at_commit(arguments.print_sdk_content_sha256)
            if digest is None:
                raise ValueError(
                    "SDK source commit is unavailable locally; fetch/review it before "
                    "recording release metadata"
                )
            print(digest)
            return 0
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
