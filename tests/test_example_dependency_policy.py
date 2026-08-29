"""Standalone example courses stay inside dependency-policy supported ranges.

The platform's dependency governance (dependency-policy.toml, the canary, the
certified lock) covers the SDK and templates, and the deepagents accelerator
cross-checks its ``%pip`` pins — but the standalone example courses carry
their own pyproject/uv.lock channels. Without this check a course can pin a
governed package outside the supported range and hold it there silently,
which is how the fine-tuning course came to pin mlflow 3.14.0 after the
policy had certified 3.15.1.

A course that deliberately holds a governed package back gets an entry in
EXEMPT_PINS naming the exact version and the reason. The exemption fails the
suite the moment it goes stale — the course upgrades, the pin drifts further,
or the policy range moves to include it — so an exemption cannot outlive its
rationale.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
#: Derived, not enumerated: a new course with its own pyproject.toml is
#: governed the day it lands, the same way fork_review_drift derives its
#: checked set from opt_out.
COURSES = tuple(
    sorted(path.parent.name for path in (ROOT / "examples").glob("*/pyproject.toml"))
)
_NAME_NORMALIZER = re.compile(r"[-_.]+")

#: (course, normalized package) -> (exact pinned version, reason).
EXEMPT_PINS: dict[tuple[str, str], tuple[str, str]] = {
    ("fine-tuning", "mlflow"): (
        "3.14.0",
        "the course landed on 3.14.0 before the policy certified 3.15.1; "
        "upgrading requires regenerating the course lock and re-verifying "
        "every lesson against 3.15.x, tracked as course follow-up work",
    ),
}


def _normalized(name: str) -> str:
    return _NAME_NORMALIZER.sub("-", name).lower()


def _supported_ranges() -> dict[str, SpecifierSet]:
    policy = tomllib.loads(
        (ROOT / "dependency-policy.toml").read_text(encoding="utf-8")
    )["packages"]
    return {
        _normalized(name): SpecifierSet(config["supported"])
        for name, config in policy.items()
    }


def _course_pins(course: str) -> list[tuple[str, str, Version]]:
    """(dependency group, normalized package, exact pin) for every `==` pin."""

    project = tomllib.loads(
        (ROOT / "examples" / course / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    groups: dict[str, list[str]] = {"default": list(project.get("dependencies", []))}
    groups.update(project.get("optional-dependencies", {}))
    pins = []
    for group, requirements in groups.items():
        for raw in requirements:
            requirement = Requirement(raw)
            for specifier in requirement.specifier:
                if specifier.operator == "==":
                    pins.append(
                        (
                            group,
                            _normalized(requirement.name),
                            Version(specifier.version),
                        )
                    )
    return pins


def test_course_pins_of_governed_packages_stay_inside_supported_ranges():
    ranges = _supported_ranges()
    violations = []
    for course in COURSES:
        for group, package, pinned in _course_pins(course):
            supported = ranges.get(package)
            if supported is None or (course, package) in EXEMPT_PINS:
                continue
            if pinned not in supported:
                violations.append(
                    f"{course} [{group}] pins {package}=={pinned} outside the "
                    f"dependency-policy supported range {supported}; upgrade "
                    "the course or add an EXEMPT_PINS entry with the reason"
                )
    assert not violations, "\n".join(violations)


def test_exemptions_match_the_course_and_stay_justified():
    ranges = _supported_ranges()
    for (course, package), (version, reason) in EXEMPT_PINS.items():
        assert reason.strip(), f"exemption for {course}/{package} needs a reason"
        assert course in COURSES, (
            f"exemption names course {course!r}, which has no pyproject.toml "
            "under examples/; remove the EXEMPT_PINS entry"
        )
        supported = ranges.get(package)
        assert supported is not None, (
            f"{package} is no longer governed by dependency-policy.toml; the "
            f"{course} exemption is stale — remove it"
        )
        pinned = {str(pin) for _, name, pin in _course_pins(course) if name == package}
        assert pinned == {version}, (
            f"{course} pins {package} to {sorted(pinned)}, not the exempted "
            f"{version}; update or remove the EXEMPT_PINS entry"
        )
        assert Version(version) not in supported, (
            f"{course}/{package}=={version} is inside the supported range "
            f"{supported}; the exemption is stale — remove it"
        )
