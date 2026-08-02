"""Give beginner-friendly diagnostics for the local classification course."""

from __future__ import annotations

import json
import os
import platform
import sys
from importlib import import_module, metadata
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_ENVIRONMENT = PROJECT / ".venv"
EXPECTED_PYTHON = EXPECTED_ENVIRONMENT / "bin" / "python"
EXPECTED_KERNEL = "aai-local-classification"
EXPECTED_DISPLAY_NAME = "AAI Local Classification"
EXPECTED_COURSE_ROOT = (PROJECT / ".aai" / "course-v2").resolve()
KERNEL_SPEC = (
    PROJECT
    / ".venv"
    / "share"
    / "jupyter"
    / "kernels"
    / EXPECTED_KERNEL
    / "kernel.json"
)


class Diagnostics:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def pass_check(self, label: str, detail: str) -> None:
        print(f"[PASS] {label}: {detail}")

    def fail_check(self, label: str, detail: str, guidance: str) -> None:
        print(f"[FAIL] {label}: {detail}")
        print(f"       Fix: {guidance}")
        self.failures.append(label)


def _check_python(diagnostics: Diagnostics) -> None:
    version = sys.version_info
    if (version.major, version.minor) in {(3, 11), (3, 12)}:
        diagnostics.pass_check(
            "Python",
            f"{platform.python_version()} is supported",
        )
    else:
        diagnostics.fail_check(
            "Python",
            f"{platform.python_version()} is outside the supported 3.11/3.12 range",
            "install the course again with `make install`",
        )

    executable = Path(sys.executable)
    environment = Path(sys.prefix).resolve()
    if environment == EXPECTED_ENVIRONMENT.resolve():
        diagnostics.pass_check("Environment", f"using {executable}")
    else:
        diagnostics.fail_check(
            "Environment",
            f"doctor is running with {executable}",
            f"run `{EXPECTED_PYTHON} scripts/doctor.py` or `make doctor`",
        )


def _check_dependencies(diagnostics: Diagnostics) -> None:
    packages = (
        ("aai-local-classification", "aai_local_classification.workflow"),
        ("ipykernel", "ipykernel"),
        ("jupyterlab", "jupyterlab"),
        ("mlflow", "mlflow"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
    )
    broken: list[str] = []
    versions: list[str] = []
    for package, module in packages:
        try:
            import_module(module)
            versions.append(f"{package} {metadata.version(package)}")
        except Exception as error:
            broken.append(f"{package} ({type(error).__name__}: {error})")
    if broken:
        diagnostics.fail_check(
            "Dependencies",
            "could not import " + ", ".join(broken),
            "run `make install`, then rerun `make doctor`",
        )
    else:
        diagnostics.pass_check("Dependencies", ", ".join(versions))


def _check_state_root(diagnostics: Diagnostics) -> None:
    configured = Path(
        os.environ.get("AAI_CLASSIFICATION_PROJECT_ROOT", str(EXPECTED_COURSE_ROOT))
    ).resolve()
    if configured != EXPECTED_COURSE_ROOT:
        diagnostics.fail_check(
            "Course state",
            f"configured as {configured}",
            "unset AAI_CLASSIFICATION_PROJECT_ROOT or set it to "
            f"{EXPECTED_COURSE_ROOT}",
        )
    elif not os.access(PROJECT, os.W_OK):
        diagnostics.fail_check(
            "Course state",
            f"{PROJECT} is not writable",
            "move the repository to a writable directory",
        )
    else:
        diagnostics.pass_check(
            "Course state",
            f"isolated, ignored state will use {EXPECTED_COURSE_ROOT}",
        )


def _check_kernel(diagnostics: Diagnostics) -> None:
    if not KERNEL_SPEC.is_file():
        diagnostics.fail_check(
            "Jupyter kernel",
            f"{EXPECTED_KERNEL} is not registered in the course environment",
            "run `make notebook-kernel`, then rerun `make doctor`",
        )
        return
    try:
        spec = json.loads(KERNEL_SPEC.read_text(encoding="utf-8"))
        kernel_python = Path(spec["argv"][0])
        display_name = spec["display_name"]
    except (KeyError, IndexError, json.JSONDecodeError, OSError) as error:
        diagnostics.fail_check(
            "Jupyter kernel",
            f"the kernel specification is invalid ({error})",
            "run `make notebook-kernel` to replace it",
        )
        return
    if kernel_python != EXPECTED_PYTHON or display_name != EXPECTED_DISPLAY_NAME:
        diagnostics.fail_check(
            "Jupyter kernel",
            f"it uses {kernel_python} and is named {display_name!r}",
            "run `make notebook-kernel` to register the course's exact environment",
        )
        return
    diagnostics.pass_check(
        "Jupyter kernel",
        f"{EXPECTED_DISPLAY_NAME} uses {EXPECTED_PYTHON}",
    )


def _check_notebooks(diagnostics: Diagnostics) -> None:
    notebooks = sorted((PROJECT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    prefixes = [path.name[:2] for path in notebooks]
    expected = [f"{number:02d}" for number in range(10)]
    if prefixes == expected:
        diagnostics.pass_check("Lessons", "found the complete 00-09 notebook course")
    else:
        diagnostics.fail_check(
            "Lessons",
            f"found lesson prefixes {prefixes}",
            "restore the tracked notebooks, then rerun `make doctor`",
        )


def main() -> int:
    print("Local classification course doctor")
    print("==================================")
    diagnostics = Diagnostics()
    _check_python(diagnostics)
    _check_dependencies(diagnostics)
    _check_state_root(diagnostics)
    _check_kernel(diagnostics)
    _check_notebooks(diagnostics)
    print()
    if diagnostics.failures:
        print(
            "Setup needs attention: "
            + ", ".join(diagnostics.failures)
            + ". Apply the fixes above and rerun `make doctor`."
        )
        return 1
    print("All checks passed. In JupyterLab, the kernel at the top right should say")
    print(f'"{EXPECTED_DISPLAY_NAME}". If it does not, choose Kernel > Change Kernel.')
    print("You are ready: run `make notebook` and start with lesson 00.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
