"""Course-specific, network-free environment diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import subprocess
import sys
from importlib import import_module, metadata
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parents[1]


def _load_setup():
    path = PROJECT / "notebook_setup.py"
    spec = importlib.util.spec_from_file_location("agentic_ops_rag_doctor_setup", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load notebook_setup.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Diagnostics:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def passed(self, label: str, detail: str) -> None:
        print(f"[PASS] {label}: {detail}")

    def info(self, label: str, detail: str) -> None:
        print(f"[INFO] {label}: {detail}")

    def failed(self, label: str, detail: str, fix: str) -> None:
        print(f"[FAIL] {label}: {detail}")
        print(f"       Fix: {fix}")
        self.failures.append(label)


def _check_python(diagnostics: Diagnostics) -> None:
    version = sys.version_info[:2]
    if version in {(3, 11), (3, 12)}:
        diagnostics.passed("Python", f"{platform.python_version()} is supported")
    else:
        diagnostics.failed(
            "Python",
            f"{platform.python_version()} is outside 3.11/3.12",
            "run `make ops-rag-install` with a supported repository toolchain",
        )
    expected = (REPOSITORY / ".venv").resolve()
    if Path(sys.prefix).resolve() == expected:
        diagnostics.passed("Environment", f"using {sys.executable}")
    else:
        diagnostics.failed(
            "Environment",
            f"using {sys.executable}",
            "run `make ops-rag-doctor` from the repository root",
        )


def _check_dependencies(diagnostics: Diagnostics) -> None:
    packages = (
        ("aai-core", "aai_core"),
        ("mlflow", "mlflow"),
        ("jupyterlab", "jupyterlab"),
        ("azure-identity", "azure.identity"),
        ("azure-search-documents", "azure.search.documents"),
        ("databricks-sdk", "databricks.sdk"),
        ("openai", "openai"),
    )
    missing: list[str] = []
    installed: list[str] = []
    for package, module in packages:
        try:
            import_module(module)
            installed.append(f"{package} {metadata.version(package)}")
        except Exception as error:
            missing.append(f"{package} ({type(error).__name__})")
    if missing:
        diagnostics.failed(
            "Dependencies",
            "missing " + ", ".join(missing),
            "run `make ops-rag-install`",
        )
    else:
        diagnostics.passed("Dependencies", ", ".join(installed))


def _check_course(diagnostics: Diagnostics, *, connected: bool) -> None:
    setup = _load_setup()
    try:
        session = setup.prepare_notebook_environment(PROJECT)
        pipeline = session.offline_pipeline()
    except Exception as error:
        diagnostics.failed(
            "Course data and config",
            f"{type(error).__name__}: {error}",
            "restore the tracked config/data files, then rerun the doctor",
        )
        return
    diagnostics.passed(
        "Course data and config",
        f"loaded {len(pipeline.retriever.documents)} synthetic documents",
    )
    if session.connected_ready:
        diagnostics.passed("Connected config", "all selected identifiers are filled")
    elif connected:
        diagnostics.failed(
            "Connected config",
            "the selected configuration still contains placeholders",
            "copy an example to config/aai-platform.yml and fill approved resources",
        )
    else:
        diagnostics.info(
            "Connected config",
            "placeholders remain; this is expected for the offline course",
        )


def _check_notebooks(diagnostics: Diagnostics) -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "check_notebooks.py")],
        cwd=PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        diagnostics.failed(
            "Notebooks",
            (result.stderr or result.stdout).strip(),
            "rerender with `make ops-rag-render`, then rerun the doctor",
        )
    else:
        diagnostics.passed("Notebooks", "six clean generated lessons verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connected",
        action="store_true",
        help="Require a filled connected config; still make no network request.",
    )
    args = parser.parse_args()
    print("Agentic operations RAG workshop doctor")
    print("=======================================")
    diagnostics = Diagnostics()
    _check_python(diagnostics)
    _check_dependencies(diagnostics)
    _check_course(diagnostics, connected=args.connected)
    _check_notebooks(diagnostics)
    if diagnostics.failures:
        print("\nSetup needs attention: " + ", ".join(diagnostics.failures))
        return 1
    print("\nAll required checks passed. No cloud call was made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
