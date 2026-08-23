"""Execute all lessons sequentially, then rerun every lesson in the same state."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from notebook_content import LESSONS

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT / "notebooks"
KERNEL_NAME = "aai-fine-tuning"
# Generous headroom over the sibling course's 300s: importing torch inside a
# fresh kernel costs several seconds before a lesson runs its first cell.
TIMEOUT_SECONDS = 600


def _tail(value: str, lines: int = 100) -> str:
    selected = value.strip().splitlines()[-lines:]
    return "\n".join(selected) if selected else "(no diagnostic output)"


def execute(
    notebook: Path,
    output: Path,
    environment: dict[str, str],
    *,
    phase: str,
    position: int,
    total: int,
) -> None:
    print(f"[{phase} {position:02d}/{total:02d}] {notebook.name}", flush=True)
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--execute",
        "--to",
        "notebook",
        f"--ExecutePreprocessor.timeout={TIMEOUT_SECONDS}",
        f"--ExecutePreprocessor.kernel_name={KERNEL_NAME}",
        f"--output={phase}-{notebook.name}",
        f"--output-dir={output}",
        str(notebook),
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"[PASS] {phase}: {notebook.name}", flush=True)
        return

    diagnostics = "\n".join(
        part for part in (result.stdout.strip(), result.stderr.strip()) if part
    )
    print(file=sys.stderr)
    print("Notebook execution failed", file=sys.stderr)
    print(f"  phase:    {phase}", file=sys.stderr)
    print(f"  notebook: {notebook}", file=sys.stderr)
    print(f"  exit:     {result.returncode}", file=sys.stderr)
    print("  final diagnostic output:", file=sys.stderr)
    print(_tail(diagnostics), file=sys.stderr)
    raise SystemExit(result.returncode or 1)


def _verify_rendered_sources() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/render_notebooks.py", "--check"],
        cwd=PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        diagnostics = "\n".join(
            part for part in (result.stdout.strip(), result.stderr.strip()) if part
        )
        print("Notebook source verification failed:", file=sys.stderr)
        print(_tail(diagnostics), file=sys.stderr)
        raise SystemExit(result.returncode)


def _expected_lessons() -> list[str]:
    """The lesson list from the single source of truth, sanity-checked."""
    names = sorted(LESSONS)
    prefixes = [name[:2] for name in names]
    if prefixes != [f"{number:02d}" for number in range(len(names))]:
        raise SystemExit(
            f"LESSONS must form one contiguous numbered course; found {names}"
        )
    return names


def main() -> int:
    _verify_rendered_sources()
    expected = _expected_lessons()
    found = sorted(path.name for path in NOTEBOOKS.glob("*.ipynb"))
    if found != expected:
        raise SystemExit(
            "The notebooks directory must contain exactly the rendered lessons; "
            f"expected {expected}, found {found}"
        )
    notebooks = [NOTEBOOKS / name for name in expected]

    with tempfile.TemporaryDirectory(prefix="aai-finetune-notebooks-") as raw:
        temporary = Path(raw)
        output = temporary / "executed"
        output.mkdir()
        environment = os.environ.copy()
        environment["AAI_FINETUNE_PROJECT_ROOT"] = str(temporary / "course-v1-state")
        environment["HF_HUB_OFFLINE"] = "1"
        environment["HF_HOME"] = str(temporary / "course-v1-state" / "hf")
        environment["JUPYTER_PATH"] = str(PROJECT / ".venv" / "share" / "jupyter")

        total = len(notebooks)
        for position, notebook in enumerate(notebooks, start=1):
            execute(
                notebook,
                output,
                environment,
                phase="sequential",
                position=position,
                total=total,
            )
        for position, notebook in enumerate(notebooks, start=1):
            execute(
                notebook,
                output,
                environment,
                phase="rerun",
                position=position,
                total=total,
            )

    print(f"All {total} lessons executed in order and every lesson reran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
