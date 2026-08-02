"""Execute all lessons sequentially, then rerun every lesson in the same state."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT / "notebooks"
KERNEL_NAME = "aai-local-classification"
TIMEOUT_SECONDS = 300


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


def main() -> int:
    _verify_rendered_sources()
    notebooks = sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))
    if [path.name[:2] for path in notebooks] != [
        f"{number:02d}" for number in range(10)
    ]:
        raise SystemExit(
            "Expected the complete numbered 00-09 course; "
            f"found {[path.name for path in notebooks]}"
        )

    with tempfile.TemporaryDirectory(prefix="aai-classification-notebooks-") as raw:
        temporary = Path(raw)
        output = temporary / "executed"
        output.mkdir()
        environment = os.environ.copy()
        environment["AAI_CLASSIFICATION_PROJECT_ROOT"] = str(
            temporary / "course-v2-state"
        )
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

    print("All 10 lessons executed in order and every lesson reran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
