"""Execute every tracked lesson in order against isolated temporary state."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT / "notebooks"


def main() -> int:
    subprocess.run(
        [sys.executable, "scripts/render_notebooks.py", "--check"],
        cwd=PROJECT,
        check=True,
    )
    notebooks = sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))
    if len(notebooks) != 10:
        raise SystemExit(f"Expected 10 lessons, found {len(notebooks)}")
    with tempfile.TemporaryDirectory(prefix="aai-classification-notebooks-") as raw:
        temporary = Path(raw)
        output = temporary / "executed"
        output.mkdir()
        environment = os.environ.copy()
        environment["AAI_CLASSIFICATION_PROJECT_ROOT"] = str(temporary / "state")
        environment["JUPYTER_PATH"] = str(PROJECT / ".venv" / "share" / "jupyter")
        for notebook in notebooks:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--execute",
                    "--to",
                    "notebook",
                    "--ExecutePreprocessor.timeout=240",
                    "--ExecutePreprocessor.kernel_name=aai-local-classification",
                    f"--output-dir={output}",
                    str(notebook),
                ],
                cwd=PROJECT,
                env=environment,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            print(f"Executed {notebook.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
