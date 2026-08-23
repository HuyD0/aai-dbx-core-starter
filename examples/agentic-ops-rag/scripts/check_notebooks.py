"""Verify and optionally execute every credential-free workshop notebook."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT / "notebooks"
EXPECTED = (
    "00_environment_and_stack_map.ipynb",
    "01_routing_filters_and_action_boundaries.ipynb",
    "02_chunking_embeddings_and_index_release.ipynb",
    "03_hybrid_retrieval_and_reranking.ipynb",
    "04_mlflow_tracing_guardrails_and_evaluation.ipynb",
    "05_capstone_release_decision.ipynb",
    "06_confidence_intervals_for_release_gates.ipynb",
)


def verify_notebook(path: Path, *, execute: bool) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not notebook.get("cells"):
        raise ValueError(f"{path.name}: expected a non-empty nbformat 4 notebook")
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    if not all(cell_ids) or len(cell_ids) != len(set(cell_ids)):
        raise ValueError(f"{path.name}: cell IDs must be present and unique")

    namespace: dict[str, object] = {"__name__": f"workshop_{path.stem}"}
    for position, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise ValueError(f"{path.name}: tracked notebooks must be output-free")
        source = "".join(cell.get("source", []))
        compiled = compile(
            source,
            f"{path.name}:code-cell-{position}",
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        if execute:
            result = eval(compiled, namespace)
            if result is not None:
                raise RuntimeError(
                    f"{path.name}: top-level await is not supported by this checker"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rendered = subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "render_notebooks.py"), "--check"],
        cwd=PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )
    if rendered.returncode:
        print(rendered.stdout, file=sys.stderr)
        print(rendered.stderr, file=sys.stderr)
        return rendered.returncode

    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    if tuple(path.name for path in paths) != EXPECTED:
        raise SystemExit(
            "Expected the complete ordered 00-06 course; found "
            + ", ".join(path.name for path in paths)
        )
    original = Path.cwd()
    try:
        os.chdir(PROJECT)
        for path in paths:
            verify_notebook(path, execute=args.execute)
            print(f"[PASS] {'executed' if args.execute else 'verified'} {path.name}")
    finally:
        os.chdir(original)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
