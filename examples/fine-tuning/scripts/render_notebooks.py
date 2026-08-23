"""Render the deterministic fine-tuning notebook course."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import black
import nbformat
from notebook_content import LESSONS

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT / "notebooks"


def _cell_id(notebook: str, position: int, text: str) -> str:
    digest = hashlib.sha256(f"{notebook}:{position}:{text}".encode()).hexdigest()[:12]
    return f"aai-{position:02d}-{digest}"


def markdown(notebook: str, position: int, text: str):
    return nbformat.v4.new_markdown_cell(
        source=text.strip() + "\n",
        id=_cell_id(notebook, position, text),
    )


def code(notebook: str, position: int, text: str):
    source = text.strip()
    try:
        formatted = black.format_cell(
            source,
            fast=True,
            mode=black.Mode(line_length=88),
        )
    except black.NothingChanged:
        formatted = source
    cell = nbformat.v4.new_code_cell(
        source=formatted.rstrip() + "\n",
        execution_count=None,
        outputs=[],
        id=_cell_id(notebook, position, text),
    )
    if source.lstrip().startswith("# Preflight"):
        cell.metadata["tags"] = ["preflight"]
    elif source.lstrip().startswith("# Reference solution"):
        cell.metadata["tags"] = ["solution"]
    return cell


def render_lesson(filename: str, cells: list[tuple[str, str]]) -> str:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        (
            markdown(filename, index, text)
            if kind == "markdown"
            else code(filename, index, text)
        )
        for index, (kind, text) in enumerate(cells)
    ]
    notebook.metadata = {
        "kernelspec": {
            "display_name": "AAI Fine-Tuning",
            "language": "python",
            "name": "aai-fine-tuning",
        },
        "language_info": {"name": "python"},
        "aai_course": {
            "schema_version": 1,
            "network_required_after_install": False,
            "cloud_credentials_required": False,
            "audience": "fine-tuning beginner",
        },
    }
    return nbformat.writes(notebook, version=4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for filename, cells in LESSONS.items():
        rendered = render_lesson(filename, cells)
        path = NOTEBOOKS / filename
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                failures.append(filename)
        else:
            path.write_text(rendered, encoding="utf-8")
    if failures:
        raise SystemExit(f"Notebook sources are stale: {', '.join(failures)}")
    verb = "Verified" if args.check else "Rendered"
    print(f"{verb} {len(LESSONS)} fine-tuning lessons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
