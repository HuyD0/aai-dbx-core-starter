"""Render the clean, deterministic agentic operations RAG notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from notebook_content import LESSONS

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT / "notebooks"
EXPECTED_FILENAMES = (
    "00_environment_and_stack_map.ipynb",
    "01_routing_filters_and_action_boundaries.ipynb",
    "02_chunking_embeddings_and_index_release.ipynb",
    "03_hybrid_retrieval_and_reranking.ipynb",
    "04_mlflow_tracing_guardrails_and_evaluation.ipynb",
    "05_capstone_release_decision.ipynb",
)


def _cell_id(notebook: str, position: int, source: str) -> str:
    digest = hashlib.sha256(f"{notebook}:{position}:{source}".encode()).hexdigest()[:12]
    return f"aai-ops-rag-{position:02d}-{digest}"


def _cell(notebook: str, position: int, kind: str, source: str) -> dict[str, object]:
    normalized = source.strip() + "\n"
    metadata: dict[str, object] = {}
    if kind == "code":
        stripped = normalized.lstrip()
        if stripped.startswith("# Notebook preflight"):
            metadata["tags"] = ["preflight"]
        elif stripped.startswith("# YOUR TURN"):
            metadata["tags"] = ["exercise"]
        elif stripped.startswith("# CHECK YOUR WORK"):
            metadata["tags"] = ["check"]
        elif stripped.startswith("# Reference solution"):
            metadata["tags"] = ["solution"]
        elif (
            "RUN_CONNECTED = False" in normalized or "RUN_MLFLOW = False" in normalized
        ):
            metadata["tags"] = ["connected"]
        return {
            "cell_type": "code",
            "execution_count": None,
            "id": _cell_id(notebook, position, normalized),
            "metadata": metadata,
            "outputs": [],
            "source": normalized.splitlines(keepends=True),
        }
    return {
        "cell_type": "markdown",
        "id": _cell_id(notebook, position, normalized),
        "metadata": metadata,
        "source": normalized.splitlines(keepends=True),
    }


def render(filename: str, cells: list[tuple[str, str]]) -> str:
    notebook = {
        "cells": [
            _cell(filename, position, kind, source)
            for position, (kind, source) in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "AAI Agentic Operations RAG",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "aai_agentic_ops_rag": {
                "schema_version": 1,
                "offline_default": True,
                "connected_calls_opt_in": True,
                "synthetic_data_only": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if tuple(sorted(LESSONS)) != EXPECTED_FILENAMES:
        raise SystemExit("Notebook content must define the ordered 00-05 course")
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for filename, cells in sorted(LESSONS.items()):
        rendered = render(filename, cells)
        destination = NOTEBOOKS / filename
        if args.check:
            if not destination.is_file() or destination.read_text() != rendered:
                stale.append(filename)
        else:
            destination.write_text(rendered, encoding="utf-8")
    if stale:
        raise SystemExit("Notebook sources are stale: " + ", ".join(stale))
    print(f"{'Verified' if args.check else 'Rendered'} {len(LESSONS)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
