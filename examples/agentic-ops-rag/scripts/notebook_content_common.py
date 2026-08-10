"""Shared cell builders for the agentic operations RAG workshop."""

from __future__ import annotations

Cell = tuple[str, str]


def m(text: str) -> Cell:
    return ("markdown", text)


def c(text: str) -> Cell:
    return ("code", text)


def preflight() -> str:
    return r"""
# Notebook preflight — configuration only; this cell makes no cloud request.
import importlib.util
import sys
from pathlib import Path

setup_path = next(
    path
    for parent in (Path.cwd(), *Path.cwd().parents)
    for path in (
        parent / "notebook_setup.py",
        parent / "examples" / "agentic-ops-rag" / "notebook_setup.py",
    )
    if path.is_file()
)
spec = importlib.util.spec_from_file_location("agentic_ops_rag_setup", setup_path)
setup = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = setup
spec.loader.exec_module(setup)
course_root = setup.find_course_root(setup_path.parent)
session = setup.prepare_notebook_environment(course_root)
session.safe_summary()
"""


def knowledge_check(*questions: str) -> str:
    numbered = "\n".join(
        f"{number}. {question}" for number, question in enumerate(questions, 1)
    )
    return f"""
## Knowledge check

Answer from the evidence you produced, not from memory:

{numbered}

<details>
<summary>How to use this check</summary>

If an answer cannot point to a row, trace, contract, or failed check from this
lesson, revisit the exercise before moving on.

</details>
"""
