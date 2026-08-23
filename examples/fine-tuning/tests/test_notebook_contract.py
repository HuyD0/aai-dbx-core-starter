"""Teaching contract for the rendered fine-tuning lessons."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
from notebook_content import EXPECTED_LESSON_COUNT, LESSONS

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT / "notebooks"
CELL_ID = re.compile(r"^aai-\d{2}-[0-9a-f]{12}$")

# Every lesson must carry each teaching marker, recognised by any alias.
MARKERS = {
    "vocabulary": ("Words introduced",),
    "predict": ("Before you run this",),
    "expected output": ("What you should see",),
    "interpret": ("How to interpret",),
    "guided exercise": ("Guided exercise",),
    "solution": ("Solution",),
    "recap": ("Recap",),
}

# The course must stay credential-free and offline: no pretrained downloads,
# no tokens. These strings may not appear anywhere in a rendered lesson.
FORBIDDEN = (
    "from_pretrained",
    "DATABRICKS_TOKEN",
    "AZURE_CLIENT_SECRET",
    "OPENAI_API_KEY",
)


def _notebooks() -> list[Path]:
    return sorted(NOTEBOOKS.glob("*.ipynb"))


def _source(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else str(source)


def _marker_index(cells: list[dict], marker: str) -> int:
    aliases = MARKERS[marker]
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "markdown":
            continue
        text = _source(cell)
        if any(
            re.search(rf"(?im)^\s*#{{1,6}}\s+{re.escape(alias)}\b", text)
            for alias in aliases
        ):
            return index
    raise AssertionError(f"missing {marker!r}; expected one of: {', '.join(aliases)}")


def test_lessons_are_one_contiguous_course_with_no_orphans():
    assert len(LESSONS) == EXPECTED_LESSON_COUNT
    names = sorted(LESSONS)
    assert [name[:2] for name in names] == [
        f"{number:02d}" for number in range(len(names))
    ]
    # Set equality with the directory: a deleted lesson or an orphan notebook
    # both fail, so the LESSONS dict cannot silently drift from disk.
    assert sorted(path.name for path in _notebooks()) == names


@pytest.mark.parametrize("path", _notebooks(), ids=lambda path: path.name)
def test_lesson_metadata_declares_the_offline_course_contract(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["kernelspec"]["name"] == "aai-fine-tuning"
    assert notebook["metadata"]["kernelspec"]["display_name"] == "AAI Fine-Tuning"
    course = notebook["metadata"]["aai_course"]
    assert course["schema_version"] == 1
    assert course["network_required_after_install"] is False
    assert course["cloud_credentials_required"] is False


@pytest.mark.parametrize("path", _notebooks(), ids=lambda path: path.name)
def test_lesson_cells_are_clean_addressable_and_compilable(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    ids = [cell["id"] for cell in cells]
    assert len(ids) == len(set(ids))
    assert all(CELL_ID.fullmatch(cell_id) for cell_id in ids)
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse(_source(cell), filename=f"{path.name}:{cell['id']}")


@pytest.mark.parametrize("path", _notebooks(), ids=lambda path: path.name)
def test_lesson_carries_every_teaching_marker_in_a_sane_order(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    positions = {marker: _marker_index(cells, marker) for marker in MARKERS}
    assert positions["guided exercise"] < positions["solution"]
    assert positions["recap"] == max(positions.values())


@pytest.mark.parametrize("path", _notebooks(), ids=lambda path: path.name)
def test_first_code_cell_is_the_tagged_preflight_check(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "preflight" in first_code.get("metadata", {}).get("tags", [])
    assert "HF_HUB_OFFLINE" in _source(first_code)


@pytest.mark.parametrize("path", _notebooks(), ids=lambda path: path.name)
def test_lesson_never_downloads_weights_or_names_credentials(path):
    text = path.read_text(encoding="utf-8")
    for token in FORBIDDEN:
        assert token not in text, f"{path.name} must not contain {token!r}"
