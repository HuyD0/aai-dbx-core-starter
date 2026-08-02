from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parents[1]
NOTEBOOKS = PROJECT / "notebooks"
CELL_ID = re.compile(r"^aai-\d{2}-[0-9a-f]{12}$")
WORD = re.compile(r"\b[\w'-]+\b")

MARKERS = {
    "vocabulary": ("Words introduced", "Vocabulary"),
    "predict": ("Before you run this", "Before you run", "Predict"),
    "expected output": ("What you should see", "Expected output"),
    "interpret": (
        "How to interpret",
        "How to read the output",
        "What you should notice",
        "Interpretation",
        "Interpret",
    ),
    "guided exercise": ("Guided exercise",),
    "solution": ("Solution",),
    "recap": ("Recap", "Final recap"),
}


def _source(cell: dict[str, object]) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else str(source)


def _contains_marker(text: str, aliases: tuple[str, ...]) -> bool:
    return any(
        re.search(
            rf"(?im)(?:^\s*#{{2,6}}\s+|^\s*\*\*|<summary>\s*)" rf"{re.escape(alias)}\b",
            text,
        )
        for alias in aliases
    )


def _marker_cell(cells: list[dict[str, object]], marker: str) -> int:
    aliases = MARKERS[marker]
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and _contains_marker(_source(cell), aliases):
            return index
    raise AssertionError(f"missing {marker!r}; expected one of: {', '.join(aliases)}")


def _marker_cells(cells: list[dict[str, object]], marker: str) -> list[int]:
    aliases = MARKERS[marker]
    return [
        index
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "markdown" and _contains_marker(_source(cell), aliases)
    ]


def _maximum_consecutive_code_cells(cells: list[dict[str, object]]) -> int:
    longest = 0
    current = 0
    for cell in cells:
        if cell["cell_type"] == "code":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def test_numbered_course_is_clean_offline_and_genuinely_instructive():
    paths = sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))
    assert [path.name[:2] for path in paths] == [
        f"{number:02d}" for number in range(10)
    ]

    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        metadata = notebook["metadata"]["aai_course"]
        assert metadata["schema_version"] == 2, f"{path.name}: use course schema v2"
        assert (
            metadata["network_required_after_install"] is False
        ), f"{path.name}: must work offline after installation"
        assert (
            metadata["cloud_credentials_required"] is False
        ), f"{path.name}: must not require cloud credentials"

        cells = notebook["cells"]
        assert (
            cells and cells[0]["cell_type"] == "markdown"
        ), f"{path.name}: begin by orienting the learner"
        cell_ids = [cell["id"] for cell in cells]
        assert len(cell_ids) == len(set(cell_ids)), f"{path.name}: duplicate cell IDs"
        assert all(
            CELL_ID.fullmatch(cell_id) for cell_id in cell_ids
        ), f"{path.name}: cell IDs must use stable aai-NN-<digest> values"

        markdown_cells = [cell for cell in cells if cell["cell_type"] == "markdown"]
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        markdown = "\n".join(_source(cell) for cell in markdown_cells)
        word_count = len(WORD.findall(markdown))
        assert len(markdown_cells) >= 8, (
            f"{path.name}: needs at least 8 teaching/narration cells, found "
            f"{len(markdown_cells)}"
        )
        assert (
            word_count >= 500
        ), f"{path.name}: needs at least 500 markdown words, found {word_count}"
        assert len(code_cells) >= 2, f"{path.name}: needs executable demonstrations"
        assert (
            _maximum_consecutive_code_cells(cells) <= 2
        ), f"{path.name}: explain and interpret code instead of stacking code blocks"

        marker_cells: dict[str, int] = {}
        for marker in MARKERS:
            try:
                marker_cells[marker] = _marker_cell(cells, marker)
            except AssertionError as error:
                raise AssertionError(f"{path.name}: {error}") from error

        predictions = _marker_cells(cells, "predict")
        expected_outputs = _marker_cells(cells, "expected output")
        code_positions = [
            index for index, cell in enumerate(cells) if cell["cell_type"] == "code"
        ]
        assert any(
            prediction < code_position < expected
            for prediction in predictions
            for code_position in code_positions
            for expected in expected_outputs
        ), f"{path.name}: include a predict, run, then expected-output sequence"
        interpretations = _marker_cells(cells, "interpret")
        assert any(
            code_position < interpretation
            for code_position in code_positions
            for interpretation in interpretations
        ), f"{path.name}: interpret a result after running code"
        assert (
            marker_cells["guided exercise"]
            < marker_cells["solution"]
            < marker_cells["recap"]
        ), f"{path.name}: order the guided exercise, solution, then recap"

        for position, cell in enumerate(code_cells, start=1):
            source = _source(cell)
            nonblank_lines = sum(bool(line.strip()) for line in source.splitlines())
            tags = cell.get("metadata", {}).get("tags", [])
            line_limit = 30 if "preflight" in tags else 20
            assert nonblank_lines <= line_limit, (
                f"{path.name}: code cell {position} has {nonblank_lines} nonblank "
                f"lines; limit is {line_limit}"
            )
            if not {"preflight", "solution"}.intersection(tags):
                assert not re.search(
                    r"(?m)^\s*raise\b", source
                ), f"{path.name}: code cell {position} intentionally raises an error"
            assert not re.search(
                r"(?m)^\s*assert\s+False\b", source
            ), f"{path.name}: code cell {position} intentionally fails"
            assert not re.search(
                r"(?im)(?:^\s*[%!]\s*(?:pip|conda|uv)\b|\bpip\s+install\b|"
                r"\buv\s+(?:sync|pip)\b)",
                source,
            ), f"{path.name}: dependencies belong in `make install`, not a lesson cell"
            assert (
                cell["execution_count"] is None
            ), f"{path.name}: tracked notebooks must have clean execution counts"
            assert (
                cell["outputs"] == []
            ), f"{path.name}: tracked notebooks must have clean outputs"

    subprocess.run(
        [sys.executable, "scripts/render_notebooks.py", "--check"],
        cwd=PROJECT,
        check=True,
    )


def test_course_state_and_root_shortcuts_use_the_v2_recoverable_workflow():
    course_makefile = (PROJECT / "Makefile").read_text(encoding="utf-8")
    root_makefile = (REPOSITORY / "Makefile").read_text(encoding="utf-8")

    assert "COURSE_ROOT := $(CURDIR)/.aai/course-v2" in course_makefile
    assert "course-reset:" in course_makefile
    assert 'mv "$$course_root" "$$destination"' in course_makefile
    assert "classification-doctor:" in root_makefile
    assert "classification-reset:" in root_makefile


def test_course_reset_archives_only_v2_state(tmp_path: Path):
    course_v2 = tmp_path / ".aai" / "course-v2"
    course_v1 = tmp_path / ".aai" / "course-v1"
    course_v2.mkdir(parents=True)
    course_v1.mkdir()
    (course_v2 / "evidence.txt").write_text("recover me\n", encoding="utf-8")
    (course_v1 / "legacy.txt").write_text("leave me\n", encoding="utf-8")

    result = subprocess.run(
        ["make", "-f", str(PROJECT / "Makefile"), "course-reset"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    archives = list((tmp_path / ".aai" / "archive").glob("course-v2-*"))
    assert len(archives) == 1
    assert (archives[0] / "evidence.txt").read_text(encoding="utf-8") == "recover me\n"
    assert not course_v2.exists()
    assert (course_v1 / "legacy.txt").read_text(encoding="utf-8") == "leave me\n"
    assert "Nothing was deleted" in result.stdout
