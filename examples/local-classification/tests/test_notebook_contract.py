from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT / "notebooks"


def test_numbered_course_is_clean_offline_and_deterministically_rendered():
    paths = sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))
    assert [path.name[:2] for path in paths] == [
        f"{number:02d}" for number in range(10)
    ]

    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["metadata"]["aai_course"] == {
            "cloud_credentials_required": False,
            "network_required": False,
            "schema_version": 1,
        }
        cell_ids = [cell["id"] for cell in notebook["cells"]]
        assert len(cell_ids) == len(set(cell_ids))
        assert all(cell_id.startswith("aai-") for cell_id in cell_ids)
        combined = "\n".join(
            (
                "".join(cell["source"])
                if isinstance(cell["source"], list)
                else cell["source"]
            )
            for cell in notebook["cells"]
        )
        assert "### Exercise" in combined
        assert "**Hint:**" in combined
        assert "**Checkpoint:**" in combined
        assert "%pip" not in combined
        assert "pip install" not in combined
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []

    subprocess.run(
        [sys.executable, "scripts/render_notebooks.py", "--check"],
        cwd=PROJECT,
        check=True,
    )
