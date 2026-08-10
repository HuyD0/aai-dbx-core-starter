"""Adapters that turn the thing under evaluation into a predict_fn.

Target resolution lives in the SDK so one `agent:` value in agentkit.yaml
covers a local callable, a Databricks serving endpoint, a Unity Catalog
model, and any HTTP/JSON endpoint. These wrappers keep the project-local
names used by the notebooks.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from aai_core.agentkit.config import ProjectContext
from aai_core.agentkit.targets import build_predict_fn, resolve_target

ROOT = Path(__file__).resolve().parents[2]


def answer_sheet_predict_fn(path: str | Path) -> Callable[[str], str]:
    """Replay recorded answers (question -> answer) — deterministic, offline."""

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    answers = {record["question"]: record["answer"] for record in records}

    def predict(question: str) -> str:
        return answers.get(question, "")

    return predict


def agent_predict_fn(project: ProjectContext | None = None) -> Callable[..., object]:
    """Call whatever `agent:` in agentkit.yaml resolves to."""

    project = project or ProjectContext.load(ROOT / "agentkit.yaml")
    target = resolve_target(
        project.config.agent, root=project.root, settings=project.settings
    )
    return build_predict_fn(target, project=project)


# Backwards-compatible name from the pre-agentkit template.
endpoint_predict_fn = agent_predict_fn
