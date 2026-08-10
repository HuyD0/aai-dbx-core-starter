"""LLM judge wiring (tier 2 only; model access is required).

Judges are versioned registry entries, not per-project definitions. Their
names, model bindings, instructions, and scales are governed platform assets,
so the same metric means the same thing across projects. This module resolves
the current dataset's plan and builds those registered scorers.

Judges route through the gateway-fronted ``judge-model`` logical resource in
``aai-platform.yml``. Calibrate them against human labels before using them in
a release threshold; see ``notebooks/01_align_judge.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aai_core import PlatformSettings
from aai_core.agentkit.catalog import build_scorer, select_scorers
from aai_core.agentkit.config import ProjectContext
from aai_core.agentkit.datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]


def _project_context(
    settings: PlatformSettings | None,
    project: ProjectContext | None,
) -> ProjectContext:
    """Resolve one authoritative project context without ignoring arguments."""

    if settings is not None and project is not None:
        raise ValueError("pass settings or project, not both")
    if project is not None:
        return project
    loaded = ProjectContext.load(ROOT / "agentkit.yaml")
    if settings is None:
        return loaded
    return ProjectContext(config=loaded.config, settings=settings, root=loaded.root)


def judge_model_uri(
    settings: PlatformSettings | None = None,
    project: ProjectContext | None = None,
) -> str:
    """Return the approved judge endpoint for this project."""

    return _project_context(settings, project).judge_model_uri()


def judge_scorers(
    settings: PlatformSettings | None = None,
    project: ProjectContext | None = None,
) -> list[Any]:
    """Build every judge selected for this project's dataset shape."""

    context = _project_context(settings, project)
    dataset = load_dataset(context.config.dataset, root=context.root)
    plan = select_scorers(
        dataset.shape,
        context.config,
        mode="live",
        judges_enabled=True,
    )
    model = context.judge_model_uri()
    return [
        build_scorer(
            entry.spec,
            judge_model_uri=model,
            guidelines=context.config.scorers.guidelines,
        )
        for entry in plan.entries
        if entry.spec.judge is not None
    ]
