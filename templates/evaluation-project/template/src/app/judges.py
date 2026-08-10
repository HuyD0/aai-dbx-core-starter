"""LLM judge wiring (tier 2 only — needs model access).

Judges are versioned registry entries, not per-project definitions: the
scorer name, its judge model binding, its instructions and its scale are
governed platform assets, so a score means the same thing on every team.
This module resolves the plan for the current dataset and builds the
executable scorers; it does not define new ones.

Judges route through the platform's gateway-fronted judge endpoint (the
`judge-model` logical name in aai-platform.yml). Calibrate against human
labels before trusting a judge in the gate — see notebooks/01_align_judge.py.
"""

from __future__ import annotations

from pathlib import Path

from aai_core.agentkit.catalog import build_scorer, select_scorers
from aai_core.agentkit.config import ProjectContext
from aai_core.agentkit.datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]


def judge_model_uri(settings=None, project: ProjectContext | None = None) -> str:
    """The approved judge endpoint for this project."""

    project = project or ProjectContext.load(ROOT / "agentkit.yaml")
    return project.judge_model_uri()


def judge_scorers(settings=None, project: ProjectContext | None = None) -> list:
    """Every judge the registry selects for this project's dataset."""

    project = project or ProjectContext.load(ROOT / "agentkit.yaml")
    dataset = load_dataset(project.config.dataset, root=project.root)
    plan = select_scorers(
        dataset.shape, project.config, mode="live", judges_enabled=True
    )
    model = project.judge_model_uri()
    return [
        build_scorer(
            entry.spec,
            judge_model_uri=model,
            guidelines=project.config.scorers.guidelines,
        )
        for entry in plan.entries
        if entry.spec.judge is not None
    ]
