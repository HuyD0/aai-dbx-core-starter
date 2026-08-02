"""Tiny display and state helpers used only by the teaching notebooks."""

from __future__ import annotations

import json
import os
from pathlib import Path

from aai_local_classification.settings import PROJECT_ROOT
from aai_local_classification.tracking import local_paths


def study_root() -> Path:
    """Use an isolated checker root when set, otherwise the course directory."""

    override = os.getenv("AAI_CLASSIFICATION_PROJECT_ROOT")
    return Path(override).resolve() if override else PROJECT_ROOT


def state_exists(name: str) -> bool:
    return (local_paths(study_root()).state_root / name).is_file()


def read_state(name: str) -> dict[str, object]:
    path = local_paths(study_root()).state_root / name
    return json.loads(path.read_text(encoding="utf-8"))


def short_digest(value: str, width: int = 12) -> str:
    return f"{value[:width]}…"
