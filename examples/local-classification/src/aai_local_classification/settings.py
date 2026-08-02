"""Load the course configuration from one strict, clone-local YAML file."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from aai_local_classification.contracts import ProjectSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "project.yaml"


def load_settings(path: Path | None = None) -> ProjectSettings:
    """Parse YAML through JSON so strict models retain JSON date/tuple semantics."""

    config_path = path or DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return ProjectSettings.model_validate_json(
        json.dumps(raw, default=str),
        strict=True,
    )
