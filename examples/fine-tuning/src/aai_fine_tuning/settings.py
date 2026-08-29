"""Resolve where the course stores its isolated, ignored local state."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
COURSE_ROOT_VARIABLE = "AAI_FINETUNE_PROJECT_ROOT"
DEFAULT_COURSE_ROOT = PROJECT / ".aai" / "course-v1"


@dataclass(frozen=True)
class LocalPaths:
    """Locations for course state, all under one disposable root."""

    root: Path
    mlflow_dir: Path
    mlflow_artifacts: Path
    hf_home: Path

    @property
    def mlflow_uri(self) -> str:
        return f"sqlite:///{self.mlflow_dir / 'mlflow.db'}"


def course_root() -> Path:
    """The state root: the Makefile-exported override, or the tracked default."""
    configured = os.environ.get(COURSE_ROOT_VARIABLE, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_COURSE_ROOT


def local_paths(root: Path | None = None) -> LocalPaths:
    resolved = root if root is not None else course_root()
    return LocalPaths(
        root=resolved,
        mlflow_dir=resolved / "mlflow",
        mlflow_artifacts=resolved / "mlflow" / "artifacts",
        hf_home=resolved / "hf",
    )


def ensure_local_paths(root: Path | None = None) -> LocalPaths:
    """Create the state directories a lesson is about to write into."""
    paths = local_paths(root)
    for directory in (paths.root, paths.mlflow_dir, paths.mlflow_artifacts):
        directory.mkdir(parents=True, exist_ok=True)
    return paths
