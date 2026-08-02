"""Stable digests that bind persisted evidence to code, lock, and policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aai_local_classification.contracts import ProjectSettings
from aai_local_classification.settings import PROJECT_ROOT


def _sha256_payload(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_source_digests() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        name: _file_sha256(package / name)
        for name in (
            "contracts.py",
            "evaluation.py",
            "modeling.py",
            "workflow.py",
        )
    }


def selection_policy_sha256(settings: ProjectSettings) -> str:
    """Bind selection evidence to its executable and declared controls."""

    lock = PROJECT_ROOT / "uv.lock"
    return _sha256_payload(
        {
            "schema_version": settings.schema_version,
            "random_seed": settings.random_seed,
            "features": settings.features.model_dump(mode="json"),
            "selection": settings.selection.model_dump(mode="json"),
            "source_sha256": _training_source_digests(),
            "uv_lock_sha256": _file_sha256(lock),
        }
    )


def gate_policy_sha256(settings: ProjectSettings) -> str:
    """Bind release evidence to selection plus the declared promotion gate."""

    return _sha256_payload(
        {
            "selection_policy_sha256": selection_policy_sha256(settings),
            "promotion_gate": settings.promotion_gate.model_dump(mode="json"),
        }
    )
