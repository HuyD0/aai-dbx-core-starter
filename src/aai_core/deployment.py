"""Immutable application release manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ApplicationRelease:
    application: str
    release: str
    source_commit: str
    core_sdk_version: str
    model: Mapping[str, Any]
    prompt: Mapping[str, Any]
    retrieval: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    environment: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        serialized = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(serialized).hexdigest()

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.write_text(
            json.dumps(
                {**self.as_dict(), "digest": self.digest},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
