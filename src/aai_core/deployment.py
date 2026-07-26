"""Immutable application release manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, field_serializer, field_validator

from aai_core.contracts import ContractModel, freeze_value, thaw_value


class ApplicationRelease(ContractModel):
    """Persistable evidence for one reproducible AI application release."""

    application: str = Field(min_length=1)
    release: str = Field(min_length=1)
    source_commit: str = Field(min_length=1)
    core_sdk_version: str = Field(min_length=1)
    model: dict[str, JsonValue]
    prompt: dict[str, JsonValue]
    retrieval: dict[str, JsonValue]
    evaluation: dict[str, JsonValue]
    environment: str = Field(min_length=1)
    schema_version: Literal["1"] = "1"

    @field_validator(
        "model",
        "prompt",
        "retrieval",
        "evaluation",
        mode="after",
    )
    @classmethod
    def freeze_evidence(cls, value: Mapping[str, JsonValue]):
        return freeze_value(value)

    @field_serializer("model", "prompt", "retrieval", "evaluation")
    def serialize_evidence(self, value: Mapping[str, JsonValue]):
        return thaw_value(value)

    def as_dict(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json")

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
