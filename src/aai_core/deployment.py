"""Immutable application release manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.manifest import _reject_secret_like_keys, _reject_secret_like_values

__all__ = ["ApplicationRelease"]


class ApplicationRelease(ContractModel):
    """Persistable evidence for one reproducible AI application release."""

    model_config = ConfigDict(hide_input_in_errors=True)

    application: str = Field(min_length=1)
    release: str = Field(min_length=1)
    source_commit: str = Field(min_length=1)
    core_sdk_version: str = Field(min_length=1)
    model: dict[str, JsonValue]
    prompt: dict[str, JsonValue]
    retrieval: dict[str, JsonValue]
    evaluation: dict[str, JsonValue]
    world: dict[str, JsonValue] = Field(default_factory=dict)
    tools: dict[str, JsonValue] = Field(default_factory=dict)
    control: dict[str, JsonValue] = Field(default_factory=dict)
    environment: str = Field(min_length=1)
    schema_version: Literal["1", "2"] = "2"

    @model_validator(mode="before")
    @classmethod
    def reject_secret_material(cls, value: Any) -> Any:
        _reject_secret_like_keys(value)
        _reject_secret_like_values(value)
        return value

    @field_validator(
        "model",
        "prompt",
        "retrieval",
        "evaluation",
        "world",
        "tools",
        "control",
        mode="after",
    )
    @classmethod
    def freeze_evidence(
        cls,
        value: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        return cast(Mapping[str, JsonValue], freeze_value(value))

    @field_serializer(
        "model",
        "prompt",
        "retrieval",
        "evaluation",
        "world",
        "tools",
        "control",
    )
    def serialize_evidence(
        self,
        value: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], thaw_value(value))

    @model_validator(mode="after")
    def version_matches_evidence(self) -> ApplicationRelease:
        if self.schema_version == "1" and (self.world or self.tools or self.control):
            raise ValueError(
                "ApplicationRelease schema version 1 cannot contain world, tools, "
                "or control evidence"
            )
        return self

    def as_dict(self) -> dict[str, JsonValue]:
        document = self.model_dump(mode="json")
        if self.schema_version == "1":
            for field_name in ("world", "tools", "control"):
                document.pop(field_name, None)
        return document

    @property
    def world_digest(self) -> str:
        """Digest the external state the application observed or depended on."""

        evidence = {"retrieval": thaw_value(self.retrieval)}
        if self.schema_version == "2":
            evidence["world"] = thaw_value(self.world)
        return _canonical_digest(evidence)

    @property
    def learning_digest(self) -> str:
        """Digest the code and adaptive-system configuration."""

        return _canonical_digest(
            {
                "source_commit": self.source_commit,
                "core_sdk_version": self.core_sdk_version,
                "model": thaw_value(self.model),
                "prompt": thaw_value(self.prompt),
                "retrieval": thaw_value(self.retrieval),
                "tools": thaw_value(self.tools),
            }
        )

    @property
    def control_digest(self) -> str:
        """Digest the evaluation and governance controls for the release."""

        return _canonical_digest(
            {
                "evaluation": thaw_value(self.evaluation),
                "control": thaw_value(self.control),
            }
        )

    @property
    def clock_digests(self) -> dict[str, str]:
        """Stable World, Learning, and Control clock join keys."""

        return {
            "world": self.world_digest,
            "learning": self.learning_digest,
            "control": self.control_digest,
        }

    @property
    def digest(self) -> str:
        return _canonical_digest(self.as_dict())

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        document: dict[str, Any] = {**self.as_dict(), "digest": self.digest}
        if self.schema_version == "2":
            document["clock_digests"] = self.clock_digests
        destination.write_text(
            json.dumps(
                document,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _canonical_digest(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
