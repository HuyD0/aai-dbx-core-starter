"""User-scoped long-term memory tools over a LangGraph ``BaseStore``.

Ordinary application code, like every tool in this template: adapt the specs
to the generated app's ``AsyncToolRegistry``, to LangChain tools, or bind the
handlers directly inside graph nodes. The store is durable and shared across
conversations; each tool closes over one user's namespace so the model can
only ever read and write that user's memories.

Memories are evidence, not just convenience. A ``decision`` memory records
what a reviewer approved or rejected — with the reason and the originating
request — so a later session can retrieve why, and a review can trace which
signal changed which behavior. The ``user_id`` is an opaque identifier the
serving layer resolves (for example from the Databricks Apps forwarded
identity); it never belongs in resource tags, trace inputs, or tool output
metadata.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langgraph.store.base import BaseStore
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_MEMORY_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_OPAQUE_ID = re.compile(r"^[^\x00\r\n]{1,256}$")


def _namespace_label(value: str) -> str:
    """Encode an opaque identifier as a store namespace label.

    LangGraph store namespace labels must not contain ``.``, and the check
    fires only on writes — an unencoded email-shaped ``user_id`` would read
    "not found" forever and crash on the first save. The encoding is
    injective ("a.b" and "a%2Eb" stay distinct) so users can never collide.
    """

    return value.replace("%", "%25").replace(".", "%2E")


class MemoryKind(StrEnum):
    """What a stored memory is evidence of."""

    PREFERENCE = "preference"
    DECISION = "decision"


class GetMemoryInput(BaseModel):
    """Strict tool boundary; unexpected or coerced arguments are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    key: str = Field(description="Memory key, e.g. preferred-region")

    @field_validator("key")
    @classmethod
    def _valid_key(cls, value: str) -> str:
        if not _MEMORY_KEY.fullmatch(value):
            raise ValueError(
                "memory key must be lowercase alphanumeric with ._- separators"
            )
        return value


class SaveMemoryInput(BaseModel):
    """Strict tool boundary; unexpected or coerced arguments are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    key: str = Field(description="Memory key, e.g. preferred-region")
    content: str = Field(max_length=2_000, description="What to remember")
    kind: MemoryKind = Field(
        default=MemoryKind.PREFERENCE,
        description="preference for durable user context, decision for a "
        "reviewed approval or rejection",
    )
    reason_code: str = Field(
        default="",
        max_length=64,
        description="For decisions: the review's reason_code",
    )
    request_id: str = Field(
        default="",
        max_length=256,
        description="For decisions: the request the decision was made on",
    )

    @field_validator("key")
    @classmethod
    def _valid_key(cls, value: str) -> str:
        if not _MEMORY_KEY.fullmatch(value):
            raise ValueError(
                "memory key must be lowercase alphanumeric with ._- separators"
            )
        return value

    @field_validator("kind", mode="before")
    @classmethod
    def _kind_from_wire(cls, value: Any) -> Any:
        # Tool arguments arrive as JSON; accept the wire string for the enum
        # while strict mode still rejects coerced and unknown values.
        if isinstance(value, str) and not isinstance(value, MemoryKind):
            return MemoryKind(value)
        return value

    @model_validator(mode="after")
    def _decisions_carry_lineage(self) -> SaveMemoryInput:
        if self.kind is MemoryKind.DECISION:
            if not self.reason_code or not self.request_id:
                raise ValueError(
                    "a decision memory needs reason_code and request_id so "
                    "the review stays traceable to its request"
                )
        elif self.reason_code or self.request_id:
            raise ValueError(
                "reason_code and request_id are reserved for decision memories"
            )
        return self


class DeleteMemoryInput(BaseModel):
    """Strict tool boundary; unexpected or coerced arguments are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    key: str = Field(description="Memory key to delete")

    @field_validator("key")
    @classmethod
    def _valid_key(cls, value: str) -> str:
        if not _MEMORY_KEY.fullmatch(value):
            raise ValueError(
                "memory key must be lowercase alphanumeric with ._- separators"
            )
        return value


@dataclass(frozen=True)
class MemoryToolSpec:
    """Framework-neutral tool description mirroring the app's ``ToolSpec``."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[..., Awaitable[dict[str, Any]]]


def build_user_memory_tools(
    store: BaseStore,
    *,
    user_id: str,
    namespace_prefix: tuple[str, ...] = ("memories",),
) -> tuple[MemoryToolSpec, MemoryToolSpec, MemoryToolSpec]:
    """Build get/save/delete memory tools scoped to one user's namespace.

    Every handler re-validates its arguments through the strict input models
    above — a handler bound directly inside a graph node gets the same
    boundary as one behind a tool registry. Handlers use only the store's
    async API and behave defensively: a missing memory is a structured
    not-found result and deletion is idempotent, so a degraded memory never
    crashes the agent loop.
    """

    if not _OPAQUE_ID.fullmatch(user_id or ""):
        raise ValueError("user_id must be a non-empty opaque identifier")
    if not namespace_prefix:
        raise ValueError("namespace_prefix must not be empty")
    for part in namespace_prefix:
        if not _OPAQUE_ID.fullmatch(part or "") or "." in part:
            raise ValueError(
                "namespace_prefix parts must be non-empty text without periods"
            )
    if namespace_prefix[0] == "langgraph":
        raise ValueError("namespace_prefix must not start with the reserved label")
    namespace = (*namespace_prefix, _namespace_label(user_id))

    async def get_user_memory(*, key: str) -> dict[str, Any]:
        inputs = GetMemoryInput(key=key)
        item = await store.aget(namespace, inputs.key)
        if item is None:
            return {"found": False, "key": inputs.key}
        return {"found": True, "key": inputs.key, "memory": dict(item.value)}

    async def save_user_memory(
        *,
        key: str,
        content: str,
        kind: MemoryKind | str = MemoryKind.PREFERENCE,
        reason_code: str = "",
        request_id: str = "",
    ) -> dict[str, Any]:
        inputs = SaveMemoryInput(
            key=key,
            content=content,
            kind=kind,
            reason_code=reason_code,
            request_id=request_id,
        )
        value: dict[str, Any] = {"content": inputs.content, "kind": str(inputs.kind)}
        if inputs.kind is MemoryKind.DECISION:
            value["reason_code"] = inputs.reason_code
            value["request_id"] = inputs.request_id
        await store.aput(namespace, inputs.key, value)
        return {"saved": True, "key": inputs.key, "kind": str(inputs.kind)}

    async def delete_user_memory(*, key: str) -> dict[str, Any]:
        inputs = DeleteMemoryInput(key=key)
        await store.adelete(namespace, inputs.key)
        return {"deleted": True, "key": inputs.key}

    return (
        MemoryToolSpec(
            name="get_user_memory",
            description="Retrieve one of the user's saved memories by key.",
            input_model=GetMemoryInput,
            handler=get_user_memory,
        ),
        MemoryToolSpec(
            name="save_user_memory",
            description="Save or update a memory for the user. Use kind "
            "'decision' with reason_code and request_id to record a reviewed "
            "approval or rejection.",
            input_model=SaveMemoryInput,
            handler=save_user_memory,
        ),
        MemoryToolSpec(
            name="delete_user_memory",
            description="Delete one of the user's saved memories by key.",
            input_model=DeleteMemoryInput,
            handler=delete_user_memory,
        ),
    )
