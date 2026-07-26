"""Shared validation defaults for public, persisted SDK contracts."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Strict immutable base for configuration and evidence boundaries.

    Provider and MLflow response objects intentionally do not inherit from
    this class. They remain native objects so the SDK does not create stale
    mirrors of upstream APIs.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )


class FrozenMapping(Mapping[str, Any]):
    """Deeply immutable mapping that remains copyable and pickle-safe.

    ``MappingProxyType`` is immutable but cannot be deep-copied or pickled,
    which breaks standard Pydantic model operations. This small value type
    preserves mapping behavior while reconstructing from ordinary Python
    containers for serialization and copying.
    """

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, Any]) -> None:
        object.__setattr__(
            self,
            "_data",
            {str(key): freeze_value(item) for key, item in value.items()},
        )

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMapping({self._data!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())

    def __reduce__(self):
        return (type(self), (thaw_value(self),))

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenMapping:
        copied = type(self)(copy.deepcopy(thaw_value(self), memo))
        memo[id(self)] = copied
        return copied

    def copy(self) -> dict[str, Any]:
        """Return a mutable deep copy, matching ``MappingProxyType.copy()``."""

        return copy.deepcopy(thaw_value(self))


def freeze_value(value: Any) -> Any:
    """Recursively freeze JSON/configuration containers."""

    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, list | tuple):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(freeze_value(item) for item in value)
    return value


def thaw_value(value: Any) -> Any:
    """Convert frozen containers back to ordinary serializable containers."""

    if isinstance(value, Mapping):
        return {str(key): thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    if isinstance(value, frozenset):
        return [thaw_value(item) for item in value]
    return value
