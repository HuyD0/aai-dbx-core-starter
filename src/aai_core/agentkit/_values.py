"""Small value-shape helpers shared across AgentKit boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def is_missing_scalar(value: Any) -> bool:
    """Whether a scalar is one of the null values dataframe rows can carry.

    This stays dependency-free: numpy and pandas are optional MLflow
    dependencies, while Python ``float``, numpy floating scalars, Decimal
    NaNs, numpy NaT, pandas NA, and pandas NaT all need the same treatment.
    Array-like and container values are not scalar nulls; coercing their
    elementwise comparisons to bool is ambiguous and can itself raise.
    """

    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return True
    if isinstance(value, (str, bytes, Mapping)):
        return False
    if isinstance(value, Sequence):
        return False
    shape = getattr(value, "shape", None)
    if shape not in (None, ()):
        return False
    try:
        unequal = value != value
        return bool(unequal)
    except (TypeError, ValueError):
        return False
