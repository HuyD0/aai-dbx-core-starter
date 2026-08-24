"""Small value-shape helpers shared across AgentKit boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
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
    if isinstance(value, Decimal):
        # Decimal signaling NaNs raise InvalidOperation when compared. The
        # explicit predicate handles both quiet and signaling NaNs safely.
        return value.is_nan()
    shape = getattr(value, "shape", None)
    if shape not in (None, ()):
        return False
    try:
        unequal = value != value
        return bool(unequal)
    except (TypeError, ValueError):
        return False


def numeric_score(value: Any) -> float | None:
    """A scorer verdict as a number, matching MLflow's mean aggregation.

    Booleans and yes/no verdicts become 1/0; unsupported categorical values
    stay ``None`` rather than being assigned an invented ordering.
    """

    if is_missing_scalar(value):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"yes", "true", "pass", "passed"}:
            return 1.0
        if normalized in {"no", "false", "fail", "failed"}:
            return 0.0
    return None
