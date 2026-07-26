"""Bounded formatting for full-evaluation scorer failures."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

_NEGATIVE_RATINGS = {"false", "fail", "failed", "no"}
_MAX_TRIAGE_ITEMS = 20
_MAX_TRIAGE_DETAIL_CHARS = 240


def print_failure_triage(
    report,
    *,
    max_items: int = _MAX_TRIAGE_ITEMS,
    include_details: bool = False,
) -> None:
    """Print bounded failures without directly reading raw input/output columns.

    Scorer rationales and errors remain governed log data and may themselves
    quote source content.
    """

    frame = getattr(report, "result_df", None)
    if frame is None:
        print("Per-row scorer triage unavailable (no result dataframe).")
        return

    findings: list[tuple[int, str, str | None]] = []
    scorer_names = {
        str(column).removesuffix("/value")
        for column in frame.columns
        if str(column).endswith("/value")
    }
    scorer_names.update(
        str(column).removesuffix("/error_message")
        for column in frame.columns
        if str(column).endswith("/error_message")
    )
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        for scorer_name in sorted(scorer_names):
            value = row.get(f"{scorer_name}/value")
            error = row.get(f"{scorer_name}/error_message")
            if not _has_value(error) and not _is_explicit_failure(value):
                continue
            rationale = row.get(f"{scorer_name}/rationale")
            detail = error if _has_value(error) else rationale
            findings.append(
                (
                    row_number,
                    scorer_name,
                    _one_line(detail) if include_details else None,
                )
            )

    if not findings:
        print("Per-row scorer triage: no explicit scorer failures.")
        return

    detail_label = (
        "rationale/error details enabled; raw columns omitted"
        if include_details
        else "details omitted; use --show-triage-details only under log data policy"
    )
    print(f"Per-row scorer triage (row number, scorer; {detail_label}):")
    for row_number, scorer_name, detail in findings[:max_items]:
        suffix = f" — {detail}" if detail else ""
        print(f"- row {row_number}: {scorer_name}{suffix}")
    omitted = len(findings) - max_items
    if omitted > 0:
        print(f"- ... {omitted} additional failure(s) omitted")


def _is_explicit_failure(value: Any) -> bool:
    if not _has_value(value):
        return False
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value.strip().lower() in _NEGATIVE_RATINGS
    if isinstance(value, Real):
        return float(value) == 0.0
    return False


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Real) and not isinstance(value, bool):
        return not math.isnan(float(value))
    text = str(value).strip().lower()
    return text not in {"", "nan", "nat", "none", "<na>"}


def _one_line(value: Any) -> str:
    if not _has_value(value):
        return "failed (no rationale supplied)"
    text = " ".join(str(value).split())
    if len(text) <= _MAX_TRIAGE_DETAIL_CHARS:
        return text
    return text[: _MAX_TRIAGE_DETAIL_CHARS - 3] + "..."
