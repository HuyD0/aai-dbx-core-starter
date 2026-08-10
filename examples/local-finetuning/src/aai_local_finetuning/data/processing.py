"""Sanitize source rows and convert them to stable logical records."""

from __future__ import annotations

from collections import Counter

from .normalization import (
    content_id,
    inferred_template,
    mask_sensitive_text,
    normalize_label,
    template_id,
)
from .schemas import CanonicalizationResult, CanonicalRecord, RawBitextRow

_FIELDS = ("flags", "instruction", "category", "intent", "response")
_REQUIRED_FIELDS = ("instruction", "category", "intent", "response")


def canonicalize_bitext(rows: tuple[RawBitextRow, ...]) -> CanonicalizationResult:
    """Normalize, mask, validate, and fingerprint raw Bitext rows."""

    missing: Counter[str] = Counter()
    sensitive: Counter[str] = Counter()
    records_with_sensitive_patterns = 0
    invalid_record_count = 0
    records: list[CanonicalRecord] = []

    for row in rows:
        raw = {field: getattr(row, field) for field in _FIELDS}
        for field, value in raw.items():
            if not value.strip():
                missing[field] += 1
        if any(not raw[field].strip() for field in _REQUIRED_FIELDS):
            invalid_record_count += 1
            continue

        masked_fields: dict[str, str] = {}
        row_sensitive: Counter[str] = Counter()
        for field in ("flags", "instruction", "response"):
            masked_fields[field], counts = mask_sensitive_text(raw[field])
            row_sensitive.update(counts)
        sensitive.update(row_sensitive)
        records_with_sensitive_patterns += bool(row_sensitive)

        category = normalize_label(raw["category"])
        intent = normalize_label(raw["intent"])
        instruction = masked_fields["instruction"]
        response = masked_fields["response"]
        if not instruction or not response or not category or not intent:
            invalid_record_count += 1
            continue

        template_text = inferred_template(instruction)
        if not template_text:
            invalid_record_count += 1
            continue
        records.append(
            CanonicalRecord(
                example_id=content_id(
                    instruction=instruction,
                    category=category,
                    intent=intent,
                    response=response,
                ),
                flags=masked_fields["flags"],
                instruction=instruction,
                category=category,
                intent=intent,
                response=response,
                template_id=template_id(template_text),
                template_text=template_text,
            )
        )

    return CanonicalizationResult(
        records=tuple(records),
        missing_by_field={field: missing[field] for field in _FIELDS},
        invalid_record_count=invalid_record_count,
        sensitive_pattern_counts=dict(sorted(sensitive.items())),
        records_with_sensitive_patterns=records_with_sensitive_patterns,
    )


def deduplicate_exact(
    records: tuple[CanonicalRecord, ...],
) -> tuple[tuple[CanonicalRecord, ...], int]:
    """Remove exact logical duplicates independent of their source row order."""

    by_id: dict[str, CanonicalRecord] = {}
    for record in records:
        existing = by_id.get(record.example_id)
        if existing is None or record.flags < existing.flags:
            by_id[record.example_id] = record
    unique = tuple(by_id[record_id] for record_id in sorted(by_id))
    return unique, len(records) - len(unique)
