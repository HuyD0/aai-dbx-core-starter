"""Provenance records and the footer every answer carries.

Consumers judge trustworthiness from the footer: which source tier answered
(semantic layer › curated reference › raw table), which objects were read,
who owns them, how fresh the data is, and the exact SQL. The footer is
rendered by code from tool-recorded evidence — never composed by the model —
and parses back losslessly because the deterministic scorers grade it.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

FOOTER_OPEN = "[provenance]"
FOOTER_CLOSE = "[/provenance]"
_FIELD = re.compile(r"^(tier|sources|owner|freshness|rows|value|sql): ?(.*)$")


class SourceTier(StrEnum):
    """Descending trust order, matching the runbook's search order."""

    SEMANTIC_LAYER = "semantic_layer"
    CURATED_REFERENCE = "curated_reference"
    RAW_TABLE = "raw_table"


class ProvenanceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: SourceTier
    sources: tuple[str, ...] = Field(min_length=1)
    owner: str | None = None
    freshness: str | None = None
    rows: int | None = None
    value: str | None = None
    sql: str | None = None


def render_footer(records: tuple[ProvenanceRecord, ...]) -> str:
    """Deterministic, single-line-per-field footer appended to answers."""

    blocks: list[str] = []
    for record in records:
        lines = [FOOTER_OPEN, f"tier: {record.tier.value}"]
        lines.append("sources: " + ", ".join(record.sources))
        if record.owner:
            lines.append(f"owner: {_flatten(record.owner)}")
        if record.freshness:
            lines.append(f"freshness: {_flatten(record.freshness)}")
        if record.rows is not None:
            lines.append(f"rows: {record.rows}")
        if record.value is not None:
            lines.append(f"value: {_flatten(record.value)}")
        if record.sql:
            lines.append(f"sql: {_flatten(record.sql)}")
        lines.append(FOOTER_CLOSE)
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def parse_footer(text: str) -> tuple[ProvenanceRecord, ...]:
    """Recover the records from an answer; scorers rely on this round-trip."""

    records: list[ProvenanceRecord] = []
    fields: dict[str, str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == FOOTER_OPEN:
            fields = {}
            continue
        if line == FOOTER_CLOSE:
            if fields is not None and "tier" in fields and "sources" in fields:
                records.append(
                    ProvenanceRecord(
                        tier=SourceTier(fields["tier"]),
                        sources=tuple(
                            part.strip()
                            for part in fields["sources"].split(",")
                            if part.strip()
                        ),
                        owner=fields.get("owner"),
                        freshness=fields.get("freshness"),
                        rows=(
                            int(fields["rows"])
                            if fields.get("rows", "").isdigit()
                            else None
                        ),
                        value=fields.get("value"),
                        sql=fields.get("sql"),
                    )
                )
            fields = None
            continue
        if fields is None:
            continue
        if match := _FIELD.match(line):
            fields[match.group(1)] = match.group(2)
    return tuple(records)


def strip_footer(text: str) -> str:
    """The prose part of an answer, without provenance blocks."""

    kept: list[str] = []
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == FOOTER_OPEN:
            inside = True
            continue
        if stripped == FOOTER_CLOSE:
            inside = False
            continue
        if not inside:
            kept.append(line)
    return "\n".join(kept).strip()


def _flatten(value: str) -> str:
    return " ".join(value.split())
