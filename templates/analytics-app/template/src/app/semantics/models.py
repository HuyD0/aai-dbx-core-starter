"""Strict contract for the neutral semantic model.

The YAML in semantics/semantic_model.yml is validated through these models
before anything queries a warehouse. Definitions are human-curated; this
module only enforces their internal consistency (sources exist, joins are
declared, names are addressable). Values come from ``yaml.safe_load`` so the
models validate plain Python data; extras are forbidden and instances are
frozen, mirroring the platform's contract discipline for application code.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_TABLE = re.compile(r"^[A-Za-z0-9_{}.-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")
_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Aggregation(StrEnum):
    SUM = "sum"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class MetricFilter(BaseModel):
    """Structured, dialect-neutral filter — deliberately not free SQL."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str = Field(min_length=1)
    op: Literal["=", "<>", ">", ">=", "<", "<="]
    value: str = Field(min_length=1)


class SourceTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    table: str = Field(min_length=5)
    grain: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    freshness_sla_hours: int = Field(gt=0)
    loaded_at_column: str | None = None
    gotchas: tuple[str, ...] = ()

    @model_validator(mode="after")
    def check_table_reference(self) -> SourceTable:
        if not _TABLE.match(self.table):
            raise ValueError(
                f"source table {self.table!r} must be a three-part "
                "catalog.schema.table name"
            )
        if self.loaded_at_column is not None and not _COLUMN.fullmatch(
            self.loaded_at_column
        ):
            raise ValueError("loaded_at_column must be a simple column identifier")
        return self

    @property
    def table_name(self) -> str:
        return self.table.rsplit(".", 1)[-1]


class Join(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    from_column: str = Field(min_length=1)
    to_column: str = Field(min_length=1)


class Dimension(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    column: str = Field(min_length=1)
    type: Literal["string", "date", "number"]
    encodings: Mapping[str, str] = Field(default_factory=dict)
    join: Join | None = None


class DetailField(BaseModel):
    """Governed row-level field exposed to ``query_rows``.

    Logical names are model-facing. Physical columns remain human-curated in
    the semantic model and are never accepted from a model-generated request.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    column: str = Field(min_length=1)
    type: Literal["string", "date", "number", "boolean"]

    @model_validator(mode="after")
    def check_column(self) -> DetailField:
        if not _COLUMN.fullmatch(self.column):
            raise ValueError("detail field column must be a simple identifier")
        return self


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    aggregation: Aggregation
    expr: str = Field(min_length=1)
    filter: MetricFilter | None = None
    description: str = Field(min_length=1)


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: int = Field(gt=0)


class SemanticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_model: ModelInfo
    sources: Mapping[str, SourceTable]
    dimensions: Mapping[str, Dimension]
    detail_fields: Mapping[str, DetailField] = Field(default_factory=dict)
    metrics: Mapping[str, Metric]

    @model_validator(mode="after")
    def check_cross_references(self) -> SemanticModel:
        for group in (
            self.sources,
            self.dimensions,
            self.detail_fields,
            self.metrics,
        ):
            for name in group:
                if not _NAME.match(name):
                    raise ValueError(f"name {name!r} must be lowercase snake_case")
        for name, dimension in self.dimensions.items():
            if dimension.source not in self.sources:
                raise ValueError(
                    f"dimension {name!r} references unknown source "
                    f"{dimension.source!r}"
                )
        for name, metric in self.metrics.items():
            if metric.source not in self.sources:
                raise ValueError(
                    f"metric {name!r} references unknown source {metric.source!r}"
                )
        for name, field in self.detail_fields.items():
            if field.source not in self.sources:
                raise ValueError(
                    f"detail field {name!r} references unknown source "
                    f"{field.source!r}"
                )
        if not self.metrics:
            raise ValueError("a semantic model requires at least one metric")
        return self

    def table_names(self) -> frozenset[str]:
        return frozenset(source.table_name for source in self.sources.values())

    def metric_catalog(self) -> str:
        """Compact, prompt-ready summary — never the full YAML."""

        lines = []
        for name, metric in sorted(self.metrics.items()):
            lines.append(
                f"- {name} ({metric.aggregation.value} of {metric.expr} on "
                f"{metric.source}): {metric.description}"
            )
        dimension_names = ", ".join(sorted(self.dimensions))
        lines.append(f"- dimensions available: {dimension_names}")
        by_source: dict[str, list[str]] = {}
        for name, field in sorted(self.detail_fields.items()):
            by_source.setdefault(field.source, []).append(name)
        for source, names in sorted(by_source.items()):
            lines.append(f"- governed row fields on {source}: {', '.join(names)}")
        return "\n".join(lines)


def load_semantic_model(path: str | Path) -> SemanticModel:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("semantic model file must contain a YAML mapping")
    return SemanticModel.model_validate(payload)
