"""The knowledge router: curated reference docs, loaded on demand.

This is the context-engineering half of the skills layer. The router indexes
knowledge/*.md front-matter (topic, covered tables and metrics, keywords) so
the system prompt carries only a thin index summary; a full document enters
the model's context only when the lookup_reference tool asks for it. The
same front-matter powers the anti-staleness check in evals/offline_checks.py:
a doc that names a table or metric missing from the semantic model fails CI,
keeping definitions, docs, and data model in one reviewable diff.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.semantics.models import SemanticModel


class KnowledgeDoc(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: str = Field(min_length=1)
    title: str = Field(min_length=1)
    covers_tables: tuple[str, ...] = ()
    covers_metrics: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    body: str = Field(min_length=1)


class KnowledgeRouter:
    def __init__(self, root: str | Path) -> None:
        self._docs: dict[str, KnowledgeDoc] = {}
        for path in sorted(Path(root).glob("*.md")):
            doc = _parse_doc(path)
            if doc.topic in self._docs:
                raise ValueError(f"duplicate knowledge topic {doc.topic!r}")
            self._docs[doc.topic] = doc

    @property
    def topics(self) -> tuple[str, ...]:
        return tuple(self._docs)

    def index_summary(self) -> str:
        """One line per doc — this is all the system prompt carries."""

        lines = []
        for doc in self._docs.values():
            keywords = ", ".join(doc.keywords) if doc.keywords else "general"
            lines.append(f"- {doc.topic}: {doc.title} (keywords: {keywords})")
        return "\n".join(lines)

    def load(self, topic: str) -> KnowledgeDoc:
        if topic not in self._docs:
            known = ", ".join(self._docs)
            raise KeyError(f"unknown knowledge topic {topic!r}; known: {known}")
        return self._docs[topic]

    def cross_reference_issues(self, model: SemanticModel) -> list[str]:
        """Front-matter references that drifted from the semantic model."""

        issues: list[str] = []
        tables = model.table_names()
        metrics = set(model.metrics)
        for doc in self._docs.values():
            for table in doc.covers_tables:
                if table not in tables:
                    issues.append(
                        f"{doc.topic}: covers_tables names {table!r}, which is "
                        "not a semantic model source table"
                    )
            for metric in doc.covers_metrics:
                if metric not in metrics:
                    issues.append(
                        f"{doc.topic}: covers_metrics names {metric!r}, which "
                        "is not a semantic model metric"
                    )
        return issues


def _parse_doc(path: Path) -> KnowledgeDoc:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path.name} is missing YAML front-matter")
    try:
        _, front_matter, body = text.split("---", 2)
    except ValueError as error:
        raise ValueError(f"{path.name} front-matter is not closed") from error
    metadata = yaml.safe_load(front_matter) or {}
    if not isinstance(metadata, dict):
        raise TypeError(f"{path.name} front-matter must be a mapping")
    return KnowledgeDoc(
        topic=path.stem,
        title=str(metadata.get("title", "")),
        covers_tables=tuple(metadata.get("covers_tables", ())),
        covers_metrics=tuple(metadata.get("covers_metrics", ())),
        keywords=tuple(metadata.get("keywords", ())),
        body=body.strip(),
    )
