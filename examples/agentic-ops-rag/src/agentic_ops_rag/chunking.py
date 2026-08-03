"""Structure-aware chunking used by the workshop and its release exercise."""

from __future__ import annotations

import hashlib
import re

from aai_core.rag import RAGDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def structural_chunks(
    markdown: str,
    *,
    document_id: str,
    doc_uri: str,
    max_characters: int = 900,
) -> list[RAGDocument]:
    """Split at headings, then bound oversized sections without losing lineage."""

    if max_characters < 100:
        raise ValueError("max_characters must be at least 100")
    sections: list[tuple[tuple[str, ...], str]] = []
    heading_path: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append((tuple(heading_path), text))
        buffer.clear()

    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            del heading_path[level - 1 :]
            heading_path.append(match.group(2))
        buffer.append(line)
    flush()

    chunks: list[RAGDocument] = []
    for path, section in sections:
        for part_number, part in enumerate(_bounded_parts(section, max_characters), 1):
            seed = f"{document_id}:{'/'.join(path)}:{part_number}:{part}"
            chunk_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                RAGDocument(
                    document_id=f"{document_id}:{chunk_id}",
                    page_content=part,
                    doc_uri=doc_uri,
                    chunk_id=chunk_id,
                    metadata={
                        "heading_path": list(path),
                        "part_number": part_number,
                    },
                )
            )
    return chunks


def _bounded_parts(text: str, maximum: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph]
    parts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        proposed = f"{current}\n\n{paragraph}".strip()
        if current and len(proposed) > maximum:
            parts.append(current)
            current = paragraph
        else:
            current = proposed
    if current:
        parts.append(current)
    return parts or [text]
