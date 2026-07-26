"""Chunking pipeline: governed source table -> CDF-enabled chunk table.

The chunk table feeds the DELTA_SYNC vector search index
(resources/index.yml). Change Data Feed must stay enabled so the index can
sync incrementally. The chunking profile is an application release input —
change it deliberately and re-run the evaluation gate
(scripts/create_release.py records it).
"""

from __future__ import annotations

from aai_core.rag import ChunkingProfile

CHUNKING = ChunkingProfile(
    name="fixed-window",
    version="1",
    chunk_size=1000,
    chunk_overlap=200,
    parser="plain-text",
)


def chunk_text(text: str, profile: ChunkingProfile = CHUNKING) -> list[str]:
    """Deterministic fixed-size chunking with overlap (pure function)."""

    if not text:
        return []
    step = profile.chunk_size - profile.chunk_overlap
    return [
        text[start : start + profile.chunk_size] for start in range(0, len(text), step)
    ]


def main() -> None:
    from pyspark.sql import Row, SparkSession

    from app.config import CHUNK_TABLE, SOURCE_TABLE

    spark = SparkSession.builder.getOrCreate()
    documents = spark.table(SOURCE_TABLE).select("id", "content", "source_uri")

    rows = []
    for document in documents.collect():
        for index, chunk in enumerate(chunk_text(document.content or "")):
            rows.append(
                Row(
                    id=f"{document.id}-{index}",
                    content=chunk,
                    source_uri=document.source_uri,
                    chunk_id=f"chunk-{index}",
                )
            )
    (
        spark.createDataFrame(rows)
        .write.mode("overwrite")
        .option("delta.enableChangeDataFeed", "true")
        .saveAsTable(CHUNK_TABLE)
    )
    print({"source": SOURCE_TABLE, "chunks": len(rows), "table": CHUNK_TABLE})


if __name__ == "__main__":
    main()
