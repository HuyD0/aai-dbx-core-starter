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
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, StringType

    from app.config import CHUNK_TABLE, SOURCE_TABLE

    spark = SparkSession.builder.getOrCreate()
    # Chunking runs distributed on the executors — the corpus never
    # materializes on the driver, so table size is bounded by the cluster,
    # not driver memory.
    chunk_udf = F.udf(lambda text: chunk_text(text or ""), ArrayType(StringType()))
    chunks = (
        spark.table(SOURCE_TABLE)
        .select(
            F.col("id").alias("document_id"),
            "source_uri",
            F.posexplode(chunk_udf(F.col("content"))).alias("position", "content"),
        )
        .select(
            F.concat_ws("-", F.col("document_id"), F.col("position")).alias("id"),
            "content",
            "source_uri",
            F.concat(F.lit("chunk-"), F.col("position")).alias("chunk_id"),
        )
    )
    (
        chunks.write.mode("overwrite")
        .option("delta.enableChangeDataFeed", "true")
        .saveAsTable(CHUNK_TABLE)
    )
    written = spark.table(CHUNK_TABLE).count()
    print({"source": SOURCE_TABLE, "chunks": written, "table": CHUNK_TABLE})


if __name__ == "__main__":
    main()
