"""Chunking pipeline: governed source table -> idempotent Delta chunk sync."""

from __future__ import annotations

import hashlib

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


def stable_chunk_id(document_id: str, position: int) -> str:
    """Stable merge key: content changes update a chunk rather than duplicate it."""

    if not document_id.strip() or position < 0:
        raise ValueError(
            "chunk identity requires a document id and nonnegative position"
        )
    payload = f"{document_id}:{position}".encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    from delta.tables import DeltaTable
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from app.config import CHUNK_TABLE, SOURCE_TABLE

    spark = SparkSession.builder.getOrCreate()
    source = (
        spark.table(SOURCE_TABLE)
        .select(
            F.col("id").cast("string").alias("document_id"),
            F.col("source_uri").cast("string").alias("source_uri"),
            F.col("content").cast("string").alias("document_content"),
        )
        .where(F.col("document_id").isNotNull())
    )
    duplicate = (
        source.groupBy("document_id").count().where(F.col("count") > 1).limit(1).count()
    )
    if duplicate:
        raise ValueError("source table contains duplicate document ids")

    step = CHUNKING.chunk_size - CHUNKING.chunk_overlap
    chunks = (
        source.where(F.length("document_content") > 0)
        .withColumn(
            "_starts",
            F.sequence(
                F.lit(0),
                F.length("document_content") - F.lit(1),
                F.lit(step),
            ),
        )
        .select(
            "document_id",
            "source_uri",
            "document_content",
            F.posexplode("_starts").alias("position", "start"),
        )
        .select(
            F.sha2(
                F.concat_ws(":", "document_id", F.col("position").cast("string")),
                256,
            ).alias("id"),
            F.expr(
                f"substring(document_content, start + 1, {CHUNKING.chunk_size})"
            ).alias("content"),
            "source_uri",
            F.concat_ws(":", "document_id", F.col("position").cast("string")).alias(
                "chunk_id"
            ),
        )
    )

    if not spark.catalog.tableExists(CHUNK_TABLE):
        (
            chunks.write.format("delta")
            .option("delta.enableChangeDataFeed", "true")
            .saveAsTable(CHUNK_TABLE)
        )
    else:
        target = DeltaTable.forName(spark, CHUNK_TABLE)
        (
            target.alias("target")
            .merge(chunks.alias("source"), "target.id = source.id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .whenNotMatchedBySourceDelete()
            .execute()
        )
    written = spark.table(CHUNK_TABLE).count()
    print({"source": SOURCE_TABLE, "chunks": written, "table": CHUNK_TABLE})


if __name__ == "__main__":
    main()
