"""Chunking is deterministic, overlapping, and covers the text — no Spark."""

from aai_core.rag import ChunkingProfile
from jobs.build_chunks import chunk_text

PROFILE = ChunkingProfile(
    name="test", version="1", chunk_size=10, chunk_overlap=4, parser="plain-text"
)


def test_chunks_cover_the_whole_text_with_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunk_text(text, PROFILE)

    assert chunks[0] == "abcdefghij"
    assert chunks[1][:4] == chunks[0][-4:]  # overlap preserved
    assert text.endswith(chunks[-1][-1])
    assert chunk_text(text, PROFILE) == chunks  # deterministic


def test_empty_text_produces_no_chunks():
    assert chunk_text("", PROFILE) == []
