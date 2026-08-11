"""Chunking is deterministic, overlapping, and covers the text — no Spark."""

import pytest

from aai_core.rag import ChunkingProfile
from jobs.build_chunks import chunk_text, stable_chunk_id

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


def test_chunk_identity_is_stable_across_content_updates():
    first = stable_chunk_id("document-1", 2)

    assert stable_chunk_id("document-1", 2) == first
    assert stable_chunk_id("document-1", 3) != first
    assert stable_chunk_id("document-2", 2) != first


def test_chunk_identity_rejects_invalid_keys():
    with pytest.raises(ValueError):
        stable_chunk_id("", 0)
    with pytest.raises(ValueError):
        stable_chunk_id("document-1", -1)
