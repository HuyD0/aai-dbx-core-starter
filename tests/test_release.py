import json

from aai_core.deployment import ApplicationRelease
from aai_core.rag import ChunkingProfile, EmbeddingProfile


def release():
    return ApplicationRelease(
        application="claims-agent",
        release="1.0.0",
        source_commit="abc123",
        core_sdk_version="0.1.0",
        model={"logical_name": "general-chat", "deployment": "chat"},
        prompt={"name": "claims", "version": 3},
        retrieval={"index": "claims-v1"},
        evaluation={"dataset": "release-suite", "run_id": "run-1"},
        environment="dev",
    )


def test_release_digest_is_stable_and_written(tmp_path):
    first = release()
    second = release()
    destination = tmp_path / "release.json"

    first.write(destination)

    document = json.loads(destination.read_text())
    assert first.digest == second.digest
    assert document["digest"] == first.digest


def test_embedding_compatibility_and_chunk_validation():
    profile = EmbeddingProfile("embedding", "foundry", "model", 1536, True, "1")
    profile.assert_compatible(
        EmbeddingProfile("other", "databricks", "model", 1536, True, "2")
    )
    chunking = ChunkingProfile("documents", "1", 800, 100, "markdown")
    assert chunking.chunk_overlap == 100
