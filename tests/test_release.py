import copy
import importlib.util
import json
import pickle
import zipfile
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from aai_core.deployment import ApplicationRelease
from aai_core.rag import ChunkingProfile, EmbeddingProfile

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validate_release", ROOT / "scripts" / "validate_release.py"
)
release_validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_validation)


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
    assert document["schema_version"] == "1"


def test_release_contract_is_strict_and_copies_input():
    model = {
        "logical_name": "general-chat",
        "configuration": {"stop": ["END"]},
    }
    application_release = ApplicationRelease(
        application="claims-agent",
        release="1.0.0",
        source_commit="abc123",
        core_sdk_version="0.3.0",
        model=model,
        prompt={},
        retrieval={},
        evaluation={},
        environment="dev",
    )

    model["logical_name"] = "changed-after-validation"
    assert application_release.model["logical_name"] == "general-chat"
    digest = application_release.digest
    configuration = application_release.model["configuration"]
    assert isinstance(configuration, Mapping)
    with pytest.raises(TypeError):
        configuration["stop"] = ["CHANGED"]
    assert application_release.digest == digest
    assert copy.deepcopy(application_release) == application_release
    assert application_release.model_copy(deep=True) == application_release
    assert pickle.loads(pickle.dumps(application_release)) == application_release

    with pytest.raises(ValidationError):
        ApplicationRelease(
            application="claims-agent",
            release="1.0.0",
            source_commit="abc123",
            core_sdk_version="0.3.0",
            model={},
            prompt={},
            retrieval={},
            evaluation={},
            environment="dev",
            invented_term="value",
        )


def test_embedding_compatibility_and_chunk_validation():
    profile = EmbeddingProfile("embedding", "foundry", "model", 1536, True, "1")
    profile.assert_compatible(
        EmbeddingProfile("other", "databricks", "model", 1536, True, "2")
    )
    chunking = ChunkingProfile("documents", "1", 800, 100, "markdown")
    assert chunking.chunk_overlap == 100


def _write_test_wheel(path: Path, version: str) -> None:
    dist_info = f"aai_core-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            (
                "Metadata-Version: 2.4\n"
                "Name: aai-core\n"
                f"Version: {version}\n"
                "Requires-Python: >=3.11,<3.13\n\n"
            ),
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            ("Wheel-Version: 1.0\n" "Root-Is-Purelib: true\n" "Tag: py3-none-any\n\n"),
        )


def test_release_wheel_validation_and_manifest(tmp_path):
    version = json.loads((ROOT / "compatibility.json").read_text())["sdk"]["version"]
    wheel = tmp_path / f"aai_core-{version}-py3-none-any.whl"
    _write_test_wheel(wheel, version)

    details = release_validation.validate_wheel(wheel, version)
    manifest = tmp_path / "release-manifest.json"
    compatibility = json.loads((ROOT / "compatibility.json").read_text())
    release_validation.write_manifest(
        manifest,
        version=version,
        commit="abc123",
        wheel=details,
        compatibility=compatibility,
    )

    document = json.loads(manifest.read_text())
    assert document["wheel"]["filename"] == wheel.name
    assert document["wheel"]["sha256"] == details["sha256"]
    assert document["source_commit"] == "abc123"


def test_release_wheel_rejects_wrong_version(tmp_path):
    wheel = tmp_path / "aai_core-9.9.9-py3-none-any.whl"
    _write_test_wheel(wheel, "9.9.9")

    with pytest.raises(ValueError, match="wheel version"):
        release_validation.validate_wheel(wheel, "0.3.0")


def test_release_requires_an_annotated_tag(monkeypatch):
    monkeypatch.setattr(
        release_validation,
        "git_output",
        lambda *arguments: (
            "abc123"
            if arguments == ("rev-parse", "HEAD")
            else (
                "commit"
                if arguments == ("cat-file", "-t", "refs/tags/v0.3.0")
                else "abc123"
            )
        ),
    )

    with pytest.raises(ValueError, match="annotated tag"):
        release_validation.validate_release_version("0.3.0", require_tag=True)
