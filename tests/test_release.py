import copy
import hashlib
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


def release(**overrides):
    values = {
        "application": "claims-agent",
        "release": "1.0.0",
        "source_commit": "abc123",
        "core_sdk_version": "0.1.0",
        "model": {"logical_name": "general-chat", "deployment": "chat"},
        "prompt": {"name": "claims", "version": 3},
        "retrieval": {"index": "claims-v1"},
        "evaluation": {"dataset": "release-suite", "run_id": "run-1"},
        "environment": "dev",
    }
    values.update(overrides)
    return ApplicationRelease(
        **values,
    )


def test_release_digest_is_stable_and_written(tmp_path):
    first = release()
    second = release()
    destination = tmp_path / "release.json"

    first.write(destination)

    document = json.loads(destination.read_text())
    assert first.digest == second.digest
    assert document["digest"] == first.digest
    assert document["schema_version"] == "2"
    assert document["clock_digests"] == first.clock_digests


def test_v1_release_documents_keep_their_original_canonical_digest(tmp_path):
    legacy = release(schema_version="1")
    document = legacy.as_dict()
    expected = {
        "application": "claims-agent",
        "release": "1.0.0",
        "source_commit": "abc123",
        "core_sdk_version": "0.1.0",
        "model": {"logical_name": "general-chat", "deployment": "chat"},
        "prompt": {"name": "claims", "version": 3},
        "retrieval": {"index": "claims-v1"},
        "evaluation": {"dataset": "release-suite", "run_id": "run-1"},
        "environment": "dev",
        "schema_version": "1",
    }
    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()

    assert document == expected
    assert legacy.digest == hashlib.sha256(canonical).hexdigest()
    destination = tmp_path / "release-v1.json"
    legacy.write(destination)
    assert json.loads(destination.read_text()) == {
        **expected,
        "digest": legacy.digest,
    }
    with pytest.raises(ValidationError, match="schema version 1"):
        release(schema_version="1", world={"source_snapshot": "2026-08-08"})


def test_three_clock_digests_change_only_with_their_evidence_domains():
    baseline = release(
        world={"source_snapshot": "2026-08-08"},
        tools={"schema_digest": "tools-v1"},
        control={"manifest_digest": "manifest-v1"},
    )

    world_changed = release(
        world={"source_snapshot": "2026-08-09"},
        tools={"schema_digest": "tools-v1"},
        control={"manifest_digest": "manifest-v1"},
    )
    retrieval_changed = release(
        world={"source_snapshot": "2026-08-08"},
        retrieval={"index": "claims-v2"},
        tools={"schema_digest": "tools-v1"},
        control={"manifest_digest": "manifest-v1"},
    )
    tools_changed = release(
        world={"source_snapshot": "2026-08-08"},
        tools={"schema_digest": "tools-v2"},
        control={"manifest_digest": "manifest-v1"},
    )
    control_changed = release(
        world={"source_snapshot": "2026-08-08"},
        tools={"schema_digest": "tools-v1"},
        control={"manifest_digest": "manifest-v2"},
    )

    assert world_changed.world_digest != baseline.world_digest
    assert world_changed.learning_digest == baseline.learning_digest
    assert world_changed.control_digest == baseline.control_digest

    assert retrieval_changed.world_digest != baseline.world_digest
    assert retrieval_changed.learning_digest != baseline.learning_digest
    assert retrieval_changed.control_digest == baseline.control_digest

    assert tools_changed.world_digest == baseline.world_digest
    assert tools_changed.learning_digest != baseline.learning_digest
    assert tools_changed.control_digest == baseline.control_digest

    assert control_changed.world_digest == baseline.world_digest
    assert control_changed.learning_digest == baseline.learning_digest
    assert control_changed.control_digest != baseline.control_digest


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


@pytest.mark.parametrize("schema_version", ["1", "2"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("model", {"api_key": "do-not-persist"}),
        ("prompt", {"headers": {"authorization": "Bearer abcdefghijk"}}),
        (
            "control",
            {"provider_value": "github_pat_abcdefghijklmnopqrstuvwxyz"},
        ),
    ],
)
def test_release_rejects_secret_material_without_echoing_it(
    schema_version,
    field,
    value,
):
    overrides = {field: value, "schema_version": schema_version}
    if schema_version == "1" and field == "control":
        overrides = {
            "evaluation": value,
            "schema_version": schema_version,
        }

    with pytest.raises(ValidationError) as captured:
        release(**overrides)

    message = str(captured.value)
    assert "do-not-persist" not in message
    assert "Bearer abcdefghijk" not in message
    assert "github_pat_abcdefghijklmnopqrstuvwxyz" not in message


def test_release_allows_token_usage_and_secret_references():
    application_release = release(
        evaluation={"total_tokens": 42, "token_count": 42},
        model={"credential_ref": "keyvault://platform/model-auth"},
    )

    assert application_release.evaluation["total_tokens"] == 42
    assert application_release.model["credential_ref"].startswith("keyvault://")


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
    assert document["schema_version"] == 2
    identifiers = json.loads((ROOT / "platform-identifiers.json").read_text())
    assert document["sdk_artifact_volume"] == identifiers["sdk_artifact_volume"]
    assert (
        document["dependency_lock_sha256"]
        == hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    )


def test_release_wheel_rejects_wrong_version(tmp_path):
    wheel = tmp_path / "aai_core-9.9.9-py3-none-any.whl"
    _write_test_wheel(wheel, "9.9.9")

    with pytest.raises(ValueError, match="wheel version"):
        release_validation.validate_wheel(wheel, "0.3.0")


def test_current_sdk_version_has_a_changelog_release_section():
    version = json.loads((ROOT / "compatibility.json").read_text())["sdk"]["version"]

    commit = release_validation.validate_release_version(version, require_tag=False)

    assert commit == release_validation.git_output("rev-parse", "HEAD")


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
