import copy
import hashlib
import importlib.util
import json
import pickle
import zipfile
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml
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
    profile = EmbeddingProfile("embedding", "azure_apim", "model", 1536, True, "1")
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


def test_generated_project_sdk_default_is_an_offline_verified_candidate():
    compatibility = json.loads((ROOT / "compatibility.json").read_text())
    generated = compatibility["sdk"]["generated_project_default"]

    issues = release_validation.generated_sdk_default_issues(
        compatibility,
        current_sdk_version=compatibility["sdk"]["version"],
        pinned_content_sha256=release_validation.sdk_content_sha256_at_commit(
            generated["source"]["ref"]
        ),
        pinned_sdk_version=release_validation.sdk_version_at_commit(
            generated["source"]["ref"]
        ),
    )

    assert not issues
    assert generated["status"] == "release-candidate"
    assert generated["source"]["kind"] == "git-commit"


def test_sdk_content_digest_command_uses_the_local_candidate_commit(capsys):
    compatibility = json.loads((ROOT / "compatibility.json").read_text())
    source = compatibility["sdk"]["generated_project_default"]["source"]

    result = release_validation.main(["--print-sdk-content-sha256", source["ref"]])

    assert result == 0
    assert capsys.readouterr().out.strip() == source["content_sha256"]


def test_release_validation_jobs_fetch_candidate_commit_history():
    """validate_release.py hashes the pinned release-candidate commit to confirm
    the digest in compatibility.json describes it. On a shallow checkout that
    lookup returns None and both cross-checks are skipped *silently*, so any job
    running it needs full history — deploy.yml's build job did not have it.
    """

    # Both entry points reach sdk_content_sha256_at_commit(): the script directly,
    # and cloud-verify.sh through the suite it runs.
    signals = ("validate_release.py", "cloud-verify.sh")
    checked = []
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        for job_name, job in workflow.get("jobs", {}).items():
            steps = job.get("steps", [])
            if not any(
                signal in (step.get("run") or "")
                for step in steps
                for signal in signals
            ):
                continue
            checkouts = [
                step
                for step in steps
                if str(step.get("uses", "")).startswith("actions/checkout@")
            ]
            assert checkouts, f"{path.name}:{job_name} validates without a checkout"
            for checkout in checkouts:
                assert (checkout.get("with") or {}).get("fetch-depth") == 0, (
                    f"{path.name}:{job_name} runs validate_release.py on a "
                    "shallow checkout, which skips the digest cross-checks"
                )
            checked.append(f"{path.name}:{job_name}")

    assert {
        "ci.yml:lint-test",
        "deploy.yml:build",
        "publish-sdk.yml:build",
    } <= set(checked), f"release-validation job detection regressed: {checked}"

    # python-311 runs the repository suite, which hashes the pinned commit through
    # test_release_validation_reports_the_pinned_candidate_digest.
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    checkout = ci["jobs"]["python-311"]["steps"][0]
    assert checkout["with"]["fetch-depth"] == 0


def test_certified_lock_divergence_is_reported():
    policy = {
        "resolution": {"certified_lock": "uv.lock"},
        "packages": {"mlflow": {"certified": "3.15.1"}},
    }
    lock = {"package": [{"name": "mlflow", "version": "3.16.0"}]}

    issues = release_validation.certified_lock_divergences(policy, lock)

    assert len(issues) == 1
    assert "mlflow" in issues[0]
    assert "3.16.0" in issues[0] and "3.15.1" in issues[0]


def test_certified_lock_split_resolution_counts_as_divergence():
    """A universal lock may resolve one package to several versions across
    environment markers; if any of them is uncertified, CI partially tests an
    uncertified version."""

    policy = {
        "resolution": {"certified_lock": "uv.lock"},
        "packages": {"numpy": {"certified": "2.4.1"}},
    }
    lock = {
        "package": [
            {"name": "numpy", "version": "2.4.1"},
            {"name": "numpy", "version": "2.5.1"},
        ]
    }

    assert release_validation.certified_lock_divergences(policy, lock)


def test_certified_lock_check_skips_template_only_packages():
    policy = {
        "resolution": {"certified_lock": "uv.lock"},
        "packages": {
            "mlflow": {"certified": "3.15.1"},
            "langgraph": {"certified": "1.2.9"},
        },
    }
    lock = {"package": [{"name": "mlflow", "version": "3.15.1"}]}

    assert release_validation.certified_lock_divergences(policy, lock) == []


def test_certified_lock_agrees_with_dependency_policy():
    policy = release_validation.load_toml(ROOT / "dependency-policy.toml")
    lock = release_validation.load_toml(ROOT / policy["resolution"]["certified_lock"])

    assert release_validation.certified_lock_divergences(policy, lock) == []


def test_dependency_canary_workflow_matches_release_acceptance_metadata():
    compatibility = json.loads((ROOT / "compatibility.json").read_text())
    canary = compatibility["release_acceptance"]["dependency_canary"]
    workflow = ROOT / ".github" / "workflows" / "dependency-canary.yml"

    assert release_validation.workflow_matrix_values(workflow, "python") == set(
        canary["python"]
    )
    assert release_validation.workflow_matrix_values(workflow, "resolution") == set(
        canary["resolutions"]
    )


def test_generated_project_candidate_requires_a_full_commit_and_matching_digest():
    compatibility = json.loads((ROOT / "compatibility.json").read_text())
    generated = compatibility["sdk"]["generated_project_default"]
    generated["source"]["ref"] = "main"
    generated["source"]["content_sha256"] = "0" * 64

    issues = release_validation.generated_sdk_default_issues(
        compatibility,
        current_sdk_version=compatibility["sdk"]["version"],
        pinned_content_sha256="1" * 64,
        pinned_sdk_version="9.9.9",
    )

    assert "release-candidate SDK ref must be a full commit SHA" in issues
    assert (
        "generated-project SDK content digest does not describe its pinned commit"
        in issues
    )
    assert (
        "generated-project SDK version does not match its pinned commit metadata"
        in issues
    )


def test_published_default_requires_the_exact_annotated_version_tag():
    compatibility = json.loads((ROOT / "compatibility.json").read_text())
    generated = compatibility["sdk"]["generated_project_default"]
    generated["status"] = "published"
    generated["source"] = {
        "kind": "git-tag",
        "ref": "v9.9.9",
        "content_sha256": "0" * 64,
        "annotated": False,
    }

    issues = release_validation.generated_sdk_default_issues(
        compatibility,
        current_sdk_version=compatibility["sdk"]["version"],
        pinned_content_sha256=None,
        pinned_sdk_version=None,
    )

    assert "published SDK ref must be the exact v<version> tag" in issues
    assert "published SDK source must declare an annotated tag" in issues
    assert (
        "the checkout cannot mark its own SDK version both unreleased and published"
        in issues
    )


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
