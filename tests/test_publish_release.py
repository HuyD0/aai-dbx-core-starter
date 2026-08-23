"""Release immutability (AGENTS.md section 4 rule 12).

The publication logic used to be inline workflow shell: never unit-tested, and
never executed, because the repository has not cut a release tag. These tests
drive the extracted implementation against an in-memory volume so the refusal
paths are exercised before a real credentialed run depends on them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "publish_release", ROOT / "scripts" / "publish_release.py"
)
publish_release = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(publish_release)

VERSION = "9.9.9"
VOLUME = "/Volumes/catalog/schema/volume"


class FakeVolume:
    """An in-memory `databricks fs`, recording the order of every operation."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.operations: list[tuple[str, str]] = []

    def exists(self, remote: str) -> bool:
        self.operations.append(("exists", remote))
        return remote in self.files

    def mkdir(self, remote: str) -> None:
        self.operations.append(("mkdir", remote))

    def upload(self, local: Path, remote: str) -> None:
        self.operations.append(("upload", remote))
        self.files[remote] = local.read_bytes()

    def download(self, remote: str, local: Path) -> None:
        self.operations.append(("download", remote))
        if remote not in self.files:
            raise publish_release.ReleaseError(f"missing remote file {remote}")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self.files[remote])

    def uploaded(self) -> list[str]:
        return [
            remote for operation, remote in self.operations if operation == "upload"
        ]


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    """A complete, self-consistent release directory."""

    wheel, checksum, manifest = publish_release.release_files(VERSION)
    directory = tmp_path / "dist"
    directory.mkdir()
    payload = b"wheel bytes for " + VERSION.encode()
    (directory / wheel).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (directory / checksum).write_text(f"{digest}  {wheel}\n", encoding="utf-8")
    (directory / manifest).write_text('{"version": "9.9.9"}\n', encoding="utf-8")
    return directory


def _publish(volume: FakeVolume, artifacts: Path, scratch: Path) -> str:
    return publish_release.publish(
        volume,
        artifact_dir=artifacts,
        volume=VOLUME,
        version=VERSION,
        scratch=scratch,
    )


def test_publishes_the_manifest_last_so_it_marks_completion(
    artifacts: Path, tmp_path: Path
) -> None:
    volume = FakeVolume()
    destination = _publish(volume, artifacts, tmp_path / "scratch1")

    wheel, checksum, manifest = publish_release.release_files(VERSION)
    assert destination == f"dbfs:{VOLUME}/aai_core/{VERSION}"
    assert volume.uploaded() == [
        f"{destination}/{wheel}",
        f"{destination}/{checksum}",
        f"{destination}/{manifest}",
    ], "the completion marker must be written after the artifacts it marks"


def test_refuses_a_version_that_is_already_complete(
    artifacts: Path, tmp_path: Path
) -> None:
    volume = FakeVolume()
    _publish(volume, artifacts, tmp_path / "scratch1")
    before = dict(volume.files)

    with pytest.raises(publish_release.ReleaseError, match="releases are immutable"):
        _publish(volume, artifacts, tmp_path / "scratch2")
    assert volume.files == before, "a refused republication must change nothing"


def test_resumes_a_partial_release_whose_files_are_identical(
    artifacts: Path, tmp_path: Path
) -> None:
    wheel, checksum, manifest = publish_release.release_files(VERSION)
    volume = FakeVolume()
    destination = f"dbfs:{VOLUME}/aai_core/{VERSION}"
    # Interrupted after the wheel, before the checksum and the marker.
    volume.files[f"{destination}/{wheel}"] = (artifacts / wheel).read_bytes()

    _publish(volume, artifacts, tmp_path / "scratch")

    assert volume.uploaded() == [
        f"{destination}/{checksum}",
        f"{destination}/{manifest}",
    ], "an identical file must be accepted, not re-uploaded"
    assert set(volume.files) == {
        f"{destination}/{wheel}",
        f"{destination}/{checksum}",
        f"{destination}/{manifest}",
    }


def test_refuses_a_partial_release_containing_a_different_build(
    artifacts: Path, tmp_path: Path
) -> None:
    wheel, _, manifest = publish_release.release_files(VERSION)
    volume = FakeVolume()
    destination = f"dbfs:{VOLUME}/aai_core/{VERSION}"
    volume.files[f"{destination}/{wheel}"] = b"a different build of the same version"

    with pytest.raises(publish_release.ReleaseError, match="refusing to overwrite"):
        _publish(volume, artifacts, tmp_path / "scratch")

    assert volume.files[f"{destination}/{wheel}"] == (
        b"a different build of the same version"
    ), "the published artifact must be left untouched"
    assert f"{destination}/{manifest}" not in volume.files


def test_refuses_when_the_volume_copy_fails_its_own_checksum(
    artifacts: Path, tmp_path: Path
) -> None:
    """Verification reads back what the volume holds, not what we meant to send."""

    wheel, _, manifest = publish_release.release_files(VERSION)
    destination = f"dbfs:{VOLUME}/aai_core/{VERSION}"

    class CorruptingVolume(FakeVolume):
        def upload(self, local: Path, remote: str) -> None:
            super().upload(local, remote)
            if remote.endswith(wheel):
                self.files[remote] = b"corrupted in transit"

    volume = CorruptingVolume()
    with pytest.raises(
        publish_release.ReleaseError, match="does not match its checksum"
    ):
        _publish(volume, artifacts, tmp_path / "scratch")
    assert f"{destination}/{manifest}" not in volume.files


@pytest.mark.parametrize("missing", publish_release.release_files(VERSION))
def test_refuses_an_incomplete_artifact_directory(
    artifacts: Path, tmp_path: Path, missing: str
) -> None:
    (artifacts / missing).unlink()
    volume = FakeVolume()
    with pytest.raises(publish_release.ReleaseError, match=f"{missing} is missing"):
        _publish(volume, artifacts, tmp_path / "scratch")
    assert not volume.uploaded()


def test_requires_a_configured_volume() -> None:
    with pytest.raises(publish_release.ReleaseError, match="is not configured"):
        publish_release.destination("", VERSION)


def test_workflow_delegates_publication_to_this_script() -> None:
    """The rule is only enforced where the release actually runs."""

    workflow = (ROOT / ".github/workflows/publish-sdk.yml").read_text(encoding="utf-8")
    assert "scripts/publish_release.py" in workflow
    for reimplemented in ("publish_or_resume", "databricks fs cp"):
        assert (
            reimplemented not in workflow
        ), "publication logic belongs in the tested script, not in workflow shell"


def test_uses_no_shell_and_a_fixed_executable() -> None:
    source = (ROOT / "scripts" / "publish_release.py").read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert shutil.which("sha256sum"), "verification depends on sha256sum"
