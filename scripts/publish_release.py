"""Publish one immutable aai-core release to the Unity Catalog artifact volume.

This is the enforcement point for AGENTS.md section 4 rule 12 — releases are
immutable. It ran as inline workflow shell until it was extracted here, which
meant the one hard rule with the most expensive failure mode (silently replacing
a published wheel) had no test and, since the repository has never cut a tag, had
never executed at all.

The publication contract:

* A version whose manifest already exists is complete. Refuse it. The manifest is
  written last precisely so its presence means "nothing further to do".
* A partial release may be resumed, but only when every file already uploaded is
  byte-identical to what this run would upload. A differing file means two
  different builds claim one version; refuse rather than overwrite.
* Verify what the volume actually holds — read the wheel back and check it
  against its own checksum file — before writing the completion marker.

Every remote operation goes through a `Runner`, so the tests exercise the real
ordering and refusal logic against an in-memory volume rather than a workspace.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class ReleaseError(RuntimeError):
    """A publication was refused. The message is the operator-facing reason."""


class Runner(Protocol):
    """The subset of `databricks fs` this publication needs."""

    def exists(self, remote: str) -> bool:
        """Whether the remote path is present in the volume."""
        raise NotImplementedError

    def mkdir(self, remote: str) -> None:
        """Create the remote directory, succeeding if it already exists."""
        raise NotImplementedError

    def upload(self, local: Path, remote: str) -> None:
        """Copy a local file to the volume without overwriting."""
        raise NotImplementedError

    def download(self, remote: str, local: Path) -> None:
        """Copy a volume file to a local path, overwriting it."""
        raise NotImplementedError


class DatabricksRunner:
    """Real `databricks fs` calls. Authentication comes from the environment."""

    def __init__(self, executable: str = "databricks") -> None:
        self._executable = executable

    def _run(self, *arguments: str, check: bool = True) -> int:
        completed = subprocess.run(  # noqa: S603 - fixed executable, no shell
            [self._executable, "fs", *arguments],
            capture_output=True,
            text=True,
        )
        if check and completed.returncode != 0:
            raise ReleaseError(
                f"`databricks fs {' '.join(arguments)}` failed: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return completed.returncode

    def exists(self, remote: str) -> bool:
        return self._run("ls", remote, check=False) == 0

    def mkdir(self, remote: str) -> None:
        self._run("mkdir", remote)

    def upload(self, local: Path, remote: str) -> None:
        self._run("cp", str(local), remote)

    def download(self, remote: str, local: Path) -> None:
        self._run("cp", "--overwrite", remote, str(local))


def release_files(version: str) -> tuple[str, str, str]:
    """(wheel, checksum, manifest) names, in the order they must be published."""

    wheel = f"aai_core-{version}-py3-none-any.whl"
    return wheel, f"{wheel}.sha256", "release-manifest.json"


def destination(volume: str, version: str) -> str:
    if not volume:
        raise ReleaseError("SDK_ARTIFACT_VOLUME is not configured")
    return f"dbfs:{volume}/aai_core/{version}"


def _publish_or_resume(
    runner: Runner, source: Path, remote: str, scratch: Path
) -> None:
    """Upload one file, or accept an identical one already in the volume."""

    if not runner.exists(remote):
        runner.upload(source, remote)
        return
    readback = scratch / f"resume-{source.name}"
    runner.download(remote, readback)
    if not filecmp.cmp(source, readback, shallow=False):
        raise ReleaseError(
            f"partial release already contains a different {source.name}; "
            "refusing to overwrite a published artifact"
        )


def _verify_uploaded_checksum(
    runner: Runner, remote_dir: str, wheel: str, checksum: str, scratch: Path
) -> None:
    """Read both files back out of the volume and check one against the other."""

    verify_dir = scratch / "verify"
    verify_dir.mkdir(exist_ok=True)
    runner.download(f"{remote_dir}/{wheel}", verify_dir / wheel)
    runner.download(f"{remote_dir}/{checksum}", verify_dir / checksum)
    completed = subprocess.run(  # noqa: S603 - fixed executable, no shell
        ["sha256sum", "--check", checksum],
        cwd=verify_dir,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ReleaseError(
            "published wheel does not match its checksum: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def publish(
    runner: Runner,
    *,
    artifact_dir: Path,
    volume: str,
    version: str,
    scratch: Path | None = None,
) -> str:
    """Publish `version` from `artifact_dir`, or raise ReleaseError.

    Returns the destination the release was published to.
    """

    wheel, checksum, manifest = release_files(version)
    for name in (wheel, checksum, manifest):
        if not (artifact_dir / name).is_file():
            raise ReleaseError(f"{name} is missing from {artifact_dir}")

    remote_dir = destination(volume, version)
    runner.mkdir(remote_dir)

    # The manifest is the completion marker: present means published.
    if runner.exists(f"{remote_dir}/{manifest}"):
        raise ReleaseError(f"version {version} is complete; releases are immutable")

    owned_scratch = scratch is None
    working = Path(tempfile.mkdtemp()) if owned_scratch else scratch
    working.mkdir(parents=True, exist_ok=True)
    try:
        _publish_or_resume(
            runner, artifact_dir / wheel, f"{remote_dir}/{wheel}", working
        )
        _publish_or_resume(
            runner, artifact_dir / checksum, f"{remote_dir}/{checksum}", working
        )
        _verify_uploaded_checksum(runner, remote_dir, wheel, checksum, working)
        # Written last, and only after the volume's own copy verified.
        runner.upload(artifact_dir / manifest, f"{remote_dir}/{manifest}")
        readback = working / f"final-{manifest}"
        runner.download(f"{remote_dir}/{manifest}", readback)
        if not filecmp.cmp(artifact_dir / manifest, readback, shallow=False):
            raise ReleaseError("published manifest does not match the reviewed one")
    finally:
        if owned_scratch:
            shutil.rmtree(working, ignore_errors=True)
    return remote_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("dist"))
    parser.add_argument("--volume", required=True)
    parser.add_argument("--version", required=True)
    arguments = parser.parse_args(argv)
    try:
        published = publish(
            DatabricksRunner(),
            artifact_dir=arguments.artifact_dir,
            volume=arguments.volume,
            version=arguments.version,
        )
    except ReleaseError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    print(f"published {arguments.version} to {published}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
