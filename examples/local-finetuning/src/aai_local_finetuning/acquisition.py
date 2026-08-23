"""Online-only acquisition of pinned public data and model assets."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from .settings import ProjectSettings, sha256_file


class AcquisitionError(RuntimeError):
    """Raised when an online source does not match the reviewed manifest."""


def _already_verified(path: Path, expected_sha256: str) -> bool:
    return path.is_file() and sha256_file(path) == expected_sha256


def acquire_bitext(settings: ProjectSettings) -> tuple[Path, Path]:
    """Download through the current Kaggle CLI and extract one pinned CSV."""

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    if not _already_verified(settings.archive_path, settings.dataset.archive_sha256):
        kaggle = Path(sys.executable).with_name("kaggle")
        if not kaggle.is_file():
            raise AcquisitionError(
                "the locked Kaggle CLI is absent; run `uv sync --all-extras --locked`"
            )
        with tempfile.TemporaryDirectory(prefix="aai-kaggle-") as temporary:
            download_dir = Path(temporary)
            command = [
                str(kaggle),
                "datasets",
                "download",
                settings.dataset.ref,
                "--path",
                str(download_dir),
                "--quiet",
            ]
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode:
                raise AcquisitionError(
                    "Kaggle download failed. The public dataset currently supports "
                    "anonymous download; if that changes, authenticate with a current "
                    "Kaggle mechanism outside this project and retry.\n"
                    f"CLI output: {result.stdout.strip()}"
                )
            archives = sorted(download_dir.glob("*.zip"))
            if len(archives) != 1:
                raise AcquisitionError(
                    "expected one Kaggle archive, found "
                    f"{[path.name for path in archives]}"
                )
            archive = archives[0]
            actual = sha256_file(archive)
            if actual != settings.dataset.archive_sha256:
                raise AcquisitionError(
                    "the Kaggle archive changed; do not silently update frozen study "
                    f"data (expected {settings.dataset.archive_sha256}, got {actual})"
                )
            shutil.copy2(archive, settings.archive_path)

    if not _already_verified(settings.csv_path, settings.dataset.csv_sha256):
        with zipfile.ZipFile(settings.archive_path) as archive:
            names = archive.namelist()
            if names != [settings.dataset.csv_name]:
                raise AcquisitionError(
                    f"archive members changed; expected one CSV, found {names}"
                )
            with archive.open(settings.dataset.csv_name) as source:
                with tempfile.NamedTemporaryFile(
                    dir=settings.raw_dir,
                    prefix="bitext-",
                    suffix=".csv",
                    delete=False,
                ) as destination:
                    shutil.copyfileobj(source, destination)
                    temporary_csv = Path(destination.name)
        actual = sha256_file(temporary_csv)
        if actual != settings.dataset.csv_sha256:
            temporary_csv.unlink(missing_ok=True)
            raise AcquisitionError(
                "extracted CSV changed "
                f"(expected {settings.dataset.csv_sha256}, got {actual})"
            )
        temporary_csv.replace(settings.csv_path)
    return settings.archive_path, settings.csv_path


def acquire_model(settings: ProjectSettings) -> Path:
    """Materialize one exact Hugging Face commit into a real local directory."""

    weight_path = settings.model_dir / settings.model.primary_weight
    revision_file = settings.model_dir / "LOCAL_REVISION"
    runtime_files = settings.model.verified_runtime_files
    all_runtime_files_verified = all(
        _already_verified(settings.model_dir / name, expected)
        for name, expected in runtime_files.items()
    )
    if all_runtime_files_verified:
        revision_file.write_text(settings.model.revision + "\n", encoding="utf-8")
        return settings.model_dir

    from huggingface_hub import snapshot_download

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=settings.model.repo,
        revision=settings.model.revision,
        local_dir=settings.model_dir,
        local_files_only=False,
        force_download=True,
    )
    mismatches = [
        name
        for name, expected in runtime_files.items()
        if not _already_verified(settings.model_dir / name, expected)
    ]
    if mismatches:
        raise AcquisitionError(
            "downloaded model runtime files did not match the pinned revision: "
            + ", ".join(mismatches)
        )
    if weight_path.stat().st_size != settings.model.primary_weight_bytes:
        raise AcquisitionError(
            "model weight size changed: "
            f"expected {settings.model.primary_weight_bytes}, "
            f"got {weight_path.stat().st_size}"
        )
    actual = sha256_file(weight_path)
    if actual != settings.model.primary_weight_sha256:
        raise AcquisitionError(
            "model weights changed: "
            f"expected {settings.model.primary_weight_sha256}, got {actual}"
        )
    revision_file.write_text(settings.model.revision + "\n", encoding="utf-8")
    return settings.model_dir
