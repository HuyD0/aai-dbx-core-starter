"""Strict project settings loaded from the tracked study manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "project.yaml"


class SplitCounts(BaseModel):
    """Balanced record targets for every intent."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    train: int = Field(gt=0)
    validation: int = Field(gt=0)
    test: int = Field(gt=0)


class DatasetSettings(BaseModel):
    """Pinned Kaggle dataset identity and integrity evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    title: str
    owner: str
    ref: str
    version: int = Field(gt=0)
    url: str
    archive_name: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    csv_name: str
    csv_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str
    license_spdx: str
    accessed_on: str
    language: str
    split_seed: int
    per_intent: SplitCounts


class ModelSettings(BaseModel):
    """Exact local model revision required by the offline path."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str
    directory: str
    primary_weight: str
    primary_weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_weight_bytes: int = Field(gt=0)
    runtime_files: dict[str, str]

    @property
    def verified_runtime_files(self) -> dict[str, str]:
        """Return validated relative runtime files, including the primary weight."""

        if not self.runtime_files:
            raise ValueError("model.runtime_files must not be empty")
        for name, digest in self.runtime_files.items():
            if Path(name).name != name or name.startswith("."):
                raise ValueError(f"model runtime filename is unsafe: {name!r}")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"model runtime hash is invalid for {name!r}")
        if self.runtime_files.get(self.primary_weight) != self.primary_weight_sha256:
            raise ValueError("primary model weight must appear in model.runtime_files")
        return dict(self.runtime_files)


class TrackingSettings(BaseModel):
    """Repository-local MLflow configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    uri: str
    artifact_root: str
    experiment: str


class ProjectSettings(BaseModel):
    """Versioned offline-study contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str
    dataset: DatasetSettings
    model: ModelSettings
    tracking: TrackingSettings

    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw" / "bitext"

    @property
    def archive_path(self) -> Path:
        return self.raw_dir / self.dataset.archive_name

    @property
    def csv_path(self) -> Path:
        return self.raw_dir / self.dataset.csv_name

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed" / "bitext-v1"

    @property
    def model_dir(self) -> Path:
        return PROJECT_ROOT / self.model.directory

    @property
    def adapter_dir(self) -> Path:
        return PROJECT_ROOT / "artifacts" / "adapters" / "bitext-lora-v1"

    @property
    def preflight_adapter_dir(self) -> Path:
        return PROJECT_ROOT / "artifacts" / "adapters" / "preflight-smoke"

    @property
    def capstone_adapter_dir(self) -> Path:
        return PROJECT_ROOT / "artifacts" / "adapters" / "capstone-lora-v1"


def load_settings(path: Path = DEFAULT_CONFIG) -> ProjectSettings:
    """Load and strictly validate the tracked project settings."""

    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return ProjectSettings.model_validate(payload, strict=True)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading large assets in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
