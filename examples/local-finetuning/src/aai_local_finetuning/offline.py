"""Network denial and immutable-asset checks for plane-safe study."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import platform
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .data.manifests import DatasetIntegrityError, require_valid_manifest
from .settings import PROJECT_ROOT, ProjectSettings, sha256_file

OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DATASETS_OFFLINE": "1",
    "DO_NOT_TRACK": "1",
    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
    "UV_OFFLINE": "1",
    "UV_PYTHON_DOWNLOADS": "never",
}


class OfflineAssetError(RuntimeError):
    """Raised when a required local asset is absent or has changed."""


class AssetCheck(BaseModel):
    """One immutable readiness check."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    path: str
    ready: bool
    detail: str


class FlightManifest(BaseModel):
    """Evidence that the online preparation phase completed."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str
    python: str
    platform: str
    project_lock_sha256: str
    dataset_archive_sha256: str
    dataset_csv_sha256: str
    model_revision: str
    model_weight_sha256: str
    packages: dict[str, str]
    model_files: dict[str, str]
    processed_files: dict[str, str]
    preflight_adapter_files: dict[str, str]


def enable_offline_environment() -> None:
    """Set supported library controls before importing model/tracking packages."""

    os.environ.update(OFFLINE_ENVIRONMENT)
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"


@contextlib.contextmanager
def deny_network() -> Iterator[None]:
    """Fail every Python socket connection during an offline study operation."""

    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def blocked_connect(*_args: Any, **_kwargs: Any) -> None:
        raise OfflineAssetError("network access is blocked in offline study mode")

    socket.socket.connect = blocked_connect  # type: ignore[method-assign]
    socket.create_connection = blocked_connect  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.create_connection = original_create_connection  # type: ignore[assignment]


def _check_hash(name: str, path: Path, expected: str) -> AssetCheck:
    if not path.is_file():
        return AssetCheck(
            name=name,
            path=str(path),
            ready=False,
            detail="missing",
        )
    actual = sha256_file(path)
    return AssetCheck(
        name=name,
        path=str(path),
        ready=actual == expected,
        detail="sha256 verified" if actual == expected else f"sha256 was {actual}",
    )


def prepared_dataset_check(processed_dir: Path) -> AssetCheck:
    """Verify prepared artifacts, ordered split IDs, and content hashes."""

    manifest_path = processed_dir / "manifest.json"
    try:
        verification = require_valid_manifest(processed_dir)
    except DatasetIntegrityError as error:
        return AssetCheck(
            name="processed dataset integrity",
            path=str(manifest_path),
            ready=False,
            detail=str(error).replace("\n", "; "),
        )
    return AssetCheck(
        name="processed dataset integrity",
        path=str(manifest_path),
        ready=True,
        detail=f"verified {verification.checked_files} content-addressed files",
    )


def asset_checks(settings: ProjectSettings) -> list[AssetCheck]:
    """Check every asset required for data work and local model execution."""

    checks = [
        _check_hash(
            "Kaggle archive",
            settings.archive_path,
            settings.dataset.archive_sha256,
        ),
        _check_hash("Bitext CSV", settings.csv_path, settings.dataset.csv_sha256),
    ]
    checks.extend(
        _check_hash(
            f"MLX runtime {name}",
            settings.model_dir / name,
            expected,
        )
        for name, expected in settings.model.verified_runtime_files.items()
    )
    revision_path = settings.model_dir / "LOCAL_REVISION"
    revision_matches = (
        revision_path.is_file()
        and revision_path.read_text(encoding="utf-8").strip() == settings.model.revision
    )
    checks.append(
        AssetCheck(
            name="MLX model revision",
            path=str(revision_path),
            ready=revision_matches,
            detail="revision verified" if revision_matches else "missing or changed",
        )
    )
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl", "manifest.json"):
        path = settings.processed_dir / name
        checks.append(
            AssetCheck(
                name=f"processed {name}",
                path=str(path),
                ready=path.is_file() and path.stat().st_size > 0,
                detail="present" if path.is_file() else "missing",
            )
        )
    checks.append(prepared_dataset_check(settings.processed_dir))
    preflight_weights = settings.preflight_adapter_dir / "adapters.safetensors"
    checks.append(
        AssetCheck(
            name="MLX preflight adapter",
            path=str(preflight_weights),
            ready=preflight_weights.is_file() and preflight_weights.stat().st_size > 0,
            detail="present" if preflight_weights.is_file() else "missing",
        )
    )
    mlflow_database = PROJECT_ROOT / ".aai" / "mlflow.db"
    checks.append(
        AssetCheck(
            name="local MLflow store",
            path=str(mlflow_database),
            ready=mlflow_database.is_file() and mlflow_database.stat().st_size > 0,
            detail="present" if mlflow_database.is_file() else "missing",
        )
    )
    return checks


def require_assets(settings: ProjectSettings) -> list[AssetCheck]:
    """Raise once with a complete list instead of failing asset by asset."""

    checks = asset_checks(settings)
    missing = [check for check in checks if not check.ready]
    if missing:
        detail = "\n".join(
            f"- {item.name}: {item.path} ({item.detail})" for item in missing
        )
        raise OfflineAssetError(
            "offline assets are incomplete; run `make prepare-flight` while online:\n"
            + detail
        )
    return checks


def _current_flight_manifest(settings: ProjectSettings) -> FlightManifest:
    processed = {
        str(path.relative_to(settings.processed_dir)): sha256_file(path)
        for path in sorted(settings.processed_dir.rglob("*"))
        if path.is_file()
    }
    packages = {
        (distribution.metadata.get("Name") or "unknown").lower(): distribution.version
        for distribution in importlib.metadata.distributions()
    }
    model_files = {
        str(path.relative_to(settings.model_dir)): sha256_file(path)
        for path in sorted(settings.model_dir.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(settings.model_dir).parts
    }
    preflight_adapter_files = {
        str(path.relative_to(settings.preflight_adapter_dir)): sha256_file(path)
        for path in sorted(settings.preflight_adapter_dir.rglob("*"))
        if path.is_file()
    }
    return FlightManifest(
        schema_version="1.0.0",
        python=platform.python_version(),
        platform=f"{platform.system()}-{platform.machine()}",
        project_lock_sha256=sha256_file(PROJECT_ROOT / "uv.lock"),
        dataset_archive_sha256=sha256_file(settings.archive_path),
        dataset_csv_sha256=sha256_file(settings.csv_path),
        model_revision=settings.model.revision,
        model_weight_sha256=sha256_file(
            settings.model_dir / settings.model.primary_weight
        ),
        packages=packages,
        model_files=model_files,
        processed_files=processed,
        preflight_adapter_files=preflight_adapter_files,
    )


def write_flight_manifest(settings: ProjectSettings) -> Path:
    """Record hashes for the assets that will be used away from a network."""

    payload = _current_flight_manifest(settings)
    path = PROJECT_ROOT / "artifacts" / "flight-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def verify_flight_manifest(settings: ProjectSettings) -> FlightManifest:
    """Validate preparation evidence and all current file hashes."""

    path = PROJECT_ROOT / "artifacts" / "flight-manifest.json"
    if not path.is_file():
        raise OfflineAssetError(f"missing preparation manifest: {path}")
    manifest = FlightManifest.model_validate_json(path.read_text(encoding="utf-8"))
    current = _current_flight_manifest(settings)
    if manifest != current:
        raise OfflineAssetError(
            "local assets or the locked environment changed after flight preparation"
        )
    return manifest


def apple_silicon_status() -> AssetCheck:
    """Report whether real MLX execution is supported on this machine."""

    supported = platform.system() == "Darwin" and platform.machine() == "arm64"
    return AssetCheck(
        name="Apple silicon",
        path="local machine",
        ready=supported,
        detail=f"{platform.system()} {platform.machine()}",
    )


def prove_socket_denial() -> None:
    """Exercise the guard without contacting any external host."""

    try:
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
    except OfflineAssetError:
        return
    except OSError as exc:  # pragma: no cover - only possible without the guard
        raise OfflineAssetError("socket denial guard was not installed") from exc
    raise OfflineAssetError("socket denial guard allowed a connection")


def load_json(path: Path) -> Any:
    """Read a small local JSON artifact."""

    return json.loads(path.read_text(encoding="utf-8"))
