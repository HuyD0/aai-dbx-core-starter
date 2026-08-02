"""Safe wrapper around the pinned MLX-LM LoRA command."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .settings import PROJECT_ROOT, load_settings

TRAINING_MANIFEST_NAME = "training-manifest.json"
_MODEL_REVISION_FILE = "LOCAL_REVISION"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_REVISION_PATTERN = r"^[0-9a-f]{40}$"


class TrainingManifestError(RuntimeError):
    """A trained adapter does not match its successful-run evidence."""


class ExpectedFileHash(BaseModel):
    """One trusted model-runtime hash supplied independently of training YAML."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _require_safe_path(self) -> ExpectedFileHash:
        _require_safe_relative_path(self.path, "model runtime file")
        if PurePosixPath(self.path).name != self.path:
            raise ValueError("model runtime files must be top-level filenames")
        return self


class TrainingInputContract(BaseModel):
    """Trusted model and task-data expectations outside the training YAML."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model_path: str = Field(min_length=1)
    model_revision: str = Field(pattern=_REVISION_PATTERN)
    model_runtime_files: tuple[ExpectedFileHash, ...] = Field(min_length=1)
    data_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def _require_canonical_contract(self) -> TrainingInputContract:
        _require_safe_relative_path(self.model_path, "expected model path")
        _require_safe_relative_path(self.data_path, "expected data path")
        paths = tuple(item.path for item in self.model_runtime_files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError(
                "model_runtime_files must have unique paths in sorted order"
            )
        return self


class TrainingFileEvidence(BaseModel):
    """Content and size evidence for one model or dataset file."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _require_safe_path(self) -> TrainingFileEvidence:
        _require_safe_relative_path(self.path, "bound training input")
        return self


class TrainingManifest(BaseModel):
    """Immutable binding between adapter bytes and effective training inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["2.0.0"] = "2.0.0"
    adapter_path: str = Field(min_length=1)
    adapter_sha256: str = Field(pattern=_SHA256_PATTERN)
    adapter_size_bytes: int = Field(gt=0)
    adapter_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    adapter_config_size_bytes: int = Field(gt=0)
    source_config_path: str = Field(min_length=1)
    source_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_config_size_bytes: int = Field(gt=0)
    effective_config: dict[str, JsonValue]
    effective_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_inputs_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_path: str = Field(min_length=1)
    model_revision: str = Field(pattern=_REVISION_PATTERN)
    model_files: tuple[TrainingFileEvidence, ...] = Field(min_length=1)
    data_path: str = Field(min_length=1)
    data_files: tuple[TrainingFileEvidence, ...] = Field(min_length=1)
    data_manifest_path: str = Field(min_length=1)
    data_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _require_canonical_file_evidence(self) -> TrainingManifest:
        _require_safe_relative_path(self.adapter_path, "adapter path")
        _require_safe_relative_path(self.source_config_path, "source config path")
        _require_safe_relative_path(self.model_path, "model path")
        _require_safe_relative_path(self.data_path, "data path")
        _require_safe_relative_path(self.data_manifest_path, "data manifest path")
        for label, evidence in (
            ("model_files", self.model_files),
            ("data_files", self.data_files),
        ):
            paths = tuple(item.path for item in evidence)
            if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
                raise ValueError(f"{label} must have unique paths in sorted order")
        return self


class TrainingEvidence(BaseModel):
    """Small persisted summary parsed from an MLX-LM training run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    command: list[str]
    return_code: Literal[0]
    iterations: int = Field(gt=0)
    train_losses: list[float]
    validation_losses: list[float]
    peak_memory_gb: float | None
    log_path: str
    training_manifest_path: str
    training_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    adapter_sha256: str = Field(pattern=_SHA256_PATTERN)
    effective_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_inputs_sha256: str = Field(pattern=_SHA256_PATTERN)


@dataclass(frozen=True, slots=True)
class ValidatedTrainingSnapshot:
    """Exact immutable manifest snapshot validated against trusted expectations."""

    manifest: TrainingManifest
    raw_manifest_bytes: bytes
    manifest_sha256: str
    adapter_path: Path
    config_path: Path
    expected_iterations: int | None
    expected_adapter_path: Path | None
    expected_inputs: TrainingInputContract


@dataclass(frozen=True, slots=True)
class _CapturedFile:
    evidence: TrainingFileEvidence
    device: int
    inode: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _TrainingPlan:
    config_path: Path
    config_file: _CapturedFile
    source_config_path: str
    target_adapter_path: Path
    target_adapter_relative_path: str
    iterations: int
    effective_config: dict[str, JsonValue]
    effective_config_sha256: str
    expected_inputs: TrainingInputContract
    expected_inputs_sha256: str
    model_path: Path
    model_files: tuple[_CapturedFile, ...]
    data_path: Path
    data_files: tuple[_CapturedFile, ...]
    data_manifest_relative_path: str
    data_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _PriorSuccessEvidence:
    manifest_bytes: bytes | None
    evidence_bytes: bytes | None


@dataclass(frozen=True, slots=True)
class _AdapterPromotion:
    target: Path
    backup: Path | None


@dataclass(frozen=True, slots=True)
class _LocalLockToken:
    kind: Literal["reader", "writer"]


class _ProcessLockState:
    """A fair-enough reader/writer lock for threads sharing one process."""

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._condition = threading.Condition()
        self._readers: dict[int, int] = {}
        self._writer_thread: int | None = None
        self._writer_depth = 0
        self._waiting_writers = 0
        self._acquiring_file_lock = False
        self._descriptor: int | None = None
        self._flock_module = None

    def acquire(self, *, exclusive: bool) -> _LocalLockToken:
        thread_id = threading.get_ident()
        with self._condition:
            if self._writer_thread == thread_id:
                self._writer_depth += 1
                return _LocalLockToken(kind="writer")
            reader_depth = self._readers.get(thread_id, 0)
            if reader_depth:
                if exclusive:
                    raise RuntimeError(
                        "cannot upgrade a shared adapter lock to exclusive"
                    )
                self._readers[thread_id] = reader_depth + 1
                return _LocalLockToken(kind="reader")
            if exclusive:
                self._waiting_writers += 1
                try:
                    self._condition.wait_for(
                        lambda: (
                            self._writer_thread is None
                            and not self._readers
                            and not self._acquiring_file_lock
                        )
                    )
                    self._writer_thread = thread_id
                    self._writer_depth = 1
                    self._acquiring_file_lock = True
                finally:
                    self._waiting_writers -= 1
                token = _LocalLockToken(kind="writer")
            else:
                self._condition.wait_for(
                    lambda: (
                        self._writer_thread is None
                        and not self._waiting_writers
                        and not self._acquiring_file_lock
                    )
                )
                if self._readers:
                    self._readers[thread_id] = 1
                    return _LocalLockToken(kind="reader")
                self._readers[thread_id] = 1
                self._acquiring_file_lock = True
                token = _LocalLockToken(kind="reader")

        try:
            descriptor, flock_module = _acquire_file_lock(
                self._lock_path,
                exclusive=exclusive,
            )
        except BaseException:
            with self._condition:
                self._acquiring_file_lock = False
                if token.kind == "writer":
                    self._writer_thread = None
                    self._writer_depth = 0
                else:
                    self._readers.pop(thread_id, None)
                self._condition.notify_all()
            raise
        with self._condition:
            self._descriptor = descriptor
            self._flock_module = flock_module
            self._acquiring_file_lock = False
            self._condition.notify_all()
        return token

    def release(self, token: _LocalLockToken) -> None:
        thread_id = threading.get_ident()
        release_error: BaseException | None = None
        with self._condition:
            if token.kind == "writer":
                if self._writer_thread != thread_id or self._writer_depth < 1:
                    raise RuntimeError("exclusive adapter lock ownership was lost")
                self._writer_depth -= 1
                if self._writer_depth == 0:
                    try:
                        self._release_file_lock()
                    except BaseException as error:
                        release_error = error
                    finally:
                        self._writer_thread = None
                        self._condition.notify_all()
            else:
                reader_depth = self._readers.get(thread_id, 0)
                if reader_depth < 1:
                    raise RuntimeError("shared adapter lock ownership was lost")
                if reader_depth == 1:
                    del self._readers[thread_id]
                    if not self._readers:
                        try:
                            self._release_file_lock()
                        except BaseException as error:
                            release_error = error
                        finally:
                            self._condition.notify_all()
                else:
                    self._readers[thread_id] = reader_depth - 1
        if release_error is not None:
            raise release_error

    def _release_file_lock(self) -> None:
        descriptor = self._descriptor
        flock_module = self._flock_module
        self._descriptor = None
        self._flock_module = None
        if descriptor is None or flock_module is None:
            raise RuntimeError("adapter file lock ownership was lost")
        try:
            flock_module.flock(descriptor, flock_module.LOCK_UN)
        finally:
            os.close(descriptor)

    def close_in_child_after_fork(self) -> None:
        descriptor = self._descriptor
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


_PROCESS_LOCK_REGISTRY_GUARD = threading.Lock()
_PROCESS_LOCK_REGISTRY: dict[Path, _ProcessLockState] = {}


def _reset_process_lock_registry() -> None:
    global _PROCESS_LOCK_REGISTRY_GUARD
    global _PROCESS_LOCK_REGISTRY

    for state in _PROCESS_LOCK_REGISTRY.values():
        state.close_in_child_after_fork()
    _PROCESS_LOCK_REGISTRY_GUARD = threading.Lock()
    _PROCESS_LOCK_REGISTRY = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_process_lock_registry)


@contextmanager
def shared_adapter_lock(adapter_path: Path) -> Iterator[None]:
    """Hold a shared process lock across adapter inference and evidence writes."""

    with _adapter_process_lock(adapter_path, exclusive=False):
        yield


@contextmanager
def exclusive_adapter_lock(adapter_path: Path) -> Iterator[None]:
    """Hold an exclusive process lock while replacing or removing an adapter."""

    with _adapter_process_lock(adapter_path, exclusive=True):
        yield


@contextmanager
def _exclusive_adapter_lock(adapter_path: Path) -> Iterator[None]:
    """Compatibility alias for the original private lock helper."""

    with exclusive_adapter_lock(adapter_path):
        yield


@contextmanager
def _adapter_process_lock(
    adapter_path: Path,
    *,
    exclusive: bool,
) -> Iterator[None]:
    resolved_adapter = _project_path(adapter_path, "adapter path")
    lock_path = _adapter_lock_path(resolved_adapter)
    state = _process_lock_state(lock_path)
    token = state.acquire(exclusive=exclusive)
    try:
        yield
    finally:
        state.release(token)


def _adapter_lock_path(adapter_path: Path) -> Path:
    """Return a stable sibling lock path outside the replaceable directory."""

    resolved_adapter = _project_path(adapter_path, "adapter path")
    return resolved_adapter.parent / f".{resolved_adapter.name}.lock"


def _process_lock_state(lock_path: Path) -> _ProcessLockState:
    with _PROCESS_LOCK_REGISTRY_GUARD:
        state = _PROCESS_LOCK_REGISTRY.get(lock_path)
        if state is None:
            state = _ProcessLockState(lock_path)
            _PROCESS_LOCK_REGISTRY[lock_path] = state
        return state


def _acquire_file_lock(lock_path: Path, *, exclusive: bool):
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        raise RuntimeError(
            "adapter process locking requires fcntl on supported macOS or Linux"
        )
    try:
        import fcntl
    except ImportError as error:
        raise RuntimeError(
            "adapter process locking requires the fcntl standard library module"
        ) from error

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        opened = os.fstat(descriptor)
        linked = os.lstat(lock_path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(linked.st_mode)
            or opened.st_dev != linked.st_dev
            or opened.st_ino != linked.st_ino
        ):
            raise RuntimeError(f"adapter lock path is unsafe: {lock_path}")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(descriptor, operation)
        return descriptor, fcntl
    except BaseException as error:
        if descriptor is not None:
            os.close(descriptor)
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(f"could not acquire adapter lock: {lock_path}") from error


def require_apple_silicon() -> None:
    """Fail before importing MLX on an unsupported platform."""

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX-LM training requires macOS on Apple silicon")


def run_lora(
    *,
    iterations: int | None = None,
    config_path: Path | None = None,
    adapter_path: Path | None = None,
    log_name: str = "latest",
    expected_inputs: TrainingInputContract | None = None,
) -> TrainingEvidence:
    """Train into staging and transactionally publish success-bound evidence."""

    require_apple_silicon()
    config = config_path or PROJECT_ROOT / "configs" / "training" / "lora.yaml"
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", log_name):
        raise ValueError("log_name must contain lowercase letters, numbers, _ or -")
    plan = _build_training_plan(
        config,
        iterations=iterations,
        adapter_path=adapter_path,
        expected_inputs=expected_inputs,
    )
    with _exclusive_adapter_lock(plan.target_adapter_path):
        return _run_lora_locked(
            plan,
            iterations_overridden=iterations is not None,
            adapter_path_overridden=adapter_path is not None,
            log_name=log_name,
        )


def _run_lora_locked(
    plan: _TrainingPlan,
    *,
    iterations_overridden: bool,
    adapter_path_overridden: bool,
    log_name: str,
) -> TrainingEvidence:
    log_dir = PROJECT_ROOT / "artifacts" / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}.log"
    evidence_path = log_dir / f"{log_name}.json"
    plan.target_adapter_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            dir=plan.target_adapter_path.parent,
            prefix=f".{plan.target_adapter_path.name}.training-",
        )
    )
    try:
        command = [
            sys.executable,
            "-m",
            "mlx_lm",
            "lora",
            "--config",
            str(plan.config_path),
        ]
        recorded_command = [
            "<python>",
            "-m",
            "mlx_lm",
            "lora",
            "--config",
            plan.source_config_path,
        ]
        if iterations_overridden:
            command.extend(["--iters", str(plan.iterations)])
            recorded_command.extend(["--iters", str(plan.iterations)])
        command.extend(["--adapter-path", str(staging_path)])
        recorded_command.extend(["--adapter-path", plan.target_adapter_relative_path])
        if adapter_path_overridden:
            command.extend(["--save-every", str(plan.iterations)])
            recorded_command.extend(["--save-every", str(plan.iterations)])

        prior = _invalidate_success_evidence(
            plan.target_adapter_path,
            evidence_path,
        )
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = result.stdout or ""
        _write_text_atomic(log_path, output)
        if result.returncode:
            raise RuntimeError(f"MLX-LM training failed; see {log_path}")

        adapter_file, adapter_config_file = _capture_adapter_outputs(staging_path)
        _require_unchanged_training_inputs(plan)
        manifest = _training_manifest(
            plan,
            adapter_file=adapter_file,
            adapter_config_file=adapter_config_file,
        )
        raw_manifest = _model_json_bytes(manifest)
        manifest_path = staging_path / TRAINING_MANIFEST_NAME
        _write_bytes_atomic(manifest_path, raw_manifest)
        _require_unchanged_adapter_outputs(
            staging_path,
            adapter_file=adapter_file,
            adapter_config_file=adapter_config_file,
        )

        manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
        evidence = _training_evidence(
            plan,
            command=recorded_command,
            output=output,
            log_path=log_path,
            manifest_sha256=manifest_sha256,
            adapter_sha256=adapter_file.evidence.sha256,
        )
        promotion = _promote_adapter_directory(
            staging_path,
            plan.target_adapter_path,
        )
        try:
            _write_json_atomic(evidence_path, evidence)
        except BaseException as error:
            try:
                _rollback_published_success(
                    promotion,
                    prior=prior,
                    evidence_path=evidence_path,
                )
            except Exception as rollback_error:
                raise RuntimeError(
                    "training evidence publication failed; the new success manifest "
                    "was invalidated but prior-state restoration also failed"
                ) from rollback_error
            raise error
        _commit_adapter_promotion(promotion)
        return evidence
    finally:
        if staging_path.exists() or staging_path.is_symlink():
            _remove_path(staging_path)


def require_valid_training_snapshot(
    adapter_path: Path,
    *,
    config_path: Path | None = None,
    expected_iterations: int | None = None,
    expected_adapter_path: Path | None = None,
    expected_inputs: TrainingInputContract | None = None,
) -> ValidatedTrainingSnapshot:
    """Validate current evidence and retain its exact bytes for later rechecks."""

    try:
        with shared_adapter_lock(adapter_path):
            return _require_valid_training_snapshot(
                adapter_path,
                config_path=config_path,
                expected_iterations=expected_iterations,
                expected_adapter_path=expected_adapter_path,
                expected_inputs=expected_inputs,
            )
    except TrainingManifestError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingManifestError(
            "trained adapter evidence is stale or mismatched "
            f"({type(error).__name__}: {error})"
        ) from error


def verify_training_manifest(
    adapter_path: Path,
    *,
    config_path: Path | None = None,
    expected_iterations: int | None = None,
    expected_adapter_path: Path | None = None,
    expected_inputs: TrainingInputContract | None = None,
) -> TrainingManifest:
    """Compatibility wrapper returning the parsed current training manifest."""

    return require_valid_training_snapshot(
        adapter_path,
        config_path=config_path,
        expected_iterations=expected_iterations,
        expected_adapter_path=expected_adapter_path,
        expected_inputs=expected_inputs,
    ).manifest


def recheck_training_snapshot(
    snapshot: ValidatedTrainingSnapshot,
) -> ValidatedTrainingSnapshot:
    """Require the exact previously validated manifest and all bound bytes again."""

    with shared_adapter_lock(snapshot.adapter_path):
        captured_sha256 = hashlib.sha256(snapshot.raw_manifest_bytes).hexdigest()
        if captured_sha256 != snapshot.manifest_sha256:
            raise TrainingManifestError(
                "captured training manifest bytes are inconsistent"
            )
        current = require_valid_training_snapshot(
            snapshot.adapter_path,
            config_path=snapshot.config_path,
            expected_iterations=snapshot.expected_iterations,
            expected_adapter_path=snapshot.expected_adapter_path,
            expected_inputs=snapshot.expected_inputs,
        )
        if (
            current.manifest_sha256 != snapshot.manifest_sha256
            or current.raw_manifest_bytes != snapshot.raw_manifest_bytes
            or current.manifest != snapshot.manifest
        ):
            raise TrainingManifestError(
                "training manifest changed after the validated snapshot was captured"
            )
        return snapshot


def _require_valid_training_snapshot(
    adapter_path: Path,
    *,
    config_path: Path | None,
    expected_iterations: int | None,
    expected_adapter_path: Path | None,
    expected_inputs: TrainingInputContract | None,
) -> ValidatedTrainingSnapshot:
    supplied_adapter = _project_path(adapter_path, "adapter path")
    manifest_path = supplied_adapter / TRAINING_MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise TrainingManifestError(
            f"missing successful training manifest: {manifest_path}"
        )
    manifest_file, raw_manifest = _capture_file_bytes(
        manifest_path,
        TRAINING_MANIFEST_NAME,
    )
    config = config_path or PROJECT_ROOT / "configs" / "training" / "lora.yaml"
    plan = _build_training_plan(
        config,
        iterations=expected_iterations,
        adapter_path=expected_adapter_path,
        expected_inputs=expected_inputs,
    )
    if supplied_adapter != plan.target_adapter_path:
        raise TrainingManifestError(
            "trusted expectations point at a different adapter directory"
        )
    try:
        manifest = TrainingManifest.model_validate_json(raw_manifest)
    except ValueError as error:
        raise TrainingManifestError(
            f"invalid successful training manifest: {manifest_path}"
        ) from error

    adapter_file, adapter_config_file = _capture_adapter_outputs(supplied_adapter)
    model_evidence = tuple(item.evidence for item in plan.model_files)
    data_evidence = tuple(item.evidence for item in plan.data_files)
    mismatches: list[str] = []
    if manifest.adapter_path != plan.target_adapter_relative_path:
        mismatches.append("adapter path")
    if manifest.adapter_sha256 != adapter_file.evidence.sha256:
        mismatches.append("adapter SHA-256")
    if manifest.adapter_size_bytes != adapter_file.evidence.size_bytes:
        mismatches.append("adapter size")
    if manifest.adapter_config_sha256 != adapter_config_file.evidence.sha256:
        mismatches.append("adapter configuration SHA-256")
    if manifest.adapter_config_size_bytes != adapter_config_file.evidence.size_bytes:
        mismatches.append("adapter configuration size")
    if manifest.source_config_path != plan.source_config_path:
        mismatches.append("training configuration path")
    if manifest.source_config_sha256 != plan.config_file.evidence.sha256:
        mismatches.append("training configuration SHA-256")
    if manifest.source_config_size_bytes != plan.config_file.evidence.size_bytes:
        mismatches.append("training configuration size")
    if manifest.effective_config != plan.effective_config:
        mismatches.append("effective training configuration")
    if _json_sha256(manifest.effective_config) != manifest.effective_config_sha256:
        mismatches.append("recorded effective configuration SHA-256")
    if manifest.effective_config_sha256 != plan.effective_config_sha256:
        mismatches.append("effective configuration SHA-256")
    if manifest.expected_inputs_sha256 != plan.expected_inputs_sha256:
        mismatches.append("trusted input contract SHA-256")
    if manifest.model_path != plan.expected_inputs.model_path:
        mismatches.append("model path")
    if manifest.model_revision != plan.expected_inputs.model_revision:
        mismatches.append("model revision")
    if manifest.model_files != model_evidence:
        mismatches.append("model file evidence")
    if manifest.data_path != plan.expected_inputs.data_path:
        mismatches.append("training data path")
    if manifest.data_files != data_evidence:
        mismatches.append("training data file evidence")
    if manifest.data_manifest_path != plan.data_manifest_relative_path:
        mismatches.append("training data manifest path")
    if manifest.data_manifest_sha256 != plan.data_manifest_sha256:
        mismatches.append("training data manifest SHA-256")
    if mismatches:
        raise TrainingManifestError(
            "trained adapter evidence is stale or mismatched: " + ", ".join(mismatches)
        )
    _require_unchanged_training_inputs(plan)
    return ValidatedTrainingSnapshot(
        manifest=manifest,
        raw_manifest_bytes=raw_manifest,
        manifest_sha256=manifest_file.evidence.sha256,
        adapter_path=supplied_adapter,
        config_path=plan.config_path,
        expected_iterations=expected_iterations,
        expected_adapter_path=(
            _project_path(expected_adapter_path, "expected adapter path")
            if expected_adapter_path is not None
            else None
        ),
        expected_inputs=plan.expected_inputs,
    )


def _build_training_plan(
    config_path: Path,
    *,
    iterations: int | None,
    adapter_path: Path | None,
    expected_inputs: TrainingInputContract | None,
) -> _TrainingPlan:
    config = _project_path(config_path, "training configuration")
    if not config.is_file() or config.is_symlink():
        raise FileNotFoundError(f"training configuration is missing: {config}")
    source_config_path = _project_relative(config)
    config_file, config_bytes = _capture_file_bytes(config, source_config_path)
    try:
        configured = yaml.safe_load(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("training configuration must be valid UTF-8 YAML") from error
    if not isinstance(configured, dict) or any(
        not isinstance(key, str) for key in configured
    ):
        raise ValueError("training configuration must be a string-keyed mapping")
    if configured.get("resume_adapter_file") is not None:
        raise ValueError(
            "resume_adapter_file is not supported because its bytes are not bound"
        )

    contract = expected_inputs or _default_training_input_contract(config)
    model_path = _project_path(contract.model_path, "expected model path")
    data_path = _project_path(contract.data_path, "expected data path")
    configured_model = configured.get("model")
    configured_data = configured.get("data")
    if configured_model != contract.model_path:
        raise ValueError(
            "training configuration model must exactly match the trusted model path"
        )
    if configured_data != contract.data_path:
        raise ValueError(
            "training configuration data must exactly match the trusted task data path"
        )

    model_files = _capture_model_files(
        model_path,
        expected_inputs=contract,
    )
    data_files = _capture_directory_files(data_path, "training data")
    _require_dataset_files(data_files)

    configured_iterations = configured.get("iters")
    if not isinstance(configured_iterations, int) or configured_iterations < 1:
        raise ValueError("training configuration iters must be a positive integer")
    if iterations is not None and iterations < 1:
        raise ValueError("iterations must be positive")
    effective_iterations = configured_iterations if iterations is None else iterations
    configured_adapter = configured.get("adapter_path")
    if adapter_path is None and not isinstance(configured_adapter, str):
        raise ValueError("training configuration adapter_path must be a string")
    target_adapter = _project_path(
        adapter_path if adapter_path is not None else configured_adapter,
        "adapter path",
    )
    target_adapter_relative_path = _project_relative(target_adapter)
    effective_config: dict[str, JsonValue] = dict(configured)
    effective_config["iters"] = effective_iterations
    effective_config["adapter_path"] = target_adapter_relative_path
    if adapter_path is not None:
        effective_config["save_every"] = effective_iterations
    try:
        effective_config_sha256 = _json_sha256(effective_config)
    except (TypeError, ValueError) as error:
        raise ValueError("training configuration must contain JSON values") from error

    data_manifest = _evidence_by_path(data_files)["manifest.json"]
    return _TrainingPlan(
        config_path=config,
        config_file=config_file,
        source_config_path=source_config_path,
        target_adapter_path=target_adapter,
        target_adapter_relative_path=target_adapter_relative_path,
        iterations=effective_iterations,
        effective_config=effective_config,
        effective_config_sha256=effective_config_sha256,
        expected_inputs=contract,
        expected_inputs_sha256=_model_sha256(contract),
        model_path=model_path,
        model_files=model_files,
        data_path=data_path,
        data_files=data_files,
        data_manifest_relative_path=(f"{contract.data_path}/manifest.json"),
        data_manifest_sha256=data_manifest.evidence.sha256,
    )


def _default_training_input_contract(config_path: Path) -> TrainingInputContract:
    tracked = {
        (PROJECT_ROOT / "configs" / "training" / "lora.yaml").resolve(): (
            "data/processed/bitext-v1"
        ),
        (
            PROJECT_ROOT / "configs" / "training" / "capstone-lora.yaml"
        ).resolve(): "data/processed/capstone-mlx-v1",
    }
    data_path = tracked.get(config_path.resolve())
    if data_path is None:
        raise ValueError(
            "expected_inputs is required for a custom training configuration"
        )
    settings = load_settings(PROJECT_ROOT / "configs" / "project.yaml")
    runtime_files = tuple(
        ExpectedFileHash(path=path, sha256=digest)
        for path, digest in sorted(settings.model.verified_runtime_files.items())
    )
    return TrainingInputContract(
        model_path=settings.model.directory,
        model_revision=settings.model.revision,
        model_runtime_files=runtime_files,
        data_path=data_path,
    )


def _capture_model_files(
    model_path: Path,
    *,
    expected_inputs: TrainingInputContract,
) -> tuple[_CapturedFile, ...]:
    if not model_path.is_dir() or model_path.is_symlink():
        raise FileNotFoundError(f"model directory is missing: {model_path}")
    expected_paths = tuple(
        sorted(
            {
                *(item.path for item in expected_inputs.model_runtime_files),
                _MODEL_REVISION_FILE,
            }
        )
    )
    model_files = tuple(
        _capture_file(model_path / relative_path, relative_path)
        for relative_path in expected_paths
    )
    evidence = _evidence_by_path(model_files)
    for expected in expected_inputs.model_runtime_files:
        captured = evidence.get(expected.path)
        if captured is None:
            raise FileNotFoundError(
                f"expected model runtime file is missing: {model_path / expected.path}"
            )
        if captured.evidence.sha256 != expected.sha256:
            raise ValueError(f"model runtime SHA-256 mismatch: {expected.path}")
        if captured.evidence.size_bytes == 0:
            raise ValueError(f"model runtime file is empty: {expected.path}")
    revision = evidence.get(_MODEL_REVISION_FILE)
    if revision is None:
        raise FileNotFoundError(
            f"model revision file is missing: {model_path / _MODEL_REVISION_FILE}"
        )
    _, revision_bytes = _capture_file_bytes(
        model_path / _MODEL_REVISION_FILE,
        _MODEL_REVISION_FILE,
    )
    try:
        actual_revision = revision_bytes.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError("model revision file must be UTF-8") from error
    if actual_revision != expected_inputs.model_revision:
        raise ValueError("model revision does not match the trusted expectation")
    if hashlib.sha256(revision_bytes).hexdigest() != revision.evidence.sha256:
        raise RuntimeError("model revision file changed while it was captured")
    return model_files


def _require_dataset_files(data_files: tuple[_CapturedFile, ...]) -> None:
    evidence = _evidence_by_path(data_files)
    required = ("manifest.json", "train.jsonl", "valid.jsonl")
    missing = [name for name in required if name not in evidence]
    if missing:
        raise FileNotFoundError(
            "training data files are missing: " + ", ".join(missing)
        )
    empty = [name for name in required if evidence[name].evidence.size_bytes == 0]
    if empty:
        raise ValueError("training data files are empty: " + ", ".join(empty))


def _training_manifest(
    plan: _TrainingPlan,
    *,
    adapter_file: _CapturedFile,
    adapter_config_file: _CapturedFile,
) -> TrainingManifest:
    return TrainingManifest(
        adapter_path=plan.target_adapter_relative_path,
        adapter_sha256=adapter_file.evidence.sha256,
        adapter_size_bytes=adapter_file.evidence.size_bytes,
        adapter_config_sha256=adapter_config_file.evidence.sha256,
        adapter_config_size_bytes=adapter_config_file.evidence.size_bytes,
        source_config_path=plan.source_config_path,
        source_config_sha256=plan.config_file.evidence.sha256,
        source_config_size_bytes=plan.config_file.evidence.size_bytes,
        effective_config=plan.effective_config,
        effective_config_sha256=plan.effective_config_sha256,
        expected_inputs_sha256=plan.expected_inputs_sha256,
        model_path=plan.expected_inputs.model_path,
        model_revision=plan.expected_inputs.model_revision,
        model_files=tuple(item.evidence for item in plan.model_files),
        data_path=plan.expected_inputs.data_path,
        data_files=tuple(item.evidence for item in plan.data_files),
        data_manifest_path=plan.data_manifest_relative_path,
        data_manifest_sha256=plan.data_manifest_sha256,
    )


def _training_evidence(
    plan: _TrainingPlan,
    *,
    command: list[str],
    output: str,
    log_path: Path,
    manifest_sha256: str,
    adapter_sha256: str,
) -> TrainingEvidence:
    train_losses = [
        float(value)
        for value in re.findall(r"Train loss ([0-9]+(?:\.[0-9]+)?)", output)
    ]
    validation_losses = [
        float(value) for value in re.findall(r"Val loss ([0-9]+(?:\.[0-9]+)?)", output)
    ]
    peaks = [
        float(value)
        for value in re.findall(r"Peak mem ([0-9]+(?:\.[0-9]+)?) GB", output)
    ]
    published_manifest = plan.target_adapter_path / TRAINING_MANIFEST_NAME
    return TrainingEvidence(
        command=command,
        return_code=0,
        iterations=plan.iterations,
        train_losses=train_losses,
        validation_losses=validation_losses,
        peak_memory_gb=max(peaks) if peaks else None,
        log_path=_project_relative(log_path),
        training_manifest_path=_project_relative(published_manifest),
        training_manifest_sha256=manifest_sha256,
        adapter_sha256=adapter_sha256,
        effective_config_sha256=plan.effective_config_sha256,
        expected_inputs_sha256=plan.expected_inputs_sha256,
    )


def _capture_adapter_outputs(
    adapter_path: Path,
) -> tuple[_CapturedFile, _CapturedFile]:
    adapter = adapter_path / "adapters.safetensors"
    adapter_config = adapter_path / "adapter_config.json"
    if not adapter.is_file() or not adapter_config.is_file():
        raise RuntimeError(
            "MLX-LM training completed without the required adapter outputs"
        )
    adapter_file = _capture_file(adapter, "adapters.safetensors")
    if adapter_file.evidence.size_bytes == 0:
        raise RuntimeError("MLX-LM training wrote an empty adapter weight file")
    adapter_config_file, raw_config = _capture_file_bytes(
        adapter_config,
        "adapter_config.json",
    )
    try:
        parsed_config = json.loads(raw_config)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("MLX-LM adapter configuration is not valid JSON") from error
    if not isinstance(parsed_config, dict):
        raise RuntimeError("MLX-LM adapter configuration must be a JSON object")
    return adapter_file, adapter_config_file


def _require_unchanged_adapter_outputs(
    adapter_path: Path,
    *,
    adapter_file: _CapturedFile,
    adapter_config_file: _CapturedFile,
) -> None:
    current_adapter, current_config = _capture_adapter_outputs(adapter_path)
    if current_adapter != adapter_file or current_config != adapter_config_file:
        raise RuntimeError("adapter outputs changed before publication")


def _require_unchanged_training_inputs(plan: _TrainingPlan) -> None:
    try:
        current_config, _ = _capture_file_bytes(
            plan.config_path,
            plan.source_config_path,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "training configuration changed while MLX-LM was running"
        ) from error
    if current_config != plan.config_file:
        raise RuntimeError("training configuration changed while MLX-LM was running")
    try:
        current_model = _capture_model_files(
            plan.model_path,
            expected_inputs=plan.expected_inputs,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError("model files changed while MLX-LM was running") from error
    if current_model != plan.model_files:
        raise RuntimeError("model files changed while MLX-LM was running")
    try:
        current_data = _capture_directory_files(plan.data_path, "training data")
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "training data files changed while MLX-LM was running"
        ) from error
    if current_data != plan.data_files:
        raise RuntimeError("training data files changed while MLX-LM was running")


def _capture_directory_files(
    directory: Path,
    label: str,
) -> tuple[_CapturedFile, ...]:
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(f"{label} directory is missing: {directory}")
    paths: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} contains a symbolic link: {path}")
        if path.is_file():
            paths.append(path)
        elif path.exists() and not path.is_dir():
            raise ValueError(f"{label} contains a non-regular path: {path}")
    if not paths:
        raise FileNotFoundError(f"{label} directory contains no files: {directory}")
    return tuple(
        _capture_file(path, path.relative_to(directory).as_posix())
        for path in sorted(
            paths, key=lambda item: item.relative_to(directory).as_posix()
        )
    )


def _capture_file(path: Path, display_path: str) -> _CapturedFile:
    captured, _ = _capture_open_file(path, display_path, include_bytes=False)
    return captured


def _capture_file_bytes(path: Path, display_path: str) -> tuple[_CapturedFile, bytes]:
    captured, raw = _capture_open_file(path, display_path, include_bytes=True)
    assert raw is not None
    return captured, raw


def _capture_open_file(
    path: Path,
    display_path: str,
    *,
    include_bytes: bool,
) -> tuple[_CapturedFile, bytes | None]:
    _require_safe_relative_path(display_path, "captured file")
    if path.is_symlink():
        raise ValueError(f"symbolic links are not valid training inputs: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    chunks: list[bytes] | None = [] if include_bytes else None
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"training input is not a regular file: {path}")
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size_bytes += len(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            after = os.fstat(stream.fileno())
    except Exception:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or size_bytes != after.st_size:
        raise RuntimeError(f"training input changed while it was captured: {path}")
    captured = _CapturedFile(
        evidence=TrainingFileEvidence(
            path=display_path,
            sha256=digest.hexdigest(),
            size_bytes=size_bytes,
        ),
        device=after.st_dev,
        inode=after.st_ino,
        modified_ns=after.st_mtime_ns,
        changed_ns=after.st_ctime_ns,
    )
    return captured, b"".join(chunks) if chunks is not None else None


def _evidence_by_path(
    files: tuple[_CapturedFile, ...],
) -> dict[str, _CapturedFile]:
    return {item.evidence.path: item for item in files}


def _project_path(value: str | Path, label: str) -> Path:
    candidate = Path(value)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (PROJECT_ROOT / candidate).resolve()
    )
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the project") from error
    if resolved == PROJECT_ROOT.resolve():
        raise ValueError(f"{label} must not be the project root")
    return resolved


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path must stay inside the project: {path}") from error


def _require_safe_relative_path(value: str, label: str) -> None:
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative path")


def _json_sha256(value: dict[str, JsonValue]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_sha256(value: BaseModel) -> str:
    payload = json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_json_bytes(value: BaseModel) -> bytes:
    return (value.model_dump_json(indent=2) + "\n").encode("utf-8")


def _invalidate_success_evidence(
    adapter_path: Path,
    evidence_path: Path,
) -> _PriorSuccessEvidence:
    manifest_path = adapter_path / TRAINING_MANIFEST_NAME
    prior = _PriorSuccessEvidence(
        manifest_bytes=_read_existing_regular_file(manifest_path),
        evidence_bytes=_read_existing_regular_file(evidence_path),
    )
    for path in (manifest_path, evidence_path):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.exists():
            raise RuntimeError(f"success evidence path is not a file: {path}")
    return prior


def _read_existing_regular_file(path: Path) -> bytes | None:
    if path.is_symlink():
        return None
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError(f"success evidence path is not a file: {path}")
    _, raw = _capture_file_bytes(path, path.name)
    return raw


def _promote_adapter_directory(staging: Path, target: Path) -> _AdapterPromotion:
    backup = target.with_name(f".{target.name}.previous-{uuid.uuid4().hex}")
    moved_existing = target.exists() or target.is_symlink()
    if moved_existing:
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except BaseException:
        if moved_existing:
            os.replace(backup, target)
        raise
    return _AdapterPromotion(target=target, backup=backup if moved_existing else None)


def _commit_adapter_promotion(promotion: _AdapterPromotion) -> None:
    if promotion.backup is not None:
        _remove_path(promotion.backup, ignore_errors=True)


def _rollback_published_success(
    promotion: _AdapterPromotion,
    *,
    prior: _PriorSuccessEvidence,
    evidence_path: Path,
) -> None:
    failed = promotion.target.with_name(
        f".{promotion.target.name}.failed-{uuid.uuid4().hex}"
    )
    if promotion.target.exists() or promotion.target.is_symlink():
        try:
            os.replace(promotion.target, failed)
        except BaseException:
            published_manifest = promotion.target / TRAINING_MANIFEST_NAME
            if published_manifest.is_file() or published_manifest.is_symlink():
                published_manifest.unlink()
            raise
    failed_manifest = failed / TRAINING_MANIFEST_NAME
    if failed_manifest.is_file() or failed_manifest.is_symlink():
        failed_manifest.unlink()
    if promotion.backup is not None:
        os.replace(promotion.backup, promotion.target)
    if failed.exists() or failed.is_symlink():
        _remove_path(failed, ignore_errors=True)
    _restore_file(promotion.target / TRAINING_MANIFEST_NAME, prior.manifest_bytes)
    _restore_file(evidence_path, prior.evidence_bytes)


def _restore_file(path: Path, content: bytes | None) -> None:
    if content is None:
        if path.is_file() or path.is_symlink():
            path.unlink()
        return
    _write_bytes_atomic(path, content)


def _remove_path(path: Path, *, ignore_errors: bool = False) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    except OSError:
        if not ignore_errors:
            raise


def _write_text_atomic(path: Path, content: str) -> None:
    _write_bytes_atomic(path, content.encode("utf-8"))


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, evidence: BaseModel) -> None:
    _write_bytes_atomic(path, _model_json_bytes(evidence))
