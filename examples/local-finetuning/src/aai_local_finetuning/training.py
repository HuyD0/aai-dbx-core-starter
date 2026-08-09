"""Safe wrapper around the pinned MLX-LM LoRA command."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import types
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from . import _runtime_audit
from .settings import PROJECT_ROOT, ProjectSettings, load_settings

TRAINING_MANIFEST_NAME = "training-manifest.json"
_MODEL_REVISION_FILE = "LOCAL_REVISION"
_MODEL_NON_RUNTIME_FILES = frozenset(
    {
        ".DS_Store",
        ".gitattributes",
        "LICENSE",
        "LICENSE.md",
        "LICENSE.txt",
        "NOTICE",
        "NOTICE.md",
        "NOTICE.txt",
        "README.md",
    }
)
_MODEL_NON_RUNTIME_DIRECTORIES = frozenset({".cache"})
_SOURCE_PACKAGE_PATH = PurePosixPath("src/aai_local_finetuning")
_NOTEBOOK_SOURCE_PATHS = (
    PurePosixPath("scripts/notebook_pedagogy.py"),
    PurePosixPath("scripts/render_notebooks.py"),
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_REVISION_PATTERN = r"^[0-9a-f]{40}$"
_RUNTIME_PAYLOAD_DIGEST_CACHE_MAX_ENTRIES = 65_536
_RUNTIME_PAYLOAD_DIGEST_CACHE: OrderedDict[
    tuple[str, int, int, int, int, int],
    tuple[str, int],
] = OrderedDict()
_RUNTIME_PAYLOAD_DIGEST_CACHE_LOCK = threading.Lock()
_RUNTIME_NON_PAYLOAD_METADATA_FILES = frozenset(
    {
        "INSTALLER",
        "RECORD",
        "REQUESTED",
        "SOURCES.txt",
        "WHEEL",
        "direct_url.json",
    }
)
_RUNTIME_NON_PAYLOAD_METADATA_DIRECTORIES = frozenset({"licenses"})
_RUNTIME_TRANSIENT_DIRECTORIES = frozenset({"__pycache__"})
_RUNTIME_EXTERNAL_BOOKKEEPING_PREFIXES = (
    ("bin",),
    ("Scripts",),
    ("etc", "jupyter"),
    ("share", "applications"),
    ("share", "icons"),
    ("share", "jupyter"),
    ("share", "man"),
)
_RUNTIME_EXTERNAL_PAYLOAD_SUFFIXES = frozenset({".jar"})
_SETUPTOOLS_DISTUTILS_PTH_SOURCE = (
    "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; "
    "enabled = os.environ.get(var, 'local') == 'local'; "
    "enabled and __import__('_distutils_hack').add_shim();"
)
_COVERAGE_PTH_EMBEDDED_SOURCE = """\
import os

if os.getenv("COVERAGE_PROCESS_START") or os.getenv("COVERAGE_PROCESS_CONFIG"):
    try:
        import coverage
    except:
        pass
    else:
        coverage.process_startup(slug="pth")
"""
_BYTECODE_VALIDATOR_SCRIPT = r"""
import importlib.util
import io
import hashlib
import json
import marshal
import re
import struct
import sys
import types
from pathlib import Path


def canonical(code):
    constants = tuple(
        canonical(value) if isinstance(value, types.CodeType) else value
        for value in code.co_consts
    )
    return code.replace(co_filename="<runtime-bytecode>", co_consts=constants)


def load_code(raw):
    if len(raw) < 16 or raw[:4] != importlib.util.MAGIC_NUMBER:
        raise ValueError("invalid or foreign bytecode header")
    flags = int.from_bytes(raw[4:8], "little")
    if flags & ~3 or flags == 2:
        raise ValueError("unsupported bytecode flags")
    stream = io.BytesIO(raw[16:])
    code = marshal.load(stream)
    if stream.tell() != len(raw) - 16 or not isinstance(code, types.CodeType):
        raise ValueError("bytecode payload is not one complete code object")
    return canonical(code)


def source_path(path):
    if path.parent.name == "__pycache__":
        try:
            return Path(importlib.util.source_from_cache(str(path))), "standard"
        except ValueError:
            match = re.fullmatch(
                r"(.+)\.cpython-\d+-pytest-\d+(?:\.\d+)*\.pyc",
                path.name,
            )
            if match:
                return path.parent.parent / f"{match.group(1)}.py", "pytest"
            return None, "unknown"
    return path.with_suffix(".py"), "legacy"


def semantic(value):
    if isinstance(value, types.CodeType):
        return {
            "type": "code",
            "argcount": value.co_argcount,
            "posonlyargcount": value.co_posonlyargcount,
            "kwonlyargcount": value.co_kwonlyargcount,
            "nlocals": value.co_nlocals,
            "stacksize": value.co_stacksize,
            "flags": value.co_flags,
            "code": value.co_code.hex(),
            "consts": [semantic(item) for item in value.co_consts],
            "names": list(value.co_names),
            "varnames": list(value.co_varnames),
            "filename": "<runtime-bytecode>",
            "name": value.co_name,
            "qualname": value.co_qualname,
            "firstlineno": value.co_firstlineno,
            "linetable": value.co_linetable.hex(),
            "exceptiontable": value.co_exceptiontable.hex(),
            "freevars": list(value.co_freevars),
            "cellvars": list(value.co_cellvars),
        }
    if value is None:
        return {"type": "none"}
    if value is Ellipsis:
        return {"type": "ellipsis"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": struct.pack(">d", value).hex()}
    if isinstance(value, complex):
        return {
            "type": "complex",
            "real": struct.pack(">d", value.real).hex(),
            "imag": struct.pack(">d", value.imag).hex(),
        }
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [semantic(item) for item in value]}
    if isinstance(value, frozenset):
        items = [semantic(item) for item in value]
        return {
            "type": "frozenset",
            "value": sorted(
                items,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        }
    raise ValueError(f"unsupported bytecode constant type: {type(value).__name__}")


def semantic_sha256(code):
    payload = json.dumps(
        semantic(code),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate(path):
    source, kind = source_path(path)
    if source is None or not source.is_file():
        raise ValueError("sourceless or unrecognized bytecode is unsupported")
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:4] != importlib.util.MAGIC_NUMBER:
        raise ValueError("invalid or foreign bytecode header")
    flags = int.from_bytes(raw[4:8], "little")
    if flags & ~3 or flags == 2:
        raise ValueError("unsupported bytecode flags")
    source_bytes = source.read_bytes()
    if flags == 0:
        details = source.stat()
        expected_header = (
            (int(details.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little")
            + (len(source_bytes) & 0xFFFFFFFF).to_bytes(4, "little")
        )
        if raw[8:16] != expected_header:
            return {"status": "inactive-stale"}
    elif flags & 2 and raw[8:16] != importlib.util.source_hash(source_bytes):
        return {"status": "inactive-stale"}
    match = re.search(r"\.opt-([012])\.", path.name)
    optimize = int(match.group(1)) if match else 0
    cached = load_code(raw)
    compiled = compile(
        source_bytes,
        "<runtime-bytecode>",
        "exec",
        dont_inherit=True,
        optimize=optimize,
    )
    expected = canonical(compiled)
    if cached == expected:
        return {"status": "source-equivalent"}
    if kind != "pytest":
        raise ValueError("cached bytecode differs semantically from captured source")
    return {
        "status": "instrumented",
        "semantic_sha256": semantic_sha256(cached),
    }


requests = json.load(sys.stdin)
results = []
for raw_path in requests:
    try:
        results.append({"path": raw_path, **validate(Path(raw_path))})
    except Exception as error:
        results.append({"path": raw_path, "error": str(error)})
json.dump(results, sys.stdout, sort_keys=True, separators=(",", ":"))
"""
_SETUPTOOLS_FINDER_TEMPLATE = """\
from __future__ import annotations
import sys
from importlib.machinery import ModuleSpec, PathFinder
from importlib.machinery import all_suffixes as module_suffixes
from importlib.util import spec_from_file_location
from itertools import chain
from pathlib import Path

MAPPING: dict[str, str] = {mapping!r}
NAMESPACES: dict[str, list[str]] = {namespaces!r}
PATH_PLACEHOLDER = {name!r} + ".__path_hook__"


class _EditableFinder:
    @classmethod
    def find_spec(cls, fullname: str, path=None, target=None) -> ModuleSpec | None:
        if fullname in MAPPING:
            pkg_path = MAPPING[fullname]
            return cls._find_spec(fullname, Path(pkg_path))
        parent, _, child = fullname.rpartition(".")
        if parent and parent in MAPPING:
            return PathFinder.find_spec(fullname, path=[MAPPING[parent]])
        return None

    @classmethod
    def _find_spec(cls, fullname: str, candidate_path: Path) -> ModuleSpec | None:
        init = candidate_path / "__init__.py"
        candidates = (candidate_path.with_suffix(x) for x in module_suffixes())
        for candidate in chain([init], candidates):
            if candidate.exists():
                return spec_from_file_location(fullname, candidate)
        return None


class _EditableNamespaceFinder:
    @classmethod
    def _path_hook(cls, path) -> type[_EditableNamespaceFinder]:
        if path == PATH_PLACEHOLDER:
            return cls
        raise ImportError

    @classmethod
    def _paths(cls, fullname: str) -> list[str]:
        paths = NAMESPACES[fullname]
        if not paths and fullname in MAPPING:
            paths = [MAPPING[fullname]]
        return [*paths, PATH_PLACEHOLDER]

    @classmethod
    def find_spec(cls, fullname: str, target=None) -> ModuleSpec | None:
        if fullname in NAMESPACES:
            spec = ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = cls._paths(fullname)
            return spec
        return None

    @classmethod
    def find_module(cls, _fullname) -> None:
        return None


def install():
    if not any(finder == _EditableFinder for finder in sys.meta_path):
        sys.meta_path.append(_EditableFinder)
    if not NAMESPACES:
        return
    if not any(hook == _EditableNamespaceFinder._path_hook for hook in sys.path_hooks):
        sys.path_hooks.append(_EditableNamespaceFinder._path_hook)
    if PATH_PLACEHOLDER not in sys.path:
        sys.path.append(PATH_PLACEHOLDER)
"""


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


class RuntimePackageEvidence(BaseModel):
    """One installed distribution and its content-addressed runtime payload."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    version: str = Field(min_length=1)
    payload_file_count: int = Field(ge=0)
    payload_size_bytes: int = Field(ge=0)
    payload_files_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _require_canonical_value(self) -> RuntimePackageEvidence:
        if self.version != self.version.strip():
            raise ValueError(
                "runtime package version must not contain outer whitespace"
            )
        if self.payload_file_count == 0 and self.payload_size_bytes != 0:
            raise ValueError("an empty runtime package payload must have zero bytes")
        return self


class ExecutionContract(BaseModel):
    """Portable hashes for the source tree and exact installed package set."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["2.0.0"] = "2.0.0"
    python_version: str = Field(min_length=1)
    python_implementation: str = Field(min_length=1)
    operating_system: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    source_files: tuple[TrainingFileEvidence, ...] = Field(min_length=1)
    source_files_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_packages: tuple[RuntimePackageEvidence, ...] = Field(min_length=1)
    runtime_packages_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _require_canonical_contract(self) -> ExecutionContract:
        for field_name in (
            "python_version",
            "python_implementation",
            "operating_system",
            "machine",
        ):
            value = getattr(self, field_name)
            if value != value.strip():
                raise ValueError(f"{field_name} must not contain outer whitespace")
        source_paths = tuple(item.path for item in self.source_files)
        if source_paths != tuple(sorted(source_paths)) or len(source_paths) != len(
            set(source_paths)
        ):
            raise ValueError("source_files must have unique paths in sorted order")
        package_keys = tuple(
            _runtime_package_sort_key(item) for item in self.runtime_packages
        )
        if package_keys != tuple(sorted(package_keys)):
            raise ValueError("runtime_packages must be in canonical sorted order")
        if self.source_files_sha256 != _evidence_sequence_sha256(self.source_files):
            raise ValueError("source_files_sha256 does not match source_files")
        if self.runtime_packages_sha256 != _evidence_sequence_sha256(
            self.runtime_packages
        ):
            raise ValueError("runtime_packages_sha256 does not match runtime_packages")
        return self


class BaseModelExecutionContract(BaseModel):
    """Portable identity for the exact local base-model runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    repository: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    model_revision: str = Field(pattern=_REVISION_PATTERN)
    model_files: tuple[TrainingFileEvidence, ...] = Field(min_length=2)
    model_files_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _require_canonical_contract(self) -> BaseModelExecutionContract:
        if self.repository != self.repository.strip():
            raise ValueError("base-model repository must not contain outer whitespace")
        _require_safe_relative_path(self.model_path, "base-model path")
        paths = tuple(item.path for item in self.model_files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("model_files must have unique paths in sorted order")
        if _MODEL_REVISION_FILE not in paths:
            raise ValueError(f"model_files must include {_MODEL_REVISION_FILE}")
        if any(PurePosixPath(path).name != path for path in paths):
            raise ValueError("base-model runtime files must be top-level filenames")
        if self.model_files_sha256 != _evidence_sequence_sha256(self.model_files):
            raise ValueError("model_files_sha256 does not match model_files")
        return self


class TrainingManifest(BaseModel):
    """Immutable binding between adapter bytes and effective training inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["4.0.0"] = "4.0.0"
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
    execution_contract: ExecutionContract
    execution_contract_sha256: str = Field(pattern=_SHA256_PATTERN)

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
        if self.execution_contract_sha256 != _model_sha256(self.execution_contract):
            raise ValueError(
                "execution_contract_sha256 does not match execution_contract"
            )
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
    source_files_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_packages_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_contract_sha256: str = Field(pattern=_SHA256_PATTERN)


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
    physical_path: Path = field(repr=False)
    physical_size_bytes: int
    device: int
    inode: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _CapturedDirectory:
    path: Path
    device: int
    inode: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _CapturedPhysicalFile:
    path: Path
    size_bytes: int
    device: int
    inode: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _LoadedModuleState:
    name: str
    origin: _CapturedPhysicalFile
    module: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _LoadedSpecLessModuleState:
    name: str
    parent_name: str | None
    module: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _RuntimeNativeLoadState:
    name: str
    origin: _CapturedPhysicalFile
    module: object = field(repr=False, compare=False)
    loader: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _CapturedImportHooks:
    portable: tuple[str, ...]
    objects: tuple[object, ...] = field(repr=False, compare=False)


class _RuntimeExtensionLoader(importlib.abc.Loader):
    """Delegate one extension load and retain its resulting module identity."""

    def __init__(
        self,
        name: str,
        delegate: importlib.machinery.ExtensionFileLoader,
    ) -> None:
        self._name = name
        self._delegate = delegate

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> object:
        return self._delegate.create_module(spec)

    def exec_module(self, module: types.ModuleType) -> None:
        self._delegate.exec_module(module)
        _record_loaded_native_module(self._name, module, self)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class _RuntimeExtensionFinder(importlib.abc.MetaPathFinder):
    """Wrap extensions found by the ordinary path finder after initialization."""

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and isinstance(
            spec.loader,
            importlib.machinery.ExtensionFileLoader,
        ):
            spec.loader = _RuntimeExtensionLoader(fullname, spec.loader)
        return spec


@dataclass(frozen=True, slots=True)
class _CapturedExecutable:
    path: Path
    link_mode: int
    link_size: int
    link_device: int
    link_inode: int
    link_modified_ns: int
    link_changed_ns: int
    target: _CapturedPhysicalFile


@dataclass(frozen=True, slots=True)
class _CapturedRuntimeTree:
    files: tuple[tuple[str, Path], ...]
    bytecode_files: tuple[tuple[str, Path], ...]
    directories: tuple[_CapturedDirectory, ...]


@dataclass(frozen=True, slots=True)
class _RuntimePathConfiguration:
    captured_files: tuple[tuple[str, _CapturedFile], ...]
    explicit_files: tuple[tuple[str, Path], ...]
    exposed_trees: tuple[tuple[str, Path], ...]
    observed_directories: tuple[Path, ...]
    consumed_paths: frozenset[Path]
    import_bindings: tuple[_RuntimeImportBinding, ...]


@dataclass(frozen=True, slots=True)
class _RuntimeImportBinding:
    name: str
    physical_path: Path
    kind: Literal["module", "package", "namespace"]
    exposure: Literal["direct", "path", "finder"]
    search_root: Path | None
    namespace_placeholder: str | None = None


@dataclass(frozen=True, slots=True)
class _CapturedImportSurface:
    persistent_files: tuple[_CapturedFile, ...]
    transient_files: tuple[_CapturedFile, ...]
    directories: tuple[_CapturedDirectory, ...]
    import_bindings: tuple[_RuntimeImportBinding, ...]


@dataclass(slots=True)
class _RuntimeCaptureContext:
    """Deduplicate physical runtime reads within one evidence snapshot."""

    files: dict[Path, _CapturedFile] = field(default_factory=dict)
    trees: dict[Path, _CapturedRuntimeTree] = field(default_factory=dict)
    directories: dict[Path, _CapturedDirectory] = field(default_factory=dict)
    bytecode_files: dict[Path, _CapturedFile] = field(default_factory=dict)
    bytecode_semantics: dict[Path, str | None] = field(default_factory=dict)

    def capture_file(self, path: Path, display_path: str) -> _CapturedFile:
        captured = self.files.get(path)
        if captured is None:
            captured = _capture_runtime_payload_file(path, display_path)
            self.files[path] = captured
        return _captured_file_with_display_path(captured, display_path)

    def capture_tree(self, root: Path) -> _CapturedRuntimeTree:
        captured = self.trees.get(root)
        if captured is None:
            captured = _capture_runtime_tree(root)
            self.trees[root] = captured
            for directory in captured.directories:
                self.observe_directory(directory)
        return captured

    def capture_bytecode(self, path: Path, logical_path: str) -> _CapturedFile:
        display_path = _runtime_payload_display_path(
            f"validated-bytecode/{logical_path}"
        )
        captured = self.capture_file(path, display_path)
        if path not in self.bytecode_files:
            self.bytecode_files[path] = captured
        return captured

    def observe_directory(self, captured: _CapturedDirectory) -> None:
        previous = self.directories.setdefault(captured.path, captured)
        if previous != captured:
            raise RuntimeError(
                "runtime package directory changed while it was captured: "
                f"{captured.path}"
            )

    def verify_directories(self) -> None:
        self.validate_bytecode()
        for path, captured in sorted(self.directories.items()):
            if _capture_directory_identity(path) != captured:
                raise RuntimeError(
                    "runtime package directory changed while it was captured: "
                    f"{path}"
                )
        for path, captured in sorted(self.files.items()):
            if not _captured_file_identity_matches(captured):
                raise RuntimeError(
                    f"runtime package file changed while it was captured: {path}"
                )

    def validate_bytecode(self) -> dict[Path, str | None]:
        missing = tuple(
            sorted(set(self.bytecode_files).difference(self.bytecode_semantics))
        )
        if missing:
            self.bytecode_semantics.update(_validate_runtime_bytecode(missing))
        return self.bytecode_semantics


@dataclass(frozen=True, slots=True)
class _RuntimeDistribution:
    evidence: RuntimePackageEvidence
    metadata_path: Path
    metadata_file: _CapturedFile
    install_root: Path
    install_root_identity: _CapturedDirectory
    payload_files: tuple[_CapturedFile, ...]
    transient_files: tuple[_CapturedFile, ...]
    runtime_roots: tuple[_CapturedDirectory, ...]
    import_bindings: tuple[_RuntimeImportBinding, ...]
    import_environment: _CapturedImportEnvironment | None = None
    inventory_backed: bool = True


@dataclass(frozen=True, slots=True)
class _CapturedImportEnvironment:
    signature: str
    active_tokens: tuple[str, ...]
    root_tokens: tuple[tuple[Path, str], ...]
    protected_origins: tuple[tuple[str, str], ...]
    origin_paths: tuple[tuple[Path, str], ...]
    namespace_placeholders: tuple[str, ...]
    captured_files: tuple[_CapturedPhysicalFile, ...]
    loaded_modules: tuple[_LoadedModuleState, ...]
    spec_less_modules: tuple[_LoadedSpecLessModuleState, ...]
    meta_path: _CapturedImportHooks
    path_hooks: _CapturedImportHooks
    execution_counts: tuple[tuple[Path, int], ...]
    bytecode_writes_disabled: bool


_RUNTIME_PROVENANCE_LOCK = threading.RLock()
_RUNTIME_EXECUTED_MODULE_CODE: dict[Path, list[types.CodeType]] = {}
_RUNTIME_EXECUTED_MODULE_CODE_OVERFLOW: set[Path] = set()
_RUNTIME_EXECUTION_COUNTS: dict[Path, int] = {}
_RUNTIME_IMPORTED_NATIVE_FILES: dict[
    tuple[str, Path],
    list[_CapturedPhysicalFile],
] = {}
_RUNTIME_IMPORTED_NATIVE_FILES_OVERFLOW: set[tuple[str, Path]] = set()
_RUNTIME_IMPORTED_NATIVE_MODULES: dict[
    tuple[str, Path],
    list[_RuntimeNativeLoadState],
] = {}
_RUNTIME_IMPORTED_NATIVE_MODULES_OVERFLOW: set[tuple[str, Path]] = set()
_RUNTIME_MLX_CHILD_BINDINGS: dict[str, list[tuple[object, object]]] = {}
_RUNTIME_MLX_CHILD_BINDINGS_OVERFLOW: set[str] = set()
_RUNTIME_INITIAL_MODULE_FILES: dict[
    tuple[str, Path],
    _CapturedPhysicalFile,
] = {}
_RUNTIME_INITIAL_MODULE_OBJECTS: dict[tuple[str, Path], object] = {}
_RUNTIME_INITIAL_SPECLESS_MODULES: dict[str, object] = {}
_RUNTIME_PROVENANCE_CACHE_MAX_ENTRIES = 8_192
_RUNTIME_PROVENANCE_CACHE: OrderedDict[tuple[object, ...], None] = OrderedDict()
_RUNTIME_PROVENANCE_AUDIT_INSTALLED = False
_RUNTIME_EXTENSION_FINDER: _RuntimeExtensionFinder | None = None
_RUNTIME_SPECLESS_MODULE_NAMES = frozenset({"__main__"})
_RUNTIME_SPECLESS_PARENT_ALIASES = {
    "pyexpat.errors": ("pyexpat", "errors"),
    "pyexpat.model": ("pyexpat", "model"),
    "typing.io": ("typing", "io"),
    "typing.re": ("typing", "re"),
}
_RUNTIME_MLX_SPECLESS_CHILDREN = frozenset(
    {
        "mlx.core.cuda",
        "mlx.core.distributed",
        "mlx.core.fast",
        "mlx.core.fft",
        "mlx.core.linalg",
        "mlx.core.metal",
        "mlx.core.random",
    }
)


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    """Transient source/runtime identity snapshot for a bounded operation.

    The portable contract is safe to persist. File and directory identities stay
    transient so source or installed-package metadata changed and restored during
    an operation still invalidates it without leaking machine-specific paths or
    inode data into evidence artifacts.
    """

    execution_contract: ExecutionContract
    execution_contract_sha256: str
    _source_files: tuple[_CapturedFile, ...] = field(repr=False)
    _source_transient_files: tuple[_CapturedFile, ...] = field(repr=False)
    _source_directories: tuple[_CapturedDirectory, ...] = field(repr=False)
    _runtime_package_metadata: tuple[_CapturedFile, ...] = field(repr=False)
    _runtime_package_payloads: tuple[_CapturedFile, ...] = field(repr=False)
    _runtime_package_roots: tuple[_CapturedDirectory, ...] = field(repr=False)
    _runtime_metadata_paths: tuple[Path, ...] = field(repr=False)
    _import_environment: _CapturedImportEnvironment = field(repr=False)


@dataclass(frozen=True, slots=True)
class BaseModelSnapshot:
    """Portable model evidence plus transient local identities for rechecking."""

    execution_contract: BaseModelExecutionContract
    execution_contract_sha256: str
    _model_path: Path = field(repr=False)
    _model_directory: _CapturedDirectory = field(repr=False)
    _model_files: tuple[_CapturedFile, ...] = field(repr=False)


_CapturedExecutionContract = tuple[tuple[_CapturedFile, ...], ExecutionContract]


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
    model_directory: _CapturedDirectory
    model_files: tuple[_CapturedFile, ...]
    data_path: Path
    data_files: tuple[_CapturedFile, ...]
    data_manifest_relative_path: str
    data_manifest_sha256: str
    source_files: tuple[_CapturedFile, ...]
    execution_snapshot: ExecutionSnapshot
    execution_contract: ExecutionContract
    execution_contract_sha256: str
    child_python: _CapturedExecutable
    child_environment: tuple[tuple[str, str], ...] = field(repr=False)


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


def _capture_child_python() -> _CapturedExecutable:
    path = Path(sys.executable)
    if not path.is_absolute():
        raise RuntimeError("Python executable path must be absolute")
    _require_no_lexical_symlink_components(
        path.parent,
        label="Python executable parent",
    )
    linked = path.lstat()
    if not (stat.S_ISREG(linked.st_mode) or stat.S_ISLNK(linked.st_mode)):
        raise ValueError(f"Python executable is not a regular file: {path}")
    target_path = path.resolve(strict=True)
    target = _capture_physical_file_identity(
        target_path,
        label="Python executable target",
    )
    return _CapturedExecutable(
        path=path,
        link_mode=linked.st_mode,
        link_size=linked.st_size,
        link_device=linked.st_dev,
        link_inode=linked.st_ino,
        link_modified_ns=linked.st_mtime_ns,
        link_changed_ns=linked.st_ctime_ns,
        target=target,
    )


def _captured_executable_matches(captured: _CapturedExecutable) -> bool:
    if Path(sys.executable) != captured.path:
        return False
    try:
        linked = captured.path.lstat()
        target_path = captured.path.resolve(strict=True)
        target = _capture_physical_file_identity(
            target_path,
            label="Python executable target",
        )
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        linked.st_mode == captured.link_mode
        and linked.st_size == captured.link_size
        and linked.st_dev == captured.link_device
        and linked.st_ino == captured.link_inode
        and linked.st_mtime_ns == captured.link_modified_ns
        and linked.st_ctime_ns == captured.link_changed_ns
        and target == captured.target
    )


def _child_environment() -> tuple[tuple[str, str], ...]:
    """Return a fixed environment with Python import overrides removed."""

    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if not name.upper().startswith("PYTHON")
        )
    )


def _disable_runtime_bytecode_writes() -> None:
    """Prevent lazy imports from mutating governed cache directories."""

    sys.dont_write_bytecode = True


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
        if not _captured_executable_matches(plan.child_python):
            raise RuntimeError("Python executable changed before MLX-LM launch")
        command = [
            str(plan.child_python.path),
            "-I",
            "-B",
            "-m",
            "mlx_lm",
            "lora",
            "--config",
            str(plan.config_path),
        ]
        recorded_command = [
            "<python>",
            "-I",
            "-B",
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
            env=dict(plan.child_environment),
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
    if manifest.execution_contract.source_files != plan.execution_contract.source_files:
        mismatches.append("evaluator and training source code")
    if (
        manifest.execution_contract.source_files_sha256
        != plan.execution_contract.source_files_sha256
    ):
        mismatches.append("source-code set SHA-256")
    if (
        manifest.execution_contract.runtime_packages
        != plan.execution_contract.runtime_packages
    ):
        mismatches.append("runtime package set")
    if (
        manifest.execution_contract.runtime_packages_sha256
        != plan.execution_contract.runtime_packages_sha256
    ):
        mismatches.append("runtime package-set SHA-256")
    if manifest.execution_contract_sha256 != plan.execution_contract_sha256:
        mismatches.append("source/runtime execution contract SHA-256")
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

    model_directory_before = _capture_directory_identity(
        model_path,
        label="base-model directory",
    )
    model_files = _capture_model_files(
        model_path,
        expected_inputs=contract,
    )
    model_directory_after = _capture_directory_identity(
        model_path,
        label="base-model directory",
    )
    if model_directory_after != model_directory_before:
        raise RuntimeError("base-model directory changed while it was captured")
    data_files = _capture_directory_files(data_path, "training data")
    _require_dataset_files(data_files)
    execution_snapshot = capture_execution_snapshot()
    source_files = execution_snapshot._source_files
    execution_contract = execution_snapshot.execution_contract

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
    child_python = _capture_child_python()
    child_environment = _child_environment()
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
        model_directory=model_directory_before,
        model_files=model_files,
        data_path=data_path,
        data_files=data_files,
        data_manifest_relative_path=(f"{contract.data_path}/manifest.json"),
        data_manifest_sha256=data_manifest.evidence.sha256,
        source_files=source_files,
        execution_snapshot=execution_snapshot,
        execution_contract=execution_contract,
        execution_contract_sha256=_model_sha256(execution_contract),
        child_python=child_python,
        child_environment=child_environment,
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
    return _capture_verified_model_files(
        model_path,
        model_revision=expected_inputs.model_revision,
        model_runtime_files=expected_inputs.model_runtime_files,
    )


def _capture_verified_model_files(
    model_path: Path,
    *,
    model_revision: str,
    model_runtime_files: tuple[ExpectedFileHash, ...],
) -> tuple[_CapturedFile, ...]:
    """Capture only independently verified files used by the local runtime."""

    if not model_path.is_dir() or model_path.is_symlink():
        raise FileNotFoundError(f"model directory is missing: {model_path}")
    expected_paths = tuple(
        sorted(
            {
                *(item.path for item in model_runtime_files),
                _MODEL_REVISION_FILE,
            }
        )
    )
    _require_verified_model_directory_surface(model_path, expected_paths)
    model_files = tuple(
        _capture_file(model_path / relative_path, relative_path)
        for relative_path in expected_paths
    )
    evidence = _evidence_by_path(model_files)
    for expected in model_runtime_files:
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
    if actual_revision != model_revision:
        raise ValueError("model revision does not match the trusted expectation")
    if hashlib.sha256(revision_bytes).hexdigest() != revision.evidence.sha256:
        raise RuntimeError("model revision file changed while it was captured")
    return model_files


def _require_verified_model_directory_surface(
    model_path: Path,
    expected_runtime_paths: tuple[str, ...],
) -> None:
    """Reject unverified entries that a local model/tokenizer loader could consume."""

    expected = set(expected_runtime_paths)
    unexpected: list[str] = []
    for entry in model_path.iterdir():
        if entry.name in expected:
            continue
        if (
            entry.name in _MODEL_NON_RUNTIME_FILES
            and entry.is_file()
            and not entry.is_symlink()
        ):
            continue
        if (
            entry.name in _MODEL_NON_RUNTIME_DIRECTORIES
            and entry.is_dir()
            and not entry.is_symlink()
        ):
            continue
        unexpected.append(entry.name)
    if unexpected:
        raise ValueError(
            "base-model directory contains unverified entries: "
            + ", ".join(sorted(unexpected))
        )


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
        execution_contract=plan.execution_contract,
        execution_contract_sha256=plan.execution_contract_sha256,
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
        source_files_sha256=plan.execution_contract.source_files_sha256,
        runtime_packages_sha256=(plan.execution_contract.runtime_packages_sha256),
        execution_contract_sha256=plan.execution_contract_sha256,
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
    if not _captured_executable_matches(plan.child_python):
        raise RuntimeError("Python executable changed while MLX-LM was running")
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
        current_model_directory_before = _capture_directory_identity(
            plan.model_path,
            label="base-model directory",
        )
        current_model = _capture_model_files(
            plan.model_path,
            expected_inputs=plan.expected_inputs,
        )
        current_model_directory_after = _capture_directory_identity(
            plan.model_path,
            label="base-model directory",
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError("model files changed while MLX-LM was running") from error
    if (
        current_model_directory_before != plan.model_directory
        or current_model_directory_after != current_model_directory_before
        or current_model != plan.model_files
    ):
        raise RuntimeError("model files changed while MLX-LM was running")
    try:
        current_data = _capture_directory_files(plan.data_path, "training data")
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "training data files changed while MLX-LM was running"
        ) from error
    if current_data != plan.data_files:
        raise RuntimeError("training data files changed while MLX-LM was running")
    try:
        recheck_execution_snapshot(plan.execution_snapshot)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "source code or runtime package set changed while MLX-LM was running"
        ) from error
    if (
        plan.execution_snapshot.execution_contract != plan.execution_contract
        or plan.execution_snapshot._source_files != plan.source_files
    ):
        raise RuntimeError(
            "captured source/runtime execution contract is internally inconsistent"
        )


def _capture_execution_contract() -> _CapturedExecutionContract:
    """Capture portable source bytes and installed distribution payloads."""

    _disable_runtime_bytecode_writes()
    context = _RuntimeCaptureContext()
    source_files, _source_transient, _source_directories = (
        _capture_governed_source_state(context=context)
    )
    distributions = _runtime_distribution_inventory(context=context)
    source_files = _bind_instrumented_bytecode(
        source_files,
        _source_transient,
        context=context,
    )
    context.verify_directories()
    packages = tuple(item.evidence for item in distributions)
    contract = _build_execution_contract(source_files, packages)
    return source_files, contract


def _build_execution_contract(
    source_files: tuple[_CapturedFile, ...],
    packages: tuple[RuntimePackageEvidence, ...],
) -> ExecutionContract:
    if not packages:
        raise RuntimeError("installed runtime package set is empty")
    source_evidence = tuple(item.evidence for item in source_files)
    return ExecutionContract(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        operating_system=platform.system(),
        machine=platform.machine(),
        source_files=source_evidence,
        source_files_sha256=_evidence_sequence_sha256(source_evidence),
        runtime_packages=packages,
        runtime_packages_sha256=_evidence_sequence_sha256(packages),
    )


def capture_execution_contract() -> ExecutionContract:
    """Return the current portable source/runtime contract for evaluation evidence."""

    _source_files, contract = _capture_execution_contract()
    return contract


def capture_execution_snapshot() -> ExecutionSnapshot:
    """Capture portable evidence plus transient identities for later rechecking."""

    _disable_runtime_bytecode_writes()
    context = _RuntimeCaptureContext()
    source_files, source_transient_files, source_directories = (
        _capture_governed_source_state(context=context)
    )
    distributions = _runtime_distribution_inventory(context=context)
    source_files = _bind_instrumented_bytecode(
        source_files,
        source_transient_files,
        context=context,
    )
    context.verify_directories()
    packages = tuple(item.evidence for item in distributions)
    contract = _build_execution_contract(source_files, packages)
    package_metadata, package_payloads, package_roots = _runtime_package_state(
        distributions
    )
    import_environments = tuple(
        item.import_environment
        for item in distributions
        if item.import_environment is not None
    )
    if len(import_environments) != 1:
        raise RuntimeError("runtime import environment evidence is not unique")
    runtime_metadata_paths = tuple(
        sorted(
            (item.metadata_path for item in distributions if item.inventory_backed),
            key=lambda path: path.as_posix(),
        )
    )
    return ExecutionSnapshot(
        execution_contract=contract,
        execution_contract_sha256=execution_contract_sha256(contract),
        _source_files=source_files,
        _source_transient_files=source_transient_files,
        _source_directories=source_directories,
        _runtime_package_metadata=package_metadata,
        _runtime_package_payloads=package_payloads,
        _runtime_package_roots=package_roots,
        _runtime_metadata_paths=runtime_metadata_paths,
        _import_environment=import_environments[0],
    )


def recheck_execution_snapshot(snapshot: ExecutionSnapshot) -> ExecutionSnapshot:
    """Fail when source/runtime identity changed, even if bytes were restored."""

    captured_digest = execution_contract_sha256(snapshot.execution_contract)
    if captured_digest != snapshot.execution_contract_sha256:
        raise RuntimeError("captured execution snapshot is internally inconsistent")
    if tuple(item.evidence for item in snapshot._source_files) != (
        snapshot.execution_contract.source_files
    ):
        raise RuntimeError("captured execution snapshot is internally inconsistent")
    try:
        if not snapshot._import_environment.bytecode_writes_disabled:
            raise RuntimeError("captured bytecode-write policy is invalid")
        if not sys.dont_write_bytecode:
            raise RuntimeError("Python bytecode writes were re-enabled")
        files = (
            *snapshot._source_files,
            *snapshot._source_transient_files,
            *snapshot._runtime_package_metadata,
            *snapshot._runtime_package_payloads,
        )
        checked_files: set[Path] = set()
        for captured in files:
            if captured.physical_path in checked_files:
                continue
            checked_files.add(captured.physical_path)
            if not _captured_file_identity_matches(captured):
                raise RuntimeError("captured runtime file identity changed")
        directories = {
            item.path: item
            for item in (
                *snapshot._source_directories,
                *snapshot._runtime_package_roots,
            )
        }
        for path, captured in directories.items():
            if _capture_directory_identity(path) != captured:
                raise RuntimeError("captured runtime directory identity changed")
        if _current_runtime_metadata_paths() != snapshot._runtime_metadata_paths:
            raise RuntimeError("installed runtime package inventory changed")
        current_tokens, _active_paths, active_path_tokens = _active_import_search_state(
            dict(snapshot._import_environment.root_tokens),
            allowed_namespace_placeholders=frozenset(
                snapshot._import_environment.namespace_placeholders
            ),
        )
        current_signature = hashlib.sha256(
            json.dumps(
                list(current_tokens),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        if current_signature != snapshot._import_environment.signature:
            raise RuntimeError("active Python import precedence changed")
        _require_unchanged_import_hooks(
            snapshot._import_environment.meta_path,
            tuple(sys.meta_path),
            label="sys.meta_path",
        )
        _require_unchanged_import_hooks(
            snapshot._import_environment.path_hooks,
            tuple(sys.path_hooks),
            label="sys.path_hooks",
        )
        _reject_import_precedence_overlaps(
            active_path_tokens,
            dict(snapshot._import_environment.protected_origins),
        )
        _validate_loaded_module_origins(snapshot._import_environment)
        _require_bounded_runtime_execution(snapshot._import_environment)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "source code or runtime package files changed during the operation"
        ) from error
    return snapshot


def execution_contract_sha256(contract: ExecutionContract) -> str:
    """Hash a validated execution contract using canonical portable JSON."""

    return _model_sha256(contract)


def capture_base_model_snapshot(
    settings: ProjectSettings,
) -> BaseModelSnapshot:
    """Verify and capture the pinned files used by local model execution."""

    model_directory = settings.model.directory
    _require_safe_relative_path(model_directory, "base-model path")
    configured_path = PROJECT_ROOT / model_directory
    if configured_path.is_symlink():
        raise ValueError(f"base-model directory is a symbolic link: {configured_path}")
    model_path = _project_path(model_directory, "base-model path")
    expected_files = tuple(
        ExpectedFileHash(path=path, sha256=digest)
        for path, digest in sorted(settings.model.verified_runtime_files.items())
    )
    directory_before = _capture_directory_identity(
        model_path,
        label="base-model directory",
    )
    model_files = _capture_verified_model_files(
        model_path,
        model_revision=settings.model.revision,
        model_runtime_files=expected_files,
    )
    directory_after = _capture_directory_identity(
        model_path,
        label="base-model directory",
    )
    if directory_after != directory_before:
        raise RuntimeError("base-model directory changed while it was captured")
    file_evidence = tuple(item.evidence for item in model_files)
    contract = BaseModelExecutionContract(
        repository=settings.model.repo,
        model_path=model_directory,
        model_revision=settings.model.revision,
        model_files=file_evidence,
        model_files_sha256=_evidence_sequence_sha256(file_evidence),
    )
    return BaseModelSnapshot(
        execution_contract=contract,
        execution_contract_sha256=base_model_execution_contract_sha256(contract),
        _model_path=model_path,
        _model_directory=directory_before,
        _model_files=model_files,
    )


def recheck_base_model_snapshot(snapshot: BaseModelSnapshot) -> BaseModelSnapshot:
    """Fail if the verified model directory, revision, or runtime files drifted."""

    captured_digest = base_model_execution_contract_sha256(snapshot.execution_contract)
    if captured_digest != snapshot.execution_contract_sha256:
        raise RuntimeError("captured base-model snapshot is internally inconsistent")
    expected_files = tuple(
        ExpectedFileHash(path=item.path, sha256=item.sha256)
        for item in snapshot.execution_contract.model_files
        if item.path != _MODEL_REVISION_FILE
    )
    try:
        current_directory = _capture_directory_identity(
            snapshot._model_path,
            label="base-model directory",
        )
        current_files = _capture_verified_model_files(
            snapshot._model_path,
            model_revision=snapshot.execution_contract.model_revision,
            model_runtime_files=expected_files,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "base-model revision or runtime files changed during the operation"
        ) from error
    current_evidence = tuple(item.evidence for item in current_files)
    current_contract = BaseModelExecutionContract(
        repository=snapshot.execution_contract.repository,
        model_path=snapshot.execution_contract.model_path,
        model_revision=snapshot.execution_contract.model_revision,
        model_files=current_evidence,
        model_files_sha256=_evidence_sequence_sha256(current_evidence),
    )
    if (
        current_directory != snapshot._model_directory
        or current_files != snapshot._model_files
        or current_contract != snapshot.execution_contract
    ):
        raise RuntimeError(
            "base-model revision or runtime files changed during the operation"
        )
    return snapshot


def base_model_execution_contract_sha256(
    contract: BaseModelExecutionContract,
) -> str:
    """Hash portable base-model evidence using canonical JSON."""

    return _model_sha256(contract)


def _capture_governed_source_state(
    *,
    context: _RuntimeCaptureContext | None = None,
) -> tuple[
    tuple[_CapturedFile, ...],
    tuple[_CapturedFile, ...],
    tuple[_CapturedDirectory, ...],
]:
    capture_context = context or _RuntimeCaptureContext()
    source_root = _project_path(_SOURCE_PACKAGE_PATH.as_posix(), "source package")
    if not source_root.is_dir() or source_root.is_symlink():
        raise FileNotFoundError(f"source package directory is missing: {source_root}")
    tree = capture_context.capture_tree(source_root)
    paths = [path for _relative, path in tree.files]
    if not any(path.suffix == ".py" for path in paths):
        raise FileNotFoundError(
            f"source package contains no Python files: {source_root}"
        )
    notebook_paths = tuple(
        _project_path(relative.as_posix(), "notebook source")
        for relative in _NOTEBOOK_SOURCE_PATHS
    )
    notebook_directories: list[_CapturedDirectory] = []
    for directory in sorted({path.parent for path in notebook_paths}):
        notebook_directories.append(
            _capture_directory_identity(directory, label="notebook source directory")
        )
    files = tuple(
        capture_context.capture_file(path, _project_relative(path))
        for path in sorted((*paths, *notebook_paths), key=_project_relative)
    )
    for captured_directory in notebook_directories:
        after = _capture_directory_identity(
            captured_directory.path,
            label="notebook source directory",
        )
        if after != captured_directory:
            raise RuntimeError(
                "notebook source directory changed while it was captured: "
                f"{captured_directory.path}"
            )
        capture_context.observe_directory(captured_directory)
    transient_files = tuple(
        capture_context.capture_bytecode(path, _project_relative(path))
        for _relative, path in tree.bytecode_files
    )
    directories = tuple(
        sorted(
            (*tree.directories, *notebook_directories),
            key=lambda item: item.path.as_posix(),
        )
    )
    if context is None:
        files = _bind_instrumented_bytecode(
            files,
            transient_files,
            context=capture_context,
        )
        capture_context.verify_directories()
    return files, transient_files, directories


def _capture_runtime_packages() -> tuple[RuntimePackageEvidence, ...]:
    distributions = _runtime_distribution_inventory()
    if not distributions:
        raise RuntimeError("installed runtime package set is empty")
    return tuple(item.evidence for item in distributions)


def _runtime_package_state(
    distributions: tuple[_RuntimeDistribution, ...],
) -> tuple[
    tuple[_CapturedFile, ...],
    tuple[_CapturedFile, ...],
    tuple[_CapturedDirectory, ...],
]:
    """Return package metadata, payload files, and installation-root identities."""

    metadata_files: list[_CapturedFile] = []
    payload_files: list[_CapturedFile] = []
    roots: dict[Path, _CapturedDirectory] = {}
    for distribution in distributions:
        metadata_files.append(distribution.metadata_file)
        payload_files.extend(distribution.payload_files)
        payload_files.extend(distribution.transient_files)
        for root in (
            distribution.install_root_identity,
            *distribution.runtime_roots,
        ):
            previous = roots.setdefault(root.path, root)
            if previous != root:
                raise RuntimeError(
                    "runtime package directory identity is inconsistent: "
                    f"{root.path}"
                )

    captured_roots = tuple(
        roots[path] for path in sorted(roots, key=lambda item: item.as_posix())
    )
    if not captured_roots:
        raise RuntimeError("installed runtime package roots are missing")
    for captured in captured_roots:
        if _capture_directory_identity(captured.path) != captured:
            raise RuntimeError(
                "runtime package directory changed while it was captured: "
                f"{captured.path}"
            )
    return (
        tuple(metadata_files),
        tuple(payload_files),
        captured_roots,
    )


def _runtime_distribution_inventory(
    *,
    context: _RuntimeCaptureContext | None = None,
) -> tuple[_RuntimeDistribution, ...]:
    """Return distinct metadata installations in a deterministic order.

    Provider libraries can temporarily add a vendored distribution root to
    ``sys.path``. The same normalized project name can then legitimately appear
    at multiple versions. Preserve each distinct metadata file in the portable
    package multiset, while collapsing repeated discovery of the same physical
    metadata path.
    """

    capture_context = context or _RuntimeCaptureContext()
    by_metadata_path: dict[Path, _RuntimeDistribution] = {}
    owned_top_levels: dict[Path, set[str]] = {}
    for distribution in importlib.metadata.distributions():
        inventory = _require_distribution_file_inventory(distribution)
        located_root = Path(distribution.locate_file(""))
        _require_no_lexical_symlink_components(
            located_root,
            label="runtime package installation root",
        )
        if located_root.is_symlink():
            raise ValueError(
                f"runtime package installation root is a symbolic link: {located_root}"
            )
        install_root = located_root.resolve(strict=True)
        install_root_identity = _capture_directory_identity(
            install_root,
            label="runtime package installation root",
        )
        capture_context.observe_directory(install_root_identity)
        owned_top_levels.setdefault(install_root, set()).update(
            _distribution_owned_top_levels(inventory)
        )
        metadata_path = _distribution_metadata_path(distribution, inventory)
        resolved_metadata = _require_runtime_location(
            metadata_path,
            install_root=install_root,
            label="runtime package metadata",
        )
        if resolved_metadata in by_metadata_path:
            continue
        (
            payload_files,
            transient_files,
            runtime_roots,
            import_bindings,
        ) = _capture_runtime_package_payloads(
            distribution,
            install_root=install_root,
            inventory=inventory,
            metadata_path=resolved_metadata,
            context=capture_context,
        )
        evidence = _runtime_package_evidence(distribution, payload_files)
        metadata_display_path = (
            "runtime-package-metadata/"
            + hashlib.sha256(resolved_metadata.as_posix().encode()).hexdigest()
        )
        captured = _RuntimeDistribution(
            evidence=evidence,
            metadata_path=resolved_metadata,
            metadata_file=capture_context.capture_file(
                resolved_metadata,
                metadata_display_path,
            ),
            install_root=install_root,
            install_root_identity=install_root_identity,
            payload_files=payload_files,
            transient_files=transient_files,
            runtime_roots=runtime_roots,
            import_bindings=import_bindings,
        )
        by_metadata_path[resolved_metadata] = captured
    unowned_entries = _runtime_installation_unowned_entries(owned_top_levels)
    for install_root, entries in unowned_entries.items():
        bootstrap = _capture_virtualenv_bootstrap(
            install_root,
            entries=entries,
            context=capture_context,
        )
        if bootstrap.metadata_path in by_metadata_path:
            raise RuntimeError(
                "runtime environment bootstrap collides with distribution metadata"
            )
        by_metadata_path[bootstrap.metadata_path] = bootstrap
    captured_distributions = _bind_distribution_bytecode(
        tuple(by_metadata_path.values()),
        context=capture_context,
    )
    import_environment = _capture_import_environment(
        captured_distributions,
        context=capture_context,
    )
    result = tuple(
        sorted(
            (*captured_distributions, import_environment),
            key=lambda item: (
                *_runtime_package_sort_key(item.evidence),
                item.metadata_path.as_posix(),
            ),
        )
    )
    if context is None:
        capture_context.verify_directories()
    return result


def _runtime_package_evidence(
    distribution: importlib.metadata.Distribution,
    payload_files: tuple[_CapturedFile, ...],
) -> RuntimePackageEvidence:
    raw_name = distribution.metadata.get("Name")
    raw_version = distribution.version
    if not raw_name or not raw_version:
        raise RuntimeError("installed distribution metadata is incomplete")
    name = re.sub(r"[-_.]+", "-", raw_name).lower()
    payload_evidence = tuple(item.evidence for item in payload_files)
    return RuntimePackageEvidence(
        name=name,
        version=raw_version,
        payload_file_count=len(payload_evidence),
        payload_size_bytes=sum(item.size_bytes for item in payload_evidence),
        payload_files_sha256=_evidence_sequence_sha256(payload_evidence),
    )


def _runtime_package_sort_key(
    package: RuntimePackageEvidence,
) -> tuple[str, str, str, int, int]:
    return (
        package.name,
        package.version,
        package.payload_files_sha256,
        package.payload_file_count,
        package.payload_size_bytes,
    )


def _bind_instrumented_bytecode(
    persistent_files: tuple[_CapturedFile, ...],
    bytecode_files: tuple[_CapturedFile, ...],
    *,
    context: _RuntimeCaptureContext,
) -> tuple[_CapturedFile, ...]:
    semantics = context.validate_bytecode()
    markers = [
        _captured_file_with_canonical_bytes(
            captured,
            captured.evidence.path,
            f"python-bytecode-semantic-v1:{digest}".encode("ascii"),
        )
        for captured in bytecode_files
        if (digest := semantics[captured.physical_path]) is not None
    ]
    return tuple(
        sorted(
            (*persistent_files, *markers),
            key=lambda item: item.evidence.path,
        )
    )


def _bind_distribution_bytecode(
    distributions: tuple[_RuntimeDistribution, ...],
    *,
    context: _RuntimeCaptureContext,
) -> tuple[_RuntimeDistribution, ...]:
    result: list[_RuntimeDistribution] = []
    for distribution in distributions:
        payload_files = _bind_instrumented_bytecode(
            distribution.payload_files,
            distribution.transient_files,
            context=context,
        )
        evidence_files = tuple(item.evidence for item in payload_files)
        evidence = RuntimePackageEvidence(
            name=distribution.evidence.name,
            version=distribution.evidence.version,
            payload_file_count=len(payload_files),
            payload_size_bytes=sum(item.size_bytes for item in evidence_files),
            payload_files_sha256=_evidence_sequence_sha256(evidence_files),
        )
        result.append(
            replace(
                distribution,
                evidence=evidence,
                payload_files=payload_files,
            )
        )
    return tuple(result)


def _capture_loaded_module_bytecode_caches(
    protected_origins: dict[str, str],
    *,
    context: _RuntimeCaptureContext,
) -> tuple[_CapturedFile, ...]:
    captured: dict[Path, _CapturedFile] = {}
    for module_name, module in _runtime_loaded_modules():
        if module_name.partition(".")[0] not in protected_origins:
            continue
        spec = getattr(module, "__spec__", None)
        if spec is None:
            continue
        try:
            source_path = _loaded_module_origin_path(spec)
        except (OSError, RuntimeError, ValueError):
            continue
        if source_path is None or source_path.suffix != ".py":
            continue
        raw_cache = getattr(module, "__cached__", None)
        if not isinstance(raw_cache, str) or not raw_cache:
            continue
        cache_path = Path(raw_cache)
        if not cache_path.is_absolute():
            cache_path = Path(os.path.abspath(cache_path))
        _require_no_lexical_symlink_components(
            cache_path,
            label="loaded module bytecode cache",
        )
        if cache_path.is_symlink() or not cache_path.is_file():
            continue
        cache_path = cache_path.resolve(strict=True)
        logical_path = "loaded-module/" + module_name.replace(".", "/") + ".pyc"
        captured[cache_path] = context.capture_bytecode(cache_path, logical_path)
        context.observe_directory(
            _capture_directory_identity(
                cache_path.parent,
                label="loaded module bytecode directory",
            )
        )
    return tuple(
        captured[path] for path in sorted(captured, key=lambda item: item.as_posix())
    )


def _portable_import_hook(item: object, *, label: str) -> str:
    if isinstance(item, type):
        kind = "class"
        module = getattr(item, "__module__", None)
        qualname = getattr(item, "__qualname__", None)
    elif isinstance(item, types.MethodType):
        kind = "method"
        function = item.__func__
        owner = item.__self__
        module = getattr(function, "__module__", None)
        qualname = getattr(function, "__qualname__", None)
        owner_type = owner if isinstance(owner, type) else type(owner)
        owner_module = getattr(owner_type, "__module__", None)
        owner_qualname = getattr(owner_type, "__qualname__", None)
        if not isinstance(owner_module, str) or not isinstance(owner_qualname, str):
            raise RuntimeError(f"{label} contains an unidentifiable bound method")
        qualname = f"{owner_module}.{owner_qualname}:{qualname}"
    elif isinstance(item, types.FunctionType):
        kind = "function"
        module = getattr(item, "__module__", None)
        qualname = getattr(item, "__qualname__", None)
    else:
        kind = "instance"
        item_type = type(item)
        module = getattr(item_type, "__module__", None)
        qualname = getattr(item_type, "__qualname__", None)
    if (
        not isinstance(module, str)
        or not module
        or not isinstance(qualname, str)
        or not qualname
    ):
        raise RuntimeError(f"{label} contains an unidentifiable import hook")
    return f"{kind}:{module}:{qualname}"


def _capture_import_hooks(
    items: tuple[object, ...],
    *,
    label: str,
) -> _CapturedImportHooks:
    return _CapturedImportHooks(
        portable=tuple(_portable_import_hook(item, label=label) for item in items),
        objects=items,
    )


def _require_unchanged_import_hooks(
    captured: _CapturedImportHooks,
    current: tuple[object, ...],
    *,
    label: str,
) -> None:
    observed = _capture_import_hooks(current, label=label)
    if observed.portable != captured.portable or len(current) != len(captured.objects):
        raise RuntimeError(f"{label} import precedence changed")
    pairs = zip(current, captured.objects, strict=True)
    if any(actual is not expected for actual, expected in pairs):
        raise RuntimeError(f"{label} import hook identity changed")


def _runtime_execution_counts() -> tuple[tuple[Path, int], ...]:
    with _RUNTIME_PROVENANCE_LOCK:
        return tuple(
            sorted(
                _RUNTIME_EXECUTION_COUNTS.items(),
                key=lambda item: item[0].as_posix(),
            )
        )


def _capture_import_environment(
    distributions: tuple[_RuntimeDistribution, ...],
    *,
    context: _RuntimeCaptureContext,
) -> _RuntimeDistribution:
    """Bind effective import precedence and the covered origin for each name."""

    _install_runtime_extension_finder()
    if not distributions:
        raise RuntimeError("installed runtime package set is empty")
    by_install_root: dict[Path, list[_RuntimeDistribution]] = {}
    by_path_root: dict[Path, list[_RuntimeDistribution]] = {}
    for distribution in distributions:
        by_install_root.setdefault(distribution.install_root, []).append(distribution)
        for binding in distribution.import_bindings:
            if binding.exposure == "path":
                if binding.search_root is None:
                    raise RuntimeError("runtime path import is missing its search root")
                by_path_root.setdefault(binding.search_root, []).append(distribution)

    governed_root = _project_path(
        _SOURCE_PACKAGE_PATH.as_posix(),
        "source package",
    )
    governed_search_root = governed_root.parent
    governed_name = governed_root.name
    governed_token = _project_import_token(governed_search_root)
    root_tokens = _python_stdlib_root_tokens()
    for prefix, grouped in (
        ("packages", by_install_root),
        ("editable-path", by_path_root),
    ):
        for root, providers in grouped.items():
            canonical = json.dumps(
                [
                    item.evidence.model_dump(mode="json")
                    for item in sorted(
                        {item.metadata_path: item for item in providers}.values(),
                        key=lambda item: _runtime_package_sort_key(item.evidence),
                    )
                ],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            token = (
                governed_token
                if prefix == "editable-path" and root == governed_search_root
                else f"{prefix}:{hashlib.sha256(canonical).hexdigest()}"
            )
            previous = root_tokens.setdefault(root, token)
            if previous != token:
                raise RuntimeError("runtime import root has ambiguous ownership")

    previous = root_tokens.setdefault(governed_search_root, governed_token)
    if previous != governed_token:
        raise RuntimeError("governed source import root has ambiguous ownership")

    active_candidates = _active_existing_search_roots()
    governed_surface = _capture_python_import_surface(
        governed_search_root,
        logical_prefix="governed-source-entry",
        context=context,
    )
    surfaces: list[tuple[Path, str, _CapturedImportSurface]] = [
        (governed_search_root, governed_token, governed_surface)
    ]
    project_root = PROJECT_ROOT.resolve(strict=True)
    environment_scripts = Path(sysconfig.get_path("scripts")).resolve(strict=True)
    for surface_root, role in (
        (project_root, "project-entry"),
        (environment_scripts, "environment-scripts"),
    ):
        if surface_root not in active_candidates or surface_root in root_tokens:
            continue
        surface = _capture_python_import_surface(
            surface_root,
            logical_prefix=role,
            context=context,
        )
        token = _captured_import_surface_token(
            surface,
            root=surface_root,
            role=role,
        )
        root_tokens[surface_root] = token
        surfaces.append((surface_root, token, surface))

    allowed_namespace_placeholders = frozenset(
        binding.namespace_placeholder
        for distribution in distributions
        for binding in distribution.import_bindings
        if binding.namespace_placeholder is not None
    )

    active_tokens, active_paths, active_path_tokens = _active_import_search_state(
        root_tokens,
        allowed_namespace_placeholders=allowed_namespace_placeholders,
    )
    active_install_roots = set(by_install_root).intersection(active_paths)
    protected: dict[str, set[str]] = {}
    protected_owners: dict[str, set[tuple[str, Path]]] = {}
    origin_paths: dict[tuple[Path, str], None] = {}
    canonical_bindings: list[tuple[str, str, str, str]] = []
    for distribution in distributions:
        if distribution.install_root not in active_install_roots:
            continue
        finder_token = (
            "editable-finder:"
            + hashlib.sha256(
                json.dumps(
                    distribution.evidence.model_dump(mode="json"),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
            ).hexdigest()
        )
        for binding in distribution.import_bindings:
            if (
                binding.name == governed_name
                and binding.physical_path.resolve(strict=True) == governed_root
            ):
                mechanism = governed_token
                origin_root = governed_search_root
                relative = governed_root.name
                owner = ("governed", governed_search_root)
            elif binding.exposure == "direct":
                mechanism = root_tokens[distribution.install_root]
                origin_root = distribution.install_root
                relative = binding.physical_path.relative_to(origin_root).as_posix()
                owner = ("root", origin_root)
            elif binding.exposure == "path":
                assert binding.search_root is not None
                if binding.search_root not in active_paths:
                    raise RuntimeError(
                        "runtime path configuration is not present in active sys.path: "
                        f"{binding.search_root}"
                    )
                mechanism = root_tokens[binding.search_root]
                origin_root = binding.search_root
                relative = binding.physical_path.relative_to(origin_root).as_posix()
                owner = ("root", origin_root)
            else:
                mechanism = finder_token
                origin_root = binding.physical_path
                relative = f"finder/{binding.name}/{binding.kind}"
                owner = ("finder", distribution.metadata_path)
            protected.setdefault(binding.name, set()).add(mechanism)
            protected_owners.setdefault(binding.name, set()).add(owner)
            origin_paths[(origin_root, mechanism)] = None
            canonical_bindings.append((binding.name, mechanism, binding.kind, relative))

    surface_persistent_files: list[_CapturedFile] = []
    surface_transient_files: list[_CapturedFile] = []
    surface_directories: dict[Path, _CapturedDirectory] = {}
    for surface_root, mechanism, surface in surfaces:
        surface_persistent_files.extend(surface.persistent_files)
        surface_transient_files.extend(surface.transient_files)
        for directory in surface.directories:
            surface_directories[directory.path] = directory
        for binding in surface.import_bindings:
            if (
                surface_root == governed_search_root
                and binding.name == governed_name
                and binding.physical_path.resolve(strict=True) == governed_root
            ):
                continue
            protected.setdefault(binding.name, set()).add(mechanism)
            protected_owners.setdefault(binding.name, set()).add(("root", surface_root))
            origin_paths[(surface_root, mechanism)] = None
            relative = binding.physical_path.relative_to(surface_root).as_posix()
            canonical_bindings.append((binding.name, mechanism, binding.kind, relative))

    protected.setdefault(governed_name, set()).add(governed_token)
    protected_owners.setdefault(governed_name, set()).add(
        ("governed", governed_search_root)
    )
    origin_paths[(governed_search_root, governed_token)] = None
    # An embedding caller may supply a governed project root after this package
    # was imported. Retain the package's actual source root as an equivalent
    # governed origin; the normal application resolves both paths identically.
    loaded_source_root = Path(__file__).resolve(strict=True).parents[1]
    origin_paths[(loaded_source_root, governed_token)] = None
    canonical_bindings.append(
        (governed_name, governed_token, "package", governed_root.name)
    )

    for root, token in active_path_tokens:
        if not token.startswith("python-runtime:"):
            origin_paths[(root, token)] = None
            captured_root = _capture_directory_identity(
                root,
                label="governed Python import root",
            )
            context.observe_directory(captured_root)
            surface_directories.setdefault(root, captured_root)

    overlaps = {
        name
        for name, mechanisms in protected.items()
        if len(mechanisms) != 1 or len(protected_owners.get(name, ())) != 1
    }
    if overlaps:
        raise RuntimeError(
            "runtime import names have overlapping origins: "
            + ", ".join(sorted(overlaps))
        )
    protected_origins = {
        name: next(iter(mechanisms)) for name, mechanisms in protected.items()
    }
    _reject_import_precedence_overlaps(
        active_path_tokens,
        protected_origins,
    )
    surface_transient_files.extend(
        _capture_loaded_module_bytecode_caches(
            protected_origins,
            context=context,
        )
    )
    context.validate_bytecode()
    meta_path = _capture_import_hooks(tuple(sys.meta_path), label="sys.meta_path")
    path_hooks = _capture_import_hooks(
        tuple(sys.path_hooks),
        label="sys.path_hooks",
    )
    state_payload = {
        "schema": 2,
        "active_sys_path": list(active_tokens),
        "meta_path": list(meta_path.portable),
        "path_hooks": list(path_hooks.portable),
        "bytecode_writes_disabled": True,
        "origins": sorted(canonical_bindings),
    }
    canonical = json.dumps(
        state_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    anchor = min(distributions, key=lambda item: item.metadata_path.as_posix())
    marker = _captured_file_with_canonical_bytes(
        anchor.metadata_file,
        _runtime_payload_display_path("python-import-environment.json"),
        canonical,
    )
    payload_files = tuple(
        sorted(
            (marker, *surface_persistent_files),
            key=lambda item: item.evidence.path,
        )
    )
    payload_evidence = tuple(item.evidence for item in payload_files)
    evidence = RuntimePackageEvidence(
        name="python-import-environment",
        version="1",
        payload_file_count=len(payload_files),
        payload_size_bytes=sum(item.size_bytes for item in payload_evidence),
        payload_files_sha256=_evidence_sequence_sha256(payload_evidence),
    )
    captured_files = _captured_physical_file_set(
        (
            *context.files.values(),
            *(
                captured
                for distribution in distributions
                for captured in (
                    distribution.metadata_file,
                    *distribution.payload_files,
                    *distribution.transient_files,
                )
            ),
            *surface_persistent_files,
            *surface_transient_files,
        )
    )
    state = _CapturedImportEnvironment(
        signature=hashlib.sha256(
            json.dumps(
                list(active_tokens),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
        active_tokens=active_tokens,
        root_tokens=tuple(
            sorted(root_tokens.items(), key=lambda item: item[0].as_posix())
        ),
        protected_origins=tuple(sorted(protected_origins.items())),
        origin_paths=tuple(
            sorted(origin_paths, key=lambda item: (item[0].as_posix(), item[1]))
        ),
        namespace_placeholders=tuple(sorted(allowed_namespace_placeholders)),
        captured_files=captured_files,
        loaded_modules=(),
        spec_less_modules=(),
        meta_path=meta_path,
        path_hooks=path_hooks,
        execution_counts=_runtime_execution_counts(),
        bytecode_writes_disabled=True,
    )
    loaded_modules, spec_less_modules = _validate_loaded_module_origins(
        state,
        tuple(origin_paths),
        capture_initial=True,
    )
    state = replace(
        state,
        loaded_modules=loaded_modules,
        spec_less_modules=spec_less_modules,
    )
    return _RuntimeDistribution(
        evidence=evidence,
        metadata_path=anchor.metadata_path.with_name(
            anchor.metadata_path.name + ".import-environment"
        ),
        metadata_file=marker,
        install_root=anchor.install_root,
        install_root_identity=anchor.install_root_identity,
        payload_files=payload_files,
        transient_files=tuple(
            sorted(
                surface_transient_files,
                key=lambda item: item.evidence.path,
            )
        ),
        runtime_roots=tuple(
            surface_directories[path]
            for path in sorted(surface_directories, key=lambda item: item.as_posix())
        ),
        import_bindings=(),
        import_environment=state,
        inventory_backed=False,
    )


def _current_runtime_metadata_paths() -> tuple[Path, ...]:
    """Rediscover distribution identities without rehashing their payload trees."""

    paths: set[Path] = set()
    for distribution in importlib.metadata.distributions():
        inventory = _require_distribution_file_inventory(distribution)
        located_root = Path(distribution.locate_file(""))
        _require_no_lexical_symlink_components(
            located_root,
            label="runtime package installation root",
        )
        install_root = located_root.resolve(strict=True)
        metadata_path = _distribution_metadata_path(distribution, inventory)
        paths.add(
            _require_runtime_location(
                metadata_path,
                install_root=install_root,
                label="runtime package metadata",
            )
        )
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _project_import_token(path: Path) -> str:
    project_root = PROJECT_ROOT.resolve(strict=True)
    relative = path.relative_to(project_root)
    return "project:" + (relative.as_posix() if relative.parts else ".")


def _python_stdlib_root_tokens() -> dict[Path, str]:
    """Return only the interpreter roots that are valid stdlib search entries."""

    configured = (
        ("stdlib", sysconfig.get_path("stdlib")),
        ("extensions", sysconfig.get_config_var("DESTSHARED")),
    )
    roles_by_root: dict[Path, list[str]] = {}
    for role, raw_path in configured:
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeError(f"Python {role} import root is unavailable")
        candidate = Path(raw_path)
        _require_no_lexical_symlink_components(
            candidate,
            label=f"Python {role} import root",
        )
        if candidate.is_symlink():
            raise ValueError(
                f"Python {role} import root is a symbolic link: {candidate}"
            )
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(
                f"Python {role} import root is not a directory: {candidate}"
            )
        roles_by_root.setdefault(resolved, []).append(role)
    return {
        root: "python-runtime:" + "+".join(sorted(roles))
        for root, roles in roles_by_root.items()
    }


def _python_missing_stdlib_archives(base_prefix: Path) -> dict[Path, str]:
    version = f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    candidates = (
        base_prefix / version,
        base_prefix / "lib" / version,
    )
    return {
        candidate: f"python-runtime-missing-zip:{index}"
        for index, candidate in enumerate(candidates, 1)
    }


def _active_existing_search_roots() -> frozenset[Path]:
    """Resolve existing filesystem entries only to select roots for capture."""

    roots: set[Path] = set()
    for raw_entry in sys.path:
        if not isinstance(raw_entry, str) or raw_entry.endswith(".__path_hook__"):
            continue
        candidate = Path(raw_entry or os.getcwd())
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved.is_dir():
            roots.add(resolved)
    return frozenset(roots)


def _capture_python_import_surface(
    root: Path,
    *,
    logical_prefix: str,
    context: _RuntimeCaptureContext,
) -> _CapturedImportSurface:
    """Capture code reachable through a governed interpreter entry directory."""

    root_before = _capture_directory_identity(root, label="Python entry import root")
    context.observe_directory(root_before)
    persistent: dict[str, _CapturedFile] = {}
    transient: dict[str, _CapturedFile] = {}
    directories: dict[Path, _CapturedDirectory] = {root: root_before}
    bindings: list[_RuntimeImportBinding] = []

    def logical_path(relative: PurePosixPath) -> str:
        return PurePosixPath(logical_prefix, relative).as_posix()

    def add_regular_file(path: Path, relative: PurePosixPath) -> bool:
        binding = _runtime_import_binding(path)
        if binding is None and path.suffix != ".pyc":
            return False
        portable = logical_path(relative)
        if path.suffix == ".pyc":
            transient[portable] = context.capture_bytecode(path, portable)
        else:
            persistent[portable] = context.capture_file(
                path,
                _runtime_payload_display_path(portable),
            )
        return True

    def add_package_tree(path: Path, relative: PurePosixPath) -> None:
        tree = context.capture_tree(path)
        for directory in tree.directories:
            directories[directory.path] = directory
        for child, physical in tree.files:
            child_relative = relative / PurePosixPath(child)
            portable = logical_path(child_relative)
            persistent[portable] = context.capture_file(
                physical,
                _runtime_payload_display_path(portable),
            )
        for child, physical in tree.bytecode_files:
            child_relative = relative / PurePosixPath(child)
            portable = logical_path(child_relative)
            transient[portable] = context.capture_bytecode(physical, portable)

    def visit_namespace(path: Path, relative: PurePosixPath) -> None:
        before = _capture_directory_identity(path, label="Python namespace directory")
        context.observe_directory(before)
        directories[path] = before
        with os.scandir(path) as entries:
            ordered_entries = sorted(entries, key=lambda item: item.name)
        for entry in ordered_entries:
            physical = Path(entry.path)
            child_relative = relative / entry.name
            if entry.is_symlink():
                if (
                    _is_importable_top_level(physical)
                    or physical.suffix == ".pyc"
                    or entry.name == "__pycache__"
                ):
                    raise ValueError(
                        "Python entry import surface contains a symbolic link: "
                        f"{physical}"
                    )
                continue
            if entry.is_dir(follow_symlinks=False):
                if entry.name == "__pycache__":
                    cache_before = _capture_directory_identity(
                        physical,
                        label="Python entry bytecode cache",
                    )
                    context.observe_directory(cache_before)
                    directories[physical] = cache_before
                    with os.scandir(physical) as cache_entries:
                        ordered_cache = sorted(
                            cache_entries, key=lambda item: item.name
                        )
                    for cache_entry in ordered_cache:
                        cache_path = Path(cache_entry.path)
                        if cache_entry.is_symlink() or not cache_entry.is_file(
                            follow_symlinks=False
                        ):
                            raise ValueError(
                                "Python entry bytecode cache contains an unsupported "
                                f"path: {cache_path}"
                            )
                        if cache_path.suffix != ".pyc":
                            raise ValueError(
                                "Python entry bytecode cache contains a non-bytecode "
                                f"file: {cache_path}"
                            )
                        add_regular_file(cache_path, child_relative / cache_entry.name)
                    cache_after = _capture_directory_identity(
                        physical,
                        label="Python entry bytecode cache",
                    )
                    if cache_after != cache_before:
                        raise RuntimeError(
                            "Python entry bytecode cache changed while it was "
                            f"captured: {physical}"
                        )
                    continue
                if not entry.name.isidentifier():
                    continue
                initializer = physical / "__init__.py"
                if initializer.is_symlink():
                    raise ValueError(
                        "Python entry package initializer is a symbolic link: "
                        f"{initializer}"
                    )
                if initializer.is_file():
                    add_package_tree(physical, child_relative)
                else:
                    visit_namespace(physical, child_relative)
            elif entry.is_file(follow_symlinks=False):
                add_regular_file(physical, child_relative)
            else:
                raise ValueError(
                    "Python entry import surface contains a non-regular path: "
                    f"{physical}"
                )
        after = _capture_directory_identity(path, label="Python namespace directory")
        if after != before:
            raise RuntimeError(
                f"Python namespace directory changed while it was captured: {path}"
            )

    with os.scandir(root) as entries:
        root_entries = sorted(entries, key=lambda item: item.name)
    for entry in root_entries:
        path = Path(entry.path)
        relative = PurePosixPath(entry.name)
        if entry.is_symlink():
            if _is_importable_top_level(path) or path.suffix == ".pyc":
                raise ValueError(
                    f"Python entry import root contains a symbolic link: {path}"
                )
            continue
        if entry.is_dir(follow_symlinks=False) and entry.name == "__pycache__":
            visit_namespace(path, relative)
            continue
        binding = _runtime_import_binding(
            path,
            exposure="path",
            search_root=root,
        )
        if binding is None:
            continue
        if path.is_dir():
            initializer = path / "__init__.py"
            if initializer.is_symlink():
                raise ValueError(
                    "Python entry package initializer is a symbolic link: "
                    f"{initializer}"
                )
            if initializer.is_file():
                add_package_tree(path, relative)
            else:
                visit_namespace(path, relative)
        else:
            add_regular_file(path, relative)
        bindings.append(binding)
    root_after = _capture_directory_identity(root, label="Python entry import root")
    if root_after != root_before:
        raise RuntimeError(f"Python entry import root changed while captured: {root}")
    persistent_result = tuple(persistent[path] for path in sorted(persistent))
    transient_result = tuple(transient[path] for path in sorted(transient))
    persistent_result = _bind_instrumented_bytecode(
        persistent_result,
        transient_result,
        context=context,
    )
    return _CapturedImportSurface(
        persistent_files=persistent_result,
        transient_files=transient_result,
        directories=tuple(
            directories[path]
            for path in sorted(directories, key=lambda item: item.as_posix())
        ),
        import_bindings=tuple(
            sorted(
                bindings, key=lambda item: (item.name, item.physical_path.as_posix())
            )
        ),
    )


def _captured_import_surface_token(
    surface: _CapturedImportSurface,
    *,
    root: Path,
    role: str,
) -> str:
    canonical = {
        "schema": 1,
        "role": role,
        "files": [
            item.evidence.model_dump(mode="json") for item in surface.persistent_files
        ],
        "directories": sorted(
            directory.path.relative_to(root).as_posix()
            for directory in surface.directories
        ),
        "bindings": sorted(
            (
                binding.name,
                binding.kind,
                binding.physical_path.relative_to(root).as_posix(),
            )
            for binding in surface.import_bindings
        ),
    }
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return f"{role}:{digest}"


def _active_import_search_state(
    root_tokens: dict[Path, str],
    *,
    allowed_namespace_placeholders: frozenset[str],
) -> tuple[
    tuple[str, ...],
    frozenset[Path],
    tuple[tuple[Path, str], ...],
]:
    """Canonicalize only effective sys.path entries, preserving precedence."""

    base_prefix = Path(sys.base_prefix).resolve(strict=True)
    tokens: list[str] = []
    active_paths: set[Path] = set()
    active_path_tokens: list[tuple[Path, str]] = []
    seen: set[Path | str] = set()
    for raw_entry in sys.path:
        if not isinstance(raw_entry, str):
            raise ValueError("active Python import root must be a string path")
        if raw_entry.endswith(".__path_hook__") and re.fullmatch(
            r"__editable__\.[A-Za-z0-9_.-]+\.finder\.__path_hook__",
            raw_entry,
        ):
            if raw_entry not in allowed_namespace_placeholders:
                raise ValueError(
                    "active setuptools namespace hook is not covered by "
                    f"editable evidence: {raw_entry}"
                )
            marker = f"setuptools-namespace:{raw_entry}"
            if marker not in seen:
                seen.add(marker)
                tokens.append(marker)
            continue
        candidate = Path(raw_entry or os.getcwd())
        if not candidate.is_absolute():
            candidate = Path(os.path.abspath(candidate))
        _require_no_lexical_symlink_components(
            candidate,
            label="active Python import root",
        )
        if not candidate.exists():
            missing_archives = _python_missing_stdlib_archives(base_prefix)
            if candidate not in missing_archives:
                raise ValueError(
                    f"active Python import root does not exist: {candidate}"
                )
            token = missing_archives[candidate]
            if token not in seen:
                seen.add(token)
                tokens.append(token)
            continue
        if candidate.is_symlink():
            raise ValueError(
                f"active Python import root is a symbolic link: {candidate}"
            )
        resolved = candidate.resolve(strict=True)
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved in root_tokens:
            token = root_tokens[resolved]
        else:
            raise ValueError(
                f"active Python import root is not governed by evidence: {resolved}"
            )
        if resolved.is_file():
            raise ValueError(f"active Python import archive is unsupported: {resolved}")
        if not resolved.is_dir():
            raise ValueError(
                f"active Python import root is not a directory: {resolved}"
            )
        active_paths.add(resolved)
        active_path_tokens.append((resolved, token))
        tokens.append(token)
    return tuple(tokens), frozenset(active_paths), tuple(active_path_tokens)


def _reject_import_precedence_overlaps(
    active_path_tokens: tuple[tuple[Path, str], ...],
    protected_origins: dict[str, str],
) -> None:
    for path, token in active_path_tokens:
        for name, expected in protected_origins.items():
            if token == expected:
                continue
            if _runtime_import_candidate(path, name) is not None:
                raise RuntimeError(
                    "active Python import precedence can shadow governed runtime "
                    f"name {name!r}: {path}"
                )


def _runtime_import_candidate(root: Path, name: str) -> Path | None:
    package = root / name
    if package.is_symlink():
        raise ValueError(
            f"active Python import root contains a symbolic-link candidate: {package}"
        )
    if package.is_dir():
        return package
    for suffix in (".py", ".pyc", *importlib.machinery.EXTENSION_SUFFIXES):
        candidate = root / f"{name}{suffix}"
        if candidate.is_symlink():
            raise ValueError(
                "active Python import root contains a symbolic-link candidate: "
                f"{candidate}"
            )
        if candidate.is_file():
            return candidate
    return None


def _validate_loaded_module_origins(
    state: _CapturedImportEnvironment,
    origin_paths: tuple[tuple[Path, str], ...] | None = None,
    *,
    capture_initial: bool = False,
) -> tuple[tuple[_LoadedModuleState, ...], tuple[_LoadedSpecLessModuleState, ...]]:
    # Initial capture passes exact finder/source roots. Recheck reconstructs the
    # direct/path roots from the stable root-token mapping below.
    roots = list(origin_paths if origin_paths is not None else state.origin_paths)
    expected = dict(state.protected_origins)
    captured_by_path = {item.path: item for item in state.captured_files}
    initial_modules = {item.name: item for item in state.loaded_modules}
    initial_spec_less = {item.name: item for item in state.spec_less_modules}
    if len(initial_modules) != len(state.loaded_modules):
        raise RuntimeError("captured loaded-module state contains duplicate names")
    if len(initial_spec_less) != len(state.spec_less_modules):
        raise RuntimeError("captured spec-less module state contains duplicate names")
    loaded_modules = _runtime_loaded_modules()
    observed_initial: set[str] = set()
    observed_spec_less: set[str] = set()
    captured_modules: list[_LoadedModuleState] = []
    captured_spec_less: list[_LoadedSpecLessModuleState] = []
    for module_name, module in loaded_modules:
        top_level = module_name.partition(".")[0]
        expected_token = expected.get(top_level)
        spec = getattr(module, "__spec__", None)
        if expected_token is None:
            if spec is None:
                captured_dynamic = _validate_spec_less_module(
                    module_name,
                    module,
                    state=state,
                    roots=roots,
                    captured_by_path=captured_by_path,
                    initial=initial_spec_less.get(module_name),
                )
                if module_name in initial_spec_less:
                    observed_spec_less.add(module_name)
                if capture_initial:
                    captured_spec_less.append(captured_dynamic)
            else:
                _reject_unbound_loaded_origin(
                    module_name,
                    spec,
                    state=state,
                )
            continue
        if spec is None:
            captured_dynamic = _validate_spec_less_module(
                module_name,
                module,
                state=state,
                roots=roots,
                captured_by_path=captured_by_path,
                initial=initial_spec_less.get(module_name),
            )
            if module_name in initial_spec_less:
                observed_spec_less.add(module_name)
            if capture_initial:
                captured_spec_less.append(captured_dynamic)
            continue
        origin = getattr(spec, "origin", None)
        if origin in {"built-in", "frozen"}:
            raise RuntimeError(
                f"loaded module origin violates captured precedence: {module_name}"
            )
        if origin is None:
            locations = tuple(getattr(spec, "submodule_search_locations", ()) or ())
            if not locations:
                raise RuntimeError(
                    f"loaded module lacks a resolvable origin: {module_name}"
                )
            for location in locations:
                if (
                    expected_token.startswith("editable-finder:")
                    and re.fullmatch(
                        r"__editable__\.[A-Za-z0-9_.-]+\.finder\.__path_hook__",
                        str(location),
                    )
                    and str(location) in state.namespace_placeholders
                ):
                    continue
                actual_token = _loaded_origin_token(Path(location), roots)
                if actual_token != expected_token:
                    raise RuntimeError(
                        "loaded namespace origin violates captured precedence: "
                        f"{module_name}"
                    )
            continue
        path = _loaded_module_origin_path(spec)
        if path is None:
            raise RuntimeError(f"loaded module lacks a file origin: {module_name}")
        actual_token = _loaded_origin_token(path, roots)
        if actual_token != expected_token:
            raise RuntimeError(
                f"loaded module origin violates captured precedence: {module_name}"
            )
        initial = initial_modules.get(module_name)
        captured = _validate_loaded_module_file(
            module_name,
            module,
            path,
            captured_by_path=captured_by_path,
            initial=initial,
        )
        if initial is not None:
            observed_initial.add(module_name)
        if capture_initial:
            captured_modules.append(captured)

    mlx_core = sys.modules.get("mlx.core")
    if mlx_core is not None:
        for child_module_name in _RUNTIME_MLX_SPECLESS_CHILDREN:
            child_name = child_module_name.rpartition(".")[2]
            child = getattr(mlx_core, child_name, None)
            if child is not None and sys.modules.get(child_module_name) is not child:
                raise RuntimeError(
                    "loaded native child was removed or replaced: "
                    f"{child_module_name}"
                )

    if not capture_initial:
        missing = set(initial_modules).difference(observed_initial)
        if missing:
            raise RuntimeError(
                "loaded modules present at evidence capture were removed: "
                + ", ".join(sorted(missing))
            )
        missing_spec_less = set(initial_spec_less).difference(observed_spec_less)
        if missing_spec_less:
            raise RuntimeError(
                "spec-less modules present at evidence capture were removed: "
                + ", ".join(sorted(missing_spec_less))
            )
    return (
        tuple(sorted(captured_modules, key=lambda item: item.name)),
        tuple(sorted(captured_spec_less, key=lambda item: item.name)),
    )


def _supported_spec_less_module_name(name: str) -> bool:
    return (
        name in _RUNTIME_SPECLESS_MODULE_NAMES
        or name == "cython_runtime"
        or re.fullmatch(
            r"_cython_\d+(?:_\d+)+",
            name,
        )
        is not None
    )


def _validate_spec_less_module(
    name: str,
    module: object,
    *,
    state: _CapturedImportEnvironment,
    roots: list[tuple[Path, str]],
    captured_by_path: dict[Path, _CapturedPhysicalFile],
    initial: _LoadedSpecLessModuleState | None,
) -> _LoadedSpecLessModuleState:
    if initial is not None and module is not initial.module:
        raise RuntimeError(f"loaded spec-less module identity changed: {name}")

    parent_alias = _RUNTIME_SPECLESS_PARENT_ALIASES.get(name)
    if parent_alias is not None:
        parent = sys.modules.get(parent_alias[0])
        if parent is not None and getattr(parent, parent_alias[1], None) is module:
            return _LoadedSpecLessModuleState(
                name=name,
                parent_name=parent_alias[0],
                module=module,
            )

    if name in _RUNTIME_MLX_SPECLESS_CHILDREN:
        parent_name, _, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is None or getattr(parent, child_name, None) is not module:
            raise RuntimeError(f"loaded native child lacks its parent binding: {name}")
        with _RUNTIME_PROVENANCE_LOCK:
            bindings_overflowed = name in _RUNTIME_MLX_CHILD_BINDINGS_OVERFLOW
            bindings = tuple(_RUNTIME_MLX_CHILD_BINDINGS.get(name, ()))
        if (
            bindings_overflowed
            or len(bindings) != 1
            or bindings[0][0] is not parent
            or bindings[0][1] is not module
        ):
            raise RuntimeError(
                f"loaded native child lacks completed parent provenance: {name}"
            )
        spec = getattr(parent, "__spec__", None)
        path = _loaded_module_origin_path(spec) if spec is not None else None
        if path is None or not any(
            path.name.endswith(suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        ):
            raise RuntimeError(f"loaded native child has an invalid parent: {name}")
        expected_token = dict(state.protected_origins).get(name.partition(".")[0])
        if (
            expected_token is None
            or _loaded_origin_token(path, roots) != expected_token
        ):
            raise RuntimeError(
                f"loaded native child parent violates captured precedence: {name}"
            )
        initial_parent = next(
            (item for item in state.loaded_modules if item.name == parent_name),
            None,
        )
        _validate_loaded_module_file(
            parent_name,
            parent,
            path,
            captured_by_path=captured_by_path,
            initial=initial_parent,
        )
        _require_inert_spec_less_shape(name, module)
        return _LoadedSpecLessModuleState(
            name=name,
            parent_name=parent_name,
            module=module,
        )

    baseline = _RUNTIME_INITIAL_SPECLESS_MODULES.get(name)
    if name in _RUNTIME_SPECLESS_MODULE_NAMES and baseline is module:
        return _LoadedSpecLessModuleState(name=name, parent_name=None, module=module)
    if (
        (
            name == "cython_runtime"
            or re.fullmatch(r"_cython_\d+(?:_\d+)+", name) is not None
        )
        and baseline is module
        and not _runtime_audit.was_preexisting(name, module)
    ):
        _require_inert_spec_less_shape(name, module)
        return _LoadedSpecLessModuleState(name=name, parent_name=None, module=module)
    raise RuntimeError(f"loaded module lacks origin metadata: {name}")


def _require_inert_spec_less_shape(name: str, module: object) -> None:
    if (
        type(module) is not types.ModuleType
        or getattr(module, "__file__", None) is not None
        or getattr(module, "__loader__", None) is not None
        or getattr(module, "__package__", None) is not None
    ):
        raise RuntimeError(f"loaded spec-less module has executable metadata: {name}")


def _require_bounded_runtime_execution(state: _CapturedImportEnvironment) -> None:
    baseline = dict(state.execution_counts)
    current = dict(_runtime_execution_counts())
    changed = {
        path: (baseline.get(path, 0), count)
        for path, count in current.items()
        if count > baseline.get(path, 0)
    }
    if not changed:
        return

    roots = list(state.root_tokens)
    initial_paths = {item.origin.path for item in state.loaded_modules}
    loaded_paths: set[Path] = set()
    for _name, module in _runtime_loaded_modules():
        spec = getattr(module, "__spec__", None)
        if spec is None:
            continue
        try:
            path = _loaded_module_origin_path(spec)
        except (OSError, RuntimeError, ValueError):
            continue
        if path is not None:
            loaded_paths.add(path)

    for path, (before, after) in sorted(
        changed.items(),
        key=lambda item: item[0].as_posix(),
    ):
        matched = _matched_loaded_origin(path, roots)
        if matched is not None and matched[1].startswith("python-runtime:"):
            continue
        if before or path in initial_paths or after - before > 1:
            raise RuntimeError(
                f"governed module code executed again during the operation: {path}"
            )
        if path not in loaded_paths:
            raise RuntimeError(
                "governed module code was executed and then removed during the "
                f"operation: {path}"
            )


def _validate_loaded_module_file(
    module_name: str,
    module: object,
    path: Path,
    *,
    captured_by_path: dict[Path, _CapturedPhysicalFile],
    initial: _LoadedModuleState | None,
) -> _LoadedModuleState:
    expected = captured_by_path.get(path)
    if expected is None:
        raise RuntimeError(
            f"loaded module file is not bound to captured bytes: {module_name}"
        )
    current = _capture_physical_file_identity(path, label="loaded module origin")
    if current != expected:
        raise RuntimeError(
            f"loaded module file identity differs from captured bytes: {module_name}"
        )
    if initial is not None and (
        module is not initial.module or current != initial.origin
    ):
        raise RuntimeError(
            f"loaded module identity changed during the operation: {module_name}"
        )
    _require_loaded_code_provenance(
        module_name,
        module,
        path,
        current,
    )
    return _LoadedModuleState(name=module_name, origin=current, module=module)


def _reject_unbound_loaded_origin(
    module_name: str,
    spec: object,
    *,
    state: _CapturedImportEnvironment,
) -> None:
    """Reject standard imports from governed roots without a captured binding."""

    origin = getattr(spec, "origin", None)
    if origin in {None, "built-in", "frozen"}:
        if origin is not None:
            return
        locations = tuple(getattr(spec, "submodule_search_locations", ()) or ())
        if not locations:
            raise RuntimeError(
                f"loaded module lacks a resolvable origin: {module_name}"
            )
        for location in locations:
            if str(location).endswith(".__path_hook__"):
                if str(location) not in state.namespace_placeholders:
                    raise RuntimeError(
                        "loaded namespace uses an unbound path hook: " f"{module_name}"
                    )
                continue
            _reject_unbound_loaded_path(
                module_name,
                Path(location),
                state=state,
            )
        return
    path = _loaded_module_origin_path(spec)
    if path is None:
        raise RuntimeError(f"loaded module lacks a file origin: {module_name}")
    _reject_unbound_loaded_path(
        module_name,
        path,
        state=state,
    )


def _reject_unbound_loaded_path(
    module_name: str,
    path: Path,
    *,
    state: _CapturedImportEnvironment,
) -> None:
    roots = list(state.root_tokens)
    matched = _matched_loaded_origin(path, roots)
    if matched is None:
        raise RuntimeError(
            f"loaded module origin is outside every captured root: {module_name}"
        )
    _root, token = matched
    if token.startswith("python-runtime:"):
        return
    raise RuntimeError(
        f"loaded module is not bound to captured import evidence: {module_name}"
    )


def _runtime_loaded_modules() -> tuple[tuple[str, object], ...]:
    """Return a stable view of loaded modules for origin validation."""

    return tuple(sys.modules.items())


def _loaded_module_origin_path(spec: object) -> Path | None:
    origin = getattr(spec, "origin", None)
    if not isinstance(origin, str) or origin in {"built-in", "frozen"}:
        return None
    path = Path(origin)
    if not path.is_absolute():
        path = Path(os.path.abspath(path))
    if path.parent.name == "__pycache__" and path.suffix == ".pyc":
        try:
            source = Path(importlib.util.source_from_cache(path.as_posix()))
        except ValueError:
            pass
        else:
            if source.is_file():
                path = source
    _require_no_lexical_symlink_components(path, label="loaded module origin")
    if path.is_symlink():
        raise ValueError(f"loaded module origin is a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"loaded module origin is not a regular file: {path}")
    return resolved


def _captured_physical_file_set(
    captures: Iterator[_CapturedFile] | tuple[_CapturedFile, ...],
) -> tuple[_CapturedPhysicalFile, ...]:
    by_path: dict[Path, _CapturedPhysicalFile] = {}
    for captured in captures:
        physical = _physical_file_from_capture(captured)
        previous = by_path.setdefault(physical.path, physical)
        if previous != physical:
            raise RuntimeError(
                "captured runtime file has inconsistent physical identities: "
                f"{physical.path}"
            )
    return tuple(by_path[path] for path in sorted(by_path, key=Path.as_posix))


def _physical_file_from_capture(captured: _CapturedFile) -> _CapturedPhysicalFile:
    return _CapturedPhysicalFile(
        path=captured.physical_path,
        size_bytes=captured.physical_size_bytes,
        device=captured.device,
        inode=captured.inode,
        modified_ns=captured.modified_ns,
        changed_ns=captured.changed_ns,
    )


def _capture_physical_file_identity(
    path: Path,
    *,
    label: str,
) -> _CapturedPhysicalFile:
    _require_no_lexical_symlink_components(path, label=label)
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode):
        raise ValueError(f"{label} is a symbolic link: {path}")
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    return _CapturedPhysicalFile(
        path=path,
        size_bytes=details.st_size,
        device=details.st_dev,
        inode=details.st_ino,
        modified_ns=details.st_mtime_ns,
        changed_ns=details.st_ctime_ns,
    )


def _runtime_exec_audit_hook(event: str, args: tuple[object, ...]) -> None:
    """Retain the exact top-level Python code objects executed after import."""

    if event == "import" and len(args) >= 2 and isinstance(args[0], str):
        name = args[0]
        raw_path = args[1]
        if isinstance(raw_path, str):
            path = Path(raw_path)
            completed_import = True
        elif raw_path is None and len(args) >= 3:
            path = _find_native_import_candidate(name, args[2])
            if path is None:
                return
            completed_import = False
        else:
            return
        if not path.is_absolute():
            path = Path(os.path.abspath(path))
        path = path.resolve(strict=False)
        if path.suffix == ".py":
            return
        try:
            captured = _capture_physical_file_identity(
                path,
                label="imported native module origin",
            )
        except (OSError, RuntimeError, ValueError):
            return
        if completed_import:
            _record_runtime_execution(path)
        _record_native_import_identity(name, captured)
        return
    if event != "exec" or len(args) != 1 or not isinstance(args[0], types.CodeType):
        return
    code = args[0]
    if code.co_name != "<module>":
        return
    raw_path = code.co_filename
    if not raw_path or raw_path.startswith("<"):
        return
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(os.path.abspath(path))
    path = path.resolve(strict=False)
    with _RUNTIME_PROVENANCE_LOCK:
        _RUNTIME_EXECUTION_COUNTS[path] = _RUNTIME_EXECUTION_COUNTS.get(path, 0) + 1
        observed = _RUNTIME_EXECUTED_MODULE_CODE.setdefault(path, [])
        if len(observed) >= 64:
            _RUNTIME_EXECUTED_MODULE_CODE_OVERFLOW.add(path)
            return
        observed.append(code)


def _record_runtime_execution(path: Path) -> None:
    with _RUNTIME_PROVENANCE_LOCK:
        _RUNTIME_EXECUTION_COUNTS[path] = _RUNTIME_EXECUTION_COUNTS.get(path, 0) + 1


def _find_native_import_candidate(name: str, raw_roots: object) -> Path | None:
    """Resolve an ordinary extension-module candidate before its loader runs."""

    if not isinstance(raw_roots, (list, tuple)):
        return None
    parts = name.split(".")
    if not all(part.isidentifier() for part in parts):
        return None
    candidates: set[Path] = set()
    for raw_root in raw_roots:
        if not isinstance(raw_root, str) or raw_root.endswith(".__path_hook__"):
            continue
        root = Path(raw_root or os.getcwd())
        if not root.is_absolute():
            root = Path(os.path.abspath(root))
        module_stem = root.joinpath(*parts[:-1], parts[-1])
        package_stem = root.joinpath(*parts, "__init__")
        for stem in (module_stem, package_stem):
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                candidate = stem.with_name(stem.name + suffix)
                if candidate.is_file() and not candidate.is_symlink():
                    candidates.add(candidate.resolve(strict=True))
    if len(candidates) != 1:
        return None
    return next(iter(candidates))


def _record_native_import_identity(
    name: str,
    captured: _CapturedPhysicalFile,
) -> None:
    key = (name, captured.path)
    with _RUNTIME_PROVENANCE_LOCK:
        observed_files = _RUNTIME_IMPORTED_NATIVE_FILES.setdefault(key, [])
        if captured in observed_files:
            return
        if len(observed_files) >= 64:
            _RUNTIME_IMPORTED_NATIVE_FILES_OVERFLOW.add(key)
            return
        observed_files.append(captured)


def _record_loaded_native_module(
    name: str,
    module: types.ModuleType,
    loader: _RuntimeExtensionLoader,
) -> None:
    spec = getattr(module, "__spec__", None)
    path = _loaded_module_origin_path(spec) if spec is not None else None
    if path is None or not any(
        path.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise RuntimeError(f"loaded extension has an invalid origin: {name}")
    captured = _capture_physical_file_identity(
        path,
        label="loaded native module origin",
    )
    state = _RuntimeNativeLoadState(
        name=name,
        origin=captured,
        module=module,
        loader=loader,
    )
    key = (name, path)
    with _RUNTIME_PROVENANCE_LOCK:
        observed = _RUNTIME_IMPORTED_NATIVE_MODULES.setdefault(key, [])
        if len(observed) >= 64:
            _RUNTIME_IMPORTED_NATIVE_MODULES_OVERFLOW.add(key)
        else:
            observed.append(state)
    if name == "mlx.core":
        _record_mlx_child_bindings(module)


def _record_mlx_child_bindings(parent: types.ModuleType) -> None:
    for child_module_name in _RUNTIME_MLX_SPECLESS_CHILDREN:
        child_name = child_module_name.rpartition(".")[2]
        child = getattr(parent, child_name, None)
        if child is None or sys.modules.get(child_module_name) is not child:
            raise RuntimeError(
                f"loaded MLX native child lacks its parent binding: {child_module_name}"
            )
        _require_inert_spec_less_shape(child_module_name, child)
        with _RUNTIME_PROVENANCE_LOCK:
            observed = _RUNTIME_MLX_CHILD_BINDINGS.setdefault(
                child_module_name,
                [],
            )
            if len(observed) >= 64:
                _RUNTIME_MLX_CHILD_BINDINGS_OVERFLOW.add(child_module_name)
            else:
                observed.append((parent, child))


def _install_runtime_extension_finder() -> None:
    global _RUNTIME_EXTENSION_FINDER
    if _RUNTIME_EXTENSION_FINDER is not None:
        tracker_indices = [
            index
            for index, item in enumerate(sys.meta_path)
            if item is _RUNTIME_EXTENSION_FINDER
        ]
        path_finder_indices = [
            index
            for index, item in enumerate(sys.meta_path)
            if item is importlib.machinery.PathFinder
        ]
        if (
            len(tracker_indices) != 1
            or len(path_finder_indices) != 1
            or tracker_indices[0] + 1 != path_finder_indices[0]
        ):
            raise RuntimeError("runtime extension import tracker order changed")
        return
    path_finder_indices = [
        index
        for index, item in enumerate(sys.meta_path)
        if item is importlib.machinery.PathFinder
    ]
    if len(path_finder_indices) != 1:
        raise RuntimeError("Python path finder order is unsupported")
    finder = _RuntimeExtensionFinder()
    sys.meta_path.insert(path_finder_indices[0], finder)
    _RUNTIME_EXTENSION_FINDER = finder


def _install_runtime_provenance_audit() -> None:
    """Establish the one-time loaded-module boundary for this interpreter."""

    global _RUNTIME_PROVENANCE_AUDIT_INSTALLED
    with _RUNTIME_PROVENANCE_LOCK:
        if _RUNTIME_PROVENANCE_AUDIT_INSTALLED:
            return
        _install_runtime_extension_finder()
        initial: dict[tuple[str, Path], _CapturedPhysicalFile] = {}
        initial_modules: dict[tuple[str, Path], object] = {}
        for name, module in sys.modules.items():
            spec = getattr(module, "__spec__", None)
            if spec is None:
                if _supported_spec_less_module_name(name):
                    _RUNTIME_INITIAL_SPECLESS_MODULES[name] = module
                continue
            try:
                path = _loaded_module_origin_path(spec)
                if path is None:
                    continue
                initial[(name, path)] = _capture_physical_file_identity(
                    path,
                    label="initial loaded module origin",
                )
                initial_modules[(name, path)] = module
            except (OSError, RuntimeError, ValueError):
                # An unsafe or already-missing origin remains unproven and will
                # fail closed if it participates in governed execution later.
                continue
        _RUNTIME_INITIAL_MODULE_FILES.update(initial)
        _RUNTIME_INITIAL_MODULE_OBJECTS.update(initial_modules)
        mlx_core = sys.modules.get("mlx.core")
        if isinstance(mlx_core, types.ModuleType):
            if not _runtime_audit.was_preexisting("mlx.core", mlx_core):
                _record_mlx_child_bindings(mlx_core)
        _runtime_audit.activate(_runtime_exec_audit_hook)
        _RUNTIME_PROVENANCE_AUDIT_INSTALLED = True


def _require_loaded_code_provenance(
    module_name: str,
    module: object,
    path: Path,
    current: _CapturedPhysicalFile,
) -> None:
    with _RUNTIME_PROVENANCE_LOCK:
        overflow = path in _RUNTIME_EXECUTED_MODULE_CODE_OVERFLOW
        executed = tuple(_RUNTIME_EXECUTED_MODULE_CODE.get(path, ()))
        native_key = (module_name, path)
        native_overflow = native_key in _RUNTIME_IMPORTED_NATIVE_FILES_OVERFLOW
        native_files = tuple(_RUNTIME_IMPORTED_NATIVE_FILES.get(native_key, ()))
        native_module_overflow = native_key in _RUNTIME_IMPORTED_NATIVE_MODULES_OVERFLOW
        native_modules = tuple(_RUNTIME_IMPORTED_NATIVE_MODULES.get(native_key, ()))
        initial = _RUNTIME_INITIAL_MODULE_FILES.get((module_name, path))
        initial_module = _RUNTIME_INITIAL_MODULE_OBJECTS.get((module_name, path))
        module_spec = getattr(module, "__spec__", None)
        cache_key = (
            module_name,
            path,
            current,
            id(module),
            id(module_spec),
            id(getattr(module_spec, "loader", None)),
            id(getattr(module, "__loader__", None)),
            overflow,
            native_overflow,
            native_module_overflow,
            tuple(id(code) for code in executed),
            native_files,
            tuple(
                (item.origin, id(item.module), id(item.loader))
                for item in native_modules
            ),
            initial,
            id(initial_module),
        )
        if cache_key in _RUNTIME_PROVENANCE_CACHE:
            _RUNTIME_PROVENANCE_CACHE.move_to_end(cache_key)
            return
    if overflow or native_overflow or native_module_overflow:
        raise RuntimeError(
            f"loaded module execution provenance overflowed: {module_name}"
        )
    if executed:
        if path.suffix != ".py":
            raise RuntimeError(
                f"loaded non-source module lacks import provenance: {module_name}"
            )
        current_code = _source_module_semantic_code(path, current)
        executed_code = {_canonical_code_object(code) for code in executed}
        if executed_code != {current_code}:
            raise RuntimeError(
                "loaded Python code does not match captured source bytes: "
                f"{module_name}"
            )
        _cache_runtime_provenance(cache_key)
        return
    if native_files:
        if set(native_files) != {current}:
            raise RuntimeError(
                "loaded native code does not match captured module bytes: "
                f"{module_name}"
            )
        spec = getattr(module, "__spec__", None)
        active_loader = getattr(spec, "loader", None)
        module_loader = getattr(module, "__loader__", None)
        tracked_load = (
            len(native_modules) == 1
            and native_modules[0].origin == current
            and native_modules[0].module is module
            and native_modules[0].loader is active_loader
            and module_loader is active_loader
        )
        captured_during_bootstrap = (
            not native_modules
            and initial == current
            and initial_module is module
            and not _runtime_audit.was_preexisting(module_name, module)
        )
        if not tracked_load and not captured_during_bootstrap:
            raise RuntimeError(
                "loaded native module identity lacks completed import provenance: "
                f"{module_name}"
            )
        _cache_runtime_provenance(cache_key)
        return
    if initial == current and _trusted_pre_audit_module(module_name, module):
        _cache_runtime_provenance(cache_key)
        return
    raise RuntimeError(
        "loaded module predates execution evidence without matching import "
        f"provenance: {module_name}"
    )


def _trusted_pre_audit_module(module_name: str, module: object) -> bool:
    if not _runtime_audit.was_preexisting(module_name, module):
        return False
    if module_name in {
        "_virtualenv",
        "aai_local_finetuning",
        "aai_local_finetuning._runtime_audit",
    }:
        return True
    if module_name == "_distutils_hack" or module_name.startswith("_distutils_hack."):
        return True
    return (
        re.fullmatch(
            r"__editable___[A-Za-z0-9_]+_finder",
            module_name,
        )
        is not None
    )


def _cache_runtime_provenance(cache_key: tuple[object, ...]) -> None:
    with _RUNTIME_PROVENANCE_LOCK:
        _RUNTIME_PROVENANCE_CACHE[cache_key] = None
        _RUNTIME_PROVENANCE_CACHE.move_to_end(cache_key)
        while len(_RUNTIME_PROVENANCE_CACHE) > _RUNTIME_PROVENANCE_CACHE_MAX_ENTRIES:
            _RUNTIME_PROVENANCE_CACHE.popitem(last=False)


def _source_module_semantic_code(
    path: Path,
    expected: _CapturedPhysicalFile,
) -> types.CodeType:
    display = (
        "loaded-module-source/"
        + hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()
    )
    captured, raw = _capture_file_bytes(path, display)
    if _physical_file_from_capture(captured) != expected:
        raise RuntimeError(f"loaded module source changed while read: {path}")
    try:
        code = compile(
            raw,
            path.as_posix(),
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"loaded module source cannot be compiled: {path}") from error
    return _canonical_code_object(code)


def _canonical_code_object(code: types.CodeType) -> types.CodeType:
    def canonical(value: object) -> object:
        if isinstance(value, types.CodeType):
            return value.replace(
                co_filename="<runtime-module>",
                co_consts=tuple(canonical(item) for item in value.co_consts),
            )
        return value

    normalized = canonical(code)
    assert isinstance(normalized, types.CodeType)
    return normalized


def _loaded_origin_token(
    path: Path,
    roots: list[tuple[Path, str]],
) -> str:
    matched = _matched_loaded_origin(path, roots)
    if matched is None:
        raise RuntimeError(f"loaded module origin is outside captured roots: {path}")
    _root, token = matched
    return token


def _matched_loaded_origin(
    path: Path,
    roots: list[tuple[Path, str]],
) -> tuple[Path, str] | None:
    _require_no_lexical_symlink_components(path, label="loaded module origin")
    resolved = path.resolve(strict=True)
    matching = [
        (root, token)
        for root, token in roots
        if resolved == root or resolved.is_relative_to(root)
    ]
    if not matching:
        return None
    return max(matching, key=lambda item: len(item[0].parts))


def _capture_runtime_package_payloads(
    distribution: importlib.metadata.Distribution,
    *,
    install_root: Path,
    inventory: tuple[importlib.metadata.PackagePath, ...],
    metadata_path: Path | None = None,
    context: _RuntimeCaptureContext | None = None,
) -> tuple[
    tuple[_CapturedFile, ...],
    tuple[_CapturedFile, ...],
    tuple[_CapturedDirectory, ...],
    tuple[_RuntimeImportBinding, ...],
]:
    """Hash runtime metadata and complete import roots using portable names.

    Wheel ``RECORD`` is an inventory seed, not an authority over the live
    import tree: files may be added after installation, and editable installs
    can expose a source root through a ``.pth`` file.  Enumerating every seeded
    top-level runtime directory closes both gaps.  Generated launchers outside
    the installation root (for example ``../../../bin/<console-script>``) are
    deliberately excluded; their portable behavior is represented by
    ``entry_points.txt`` instead of a machine-specific shebang.  Known external
    runtime archives are content-addressed under a stable logical path, and any
    other escape fails closed.
    """

    capture_context = context or _RuntimeCaptureContext()
    resolved_metadata = metadata_path or _require_runtime_location(
        _distribution_metadata_path(distribution, inventory),
        install_root=install_root,
        label="runtime package metadata",
    )
    explicit_paths: dict[str, Path] = {}
    captured_files: dict[str, _CapturedFile] = {}
    transient_files: dict[str, _CapturedFile] = {}
    runtime_trees: dict[str, Path] = {}
    configured_directories: set[Path] = set()
    import_bindings: dict[tuple[str, Path], _RuntimeImportBinding] = {}
    seen_record_paths: set[tuple[bool, str]] = set()
    inside_record_paths: set[Path] = set()
    prepared_inventory: list[tuple[str, bool, Path | None]] = []

    ordered_inventory = sorted(inventory, key=lambda item: str(item).replace("\\", "/"))
    for item in ordered_inventory:
        record_path, outside_install_root = _distribution_record_path(item)
        record_key = (outside_install_root, record_path)
        if record_key in seen_record_paths:
            name = distribution.metadata.get("Name") or "<unknown>"
            raise RuntimeError(
                f"installed distribution contains a duplicate payload path: {name}"
            )
        seen_record_paths.add(record_key)
        if outside_install_root:
            resolved_external = _require_external_runtime_location(
                Path(distribution.locate_file(item)),
                install_root=install_root,
                label="external runtime package path",
            )
            portable_external = _external_runtime_environment_path(
                resolved_external,
                install_root=install_root,
            )
            if _is_external_runtime_bookkeeping(portable_external):
                continue
            if (
                PurePosixPath(portable_external).suffix
                in _RUNTIME_EXTERNAL_PAYLOAD_SUFFIXES
            ):
                explicit_paths[f"runtime-external/{portable_external}"] = (
                    resolved_external
                )
                continue
            raise ValueError(
                "installed distribution path escapes its installation root "
                f"without a portable runtime mapping: {record_path}"
            )
        if _is_distribution_metadata_path(record_path):
            continue
        parts = PurePosixPath(record_path).parts
        if any(part in _RUNTIME_TRANSIENT_DIRECTORIES for part in parts):
            continue

        is_top_level = len(parts) == 1
        located = None
        if is_top_level:
            located = _require_runtime_location(
                Path(distribution.locate_file(item)),
                install_root=install_root,
                label="runtime package payload",
            )
            inside_record_paths.add(located)
        prepared_inventory.append((record_path, is_top_level, located))

    consumed_paths: set[Path] = set()
    for record_path, is_top_level, located in prepared_inventory:
        if not is_top_level or located is None or located.suffix != ".pth":
            continue
        configuration = _capture_runtime_path_configuration(
            located,
            record_path=record_path,
            install_root=install_root,
            record_owned_paths=frozenset(inside_record_paths),
        )
        for logical_path, captured in configuration.captured_files:
            _add_runtime_capture(captured_files, logical_path, captured)
        for logical_path, physical_path in configuration.explicit_files:
            _add_runtime_path(explicit_paths, logical_path, physical_path)
        for tree_name, tree_path in configuration.exposed_trees:
            _add_runtime_path(runtime_trees, tree_name, tree_path)
        configured_directories.update(configuration.observed_directories)
        consumed_paths.update(configuration.consumed_paths)
        for binding in configuration.import_bindings:
            import_bindings[(binding.name, binding.physical_path)] = binding

    top_level_paths: dict[str, Path] = {}
    for record_path, is_top_level, located in prepared_inventory:
        if located is not None and located in consumed_paths:
            continue
        parts = PurePosixPath(record_path).parts

        top_level = parts[0]
        top_level_path = top_level_paths.get(top_level)
        if top_level_path is None:
            top_level_path = _require_runtime_location(
                Path(distribution.locate_file(top_level)),
                install_root=install_root,
                label="runtime package root",
            )
            top_level_paths[top_level] = top_level_path
        if top_level_path.is_dir():
            _add_runtime_path(runtime_trees, top_level, top_level_path)
        else:
            if not is_top_level or located is None:
                raise RuntimeError(
                    "installed distribution payload has a non-directory "
                    f"top-level parent: {record_path}"
                )
            _add_runtime_path(explicit_paths, record_path, located)
        binding = _runtime_import_binding(
            top_level_path,
            exposure="direct",
            search_root=install_root,
        )
        if binding is not None:
            import_bindings[(binding.name, binding.physical_path)] = binding

    metadata_root = resolved_metadata.parent
    record_capture = _capture_runtime_record(
        metadata_root,
        distribution=distribution,
        install_root=install_root,
        context=capture_context,
    )
    if record_capture is not None:
        _add_runtime_capture(
            captured_files,
            f"{metadata_root.name}/RECORD",
            record_capture,
        )
    _add_runtime_path(
        runtime_trees,
        metadata_root.name,
        metadata_root,
    )

    tree_identities: dict[Path, _CapturedDirectory] = {}
    for directory in sorted(configured_directories):
        captured_directory = _capture_directory_identity(
            directory,
            label="runtime package mapped-source directory",
        )
        capture_context.observe_directory(captured_directory)
        tree_identities[directory] = captured_directory
    for tree_name, tree_path in sorted(runtime_trees.items()):
        tree = capture_context.capture_tree(tree_path)
        for directory in tree.directories:
            tree_identities[directory.path] = directory
        for relative_path, physical_path in tree.files:
            logical_path = PurePosixPath(tree_name, relative_path).as_posix()
            if _is_distribution_metadata_path(logical_path) and not (
                _is_runtime_metadata_payload(logical_path)
            ):
                continue
            _add_runtime_path(explicit_paths, logical_path, physical_path)
        for relative_path, physical_path in tree.bytecode_files:
            logical_path = PurePosixPath(tree_name, relative_path).as_posix()
            transient_files[logical_path] = capture_context.capture_bytecode(
                physical_path,
                logical_path,
            )

    for logical_path, path in sorted(explicit_paths.items()):
        if logical_path in captured_files:
            continue
        if path.suffix == ".pyc":
            transient_files[logical_path] = capture_context.capture_bytecode(
                path,
                logical_path,
            )
            continue
        captured_files[logical_path] = capture_context.capture_file(
            path,
            _runtime_payload_display_path(logical_path),
        )
    if not captured_files:
        name = distribution.metadata.get("Name") or "<unknown>"
        raise RuntimeError(f"installed distribution runtime payload is empty: {name}")
    persistent_result = tuple(captured_files[path] for path in sorted(captured_files))
    transient_result = tuple(transient_files[path] for path in sorted(transient_files))
    if context is None:
        persistent_result = _bind_instrumented_bytecode(
            persistent_result,
            transient_result,
            context=capture_context,
        )
    result = (
        persistent_result,
        transient_result,
        tuple(tree_identities[path] for path in sorted(tree_identities)),
        tuple(
            import_bindings[key]
            for key in sorted(
                import_bindings,
                key=lambda item: (item[0], item[1].as_posix()),
            )
        ),
    )
    if context is None:
        capture_context.verify_directories()
    return result


def _require_distribution_file_inventory(
    distribution: importlib.metadata.Distribution,
) -> tuple[importlib.metadata.PackagePath, ...]:
    files = distribution.files
    name = distribution.metadata.get("Name") or "<unknown>"
    if files is None:
        raise RuntimeError(
            f"installed distribution file inventory is unavailable: {name}"
        )
    inventory = tuple(files)
    if not inventory:
        raise RuntimeError(f"installed distribution file inventory is empty: {name}")
    return inventory


def _distribution_owned_top_levels(
    inventory: tuple[importlib.metadata.PackagePath, ...],
) -> set[str]:
    owned: set[str] = set()
    for item in inventory:
        record_path, outside_install_root = _distribution_record_path(item)
        if outside_install_root or _is_distribution_metadata_path(record_path):
            continue
        parts = PurePosixPath(record_path).parts
        if any(part in _RUNTIME_TRANSIENT_DIRECTORIES for part in parts):
            continue
        owned.add(parts[0])
    return owned


def _runtime_installation_unowned_entries(
    owned_top_levels: dict[Path, set[str]],
) -> dict[Path, tuple[Path, ...]]:
    """Return importable installation-root entries with no RECORD owner."""

    result: dict[Path, tuple[Path, ...]] = {}
    for install_root, owned in sorted(
        owned_top_levels.items(),
        key=lambda item: item[0].as_posix(),
    ):
        unowned = tuple(
            sorted(
                (
                    path
                    for path in install_root.iterdir()
                    if _is_importable_top_level(path) and path.name not in owned
                ),
                key=lambda path: path.name,
            )
        )
        if unowned:
            result[install_root] = unowned
    return result


def _capture_virtualenv_bootstrap(
    install_root: Path,
    *,
    entries: tuple[Path, ...],
    context: _RuntimeCaptureContext,
) -> _RuntimeDistribution:
    """Bind virtualenv's generated, intentionally distribution-less hook."""

    by_name = {path.name: path for path in entries}
    expected = {"_virtualenv.pth", "_virtualenv.py"}
    if by_name.keys() != expected:
        raise RuntimeError(
            "runtime installation contains importable top-level entries "
            "without distribution inventory ownership: " + ", ".join(sorted(by_name))
        )
    pth_path = by_name["_virtualenv.pth"]
    configuration = _capture_runtime_path_configuration(
        pth_path,
        record_path=pth_path.name,
        install_root=install_root,
        record_owned_paths=frozenset(entries),
    )
    if (
        configuration.explicit_files
        or configuration.exposed_trees
        or configuration.observed_directories
    ):
        raise RuntimeError("virtualenv bootstrap unexpectedly exposes a runtime tree")
    captures = dict(configuration.captured_files)
    captures["_virtualenv.py"] = context.capture_file(
        by_name["_virtualenv.py"],
        _runtime_payload_display_path("_virtualenv.py"),
    )
    payload_files = tuple(captures[name] for name in sorted(captures))
    payload_evidence = tuple(item.evidence for item in payload_files)
    evidence = RuntimePackageEvidence(
        name="python-environment-bootstrap",
        version="1",
        payload_file_count=len(payload_files),
        payload_size_bytes=sum(item.evidence.size_bytes for item in payload_files),
        payload_files_sha256=_evidence_sequence_sha256(payload_evidence),
    )
    return _RuntimeDistribution(
        evidence=evidence,
        metadata_path=pth_path,
        metadata_file=captures["_virtualenv.pth"],
        install_root=install_root,
        install_root_identity=context.directories[install_root],
        payload_files=payload_files,
        transient_files=(),
        runtime_roots=(),
        import_bindings=(
            _RuntimeImportBinding(
                name="_virtualenv",
                physical_path=by_name["_virtualenv.py"],
                kind="module",
                exposure="direct",
                search_root=install_root,
            ),
        ),
        inventory_backed=False,
    )


def _is_importable_top_level(path: Path) -> bool:
    name = path.name
    if name in _RUNTIME_TRANSIENT_DIRECTORIES or name.endswith(
        (".dist-info", ".egg-info")
    ):
        return False
    if path.is_dir():
        return name.isidentifier()
    if not path.is_file() and not path.is_symlink():
        return False
    if name.endswith((".pth", ".egg-link")):
        return True
    if name.endswith(".py"):
        return name[:-3].isidentifier()
    if name.endswith(".pyc"):
        return name[:-4].isidentifier()
    for suffix in sorted(
        importlib.machinery.EXTENSION_SUFFIXES,
        key=len,
        reverse=True,
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)].isidentifier()
    return False


def _runtime_import_binding(
    path: Path,
    *,
    exposure: Literal["direct", "path", "finder"] = "direct",
    search_root: Path | None = None,
) -> _RuntimeImportBinding | None:
    """Describe the top-level import a regular path can provide."""

    name = path.name
    if path.is_dir() and not path.is_symlink() and name.isidentifier():
        kind: Literal["package", "namespace"] = (
            "package" if (path / "__init__.py").is_file() else "namespace"
        )
        return _RuntimeImportBinding(
            name=name,
            physical_path=path,
            kind=kind,
            exposure=exposure,
            search_root=search_root,
        )
    if not path.is_file() or path.is_symlink():
        return None
    if name.endswith(".py") and name[:-3].isidentifier():
        return _RuntimeImportBinding(
            name=name[:-3],
            physical_path=path,
            kind="module",
            exposure=exposure,
            search_root=search_root,
        )
    if name.endswith(".pyc") and name[:-4].isidentifier():
        return _RuntimeImportBinding(
            name=name[:-4],
            physical_path=path,
            kind="module",
            exposure=exposure,
            search_root=search_root,
        )
    for suffix in sorted(
        importlib.machinery.EXTENSION_SUFFIXES,
        key=len,
        reverse=True,
    ):
        if name.endswith(suffix) and name[: -len(suffix)].isidentifier():
            return _RuntimeImportBinding(
                name=name[: -len(suffix)],
                physical_path=path,
                kind="module",
                exposure=exposure,
                search_root=search_root,
            )
    return None


def _runtime_search_root_bindings(root: Path) -> tuple[_RuntimeImportBinding, ...]:
    bindings = [
        binding
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if (
            binding := _runtime_import_binding(
                path,
                exposure="path",
                search_root=root,
            )
        )
        is not None
    ]
    return tuple(bindings)


def _distribution_record_path(
    item: importlib.metadata.PackagePath,
) -> tuple[str, bool]:
    raw = str(item).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if (
        candidate.is_absolute()
        or re.match(r"^[A-Za-z]:/", raw)
        or not candidate.parts
        or candidate.as_posix() != raw
    ):
        raise ValueError(f"installed distribution path is not canonical: {raw}")
    parts = list(candidate.parts)
    parent_count = 0
    while parts and parts[0] == "..":
        parent_count += 1
        parts.pop(0)
    if ".." in parts or not parts:
        raise ValueError(
            f"installed distribution path contains unsafe traversal: {raw}"
        )
    portable = PurePosixPath(*parts).as_posix()
    return portable, parent_count > 0


def _is_distribution_metadata_path(record_path: str) -> bool:
    return any(
        part.endswith((".dist-info", ".egg-info"))
        for part in PurePosixPath(record_path).parts
    )


def _is_runtime_metadata_payload(record_path: str) -> bool:
    parts = PurePosixPath(record_path).parts
    metadata_index = next(
        (
            index
            for index, part in enumerate(parts)
            if part.endswith((".dist-info", ".egg-info"))
        ),
        None,
    )
    if metadata_index is None:
        return True
    relative = parts[metadata_index + 1 :]
    if not relative:
        return False
    if any(part in _RUNTIME_NON_PAYLOAD_METADATA_DIRECTORIES for part in relative[:-1]):
        return False
    return relative[-1] not in _RUNTIME_NON_PAYLOAD_METADATA_FILES


def _capture_runtime_record(
    metadata_root: Path,
    *,
    distribution: importlib.metadata.Distribution,
    install_root: Path,
    context: _RuntimeCaptureContext,
) -> _CapturedFile | None:
    """Bind RECORD semantics without persisting machine-specific launcher hashes."""

    record_path = metadata_root / "RECORD"
    if not record_path.exists():
        return None
    display_path = _runtime_payload_display_path(f"{metadata_root.name}/RECORD")
    raw_capture, raw = _capture_file_bytes(record_path, display_path)
    previous = context.files.setdefault(record_path, raw_capture)
    if previous != raw_capture:
        raise RuntimeError(
            f"runtime package RECORD changed while it was captured: {record_path}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"runtime package RECORD is not UTF-8: {record_path}"
        ) from error

    canonical_rows: list[tuple[str, str, str]] = []
    seen_paths: set[tuple[bool, str]] = set()
    try:
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        for row_number, row in enumerate(rows, 1):
            if len(row) != 3 or not row[0]:
                raise ValueError(
                    f"runtime package RECORD row {row_number} is malformed: "
                    f"{record_path}"
                )
            portable_path, outside_install_root = _distribution_record_path(
                importlib.metadata.PackagePath(row[0])
            )
            if outside_install_root:
                resolved_external = _require_external_runtime_location(
                    Path(
                        distribution.locate_file(importlib.metadata.PackagePath(row[0]))
                    ),
                    install_root=install_root,
                    label="external runtime package RECORD path",
                )
                portable_path = _external_runtime_environment_path(
                    resolved_external,
                    install_root=install_root,
                )
                if not _is_external_runtime_bookkeeping(portable_path) and (
                    PurePosixPath(portable_path).suffix
                    not in _RUNTIME_EXTERNAL_PAYLOAD_SUFFIXES
                ):
                    raise ValueError(
                        "runtime package RECORD path escapes its installation "
                        "root without a portable runtime mapping: "
                        f"{row[0]}"
                    )
            key = (outside_install_root, portable_path)
            if key in seen_paths:
                raise ValueError(
                    "runtime package RECORD contains duplicate canonical paths: "
                    f"{record_path}"
                )
            seen_paths.add(key)
            hash_value, size_value = row[1:]
            if hash_value and not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*=[A-Za-z0-9_-]+",
                hash_value,
            ):
                raise ValueError(
                    f"runtime package RECORD row {row_number} has an invalid hash: "
                    f"{record_path}"
                )
            if size_value and (not size_value.isascii() or not size_value.isdecimal()):
                raise ValueError(
                    f"runtime package RECORD row {row_number} has an invalid size: "
                    f"{record_path}"
                )
            prefix = "outside" if outside_install_root else "inside"
            if outside_install_root and _is_external_runtime_bookkeeping(portable_path):
                hash_value = "<portable-external-bookkeeping>" if hash_value else ""
                size_value = "<portable-external-bookkeeping>" if size_value else ""
            canonical_rows.append((f"{prefix}:{portable_path}", hash_value, size_value))
    except csv.Error as error:
        raise ValueError(
            f"runtime package RECORD is malformed: {record_path}"
        ) from error
    canonical = json.dumps(
        canonical_rows,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _captured_file_with_canonical_bytes(raw_capture, display_path, canonical)


def _is_external_runtime_bookkeeping(record_path: str) -> bool:
    parts = PurePosixPath(record_path).parts
    return any(
        parts[: len(prefix)] == prefix
        for prefix in _RUNTIME_EXTERNAL_BOOKKEEPING_PREFIXES
    )


def _require_runtime_location(
    path: Path,
    *,
    install_root: Path,
    label: str,
) -> Path:
    _require_no_lexical_symlink_components(path, label=label)
    if path.is_symlink():
        raise ValueError(f"{label} is a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(install_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its installation root: {path}") from error
    return resolved


def _require_external_runtime_location(
    path: Path,
    *,
    install_root: Path,
    label: str,
) -> Path:
    _require_no_lexical_symlink_components(path, label=label)
    if path.is_symlink():
        raise ValueError(f"{label} is a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    environment_roots = _runtime_environment_roots(install_root)
    if not any(resolved.is_relative_to(root) for root in environment_roots):
        raise ValueError(f"{label} escapes the Python environment: {path}")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    return resolved


def _runtime_environment_roots(install_root: Path) -> frozenset[Path]:
    prefix = Path(sys.prefix)
    _require_no_lexical_symlink_components(
        prefix,
        label="Python environment prefix",
    )
    environment_roots = {prefix.resolve(strict=True)}
    parts = install_root.parts
    if (
        len(parts) >= 4
        and parts[-1] in {"site-packages", "dist-packages"}
        and re.fullmatch(r"python\d+(?:\.\d+)?", parts[-2])
        and parts[-3] in {"lib", "lib64"}
    ):
        environment_roots.add(install_root.parents[2])
    elif len(parts) >= 3 and parts[-2:] == ("Lib", "site-packages"):
        environment_roots.add(install_root.parents[1])
    return frozenset(environment_roots)


def _external_runtime_environment_path(
    path: Path,
    *,
    install_root: Path,
) -> str:
    relatives = [
        path.relative_to(root)
        for root in _runtime_environment_roots(install_root)
        if path.is_relative_to(root)
    ]
    if not relatives:
        raise ValueError(
            f"external runtime path escapes the Python environment: {path}"
        )
    relative = min(relatives, key=lambda item: (len(item.parts), item.as_posix()))
    if not relative.parts:
        raise ValueError(f"external runtime path is the Python environment: {path}")
    return relative.as_posix()


def _require_no_lexical_symlink_components(path: Path, *, label: str) -> None:
    """Reject mutable lexical links before canonical path resolution."""

    lexical = path if path.is_absolute() else Path.cwd() / path
    anchor = Path(lexical.anchor)
    component = anchor
    for part in lexical.parts[1:]:
        if part == "..":
            component = component.parent
            continue
        component /= part
        try:
            details = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode):
            raise ValueError(
                f"{label} has an unsupported lexical symbolic-link component: "
                f"{component}; use the physical path instead"
            )


def _runtime_payload_display_path(logical_path: str) -> str:
    _require_safe_relative_path(logical_path, "runtime package payload")
    return (
        "runtime-package-payload/"
        + hashlib.sha256(logical_path.encode("utf-8")).hexdigest()
    )


def _add_runtime_path(paths: dict[str, Path], logical_path: str, path: Path) -> None:
    existing = paths.setdefault(logical_path, path)
    if existing != path:
        raise RuntimeError(
            "runtime package inventory maps one portable path to multiple files"
        )


def _add_runtime_capture(
    captures: dict[str, _CapturedFile],
    logical_path: str,
    captured: _CapturedFile,
) -> None:
    existing = captures.setdefault(logical_path, captured)
    if existing != captured:
        raise RuntimeError(
            "runtime package inventory maps one portable path to multiple captures"
        )


def _capture_runtime_path_configuration(
    path: Path,
    *,
    record_path: str,
    install_root: Path,
    record_owned_paths: frozenset[Path],
) -> _RuntimePathConfiguration:
    """Canonicalize a ``.pth`` file and expose its live import roots.

    Absolute editable-source paths are intentionally replaced by stable ordinal
    markers.  The referenced source bytes are captured below those markers, so
    two equivalent environments produce the same portable digest while a
    source or mapping change still invalidates an in-flight snapshot.
    """

    display_path = _runtime_payload_display_path(record_path)
    raw_capture, raw = _capture_file_bytes(path, display_path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"runtime path configuration is not UTF-8: {path}") from error

    effective_lines = [
        (line_number, raw_line.rstrip())
        for line_number, raw_line in enumerate(text.splitlines(), 1)
        if raw_line.rstrip() and not raw_line.startswith("#")
    ]
    executable_lines = [
        (line_number, line)
        for line_number, line in effective_lines
        if line.startswith(("import ", "import\t"))
    ]
    if executable_lines:
        if len(effective_lines) != 1 or len(executable_lines) != 1:
            raise ValueError(
                "runtime path configuration mixes executable and path entries: "
                f"{path}"
            )
        _line_number, line = executable_lines[0]
        known_canonical = _known_generated_executable_pth(path, line)
        if known_canonical is not None:
            return _RuntimePathConfiguration(
                captured_files=(
                    (
                        record_path,
                        _captured_file_with_canonical_bytes(
                            raw_capture,
                            display_path,
                            known_canonical,
                        ),
                    ),
                ),
                explicit_files=(),
                exposed_trees=(),
                observed_directories=(),
                consumed_paths=frozenset({path}),
                import_bindings=(),
            )
        return _capture_setuptools_finder_configuration(
            path,
            record_path=record_path,
            install_root=install_root,
            record_owned_paths=record_owned_paths,
            raw_capture=raw_capture,
            line=line,
        )

    canonical_lines: list[str] = []
    exposed_trees: list[tuple[str, Path]] = []
    path_key = hashlib.sha256(record_path.encode("utf-8")).hexdigest()
    for ordinal, (_line_number, line) in enumerate(effective_lines, 1):
        candidate = Path(line)
        located = candidate if candidate.is_absolute() else install_root / candidate
        _require_no_lexical_symlink_components(
            located,
            label="runtime import root",
        )
        if located.is_symlink():
            raise ValueError(f"runtime import root is a symbolic link: {located}")
        resolved = located.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"runtime import root is not a directory: {located}")
        if path.name.startswith("__editable__.") and resolved.name.startswith(
            "__editable__."
        ):
            raise ValueError(
                "setuptools strict editable symlink-tree mode is unsupported; "
                "use the standard finder or path editable mode"
            )
        tree_name = f"runtime-path/{path_key}/{ordinal:06d}"
        canonical_lines.append(f"path:{ordinal:06d}")
        exposed_trees.append((tree_name, resolved))

    canonical = "\n".join(canonical_lines).encode("utf-8")
    captured = _CapturedFile(
        evidence=TrainingFileEvidence(
            path=display_path,
            sha256=hashlib.sha256(canonical).hexdigest(),
            size_bytes=len(canonical),
        ),
        physical_path=raw_capture.physical_path,
        physical_size_bytes=raw_capture.physical_size_bytes,
        device=raw_capture.device,
        inode=raw_capture.inode,
        modified_ns=raw_capture.modified_ns,
        changed_ns=raw_capture.changed_ns,
    )
    return _RuntimePathConfiguration(
        captured_files=((record_path, captured),),
        explicit_files=(),
        exposed_trees=tuple(exposed_trees),
        observed_directories=(),
        consumed_paths=frozenset({path}),
        import_bindings=tuple(
            binding
            for _tree_name, root in exposed_trees
            for binding in _runtime_search_root_bindings(root)
        ),
    )


def _known_generated_executable_pth(path: Path, line: str) -> bytes | None:
    try:
        tree = ast.parse(line, mode="exec")
    except SyntaxError:
        return None
    if path.name == "_virtualenv.pth" and ast.dump(
        tree,
        include_attributes=False,
    ) == ast.dump(
        ast.parse("import _virtualenv", mode="exec"),
        include_attributes=False,
    ):
        return ast.dump(tree, include_attributes=False).encode("utf-8")
    if path.name == "distutils-precedence.pth" and ast.dump(
        tree,
        include_attributes=False,
    ) == ast.dump(
        ast.parse(_SETUPTOOLS_DISTUTILS_PTH_SOURCE, mode="exec"),
        include_attributes=False,
    ):
        return ast.dump(tree, include_attributes=False).encode("utf-8")
    if path.name != "a1_coverage.pth" or len(tree.body) != 2:
        return None
    imported, executed = tree.body
    if not (
        isinstance(imported, ast.Import)
        and len(imported.names) == 1
        and imported.names[0].name == "sys"
        and imported.names[0].asname is None
        and isinstance(executed, ast.Expr)
        and isinstance(executed.value, ast.Call)
        and isinstance(executed.value.func, ast.Name)
        and executed.value.func.id == "exec"
        and len(executed.value.args) == 1
        and not executed.value.keywords
        and isinstance(executed.value.args[0], ast.Constant)
        and isinstance(executed.value.args[0].value, str)
    ):
        return None
    try:
        embedded = ast.parse(executed.value.args[0].value, mode="exec")
    except SyntaxError:
        return None
    if ast.dump(embedded, include_attributes=False) != ast.dump(
        ast.parse(_COVERAGE_PTH_EMBEDDED_SOURCE, mode="exec"),
        include_attributes=False,
    ):
        return None
    return ast.dump(tree, include_attributes=False).encode("utf-8")


def _capture_setuptools_finder_configuration(
    path: Path,
    *,
    record_path: str,
    install_root: Path,
    record_owned_paths: frozenset[Path],
    raw_capture: _CapturedFile,
    line: str,
) -> _RuntimePathConfiguration:
    if not path.name.startswith("__editable__.") or not path.name.endswith(".pth"):
        raise ValueError(
            f"unknown executable runtime path configuration is unsupported: {path}"
        )
    try:
        parsed_line = ast.parse(line, mode="exec")
    except SyntaxError as error:
        raise ValueError(
            f"runtime path configuration contains invalid executable syntax: {path}"
        ) from error
    expected_module = re.sub(r"\W|^(?=\d)", "_", f"{path.stem}.finder")
    if not _is_setuptools_finder_import(parsed_line, expected_module):
        raise ValueError(
            f"unknown executable runtime path configuration is unsupported: {path}"
        )

    finder_path = _require_runtime_location(
        install_root / f"{expected_module}.py",
        install_root=install_root,
        label="setuptools editable finder",
    )
    if finder_path not in record_owned_paths:
        raise ValueError(
            "setuptools editable finder is not owned by the distribution RECORD: "
            f"{finder_path.name}"
        )
    finder_display = _runtime_payload_display_path(f"{record_path}.finder")
    finder_raw_capture, finder_raw = _capture_file_bytes(finder_path, finder_display)
    try:
        finder_source = finder_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"setuptools editable finder is not UTF-8: {finder_path}"
        ) from error
    canonical_finder, mapping, namespaces = _parse_setuptools_finder(
        finder_source,
        path=finder_path,
        expected_module=expected_module,
        expected_finder_name=f"{path.stem}.finder",
    )

    canonical_pth = f"setuptools-finder:{expected_module}".encode()
    pth_capture = _captured_file_with_canonical_bytes(
        raw_capture,
        _runtime_payload_display_path(record_path),
        canonical_pth,
    )
    finder_capture = _captured_file_with_canonical_bytes(
        finder_raw_capture,
        finder_display,
        canonical_finder,
    )
    prefix = (
        "runtime-path/"
        + hashlib.sha256(record_path.encode("utf-8")).hexdigest()
        + "/finder"
    )
    explicit_files: list[tuple[str, Path]] = []
    exposed_trees: list[tuple[str, Path]] = []
    namespace_placeholder = f"{path.stem}.finder.__path_hook__" if namespaces else None
    import_bindings: list[_RuntimeImportBinding] = [
        _RuntimeImportBinding(
            name=expected_module,
            physical_path=finder_path,
            kind="module",
            exposure="direct",
            search_root=install_root,
            namespace_placeholder=namespace_placeholder,
        )
    ]
    for qualified_name, raw_source in sorted(mapping.items()):
        logical = PurePosixPath(
            prefix,
            "mapping",
            *qualified_name.split("."),
        ).as_posix()
        source_type, source = _resolve_setuptools_mapping_source(
            raw_source,
            label=f"setuptools editable mapping {qualified_name}",
            allow_namespace_directory=qualified_name in namespaces,
        )
        if source_type == "tree":
            exposed_trees.append((logical, source))
        else:
            suffix = "".join(source.suffixes) or ".py"
            explicit_files.append((logical + suffix, source))
        if "." not in qualified_name:
            import_bindings.append(
                _RuntimeImportBinding(
                    name=qualified_name,
                    physical_path=source,
                    kind="module" if source_type == "file" else "package",
                    exposure="finder",
                    search_root=None,
                    namespace_placeholder=namespace_placeholder,
                )
            )
    for qualified_name, raw_roots in sorted(namespaces.items()):
        for index, raw_root in enumerate(raw_roots):
            root = _require_editable_source_directory(
                raw_root,
                label=f"setuptools editable namespace {qualified_name}",
            )
            logical = PurePosixPath(
                prefix,
                "namespace",
                *qualified_name.split("."),
                f"{index:06d}",
            ).as_posix()
            exposed_trees.append((logical, root))
            if "." not in qualified_name:
                import_bindings.append(
                    _RuntimeImportBinding(
                        name=qualified_name,
                        physical_path=root,
                        kind="namespace",
                        exposure="finder",
                        search_root=None,
                        namespace_placeholder=namespace_placeholder,
                    )
                )
    return _RuntimePathConfiguration(
        captured_files=(
            (record_path, pth_capture),
            (f"{record_path}.finder", finder_capture),
        ),
        explicit_files=tuple(explicit_files),
        exposed_trees=tuple(exposed_trees),
        observed_directories=tuple(
            sorted({path.parent for _logical, path in explicit_files})
        ),
        consumed_paths=frozenset({path, finder_path}),
        import_bindings=tuple(import_bindings),
    )


def _is_setuptools_finder_import(tree: ast.Module, module: str) -> bool:
    if len(tree.body) != 2:
        return False
    imported, installed = tree.body
    if not isinstance(imported, ast.Import) or len(imported.names) != 1:
        return False
    alias = imported.names[0]
    if alias.name != module or alias.asname is not None:
        return False
    if not isinstance(installed, ast.Expr) or not isinstance(installed.value, ast.Call):
        return False
    call = installed.value
    return (
        not call.args
        and not call.keywords
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "install"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == module
    )


def _parse_setuptools_finder(
    source: str,
    *,
    path: Path,
    expected_module: str,
    expected_finder_name: str,
) -> tuple[bytes, dict[str, str], dict[str, list[str]]]:
    try:
        tree = ast.parse(source, filename=path.as_posix(), mode="exec")
    except SyntaxError as error:
        raise ValueError(f"setuptools editable finder is invalid: {path}") from error
    assignments: dict[str, ast.expr] = {}
    for node in tree.body:
        name: str | None = None
        value: ast.expr | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value = node.value
        if name not in {"MAPPING", "NAMESPACES"}:
            continue
        if value is None or name in assignments:
            raise ValueError(f"setuptools editable finder has ambiguous {name}: {path}")
        assignments[name] = value
    if assignments.keys() != {"MAPPING", "NAMESPACES"}:
        raise ValueError(
            f"setuptools editable finder lacks literal MAPPING/NAMESPACES: {path}"
        )
    mapping = _literal_finder_mapping(assignments["MAPPING"], path=path)
    namespaces = _literal_finder_namespaces(assignments["NAMESPACES"], path=path)
    expected_tree = ast.parse(
        _SETUPTOOLS_FINDER_TEMPLATE.format(
            mapping=mapping,
            namespaces=namespaces,
            name=expected_finder_name,
        ),
        mode="exec",
    )
    if ast.dump(tree, include_attributes=False) != ast.dump(
        expected_tree,
        include_attributes=False,
    ):
        raise ValueError(
            "setuptools editable finder implementation is not the supported "
            f"locked template: {path}"
        )
    _mask_finder_paths(assignments["MAPPING"], kind="mapping")
    _mask_finder_paths(assignments["NAMESPACES"], kind="namespace")
    if any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and (Path(node.value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", node.value))
        for node in ast.walk(tree)
    ):
        raise ValueError(
            "setuptools editable finder contains an unclassified absolute path: "
            f"{path}"
        )
    canonical = (
        f"module:{expected_module}\n"
        + ast.dump(tree, annotate_fields=True, include_attributes=False)
    ).encode("utf-8")
    return canonical, mapping, namespaces


def _literal_finder_mapping(node: ast.expr, *, path: Path) -> dict[str, str]:
    values = _literal_finder_dict(node, label="MAPPING", path=path)
    if not all(
        _is_qualified_identifier(key)
        and isinstance(value, str)
        and value
        and Path(value).is_absolute()
        for key, value in values.items()
    ):
        raise ValueError(f"setuptools editable finder MAPPING is unsafe: {path}")
    return {key: value for key, value in values.items() if isinstance(value, str)}


def _literal_finder_namespaces(
    node: ast.expr,
    *,
    path: Path,
) -> dict[str, list[str]]:
    values = _literal_finder_dict(node, label="NAMESPACES", path=path)
    if not all(
        _is_qualified_identifier(key)
        and isinstance(value, list)
        and all(
            isinstance(item, str) and item and Path(item).is_absolute()
            for item in value
        )
        for key, value in values.items()
    ):
        raise ValueError(f"setuptools editable finder NAMESPACES is unsafe: {path}")
    return {
        key: list(value) for key, value in values.items() if isinstance(value, list)
    }


def _literal_finder_dict(
    node: ast.expr,
    *,
    label: str,
    path: Path,
) -> dict[str, object]:
    if not isinstance(node, ast.Dict) or any(key is None for key in node.keys):
        raise ValueError(f"setuptools editable finder {label} is not literal: {path}")
    keys: list[object] = []
    values: list[object] = []
    try:
        for key, value in zip(node.keys, node.values, strict=True):
            assert key is not None
            keys.append(ast.literal_eval(key))
            values.append(ast.literal_eval(value))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"setuptools editable finder {label} is not literal: {path}"
        ) from error
    if not all(isinstance(key, str) for key in keys) or len(keys) != len(set(keys)):
        raise ValueError(
            f"setuptools editable finder {label} has invalid or duplicate keys: {path}"
        )
    return dict(zip(keys, values, strict=True))


def _mask_finder_paths(node: ast.expr, *, kind: str) -> None:
    assert isinstance(node, ast.Dict)
    for value_index, (key_node, value_node) in enumerate(
        zip(node.keys, node.values, strict=True)
    ):
        assert key_node is not None
        key = ast.literal_eval(key_node)
        if kind == "mapping":
            replacement: ast.expr = ast.Constant(value=f"<{kind}:{key}>")
        else:
            assert isinstance(value_node, (ast.List, ast.Tuple))
            replacement = ast.List(
                elts=[
                    ast.Constant(value=f"<{kind}:{key}:{index}>")
                    for index, _item in enumerate(value_node.elts)
                ],
                ctx=ast.Load(),
            )
        node.values[value_index] = replacement


def _is_qualified_identifier(value: object) -> bool:
    return isinstance(value, str) and all(
        part.isidentifier() for part in value.split(".")
    )


def _resolve_setuptools_mapping_source(
    raw_path: str,
    *,
    label: str,
    allow_namespace_directory: bool,
) -> tuple[Literal["file", "tree"], Path]:
    candidate = Path(raw_path)
    _require_no_lexical_symlink_components(candidate, label=label)
    if candidate.is_symlink():
        raise ValueError(f"{label} is a symbolic link: {candidate}")
    package_initializer = candidate / "__init__.py"
    if package_initializer.is_symlink():
        raise ValueError(f"{label} is a symbolic link: {package_initializer}")
    if candidate.is_dir() and package_initializer.is_file():
        return "tree", candidate.resolve(strict=True)
    possible = list(
        candidate.with_suffix(suffix) for suffix in importlib.machinery.all_suffixes()
    )
    for path in possible:
        if path.is_symlink():
            raise ValueError(f"{label} is a symbolic link: {path}")
        if path.is_file():
            return "file", path.resolve(strict=True)
        if path.exists() and not path.is_dir():
            raise ValueError(f"{label} is not a regular runtime source: {path}")
    if candidate.is_dir() and allow_namespace_directory:
        return "tree", candidate.resolve(strict=True)
    raise FileNotFoundError(f"{label} cannot be resolved: {candidate}")


def _require_editable_source_directory(raw_path: str, *, label: str) -> Path:
    candidate = Path(raw_path)
    _require_no_lexical_symlink_components(candidate, label=label)
    if candidate.is_symlink():
        raise ValueError(f"{label} is a symbolic link: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {candidate}")
    return resolved


def _capture_runtime_tree(root: Path) -> _CapturedRuntimeTree:
    files: list[tuple[str, Path]] = []
    bytecode_files: list[tuple[str, Path]] = []
    directories: list[_CapturedDirectory] = []

    def visit(
        directory: Path,
        relative: PurePosixPath,
        *,
        in_bytecode_cache: bool,
    ) -> None:
        before = _capture_directory_identity(
            directory,
            label="runtime package import directory",
        )
        with os.scandir(directory) as entries:
            ordered_entries = sorted(entries, key=lambda item: item.name)
        for entry in ordered_entries:
            path = Path(entry.path)
            child_relative = relative / entry.name
            logical_path = PurePosixPath(root.name, child_relative).as_posix()
            if entry.is_symlink():
                raise ValueError(
                    f"runtime package import root contains a symlink: {path}"
                )
            if entry.is_dir(
                follow_symlinks=False
            ) and _is_non_runtime_metadata_directory(logical_path):
                continue
            if entry.is_dir(follow_symlinks=False):
                if in_bytecode_cache:
                    raise ValueError(
                        "runtime bytecode cache contains a nested directory: " f"{path}"
                    )
                visit(
                    path,
                    child_relative,
                    in_bytecode_cache=entry.name in _RUNTIME_TRANSIENT_DIRECTORIES,
                )
            elif entry.is_file(follow_symlinks=False):
                if path.suffix == ".pyc":
                    bytecode_files.append((child_relative.as_posix(), path))
                elif in_bytecode_cache:
                    raise ValueError(
                        "runtime bytecode cache contains a non-bytecode file: "
                        f"{path}"
                    )
                else:
                    files.append((child_relative.as_posix(), path))
            else:
                raise ValueError(
                    "runtime package import root contains a non-regular path: "
                    f"{path}"
                )
        after = _capture_directory_identity(
            directory,
            label="runtime package import directory",
        )
        if after != before:
            raise RuntimeError(
                f"runtime package directory changed while it was captured: {directory}"
            )
        directories.append(after)

    visit(root, PurePosixPath(), in_bytecode_cache=False)
    return _CapturedRuntimeTree(
        files=tuple(sorted(files)),
        bytecode_files=tuple(sorted(bytecode_files)),
        directories=tuple(sorted(directories, key=lambda item: item.path.as_posix())),
    )


def _is_non_runtime_metadata_directory(logical_path: str) -> bool:
    parts = PurePosixPath(logical_path).parts
    metadata_index = next(
        (
            index
            for index, part in enumerate(parts)
            if part.endswith((".dist-info", ".egg-info"))
        ),
        None,
    )
    return metadata_index is not None and any(
        part in _RUNTIME_NON_PAYLOAD_METADATA_DIRECTORIES
        for part in parts[metadata_index + 1 :]
    )


def _distribution_metadata_path(
    distribution: importlib.metadata.Distribution,
    inventory: tuple[importlib.metadata.PackagePath, ...],
) -> Path:
    metadata_root = getattr(distribution, "_path", None)
    if metadata_root is not None:
        root = Path(metadata_root)
        candidates = [
            candidate
            for candidate in (root / "METADATA", root / "PKG-INFO")
            if candidate.is_file() and not candidate.is_symlink()
        ]
        if len(candidates) == 1:
            return candidates[0]

    candidates = []
    for item in inventory:
        record_path, outside_install_root = _distribution_record_path(item)
        if outside_install_root:
            continue
        if record_path.endswith((".dist-info/METADATA", ".egg-info/PKG-INFO")):
            candidates.append(item)
    if candidates:
        shallowest = min(len(item.parts) for item in candidates)
        candidates = [item for item in candidates if len(item.parts) == shallowest]
    if len(candidates) != 1:
        name = distribution.metadata.get("Name") or "<unknown>"
        raise RuntimeError(
            f"installed distribution metadata file is not uniquely located: {name}"
        )
    return Path(distribution.locate_file(candidates[0]))


def _capture_directory_identity(
    path: Path,
    *,
    label: str = "runtime package root",
) -> _CapturedDirectory:
    _require_no_lexical_symlink_components(path, label=label)
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode):
        raise ValueError(f"{label} is a symbolic link: {path}")
    if not stat.S_ISDIR(details.st_mode):
        raise ValueError(f"{label} is not a directory: {path}")
    return _CapturedDirectory(
        path=path,
        device=details.st_dev,
        inode=details.st_ino,
        modified_ns=details.st_mtime_ns,
        changed_ns=details.st_ctime_ns,
    )


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


def _captured_file_with_display_path(
    captured: _CapturedFile,
    display_path: str,
) -> _CapturedFile:
    _require_safe_relative_path(display_path, "captured file")
    if captured.evidence.path == display_path:
        return captured
    return _CapturedFile(
        evidence=TrainingFileEvidence(
            path=display_path,
            sha256=captured.evidence.sha256,
            size_bytes=captured.evidence.size_bytes,
        ),
        physical_path=captured.physical_path,
        physical_size_bytes=captured.physical_size_bytes,
        device=captured.device,
        inode=captured.inode,
        modified_ns=captured.modified_ns,
        changed_ns=captured.changed_ns,
    )


def _captured_file_with_canonical_bytes(
    captured: _CapturedFile,
    display_path: str,
    canonical: bytes,
) -> _CapturedFile:
    _require_safe_relative_path(display_path, "captured file")
    return _CapturedFile(
        evidence=TrainingFileEvidence(
            path=display_path,
            sha256=hashlib.sha256(canonical).hexdigest(),
            size_bytes=len(canonical),
        ),
        physical_path=captured.physical_path,
        physical_size_bytes=captured.physical_size_bytes,
        device=captured.device,
        inode=captured.inode,
        modified_ns=captured.modified_ns,
        changed_ns=captured.changed_ns,
    )


def _capture_runtime_payload_file(path: Path, display_path: str) -> _CapturedFile:
    captured, _ = _capture_open_file(
        path,
        display_path,
        include_bytes=False,
        use_runtime_digest_cache=True,
    )
    return captured


def _capture_file_bytes(path: Path, display_path: str) -> tuple[_CapturedFile, bytes]:
    captured, raw = _capture_open_file(path, display_path, include_bytes=True)
    assert raw is not None
    return captured, raw


def _validate_runtime_bytecode_details(
    paths: tuple[Path, ...],
) -> dict[Path, tuple[str, str | None]]:
    if not paths:
        return {}
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", _BYTECODE_VALIDATOR_SCRIPT],
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(
            json.dumps([path.as_posix() for path in paths]),
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise RuntimeError("isolated runtime bytecode validation timed out") from error
    if process.returncode != 0:
        raise RuntimeError(
            "isolated runtime bytecode validation failed: "
            + (stderr.strip() or f"exit {process.returncode}")
        )
    try:
        results = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "isolated runtime bytecode validation returned invalid output"
        ) from error
    expected_paths = [path.as_posix() for path in paths]
    if (
        not isinstance(results, list)
        or not all(isinstance(item, dict) for item in results)
        or [item.get("path") for item in results] != expected_paths
    ):
        raise RuntimeError("isolated runtime bytecode validation was incomplete")
    failures = [
        f"{item['path']}: {item['error']}"
        for item in results
        if isinstance(item, dict) and "error" in item
    ]
    if failures:
        raise ValueError("runtime bytecode validation failed: " + "; ".join(failures))
    details: dict[Path, tuple[str, str | None]] = {}
    for path, item in zip(paths, results, strict=True):
        status = item.get("status")
        digest = item.get("semantic_sha256")
        if status in {"source-equivalent", "inactive-stale"} and digest is None:
            details[path] = (status, None)
        elif (
            status == "instrumented"
            and isinstance(digest, str)
            and re.fullmatch(_SHA256_PATTERN, digest)
        ):
            details[path] = (status, digest)
        else:
            raise RuntimeError("isolated runtime bytecode validation was incomplete")
    return details


def _validate_runtime_bytecode(paths: tuple[Path, ...]) -> dict[Path, str | None]:
    return {
        path: digest
        for path, (_status, digest) in _validate_runtime_bytecode_details(paths).items()
    }


def _captured_file_identity_matches(captured: _CapturedFile) -> bool:
    _require_no_lexical_symlink_components(
        captured.physical_path,
        label="captured file",
    )
    details = captured.physical_path.lstat()
    return (
        stat.S_ISREG(details.st_mode)
        and details.st_dev == captured.device
        and details.st_ino == captured.inode
        and details.st_size == captured.physical_size_bytes
        and details.st_mtime_ns == captured.modified_ns
        and details.st_ctime_ns == captured.changed_ns
    )


def _capture_open_file(
    path: Path,
    display_path: str,
    *,
    include_bytes: bool,
    use_runtime_digest_cache: bool = False,
) -> tuple[_CapturedFile, bytes | None]:
    _require_safe_relative_path(display_path, "captured file")
    if include_bytes and use_runtime_digest_cache:
        raise ValueError("captured bytes cannot use the runtime digest cache")
    _require_no_lexical_symlink_components(path, label="captured file")
    if path.is_symlink():
        raise ValueError(f"symbolic links are not valid training inputs: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    chunks: list[bytes] | None = [] if include_bytes else None
    digest_hex: str | None = None
    size_bytes: int | None = None
    cache_key: tuple[str, int, int, int, int, int] | None = None
    try:
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"training input is not a regular file: {path}")
            if use_runtime_digest_cache:
                cache_key = (
                    os.path.abspath(path),
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                with _RUNTIME_PAYLOAD_DIGEST_CACHE_LOCK:
                    cached = _RUNTIME_PAYLOAD_DIGEST_CACHE.get(cache_key)
                    if cached is not None:
                        _RUNTIME_PAYLOAD_DIGEST_CACHE.move_to_end(cache_key)
                if cached is not None:
                    digest_hex, size_bytes = cached
            if digest_hex is None:
                digest = hashlib.sha256()
                size_bytes = 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size_bytes += len(chunk)
                    if chunks is not None:
                        chunks.append(chunk)
                digest_hex = digest.hexdigest()
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
    assert digest_hex is not None
    assert size_bytes is not None
    if identity_before != identity_after or size_bytes != after.st_size:
        raise RuntimeError(f"training input changed while it was captured: {path}")
    if cache_key is not None:
        with _RUNTIME_PAYLOAD_DIGEST_CACHE_LOCK:
            _RUNTIME_PAYLOAD_DIGEST_CACHE[cache_key] = (digest_hex, size_bytes)
            _RUNTIME_PAYLOAD_DIGEST_CACHE.move_to_end(cache_key)
            while (
                len(_RUNTIME_PAYLOAD_DIGEST_CACHE)
                > _RUNTIME_PAYLOAD_DIGEST_CACHE_MAX_ENTRIES
            ):
                _RUNTIME_PAYLOAD_DIGEST_CACHE.popitem(last=False)
    captured = _CapturedFile(
        evidence=TrainingFileEvidence(
            path=display_path,
            sha256=digest_hex,
            size_bytes=size_bytes,
        ),
        physical_path=path,
        physical_size_bytes=after.st_size,
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
    lexical = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    _require_no_lexical_symlink_components(lexical, label=label)
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


def _evidence_sequence_sha256(values: tuple[BaseModel, ...]) -> str:
    payload = json.dumps(
        [value.model_dump(mode="json") for value in values],
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


_install_runtime_provenance_audit()
