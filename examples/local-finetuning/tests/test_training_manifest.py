"""Success-bound evidence tests for local MLX-LM adapter training."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from importlib.metadata import PathDistribution
from pathlib import Path

import pytest

from aai_local_finetuning import training

_REVISION = "a" * 40
_ORIGINAL_SYS_PATH = tuple(sys.path)


@dataclass(frozen=True)
class _Inputs:
    config_path: Path
    contract: training.TrainingInputContract
    model_dir: Path
    data_dir: Path
    adapter_dir: Path


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_training_inputs(
    root: Path,
    *,
    iterations: int = 12,
    config_name: str = "lora.yaml",
    resume_adapter_file: str = "null",
) -> _Inputs:
    source_dir = root / "src" / "aai_local_finetuning"
    source_dir.mkdir(parents=True)
    (source_dir / "training.py").write_text(
        '"""Fixture training source."""\n',
        encoding="utf-8",
    )
    scripts_dir = root / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "notebook_pedagogy.py").write_text(
        '"""Fixture notebook pedagogy source."""\n',
        encoding="utf-8",
    )
    (scripts_dir / "render_notebooks.py").write_text(
        '"""Fixture notebook renderer source."""\n',
        encoding="utf-8",
    )
    model_dir = root / "models" / "tiny"
    model_dir.mkdir(parents=True)
    runtime = {
        "config.json": b'{"model_type":"tiny"}\n',
        "model.safetensors": b"base-model-v1",
    }
    for name, content in runtime.items():
        (model_dir / name).write_bytes(content)
    (model_dir / "LOCAL_REVISION").write_text(_REVISION + "\n", encoding="utf-8")
    (model_dir / "README.md").write_text("local test model\n", encoding="utf-8")

    data_dir = root / "data" / "processed" / "study-v1"
    data_dir.mkdir(parents=True)
    dataset = {
        "manifest.json": b'{"dataset_fingerprint":"study-v1"}\n',
        "train.jsonl": b'{"split":"train"}\n',
        "valid.jsonl": b'{"split":"valid"}\n',
        "test.jsonl": b'{"split":"test"}\n',
        "notes.txt": b"all dataset files are bound\n",
    }
    for name, content in dataset.items():
        (data_dir / name).write_bytes(content)

    config_path = root / "configs" / "training" / config_name
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            (
                "model: models/tiny",
                "data: data/processed/study-v1",
                f"iters: {iterations}",
                "save_every: 4",
                f"resume_adapter_file: {resume_adapter_file}",
                "adapter_path: artifacts/adapters/study-lora-v1",
                "",
            )
        ),
        encoding="utf-8",
    )
    contract = training.TrainingInputContract(
        model_path="models/tiny",
        model_revision=_REVISION,
        model_runtime_files=tuple(
            training.ExpectedFileHash(path=name, sha256=_sha256(content))
            for name, content in sorted(runtime.items())
        ),
        data_path="data/processed/study-v1",
    )
    return _Inputs(
        config_path=config_path,
        contract=contract,
        model_dir=model_dir,
        data_dir=data_dir,
        adapter_dir=root / "artifacts" / "adapters" / "study-lora-v1",
    )


def _successful_mlx_run(
    *,
    adapter_bytes: bytes = b"adapter-v1",
    adapter_config_bytes: bytes = b'{"rank":8}\n',
    after_write: Callable[[], None] | None = None,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        staging = Path(command[command.index("--adapter-path") + 1])
        (staging / "adapters.safetensors").write_bytes(adapter_bytes)
        (staging / "adapter_config.json").write_bytes(adapter_config_bytes)
        if after_write is not None:
            after_write()
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=("Train loss 1.25\n" "Val loss 1.50\n" "Peak mem 2.00 GB\n"),
        )

    return run


@pytest.fixture
def training_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(training, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(training, "require_apple_silicon", lambda: None)
    install_root = tmp_path / "site-packages"
    package_root = install_root / "manifest_runtime"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text(
        '"""Synthetic manifest-test runtime."""\n',
        encoding="utf-8",
    )
    metadata_root = install_root / "manifest_runtime-1.0.0.dist-info"
    metadata_root.mkdir()
    (metadata_root / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: manifest-runtime\nVersion: 1.0.0\n",
        encoding="utf-8",
    )
    (metadata_root / "RECORD").write_text(
        (
            "manifest_runtime/__init__.py,,\n"
            "manifest_runtime-1.0.0.dist-info/METADATA,,\n"
            "manifest_runtime-1.0.0.dist-info/RECORD,,\n"
        ),
        encoding="utf-8",
    )
    distribution = PathDistribution(metadata_root)
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (distribution,),
    )
    base_prefix = Path(sys.base_prefix).resolve(strict=True)
    stdlib_roots = tuple(training._python_stdlib_root_tokens())
    initial_module_names = frozenset(training.sys.modules)
    inherited_loaded_modules = training._runtime_loaded_modules
    allowed_module_roots = (
        *stdlib_roots,
        (tmp_path / "src").resolve(strict=False),
        install_root.resolve(strict=True),
    )

    def loaded_modules() -> tuple[tuple[str, object], ...]:
        selected: list[tuple[str, object]] = []
        for name, module in inherited_loaded_modules():
            if name not in initial_module_names:
                selected.append((name, module))
                continue
            spec = getattr(module, "__spec__", None)
            origin = getattr(spec, "origin", None)
            if spec is None or origin in {"built-in", "frozen"}:
                selected.append((name, module))
                continue
            if origin is None:
                locations = tuple(getattr(spec, "submodule_search_locations", ()) or ())
                if not locations:
                    selected.append((name, module))
                    continue
                paths = tuple(
                    Path(location).resolve(strict=False) for location in locations
                )
            else:
                paths = (Path(origin).resolve(strict=False),)
            if all(
                any(
                    path == root or path.is_relative_to(root)
                    for root in allowed_module_roots
                )
                for path in paths
            ):
                selected.append((name, module))
        return tuple(selected)

    monkeypatch.setattr(training, "_runtime_loaded_modules", loaded_modules)
    stdlib_paths: list[str] = []
    for raw_path in _ORIGINAL_SYS_PATH:
        path = Path(raw_path or os.getcwd())
        if not path.is_absolute():
            path = Path(os.path.abspath(path))
        if path in stdlib_roots or path in training._python_missing_stdlib_archives(
            base_prefix
        ):
            stdlib_paths.append(path.as_posix())
    monkeypatch.setattr(
        training.sys,
        "path",
        [
            (tmp_path / "src").as_posix(),
            install_root.as_posix(),
            *stdlib_paths,
        ],
    )
    return tmp_path


def _run(
    inputs: _Inputs,
    **kwargs: object,
) -> training.TrainingEvidence:
    return training.run_lora(
        config_path=inputs.config_path,
        expected_inputs=inputs.contract,
        **kwargs,  # type: ignore[arg-type]
    )


def _verify(inputs: _Inputs) -> training.TrainingManifest:
    return training.verify_training_manifest(
        inputs.adapter_dir,
        config_path=inputs.config_path,
        expected_inputs=inputs.contract,
    )


def test_success_manifest_binds_all_model_data_and_adapter_bytes(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(training.subprocess, "run", _successful_mlx_run())

    evidence = _run(inputs)
    snapshot = training.require_valid_training_snapshot(
        inputs.adapter_dir,
        config_path=inputs.config_path,
        expected_inputs=inputs.contract,
    )
    manifest = snapshot.manifest

    assert {item.path for item in manifest.model_files} == {
        "LOCAL_REVISION",
        "config.json",
        "model.safetensors",
    }
    assert {item.path for item in manifest.data_files} == {
        "manifest.json",
        "notes.txt",
        "test.jsonl",
        "train.jsonl",
        "valid.jsonl",
    }
    assert manifest.model_path == "models/tiny"
    assert manifest.model_revision == _REVISION
    assert manifest.data_path == "data/processed/study-v1"
    assert manifest.adapter_size_bytes == len(b"adapter-v1")
    assert manifest.effective_config["iters"] == 12
    assert manifest.schema_version == "4.0.0"
    assert tuple(item.path for item in manifest.execution_contract.source_files) == (
        "scripts/notebook_pedagogy.py",
        "scripts/render_notebooks.py",
        "src/aai_local_finetuning/training.py",
    )
    assert manifest.execution_contract.runtime_packages
    assert manifest.execution_contract.python_version == platform.python_version()
    assert manifest.execution_contract.python_implementation == (
        platform.python_implementation()
    )
    assert evidence.execution_contract_sha256 == manifest.execution_contract_sha256
    assert evidence.training_manifest_sha256 == snapshot.manifest_sha256
    assert snapshot.manifest_sha256 == _sha256(snapshot.raw_manifest_bytes)
    assert ".training-" not in " ".join(evidence.command)
    assert _verify(inputs) == manifest


def test_training_child_ignores_mutable_python_import_overrides(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root)
    malicious_root = training_root / "unbound-pythonpath"
    fake_mlx = malicious_root / "mlx_lm"
    fake_mlx.mkdir(parents=True)
    (fake_mlx / "__main__.py").write_text(
        "raise RuntimeError('unbound mlx_lm executed')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", malicious_root.as_posix())

    def isolated_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        assert not any(name.upper().startswith("PYTHON") for name in environment)
        assert command[:4] == [sys.executable, "-I", "-m", "mlx_lm"]
        original = os.environ["PYTHONPATH"]
        os.environ["PYTHONPATH"] = (malicious_root / "replacement").as_posix()
        os.environ["PYTHONPATH"] = original
        staging = Path(command[command.index("--adapter-path") + 1])
        (staging / "adapters.safetensors").write_bytes(b"isolated-adapter")
        (staging / "adapter_config.json").write_text('{"rank":8}\n')
        return subprocess.CompletedProcess(command, 0, stdout="Train loss 1.0\n")

    monkeypatch.setattr(training.subprocess, "run", isolated_run)
    evidence = _run(inputs)

    assert evidence.command[:4] == ["<python>", "-I", "-m", "mlx_lm"]


def test_custom_configuration_requires_explicit_trusted_inputs(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root, config_name="custom.yaml")

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        pytest.fail("training launched without a trusted expected-input contract")

    monkeypatch.setattr(training.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match="expected_inputs is required"):
        training.run_lora(config_path=inputs.config_path)


@pytest.mark.parametrize(
    ("changed_input", "message"),
    (
        ("model_path", "trusted model path"),
        ("data_path", "trusted task data path"),
        ("model_revision", "model revision"),
        ("runtime_hash", "runtime SHA-256"),
        ("loader_file", "unverified entries"),
    ),
)
def test_trusted_contract_is_checked_before_process_launch(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_input: str,
    message: str,
) -> None:
    inputs = _write_training_inputs(training_root)
    if changed_input == "model_path":
        inputs.config_path.write_text(
            inputs.config_path.read_text().replace(
                "model: models/tiny",
                "model: models/other",
            )
        )
    elif changed_input == "data_path":
        inputs.config_path.write_text(
            inputs.config_path.read_text().replace(
                "data: data/processed/study-v1",
                "data: data/processed/other-v1",
            )
        )
    elif changed_input == "model_revision":
        (inputs.model_dir / "LOCAL_REVISION").write_text("b" * 40 + "\n")
    elif changed_input == "loader_file":
        (inputs.model_dir / "chat_template.jinja").write_text(
            "{{ messages }}\n",
            encoding="utf-8",
        )
    else:
        (inputs.model_dir / "model.safetensors").write_bytes(b"different-model")

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        pytest.fail("training launched with mismatched trusted inputs")

    monkeypatch.setattr(training.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match=message):
        _run(inputs)


def test_resume_adapter_is_rejected_until_its_bytes_can_be_bound(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(
        training_root,
        resume_adapter_file="artifacts/adapters/previous.safetensors",
    )

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        pytest.fail("resume training launched without binding the resume bytes")

    monkeypatch.setattr(training.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match="resume_adapter_file is not supported"):
        _run(inputs)


@pytest.mark.parametrize(
    ("changed_input", "message"),
    (
        ("config", "configuration changed"),
        ("model", "model files changed"),
        ("data", "data files changed"),
        ("model_restored", "model files changed"),
        ("runtime_restored", "runtime package set changed"),
    ),
)
def test_every_training_input_is_rechecked_after_subprocess(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_input: str,
    message: str,
) -> None:
    inputs = _write_training_inputs(training_root)

    def change_input() -> None:
        if changed_input == "config":
            inputs.config_path.write_text(
                inputs.config_path.read_text() + "# changed\n"
            )
        elif changed_input == "model":
            (inputs.model_dir / "model.safetensors").write_bytes(b"changed-model")
        elif changed_input == "data":
            (inputs.data_dir / "train.jsonl").write_bytes(b'{"changed":true}\n')
        elif changed_input == "model_restored":
            model = inputs.model_dir / "model.safetensors"
            original = model.read_bytes()
            model.write_bytes(b"temporarily-changed-model")
            model.write_bytes(original)
        else:
            runtime = (
                training_root / "site-packages" / "manifest_runtime" / "__init__.py"
            )
            original = runtime.read_bytes()
            runtime.write_bytes(b'VALUE = "temporarily changed"\n')
            runtime.write_bytes(original)

    monkeypatch.setattr(
        training.subprocess,
        "run",
        _successful_mlx_run(after_write=change_input),
    )
    with pytest.raises(RuntimeError, match=message):
        _run(inputs)

    assert not inputs.adapter_dir.exists()
    assert not (training_root / "artifacts" / "training" / "latest.json").exists()


@pytest.mark.parametrize(
    ("adapter_bytes", "adapter_config", "message"),
    (
        (b"", b'{"rank":8}\n', "empty adapter weight"),
        (b"adapter", b"not-json", "not valid JSON"),
        (b"adapter", b"[]", "must be a JSON object"),
        (b"adapter", b"", "not valid JSON"),
    ),
)
def test_invalid_adapter_outputs_are_never_published(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_bytes: bytes,
    adapter_config: bytes,
    message: str,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(
        training.subprocess,
        "run",
        _successful_mlx_run(
            adapter_bytes=adapter_bytes,
            adapter_config_bytes=adapter_config,
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        _run(inputs)

    assert not inputs.adapter_dir.exists()
    assert not (training_root / "artifacts" / "training" / "latest.json").exists()


def test_failed_retrain_invalidates_success_marker_but_preserves_old_bytes(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(training.subprocess, "run", _successful_mlx_run())
    _run(inputs)
    old_bytes = (inputs.adapter_dir / "adapters.safetensors").read_bytes()

    def fail(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="training failed\n")

    monkeypatch.setattr(training.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        _run(inputs)

    assert (inputs.adapter_dir / "adapters.safetensors").read_bytes() == old_bytes
    assert not (inputs.adapter_dir / training.TRAINING_MANIFEST_NAME).exists()
    assert not (training_root / "artifacts" / "training" / "latest.json").exists()
    with pytest.raises(training.TrainingManifestError, match="missing"):
        _verify(inputs)


def test_zero_exit_without_new_outputs_cannot_rebless_old_adapter(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(training.subprocess, "run", _successful_mlx_run())
    _run(inputs)
    old_bytes = (inputs.adapter_dir / "adapters.safetensors").read_bytes()

    def omit_outputs(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="finished\n")

    monkeypatch.setattr(training.subprocess, "run", omit_outputs)
    with pytest.raises(RuntimeError, match="without the required adapter outputs"):
        _run(inputs)

    assert (inputs.adapter_dir / "adapters.safetensors").read_bytes() == old_bytes
    assert not (inputs.adapter_dir / training.TRAINING_MANIFEST_NAME).exists()
    assert not (training_root / "artifacts" / "training" / "latest.json").exists()


@pytest.mark.parametrize("changed_input", ("adapter", "adapter_config", "config"))
def test_verifier_rejects_changed_adapter_or_configuration(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_input: str,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(training.subprocess, "run", _successful_mlx_run())
    _run(inputs)

    if changed_input == "adapter":
        (inputs.adapter_dir / "adapters.safetensors").write_bytes(b"tampered")
    elif changed_input == "adapter_config":
        (inputs.adapter_dir / "adapter_config.json").write_text('{"rank":16}\n')
    else:
        inputs.config_path.write_text(inputs.config_path.read_text() + "# changed\n")

    with pytest.raises(training.TrainingManifestError):
        _verify(inputs)


@pytest.mark.parametrize(
    ("changed_input", "message"),
    (
        ("config", "configuration changed"),
        ("model", "model files changed"),
        ("data", "data files changed"),
    ),
)
def test_snapshot_validation_rechecks_the_complete_captured_plan(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_input: str,
    message: str,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(training.subprocess, "run", _successful_mlx_run())
    _run(inputs)
    capture_directory = training._capture_directory_files
    first_capture = True

    def mutate_after_plan_capture(
        directory: Path,
        label: str,
    ) -> tuple[training._CapturedFile, ...]:
        nonlocal first_capture
        captured = capture_directory(directory, label)
        if first_capture:
            first_capture = False
            if changed_input == "config":
                inputs.config_path.write_text(
                    inputs.config_path.read_text(encoding="utf-8") + "# changed\n",
                    encoding="utf-8",
                )
            elif changed_input == "model":
                (inputs.model_dir / "model.safetensors").write_bytes(b"changed-model")
            else:
                (inputs.data_dir / "train.jsonl").write_bytes(b'{"changed":true}\n')
        return captured

    monkeypatch.setattr(
        training,
        "_capture_directory_files",
        mutate_after_plan_capture,
    )
    with pytest.raises(training.TrainingManifestError, match=message):
        training.require_valid_training_snapshot(
            inputs.adapter_dir,
            config_path=inputs.config_path,
            expected_inputs=inputs.contract,
        )


@pytest.mark.parametrize("changed_input", ("model_runtime", "data_extra"))
def test_verifier_rejects_changes_to_any_bound_input_file(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_input: str,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(training.subprocess, "run", _successful_mlx_run())
    _run(inputs)

    if changed_input == "model_runtime":
        (inputs.model_dir / "config.json").write_text('{"model_type":"changed"}\n')
    else:
        (inputs.data_dir / "notes.txt").write_text("changed dataset note\n")

    with pytest.raises(training.TrainingManifestError, match="stale or mismatched"):
        _verify(inputs)


def test_verifier_rejects_source_code_changed_after_training(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(training.subprocess, "run", _successful_mlx_run())
    _run(inputs)

    source = training_root / "src" / "aai_local_finetuning" / "training.py"
    source.write_text(source.read_text() + "# changed evaluator logic\n")

    with pytest.raises(
        training.TrainingManifestError,
        match="evaluator and training source code",
    ):
        _verify(inputs)


def test_verifier_rejects_runtime_package_set_changed_after_training(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(training.subprocess, "run", _successful_mlx_run())
    _run(inputs)
    capture_snapshot = training.capture_execution_snapshot

    def changed_snapshot() -> training.ExecutionSnapshot:
        snapshot = capture_snapshot()
        contract = snapshot.execution_contract
        changed = contract.runtime_packages + (
            training.RuntimePackageEvidence(
                name="zz-runtime-mutation-test",
                version="1.0.0",
                payload_file_count=1,
                payload_size_bytes=10,
                payload_files_sha256="f" * 64,
            ),
        )
        changed_contract = training.ExecutionContract(
            python_version=contract.python_version,
            python_implementation=contract.python_implementation,
            operating_system=contract.operating_system,
            machine=contract.machine,
            source_files=contract.source_files,
            source_files_sha256=contract.source_files_sha256,
            runtime_packages=changed,
            runtime_packages_sha256=training._evidence_sequence_sha256(changed),
        )
        return replace(
            snapshot,
            execution_contract=changed_contract,
            execution_contract_sha256=training.execution_contract_sha256(
                changed_contract
            ),
        )

    monkeypatch.setattr(training, "capture_execution_snapshot", changed_snapshot)

    with pytest.raises(training.TrainingManifestError, match="runtime package set"):
        _verify(inputs)


def test_explicit_smoke_expectations_round_trip_without_trusting_manifest(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root, iterations=12)
    smoke_adapter = training_root / "artifacts" / "adapters" / "smoke"
    monkeypatch.setattr(training.subprocess, "run", _successful_mlx_run())
    _run(inputs, iterations=1, adapter_path=smoke_adapter, log_name="smoke")

    snapshot = training.require_valid_training_snapshot(
        smoke_adapter,
        config_path=inputs.config_path,
        expected_iterations=1,
        expected_adapter_path=smoke_adapter,
        expected_inputs=inputs.contract,
    )
    assert snapshot.manifest.effective_config["iters"] == 1
    assert snapshot.manifest.adapter_path == "artifacts/adapters/smoke"
    assert (
        training.verify_training_manifest(
            smoke_adapter,
            config_path=inputs.config_path,
            expected_iterations=1,
            expected_adapter_path=smoke_adapter,
            expected_inputs=inputs.contract,
        )
        == snapshot.manifest
    )
    with pytest.raises(training.TrainingManifestError):
        training.verify_training_manifest(
            smoke_adapter,
            config_path=inputs.config_path,
            expected_iterations=2,
            expected_adapter_path=smoke_adapter,
            expected_inputs=inputs.contract,
        )
    with pytest.raises(training.TrainingManifestError, match="different adapter"):
        training.verify_training_manifest(
            smoke_adapter,
            config_path=inputs.config_path,
            expected_iterations=1,
            expected_inputs=inputs.contract,
        )


def test_recheck_requires_exact_captured_manifest_bytes(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(training.subprocess, "run", _successful_mlx_run())
    _run(inputs)
    snapshot = training.require_valid_training_snapshot(
        inputs.adapter_dir,
        config_path=inputs.config_path,
        expected_inputs=inputs.contract,
    )
    manifest_path = inputs.adapter_dir / training.TRAINING_MANIFEST_NAME
    manifest_path.write_bytes(snapshot.raw_manifest_bytes + b"\n")

    with pytest.raises(training.TrainingManifestError, match="manifest changed"):
        training.recheck_training_snapshot(snapshot)


def test_evidence_write_failure_rolls_back_adapter_manifest_and_prior_evidence(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(training.subprocess, "run", _successful_mlx_run())
    _run(inputs)
    old_adapter = (inputs.adapter_dir / "adapters.safetensors").read_bytes()
    old_manifest = (inputs.adapter_dir / training.TRAINING_MANIFEST_NAME).read_bytes()
    evidence_path = training_root / "artifacts" / "training" / "latest.json"
    old_evidence = evidence_path.read_bytes()

    monkeypatch.setattr(
        training.subprocess,
        "run",
        _successful_mlx_run(adapter_bytes=b"adapter-v2"),
    )

    def fail_evidence_write(_path: Path, _evidence: object) -> None:
        raise OSError("simulated evidence write failure")

    monkeypatch.setattr(training, "_write_json_atomic", fail_evidence_write)
    with pytest.raises(OSError, match="evidence write failure"):
        _run(inputs)

    assert (inputs.adapter_dir / "adapters.safetensors").read_bytes() == old_adapter
    assert (
        inputs.adapter_dir / training.TRAINING_MANIFEST_NAME
    ).read_bytes() == old_manifest
    assert evidence_path.read_bytes() == old_evidence
    assert _verify(inputs).adapter_sha256 == _sha256(old_adapter)
    assert not tuple(inputs.adapter_dir.parent.glob(".*.previous-*"))
    assert not tuple(inputs.adapter_dir.parent.glob(".*.failed-*"))


def test_concurrent_training_attempts_serialize_without_mixed_evidence(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root)
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    counter_lock = threading.Lock()
    call_count = 0

    def serialized_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        with counter_lock:
            call_count += 1
            call_number = call_count
        if call_number == 1:
            first_entered.set()
            if not release_first.wait(timeout=3):
                raise RuntimeError("test did not release the first training run")
        else:
            second_entered.set()
        staging = Path(command[command.index("--adapter-path") + 1])
        (staging / "adapters.safetensors").write_bytes(
            f"adapter-{call_number}".encode()
        )
        (staging / "adapter_config.json").write_text(
            json.dumps({"run": call_number}) + "\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout="Train loss 1.0\n")

    monkeypatch.setattr(training.subprocess, "run", serialized_run)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_run, inputs)
        assert first_entered.wait(timeout=3)
        second = executor.submit(_run, inputs)
        try:
            assert not second_entered.wait(timeout=0.15)
        finally:
            release_first.set()
        first_evidence = first.result(timeout=3)
        second_evidence = second.result(timeout=3)

    assert second_entered.is_set()
    assert first_evidence.adapter_sha256 == _sha256(b"adapter-1")
    assert second_evidence.adapter_sha256 == _sha256(b"adapter-2")
    final_adapter = (inputs.adapter_dir / "adapters.safetensors").read_bytes()
    final_manifest = training.TrainingManifest.model_validate_json(
        (inputs.adapter_dir / training.TRAINING_MANIFEST_NAME).read_bytes()
    )
    final_evidence = training.TrainingEvidence.model_validate_json(
        (training_root / "artifacts" / "training" / "latest.json").read_bytes()
    )
    assert final_adapter == b"adapter-2"
    assert final_manifest.adapter_sha256 == _sha256(final_adapter)
    assert final_evidence.adapter_sha256 == final_manifest.adapter_sha256
    assert final_evidence.training_manifest_sha256 == _sha256(
        (inputs.adapter_dir / training.TRAINING_MANIFEST_NAME).read_bytes()
    )


def test_exclusive_training_waits_for_public_shared_lock(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(training.subprocess, "run", _successful_mlx_run())
    _run(inputs)
    training_entered = threading.Event()
    monkeypatch.setattr(
        training.subprocess,
        "run",
        _successful_mlx_run(
            adapter_bytes=b"after-shared-lock",
            after_write=training_entered.set,
        ),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        with training.shared_adapter_lock(inputs.adapter_dir):
            snapshot = training.require_valid_training_snapshot(
                inputs.adapter_dir,
                config_path=inputs.config_path,
                expected_inputs=inputs.contract,
            )
            assert training.recheck_training_snapshot(snapshot) is snapshot
            future = executor.submit(_run, inputs)
            assert not training_entered.wait(timeout=0.15)
        assert training_entered.wait(timeout=3)
        future.result(timeout=3)

    assert (inputs.adapter_dir / "adapters.safetensors").read_bytes() == (
        b"after-shared-lock"
    )


def test_shared_verification_waits_for_exclusive_training(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root)
    training_entered = threading.Event()
    release_training = threading.Event()
    verification_started = threading.Event()
    verification_finished = threading.Event()

    def blocking_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        training_entered.set()
        if not release_training.wait(timeout=3):
            raise RuntimeError("test did not release exclusive training")
        staging = Path(command[command.index("--adapter-path") + 1])
        (staging / "adapters.safetensors").write_bytes(b"locked-adapter")
        (staging / "adapter_config.json").write_text('{"rank":8}\n')
        return subprocess.CompletedProcess(command, 0, stdout="Train loss 1.0\n")

    def verify_after_training() -> training.TrainingManifest:
        verification_started.set()
        try:
            return _verify(inputs)
        finally:
            verification_finished.set()

    monkeypatch.setattr(training.subprocess, "run", blocking_run)
    with ThreadPoolExecutor(max_workers=2) as executor:
        training_future = executor.submit(_run, inputs)
        assert training_entered.wait(timeout=3)
        verification_future = executor.submit(verify_after_training)
        assert verification_started.wait(timeout=3)
        try:
            assert not verification_finished.wait(timeout=0.15)
        finally:
            release_training.set()
        training_future.result(timeout=3)
        manifest = verification_future.result(timeout=3)

    assert verification_finished.is_set()
    assert manifest.adapter_sha256 == _sha256(b"locked-adapter")


def test_adapter_lock_path_is_stable_outside_adapter_and_rejects_symlink(
    training_root: Path,
) -> None:
    inputs = _write_training_inputs(training_root)
    lock_path = training._adapter_lock_path(inputs.adapter_dir)
    assert lock_path.parent == inputs.adapter_dir.parent
    assert lock_path.name == f".{inputs.adapter_dir.name}.lock"
    assert not lock_path.is_relative_to(inputs.adapter_dir)

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    unrelated = training_root / "unrelated-lock-target"
    unrelated.write_text("do not follow\n")
    lock_path.symlink_to(unrelated)
    with pytest.raises(RuntimeError, match="could not acquire adapter lock"):
        with training.shared_adapter_lock(inputs.adapter_dir):
            pytest.fail("shared lock followed an unsafe symbolic link")


def test_adapter_lock_fails_clearly_on_unsupported_platform(
    training_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_training_inputs(training_root)
    monkeypatch.setattr(training.sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="supported macOS or Linux"):
        with training.shared_adapter_lock(inputs.adapter_dir):
            pytest.fail("adapter lock unexpectedly worked on an unsupported platform")


@pytest.mark.skipif(
    sys.platform != "darwin" and not sys.platform.startswith("linux"),
    reason="fcntl adapter locks are supported on macOS and Linux",
)
def test_file_lock_coordinates_a_separate_process(
    training_root: Path,
) -> None:
    inputs = _write_training_inputs(training_root)
    source_root = Path(training.__file__).resolve().parents[1]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else str(source_root) + os.pathsep + existing_pythonpath
    )
    child_source = """
import sys
from pathlib import Path
from aai_local_finetuning import training

training.PROJECT_ROOT = Path(sys.argv[1])
with training.shared_adapter_lock(Path(sys.argv[2])):
    print("locked", flush=True)
    sys.stdin.readline()
"""
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child_source,
            str(training_root),
            str(inputs.adapter_dir),
        ],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    exclusive_entered = threading.Event()

    def acquire_exclusive() -> None:
        with training._exclusive_adapter_lock(inputs.adapter_dir):
            exclusive_entered.set()

    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(acquire_exclusive)
            assert not exclusive_entered.wait(timeout=0.15)
            assert child.stdin is not None
            child.stdin.write("release\n")
            child.stdin.flush()
            assert exclusive_entered.wait(timeout=3)
            future.result(timeout=3)
        _, stderr = child.communicate(timeout=3)
        assert child.returncode == 0, stderr
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=3)
