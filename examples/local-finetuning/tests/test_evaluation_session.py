"""Regression tests for inference-wide source/runtime evidence sessions."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import py_compile
import shutil
import subprocess
import sys
from argparse import Namespace
from dataclasses import dataclass
from importlib.metadata import PackagePath, PathDistribution
from inspect import Parameter, signature
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_local_finetuning import cli, training
from aai_local_finetuning.capstone import (
    DatasetSplit,
    build_records,
    deterministic_capstone_predictions,
    evaluate_capstone_predictions,
)
from aai_local_finetuning.evaluation import (
    DeterministicInferenceConfig,
    EvaluationRecord,
    Evaluator,
    Prediction,
    SupportOutput,
    evaluate_predictions,
    recheck_evaluation_session,
    start_evaluation_session,
)
from aai_local_finetuning.evaluation import session as evaluation_session_module

_ORIGINAL_SYS_PATH = tuple(sys.path)


def _activate_test_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    project_root: Path,
    install_root: Path,
    exposed_roots: tuple[Path, ...] = (),
) -> None:
    """Make a synthetic distribution inventory match effective import roots."""

    base_prefix = Path(sys.base_prefix).resolve(strict=True)
    stdlib_roots = tuple(training._python_stdlib_root_tokens())
    initial_module_names = frozenset(training.sys.modules)
    inherited_loaded_modules = training._runtime_loaded_modules
    allowed_module_roots = (
        *stdlib_roots,
        (project_root / "src").resolve(strict=True),
        install_root.resolve(strict=True),
        *(path.resolve(strict=True) for path in exposed_roots),
    )

    def loaded_modules() -> tuple[tuple[str, object], ...]:
        selected: list[tuple[str, object]] = []
        for name, module in inherited_loaded_modules():
            if name not in initial_module_names:
                selected.append((name, module))
                continue
            spec = getattr(module, "__spec__", None)
            origin = getattr(spec, "origin", None)
            if spec is None or origin in {None, "built-in", "frozen"}:
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
    stdlib_entries: list[str] = []
    for raw_entry in _ORIGINAL_SYS_PATH:
        candidate = Path(raw_entry or os.getcwd())
        if not candidate.is_absolute():
            candidate = Path(os.path.abspath(candidate))
        missing_archives = training._python_missing_stdlib_archives(base_prefix)
        if candidate in stdlib_roots or candidate in missing_archives:
            stdlib_entries.append(candidate.as_posix())
    monkeypatch.setattr(
        training.sys,
        "path",
        [
            (project_root / "src").as_posix(),
            install_root.as_posix(),
            *(path.as_posix() for path in exposed_roots),
            *stdlib_entries,
        ],
    )


@dataclass(frozen=True)
class GovernedRuntime:
    project_root: Path
    source_path: Path
    metadata_path: Path
    entry_points_path: Path
    package_payload_path: Path
    settings: object
    model_dir: Path
    model_runtime_path: Path
    revision_path: Path


def test_public_scorers_require_an_explicit_prestarted_session() -> None:
    for scorer in (
        Evaluator.evaluate,
        evaluate_predictions,
        evaluate_capstone_predictions,
    ):
        parameter = signature(scorer).parameters["evaluation_session"]
        assert parameter.kind is Parameter.KEYWORD_ONLY
        assert parameter.default is Parameter.empty


def _support_case() -> tuple[EvaluationRecord, Prediction]:
    target = SupportOutput(
        intent="recover_password",
        category="account",
        requires_escalation=False,
        response="I can help you recover access to your account.",
    )
    record = EvaluationRecord(
        example_id="session-support-1",
        input_text="I forgot my password.",
        target=target,
        source_dataset="session-fixture",
        source_version="1.0",
        system_prompt="Return valid JSON only.",
        metadata={"intent": target.intent, "category": target.category},
    )
    prediction = Prediction(
        example_id=record.example_id,
        raw_text=target.model_dump_json(),
        latency_ms=1.0,
        output_tokens=8,
        peak_memory_mb=16.0,
    )
    return record, prediction


@pytest.fixture
def governed_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> GovernedRuntime:
    """Provide the smallest governed tree and one installed distribution."""

    project_root = tmp_path / "project"
    source_path = project_root / "src" / "aai_local_finetuning" / "runtime.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text('"""Stable governed source."""\n', encoding="utf-8")
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "notebook_pedagogy.py").write_text(
        '"""Stable pedagogy source."""\n',
        encoding="utf-8",
    )
    (scripts_dir / "render_notebooks.py").write_text(
        '"""Stable notebook renderer."""\n',
        encoding="utf-8",
    )

    (
        distribution,
        metadata_path,
        entry_points_path,
        package_payload_path,
    ) = _write_distribution(
        tmp_path / "site-packages",
        name="session-runtime",
        version="1.0.0",
    )

    model_dir = project_root / "models" / "tiny-model"
    model_dir.mkdir(parents=True)
    model_runtime_path = model_dir / "model.safetensors"
    model_runtime_path.write_bytes(b"small verified model runtime\n")
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer_path.write_bytes(b'{"version":"1.0"}\n')
    revision_path = model_dir / "LOCAL_REVISION"
    revision = "a" * 40
    revision_path.write_text(revision + "\n", encoding="utf-8")
    verified_runtime_files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (model_runtime_path, tokenizer_path)
    }
    settings = SimpleNamespace(
        model=SimpleNamespace(
            repo="local/tiny-model",
            directory="models/tiny-model",
            revision=revision,
            verified_runtime_files=verified_runtime_files,
        )
    )

    monkeypatch.setattr(training, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (distribution,),
    )
    _activate_test_runtime(
        monkeypatch,
        project_root=project_root,
        install_root=metadata_path.parent.parent,
    )
    return GovernedRuntime(
        project_root=project_root,
        source_path=source_path,
        metadata_path=metadata_path,
        entry_points_path=entry_points_path,
        package_payload_path=package_payload_path,
        settings=settings,
        model_dir=model_dir,
        model_runtime_path=model_runtime_path,
        revision_path=revision_path,
    )


def _write_distribution(
    root: Path,
    *,
    name: str,
    version: str,
) -> tuple[PathDistribution, Path, Path, Path]:
    normalized_name = name.replace("-", "_")
    directory_name = f"{normalized_name}-{version}.dist-info"
    distribution_dir = root / directory_name
    distribution_dir.mkdir(parents=True)
    package_dir = root / normalized_name
    package_dir.mkdir()
    package_payload_path = package_dir / "__init__.py"
    package_payload_path.write_text(
        f'"""Runtime payload for {name} {version}."""\n',
        encoding="utf-8",
    )
    metadata_path = distribution_dir / "METADATA"
    metadata_path.write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    entry_points_path = distribution_dir / "entry_points.txt"
    entry_points_path.write_text(
        f"[console_scripts]\n{name} = {normalized_name}:main\n",
        encoding="utf-8",
    )
    (distribution_dir / "RECORD").write_text(
        (
            f"{normalized_name}/__init__.py,,\n"
            f"{directory_name}/METADATA,,\n"
            f"{directory_name}/entry_points.txt,,\n"
            f"{directory_name}/RECORD,,\n"
        ),
        encoding="utf-8",
    )
    return (
        PathDistribution(distribution_dir),
        metadata_path,
        entry_points_path,
        package_payload_path,
    )


def _replace_with_original_bytes(path: Path) -> None:
    """Change a file, then restore its bytes through a guaranteed new inode."""

    original = path.read_bytes()
    path.write_bytes(b"temporary drift that inference must not hide\n")
    restored = path.with_name(f".{path.name}.restored")
    restored.write_bytes(original)
    os.replace(restored, path)
    assert path.read_bytes() == original


def test_model_free_session_does_not_claim_or_recheck_model_lineage(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session()

    assert session.base_model_execution_contract is None
    assert session.base_model_execution_contract_sha256 is None
    _replace_with_original_bytes(governed_runtime.model_runtime_path)
    recheck_evaluation_session(session)


def test_model_session_contract_is_sorted_portable_and_content_addressed(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    contract = session.base_model_execution_contract

    assert contract is not None
    assert contract.repository == "local/tiny-model"
    assert contract.model_path == "models/tiny-model"
    assert contract.model_revision == "a" * 40
    assert tuple(item.path for item in contract.model_files) == (
        "LOCAL_REVISION",
        "model.safetensors",
        "tokenizer.json",
    )
    assert session.base_model_execution_contract_sha256 == (
        training.base_model_execution_contract_sha256(contract)
    )
    portable = json.dumps(contract.model_dump(mode="json"), sort_keys=True)
    assert str(governed_runtime.project_root) not in portable
    assert all(
        name not in portable
        for name in ("device", "inode", "modified_ns", "changed_ns")
    )
    assert repr(session) == "EvaluationSession()"


def test_model_session_rejects_runtime_bytes_outside_the_trusted_contract(
    governed_runtime: GovernedRuntime,
) -> None:
    governed_runtime.model_runtime_path.write_bytes(b"untrusted replacement\n")

    with pytest.raises(ValueError, match="runtime SHA-256 mismatch"):
        start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "relative_path",
    ("chat_template.jinja", "model-extra.safetensors"),
)
def test_model_session_rejects_unverified_loader_visible_files(
    governed_runtime: GovernedRuntime,
    relative_path: str,
) -> None:
    (governed_runtime.model_dir / relative_path).write_bytes(b"unverified runtime\n")

    with pytest.raises(ValueError, match="unverified entries"):
        start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]


def test_model_session_rejects_unverified_nested_chat_templates(
    governed_runtime: GovernedRuntime,
) -> None:
    template_dir = governed_runtime.model_dir / "additional_chat_templates"
    template_dir.mkdir()
    (template_dir / "default.jinja").write_text(
        "{{ messages }}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unverified entries"):
        start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", ("runtime", "revision"))
def test_model_session_rejects_file_drift_restored_to_the_same_bytes(
    governed_runtime: GovernedRuntime,
    target: str,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    path = (
        governed_runtime.model_runtime_path
        if target == "runtime"
        else governed_runtime.revision_path
    )

    _replace_with_original_bytes(path)

    with pytest.raises(RuntimeError, match="base-model.*changed"):
        recheck_evaluation_session(session)


def test_model_session_rejects_model_directory_replacement(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    prior_directory = governed_runtime.model_dir.with_name(".prior-model")
    os.replace(governed_runtime.model_dir, prior_directory)
    governed_runtime.model_dir.mkdir()
    for path in prior_directory.iterdir():
        os.replace(path, governed_runtime.model_dir / path.name)
    prior_directory.rmdir()

    with pytest.raises(RuntimeError, match="base-model.*changed"):
        recheck_evaluation_session(session)


def test_model_capture_is_sandwiched_by_execution_identity_rechecks(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_model = evaluation_session_module.capture_base_model_snapshot

    def capture_then_drift(settings: object) -> training.BaseModelSnapshot:
        snapshot = capture_model(settings)  # type: ignore[arg-type]
        governed_runtime.source_path.write_text(
            '"""Drift during model capture."""\n',
            encoding="utf-8",
        )
        return snapshot

    monkeypatch.setattr(
        evaluation_session_module,
        "capture_base_model_snapshot",
        capture_then_drift,
    )

    with pytest.raises(RuntimeError, match="source code.*changed"):
        start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]


def test_model_recheck_is_sandwiched_by_execution_identity_rechecks(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    recheck_model = evaluation_session_module.recheck_base_model_snapshot

    def recheck_then_drift(
        snapshot: training.BaseModelSnapshot,
    ) -> training.BaseModelSnapshot:
        checked = recheck_model(snapshot)
        governed_runtime.source_path.write_text(
            '"""Drift during model recheck."""\n',
            encoding="utf-8",
        )
        return checked

    monkeypatch.setattr(
        evaluation_session_module,
        "recheck_base_model_snapshot",
        recheck_then_drift,
    )

    with pytest.raises(RuntimeError, match="source code.*changed"):
        recheck_evaluation_session(session)


def test_prestarted_session_rejects_governed_source_drift_restored_before_scoring(
    governed_runtime: GovernedRuntime,
) -> None:
    record, prediction = _support_case()
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]

    # This stands in for inference importing or rewriting code and restoring the
    # exact original bytes before the scorer starts.
    _replace_with_original_bytes(governed_runtime.source_path)

    with pytest.raises(RuntimeError, match="changed|drift"):
        evaluate_predictions(
            [record],
            [prediction],
            evaluation_session=session,
            inference_config=DeterministicInferenceConfig(method="session-fixture"),
        )


def test_prestarted_session_rejects_package_metadata_drift_restored_before_scoring(
    governed_runtime: GovernedRuntime,
) -> None:
    records = build_records(DatasetSplit.TEST, 1)
    predictions = deterministic_capstone_predictions(records)
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]

    # The final name/version and bytes are unchanged; the session must still
    # notice that installed-package evidence changed during inference.
    _replace_with_original_bytes(governed_runtime.metadata_path)

    with pytest.raises(RuntimeError, match="changed|drift"):
        evaluate_capstone_predictions(
            records,
            predictions,
            evaluation_session=session,
            inference_config=DeterministicInferenceConfig(method="session-fixture"),
        )


def test_prestarted_session_rejects_installed_package_payload_mutation(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    package = next(
        item
        for item in session.execution_contract.runtime_packages
        if item.name == "session-runtime"
    )
    runtime_files = (
        governed_runtime.package_payload_path,
        governed_runtime.metadata_path,
        governed_runtime.entry_points_path,
        governed_runtime.metadata_path.with_name("RECORD"),
    )
    original = governed_runtime.package_payload_path.read_bytes()

    assert package.payload_file_count == len(runtime_files)
    assert package.payload_size_bytes > sum(
        len(path.read_bytes()) for path in runtime_files[:-1]
    )
    governed_runtime.package_payload_path.write_bytes(b"x" * len(original))

    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)


def test_execution_capture_fails_when_distribution_inventory_is_unavailable(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = SimpleNamespace(
        files=None,
        metadata={"Name": "session-runtime"},
    )
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (unavailable,),
    )

    with pytest.raises(RuntimeError, match="file inventory is unavailable"):
        start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]


def test_prestarted_session_rejects_unrecorded_importable_module(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    unrecorded = governed_runtime.package_payload_path.with_name("unrecorded.py")

    unrecorded.write_text('VALUE = "runtime drift"\n', encoding="utf-8")

    assert (
        training._capture_runtime_packages()
        != session.execution_contract.runtime_packages
    )
    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)


@pytest.mark.parametrize("entry_name", ("unowned_module.py", "unowned_package"))
def test_execution_capture_rejects_wholly_unrecorded_top_level_import(
    governed_runtime: GovernedRuntime,
    entry_name: str,
) -> None:
    install_root = governed_runtime.metadata_path.parent.parent
    entry = install_root / entry_name
    if entry.suffix:
        entry.write_text('VALUE = "unowned"\n', encoding="utf-8")
    else:
        entry.mkdir()
        (entry / "__init__.py").write_text(
            'VALUE = "unowned"\n',
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match="without distribution inventory ownership"):
        start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]


def test_execution_capture_rejects_unowned_top_level_sourceless_bytecode(
    governed_runtime: GovernedRuntime,
    tmp_path: Path,
) -> None:
    source = tmp_path / "rogue.py"
    source.write_text('VALUE = "rogue"\n', encoding="utf-8")
    pyc_path = governed_runtime.metadata_path.parent.parent / "rogue.pyc"
    py_compile.compile(
        source.as_posix(),
        cfile=pyc_path.as_posix(),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )

    with pytest.raises(RuntimeError, match="without distribution inventory ownership"):
        training.capture_execution_snapshot()


def test_virtualenv_distributionless_bootstrap_is_content_bound(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_root = governed_runtime.metadata_path.parent.parent
    (install_root / "_virtualenv.pth").write_text(
        "import _virtualenv\n",
        encoding="utf-8",
    )
    bootstrap = install_root / "_virtualenv.py"
    bootstrap.write_text('VALUE = "generated bootstrap"\n', encoding="utf-8")
    monkeypatch.delitem(training.sys.modules, "_virtualenv", raising=False)

    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]

    assert any(
        package.name == "python-environment-bootstrap"
        for package in session.execution_contract.runtime_packages
    )
    bootstrap.write_text('VALUE = "mutated bootstrap"\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)


def test_prestarted_session_rejects_entry_point_metadata_mutation(
    governed_runtime: GovernedRuntime,
) -> None:
    record_path = governed_runtime.metadata_path.with_name("RECORD")
    record_path.write_text(
        record_path.read_text(encoding="utf-8").replace(
            "session_runtime-1.0.0.dist-info/entry_points.txt,,\n",
            "",
        ),
        encoding="utf-8",
    )
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]

    governed_runtime.entry_points_path.write_text(
        "[console_scripts]\nsession-runtime = session_runtime:replacement\n",
        encoding="utf-8",
    )

    assert (
        training._capture_runtime_packages()
        != session.execution_contract.runtime_packages
    )
    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)


def test_prestarted_session_rejects_new_unrecorded_runtime_metadata(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]

    governed_runtime.metadata_path.with_name("new-runtime-hook.txt").write_text(
        "runtime-significant metadata\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)


def test_nested_runtime_directory_identity_detects_create_import_delete(
    governed_runtime: GovernedRuntime,
) -> None:
    nested = governed_runtime.package_payload_path.parent / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text("", encoding="utf-8")
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    transient = nested / "transient.py"
    transient.write_text('VALUE = "briefly imported"\n', encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "_transient_runtime_module",
        transient,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.VALUE == "briefly imported"
    transient.unlink()
    shutil.rmtree(nested / "__pycache__", ignore_errors=True)

    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)


def test_nested_governed_source_identity_detects_create_import_delete(
    governed_runtime: GovernedRuntime,
) -> None:
    nested = governed_runtime.source_path.parent / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text("", encoding="utf-8")
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    transient = nested / "transient.py"
    transient.write_text('VALUE = "briefly imported"\n', encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "_transient_governed_module",
        transient,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transient.unlink()
    shutil.rmtree(nested / "__pycache__", ignore_errors=True)

    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)


def test_runtime_bytecode_must_be_semantically_equal_to_captured_source(
    governed_runtime: GovernedRuntime,
    tmp_path: Path,
) -> None:
    pyc_path = Path(
        py_compile.compile(
            governed_runtime.package_payload_path.as_posix(),
            doraise=True,
        )
    )
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    recheck_evaluation_session(session)

    replacement_source = tmp_path / "replacement.py"
    replacement_source.write_text('VALUE = "unbound bytecode"\n', encoding="utf-8")
    py_compile.compile(
        replacement_source.as_posix(),
        cfile=pyc_path.as_posix(),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    with pytest.raises(ValueError, match="cached bytecode differs semantically"):
        training.capture_execution_snapshot()


def test_inactive_stale_bytecode_is_identity_tracked_but_not_executed(
    governed_runtime: GovernedRuntime,
    tmp_path: Path,
) -> None:
    replacement_source = tmp_path / "replacement.py"
    replacement_source.write_text('VALUE = "inactive bytecode"\n', encoding="utf-8")
    cache = governed_runtime.package_payload_path.parent / "__pycache__"
    cache.mkdir()
    pyc_path = cache / (f"__init__.{sys.implementation.cache_tag}.pyc")
    py_compile.compile(
        replacement_source.as_posix(),
        cfile=pyc_path.as_posix(),
        doraise=True,
    )
    raw = pyc_path.read_bytes()
    pyc_path.write_bytes(raw[:8] + (b"\x00" * 8) + raw[16:])

    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    recheck_evaluation_session(session)


def test_runtime_bytecode_identity_is_rechecked_even_when_bytes_are_restored(
    governed_runtime: GovernedRuntime,
) -> None:
    pyc_path = Path(
        py_compile.compile(
            governed_runtime.package_payload_path.as_posix(),
            doraise=True,
        )
    )
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]

    _replace_with_original_bytes(pyc_path)

    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)


def test_instrumented_bytecode_evidence_is_relocation_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def instrumented_distribution(prefix: str) -> tuple[PathDistribution, Path]:
        install_root = tmp_path / prefix / "site-packages"
        distribution, _metadata, _entry_points, package_payload = _write_distribution(
            install_root,
            name="session-runtime",
            version="1.0.0",
        )
        replacement = tmp_path / f"{prefix}-rewritten.py"
        replacement.write_text('VALUE = "instrumented"\n', encoding="utf-8")
        cache = package_payload.parent / "__pycache__"
        cache.mkdir()
        pyc_path = cache / (f"__init__.{sys.implementation.cache_tag}-pytest-9.1.1.pyc")
        py_compile.compile(
            replacement.as_posix(),
            cfile=pyc_path.as_posix(),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        )
        return distribution, install_root

    first, first_root = instrumented_distribution("first-environment")
    second, second_root = instrumented_distribution("second-environment")
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (first,),
    )
    _activate_test_runtime(
        monkeypatch,
        project_root=training.PROJECT_ROOT,
        install_root=first_root,
    )
    first_evidence = training._capture_runtime_packages()

    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (second,),
    )
    _activate_test_runtime(
        monkeypatch,
        project_root=training.PROJECT_ROOT,
        install_root=second_root,
    )

    assert training._capture_runtime_packages() == first_evidence


def test_record_hash_mutation_changes_evidence_but_launcher_hash_is_portable(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_path = governed_runtime.metadata_path.with_name("RECORD")
    distribution = PathDistribution(record_path.parent)
    install_root = record_path.parent.parent
    prefix = install_root.parent
    launcher = prefix / "bin" / "session-runtime"
    launcher.parent.mkdir()
    launcher.write_text("machine-specific launcher\n", encoding="utf-8")
    monkeypatch.setattr(training.sys, "prefix", prefix.as_posix())
    original = record_path.read_text(encoding="utf-8")
    external_row = "../bin/session-runtime,sha256=machineA,10\n"
    record_path.write_text(original + external_row, encoding="utf-8")
    first = training._capture_runtime_record(
        record_path.parent,
        distribution=distribution,
        install_root=install_root,
        context=training._RuntimeCaptureContext(),
    )
    assert first is not None

    record_path.write_text(
        original + "../bin/session-runtime,sha256=machineB,999\n",
        encoding="utf-8",
    )
    relocated = training._capture_runtime_record(
        record_path.parent,
        distribution=distribution,
        install_root=install_root,
        context=training._RuntimeCaptureContext(),
    )
    assert relocated is not None
    assert relocated.evidence == first.evidence

    record_path.write_text(
        original.replace(
            "session_runtime/__init__.py,,",
            "session_runtime/__init__.py,sha256=changed,7",
        ),
        encoding="utf-8",
    )
    mutated = training._capture_runtime_record(
        record_path.parent,
        distribution=distribution,
        install_root=install_root,
        context=training._RuntimeCaptureContext(),
    )
    assert mutated is not None
    assert mutated.evidence != first.evidence


def test_prestarted_session_rejects_record_only_mutation(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    record_path = governed_runtime.metadata_path.with_name("RECORD")
    record_path.write_text(
        record_path.read_text(encoding="utf-8").replace(
            "session_runtime/__init__.py,,",
            "session_runtime/__init__.py,sha256=changed,7",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)


def test_identity_first_recheck_does_not_rehash_runtime_payloads(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = training.capture_execution_snapshot()

    def unexpected_hash(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("stable recheck must not hash payload bytes")

    monkeypatch.setattr(training, "_capture_runtime_payload_file", unexpected_hash)

    assert training.recheck_execution_snapshot(snapshot) is snapshot


def test_external_console_launcher_record_is_portable_bookkeeping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "python-environment"
    install_root = prefix / "lib" / "python3.12" / "site-packages"
    distribution, _metadata, _entry_points, _payload = _write_distribution(
        install_root,
        name="session-runtime",
        version="1.0.0",
    )
    launcher = prefix / "bin" / "session-runtime"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("portable launcher bytes are excluded\n", encoding="utf-8")
    application = prefix / "share" / "applications" / "session.desktop"
    application.parent.mkdir(parents=True)
    application.write_text("desktop bookkeeping\n", encoding="utf-8")
    icon = prefix / "share" / "icons" / "session.png"
    icon.parent.mkdir(parents=True)
    icon.write_bytes(b"icon bookkeeping")
    monkeypatch.setattr(training.sys, "prefix", prefix.as_posix())
    inventory = training._require_distribution_file_inventory(distribution)
    before, _before_bytecode, _before_roots, _before_imports = (
        training._capture_runtime_package_payloads(
            distribution,
            install_root=install_root.resolve(strict=True),
            inventory=inventory,
        )
    )
    with_launcher, _launcher_bytecode, _launcher_roots, _launcher_imports = (
        training._capture_runtime_package_payloads(
            distribution,
            install_root=install_root.resolve(strict=True),
            inventory=inventory
            + (
                PackagePath("../../../bin/session-runtime"),
                PackagePath("../../../share/applications/session.desktop"),
                PackagePath("../../../share/icons/session.png"),
            ),
        )
    )

    assert with_launcher == before
    assert training._distribution_record_path(
        PackagePath("../../../bin/session-runtime")
    ) == ("bin/session-runtime", True)
    with pytest.raises(ValueError, match="unsafe traversal"):
        training._distribution_record_path(PackagePath("package/../../bin/tool"))
    escaped = prefix.parent / "escaped-runtime"
    escaped.write_text("must not be trusted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes the Python environment"):
        training._capture_runtime_package_payloads(
            distribution,
            install_root=install_root.resolve(strict=True),
            inventory=inventory + (PackagePath("../../../../escaped-runtime"),),
        )


def test_external_runtime_archive_is_content_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "python-environment"
    install_root = prefix / "lib" / "python3.12" / "site-packages"
    distribution, _metadata, _entry_points, _payload = _write_distribution(
        install_root,
        name="archive-runtime",
        version="1.0.0",
    )
    archive = prefix / "share" / "py4j" / "runtime.jar"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"first archive")
    monkeypatch.setattr(training.sys, "prefix", prefix.as_posix())
    inventory = training._require_distribution_file_inventory(distribution)
    inventory += (PackagePath("../../../share/py4j/runtime.jar"),)

    before, _bytecode, _roots, _imports = training._capture_runtime_package_payloads(
        distribution,
        install_root=install_root.resolve(strict=True),
        inventory=inventory,
    )
    archive.write_bytes(b"changed archive")
    after, _bytecode, _roots, _imports = training._capture_runtime_package_payloads(
        distribution,
        install_root=install_root.resolve(strict=True),
        inventory=inventory,
    )

    assert before != after


def test_editable_import_root_uses_portable_content_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def editable_distribution(prefix: str) -> tuple[PathDistribution, Path, Path]:
        source_root = tmp_path / f"{prefix}-checkout" / "src"
        package_root = source_root / "session_runtime"
        package_root.mkdir(parents=True)
        (package_root / "__init__.py").write_text(
            'VALUE = "portable editable source"\n',
            encoding="utf-8",
        )
        install_root = tmp_path / prefix / "site-packages"
        distribution_root = install_root / "session_runtime-1.0.0.dist-info"
        distribution_root.mkdir(parents=True)
        (distribution_root / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: session-runtime\nVersion: 1.0.0\n",
            encoding="utf-8",
        )
        editable_path = install_root / "_editable_impl_session_runtime.pth"
        editable_path.write_text(
            (
                f"# relocated checkout {prefix}\n\n{source_root}\n"
                if prefix.startswith("first")
                else f"\n{source_root}\n# relocated checkout {prefix}\n"
            ),
            encoding="utf-8",
        )
        (distribution_root / "RECORD").write_text(
            (
                "_editable_impl_session_runtime.pth,,\n"
                "session_runtime-1.0.0.dist-info/METADATA,,\n"
                "session_runtime-1.0.0.dist-info/RECORD,,\n"
            ),
            encoding="utf-8",
        )
        return PathDistribution(distribution_root), install_root, source_root

    first, first_install_root, first_source_root = editable_distribution(
        "first-environment"
    )
    second, second_install_root, second_source_root = editable_distribution(
        "second-environment"
    )
    _activate_test_runtime(
        monkeypatch,
        project_root=training.PROJECT_ROOT,
        install_root=first_install_root,
        exposed_roots=(first_source_root,),
    )
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (first,),
    )
    first_evidence = training._capture_runtime_packages()
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (second,),
    )
    _activate_test_runtime(
        monkeypatch,
        project_root=training.PROJECT_ROOT,
        install_root=second_install_root,
        exposed_roots=(second_source_root,),
    )

    assert training._capture_runtime_packages() == first_evidence


def test_setuptools_finder_editable_is_portable_and_binds_mapped_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def finder_distribution(
        prefix: str,
    ) -> tuple[PathDistribution, Path, Path, Path, Path]:
        checkout = tmp_path / f"{prefix}-checkout"
        package_root = checkout / "src" / "session_runtime"
        package_root.mkdir(parents=True)
        package_payload = package_root / "__init__.py"
        package_payload.write_text('VALUE = "portable"\n', encoding="utf-8")
        module_stem = checkout / "src" / "renamed"
        module_stem.mkdir()
        (module_stem / "not_imported.txt").write_text(
            "the finder selects the sibling module instead\n",
            encoding="utf-8",
        )
        module_path = module_stem.with_suffix(".py")
        module_path.write_text('RENAMED = "portable"\n', encoding="utf-8")
        namespace_root = checkout / "namespace" / "portion"
        namespace_root.mkdir(parents=True)
        (namespace_root / "feature.py").write_text(
            'FEATURE = "portable"\n',
            encoding="utf-8",
        )

        install_root = tmp_path / prefix / "site-packages"
        distribution_root = install_root / "session_runtime-1.0.0.dist-info"
        distribution_root.mkdir(parents=True)
        (distribution_root / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: session-runtime\nVersion: 1.0.0\n",
            encoding="utf-8",
        )
        pth_name = "__editable__.session_runtime-1.0.0.pth"
        finder_name = "__editable___session_runtime_1_0_0_finder"
        pth_path = install_root / pth_name
        pth_path.write_text(
            f"import {finder_name}; {finder_name}.install()\n",
            encoding="utf-8",
        )
        finder_path = install_root / f"{finder_name}.py"
        finder_path.write_text(
            training._SETUPTOOLS_FINDER_TEMPLATE.format(
                name="__editable__.session_runtime-1.0.0.finder",
                mapping={
                    "session_runtime": package_root.as_posix(),
                    "renamed": module_stem.as_posix(),
                },
                namespaces={"session_namespace": [namespace_root.as_posix()]},
            ),
            encoding="utf-8",
        )
        (distribution_root / "RECORD").write_text(
            (
                f"{pth_name},,\n"
                f"{finder_name}.py,,\n"
                "session_runtime-1.0.0.dist-info/METADATA,,\n"
                "session_runtime-1.0.0.dist-info/RECORD,,\n"
            ),
            encoding="utf-8",
        )
        return (
            PathDistribution(distribution_root),
            install_root,
            package_payload,
            module_path,
            namespace_root / "feature.py",
        )

    first, first_install_root, first_package, first_module, first_namespace = (
        finder_distribution("first-environment")
    )
    second, second_install_root, _second_package, _second_module, _second_namespace = (
        finder_distribution("second-environment")
    )
    finder_name = "__editable___session_runtime_1_0_0_finder"
    original_meta_path = list(training.sys.meta_path)
    original_path_hooks = list(training.sys.path_hooks)

    def activate(install_root: Path) -> None:
        _activate_test_runtime(
            monkeypatch,
            project_root=training.PROJECT_ROOT,
            install_root=install_root,
        )
        monkeypatch.setattr(training.sys, "meta_path", list(original_meta_path))
        monkeypatch.setattr(training.sys, "path_hooks", list(original_path_hooks))
        finder_path = install_root / f"{finder_name}.py"
        spec = importlib.util.spec_from_file_location(finder_name, finder_path)
        assert spec is not None and spec.loader is not None
        tracker = training._RUNTIME_EXTENSION_FINDER
        assert tracker is not None
        spec = tracker._tracked_spec(finder_name, spec)
        monkeypatch.setitem(training._RUNTIME_COMPLETED_LOADS, finder_name, [])
        monkeypatch.setattr(
            training,
            "_RUNTIME_COMPLETED_LOADS_OVERFLOW",
            training._RUNTIME_COMPLETED_LOADS_OVERFLOW.difference({finder_name}),
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(training.sys.modules, finder_name, module)
        spec.loader.exec_module(module)
        module.install()
        assert (
            "__editable__.session_runtime-1.0.0.finder.__path_hook__"
            in training.sys.path
        )

    activate(first_install_root)
    monkeypatch.setattr(training.importlib.metadata, "distributions", lambda: (first,))
    first_evidence = training._capture_runtime_packages()
    first_inventory = training._runtime_distribution_inventory()
    import_state = next(
        item.import_environment
        for item in first_inventory
        if item.import_environment is not None
    )
    assert any(finder_name in item for item in import_state.meta_path.portable)
    monkeypatch.setattr(
        training.sys,
        "meta_path",
        [
            item
            for item in training.sys.meta_path
            if finder_name
            not in training._portable_import_hook(
                item,
                label="test finder",
            )
        ],
    )
    with pytest.raises(RuntimeError, match="sys.meta_path import precedence changed"):
        training._require_unchanged_import_hooks(
            import_state.meta_path,
            tuple(training.sys.meta_path),
            label="sys.meta_path",
        )
    monkeypatch.setattr(training.importlib.metadata, "distributions", lambda: (second,))
    activate(second_install_root)

    assert training._capture_runtime_packages() == first_evidence

    monkeypatch.setattr(training.importlib.metadata, "distributions", lambda: (first,))
    activate(first_install_root)
    first_package.write_text('VALUE = "mutated"\n', encoding="utf-8")
    assert training._capture_runtime_packages() != first_evidence
    first_package.write_text('VALUE = "portable"\n', encoding="utf-8")
    first_module.write_text('RENAMED = "mutated"\n', encoding="utf-8")
    assert training._capture_runtime_packages() != first_evidence
    first_module.write_text('RENAMED = "portable"\n', encoding="utf-8")
    first_namespace.write_text('FEATURE = "mutated"\n', encoding="utf-8")
    assert training._capture_runtime_packages() != first_evidence

    finder_path = Path(
        first.locate_file("__editable___session_runtime_1_0_0_finder.py")
    )
    finder_path.write_text(
        "MAPPING = dict(session_runtime='/tmp/untrusted')\nNAMESPACES = {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="MAPPING is not literal"):
        training._capture_runtime_packages()

    finder_path.write_text(
        training._SETUPTOOLS_FINDER_TEMPLATE.format(
            name="__editable__.session_runtime-1.0.0.finder",
            mapping={
                "session_runtime": first_package.parent.as_posix(),
                "renamed": first_module.with_suffix("").as_posix(),
            },
            namespaces={"session_namespace": [first_namespace.parent.as_posix()]},
        )
        + "\nEXTERNAL_BEHAVIOR = __import__('os').getcwd()\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="supported locked template"):
        training._capture_runtime_packages()


def test_unknown_executable_pth_fails_closed(tmp_path: Path) -> None:
    install_root = tmp_path / "site-packages"
    distribution, _metadata, _entry_points, _payload = _write_distribution(
        install_root,
        name="session-runtime",
        version="1.0.0",
    )
    executable = install_root / "unexpected.pth"
    executable.write_text("import os; os.getcwd()\n", encoding="utf-8")
    inventory = training._require_distribution_file_inventory(distribution) + (
        PackagePath("unexpected.pth"),
    )

    with pytest.raises(ValueError, match="unknown executable"):
        training._capture_runtime_package_payloads(
            distribution,
            install_root=install_root.resolve(strict=True),
            inventory=inventory,
        )


def test_setuptools_strict_editable_tree_is_explicitly_unsupported(
    tmp_path: Path,
) -> None:
    install_root = tmp_path / "site-packages"
    distribution, _metadata, _entry_points, _payload = _write_distribution(
        install_root,
        name="session-runtime",
        version="1.0.0",
    )
    strict_tree = tmp_path / "build" / "__editable__.session_runtime-1.0.0-py3-none-any"
    strict_tree.mkdir(parents=True)
    strict_pth = install_root / "__editable__.session_runtime-1.0.0.pth"
    strict_pth.write_text(strict_tree.as_posix() + "\n", encoding="utf-8")
    inventory = training._require_distribution_file_inventory(distribution) + (
        PackagePath(strict_pth.name),
    )

    with pytest.raises(ValueError, match="strict editable symlink-tree mode"):
        training._capture_runtime_package_payloads(
            distribution,
            install_root=install_root.resolve(strict=True),
            inventory=inventory,
        )


def test_runtime_capture_rejects_lexical_ancestor_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical_root = tmp_path / "physical" / "site-packages"
    distribution, _metadata, _entry_points, _payload = _write_distribution(
        physical_root,
        name="session-runtime",
        version="1.0.0",
    )
    linked_root = tmp_path / "linked-site-packages"
    linked_root.symlink_to(physical_root, target_is_directory=True)
    linked_distribution = PathDistribution(linked_root / Path(distribution._path).name)
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (linked_distribution,),
    )

    with pytest.raises(ValueError, match="lexical symbolic-link component"):
        training._capture_runtime_packages()


def test_runtime_capture_rejects_symlink_hidden_before_parent_traversal(
    tmp_path: Path,
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(physical, target_is_directory=True)
    traversed = linked / ".." / physical.name

    with pytest.raises(ValueError, match="lexical symbolic-link component"):
        training._require_no_lexical_symlink_components(
            traversed,
            label="test path",
        )


def test_active_import_precedence_is_portable_and_order_sensitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _metadata, _entry_points, _payload = _write_distribution(
        tmp_path / "first" / "site-packages",
        name="first-runtime",
        version="1.0.0",
    )
    second, _metadata, _entry_points, _payload = _write_distribution(
        tmp_path / "second" / "site-packages",
        name="second-runtime",
        version="1.0.0",
    )
    first_root = Path(first.locate_file("")).resolve(strict=True)
    second_root = Path(second.locate_file("")).resolve(strict=True)
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (first, second),
    )
    _activate_test_runtime(
        monkeypatch,
        project_root=training.PROJECT_ROOT,
        install_root=first_root,
        exposed_roots=(second_root,),
    )
    before = training._capture_runtime_packages()

    _activate_test_runtime(
        monkeypatch,
        project_root=training.PROJECT_ROOT,
        install_root=second_root,
        exposed_roots=(first_root,),
    )
    after = training._capture_runtime_packages()

    assert tuple(
        item for item in before if item.name != "python-import-environment"
    ) == (tuple(item for item in after if item.name != "python-import-environment"))
    assert before != after


def test_active_import_roots_reject_overlapping_owned_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _metadata, _entry_points, _payload = _write_distribution(
        tmp_path / "first" / "site-packages",
        name="overlap-runtime",
        version="1.0.0",
    )
    second, _metadata, _entry_points, _payload = _write_distribution(
        tmp_path / "second" / "site-packages",
        name="overlap-runtime",
        version="2.0.0",
    )
    first_root = Path(first.locate_file("")).resolve(strict=True)
    second_root = Path(second.locate_file("")).resolve(strict=True)
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (first, second),
    )
    _activate_test_runtime(
        monkeypatch,
        project_root=training.PROJECT_ROOT,
        install_root=first_root,
        exposed_roots=(second_root,),
    )

    with pytest.raises(RuntimeError, match="overlapping origins"):
        training._capture_runtime_packages()


def test_identical_evidence_in_distinct_roots_is_still_an_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _metadata, _entry_points, _payload = _write_distribution(
        tmp_path / "first" / "site-packages",
        name="same-runtime",
        version="1.0.0",
    )
    second, _metadata, _entry_points, _payload = _write_distribution(
        tmp_path / "second" / "site-packages",
        name="same-runtime",
        version="1.0.0",
    )
    first_root = Path(first.locate_file("")).resolve(strict=True)
    second_root = Path(second.locate_file("")).resolve(strict=True)
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (first, second),
    )
    _activate_test_runtime(
        monkeypatch,
        project_root=training.PROJECT_ROOT,
        install_root=first_root,
        exposed_roots=(second_root,),
    )

    with pytest.raises(RuntimeError, match="overlapping origins"):
        training._capture_runtime_packages()


def test_active_import_search_rejects_unmatched_root(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unmatched = tmp_path / "unmatched"
    unmatched.mkdir()
    _activate_test_runtime(
        monkeypatch,
        project_root=governed_runtime.project_root,
        install_root=governed_runtime.metadata_path.parent.parent,
        exposed_roots=(unmatched,),
    )

    with pytest.raises(ValueError, match="not governed by evidence"):
        training.capture_execution_snapshot()


def test_active_import_search_rejects_arbitrary_project_subdirectory(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unmatched = governed_runtime.project_root / "helpers"
    unmatched.mkdir()
    (unmatched / "untracked.py").write_text('VALUE = "untracked"\n', encoding="utf-8")
    _activate_test_runtime(
        monkeypatch,
        project_root=governed_runtime.project_root,
        install_root=governed_runtime.metadata_path.parent.parent,
        exposed_roots=(unmatched,),
    )

    with pytest.raises(ValueError, match="not governed by evidence"):
        training.capture_execution_snapshot()


def test_active_import_search_rejects_arbitrary_base_prefix_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training.sys,
        "path",
        [Path(training.sys.base_prefix).resolve(strict=True).as_posix()],
    )

    with pytest.raises(ValueError, match="not governed by evidence"):
        training._active_import_search_state(
            training._python_stdlib_root_tokens(),
            allowed_namespace_placeholders=frozenset(),
        )


def test_active_project_entry_module_is_content_and_identity_bound(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = governed_runtime.project_root / "project_helper.py"
    helper.write_text('VALUE = "captured"\n', encoding="utf-8")
    _activate_test_runtime(
        monkeypatch,
        project_root=governed_runtime.project_root,
        install_root=governed_runtime.metadata_path.parent.parent,
        exposed_roots=(governed_runtime.project_root,),
    )
    training.sys.modules.pop("project_helper", None)
    try:
        module = importlib.import_module("project_helper")
        assert module.VALUE == "captured"
        snapshot = training.capture_execution_snapshot()
        helper.write_text('VALUE = "mutated"\n', encoding="utf-8")

        with pytest.raises(RuntimeError, match="changed during the operation"):
            training.recheck_execution_snapshot(snapshot)
    finally:
        training.sys.modules.pop("project_helper", None)


def test_loaded_unbound_module_from_governed_search_root_is_rejected(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = training.capture_execution_snapshot()
    helper = governed_runtime.project_root / "src" / "unbound_helper.py"
    helper.write_text('VALUE = "unbound"\n', encoding="utf-8")
    module = ModuleType("unbound_helper")
    module.__spec__ = SimpleNamespace(
        origin=helper.as_posix(),
        submodule_search_locations=None,
    )
    monkeypatch.setitem(training.sys.modules, "unbound_helper", module)

    with pytest.raises(RuntimeError, match="not bound to captured import evidence"):
        training._validate_loaded_module_origins(snapshot._import_environment)


def test_loaded_unbound_module_outside_every_captured_root_is_rejected(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside_helper.py"
    outside.write_text('VALUE = "outside"\n', encoding="utf-8")
    module = ModuleType("outside_helper")
    module.__spec__ = SimpleNamespace(
        origin=outside.as_posix(),
        submodule_search_locations=None,
    )
    monkeypatch.setitem(training.sys.modules, "outside_helper", module)

    with pytest.raises(RuntimeError, match="outside every captured root"):
        training.capture_execution_snapshot()


def test_loaded_originless_module_is_rejected(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("originless_helper")
    module.__spec__ = SimpleNamespace(
        origin=None,
        submodule_search_locations=None,
    )
    monkeypatch.setitem(training.sys.modules, "originless_helper", module)

    with pytest.raises(RuntimeError, match="lacks a resolvable origin"):
        training.capture_execution_snapshot()


def test_loaded_spec_less_dynamic_module_is_rejected(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("spec_less_helper")
    module.__spec__ = None
    module.VALUE = "already executed"  # type: ignore[attr-defined]
    monkeypatch.setitem(training.sys.modules, "spec_less_helper", module)

    with pytest.raises(RuntimeError, match="lacks origin metadata"):
        training.capture_execution_snapshot()


def test_protected_originless_submodule_cannot_forge_loader_provenance(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged_loader_type = type(
        "ForgedLoader",
        (),
        {"__module__": "session_runtime.loader"},
    )
    module = ModuleType("session_runtime.unbound_runtime")
    module.__spec__ = SimpleNamespace(
        origin=None,
        loader=forged_loader_type(),
        submodule_search_locations=None,
    )
    monkeypatch.setitem(
        training.sys.modules,
        "session_runtime.unbound_runtime",
        module,
    )

    with pytest.raises(RuntimeError, match="lacks a resolvable origin"):
        training.capture_execution_snapshot()


def test_capture_rejects_source_changed_after_module_execution(
    governed_runtime: GovernedRuntime,
) -> None:
    source = governed_runtime.package_payload_path
    source.write_text('VALUE = "executed"\n', encoding="utf-8")
    training.sys.modules.pop("session_runtime", None)
    try:
        module = importlib.import_module("session_runtime")
        assert module.VALUE == "executed"
        source.write_text('VALUE = "replacement"\n', encoding="utf-8")

        with pytest.raises(RuntimeError, match="does not match captured source bytes"):
            training.capture_execution_snapshot()
    finally:
        training.sys.modules.pop("session_runtime", None)


def test_capture_rejects_source_changed_before_provenance_audit(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = governed_runtime.package_payload_path
    source.write_text('VALUE = "loaded-before-audit"\n', encoding="utf-8")
    training.sys.modules.pop("session_runtime", None)
    try:
        module = importlib.import_module("session_runtime")
        assert module.VALUE == "loaded-before-audit"
        resolved = source.resolve(strict=True)
        monkeypatch.delitem(
            training._RUNTIME_EXECUTED_MODULE_CODE,
            resolved,
            raising=False,
        )
        source.write_text('VALUE = "captured-after-audit"\n', encoding="utf-8")
        monkeypatch.setitem(
            training._RUNTIME_INITIAL_MODULE_FILES,
            ("session_runtime", resolved),
            training._capture_physical_file_identity(
                resolved,
                label="synthetic pre-audit module",
            ),
        )

        with pytest.raises(RuntimeError, match="predates execution evidence"):
            training.capture_execution_snapshot()
    finally:
        training.sys.modules.pop("session_runtime", None)


def test_pre_audit_source_and_spec_less_forgery_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "preaudit_runtime.py"
    source.write_text('VALUE = "resident-a"\n', encoding="utf-8")
    script = r"""
import importlib
import py_compile
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

root = Path(sys.argv[1])
sys.path.insert(0, root.as_posix())
module = importlib.import_module("preaudit_runtime")
assert module.VALUE == "resident-a"
source = root / "preaudit_runtime.py"
source.write_text('VALUE = "captured-b"\n', encoding="utf-8")
py_compile.compile(source.as_posix(), doraise=True)
forged = ModuleType("cython_runtime")
forged.__spec__ = None
sys.modules["cython_runtime"] = forged

from aai_local_finetuning import training

path = source.resolve(strict=True)
current = training._capture_physical_file_identity(
    path,
    label="pre-audit source",
)
try:
    training._require_loaded_code_provenance(
        "preaudit_runtime",
        module,
        path,
        current,
    )
except RuntimeError as error:
    assert "predates execution evidence" in str(error)
else:
    raise AssertionError("replaceable bytecode authenticated resident code")

try:
    training._validate_spec_less_module(
        "cython_runtime",
        forged,
        state=SimpleNamespace(),
        roots=[],
        captured_by_path={},
        initial=None,
    )
except RuntimeError as error:
    assert "lacks origin metadata" in str(error)
else:
    raise AssertionError("pre-audit cython_runtime forgery was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, tmp_path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "lifecycle",
    ("remove", "replace_restore", "new_replace_restore"),
)
def test_recheck_rejects_transient_governed_module_execution(
    governed_runtime: GovernedRuntime,
    lifecycle: str,
) -> None:
    helper = governed_runtime.project_root / "src" / "transient_helper.py"
    helper.write_text('VALUE = "captured"\n', encoding="utf-8")
    training.sys.modules.pop("transient_helper", None)
    if lifecycle == "new_replace_restore":
        py_compile.compile(helper.as_posix(), doraise=True)
        baseline = None
    else:
        baseline = importlib.import_module("transient_helper")
        training.sys.modules.pop("transient_helper", None)
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    replacement = importlib.import_module("transient_helper")
    assert replacement.VALUE == "captured"
    if lifecycle == "remove":
        training.sys.modules.pop("transient_helper", None)
    elif lifecycle == "new_replace_restore":
        training.sys.modules.pop("transient_helper", None)
        baseline = importlib.import_module("transient_helper")
        training.sys.modules["transient_helper"] = replacement
    else:
        assert baseline is not None
        training.sys.modules["transient_helper"] = baseline

    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)

    training.sys.modules.pop("transient_helper", None)


@pytest.mark.parametrize("hook_name", ("meta_path", "path_hooks"))
def test_recheck_binds_import_hook_activation_and_order(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
    hook_name: str,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    current = list(getattr(training.sys, hook_name))
    assert len(current) > 1
    monkeypatch.setattr(training.sys, hook_name, list(reversed(current)))

    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)


def test_capture_requires_active_extension_tracker_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = training._RUNTIME_EXTENSION_FINDER
    assert tracker is not None
    reordered = [item for item in training.sys.meta_path if item is not tracker]
    path_finder_index = reordered.index(importlib.machinery.PathFinder)
    reordered.insert(path_finder_index + 1, tracker)
    monkeypatch.setattr(training.sys, "meta_path", reordered)

    with pytest.raises(RuntimeError, match="extension import tracker order"):
        training._install_runtime_extension_finder()


def test_real_snapshot_allows_first_mlx_native_import() -> None:
    script = r"""
import sys
from types import ModuleType
from aai_local_finetuning import training

assert "mlx.core" not in sys.modules
snapshot = training.capture_execution_snapshot()
assert sys.dont_write_bytecode is True
import mlx.core
path = training._loaded_module_origin_path(sys.modules["mlx.core"].__spec__)
assert path is not None
before = dict(snapshot._import_environment.execution_counts).get(path, 0)
after = dict(training._runtime_execution_counts()).get(path, 0)
assert after - before == 1
training.recheck_execution_snapshot(snapshot)
for name in training._RUNTIME_MLX_SPECLESS_CHILDREN:
    assert name in sys.modules
training.recheck_execution_snapshot(snapshot)
replacement = ModuleType("mlx.core.random")
replacement.__spec__ = None
replacement.unbound_callable = lambda: "not native"
mlx.core.random = replacement
sys.modules["mlx.core.random"] = replacement
try:
    training.recheck_execution_snapshot(snapshot)
except RuntimeError as error:
    assert "completed parent provenance" in str(error.__cause__)
else:
    raise AssertionError("replacement MLX child was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


def test_native_sibling_requires_a_completed_extension_load() -> None:
    script = r"""
import importlib
import importlib.util
import sys
from types import ModuleType
from aai_local_finetuning import training

snapshot = training.capture_execution_snapshot()
import PIL
assert "PIL._imaging" not in sys.modules
spec = importlib.util.find_spec("PIL._imaging")
assert spec is not None
replacement = ModuleType("PIL._imaging")
replacement.__spec__ = spec
replacement.__loader__ = spec.loader
replacement.__package__ = "PIL"
PIL._imaging = replacement
sys.modules["PIL._imaging"] = replacement
try:
    training.recheck_execution_snapshot(snapshot)
except RuntimeError as error:
    assert "matching import provenance" in str(error.__cause__)
else:
    raise AssertionError("unloaded extension sibling was accepted")

del sys.modules["PIL._imaging"]
del PIL._imaging
native = importlib.import_module("PIL._imaging")
training.recheck_execution_snapshot(snapshot)
replacement = ModuleType("PIL._imaging")
replacement.__spec__ = native.__spec__
replacement.__loader__ = native.__loader__
replacement.__package__ = "PIL"
PIL._imaging = replacement
sys.modules["PIL._imaging"] = replacement
try:
    training.recheck_execution_snapshot(snapshot)
except RuntimeError as error:
    assert "completed import provenance" in str(error.__cause__)
else:
    raise AssertionError("replacement extension module was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


def test_native_created_extension_child_binds_parent_provenance() -> None:
    script = r"""
import sys
from types import ModuleType
from aai_local_finetuning import training

snapshot = training.capture_execution_snapshot()
import charset_normalizer
import charset_normalizer.cd
import charset_normalizer.md
path = training._loaded_module_origin_path(charset_normalizer.md.__spec__)
assert path is not None
assert ("charset_normalizer.md", path) in training._RUNTIME_NATIVE_CHILD_MODULES
training.recheck_execution_snapshot(snapshot)

native = charset_normalizer.md
replacement = ModuleType("charset_normalizer.md")
replacement.__spec__ = native.__spec__
replacement.__loader__ = native.__loader__
replacement.__package__ = "charset_normalizer"
charset_normalizer.md = replacement
sys.modules["charset_normalizer.md"] = replacement
try:
    training.recheck_execution_snapshot(snapshot)
except RuntimeError as error:
    assert "native-created module identity" in str(error.__cause__)
else:
    raise AssertionError("replacement native-created module was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


def test_lazy_source_identity_and_expat_aliases_are_bound() -> None:
    script = r"""
import sys
from types import ModuleType
from aai_local_finetuning import training

snapshot = training.capture_execution_snapshot()
import PIL.ImageColor
import xml.parsers.expat
for name in ("xml.parsers.expat.errors", "xml.parsers.expat.model"):
    assert sys.modules[name] is getattr(xml.parsers.expat, name.rpartition(".")[2])
training.recheck_execution_snapshot(snapshot)

native = sys.modules["PIL.ImageColor"]
replacement = ModuleType("PIL.ImageColor")
replacement.__spec__ = native.__spec__
replacement.__loader__ = native.__loader__
replacement.__package__ = "PIL"
sys.modules["PIL.ImageColor"] = replacement
import PIL
PIL.ImageColor = replacement
try:
    training.recheck_execution_snapshot(snapshot)
except RuntimeError as error:
    assert "source module identity lacks completed import provenance" in str(
        error.__cause__
    )
else:
    raise AssertionError("replacement source module was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


def test_lazy_namespace_identity_and_lifecycle_are_bound() -> None:
    script = r"""
import sys
from types import ModuleType
from aai_local_finetuning import training

snapshot = training.capture_execution_snapshot()
assert "google" not in sys.modules
import google
training.recheck_execution_snapshot(snapshot)
captured_namespace_snapshot = training.capture_execution_snapshot()

native = google
replacement = ModuleType("google")
replacement.__spec__ = native.__spec__
replacement.__loader__ = native.__loader__
replacement.__package__ = "google"
replacement.__path__ = native.__path__
sys.modules["google"] = replacement
try:
    training.recheck_execution_snapshot(snapshot)
except RuntimeError as error:
    assert "namespace lacks completed import provenance" in str(error.__cause__)
else:
    raise AssertionError("replacement namespace module was accepted")
try:
    training.recheck_execution_snapshot(captured_namespace_snapshot)
except RuntimeError as error:
    assert "namespace module identity changed" in str(error.__cause__)
else:
    raise AssertionError("captured namespace identity was not retained")

sys.modules["google"] = native
training.recheck_execution_snapshot(snapshot)
del sys.modules["google"]
try:
    training.recheck_execution_snapshot(snapshot)
except RuntimeError as error:
    assert "loaded and removed" in str(error.__cause__)
else:
    raise AssertionError("removed lazy namespace module was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


def test_interpreter_module_inventory_and_identity_are_bound() -> None:
    script = r"""
import importlib.machinery
import sys
from types import ModuleType
from aai_local_finetuning import training

snapshot = training.capture_execution_snapshot()
assert "_statistics" not in sys.modules
assert "__hello__" not in sys.modules
import _statistics
import __hello__
training.recheck_execution_snapshot(snapshot)

for name, origin, loader in (
    ("claimed_builtin", "built-in", importlib.machinery.BuiltinImporter),
    ("claimed_frozen", "frozen", importlib.machinery.FrozenImporter),
):
    claimed = ModuleType(name)
    claimed.__spec__ = importlib.machinery.ModuleSpec(
        name,
        loader,
        origin=origin,
    )
    claimed.__loader__ = loader
    sys.modules[name] = claimed
    try:
        training.recheck_execution_snapshot(snapshot)
    except RuntimeError as error:
        assert "not in inventory" in str(error.__cause__)
    else:
        raise AssertionError(f"unlisted {origin} module was accepted")
    del sys.modules[name]

for name in ("_statistics", "__hello__"):
    native = sys.modules[name]
    replacement = ModuleType(name)
    replacement.__spec__ = native.__spec__
    replacement.__loader__ = native.__loader__
    sys.modules[name] = replacement
    try:
        training.recheck_execution_snapshot(snapshot)
    except RuntimeError as error:
        assert "interpreter module" in str(error.__cause__)
    else:
        raise AssertionError(f"replacement interpreter module was accepted: {name}")
    sys.modules[name] = native
    training.recheck_execution_snapshot(snapshot)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("drift", ("mutate", "delete"))
def test_preexisting_governed_sibling_remains_bound_after_module_removal(
    governed_runtime: GovernedRuntime,
    drift: str,
) -> None:
    helper = governed_runtime.project_root / "src" / "sibling_helper.py"
    helper.write_text('VALUE = "captured"\n', encoding="utf-8")
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    module = importlib.import_module("sibling_helper")
    assert module.VALUE == "captured"
    training.sys.modules.pop("sibling_helper", None)
    if drift == "mutate":
        helper.write_text('VALUE = "mutated"\n', encoding="utf-8")
    else:
        helper.unlink()

    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)


def test_loaded_module_origin_must_match_captured_import_root(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shadow = tmp_path / "shadow" / "session_runtime.py"
    shadow.parent.mkdir()
    shadow.write_text('VALUE = "shadow"\n', encoding="utf-8")
    module = ModuleType("session_runtime")
    module.__spec__ = SimpleNamespace(
        origin=shadow.as_posix(),
        submodule_search_locations=None,
    )
    monkeypatch.setitem(training.sys.modules, "session_runtime", module)

    with pytest.raises(RuntimeError, match="loaded module origin"):
        training.capture_execution_snapshot()


def test_runtime_capture_deduplicates_shared_namespace_tree_and_file_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_root = tmp_path / "site-packages"
    first, _metadata, _entry_points, _payload = _write_distribution(
        install_root,
        name="first-runtime",
        version="1.0.0",
    )
    second, _metadata, _entry_points, _payload = _write_distribution(
        install_root,
        name="second-runtime",
        version="1.0.0",
    )
    shared_root = install_root / "shared_namespace"
    shared_root.mkdir()
    shared_file = shared_root / "shared.py"
    shared_file.write_text('VALUE = "shared"\n', encoding="utf-8")
    for distribution in (first, second):
        record = Path(distribution._path) / "RECORD"
        record.write_text(
            "shared_namespace/shared.py,,\n" + record.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    tree_calls: list[Path] = []
    file_calls: list[Path] = []
    original_tree_capture = training._capture_runtime_tree
    original_file_capture = training._capture_runtime_payload_file

    def capture_tree(path: Path) -> object:
        tree_calls.append(path)
        return original_tree_capture(path)

    def capture_file(path: Path, display_path: str) -> object:
        file_calls.append(path)
        return original_file_capture(path, display_path)

    monkeypatch.setattr(training, "_capture_runtime_tree", capture_tree)
    monkeypatch.setattr(training, "_capture_runtime_payload_file", capture_file)
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (first, second),
    )
    _activate_test_runtime(
        monkeypatch,
        project_root=training.PROJECT_ROOT,
        install_root=install_root,
    )

    training._capture_runtime_packages()

    assert tree_calls.count(shared_root.resolve(strict=True)) == 1
    assert file_calls.count(shared_file.resolve(strict=True)) == 1


def test_runtime_payload_digest_cache_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "_RUNTIME_PAYLOAD_DIGEST_CACHE_MAX_ENTRIES", 2)
    training._RUNTIME_PAYLOAD_DIGEST_CACHE.clear()
    for index in range(3):
        path = tmp_path / f"payload-{index}.py"
        path.write_text(f"VALUE = {index}\n", encoding="utf-8")
        training._capture_runtime_payload_file(path, f"payload/{index}.py")

    assert len(training._RUNTIME_PAYLOAD_DIGEST_CACHE) == 2


def test_support_and_capstone_reports_bind_the_prestarted_session_hash(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    record, prediction = _support_case()
    capstone_records = build_records(DatasetSplit.TEST, 1)

    support_report = evaluate_predictions(
        [record],
        [prediction],
        evaluation_session=session,
        inference_config=DeterministicInferenceConfig(method="session-fixture"),
    )
    capstone_report = evaluate_capstone_predictions(
        capstone_records,
        deterministic_capstone_predictions(capstone_records),
        evaluation_session=session,
        inference_config=DeterministicInferenceConfig(method="session-fixture"),
    )

    assert (
        support_report.evaluation_execution_contract_sha256
        == session.execution_contract_sha256
    )
    assert (
        capstone_report.evaluation_execution_contract_sha256
        == session.execution_contract_sha256
    )


def test_session_preserves_distinct_vendored_versions_with_the_same_name(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = PathDistribution(governed_runtime.metadata_path.parent)
    (
        vendored,
        _vendored_metadata,
        _vendored_entry_points,
        _vendored_payload,
    ) = _write_distribution(
        governed_runtime.metadata_path.parents[2] / "provider" / "_vendor",
        name="session-runtime",
        version="0.9.0",
    )
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (primary, primary, vendored),
    )

    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]

    assert tuple(
        (package.name, package.version)
        for package in session.execution_contract.runtime_packages
        if package.name != "python-import-environment"
    ) == (
        ("session-runtime", "0.9.0"),
        ("session-runtime", "1.0.0"),
    )
    recheck_evaluation_session(session)

    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (primary,),
    )
    with pytest.raises(RuntimeError, match="changed"):
        recheck_evaluation_session(session)


@pytest.mark.parametrize("kind", ("support", "capstone"))
def test_cli_removes_prediction_and_report_after_post_persistence_session_failure(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", governed_runtime.project_root)
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    if kind == "support":
        record, prediction = _support_case()
        name = "session-support"
        output_dir = governed_runtime.project_root / "artifacts" / "evaluation"
        invoke = lambda: cli._score_predictions(  # noqa: E731
            name=name,
            records=[record],
            predictions=[prediction],
            supported_intents=[record.target.intent],
            evaluation_session=session,
            inference_config=DeterministicInferenceConfig(method=name),
        )
    else:
        records = build_records(DatasetSplit.TEST, 1)
        predictions = deterministic_capstone_predictions(records)
        name = "session-capstone"
        output_dir = governed_runtime.project_root / "artifacts" / "capstone-evaluation"
        invoke = lambda: cli._score_capstone_predictions(  # noqa: E731
            name=name,
            records=records,
            predictions=predictions,
            evaluation_session=session,
            inference_config=DeterministicInferenceConfig(method=name),
        )
    prediction_path = output_dir / f"{name}-predictions.jsonl"
    report_path = output_dir / f"{name}-report.json"
    observed_persisted_evidence = False

    def reject_after_persistence(_session: object) -> None:
        nonlocal observed_persisted_evidence
        assert prediction_path.is_file()
        assert report_path.is_file()
        observed_persisted_evidence = True
        raise RuntimeError("post-persistence evaluation-session drift")

    monkeypatch.setattr(cli, "recheck_evaluation_session", reject_after_persistence)

    with pytest.raises(RuntimeError, match="post-persistence"):
        invoke()

    assert observed_persisted_evidence
    assert not prediction_path.exists()
    assert not report_path.exists()


@pytest.mark.parametrize("kind", ("support", "capstone"))
def test_cli_revalidates_training_lineage_before_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    governed_runtime: GovernedRuntime,
    kind: str,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    assert training.PROJECT_ROOT == governed_runtime.project_root
    session = start_evaluation_session()
    invalid_snapshot = SimpleNamespace(
        manifest_sha256="a" * 64,
        manifest=SimpleNamespace(execution_contract_sha256="not-a-sha256"),
    )
    if kind == "support":
        record, prediction = _support_case()
        invoke = lambda: cli._score_predictions(  # noqa: E731
            name="validated-lineage",
            records=[record],
            predictions=[prediction],
            supported_intents=[record.target.intent],
            evaluation_session=session,
            inference_config=DeterministicInferenceConfig(method="fixture"),
            training_snapshot=invalid_snapshot,  # type: ignore[arg-type]
        )
    else:
        records = build_records(DatasetSplit.TEST, 1)
        invoke = lambda: cli._score_capstone_predictions(  # noqa: E731
            name="validated-lineage",
            records=records,
            predictions=deterministic_capstone_predictions(records),
            evaluation_session=session,
            inference_config=DeterministicInferenceConfig(method="fixture"),
            training_snapshot=invalid_snapshot,  # type: ignore[arg-type]
        )

    with pytest.raises(ValidationError):
        invoke()

    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("methods", "artifact_names"),
    (
        ("basic", ("majority", "keyword-rule", "basic")),
        ("lora", ("majority", "keyword-rule", "lora-change")),
    ),
)
def test_support_cli_invalidates_requested_evidence_before_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    methods: str,
    artifact_names: tuple[str, ...],
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    output_dir = tmp_path / "artifacts" / "evaluation"
    output_dir.mkdir(parents=True)
    stale_paths = [
        output_dir / f"{name}-{suffix}"
        for name in artifact_names
        for suffix in ("predictions.jsonl", "report.json")
    ]
    promotion_path = output_dir / "promotion.json"
    stale_paths.append(promotion_path)
    for path in stale_paths:
        path.write_text("stale\n", encoding="utf-8")

    class Settings:
        processed_dir = tmp_path / "prepared"

    def reject(_path: Path) -> None:
        assert all(not path.exists() for path in stale_paths)
        raise cli.StudyCommandError("precondition failed after invalidation")

    monkeypatch.setattr(cli, "_require_prepared_split_integrity", reject)

    with pytest.raises(cli.StudyCommandError, match="after invalidation"):
        cli._cmd_evaluate(
            Namespace(limit=1, max_tokens=32, methods=methods, track=False),
            Settings(),  # type: ignore[arg-type]
        )

    assert all(not path.exists() for path in stale_paths)


def test_baseline_rerun_invalidates_prior_promotion_before_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    promotion_path = tmp_path / "artifacts" / "evaluation" / "promotion.json"
    promotion_path.parent.mkdir(parents=True)
    promotion_path.write_text("stale\n", encoding="utf-8")

    class Settings:
        processed_dir = tmp_path / "prepared"

    def reject(_path: Path) -> None:
        assert not promotion_path.exists()
        raise cli.StudyCommandError("precondition failed after invalidation")

    monkeypatch.setattr(cli, "_require_prepared_split_integrity", reject)

    with pytest.raises(cli.StudyCommandError, match="after invalidation"):
        cli._cmd_baselines(
            Namespace(track=False),
            Settings(),  # type: ignore[arg-type]
        )

    assert not promotion_path.exists()


@pytest.mark.parametrize("method", ("basic", "hybrid", "lora"))
def test_capstone_cli_invalidates_requested_evidence_before_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    output_dir = tmp_path / "artifacts" / "capstone-evaluation"
    output_dir.mkdir(parents=True)
    artifact_name = "capstone-lora-change" if method == "lora" else method
    stale_paths = [
        output_dir / f"{artifact_name}-predictions.jsonl",
        output_dir / f"{artifact_name}-report.json",
    ]
    if method == "hybrid":
        stale_paths.append(output_dir / "hybrid-explanations.json")
    stale_paths.append(output_dir / "decision.json")
    for path in stale_paths:
        path.write_text("stale\n", encoding="utf-8")

    def reject(_settings: object) -> None:
        assert all(not path.exists() for path in stale_paths)
        raise cli.StudyCommandError("precondition failed after invalidation")

    monkeypatch.setattr(cli, "_require_current_flight_preparation", reject)

    with pytest.raises(cli.StudyCommandError, match="after invalidation"):
        cli._cmd_capstone_evaluate(
            Namespace(limit=1, max_tokens=32, methods=method),
            SimpleNamespace(),  # type: ignore[arg-type]
        )

    assert all(not path.exists() for path in stale_paths)


def _support_report_stub() -> SimpleNamespace:
    return SimpleNamespace(
        classification=SimpleNamespace(macro_f1=1.0),
        output_quality=SimpleNamespace(
            json_schema_validity_rate=1.0,
            response_policy_compliance_rate=1.0,
        ),
    )


def _capstone_report_stub() -> SimpleNamespace:
    return SimpleNamespace(
        aggregate=SimpleNamespace(
            exact_review_rate=1.0,
            status_accuracy=1.0,
            check_result_accuracy=1.0,
            schema_validity_rate=1.0,
        )
    )


def test_support_cli_starts_fresh_session_before_each_baseline_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, prediction = _support_case()
    events: list[tuple[str, object]] = []
    current_session: object | None = None

    class Settings:
        processed_dir = tmp_path / "prepared"

    class Method:
        def predict_many(self, _records: object) -> tuple[Prediction, ...]:
            assert current_session is not None
            events.append(("predict", current_session))
            return (prediction,)

    class Baseline:
        @staticmethod
        def fit(_records: object) -> Method:
            return Method()

    def start() -> object:
        nonlocal current_session
        current_session = object()
        events.append(("start", current_session))
        return current_session

    def score(**kwargs: object) -> SimpleNamespace:
        assert kwargs["evaluation_session"] is current_session
        assert current_session is not None
        events.append(("score", current_session))
        return _support_report_stub()

    monkeypatch.setattr(cli, "_load_splits", lambda _s: ([record], [], [record]))
    monkeypatch.setattr(cli, "MajorityBaseline", Baseline)
    monkeypatch.setattr(cli, "KeywordRuleBaseline", Baseline)
    monkeypatch.setattr(cli, "start_evaluation_session", start)
    monkeypatch.setattr(cli, "_score_predictions", score)

    cli._baseline_reports(Settings(), track=False)  # type: ignore[arg-type]

    assert [event for event, _session in events] == [
        "start",
        "predict",
        "score",
        "start",
        "predict",
        "score",
    ]
    assert events[0][1] is events[1][1] is events[2][1]
    assert events[3][1] is events[4][1] is events[5][1]
    assert events[0][1] is not events[3][1]


def test_support_cli_starts_session_before_model_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, prediction = _support_case()
    events: list[str] = []
    marker = object()
    inference_config = object()

    class Settings:
        processed_dir = tmp_path / "prepared"
        adapter_dir = tmp_path / "adapter"
        model_dir = tmp_path / "model"

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_require_prepared_split_integrity", lambda _p: None)
    monkeypatch.setattr(cli, "_require_current_flight_preparation", lambda _s: None)
    monkeypatch.setattr(cli, "require_assets", lambda _s: [])
    monkeypatch.setattr(cli, "_load_splits", lambda _s: ([record], [], [record]))
    monkeypatch.setattr(cli, "_baseline_reports", lambda *_a, **_kw: {})
    monkeypatch.setattr(cli, "LocalMLXPredictor", lambda _path: object())

    def start(*_args: object) -> object:
        events.append("start")
        return marker

    def predict(*_args: object, **_kwargs: object) -> tuple[Prediction, ...]:
        assert events == ["start"]
        assert _kwargs["inference_config"] is inference_config
        events.append("predict")
        return (prediction,)

    def score(**kwargs: object) -> SimpleNamespace:
        assert kwargs["evaluation_session"] is marker
        assert kwargs["inference_config"] is inference_config
        events.append("score")
        return _support_report_stub()

    def build_config(session: object, **kwargs: object) -> object:
        assert session is marker
        assert kwargs["max_tokens"] == 32
        assert kwargs["method"] == "basic"
        assert kwargs["prompt_recipe"] == "basic"
        return inference_config

    monkeypatch.setattr(cli, "start_evaluation_session", start)
    monkeypatch.setattr(cli, "recheck_evaluation_session", lambda session: session)
    monkeypatch.setattr(
        cli,
        "build_local_mlx_inference_config",
        build_config,
    )
    monkeypatch.setattr(cli, "_model_predictions", predict)
    monkeypatch.setattr(cli, "_score_predictions", score)

    cli._cmd_evaluate(
        Namespace(limit=1, max_tokens=32, methods="basic", track=False),
        Settings(),  # type: ignore[arg-type]
    )

    assert events == ["start", "predict", "score"]


def test_capstone_cli_starts_session_before_model_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = build_records(DatasetSplit.TRAIN, 1)
    records = build_records(DatasetSplit.TEST, 1)
    predictions = deterministic_capstone_predictions(records)
    events: list[str] = []
    inference_config = object()
    marker = SimpleNamespace(
        execution_contract_sha256="a" * 64,
        base_model_execution_contract=None,
    )

    class Settings:
        capstone_adapter_dir = tmp_path / "adapter"
        model_dir = tmp_path / "model"

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_require_current_flight_preparation", lambda _s: None)
    monkeypatch.setattr(cli, "require_assets", lambda _s: [])
    monkeypatch.setattr(cli, "_generate_capstone", lambda: None)
    monkeypatch.setattr(
        cli,
        "load_capstone_records",
        lambda path: train if path.name == "train.jsonl" else records,
    )
    monkeypatch.setattr(cli, "LocalMLXPredictor", lambda _path: object())
    monkeypatch.setattr(cli, "_write_capstone_decision", lambda *_a, **_kw: None)

    def start(*_args: object) -> object:
        events.append("start")
        return marker

    def predict(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        assert events == ["start"]
        assert _kwargs["inference_config"] is inference_config
        events.append("predict")
        return predictions

    def score(**kwargs: object) -> SimpleNamespace:
        assert kwargs["evaluation_session"] is marker
        assert kwargs["inference_config"] is inference_config
        events.append("score")
        report = _capstone_report_stub()
        report.inference_config = DeterministicInferenceConfig(method="basic")
        report.evaluation_execution_contract_sha256 = "a" * 64
        return report

    def build_config(session: object, **kwargs: object) -> object:
        assert session is marker
        assert kwargs["max_tokens"] == 32
        assert kwargs["method"] == "basic"
        assert kwargs["prompt_recipe"] == "basic"
        return inference_config

    monkeypatch.setattr(cli, "start_evaluation_session", start)
    monkeypatch.setattr(cli, "recheck_evaluation_session", lambda session: session)
    monkeypatch.setattr(
        cli,
        "build_local_mlx_inference_config",
        build_config,
    )
    monkeypatch.setattr(cli, "_capstone_model_predictions", predict)
    monkeypatch.setattr(cli, "_score_capstone_predictions", score)

    cli._cmd_capstone_evaluate(
        Namespace(limit=1, max_tokens=32, methods="basic"),
        Settings(),  # type: ignore[arg-type]
    )

    assert events == ["start", "predict", "score", "start"]
