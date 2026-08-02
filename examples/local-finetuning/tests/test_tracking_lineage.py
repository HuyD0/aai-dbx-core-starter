"""Local MLflow lineage tests without starting a real tracking server."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

from aai_local_finetuning import tracking


def test_change_tracking_logs_only_snapshot_bound_reloadable_adapter_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter_dir = tmp_path / "artifacts" / "adapters" / "support"
    adapter_dir.mkdir(parents=True)
    for name, content in (
        ("adapters.safetensors", b"weights"),
        ("adapter_config.json", b'{"rank":8}\n'),
        ("training-manifest.json", b'{"schema_version":"2.0.0"}\n'),
    ):
        (adapter_dir / name).write_bytes(content)
    training_config = tmp_path / "configs" / "training" / "lora.yaml"
    training_config.parent.mkdir(parents=True)
    training_config.write_text("iters: 1\n", encoding="utf-8")
    dataset_manifest = tmp_path / "data" / "manifest.json"
    dataset_manifest.parent.mkdir(parents=True)
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("{}\n", encoding="utf-8")
    unbound_evidence = tmp_path / "artifacts" / "training" / "latest.json"
    unbound_evidence.parent.mkdir(parents=True)
    unbound_evidence.write_text('{"training_manifest_sha256":"stale"}\n')

    digest = "a" * 64
    snapshot = SimpleNamespace(
        adapter_path=adapter_dir,
        config_path=training_config,
        manifest_sha256=digest,
        manifest=SimpleNamespace(
            adapter_sha256="b" * 64,
            adapter_config_sha256="c" * 64,
            source_config_sha256="d" * 64,
            effective_config_sha256="e" * 64,
        ),
    )
    settings = SimpleNamespace(
        model=SimpleNamespace(repo="local/model", revision="f" * 40),
    )

    artifacts: list[tuple[str, str | None]] = []
    parameters: dict[str, object] = {}
    fake_mlflow = ModuleType("mlflow")
    fake_mlflow.data = SimpleNamespace(  # type: ignore[attr-defined]
        from_pandas=lambda *_args, **_kwargs: object()
    )
    fake_mlflow.set_tags = lambda _tags: None  # type: ignore[attr-defined]
    fake_mlflow.log_params = parameters.update  # type: ignore[attr-defined]
    fake_mlflow.log_input = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    fake_mlflow.log_metrics = lambda _metrics: None  # type: ignore[attr-defined]
    fake_mlflow.log_artifact = (  # type: ignore[attr-defined]
        lambda path, artifact_path=None: artifacts.append(
            (Path(path).name, artifact_path)
        )
    )

    class RunContext:
        def __enter__(self):
            return SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

        def __exit__(self, *_args: object) -> None:
            return None

    fake_mlflow.start_run = lambda **_kwargs: RunContext()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(tracking, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tracking, "configure_local_mlflow", lambda _settings: None)
    monkeypatch.setattr(tracking, "recheck_training_snapshot", lambda _snapshot: None)
    monkeypatch.setattr(
        tracking,
        "shared_adapter_lock",
        lambda _adapter: nullcontext(),
    )

    run_id = tracking.log_evaluation(
        settings,  # type: ignore[arg-type]
        run_name="lora-change",
        role="change",
        method="lora-change",
        metrics={"macro_f1": 0.5},
        report={"training_manifest_sha256": digest},
        records=({"example_id": "one"},),
        manifest_path=dataset_manifest,
        prediction_path=predictions,
        model_based=True,
        training_snapshot=snapshot,  # type: ignore[arg-type]
    )

    assert run_id == "run-1"
    assert {
        ("adapters.safetensors", "change"),
        ("adapter_config.json", "change"),
        ("training-manifest.json", "change"),
        ("lora.yaml", "change"),
    } <= set(artifacts)
    assert parameters["training_manifest_sha256"] == digest
    assert parameters["adapter_config_sha256"] == "c" * 64
    assert not any(name == "latest.json" for name, _ in artifacts)
