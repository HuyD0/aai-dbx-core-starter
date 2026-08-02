"""Fast checks for the no-network study contract."""

from __future__ import annotations

import json
import os
import socket

import pytest

from aai_local_finetuning.data import DatasetIntegrityError
from aai_local_finetuning.modeling import build_messages
from aai_local_finetuning.offline import (
    OFFLINE_ENVIRONMENT,
    OfflineAssetError,
    deny_network,
    enable_offline_environment,
    prepared_dataset_check,
    prove_socket_denial,
)
from aai_local_finetuning.settings import PROJECT_ROOT, load_settings, sha256_file


def test_project_pins_reviewed_dataset_and_model_revision():
    settings = load_settings()

    assert settings.dataset.version == 1
    assert settings.dataset.license_spdx == "CDLA-Sharing-1.0"
    assert settings.dataset.per_intent.train == 40
    assert settings.dataset.per_intent.validation == 10
    assert settings.dataset.per_intent.test == 10
    assert settings.model.repo == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    assert len(settings.model.revision) == 40
    assert settings.model.directory.startswith("models/")


def test_offline_environment_removes_proxies(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")

    enable_offline_environment()

    assert "HTTPS_PROXY" not in os.environ
    assert os.environ["NO_PROXY"] == "*"
    for key, value in OFFLINE_ENVIRONMENT.items():
        assert os.environ[key] == value


def test_network_guard_denies_socket_connections():
    with deny_network():
        prove_socket_denial()
        with pytest.raises(OfflineAssetError, match="network access is blocked"):
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)


def test_hash_helper_streams_known_file(tmp_path):
    path = tmp_path / "asset.txt"
    path.write_bytes(b"offline\n")

    assert (
        sha256_file(path)
        == "c826edf35c08656aec51d506a1230b2952165170aa1eaac4aa0ff9b71425fdff"
    )


def test_readiness_reports_prepared_dataset_hash_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    def reject_manifest(_path):
        raise DatasetIntegrityError("test: SHA-256 mismatch")

    monkeypatch.setattr(
        "aai_local_finetuning.offline.require_valid_manifest",
        reject_manifest,
    )

    check = prepared_dataset_check(tmp_path)

    assert check.name == "processed dataset integrity"
    assert not check.ready
    assert "test: SHA-256 mismatch" in check.detail


def test_prompt_levels_hold_output_contract_constant():
    allowed = ["recover_password", "track_order"]
    shot = (
        "I forgot my password",
        {
            "intent": "recover_password",
            "category": "account",
            "requires_escalation": False,
            "response": "Use the account recovery flow.",
        },
    )

    basic = build_messages(
        "Where is my order?", strategy="basic", allowed_intents=allowed
    )
    strong = build_messages(
        "Where is my order?", strategy="strong", allowed_intents=allowed
    )
    few_shot = build_messages(
        "Where is my order?",
        strategy="few_shot",
        allowed_intents=allowed,
        few_shot=[shot],
    )

    assert basic[-1] == strong[-1] == few_shot[-1]
    assert len(basic) == 2
    assert len(strong) == 2
    assert len(few_shot) == 4
    assert "Allowed intents" in strong[0]["content"]
    assert json.loads(few_shot[2]["content"])["intent"] == "recover_password"


def test_sensitive_assets_are_ignored():
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    for entry in (
        "data/raw/*",
        "models/",
        "artifacts/",
        "kaggle.json",
        "access_token",
        ".env",
    ):
        assert entry in ignore


def test_no_dataset_or_model_asset_is_tracked():
    tracked_paths = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "data" / "raw" / "bitext",
    ]

    assert all(not path.exists() or path.is_dir() for path in tracked_paths)
    assert not list((PROJECT_ROOT / "data" / "raw").glob("*.zip"))


def test_nested_lock_exists_and_is_nonempty():
    lock = PROJECT_ROOT / "uv.lock"

    assert lock.stat().st_size > 10_000
    text = lock.read_text(encoding="utf-8")
    assert 'name = "mlx-lm"' in text
    assert 'version = "0.31.3"' in text
    assert 'name = "mlflow"' in text


def test_notebook_launcher_pins_the_nested_kernel_and_start_page():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "notebook-kernel:" in makefile
    assert "$(PYTHON) -m ipykernel install" in makefile
    assert '--name "$(KERNEL_NAME)"' in makefile
    assert 'PATH="$(CURDIR)/.venv/bin:$$PATH"' in makefile
    assert "notebooks/00_start_here.ipynb" in makefile
    assert "notebook-check:" in makefile
    assert "jupyter nbconvert --execute" in makefile
