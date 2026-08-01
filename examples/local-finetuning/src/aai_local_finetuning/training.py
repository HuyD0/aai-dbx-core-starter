"""Safe wrapper around the pinned MLX-LM LoRA command."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from .settings import PROJECT_ROOT


class TrainingEvidence(BaseModel):
    """Small persisted summary parsed from an MLX-LM training run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    command: list[str]
    return_code: int
    iterations: int
    train_losses: list[float]
    validation_losses: list[float]
    peak_memory_gb: float | None
    log_path: str


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
) -> TrainingEvidence:
    """Run local LoRA training, retaining a readable log and bounded evidence."""

    require_apple_silicon()
    config = config_path or PROJECT_ROOT / "configs" / "training" / "lora.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"training configuration is missing: {config}")
    configured = yaml.safe_load(config.read_text(encoding="utf-8"))
    configured_iterations = configured.get("iters")
    if not isinstance(configured_iterations, int) or configured_iterations < 1:
        raise ValueError("training configuration iters must be a positive integer")
    command = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config)]
    if iterations is not None:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        command.extend(["--iters", str(iterations)])
    if adapter_path is not None:
        command.extend(
            [
                "--adapter-path",
                str(adapter_path),
                "--save-every",
                str(iterations or configured_iterations),
            ]
        )
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", log_name):
        raise ValueError("log_name must contain lowercase letters, numbers, _ or -")
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_dir = PROJECT_ROOT / "artifacts" / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}.log"
    log_path.write_text(result.stdout, encoding="utf-8")

    train_losses = [
        float(value)
        for value in re.findall(r"Train loss ([0-9]+(?:\.[0-9]+)?)", result.stdout)
    ]
    validation_losses = [
        float(value)
        for value in re.findall(r"Val loss ([0-9]+(?:\.[0-9]+)?)", result.stdout)
    ]
    peaks = [
        float(value)
        for value in re.findall(r"Peak mem ([0-9]+(?:\.[0-9]+)?) GB", result.stdout)
    ]
    evidence = TrainingEvidence(
        command=["<python>", *command[1:]],
        return_code=result.returncode,
        iterations=iterations or configured_iterations,
        train_losses=train_losses,
        validation_losses=validation_losses,
        peak_memory_gb=max(peaks) if peaks else None,
        log_path=str(log_path.relative_to(PROJECT_ROOT)),
    )
    evidence_path = log_dir / f"{log_name}.json"
    evidence_path.write_text(
        json.dumps(evidence.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(f"MLX-LM training failed; see {log_path}")
    return evidence
