"""Test-harness normalization for strict runtime-evidence checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from aai_local_finetuning import training


@pytest.fixture(autouse=True)
def remove_pytest_only_import_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pytest's collection-only tests path out of governed runtime checks."""

    tests_root = Path(__file__).resolve(strict=True).parent

    def resolved(entry: str) -> Path:
        candidate = Path(entry or os.getcwd())
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate.resolve(strict=False)

    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if resolved(entry) != tests_root],
    )
    loaded_modules = training._runtime_loaded_modules

    def without_test_modules() -> tuple[tuple[str, object], ...]:
        selected: list[tuple[str, object]] = []
        for name, module in loaded_modules():
            spec = getattr(module, "__spec__", None)
            loader = getattr(spec, "loader", None)
            if type(loader).__module__.startswith("_pytest."):
                continue
            origin = getattr(spec, "origin", None)
            if not isinstance(origin, str) or origin in {"built-in", "frozen"}:
                selected.append((name, module))
                continue
            if not resolved(origin).is_relative_to(tests_root):
                selected.append((name, module))
        return tuple(selected)

    monkeypatch.setattr(training, "_runtime_loaded_modules", without_test_modules)
