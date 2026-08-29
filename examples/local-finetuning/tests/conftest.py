"""Test-harness normalization for strict runtime-evidence checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from aai_local_finetuning import training

_COLLECTION_MODULES: dict[str, object] = {}
_TESTS_ROOT = Path(__file__).resolve(strict=True).parent
_PROJECT_ROOT = _TESTS_ROOT.parent
_INTERPRETER_PREFIXES = tuple(
    Path(prefix).resolve(strict=False) for prefix in (sys.prefix, sys.base_prefix)
)


def _embedding_repository_root() -> Path | None:
    """Locate the repository checkout this standalone project is nested in."""

    for ancestor in _PROJECT_ROOT.parents:
        if (ancestor / ".git").exists():
            return ancestor
    return None


_EMBEDDING_REPOSITORY_ROOT = _embedding_repository_root()


def _pth_configured_roots() -> frozenset[Path]:
    """Collect import roots that installed ``.pth`` files add at startup.

    Editable installs bind these roots as runtime path configuration; the
    strict execution snapshot requires them to remain on ``sys.path``, so the
    collection-only pruning below must never remove them.
    """

    import site

    site_dirs = {Path(entry) for entry in site.getsitepackages()}
    user_site = site.getusersitepackages()
    if user_site:
        site_dirs.add(Path(user_site))
    roots: set[Path] = set()
    for site_dir in site_dirs:
        if not site_dir.is_dir():
            continue
        for pth_file in site_dir.glob("*.pth"):
            try:
                lines = pth_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                value = line.strip()
                if not value or value.startswith(("#", "import ", "import\t")):
                    continue
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = site_dir / candidate
                roots.add(candidate.resolve(strict=False))
    return frozenset(roots)


_PTH_CONFIGURED_ROOTS = _pth_configured_roots()


def _resolved_path(entry: str) -> Path:
    candidate = Path(entry or os.getcwd())
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve(strict=False)


def _collection_only_import_root(path: Path) -> bool:
    """Identify sys.path entries that exist only because pytest was invoked.

    ``python -m pytest`` prepends the invocation directory, collecting a test
    module inserts that module's directory, and the embedding repository's
    pytest configuration prepends its own ``pythonpath`` entries. None of that
    is governed runtime evidence, and every import those entries serve
    completes during collection, so the strict execution-snapshot checks must
    not observe them. This covers this project's own tests directory, the
    project root and its ancestors, any other collected tests directory, and
    the embedding repository's source roots — while keeping the interpreter
    environment and everything else inside this project untouched.
    """

    if path in (_TESTS_ROOT, _PROJECT_ROOT):
        return True
    if path in _PROJECT_ROOT.parents:
        return True
    if path.name == "tests" and (path / "conftest.py").is_file():
        return True
    if (
        _EMBEDDING_REPOSITORY_ROOT is not None
        and _EMBEDDING_REPOSITORY_ROOT in path.parents
        and _PROJECT_ROOT not in path.parents
        and path not in _PTH_CONFIGURED_ROOTS
        and not any(
            path == prefix or prefix in path.parents for prefix in _INTERPRETER_PREFIXES
        )
    ):
        return True
    return False


def pytest_collection_finish() -> None:
    """Remember collection-only imports and drop collection-only import roots."""

    _COLLECTION_MODULES.clear()
    _COLLECTION_MODULES.update(training._runtime_loaded_modules())
    sys.path[:] = [
        entry
        for entry in sys.path
        if not _collection_only_import_root(_resolved_path(entry))
    ]


@pytest.fixture(autouse=True)
def remove_pytest_only_import_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pytest's collection-only import roots out of governed runtime checks."""

    tests_root = _TESTS_ROOT

    def resolved(entry: str) -> Path:
        return _resolved_path(entry)

    monkeypatch.setattr(
        sys,
        "path",
        [
            entry
            for entry in sys.path
            if not _collection_only_import_root(resolved(entry))
        ],
    )
    loaded_modules = training._runtime_loaded_modules

    def without_test_modules() -> tuple[tuple[str, object], ...]:
        selected: list[tuple[str, object]] = []
        for name, module in loaded_modules():
            if (
                training._runtime_audit.was_preexisting(name, module)
                or _COLLECTION_MODULES.get(name) is module
            ) and name != "_virtualenv":
                continue
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
