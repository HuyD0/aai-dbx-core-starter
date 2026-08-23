"""Fail-closed dependency-audit wrapper contracts."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "templates" / "_shared" / "files" / "scripts" / "audit_dependencies.py"
POLICY = ROOT / "templates" / "_shared" / "files" / "security-audit.toml"

_spec = importlib.util.spec_from_file_location("audit_dependencies", SCRIPT)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def test_expired_exception_fails_closed(tmp_path):
    policy = tmp_path / "security-audit.toml"
    policy.write_text(
        """schema_version = 1
[[exceptions]]
id = "TEST-1"
expires = "2000-01-01"
reason = "test only"
forbidden_symbols = []
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="expired"):
        audit._active_exceptions(policy, tmp_path)


def test_exception_fails_when_affected_symbol_is_used(tmp_path):
    policy = tmp_path / "security-audit.toml"
    policy.write_text(
        """schema_version = 1
[[exceptions]]
id = "TEST-1"
expires = "2099-01-01"
reason = "test only"
forbidden_symbols = ["affected_api"]
""",
        encoding="utf-8",
    )
    (tmp_path / "application.py").write_text("affected_api()\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not applicable"):
        audit._active_exceptions(policy, tmp_path)


@pytest.mark.parametrize("mode", ["uv-project", "installed-python"])
def test_resolved_graph_modes_feed_strict_exact_audit(monkeypatch, tmp_path, mode):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        command = [str(item) for item in command]
        calls.append(command)
        if command[:3] == ["uv", "export", "--locked"]:
            Path(command[command.index("--output-file") + 1]).write_text(
                "example==1.0\n", encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0)
        if command[:3] == ["uv", "pip", "freeze"]:
            return subprocess.CompletedProcess(command, 0, stdout="example==1.0\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audit, "_active_exceptions", lambda *_: [])
    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    option = "--uv-project" if mode == "uv-project" else "--installed-python"
    value = tmp_path if mode == "uv-project" else tmp_path / "python"

    assert audit.main(["--policy", str(POLICY), option, str(value)]) == 0
    pip_audit = calls[-1]
    assert "--strict" in pip_audit
    assert "--no-deps" in pip_audit
    assert "--disable-pip" in pip_audit
    assert "--requirement" in pip_audit
