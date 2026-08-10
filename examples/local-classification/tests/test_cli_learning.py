"""Behavioral coverage for the learner-facing CLI and state helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aai_local_classification import cli, learning


@pytest.mark.parametrize("command", ["prepare", "run"])
def test_cli_emits_stable_json(monkeypatch, capsys, tmp_path, command):
    settings = object()
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(
        cli, "local_paths", lambda: SimpleNamespace(data_root=tmp_path / "data")
    )
    monkeypatch.setattr(
        cli,
        "prepare_dataset",
        lambda actual, root: SimpleNamespace(
            model_dump=lambda **_: {"command": "prepare", "root": root.name}
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_full_workflow",
        lambda actual: {"command": "run"},
    )

    assert cli.main([command]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == command
    if command == "prepare":
        assert output["root"] == "data"


def test_learning_state_helpers_use_isolated_root(monkeypatch, tmp_path):
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / "selection.json").write_text('{"decision":"adopt"}', encoding="utf-8")
    monkeypatch.setenv("AAI_CLASSIFICATION_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        learning,
        "local_paths",
        lambda root: SimpleNamespace(state_root=state_root, root=root),
    )

    assert learning.study_root() == tmp_path.resolve()
    assert learning.state_exists("selection.json")
    assert not learning.state_exists("missing.json")
    assert learning.read_state("selection.json") == {"decision": "adopt"}
    assert learning.short_digest("0123456789abcdef", width=8) == "01234567…"
