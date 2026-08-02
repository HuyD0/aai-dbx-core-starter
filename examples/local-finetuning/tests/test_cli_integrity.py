"""Command-entrypoint tests for prepared dataset integrity gates."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from aai_local_finetuning import cli


@pytest.mark.parametrize(
    ("command", "arguments"),
    (
        (cli._cmd_baselines, Namespace(track=False)),
        (cli._cmd_train, Namespace(iterations=1)),
        (
            cli._cmd_evaluate,
            Namespace(
                limit=None,
                max_tokens=32,
                methods="all",
                track=False,
            ),
        ),
    ),
)
def test_data_consumers_stop_before_work_when_split_integrity_fails(
    monkeypatch: pytest.MonkeyPatch,
    command: object,
    arguments: Namespace,
) -> None:
    class Settings:
        processed_dir = Path("prepared")

    def reject_integrity(_processed_dir: Path) -> None:
        raise cli.StudyCommandError("prepared dataset integrity check failed")

    def unexpected_work(*_args: object, **_kwargs: object) -> None:
        pytest.fail("command performed work before verifying prepared splits")

    monkeypatch.setattr(cli, "_require_prepared_split_integrity", reject_integrity)
    monkeypatch.setattr(cli, "require_assets", unexpected_work)
    monkeypatch.setattr(cli, "run_lora", unexpected_work)
    monkeypatch.setattr(cli, "_load_splits", unexpected_work)
    monkeypatch.setattr(cli, "_baseline_reports", unexpected_work)

    with pytest.raises(cli.StudyCommandError, match="dataset integrity check failed"):
        command(arguments, Settings())  # type: ignore[operator]
