"""Small command-line entry points backing the notebook course."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from aai_local_classification.data import prepare_dataset
from aai_local_classification.settings import load_settings
from aai_local_classification.tracking import local_paths
from aai_local_classification.workflow import run_full_workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = load_settings()
    if args.command == "prepare":
        manifest = prepare_dataset(settings, local_paths().data_root)
        result = manifest.model_dump(mode="json")
    else:
        result = run_full_workflow(settings)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
