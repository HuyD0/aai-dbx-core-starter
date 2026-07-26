"""Job entry point: run the tracked experiment against the workspace."""

from pathlib import Path

from aai_core import bootstrap
from app.experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    context = bootstrap(ROOT / "aai-platform.yml")
    metrics = run_experiment(context)
    print({"application": context.tags.application, "metrics": metrics})


if __name__ == "__main__":
    main()
