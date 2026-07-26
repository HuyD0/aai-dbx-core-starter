"""Release gate: run the experiment and apply deterministic thresholds.

No LLM judges here — this template's gate is plain metrics. It still runs on
the credentialed path (the experiment writes to the workspace tracking
server); pull-request CI runs only evals/offline_checks.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aai_core import bootstrap
from aai_core.evaluation import QualityThreshold, apply_thresholds
from app.experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]
GATE_CONFIG = ROOT / "evals" / "gate_config.json"
BASELINE = ROOT / "evals" / "baseline.json"


def load_thresholds() -> list[QualityThreshold]:
    config = json.loads(GATE_CONFIG.read_text(encoding="utf-8"))
    return [QualityThreshold(**threshold) for threshold in config["thresholds"]]


def load_baseline() -> dict[str, float]:
    if not BASELINE.exists():
        print(
            "No evals/baseline.json yet; regression checks activate after the "
            "first release records one with --update-baseline."
        )
        return {}
    metrics = json.loads(BASELINE.read_text(encoding="utf-8"))["metrics"]
    return {name: float(value) for name, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    context = bootstrap(ROOT / "aai-platform.yml")
    metrics = run_experiment(context)
    report = apply_thresholds(
        metrics, load_thresholds(), baseline_metrics=load_baseline()
    )
    report.require_passed()
    if args.update_baseline:
        BASELINE.write_text(
            json.dumps({"metrics": dict(report.metrics)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(
        {
            "application": context.tags.application,
            "metrics": report.metrics,
            "baseline_updated": args.update_baseline,
        }
    )


if __name__ == "__main__":
    main()
