from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from aai_core import __version__, bootstrap
from aai_core.deployment import ApplicationRelease

ROOT = Path(__file__).resolve().parents[1]


def source_commit() -> str:
    configured = os.environ.get("GIT_COMMIT")
    if configured:
        return configured
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", required=True, type=int)
    parser.add_argument("--evaluation-run", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output", default="release.json")
    arguments = parser.parse_args()

    context = bootstrap(ROOT / "aai-platform.yml")
    release = ApplicationRelease(
        application=context.tags.application,
        release=context.tags.release,
        source_commit=source_commit(),
        core_sdk_version=__version__,
        model={
            "logical_name": "general-chat",
            **context.settings.models["general-chat"],
        },
        prompt={
            "name": "agent-system",
            "version": arguments.prompt_version,
        },
        retrieval={
            "logical_name": "product-knowledge",
            **context.settings.retrievers["product-knowledge"],
            "embedding": context.settings.embeddings["knowledge-embedding"],
        },
        evaluation={
            "dataset": "release-suite",
            "dataset_version": arguments.dataset_version,
            "run_id": arguments.evaluation_run,
        },
        environment=context.tags.environment,
    )
    release.write(ROOT / arguments.output)
    print({"release": release.release, "digest": release.digest})


if __name__ == "__main__":
    main()
