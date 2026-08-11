"""Shared path bootstrap for the root-executable workshop lessons."""

from __future__ import annotations

import sys
from pathlib import Path


def run(slug: str) -> None:
    """Load the accelerator and SDK source trees, then run one lesson."""

    accelerator_root = Path(__file__).resolve().parents[1]
    repository_root = accelerator_root.parents[1]
    # The accelerator is intentionally not part of the published aai-core wheel.
    # These two local source roots make each lesson runnable from a fresh checkout.
    for source_root in (
        repository_root / "src",
        accelerator_root / "src",
    ):
        source = str(source_root)
        if source not in sys.path:
            sys.path.insert(0, source)

    from email_support_agent.workshop import emit_lesson

    emit_lesson(slug)
