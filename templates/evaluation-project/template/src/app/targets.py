"""Adapters that turn the thing under evaluation into a predict_fn."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from aai_core import PlatformContext


def answer_sheet_predict_fn(path: str | Path) -> Callable[[str], str]:
    """Replay recorded answers (question -> answer) — deterministic, offline.
    Used by tier-1 checks and for scoring an application you cannot call."""

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    answers = {record["question"]: record["answer"] for record in records}

    def predict(question: str) -> str:
        return answers.get(question, "")

    return predict


def endpoint_predict_fn(
    context: PlatformContext, logical_name: str = "target-model"
) -> Callable[[str], str]:
    """Call the application/endpoint under evaluation via its logical name
    (configure providers.models.target-model in aai-platform.yml)."""

    model = context.providers.model(logical_name)

    def predict(question: str) -> str:
        response = model.generate([{"role": "user", "content": question}])
        return response.content

    return predict
