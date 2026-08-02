"""Bounded token-length analysis using the exact prepared local tokenizer."""

from __future__ import annotations

import math
import statistics
from pathlib import Path

from .bitext import load_bitext
from .processing import canonicalize_bitext
from .schemas import LengthSummary


def summarize_instruction_tokens(
    input_path: str | Path,
    tokenizer_path: str | Path,
) -> LengthSummary:
    """Summarize masked instructions without returning or displaying their text."""

    from transformers import AutoTokenizer

    local_tokenizer = Path(tokenizer_path)
    if not local_tokenizer.is_dir():
        raise FileNotFoundError(
            f"local tokenizer directory is missing: {local_tokenizer}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        local_tokenizer,
        local_files_only=True,
    )
    canonicalized = canonicalize_bitext(load_bitext(input_path).records)
    lengths = [
        len(tokenizer.encode(record.instruction, add_special_tokens=False))
        for record in canonicalized.records
    ]
    if not lengths:
        return LengthSummary(
            minimum=0,
            maximum=0,
            mean=0.0,
            median=0.0,
            p95=0.0,
        )
    ordered = sorted(lengths)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return LengthSummary(
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=round(statistics.fmean(ordered), 3),
        median=float(statistics.median(ordered)),
        p95=float(ordered[p95_index]),
    )
