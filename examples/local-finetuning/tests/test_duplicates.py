"""Regression tests for conservative near-duplicate candidate generation."""

from __future__ import annotations

from itertools import combinations

from aai_local_finetuning.data.duplicates import (
    find_near_text_pairs,
    text_similarity,
)


def test_overflowed_buckets_preserve_near_duplicate_pair_recall() -> None:
    shared_phrase = " ".join(
        (
            "commonalpha",
            "commonbravo",
            "commoncharlie",
            "commondelta",
            "commonecho",
            "commonfoxtrot",
            "commongolf",
            "commonhotel",
            "commonindia",
            "commonjuliet",
            "commonkilo",
            "commonlima",
            "commonmike",
            "commonnovember",
            "commonoscar",
            "commonpapa",
            "commonquebec",
            "commonromeo",
            "commonsierra",
            "commontango",
        )
    )
    texts = [
        f"{_alphabetic_word(index)} {shared_phrase} {_alphabetic_word(index + 100)}"
        for index in range(70)
    ]
    expected = {
        (left_index, right_index)
        for left_index, right_index in combinations(range(len(texts)), 2)
        if text_similarity(texts[left_index], texts[right_index]) >= 0.9
    }

    actual = {
        (pair.left_index, pair.right_index)
        for pair in find_near_text_pairs(texts, threshold=0.9)
    }

    assert len(expected) == 2_415
    assert actual == expected


def _alphabetic_word(value: int) -> str:
    letters: list[str] = []
    number = value + 1
    for _ in range(12):
        number, remainder = divmod(number * 17 + 11, 26)
        letters.append(chr(ord("a") + remainder))
    return "".join(letters)
