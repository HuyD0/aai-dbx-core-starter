"""Regression tests for conservative near-duplicate candidate generation."""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from aai_local_finetuning.data import duplicates
from aai_local_finetuning.data.duplicates import (
    NearDuplicateCandidateLimitError,
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


def test_long_pair_with_disjoint_rare_terms_edges_and_simhash_bands_is_found() -> None:
    base = [f"sharedtoken{index:03d}" for index in range(120)]
    positions = (0, 23, 47, 71, 95, 119)
    left_replacements = (
        "ldtwyuqhixijxc",
        "lzilzuuqefrvvi",
        "lsotduvdssuulf",
        "lqiddciidafxnd",
        "lonntdipmhotpc",
        "lwslpiljqgburp",
    )
    right_replacements = (
        "rvojovmmydihkl",
        "rfaucdtkacigmm",
        "rdxpqvgiotgpzj",
        "rvqscnpvfrmojp",
        "rsepwisoawmnal",
        "rzidmdbgjaasqj",
    )
    left_tokens = list(base)
    right_tokens = list(base)
    for position, left_value, right_value in zip(
        positions,
        left_replacements,
        right_replacements,
        strict=True,
    ):
        left_tokens[position] = left_value
        right_tokens[position] = right_value
    left = " ".join(left_tokens)
    right = " ".join(right_tokens)

    left_hash = duplicates._simhash(duplicates._tokens(left))
    right_hash = duplicates._simhash(duplicates._tokens(right))
    assert all(
        ((left_hash >> (band * 16)) & 0xFFFF) != ((right_hash >> (band * 16)) & 0xFFFF)
        for band in range(4)
    )
    assert left_tokens[:2] != right_tokens[:2]
    assert left_tokens[-2:] != right_tokens[-2:]
    assert text_similarity(left, right) == 0.958866

    assert find_near_text_pairs([left, right], threshold=0.9) == (
        duplicates.TextSimilarityPair(
            left_index=0,
            right_index=1,
            score=0.958866,
        ),
    )


def test_short_single_token_sequence_pair_is_found() -> None:
    left = "abcdefghijklmnopqrst"
    right = "abcdefghijklmnopqrsu"

    assert text_similarity(left, right) == 0.95
    assert find_near_text_pairs([left, right], threshold=0.9) == (
        duplicates.TextSimilarityPair(left_index=0, right_index=1, score=0.95),
    )


def test_sequence_pair_at_exact_length_compatibility_boundary_is_found() -> None:
    left = "baba"
    right = "abbaba"

    assert text_similarity(left, right) == 0.8
    assert find_near_text_pairs([left, right], threshold=0.8) == (
        duplicates.TextSimilarityPair(left_index=0, right_index=1, score=0.8),
    )


def test_token_jaccard_candidate_is_not_dropped_by_character_length_filter() -> None:
    left = "alpha beta"
    right = " ".join((*("alpha" for _ in range(20)), "beta"))

    assert len(left) / len(right) < 0.9 / (2.0 - 0.9)
    assert text_similarity(left, right) == 1.0
    assert find_near_text_pairs([left, right], threshold=0.9) == (
        duplicates.TextSimilarityPair(left_index=0, right_index=1, score=1.0),
    )


@pytest.mark.parametrize("threshold", (float("nan"), -0.1, 0.0, 1.01))
def test_near_pair_threshold_must_be_in_safe_range(threshold: float) -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        find_near_text_pairs(["alpha", "alpha"], threshold=threshold)


def test_tiny_positive_threshold_uses_only_observed_ngram_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = [_unique_long_text(0), _unique_long_text(1)]
    simhashes = iter((0, sum(1 << (band * 16) for band in range(4))))
    monkeypatch.setattr(duplicates, "_simhash", lambda _tokens: next(simhashes))

    pairs = find_near_text_pairs(texts, threshold=1e-12)

    assert [(pair.left_index, pair.right_index) for pair in pairs] == [(0, 1)]


def test_long_text_exhaustive_fallback_has_a_per_record_comparison_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = 32
    texts = [_permuted_character_text(index) for index in range(budget + 1)]
    simhash_index = iter(range(len(texts)))
    confirmations = 0

    def unique_simhash(_tokens: list[str]) -> int:
        value = next(simhash_index)
        return sum(value << (band * 16) for band in range(4))

    def reject_candidate(*_args: object, **_kwargs: object) -> float:
        nonlocal confirmations
        confirmations += 1
        return 0.0

    monkeypatch.setattr(duplicates, "_simhash", unique_simhash)
    monkeypatch.setattr(duplicates, "_confirmed_similarity", reject_candidate)

    assert (
        find_near_text_pairs(
            texts,
            threshold=0.9,
            max_length_fallback_candidates=budget,
        )
        == ()
    )
    assert confirmations == budget * (budget + 1) // 2


def test_long_text_exhaustive_fallback_fails_closed_over_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = 8
    texts = [_permuted_character_text(index) for index in range(budget + 2)]
    simhash_index = iter(range(len(texts)))

    def unique_simhash(_tokens: list[str]) -> int:
        value = next(simhash_index)
        return sum(value << (band * 16) for band in range(4))

    monkeypatch.setattr(duplicates, "_simhash", unique_simhash)
    monkeypatch.setattr(
        duplicates,
        "_confirmed_similarity",
        lambda *_args, **_kwargs: 0.0,
    )

    with pytest.raises(
        NearDuplicateCandidateLimitError,
        match="larger explicit budget",
    ):
        find_near_text_pairs(
            texts,
            threshold=0.9,
            max_length_fallback_candidates=budget,
        )


def _alphabetic_word(value: int) -> str:
    letters: list[str] = []
    number = value + 1
    for _ in range(12):
        number, remainder = divmod(number * 17 + 11, 26)
        letters.append(chr(ord("a") + remainder))
    return "".join(letters)


def _unique_long_text(record_index: int) -> str:
    return " ".join(
        _alphabetic_word(record_index * 200 + token_index) for token_index in range(120)
    )


def _permuted_character_text(record_index: int) -> str:
    characters = list("abcdefghijklmnopqrstuvwxyz" * 12)
    random.Random(record_index).shuffle(characters)
    characters[156:156] = "abcd"
    return "".join(characters)
