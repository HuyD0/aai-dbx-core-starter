"""Conservative template inference and near-duplicate clustering."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

from .normalization import canonical_text, group_id
from .schemas import CanonicalRecord, GroupingResult, RecordGroup

_WORD = re.compile(r"[a-z0-9]+|<[a-z_]+>")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "for",
        "from",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "please",
        "the",
        "to",
        "want",
        "with",
        "you",
    }
)
_MAX_INDEX_BUCKET = 64
_MAX_CHARACTER_NGRAM = 4
DEFAULT_MAX_LENGTH_FALLBACK_CANDIDATES = 768


class NearDuplicateCandidateLimitError(RuntimeError):
    """A recall-preserving exhaustive candidate audit exceeded its budget."""


@dataclass(frozen=True)
class TextSimilarityPair:
    left_index: int
    right_index: int
    score: float


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def group_related_records(
    records: tuple[CanonicalRecord, ...],
    *,
    near_threshold: float = 0.9,
    max_length_fallback_candidates: int = (DEFAULT_MAX_LENGTH_FALLBACK_CANDIDATES),
) -> GroupingResult:
    """Keep exact templates and conservative near duplicates in one split group."""

    ordered = tuple(sorted(records, key=lambda record: record.example_id))
    union_find = _UnionFind(len(ordered))
    template_members: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(ordered):
        template_members[record.template_id].append(index)
    for members in template_members.values():
        for member in members[1:]:
            union_find.union(members[0], member)

    texts = [record.instruction for record in ordered]
    template_ids = [record.template_id for record in ordered]
    near_pairs = find_near_text_pairs(
        texts,
        threshold=near_threshold,
        excluded_group_keys=template_ids,
        max_length_fallback_candidates=max_length_fallback_candidates,
    )
    for pair in near_pairs:
        union_find.union(pair.left_index, pair.right_index)

    clustered: dict[int, list[CanonicalRecord]] = defaultdict(list)
    for index, record in enumerate(ordered):
        clustered[union_find.find(index)].append(record)

    groups: list[RecordGroup] = []
    root_to_group_id: dict[int, str] = {}
    for root, members in clustered.items():
        sorted_members = tuple(sorted(members, key=lambda record: record.example_id))
        labels = {(record.intent, record.category) for record in sorted_members}
        stable_group_id = group_id([record.example_id for record in sorted_members])
        root_to_group_id[root] = stable_group_id
        groups.append(
            RecordGroup(
                group_id=stable_group_id,
                records=sorted_members,
                label_conflict=len(labels) > 1,
            )
        )
    groups.sort(key=lambda group: group.group_id)

    return GroupingResult(
        groups=tuple(groups),
        near_duplicate_pair_count=len(near_pairs),
        near_duplicate_group_ids=tuple(
            sorted(
                {
                    root_to_group_id[union_find.find(pair.left_index)]
                    for pair in near_pairs
                }
            )
        ),
        inferred_template_groups=len(template_members),
        repeated_template_groups=sum(
            len(members) > 1 for members in template_members.values()
        ),
    )


def merge_record_groups(
    grouping: GroupingResult,
    links: set[tuple[str, str]],
) -> GroupingResult:
    """Merge groups linked by a post-split leakage audit and retain provenance."""

    ordered = tuple(sorted(grouping.groups, key=lambda group: group.group_id))
    positions = {group.group_id: index for index, group in enumerate(ordered)}
    union_find = _UnionFind(len(ordered))
    for left_group, right_group in sorted(links):
        if left_group in positions and right_group in positions:
            union_find.union(positions[left_group], positions[right_group])

    merged_members: dict[int, list[CanonicalRecord]] = defaultdict(list)
    old_ids_by_root: dict[int, set[str]] = defaultdict(set)
    for index, group in enumerate(ordered):
        root = union_find.find(index)
        merged_members[root].extend(group.records)
        old_ids_by_root[root].add(group.group_id)

    groups: list[RecordGroup] = []
    near_group_ids: list[str] = []
    linked_ids = {value for link in links for value in link}
    old_near_ids = set(grouping.near_duplicate_group_ids)
    for root, records in merged_members.items():
        members = tuple(sorted(records, key=lambda record: record.example_id))
        stable_group_id = group_id([record.example_id for record in members])
        labels = {(record.intent, record.category) for record in members}
        groups.append(
            RecordGroup(
                group_id=stable_group_id,
                records=members,
                label_conflict=len(labels) > 1,
            )
        )
        if old_ids_by_root[root] & (linked_ids | old_near_ids):
            near_group_ids.append(stable_group_id)
    groups.sort(key=lambda group: group.group_id)
    return GroupingResult(
        groups=tuple(groups),
        near_duplicate_pair_count=(grouping.near_duplicate_pair_count + len(links)),
        near_duplicate_group_ids=tuple(sorted(near_group_ids)),
        inferred_template_groups=grouping.inferred_template_groups,
        repeated_template_groups=grouping.repeated_template_groups,
    )


def text_similarity(left: str, right: str) -> float:
    """Return a stable lexical similarity score in the inclusive range [0, 1]."""

    normalized_left = canonical_text(left)
    normalized_right = canonical_text(right)
    if normalized_left == normalized_right:
        return 1.0
    if not normalized_left or not normalized_right:
        return 0.0
    left_tokens = set(_tokens(normalized_left))
    right_tokens = set(_tokens(normalized_right))
    union = left_tokens | right_tokens
    token_score = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence_score = SequenceMatcher(
        None,
        normalized_left,
        normalized_right,
        autojunk=False,
    ).ratio()
    return round(max(token_score, sequence_score), 6)


def find_near_text_pairs(
    texts: list[str],
    *,
    threshold: float = 0.9,
    excluded_group_keys: list[str] | None = None,
    max_length_fallback_candidates: int = (DEFAULT_MAX_LENGTH_FALLBACK_CANDIDATES),
) -> tuple[TextSimilarityPair, ...]:
    """Find likely near duplicates with an indexed fast path.

    Candidate generation combines rare-token indexing, an exact global-prefix
    filter for token-set Jaccard similarity, SimHash bands, stable edge buckets,
    and a bounded character-block fallback for sequence-similar text. The
    fallback fails closed instead of truncating when its explicit budget is
    exceeded. Every candidate is confirmed with the public lexical similarity
    function. Exact matches are returned; callers can exclude a known
    exact/template group with ``excluded_group_keys``.
    """

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in the interval (0, 1]")
    if max_length_fallback_candidates < 1:
        raise ValueError("max_length_fallback_candidates must be positive")

    normalized = [canonical_text(text) for text in texts]
    token_lists = [_tokens(text) for text in normalized]
    token_sets = [set(tokens) for tokens in token_lists]
    character_counts = [Counter(text) for text in normalized]
    frequencies: Counter[str] = Counter()
    for tokens in token_sets:
        frequencies.update(tokens)
    required_ngram_sizes = _required_ngram_sizes(
        {len(text) for text in normalized if text},
        threshold=threshold,
    )
    indexed_ngram_sizes = set(required_ngram_sizes.values())

    token_index: dict[str, list[int]] = defaultdict(list)
    token_prefix_index: dict[str, list[int]] = defaultdict(list)
    band_index: dict[tuple[int, int], list[int]] = defaultdict(list)
    edge_index: dict[tuple[str, tuple[str, ...], int], list[int]] = defaultdict(list)
    character_ngram_index: dict[tuple[int, str], list[int]] = defaultdict(list)
    pairs: list[TextSimilarityPair] = []

    for right_index, text in enumerate(normalized):
        tokens = token_lists[right_index]
        right_token_set = token_sets[right_index]
        informative = [token for token in set(tokens) if token not in _STOPWORDS]
        if not informative:
            informative = list(set(tokens))
        selected = sorted(
            informative,
            key=lambda token: (frequencies[token], -len(token), token),
        )[:4]

        candidates: set[int] = set()
        overflow_candidates: set[int] = set()
        for token in selected:
            bucket = token_index[token]
            if frequencies[token] <= _MAX_INDEX_BUCKET:
                candidates.update(bucket)
            else:
                overflow_candidates.update(bucket)
        simhash = _simhash(tokens)
        for band in range(4):
            bucket = band_index[(band, (simhash >> (band * 16)) & 0xFFFF)]
            if len(bucket) <= _MAX_INDEX_BUCKET:
                candidates.update(bucket)
            else:
                overflow_candidates.update(bucket)
        length_bucket = len(tokens) // 3
        prefix = tuple(tokens[:2])
        suffix = tuple(tokens[-2:])
        prefix_bucket = edge_index[("prefix", prefix, length_bucket)]
        suffix_bucket = edge_index[("suffix", suffix, length_bucket)]
        if len(prefix_bucket) <= _MAX_INDEX_BUCKET:
            candidates.update(prefix_bucket)
        else:
            overflow_candidates.update(prefix_bucket)
        if len(suffix_bucket) <= _MAX_INDEX_BUCKET:
            candidates.update(suffix_bucket)
        else:
            overflow_candidates.update(suffix_bucket)

        # Dropping an over-cap bucket loses recall exactly where a repeated phrase
        # family is most likely. Exhaustively confirm those bucket members instead;
        # the length and quick-ratio checks below still discard impossible matches.
        candidates.update(overflow_candidates)

        token_prefix = _token_set_prefix(
            right_token_set,
            frequencies=frequencies,
            threshold=threshold,
        )
        for token in token_prefix:
            for left_index in token_prefix_index[token]:
                if _excluded_pair(
                    left_index,
                    right_index,
                    excluded_group_keys=excluded_group_keys,
                ):
                    continue
                left_token_set = token_sets[left_index]
                if not left_token_set or not right_token_set:
                    continue
                if _token_set_similarity(left_token_set, right_token_set) >= threshold:
                    candidates.add(left_index)

        candidates.update(
            _sequence_similarity_fallback_candidates(
                right_index=right_index,
                normalized=normalized,
                character_counts=character_counts,
                character_ngram_index=character_ngram_index,
                required_ngram_size=required_ngram_sizes.get(len(text)),
                existing_candidates=candidates,
                excluded_group_keys=excluded_group_keys,
                threshold=threshold,
                max_candidates=max_length_fallback_candidates,
            )
        )

        for left_index in sorted(candidates):
            if _excluded_pair(
                left_index,
                right_index,
                excluded_group_keys=excluded_group_keys,
            ):
                continue
            left_length = len(normalized[left_index])
            right_length = len(text)
            if not left_length or not right_length:
                continue
            left_token_set = token_sets[left_index]
            token_score = _token_set_similarity(left_token_set, right_token_set)
            if token_score < threshold and (
                _maximum_sequence_similarity(left_length, right_length) < threshold
            ):
                continue
            if token_score < threshold and (
                _character_multiset_ratio(
                    character_counts[left_index],
                    character_counts[right_index],
                    left_length=left_length,
                    right_length=right_length,
                )
                < threshold
            ):
                continue
            score = _confirmed_similarity(
                normalized[left_index],
                text,
                token_sets[left_index],
                token_sets[right_index],
                threshold,
            )
            if score >= threshold:
                pairs.append(
                    TextSimilarityPair(
                        left_index=left_index,
                        right_index=right_index,
                        score=score,
                    )
                )

        for token in selected:
            token_index[token].append(right_index)
        for token in token_prefix:
            token_prefix_index[token].append(right_index)
        for band in range(4):
            band_index[(band, (simhash >> (band * 16)) & 0xFFFF)].append(right_index)
        edge_index[("prefix", prefix, length_bucket)].append(right_index)
        edge_index[("suffix", suffix, length_bucket)].append(right_index)
        for ngram_size in indexed_ngram_sizes:
            for ngram in _character_ngrams(text, ngram_size):
                character_ngram_index[(ngram_size, ngram)].append(right_index)

    return tuple(pairs)


def _tokens(value: str) -> list[str]:
    return _WORD.findall(value)


def _token_set_prefix(
    tokens: set[str],
    *,
    frequencies: Counter[str],
    threshold: float,
) -> tuple[str, ...]:
    """Return the global-order prefix required by Jaccard prefix filtering."""

    if not tokens:
        return ()
    prefix_length = max(0, len(tokens) - math.ceil(threshold * len(tokens)) + 1)
    ordered = sorted(
        tokens,
        key=lambda token: (frequencies[token], -len(token), token),
    )
    return tuple(ordered[:prefix_length])


def _token_set_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _excluded_pair(
    left_index: int,
    right_index: int,
    *,
    excluded_group_keys: list[str] | None,
) -> bool:
    return (
        excluded_group_keys is not None
        and excluded_group_keys[left_index] == excluded_group_keys[right_index]
    )


def _required_ngram_sizes(
    lengths: set[int],
    *,
    threshold: float,
) -> dict[int, int]:
    """Return the weakest sound matching-block requirement per observed length."""

    required: dict[int, int] = {}
    ordered_lengths = sorted(lengths)
    for right_length in ordered_lengths:
        compatible_sizes = (
            size
            for left_length in ordered_lengths
            if (
                size := _required_matching_block_size(
                    left_length,
                    right_length,
                    threshold=threshold,
                )
            )
            is not None
        )
        minimum_size = min(compatible_sizes, default=None)
        if minimum_size is not None:
            required[right_length] = minimum_size
    return required


def _required_matching_block_size(
    left_length: int,
    right_length: int,
    *,
    threshold: float,
) -> int | None:
    """Return a shared n-gram size required by a SequenceMatcher hit.

    A ratio at least ``threshold`` requires a minimum number of matched
    characters. Every gap between matching blocks consumes at least one
    unmatched character, which bounds the number of blocks and therefore the
    size of the largest exact shared block.
    """

    if not left_length or not right_length:
        return None
    total_length = left_length + right_length
    minimum_matches = max(1, math.ceil(threshold * total_length / 2.0))
    while (
        minimum_matches > 1
        and (2.0 * (minimum_matches - 1)) / total_length >= threshold
    ):
        minimum_matches -= 1
    while (2.0 * minimum_matches) / total_length < threshold:
        minimum_matches += 1
    if minimum_matches > min(left_length, right_length):
        return None
    maximum_unmatched = total_length - (2 * minimum_matches)
    required_size = math.ceil(minimum_matches / (maximum_unmatched + 1))
    return min(_MAX_CHARACTER_NGRAM, max(1, required_size))


def _character_ngrams(text: str, size: int) -> tuple[str, ...]:
    return tuple(
        sorted({text[index : index + size] for index in range(len(text) - size + 1)})
    )


def _character_multiset_ratio(
    left: Counter[str],
    right: Counter[str],
    *,
    left_length: int,
    right_length: int,
) -> float:
    matches = sum((left & right).values())
    return (2.0 * matches) / (left_length + right_length)


def _maximum_sequence_similarity(left_length: int, right_length: int) -> float:
    return (2.0 * min(left_length, right_length)) / (left_length + right_length)


def _sequence_similarity_fallback_candidates(
    *,
    right_index: int,
    normalized: list[str],
    character_counts: list[Counter[str]],
    character_ngram_index: dict[tuple[int, str], list[int]],
    required_ngram_size: int | None,
    existing_candidates: set[int],
    excluded_group_keys: list[str] | None,
    threshold: float,
    max_candidates: int,
) -> tuple[int, ...]:
    """Return all still-possible sequence candidates or fail closed."""

    right_length = len(normalized[right_index])
    if not right_length or required_ngram_size is None:
        return ()
    possible: set[int] = set()
    for ngram in _character_ngrams(
        normalized[right_index],
        required_ngram_size,
    ):
        possible.update(character_ngram_index[(required_ngram_size, ngram)])

    fallback: list[int] = []
    for left_index in sorted(possible):
        if left_index in existing_candidates or _excluded_pair(
            left_index,
            right_index,
            excluded_group_keys=excluded_group_keys,
        ):
            continue
        left_length = len(normalized[left_index])
        if _maximum_sequence_similarity(left_length, right_length) < threshold:
            continue
        if (
            _character_multiset_ratio(
                character_counts[left_index],
                character_counts[right_index],
                left_length=left_length,
                right_length=right_length,
            )
            < threshold
        ):
            continue
        fallback.append(left_index)
        if len(fallback) > max_candidates:
            raise NearDuplicateCandidateLimitError(
                "sequence-similarity near-duplicate audit exceeded "
                f"max_length_fallback_candidates={max_candidates} at record "
                f"index {right_index}; rerun with a larger explicit budget or "
                "partition the audit without crossing split/group boundaries"
            )
    return tuple(sorted(fallback))


def _confirmed_similarity(
    left: str,
    right: str,
    left_tokens: set[str],
    right_tokens: set[str],
    threshold: float,
) -> float:
    token_score = _token_set_similarity(left_tokens, right_tokens)
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    return round(max(token_score, matcher.ratio()), 6)


def _simhash(tokens: list[str]) -> int:
    if not tokens:
        return 0
    weights = [0] * 64
    for token in tokens:
        token_hash = int.from_bytes(
            hashlib.sha256(token.encode("utf-8")).digest()[:8], "big"
        )
        for bit in range(64):
            weights[bit] += 1 if token_hash & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result
