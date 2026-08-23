"""Privacy-preserving data-quality summaries for local study."""

from __future__ import annotations

import math
import statistics
from collections import Counter

from .normalization import SENSITIVE_PATTERN_NAMES
from .policies import classify_difficulty, parse_flags
from .schemas import (
    BitextLoadResult,
    CanonicalizationResult,
    CanonicalRecord,
    CuratedSplits,
    GroupingResult,
    LengthSummary,
    PreparationConfig,
    QualityReport,
)


def build_quality_report(
    *,
    loaded: BitextLoadResult,
    canonicalized: CanonicalizationResult,
    unique_records: tuple[CanonicalRecord, ...],
    exact_duplicate_count: int,
    grouping: GroupingResult,
    splits: CuratedSplits,
    config: PreparationConfig,
) -> QualityReport:
    """Build aggregate findings without copying source text into the report."""

    intent_distribution = Counter(record.intent for record in unique_records)
    category_distribution = Counter(record.category for record in unique_records)
    split_items = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    split_intents = {
        name: dict(sorted(Counter(item.record.intent for item in items).items()))
        for name, items in split_items.items()
    }
    curated_records = sum(len(items) for items in split_items.values())
    all_curated_items = tuple(item for items in split_items.values() for item in items)
    parsed_flags = [parse_flags(item.record.flags) for item in all_curated_items]
    flag_distribution = Counter(flag for flags in parsed_flags for flag in flags)
    difficulty_distribution = Counter(
        classify_difficulty(item.record.instruction, flags).value
        for item, flags in zip(all_curated_items, parsed_flags, strict=True)
    )
    warnings: list[str] = []
    if len(intent_distribution) != config.expected_intent_count:
        warnings.append(
            f"expected {config.expected_intent_count} intents but found "
            f"{len(intent_distribution)}"
        )
    if canonicalized.invalid_record_count or loaded.invalid_csv_rows:
        warnings.append("invalid source rows were excluded or flagged")
    if splits.conflicting_group_count:
        warnings.append("label-conflicting duplicate groups were excluded")

    requested = {
        "train": config.train_per_intent,
        "validation": config.validation_per_intent,
        "test": config.test_per_intent,
    }
    for intent in sorted(intent_distribution):
        if intent_distribution[intent] < sum(requested.values()):
            warnings.append(
                f"intent {intent!r} has fewer than the requested "
                f"{sum(requested.values())} unique examples"
            )
            continue
        for split_name, target in requested.items():
            actual = split_intents[split_name].get(intent, 0)
            if actual < target:
                warnings.append(
                    f"intent {intent!r} could not reach {split_name} target "
                    f"{target} without splitting a duplicate group"
                )

    sensitive_counts = {
        name: canonicalized.sensitive_pattern_counts.get(name, 0)
        for name in SENSITIVE_PATTERN_NAMES
    }
    valid_count = len(canonicalized.records)
    return QualityReport(
        schema_version="1.0.0",
        source_records=len(loaded.records),
        valid_records=valid_count,
        unique_records=len(unique_records),
        curated_records=curated_records,
        invalid_record_count=(
            canonicalized.invalid_record_count + loaded.invalid_csv_rows
        ),
        missing_by_field=canonicalized.missing_by_field,
        exact_duplicate_count=exact_duplicate_count,
        exact_duplicate_rate=(
            exact_duplicate_count / valid_count if valid_count else 0.0
        ),
        inferred_template_groups=grouping.inferred_template_groups,
        repeated_template_groups=grouping.repeated_template_groups,
        near_duplicate_pairs=grouping.near_duplicate_pair_count,
        near_duplicate_clusters=len(grouping.near_duplicate_group_ids),
        conflicting_group_count=splits.conflicting_group_count,
        excluded_conflicting_records=splits.excluded_conflicting_records,
        intent_distribution=dict(sorted(intent_distribution.items())),
        category_distribution=dict(sorted(category_distribution.items())),
        split_intent_distribution=split_intents,
        flag_distribution=dict(sorted(flag_distribution.items())),
        difficulty_distribution=dict(sorted(difficulty_distribution.items())),
        instruction_characters=_length_summary(
            [len(record.instruction) for record in unique_records]
        ),
        response_characters=_length_summary(
            [len(record.response) for record in unique_records]
        ),
        instruction_words=_length_summary(
            [len(record.instruction.split()) for record in unique_records]
        ),
        response_words=_length_summary(
            [len(record.response.split()) for record in unique_records]
        ),
        sensitive_pattern_counts=sensitive_counts,
        records_with_sensitive_patterns=(canonicalized.records_with_sensitive_patterns),
        warnings=tuple(warnings),
    )


def _length_summary(values: list[int]) -> LengthSummary:
    if not values:
        return LengthSummary(minimum=0, maximum=0, mean=0.0, median=0.0, p95=0.0)
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return LengthSummary(
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=round(statistics.fmean(ordered), 3),
        median=float(statistics.median(ordered)),
        p95=float(ordered[p95_index]),
    )
