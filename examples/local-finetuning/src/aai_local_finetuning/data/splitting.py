"""Deterministic, intent-balanced, duplicate-group-aware curation."""

from __future__ import annotations

from collections import defaultdict

from .normalization import stable_order_key
from .policies import (
    classify_difficulty,
    parse_flags,
    render_training_response,
    requires_escalation,
)
from .schemas import (
    ChatExample,
    ChatMessage,
    ChatMetadata,
    CuratedSplits,
    GroupedRecord,
    GroupingResult,
    MessageRole,
    PreparationConfig,
    RecordGroup,
    TrainingTarget,
)

_SPLIT_ORDER = ("train", "validation", "test")


def curate_splits(grouping: GroupingResult, config: PreparationConfig) -> CuratedSplits:
    """Build balanced per-intent splits while keeping each group atomic."""

    conflicting = [group for group in grouping.groups if group.label_conflict]
    usable = [group for group in grouping.groups if not group.label_conflict]
    by_intent: dict[str, list[RecordGroup]] = defaultdict(list)
    for group in usable:
        if not group.records:
            continue
        by_intent[group.records[0].intent].append(group)

    selected: dict[str, list[GroupedRecord]] = {
        split_name: [] for split_name in _SPLIT_ORDER
    }
    targets = {
        "train": config.train_per_intent,
        "validation": config.validation_per_intent,
        "test": config.test_per_intent,
    }
    for intent in sorted(by_intent):
        groups = sorted(
            by_intent[intent],
            key=lambda group: (
                -len(group.records),
                stable_order_key(config.seed, intent, group.group_id),
            ),
        )
        available = sum(len(group.records) for group in groups)
        desired = _scaled_targets(available, targets)
        remaining = dict(desired)

        # Reserve one independent group for each requested split before allocating
        # additional groups. This keeps small datasets evaluable instead of allowing
        # the larger training quota to consume every duplicate family.
        split_queue = [name for name in _SPLIT_ORDER if remaining[name] > 0]
        unassigned = list(groups)
        for split_name in split_queue:
            if not unassigned:
                break
            group = unassigned.pop(0)
            _select_from_group(
                selected[split_name],
                group,
                limit=remaining[split_name],
                seed=config.seed,
            )
            remaining[split_name] = max(
                0,
                desired[split_name]
                - sum(item.record.intent == intent for item in selected[split_name]),
            )

        for group in unassigned:
            candidates = [name for name in _SPLIT_ORDER if remaining[name] > 0]
            if not candidates:
                break
            split_name = max(
                candidates,
                key=lambda name: (
                    remaining[name] / desired[name],
                    remaining[name],
                    -_SPLIT_ORDER.index(name),
                ),
            )
            amount = min(remaining[split_name], len(group.records))
            _select_from_group(
                selected[split_name],
                group,
                limit=amount,
                seed=config.seed,
            )
            remaining[split_name] -= amount

    for split_name, items in selected.items():
        items.sort(
            key=lambda item: stable_order_key(
                config.seed,
                split_name,
                item.record.example_id,
            )
        )

    return CuratedSplits(
        train=tuple(selected["train"]),
        validation=tuple(selected["validation"]),
        test=tuple(selected["test"]),
        conflicting_group_count=len(conflicting),
        excluded_conflicting_records=sum(len(group.records) for group in conflicting),
    )


def to_chat_examples(
    split: tuple[GroupedRecord, ...], config: PreparationConfig
) -> tuple[ChatExample, ...]:
    """Render one curated logical split as portable chat records."""

    examples: list[ChatExample] = []
    for item in split:
        record = item.record
        escalation = requires_escalation(
            category=record.category,
            intent=record.intent,
        )
        target = TrainingTarget(
            intent=record.intent,
            category=record.category,
            requires_escalation=escalation,
            response=render_training_response(
                intent=record.intent,
                escalation=escalation,
            ),
        )
        flags = parse_flags(record.flags)
        examples.append(
            ChatExample(
                example_id=record.example_id,
                source_dataset=config.source_dataset,
                source_version=config.source_version,
                messages=(
                    ChatMessage(role=MessageRole.SYSTEM, content=config.system_prompt),
                    ChatMessage(role=MessageRole.USER, content=record.instruction),
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=target.model_dump_json(),
                    ),
                ),
                metadata=ChatMetadata(
                    intent=record.intent,
                    category=record.category,
                    split_group=item.group_id,
                    flags=flags,
                    difficulty=classify_difficulty(record.instruction, flags),
                ),
            )
        )
    return tuple(examples)


def _scaled_targets(available: int, targets: dict[str, int]) -> dict[str, int]:
    requested = sum(targets.values())
    if available >= requested:
        return dict(targets)
    if not available:
        return {name: 0 for name in _SPLIT_ORDER}

    raw = {name: available * targets[name] / requested for name in _SPLIT_ORDER}
    scaled = {name: int(raw[name]) for name in _SPLIT_ORDER}
    remaining = available - sum(scaled.values())
    order = sorted(
        _SPLIT_ORDER,
        key=lambda name: (
            -(raw[name] - scaled[name]),
            -targets[name],
            _SPLIT_ORDER.index(name),
        ),
    )
    for name in order:
        if remaining <= 0:
            break
        if targets[name] > 0:
            scaled[name] += 1
            remaining -= 1
    return scaled


def _select_from_group(
    destination: list[GroupedRecord],
    group: RecordGroup,
    *,
    limit: int,
    seed: int,
) -> None:
    members = sorted(
        group.records,
        key=lambda record: stable_order_key(seed, group.group_id, record.example_id),
    )
    destination.extend(
        GroupedRecord(record=record, group_id=group.group_id)
        for record in members[:limit]
    )
