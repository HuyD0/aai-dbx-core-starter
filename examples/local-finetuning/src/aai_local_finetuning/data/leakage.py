"""Automated leakage checks for generated chat splits and demonstrations."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from .duplicates import (
    DEFAULT_MAX_LENGTH_FALLBACK_CANDIDATES,
    find_near_text_pairs,
    text_similarity,
)
from .normalization import canonical_text, inferred_template
from .schemas import (
    ChatExample,
    LeakageFinding,
    LeakageKind,
    LeakageReport,
    TrainingTarget,
)


class DatasetLeakageError(ValueError):
    """Raised when a prepared dataset violates a split boundary."""


class _FindingCollector:
    def __init__(self, max_findings: int) -> None:
        self.max_findings = max_findings
        self.counts: Counter[str] = Counter()
        self.findings: list[LeakageFinding] = []

    def add(self, finding: LeakageFinding) -> None:
        self.counts[finding.kind.value] += 1
        if len(self.findings) < self.max_findings:
            self.findings.append(finding)

    def report(self) -> LeakageReport:
        counts = {kind.value: self.counts[kind.value] for kind in LeakageKind}
        total = sum(counts.values())
        return LeakageReport(
            passed=total == 0,
            counts=counts,
            findings=tuple(self.findings),
            truncated=total > len(self.findings),
        )


def check_split_leakage(
    splits: Mapping[str, Sequence[ChatExample]],
    *,
    few_shot: Sequence[ChatExample] = (),
    prompt_tuning: Sequence[ChatExample] = (),
    near_threshold: float = 0.9,
    max_findings: int = 1_000,
    max_length_fallback_candidates: int = (DEFAULT_MAX_LENGTH_FALLBACK_CANDIDATES),
) -> LeakageReport:
    """Check exact/template/near, group, target, and evaluation-set leakage."""

    if max_findings < 1:
        raise ValueError("max_findings must be positive")
    collector = _FindingCollector(max_findings)
    normalized_splits = {
        str(name): tuple(examples) for name, examples in sorted(splits.items())
    }

    for split_name, examples in normalized_splits.items():
        for example in examples:
            _check_target_leakage(split_name, example, collector)

    split_names = list(normalized_splits)
    for left_position, left_name in enumerate(split_names):
        for right_name in split_names[left_position + 1 :]:
            _check_split_pair(
                left_name,
                normalized_splits[left_name],
                right_name,
                normalized_splits[right_name],
                near_threshold,
                collector,
                max_length_fallback_candidates,
            )

    frozen_test = normalized_splits.get("test", ())
    _check_reference_overlap(
        "few_shot",
        few_shot,
        frozen_test,
        LeakageKind.FEW_SHOT_TEST_OVERLAP,
        near_threshold,
        collector,
    )
    _check_reference_overlap(
        "prompt_tuning",
        prompt_tuning,
        frozen_test,
        LeakageKind.PROMPT_TUNING_TEST_OVERLAP,
        near_threshold,
        collector,
    )
    return collector.report()


def assert_no_leakage(report: LeakageReport) -> None:
    """Fail fast with aggregate counts; raw text is never copied to the error."""

    if report.passed:
        return
    summary = ", ".join(
        f"{name}={count}" for name, count in report.counts.items() if count
    )
    raise DatasetLeakageError(f"dataset leakage detected: {summary}")


def load_chat_jsonl(path: str | Path) -> tuple[ChatExample, ...]:
    """Strictly validate a portable chat JSONL file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"chat JSONL does not exist: {source}")
    examples: list[ChatExample] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                examples.append(ChatExample.model_validate_json(line))
            except ValueError as error:
                raise ValueError(
                    f"invalid chat record in {source} at line {line_number}"
                ) from error
    return tuple(examples)


def check_split_files(
    output_dir: str | Path,
    *,
    few_shot_path: str | Path | None = None,
    prompt_tuning_path: str | Path | None = None,
    near_threshold: float = 0.9,
    max_length_fallback_candidates: int = (DEFAULT_MAX_LENGTH_FALLBACK_CANDIDATES),
) -> LeakageReport:
    """Load the standard local filenames and run all leakage checks."""

    root = Path(output_dir)
    splits = {
        "train": load_chat_jsonl(root / "train.jsonl"),
        "validation": load_chat_jsonl(root / "valid.jsonl"),
        "test": load_chat_jsonl(root / "test.jsonl"),
    }
    few_shot = load_chat_jsonl(few_shot_path) if few_shot_path else ()
    prompt_tuning = load_chat_jsonl(prompt_tuning_path) if prompt_tuning_path else ()
    return check_split_leakage(
        splits,
        few_shot=few_shot,
        prompt_tuning=prompt_tuning,
        near_threshold=near_threshold,
        max_length_fallback_candidates=max_length_fallback_candidates,
    )


def _check_split_pair(
    left_name: str,
    left: tuple[ChatExample, ...],
    right_name: str,
    right: tuple[ChatExample, ...],
    near_threshold: float,
    collector: _FindingCollector,
    max_length_fallback_candidates: int,
) -> None:
    _check_key_overlap(
        left_name,
        left,
        right_name,
        right,
        key=lambda example: example.example_id,
        kind=LeakageKind.EXACT_DUPLICATE,
        detail="stable example ID appears in both splits",
        collector=collector,
    )
    _check_key_overlap(
        left_name,
        left,
        right_name,
        right,
        key=lambda example: canonical_text(_user_text(example)),
        kind=LeakageKind.EXACT_DUPLICATE,
        detail="normalized customer request appears in both splits",
        collector=collector,
    )
    _check_key_overlap(
        left_name,
        left,
        right_name,
        right,
        key=lambda example: inferred_template(_user_text(example)),
        kind=LeakageKind.TEMPLATE_DUPLICATE,
        detail="inferred request template appears in both splits",
        collector=collector,
    )
    _check_key_overlap(
        left_name,
        left,
        right_name,
        right,
        key=lambda example: example.metadata.split_group,
        kind=LeakageKind.SOURCE_GROUP_OVERLAP,
        detail="duplicate/source group appears in both splits",
        collector=collector,
    )

    combined = (*left, *right)
    texts = [_user_text(example) for example in combined]
    templates = [inferred_template(text) for text in texts]
    for pair in find_near_text_pairs(
        texts,
        threshold=near_threshold,
        excluded_group_keys=templates,
        max_length_fallback_candidates=max_length_fallback_candidates,
    ):
        if pair.left_index < len(left) <= pair.right_index:
            left_example = combined[pair.left_index]
            right_example = combined[pair.right_index]
            collector.add(
                LeakageFinding(
                    kind=LeakageKind.NEAR_DUPLICATE,
                    split_left=left_name,
                    split_right=right_name,
                    example_id_left=left_example.example_id,
                    example_id_right=right_example.example_id,
                    similarity=pair.score,
                    detail="lexically near-duplicate customer requests cross splits",
                )
            )


def _check_key_overlap(
    left_name: str,
    left: tuple[ChatExample, ...],
    right_name: str,
    right: tuple[ChatExample, ...],
    *,
    key: object,
    kind: LeakageKind,
    detail: str,
    collector: _FindingCollector,
) -> None:
    # ``key`` is intentionally accepted as object to keep the public module free of
    # a runtime dependency on typing extensions on the offline machine.
    key_function = key
    left_by_key: dict[str, list[ChatExample]] = defaultdict(list)
    right_by_key: dict[str, list[ChatExample]] = defaultdict(list)
    for example in left:
        left_by_key[key_function(example)].append(example)  # type: ignore[operator]
    for example in right:
        right_by_key[key_function(example)].append(example)  # type: ignore[operator]
    for shared in sorted(set(left_by_key) & set(right_by_key)):
        for left_example in left_by_key[shared]:
            for right_example in right_by_key[shared]:
                collector.add(
                    LeakageFinding(
                        kind=kind,
                        split_left=left_name,
                        split_right=right_name,
                        example_id_left=left_example.example_id,
                        example_id_right=right_example.example_id,
                        similarity=1.0 if kind == LeakageKind.EXACT_DUPLICATE else None,
                        detail=detail,
                    )
                )


def _check_target_leakage(
    split_name: str,
    example: ChatExample,
    collector: _FindingCollector,
) -> None:
    target = TrainingTarget.model_validate_json(example.messages[-1].content)
    prompt_messages = "\n".join(message.content for message in example.messages[:-1])
    prompt = canonical_text(prompt_messages)
    response = canonical_text(target.response)
    serialized_target = canonical_text(example.messages[-1].content)
    explicit_intent = re.search(
        rf"\bintent\s*[:=]\s*['\"]?{re.escape(target.intent.casefold())}\b",
        prompt,
    )
    explicit_category = re.search(
        rf"\bcategory\s*[:=]\s*['\"]?{re.escape(target.category.casefold())}\b",
        prompt,
    )
    leaked = (
        serialized_target in prompt
        or (len(response) >= 8 and response in prompt)
        or explicit_intent is not None
        or explicit_category is not None
    )
    if leaked:
        collector.add(
            LeakageFinding(
                kind=LeakageKind.TARGET_IN_PROMPT,
                split_left=split_name,
                split_right=None,
                example_id_left=example.example_id,
                example_id_right=None,
                similarity=None,
                detail="assistant target or target label is present in prompt messages",
            )
        )


def _check_reference_overlap(
    reference_name: str,
    references: Sequence[ChatExample],
    frozen_test: Sequence[ChatExample],
    kind: LeakageKind,
    near_threshold: float,
    collector: _FindingCollector,
) -> None:
    for reference in references:
        reference_text = _user_text(reference)
        reference_template = inferred_template(reference_text)
        for test_example in frozen_test:
            test_text = _user_text(test_example)
            score = text_similarity(reference_text, test_text)
            overlap_reason: str | None = None
            if reference.example_id == test_example.example_id:
                overlap_reason = "stable example ID"
            elif reference.metadata.split_group == test_example.metadata.split_group:
                overlap_reason = "duplicate/source group"
            elif canonical_text(reference_text) == canonical_text(test_text):
                overlap_reason = "exact customer request"
            elif reference_template == inferred_template(test_text):
                overlap_reason = "inferred request template"
            elif score >= near_threshold:
                overlap_reason = "near-duplicate customer request"
            if overlap_reason:
                collector.add(
                    LeakageFinding(
                        kind=kind,
                        split_left=reference_name,
                        split_right="test",
                        example_id_left=reference.example_id,
                        example_id_right=test_example.example_id,
                        similarity=score,
                        detail=f"{overlap_reason} overlaps the frozen test set",
                    )
                )


def _user_text(example: ChatExample) -> str:
    return example.messages[1].content
