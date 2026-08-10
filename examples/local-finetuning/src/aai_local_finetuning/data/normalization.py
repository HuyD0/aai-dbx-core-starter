"""Deterministic text normalization, masking, and content fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_LABEL_SEPARATOR = re.compile(r"[\s\-/]+")
_LABEL_INVALID = re.compile(r"[^a-z0-9_]")

# Patterns intentionally favor recall. This is an educational safety screen, not a
# claim that regexes can identify every kind of personal or regulated information.
_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "url",
        re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>]+"),
        "<URL>",
    ),
    (
        "email",
        re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
        "<EMAIL>",
    ),
    (
        "ipv4",
        re.compile(
            r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
        ),
        "<IP_ADDRESS>",
    ),
    (
        "ssn",
        re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        "<SSN>",
    ),
    (
        "payment_card",
        re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
        "<PAYMENT_CARD>",
    ),
    (
        "account_identifier",
        re.compile(
            r"(?i)\b(account|customer)\s+(?:number|no\.?|id)\s*[:#-]?\s*"
            r"[A-Z0-9][A-Z0-9-]{3,}\b"
        ),
        "<ACCOUNT_IDENTIFIER>",
    ),
    (
        "phone",
        re.compile(
            r"(?<![\w])(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)"
            r"\d{3}[ .-]?\d{4}(?!\w)"
        ),
        "<PHONE>",
    ),
)
SENSITIVE_PATTERN_NAMES = tuple(item[0] for item in _SENSITIVE_PATTERNS)

_PLACEHOLDER = re.compile(r"<[^>]+>|\{[^{}]+\}|\[[A-Z][A-Z0-9_ -]*\]", re.IGNORECASE)
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-" r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_TOKEN_WITH_DIGIT = re.compile(r"\b(?=\w*[0-9])\w+(?:[-_]\w+)*\b")
_QUOTED_VALUE = re.compile(r"(['\"])[^'\"]{1,80}\1")
_TEMPLATE_PUNCTUATION = re.compile(r"[^a-z0-9<>]+")


def normalize_whitespace(value: str) -> str:
    """Apply stable Unicode/control/whitespace normalization."""

    value = unicodedata.normalize("NFKC", value)
    value = _CONTROL_CHARACTERS.sub(" ", value)
    return _WHITESPACE.sub(" ", value).strip()


def mask_sensitive_text(value: str) -> tuple[str, dict[str, int]]:
    """Mask common sensitive patterns and return non-content-bearing counts."""

    normalized = normalize_whitespace(value)
    counts: Counter[str] = Counter()
    for name, pattern, replacement in _SENSITIVE_PATTERNS:
        normalized, count = pattern.subn(replacement, normalized)
        if count:
            counts[name] += count
    return normalize_whitespace(normalized), dict(sorted(counts.items()))


def normalize_label(value: str) -> str:
    """Convert source labels to a conservative lowercase identifier."""

    label = normalize_whitespace(value).lower()
    label = _LABEL_SEPARATOR.sub("_", label)
    label = _LABEL_INVALID.sub("", label)
    return re.sub(r"_+", "_", label).strip("_")


def canonical_text(value: str) -> str:
    """Normalize text for exact comparisons without removing semantics."""

    return normalize_whitespace(value).casefold()


def inferred_template(value: str) -> str:
    """Infer a conservative prompt template by replacing dynamic values."""

    template = canonical_text(value)
    template = _UUID.sub(" <slot> ", template)
    template = _PLACEHOLDER.sub(" <slot> ", template)
    template = _QUOTED_VALUE.sub(" <slot> ", template)
    template = _TOKEN_WITH_DIGIT.sub(" <slot> ", template)
    template = _TEMPLATE_PUNCTUATION.sub(" ", template)
    return normalize_whitespace(template)


def content_id(*, instruction: str, category: str, intent: str, response: str) -> str:
    """Build a stable content-derived ID that is independent of CSV row order."""

    payload = {
        "category": normalize_label(category),
        "instruction": canonical_text(instruction),
        "intent": normalize_label(intent),
        "response": canonical_text(response),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"bitext-{digest[:24]}"


def template_id(template: str) -> str:
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return f"template-{digest[:20]}"


def group_id(member_ids: list[str] | tuple[str, ...]) -> str:
    payload = "\n".join(sorted(member_ids))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"group-{digest[:20]}"


def stable_order_key(seed: int, *values: str) -> str:
    payload = ":".join((str(seed), *values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
