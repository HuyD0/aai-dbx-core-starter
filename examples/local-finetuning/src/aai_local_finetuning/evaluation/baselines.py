"""Deterministic baselines for the support-intent task.

Label statistics are fitted on train.  The keyword baseline also applies the
explicit human-authored phrase and escalation rules in this module; callers and
teaching material must not describe the whole method as train-derived.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Self

from .models import EvaluationRecord, Prediction, SupportOutput

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "could",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "please",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "with",
    "you",
}
_ESCALATION_TRIGGERS = (
    "fraud",
    "stolen",
    "unauthorized",
    "legal action",
    "identity theft",
    "speak to a human",
    "speak to an agent",
)
_DEFAULT_PHRASE_RULES: dict[str, tuple[str, ...]] = {
    "recover_password": ("forgot password", "reset password", "cannot log in"),
    "card_arrival": ("card arrive", "card delivery", "where is my card"),
    "cash_withdrawal": ("cash withdrawal", "cash machine", "atm"),
    "cancel_transfer": ("cancel transfer", "stop transfer"),
    "cash_withdrawal_charge": ("cash withdrawal fee", "atm fee"),
    "cash_withdrawal_wrong_exchange_rate": ("withdrawal exchange rate",),
    "cash_withdrawal_not_recognised": ("cash withdrawal not mine",),
    "cash_withdrawal_amount_too_large": ("withdrawal limit",),
    "cash_withdrawal_or_transfer_pending": ("withdrawal pending",),
    "compromised_card": ("card stolen", "card compromised"),
    "card_payment_not_recognised": ("card payment not mine",),
    "declined_card_payment": ("card declined", "payment declined"),
    "refund_not_showing_up": ("refund missing", "refund not showing"),
}


class MajorityBaseline:
    """Always predict the train split's most common intent."""

    name = "majority"
    meaningful = False

    def __init__(
        self,
        *,
        intent: str,
        category: str,
        requires_escalation: bool,
        supported_intents: tuple[str, ...],
        training_example_ids: frozenset[str],
    ) -> None:
        self.intent = intent
        self.category = category
        self.requires_escalation = requires_escalation
        self.supported_intents = supported_intents
        self.training_example_ids = training_example_ids

    @classmethod
    def fit(cls, train_records: Sequence[EvaluationRecord]) -> Self:
        """Fit using an explicitly supplied train split and deterministic ties."""

        _validate_training_records(train_records)
        intent = _mode(record.target.intent for record in train_records)
        intent_records = [
            record for record in train_records if record.target.intent == intent
        ]
        return cls(
            intent=intent,
            category=_mode(record.target.category for record in intent_records),
            requires_escalation=_bool_mode(
                record.target.requires_escalation for record in intent_records
            ),
            supported_intents=tuple(
                sorted({record.target.intent for record in train_records})
            ),
            training_example_ids=frozenset(
                record.example_id for record in train_records
            ),
        )

    def predict(self, record: EvaluationRecord) -> Prediction:
        start = time.perf_counter()
        output = SupportOutput(
            intent=self.intent,
            category=self.category,
            requires_escalation=self.requires_escalation,
            response=_response_for(self.intent, self.requires_escalation),
        )
        raw_text = _serialize(output)
        return _prediction(record.example_id, raw_text, start)

    def predict_many(
        self, records: Sequence[EvaluationRecord]
    ) -> tuple[Prediction, ...]:
        return tuple(self.predict(record) for record in records)


class KeywordRuleBaseline:
    """Train-derived token log-odds plus transparent support-domain rules."""

    name = "keyword_rule"
    meaningful = True

    def __init__(
        self,
        *,
        labels: tuple[str, ...],
        label_counts: Mapping[str, int],
        term_counts: Mapping[str, Mapping[str, int]],
        global_term_counts: Mapping[str, int],
        categories: Mapping[str, str],
        escalation_by_intent: Mapping[str, bool],
        phrase_rules: Mapping[str, tuple[str, ...]],
        training_example_ids: frozenset[str],
    ) -> None:
        self.supported_intents = labels
        self.label_counts = dict(label_counts)
        self.term_counts = {
            label: dict(counts) for label, counts in term_counts.items()
        }
        self.global_term_counts = dict(global_term_counts)
        self.categories = dict(categories)
        self.escalation_by_intent = dict(escalation_by_intent)
        self.phrase_rules = dict(phrase_rules)
        self.training_example_ids = training_example_ids
        self._training_size = sum(self.label_counts.values())

    @classmethod
    def fit(
        cls,
        train_records: Sequence[EvaluationRecord],
        *,
        phrase_rules: Mapping[str, Sequence[str]] | None = None,
    ) -> Self:
        """Learn label statistics from train and attach the supplied phrase rules."""

        _validate_training_records(train_records)
        labels = tuple(sorted({record.target.intent for record in train_records}))
        label_counts: Counter[str] = Counter()
        term_counts: dict[str, Counter[str]] = defaultdict(Counter)
        global_term_counts: Counter[str] = Counter()
        by_intent: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for record in train_records:
            label = record.target.intent
            label_counts[label] += 1
            by_intent[label].append(record)
            terms = set(_features(record.input_text))
            term_counts[label].update(terms)
            global_term_counts.update(terms)

        configured = phrase_rules or _DEFAULT_PHRASE_RULES
        fitted_rules: dict[str, tuple[str, ...]] = {}
        for label in labels:
            normalized = tuple(
                sorted(
                    {
                        phrase.lower().strip()
                        for phrase in configured.get(label, ())
                        if isinstance(phrase, str) and phrase.strip()
                    }
                )
            )
            if normalized:
                fitted_rules[label] = normalized

        return cls(
            labels=labels,
            label_counts=label_counts,
            term_counts=term_counts,
            global_term_counts=global_term_counts,
            categories={
                label: _mode(record.target.category for record in records)
                for label, records in by_intent.items()
            },
            escalation_by_intent={
                label: _bool_mode(
                    record.target.requires_escalation for record in records
                )
                for label, records in by_intent.items()
            },
            phrase_rules=fitted_rules,
            training_example_ids=frozenset(
                record.example_id for record in train_records
            ),
        )

    @property
    def keywords_by_intent(self) -> dict[str, tuple[str, ...]]:
        """Expose the strongest learned terms for inspection and teaching."""

        result: dict[str, tuple[str, ...]] = {}
        for label in self.supported_intents:
            ranked = sorted(
                self.term_counts[label],
                key=lambda term: (-self._term_weight(term, label), term),
            )
            result[label] = tuple(
                term for term in ranked if self._term_weight(term, label) > 0
            )[:10]
        return result

    def predict(self, record: EvaluationRecord) -> Prediction:
        start = time.perf_counter()
        intent = self.predict_intent(record.input_text)
        normalized = record.input_text.lower()
        escalation = self.escalation_by_intent[intent] or any(
            trigger in normalized for trigger in _ESCALATION_TRIGGERS
        )
        output = SupportOutput(
            intent=intent,
            category=self.categories[intent],
            requires_escalation=escalation,
            response=_response_for(intent, escalation),
        )
        return _prediction(record.example_id, _serialize(output), start)

    def predict_intent(self, text: str) -> str:
        normalized = text.lower()
        terms = set(_features(text))
        label_count = len(self.supported_intents)
        scores: dict[str, float] = {}
        for label in self.supported_intents:
            score = math.log(
                (self.label_counts[label] + 1) / (self._training_size + label_count)
            )
            score += sum(self._term_weight(term, label) for term in terms)
            score += 8.0 * sum(
                1 for phrase in self.phrase_rules.get(label, ()) if phrase in normalized
            )
            scores[label] = score
        return max(
            self.supported_intents,
            key=lambda label: (scores[label], -self.supported_intents.index(label)),
        )

    def predict_many(
        self, records: Sequence[EvaluationRecord]
    ) -> tuple[Prediction, ...]:
        return tuple(self.predict(record) for record in records)

    def _term_weight(self, term: str, label: str) -> float:
        positive = self.term_counts[label].get(term, 0)
        negative = self.global_term_counts.get(term, 0) - positive
        positive_total = self.label_counts[label]
        negative_total = self._training_size - positive_total
        positive_rate = (positive + 1.0) / (positive_total + 2.0)
        negative_rate = (negative + 1.0) / (negative_total + 2.0)
        return math.log(positive_rate / negative_rate)


def _validate_training_records(records: Sequence[EvaluationRecord]) -> None:
    if not records:
        raise ValueError("train_records must not be empty")
    identifiers = [record.example_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("train_records contains duplicate example_id values")


def _features(text: str) -> tuple[str, ...]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    unigrams = [token for token in tokens if token not in _STOP_WORDS]
    bigrams = [
        f"{left}_{right}"
        for left, right in zip(tokens, tokens[1:], strict=False)
        if left not in _STOP_WORDS or right not in _STOP_WORDS
    ]
    return tuple(unigrams + bigrams)


def _mode(values) -> str:
    counts = Counter(values)
    if not counts:
        raise ValueError("cannot find a mode for an empty collection")
    return min(counts, key=lambda value: (-counts[value], value))


def _bool_mode(values) -> bool:
    counts = Counter(values)
    if not counts:
        raise ValueError("cannot find a mode for an empty collection")
    return counts[True] > counts[False]


def _response_for(intent: str, escalation: bool) -> str:
    if escalation:
        return "I'll route this request to a support specialist for review."
    topic = intent.replace("_", " ")
    return f"I can help with your {topic} request."


def _serialize(output: SupportOutput) -> str:
    return json.dumps(
        output.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _prediction(example_id: str, raw_text: str, start: float) -> Prediction:
    return Prediction(
        example_id=example_id,
        raw_text=raw_text,
        latency_ms=float((time.perf_counter() - start) * 1000.0),
        output_tokens=len(re.findall(r"\w+|[^\w\s]", raw_text)),
        peak_memory_mb=0.0,
    )
