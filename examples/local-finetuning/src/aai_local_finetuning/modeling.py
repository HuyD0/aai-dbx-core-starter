"""Lazy local MLX-LM inference with no remote model identifiers."""

from __future__ import annotations

import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PromptStrategy = Literal["basic", "strong", "few_shot"]


SYSTEM_PROMPT = (
    "Classify the customer request. Return one JSON object only with exactly "
    "intent, category, requires_escalation, and response. Do not use markdown."
)


@dataclass(frozen=True)
class LocalGeneration:
    """Model output and bounded local performance evidence."""

    text: str
    latency_ms: float
    output_tokens: int
    peak_memory_mb: float


def build_messages(
    utterance: str,
    *,
    strategy: PromptStrategy,
    allowed_intents: list[str],
    category_by_intent: dict[str, str] | None = None,
    few_shot: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[dict[str, str]]:
    """Build progressively stronger prompts while holding model/data fixed."""

    if strategy == "basic":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": utterance},
        ]
    contract = {
        "intent": "one allowed intent",
        "category": "lowercase broad category",
        "requires_escalation": False,
        "response": "brief safe support response",
    }
    system = (
        f"{SYSTEM_PROMPT}\nAllowed intents: {', '.join(allowed_intents)}.\n"
        f"Required shape: {json.dumps(contract, separators=(',', ':'))}\n"
        "Use requires_escalation=true for a complaint, payment failure, "
        "registration failure, or an explicit request for a human. Never invent "
        "an order number, credential, refund, or completed action."
    )
    if category_by_intent:
        system += "\nUse this train-derived intent-to-category mapping: " + json.dumps(
            category_by_intent, separators=(",", ":"), sort_keys=True
        )
    messages = [{"role": "system", "content": system}]
    if strategy == "few_shot":
        for example_input, example_output in few_shot or []:
            messages.extend(
                [
                    {"role": "user", "content": example_input},
                    {
                        "role": "assistant",
                        "content": json.dumps(example_output, separators=(",", ":")),
                    },
                ]
            )
    messages.append({"role": "user", "content": utterance})
    return messages


class LocalMLXPredictor:
    """Load one exact local model, optionally with a local LoRA adapter."""

    def __init__(self, model_path: Path, adapter_path: Path | None = None):
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"local model directory does not exist: {model_path}"
            )
        from mlx_lm import load

        kwargs = {"adapter_path": str(adapter_path)} if adapter_path else {}
        self._model, self._tokenizer = load(str(model_path), **kwargs)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 160,
    ) -> LocalGeneration:
        """Generate once and measure wall time, output tokens, and process peak RSS."""

        from mlx_lm import generate

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        started = time.perf_counter()
        text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024 * 1024
        )
        return LocalGeneration(
            text=text,
            latency_ms=latency_ms,
            output_tokens=len(token_ids),
            peak_memory_mb=peak_memory_mb,
        )
