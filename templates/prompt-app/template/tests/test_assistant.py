"""Hermetic unit tests on aai_core.testing fakes — no cloud, no registry."""

from types import SimpleNamespace

import pytest

from aai_core.testing import FakeChatModel
from app.assistant import Assistant, PromptLimits
from app.config import PROMPT_NAME


class FakePrompt:
    def __init__(self, version=3):
        self.version = version

    def format(self, **values):
        return [
            {"role": "system", "content": "Be truthful."},
            {"role": "user", "content": values["question"]},
        ]


class FakePrompts:
    def __init__(self):
        self.loads = []

    def load(self, name, **kwargs):
        assert name == PROMPT_NAME
        self.loads.append(kwargs)
        return FakePrompt()


class FakeProviders:
    def __init__(self, model):
        self._model = model

    def model(self, name):
        assert name == "general-chat"
        return self._model


def _context(model, prompts, environment="dev"):
    return SimpleNamespace(
        providers=FakeProviders(model),
        prompts=prompts,
        settings=SimpleNamespace(resource=SimpleNamespace(environment=environment)),
    )


def test_ask_formats_prompt_and_returns_model_reply():
    model = FakeChatModel(reply="a concise answer")
    prompts = FakePrompts()

    answer = Assistant(_context(model, prompts)).ask("What is an alias?")

    assert answer == "a concise answer"
    assert prompts.loads == [{"alias": "development"}]
    assert model.requests[0]["messages"][-1]["content"] == "What is an alias?"
    assert model.requests[0]["max_tokens"] == 1024


def test_production_environment_loads_production_alias():
    prompts = FakePrompts()

    Assistant(_context(FakeChatModel(), prompts, environment="prod"))

    assert prompts.loads == [{"alias": "production"}]


def test_pinned_version_bypasses_aliases():
    prompts = FakePrompts()

    Assistant(_context(FakeChatModel(), prompts), prompt_version=3)

    assert prompts.loads == [{"version": 3}]


def test_empty_question_is_rejected():
    assistant = Assistant(_context(FakeChatModel(), FakePrompts()))

    with pytest.raises(ValueError):
        assistant.ask("   ")


def test_question_and_output_bounds_fail_closed():
    prompts = FakePrompts()
    model = FakeChatModel(reply="answer")
    assistant = Assistant(
        _context(model, prompts),
        limits=PromptLimits(max_question_chars=5),
    )
    with pytest.raises(ValueError, match="character bound"):
        assistant.ask("123456")
    assert model.requests == []

    oversized = Assistant(
        _context(FakeChatModel(reply="12345"), FakePrompts()),
        limits=PromptLimits(max_output_chars=4),
    )
    with pytest.raises(RuntimeError, match="response exceeded"):
        oversized.ask("hello")


def test_empty_model_output_is_rejected():
    assistant = Assistant(_context(FakeChatModel(reply=" "), FakePrompts()))

    with pytest.raises(RuntimeError, match="empty response"):
        assistant.ask("hello")
