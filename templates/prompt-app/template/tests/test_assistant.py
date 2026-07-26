"""Hermetic unit tests on aai_core.testing fakes — no cloud, no registry."""

from types import SimpleNamespace

import pytest

from aai_core.testing import FakeChatModel
from app.assistant import Assistant
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
