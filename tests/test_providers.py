from types import SimpleNamespace

import pytest

from aai_core.providers import (
    AzureAISearchRetriever,
    ModelCapabilities,
    OpenAICompatibleChatModel,
    UnsupportedCapabilityError,
)


class FakeCompletions:
    def create(self, **request):
        assert request["model"] == "chat-deployment"
        return SimpleNamespace(
            model="resolved-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", tool_calls=None)
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=4,
                completion_tokens=2,
                total_tokens=6,
            ),
        )


class FakeOpenAI:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


class FakeSearch:
    def search(self, **options):
        assert options["filter"] == "region eq 'ca'"
        return [
            {
                "id": "doc-1",
                "content": "grounding",
                "source_uri": "https://example/doc",
                "chunk_id": "chunk-1",
                "region": "ca",
                "@search.score": 0.9,
            }
        ]


def test_openai_adapter_normalizes_response():
    model = OpenAICompatibleChatModel(
        logical_name="general-chat",
        provider="databricks",
        model="chat-deployment",
        client=FakeOpenAI(),
        capabilities=ModelCapabilities(structured_output=True),
    )

    response = model.generate([{"role": "user", "content": "question"}])

    assert response.content == "answer"
    assert response.model == "resolved-model"
    assert response.usage["total_tokens"] == 6


def test_model_capabilities_fail_before_provider_call():
    model = OpenAICompatibleChatModel(
        logical_name="limited",
        provider="foundry",
        model="deployment",
        client=FakeOpenAI(),
        capabilities=ModelCapabilities(tool_calling=False),
    )

    with pytest.raises(UnsupportedCapabilityError):
        model.generate(
            [{"role": "user", "content": "question"}],
            tools=[{"type": "function"}],
        )


def test_azure_search_normalizes_mlflow_document():
    retriever = AzureAISearchRetriever(
        logical_name="knowledge",
        client=FakeSearch(),
        content_field="content",
        id_field="id",
        source_uri_field="source_uri",
        chunk_id_field="chunk_id",
    )

    result = retriever.search(
        "question",
        mode="text",
        filters={"region": "ca"},
    )[0]

    assert result.provider == "azure_ai_search"
    assert result.metadata == {"region": "ca"}
    assert result.as_mlflow_document()["metadata"]["doc_uri"] == ("https://example/doc")
