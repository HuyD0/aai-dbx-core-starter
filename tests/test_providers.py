from types import SimpleNamespace

import pytest

from aai_core.exceptions import AaiCoreError
from aai_core.providers import (
    AzureAISearchRetriever,
    ModelCapabilities,
    OpenAICompatibleChatModel,
    UnsupportedCapabilityError,
)
from aai_core.providers.openai_compatible import call_with_resilience
from aai_core.providers.search import DatabricksAISearchRetriever
from aai_core.providers.types import (
    ProviderConfigurationError,
    ProviderError,
    ProviderRequestError,
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


def test_openai_adapter_records_standard_llm_inputs_and_outputs(
    monkeypatch,
):
    from contextlib import contextmanager

    from conftest import install_fake_module

    class FakeSpan:
        def __init__(self):
            self.inputs = None
            self.outputs = None
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_inputs(self, inputs):
            self.inputs = inputs

        def set_outputs(self, outputs):
            self.outputs = outputs

    recorded = {}

    @contextmanager
    def start_span(name, span_type):
        span = FakeSpan()
        recorded.update(name=name, span_type=span_type, span=span)
        yield span

    install_fake_module(monkeypatch, "mlflow", start_span=start_span)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_order_status",
                "parameters": {"type": "object"},
            },
        }
    ]
    model = OpenAICompatibleChatModel(
        logical_name="general-chat",
        provider="databricks",
        model="chat-deployment",
        client=FakeOpenAI(),
    )

    model.generate(
        [{"role": "user", "content": "Where is A-1001?"}],
        tools=tools,
        provider_options={"seed": 7},
    )

    assert recorded["name"] == "model.generate"
    assert recorded["span_type"] == "LLM"
    assert recorded["span"].inputs == {
        "messages": [{"role": "user", "content": "Where is A-1001?"}],
        "tools": tools,
    }
    assert recorded["span"].outputs == {"content": "answer"}
    assert recorded["span"].attributes["mlflow.chat.tokenUsage"] == {
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }


@pytest.mark.parametrize(
    "provider_options",
    [
        {"messages": [{"role": "user", "content": "hidden override"}]},
        {"model": "hidden-model"},
        {"tools": []},
        {"stream": True},
        {"extra_headers": {"authorization": "sensitive"}},
        {"extra_body": {"model": "hidden-model"}},
        {"extra_body": {"messages": [{"role": "user", "content": "hidden"}]}},
    ],
)
def test_openai_adapter_rejects_trace_bypasses_and_per_call_headers(
    provider_options,
):
    model = OpenAICompatibleChatModel(
        logical_name="general-chat",
        provider="databricks",
        model="chat-deployment",
        client=FakeOpenAI(),
    )

    with pytest.raises(ProviderConfigurationError):
        model.generate(
            [{"role": "user", "content": "governed request"}],
            provider_options=provider_options,
        )


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


class _RateLimitError(Exception):
    def __init__(self, status_code, retry_after=None):
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        headers = {"retry-after": str(retry_after)} if retry_after else {}
        self.response = SimpleNamespace(headers=headers)


def test_resilience_retries_429_honoring_retry_after():
    delays = []
    attempts = {"count": 0}

    def operation():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise _RateLimitError(429, retry_after=1.5)
        return "ok"

    result = call_with_resilience(
        operation,
        description="chat completion",
        provider="azure_apim",
        logical_name="general-chat",
        sleep=delays.append,
    )

    assert result == "ok"
    assert delays == [1.5]


def test_resilience_wraps_exhausted_retries_with_remediation():
    def operation():
        raise _RateLimitError(429)

    with pytest.raises(ProviderRequestError) as excinfo:
        call_with_resilience(
            operation,
            description="chat completion",
            provider="databricks",
            logical_name="general-chat",
            max_retries=1,
            sleep=lambda _: None,
        )

    assert isinstance(excinfo.value, ProviderError)
    assert isinstance(excinfo.value, AaiCoreError)
    assert isinstance(excinfo.value.__cause__, _RateLimitError)
    assert "rate limit" in str(excinfo.value.remediation)


def test_resilience_does_not_retry_authorization_failures():
    attempts = {"count": 0}

    def operation():
        attempts["count"] += 1
        raise _RateLimitError(403)

    with pytest.raises(ProviderRequestError) as excinfo:
        call_with_resilience(
            operation,
            description="chat completion",
            provider="databricks",
            logical_name="general-chat",
            sleep=lambda _: pytest.fail("must not sleep on 403"),
        )

    assert attempts["count"] == 1
    assert "CAN_QUERY" in str(excinfo.value.remediation)


class FakeEmbedding:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2]


def test_azure_search_embeds_query_when_vector_missing(monkeypatch):
    from conftest import install_fake_module

    class FakeVectorizedQuery:
        def __init__(self, *, vector, k_nearest_neighbors, fields):
            self.vector = vector
            self.k_nearest_neighbors = k_nearest_neighbors
            self.fields = fields

    install_fake_module(
        monkeypatch,
        "azure.search.documents.models",
        VectorizedQuery=FakeVectorizedQuery,
    )

    class VectorAwareSearch:
        def __init__(self):
            self.options = None

        def search(self, **options):
            self.options = options
            return []

    client = VectorAwareSearch()
    embedding = FakeEmbedding()
    retriever = AzureAISearchRetriever(
        logical_name="knowledge",
        client=client,
        content_field="content",
        id_field="id",
        vector_fields=["content_vector"],
        embedding_provider=embedding,
    )

    retriever.search("question")  # hybrid default, no query_vector

    assert embedding.queries == ["question"]
    assert client.options["vector_queries"][0].vector == [0.1, 0.2]


def test_azure_search_without_vector_or_embedding_says_how_to_fix():
    retriever = AzureAISearchRetriever(
        logical_name="knowledge",
        client=FakeSearch(),
        content_field="content",
        id_field="id",
        vector_fields=["content_vector"],
    )

    with pytest.raises(ProviderConfigurationError) as excinfo:
        retriever.search("question")

    assert "embedding" in str(excinfo.value.remediation)


class FakeDatabricksIndex:
    def __init__(self):
        self.options = None

    def similarity_search(self, **options):
        self.options = options
        return {
            "manifest": {"columns": [{"name": "id"}, {"name": "content"}]},
            "result": {"data_array": [["doc-1", "grounding"]]},
        }


def test_databricks_search_validates_mode_and_requires_vector_for_vector_mode():
    retriever = DatabricksAISearchRetriever(
        logical_name="knowledge",
        index=FakeDatabricksIndex(),
        columns=["id", "content"],
        content_field="content",
        id_field="id",
    )

    with pytest.raises(ValueError):
        retriever.search("question", mode="semantic")
    with pytest.raises(ProviderConfigurationError):
        retriever.search("question", mode="vector")


def test_databricks_search_hybrid_sends_text_and_optional_vector():
    index = FakeDatabricksIndex()
    retriever = DatabricksAISearchRetriever(
        logical_name="knowledge",
        index=index,
        columns=["id", "content"],
        content_field="content",
        id_field="id",
        embedding_provider=FakeEmbedding(),
    )

    results = retriever.search("question")  # hybrid default

    assert index.options["query_type"] == "HYBRID"
    assert index.options["query_text"] == "question"
    assert index.options["query_vector"] == [0.1, 0.2]
    assert results[0].document_id == "doc-1"


def test_retriever_records_documents_on_the_retriever_span(monkeypatch):
    """Groundedness judges read documents from RETRIEVER span outputs."""
    from contextlib import contextmanager

    from conftest import install_fake_module

    class FakeSpan:
        def __init__(self):
            self.attributes = {}
            self.inputs = None
            self.outputs = None

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_inputs(self, inputs):
            self.inputs = inputs

        def set_outputs(self, outputs):
            self.outputs = outputs

    recorded = {}

    @contextmanager
    def start_span(name, span_type):
        span = FakeSpan()
        recorded["span"] = span
        recorded["span_type"] = span_type
        yield span

    install_fake_module(monkeypatch, "mlflow", start_span=start_span)

    retriever = AzureAISearchRetriever(
        logical_name="knowledge",
        client=FakeSearch(),
        content_field="content",
        id_field="id",
        source_uri_field="source_uri",
        chunk_id_field="chunk_id",
    )
    retriever.search("question", mode="text", filters={"region": "ca"})

    span = recorded["span"]
    assert recorded["span_type"] == "RETRIEVER"
    assert span.inputs == {"query": "question"}
    assert span.outputs[0]["page_content"] == "grounding"
    assert span.outputs[0]["metadata"]["doc_uri"] == "https://example/doc"
