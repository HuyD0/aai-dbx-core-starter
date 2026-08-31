from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from aai_core import tracing
from aai_core.providers import (
    AzureAISearchRetriever,
    AzureSemanticRankOptions,
    DatabricksRerankOptions,
    ModelCapabilities,
    OpenAICompatibleChatModel,
    UnsupportedCapabilityError,
)
from aai_core.providers.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from aai_core.providers.search import DatabricksAISearchRetriever
from aai_core.providers.types import (
    ProviderConfigurationError,
    ProviderRequestError,
)


@contextmanager
def _sdk_trace_state():
    """Activate bounded SDK instrumentation without configuring real MLflow."""

    state = tracing.TraceState(
        metadata={},
        policy=tracing.TracePolicy(),
        integration=tracing.TraceIntegration.SDK,
    )
    token = tracing._TRACE_STATE.set(state)
    try:
        yield
    finally:
        tracing._TRACE_STATE.reset(token)


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


class FakeEmbeddings:
    def create(self, **request):
        assert request["model"] == "embedding-deployment"
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])],
            usage=SimpleNamespace(prompt_tokens=11, total_tokens=11),
        )


class FakeEmbeddingOpenAI:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


class FakeAsyncOpenAI:
    pass


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


def test_openai_adapter_exposes_caller_owned_native_async_client():
    clients = []
    model = OpenAICompatibleChatModel(
        logical_name="general-chat",
        provider="databricks",
        model="chat-deployment",
        client=FakeOpenAI(),
        async_client_factory=lambda: clients.append(FakeAsyncOpenAI()) or clients[-1],
    )

    first = model.create_native_async_client()
    second = model.create_native_async_client()

    assert isinstance(first, FakeAsyncOpenAI)
    assert isinstance(second, FakeAsyncOpenAI)
    assert first is not second


def test_sync_generate_fails_before_provider_call_inside_event_loop():
    import asyncio

    model = OpenAICompatibleChatModel(
        logical_name="general-chat",
        provider="databricks",
        model="chat-deployment",
        client=FakeOpenAI(),
        async_client_factory=FakeAsyncOpenAI,
    )

    async def invoke():
        with pytest.raises(UnsupportedCapabilityError) as excinfo:
            model.generate([{"role": "user", "content": "question"}])
        assert "create_native_async_client" in str(excinfo.value.remediation)

    asyncio.run(invoke())


def test_openai_adapter_records_standard_llm_inputs_and_outputs(
    monkeypatch,
):
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

    with _sdk_trace_state():
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
    assert recorded["span"].attributes["mlflow.llm.provider"] == "databricks"
    assert recorded["span"].attributes["mlflow.llm.model"] == "chat-deployment"
    assert recorded["span"].attributes["mlflow.message.format"] == "openai"
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
        provider="azure_apim",
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


@pytest.mark.parametrize(
    "provider_options",
    [
        {"search_text": "ungoverned query"},
        {"filter": "region eq 'other'"},
        {"top": 10_000},
        {"vector_queries": []},
        {"select": ["id"]},
    ],
)
def test_azure_search_rejects_controlled_provider_options_before_call(
    provider_options,
):
    calls = []
    client = SimpleNamespace(search=lambda **options: calls.append(options) or [])
    retriever = AzureAISearchRetriever(
        logical_name="knowledge",
        client=client,
        content_field="content",
        id_field="id",
    )

    with pytest.raises(ProviderConfigurationError, match="controlled search fields"):
        retriever.search(
            "governed query",
            mode="text",
            provider_options=provider_options,
        )

    assert calls == []


@pytest.mark.parametrize(
    "provider_options",
    [
        {"api_key": "must-not-be-forwarded"},
        {"headers": {"x-client": "must-not-be-forwarded"}},
        {"transport": {"authorizationHeader": "must-not-be-forwarded"}},
    ],
)
def test_azure_search_rejects_credential_bearing_provider_options(
    provider_options,
):
    calls = []
    client = SimpleNamespace(search=lambda **options: calls.append(options) or [])
    retriever = AzureAISearchRetriever(
        logical_name="knowledge",
        client=client,
        content_field="content",
        id_field="id",
    )

    with pytest.raises(ProviderConfigurationError, match="credentials, headers"):
        retriever.search(
            "governed query",
            mode="text",
            provider_options=provider_options,
        )

    assert calls == []


@pytest.mark.parametrize(
    "provider_options",
    [
        {1: "not-a-keyword"},
        {"body": {"filter": "ungoverned"}},
        {"params": {"top": 10_000}},
        {"transport": {"timeout": 300}},
        {"request_options": {"retry_policy": "unbounded"}},
        {"proxies": {"https": "https://unreviewed.invalid"}},
    ],
)
def test_search_rejects_raw_transport_options_before_embedding_or_call(
    provider_options,
):
    calls = []
    embedding = FakeEmbedding()
    client = SimpleNamespace(search=lambda **options: calls.append(options) or [])
    retriever = AzureAISearchRetriever(
        logical_name="knowledge",
        client=client,
        content_field="content",
        id_field="id",
        vector_fields=["content_vector"],
        embedding_provider=embedding,
    )

    with pytest.raises(ProviderConfigurationError):
        retriever.search("question", provider_options=provider_options)

    assert embedding.queries == []
    assert calls == []


def test_azure_search_forwards_additive_provider_options():
    calls = []
    client = SimpleNamespace(search=lambda **options: calls.append(options) or [])
    retriever = AzureAISearchRetriever(
        logical_name="knowledge",
        client=client,
        content_field="content",
        id_field="id",
    )

    retriever.search(
        "question",
        mode="text",
        provider_options={"include_total_count": True},
    )

    assert calls[0]["include_total_count"] is True


def test_azure_semantic_ranking_uses_typed_query_controls():
    calls = []
    client = SimpleNamespace(search=lambda **options: calls.append(options) or [])
    retriever = AzureAISearchRetriever(
        logical_name="knowledge",
        client=client,
        content_field="content",
        id_field="id",
    )

    retriever.search(
        "base query",
        mode="text",
        ranking=AzureSemanticRankOptions(
            semantic_configuration_name="knowledge-semantic",
            semantic_query="ranking query",
            error_mode="partial",
            max_wait_milliseconds=750,
        ),
    )

    assert calls[0]["search_text"] == "base query"
    assert calls[0]["semantic_query"] == "ranking query"
    assert calls[0]["semantic_configuration_name"] == "knowledge-semantic"
    assert calls[0]["semantic_error_mode"] == "partial"
    assert calls[0]["semantic_max_wait_in_milliseconds"] == 750


def test_azure_search_rejects_the_other_providers_ranking_before_call():
    calls = []
    client = SimpleNamespace(search=lambda **options: calls.append(options) or [])
    retriever = AzureAISearchRetriever(
        logical_name="knowledge",
        client=client,
        content_field="content",
        id_field="id",
    )

    with pytest.raises(UnsupportedCapabilityError, match="AzureSemanticRankOptions"):
        retriever.search(
            "question",
            mode="text",
            ranking=DatabricksRerankOptions(columns_to_rerank=("content",)),
        )

    assert calls == []


class _ProviderFailure(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}; credential=native-secret")
        self.status_code = status_code


def test_adapter_translates_final_native_failure_without_retrying():
    attempts = []

    class FailingCompletions:
        def create(self, **request):
            attempts.append(request)
            raise _ProviderFailure(429)

    client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    model = OpenAICompatibleChatModel(
        logical_name="general-chat",
        provider="databricks",
        model="chat-deployment",
        client=client,
    )

    with pytest.raises(ProviderRequestError) as excinfo:
        model.generate([{"role": "user", "content": "question"}])

    assert len(attempts) == 1
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert excinfo.value.__suppress_context__
    assert excinfo.value.provider == "databricks"
    assert excinfo.value.operation == "chat_completion"
    assert excinfo.value.status_code == 429
    assert "native-secret" not in str(excinfo.value)
    assert "rate limit" in str(excinfo.value.remediation)


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


def test_databricks_text_search_uses_full_text_without_embedding():
    index = FakeDatabricksIndex()
    embedding = FakeEmbedding()
    retriever = DatabricksAISearchRetriever(
        logical_name="knowledge",
        index=index,
        columns=["id", "content"],
        content_field="content",
        id_field="id",
        embedding_provider=embedding,
    )

    retriever.search("exact product identifier", mode="text")

    assert index.options["query_type"] == "FULL_TEXT"
    assert index.options["query_text"] == "exact product identifier"
    assert "query_vector" not in index.options
    assert embedding.queries == []


def test_databricks_hybrid_reranker_uses_governed_columns(monkeypatch):
    from conftest import install_fake_module

    class NativeReranker:
        def __init__(self, columns_to_rerank):
            self.columns_to_rerank = columns_to_rerank

    install_fake_module(
        monkeypatch,
        "databricks.ai_search.reranker",
        DatabricksReranker=NativeReranker,
    )
    index = FakeDatabricksIndex()
    retriever = DatabricksAISearchRetriever(
        logical_name="knowledge",
        index=index,
        columns=["id", "content", "summary"],
        content_field="content",
        id_field="id",
    )

    retriever.search(
        "question",
        ranking=DatabricksRerankOptions(columns_to_rerank=("content", "summary")),
    )

    assert index.options["reranker"].columns_to_rerank == ["content", "summary"]


def test_databricks_reranker_rejects_non_hybrid_and_unreviewed_columns():
    index = FakeDatabricksIndex()
    retriever = DatabricksAISearchRetriever(
        logical_name="knowledge",
        index=index,
        columns=["id", "content"],
        content_field="content",
        id_field="id",
    )

    with pytest.raises(UnsupportedCapabilityError, match="only with hybrid"):
        retriever.search(
            "question",
            mode="text",
            ranking=DatabricksRerankOptions(columns_to_rerank=("content",)),
        )
    with pytest.raises(ProviderConfigurationError, match="governed columns"):
        retriever.search(
            "question",
            ranking=DatabricksRerankOptions(columns_to_rerank=("private",)),
        )

    assert index.options is None


@pytest.mark.parametrize(
    "provider_options",
    [
        {"query_text": "ungoverned query"},
        {"filters": {"region": "other"}},
        {"num_results": 10_000},
        {"query_vector": [9.0]},
        {"columns": ["secret_column"]},
        {"query_type": "FULL_TEXT"},
    ],
)
def test_databricks_search_rejects_controlled_provider_options_before_call(
    provider_options,
):
    index = FakeDatabricksIndex()
    retriever = DatabricksAISearchRetriever(
        logical_name="knowledge",
        index=index,
        columns=["id", "content"],
        content_field="content",
        id_field="id",
    )

    with pytest.raises(ProviderConfigurationError, match="controlled search fields"):
        retriever.search(
            "governed query",
            mode="text",
            provider_options=provider_options,
        )

    assert index.options is None


def test_databricks_search_rejects_nested_credential_provider_options():
    index = FakeDatabricksIndex()
    retriever = DatabricksAISearchRetriever(
        logical_name="knowledge",
        index=index,
        columns=["id", "content"],
        content_field="content",
        id_field="id",
    )

    with pytest.raises(ProviderConfigurationError, match="credentials, headers"):
        retriever.search(
            "governed query",
            mode="text",
            provider_options={"transport": [{"clientSecret": "unsafe"}]},
        )

    assert index.options is None


def test_retriever_records_documents_on_the_retriever_span(monkeypatch):
    """Groundedness judges read documents from RETRIEVER span outputs."""
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
    with _sdk_trace_state():
        retriever.search("question", mode="text", filters={"region": "ca"})

    span = recorded["span"]
    assert recorded["span_type"] == "RETRIEVER"
    assert span.inputs == {"query": "question"}
    assert span.outputs[0]["page_content"] == "grounding"
    assert span.outputs[0]["metadata"]["doc_uri"] == "https://example/doc"


def test_embedding_span_records_billed_tokens_outside_the_chat_aggregate(
    monkeypatch,
):
    """Embedding usage is evidence, but it is not chat-model spend.

    MLflow folds ``mlflow.chat.tokenUsage`` from every span type into the
    authoritative trace-level total, which agentkit prices at the agent
    model's configured rate. The embedding side is billed differently, so
    it is recorded on the OpenTelemetry GenAI attribute instead.
    """

    from conftest import install_fake_module

    class FakeSpan:
        def __init__(self):
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

    recorded = {}

    @contextmanager
    def start_span(name, span_type):
        span = FakeSpan()
        recorded.update(name=name, span_type=span_type, span=span)
        yield span

    install_fake_module(monkeypatch, "mlflow", start_span=start_span)
    provider = OpenAICompatibleEmbeddingProvider(
        logical_name="general-embedding",
        provider="databricks",
        model="embedding-deployment",
        client=FakeEmbeddingOpenAI(),
    )

    with _sdk_trace_state():
        vectors = provider.embed_documents(["grounding text"])

    assert vectors == [[0.1, 0.2]]
    assert recorded["span_type"] == "EMBEDDING"
    assert recorded["span"].attributes["gen_ai.usage.input_tokens"] == 11
    assert "mlflow.chat.tokenUsage" not in recorded["span"].attributes


def test_embedding_span_omits_usage_a_provider_does_not_report(monkeypatch):
    from conftest import install_fake_module

    class FakeSpan:
        def __init__(self):
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

    recorded = {}

    @contextmanager
    def start_span(name, span_type):
        span = FakeSpan()
        recorded.update(span=span)
        yield span

    class UsagelessEmbeddings:
        def create(self, **request):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.3])])

    install_fake_module(monkeypatch, "mlflow", start_span=start_span)
    provider = OpenAICompatibleEmbeddingProvider(
        logical_name="general-embedding",
        provider="databricks",
        model="embedding-deployment",
        client=SimpleNamespace(embeddings=UsagelessEmbeddings()),
    )

    with _sdk_trace_state():
        provider.embed_documents(["grounding text"])

    assert "gen_ai.usage.input_tokens" not in recorded["span"].attributes
