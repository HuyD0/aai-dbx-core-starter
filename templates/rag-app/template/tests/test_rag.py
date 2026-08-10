from types import SimpleNamespace

import pytest

from aai_core.agents import AgentRequest
from aai_core.providers import ModelResponse, SearchResult
from app.rag import RAGAgent, RAGLimits, rag_limit_parameters


class FakeModel:
    def __init__(self):
        self.requests = []

    def generate(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        return ModelResponse(
            content="grounded answer",
            provider="test",
            logical_name="general-chat",
            model="fake",
            latency_ms=1,
        )


class FakeEmbedding:
    dimensions = 2

    def __init__(self, vector=(0.1, 0.2)):
        self.vector = list(vector)
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return self.vector


class FakeRetriever:
    def __init__(self, results=None, provider="databricks_ai_search"):
        self.provider = provider
        self.results = (
            list(results)
            if results is not None
            else [_result("doc-1", "grounding", "chunk-1")]
        )
        self.requests = []

    def search(self, query, **kwargs):
        self.requests.append((query, kwargs))
        return list(self.results)


class FakePrompt:
    def format(self, **values):
        return [
            {"role": "system", "content": "Use supplied evidence."},
            {
                "role": "user",
                "content": f"{values['question']} {values['context']}",
            },
        ]


class FakePrompts:
    def load(self, name, **kwargs):
        assert name == "agent-system"
        assert kwargs["alias"] == "development"
        return FakePrompt()


class FakeProviders:
    def __init__(self, model, embedding, retriever):
        self._model = model
        self._embedding = embedding
        self._retriever = retriever

    def model(self, name):
        return self._model

    def embedding(self, name):
        return self._embedding

    def retriever(self, name):
        return self._retriever


def _result(document_id, content, chunk_id, source_uri="https://example/doc"):
    return SearchResult(
        document_id=document_id,
        content=content,
        score=1.0,
        source_uri=source_uri,
        chunk_id=chunk_id,
    )


def _agent(*, results=None, provider="databricks_ai_search", filters=None, limits=None):
    model = FakeModel()
    embedding = FakeEmbedding()
    retriever = FakeRetriever(results, provider)
    context = SimpleNamespace(
        providers=FakeProviders(model, embedding, retriever),
        prompts=FakePrompts(),
        settings=SimpleNamespace(resource=SimpleNamespace(environment="dev")),
    )
    agent = RAGAgent(
        context,
        retrieval_filters=filters,
        limits=limits or RAGLimits(),
    )
    return agent, model, embedding, retriever


def _request(question="question"):
    return AgentRequest(messages=[{"role": "user", "content": question}])


def test_agent_returns_grounded_answer_and_citation_with_bounded_generation():
    agent, model, _, retriever = _agent()

    response = agent.invoke(_request())

    assert response.content == "grounded answer"
    assert response.citations[0]["chunk_id"] == "chunk-1"
    assert retriever.requests[0][1]["top_k"] == 20
    assert model.requests[0][1]["max_tokens"] == 1024
    assert "untrusted data" in model.requests[0][0][-1]["content"]


def test_azure_hybrid_retrieval_requests_fifty_candidates_but_selects_eight():
    results = [
        _result(f"doc-{index}", f"evidence {index}", f"c-{index}")
        for index in range(12)
    ]
    agent, _, _, retriever = _agent(results=results, provider="azure_ai_search")

    response = agent.invoke(_request())

    assert retriever.requests[0][1]["top_k"] == 50
    assert len(response.citations) == 8
    assert [item["document_id"] for item in response.citations] == [
        f"doc-{index}" for index in range(8)
    ]


def test_context_deduplicates_and_enforces_document_and_total_budgets():
    limits = RAGLimits(
        context_k=3,
        max_document_chars=5,
        max_context_chars=8,
    )
    results = [
        _result("doc-1", "abcdefghij", "chunk-1"),
        _result("doc-1", "duplicate", "chunk-1"),
        _result("doc-2", "klmnop", "chunk-2"),
        _result("doc-3", "ignored", "chunk-3"),
    ]
    agent, model, _, _ = _agent(results=results, limits=limits)

    response = agent.invoke(_request())
    context = model.requests[0][0][-1]["content"]

    assert [item["document_id"] for item in response.citations] == ["doc-1", "doc-2"]
    assert "abcde" in context
    assert "klm" in context
    assert "duplicate" not in context
    assert "ignored" not in context


def test_security_filters_are_fixed_at_construction_and_passed_to_retrieval():
    filters = {"tenant_id": "tenant-a", "is_public": False}
    agent, _, _, retriever = _agent(filters=filters)

    agent.invoke(_request())

    assert retriever.requests[0][1]["filters"] == filters
    with pytest.raises(ValueError, match="unsafe retrieval filter"):
        _agent(filters={"tenant_id or true": "tenant-a"})
    with pytest.raises(ValueError, match="finite"):
        _agent(filters={"relevance_floor": float("nan")})


def test_query_and_embedding_dimension_failures_happen_before_retrieval():
    limits = RAGLimits(max_query_chars=5)
    agent, _, embedding, retriever = _agent(limits=limits)
    with pytest.raises(ValueError, match="character bound"):
        agent.invoke(_request("123456"))
    assert embedding.queries == []
    assert retriever.requests == []

    agent, _, embedding, retriever = _agent()
    embedding.vector = [0.1]
    with pytest.raises(ValueError, match="dimensions"):
        agent.invoke(_request())
    assert retriever.requests == []


def test_empty_results_produce_no_citations_and_explicit_empty_context():
    agent, model, _, _ = _agent(results=[])

    response = agent.invoke(_request())

    assert response.citations == ()
    assert "[NO_RETRIEVED_EVIDENCE]" in model.requests[0][0][-1]["content"]


def test_limit_evidence_is_stable_and_covers_every_runtime_bound():
    limits = RAGLimits(max_output_tokens=512)

    assert set(limits.as_dict()) == set(vars(limits))
    assert len(limits.digest) == 64
    assert rag_limit_parameters(limits)["limit_max_output_tokens"] == "512"
    assert RAGLimits(max_output_tokens=513).digest != limits.digest


@pytest.mark.parametrize(
    "options, message",
    [
        ({"max_query_chars": 0}, "positive"),
        ({"context_k": 21}, "max_context_k"),
        ({"candidate_k": 7}, "candidate_k"),
        ({"azure_semantic_candidate_k": 7}, "Azure candidate_k"),
    ],
)
def test_invalid_limit_relationships_fail_at_construction(options, message):
    with pytest.raises(ValueError, match=message):
        RAGLimits(**options)


def test_retrieval_filter_count_type_and_string_bounds_are_enforced():
    with pytest.raises(ValueError, match="at most 20"):
        _agent(filters={f"field_{index}": index for index in range(21)})
    with pytest.raises(TypeError, match="unsupported"):
        _agent(filters={"tenant_ids": ["tenant-a"]})
    with pytest.raises(ValueError, match="512"):
        _agent(filters={"tenant_id": "x" * 513})
