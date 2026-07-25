from types import SimpleNamespace

from aai_core.agents import AgentRequest
from aai_core.providers import ModelResponse, SearchResult
from app.agent import RAGAgent


class FakeModel:
    def generate(self, messages, **kwargs):
        assert "grounding" in messages[-1]["content"]
        return ModelResponse(
            content="grounded answer",
            provider="test",
            logical_name="general-chat",
            model="fake",
            latency_ms=1,
        )


class FakeEmbedding:
    def embed_query(self, text):
        return [0.1, 0.2]


class FakeRetriever:
    def search(self, query, **kwargs):
        return [
            SearchResult(
                document_id="doc-1",
                content="grounding",
                score=1.0,
                source_uri="https://example/doc",
                chunk_id="chunk-1",
            )
        ]


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
    def model(self, name):
        return FakeModel()

    def embedding(self, name):
        return FakeEmbedding()

    def retriever(self, name):
        return FakeRetriever()


def test_agent_returns_grounded_answer_and_citation():
    context = SimpleNamespace(
        providers=FakeProviders(),
        prompts=FakePrompts(),
        settings=SimpleNamespace(resource=SimpleNamespace(environment="dev")),
    )
    response = RAGAgent(context).invoke(
        AgentRequest(messages=[{"role": "user", "content": "question"}])
    )

    assert response.content == "grounded answer"
    assert response.citations[0]["chunk_id"] == "chunk-1"
