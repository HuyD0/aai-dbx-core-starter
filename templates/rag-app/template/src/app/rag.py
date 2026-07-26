from __future__ import annotations

from typing import Any

from aai_core import PlatformContext, bootstrap
from aai_core.agents import AgentRequest, AgentResponse
from aai_core.rag import mlflow_documents
from aai_core.tracing import traced


class RAGAgent:
    def __init__(
        self,
        context: PlatformContext | None = None,
        *,
        prompt_version: int | None = None,
    ) -> None:
        self.context = context or bootstrap()
        self.model = self.context.providers.model("general-chat")
        self.embedding = self.context.providers.embedding("knowledge-embedding")
        self.retriever = self.context.providers.retriever("product-knowledge")
        if prompt_version is not None:
            # Evaluation pins an exact version so results stay reproducible.
            self.prompt = self.context.prompts.load(
                "agent-system", version=prompt_version
            )
        else:
            prompt_alias = (
                "production"
                if self.context.settings.resource.environment in {"prod", "production"}
                else "development"
            )
            self.prompt = self.context.prompts.load(
                "agent-system",
                alias=prompt_alias,
            )

    @traced(name="agent.invoke", span_type="CHAIN")
    def invoke(self, request: AgentRequest) -> AgentResponse:
        query = _latest_user_message(request)
        vector = self.embedding.embed_query(query)
        results = self.retriever.search(
            query,
            query_vector=vector,
            mode="hybrid",
            top_k=8,
        )
        documents = mlflow_documents(results)
        messages = self.prompt.format(
            question=query,
            context=documents,
        )
        response = self.model.generate(
            messages,
            temperature=0.1,
        )
        citations = [
            {
                "document_id": result.document_id,
                "source_uri": result.source_uri,
                "chunk_id": result.chunk_id,
            }
            for result in results
        ]
        return AgentResponse(
            content=response.content,
            citations=citations,
            metadata={
                "model_provider": response.provider,
                "model": response.model,
                "latency_ms": response.latency_ms,
            },
        )


def _latest_user_message(request: AgentRequest) -> str:
    for message in reversed(request.messages):
        if message.get("role") == "user":
            content: Any = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise ValueError("AgentRequest requires a non-empty user message")
