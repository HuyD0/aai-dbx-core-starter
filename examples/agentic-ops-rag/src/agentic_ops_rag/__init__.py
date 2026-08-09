"""Teaching application for the agentic operations and RAG workshop."""

from agentic_ops_rag.chunking import structural_chunks
from agentic_ops_rag.contracts import (
    EvaluationCase,
    MeasurementSource,
    OperationDocument,
    PipelineResult,
    QueryKind,
    RetrievalMode,
)
from agentic_ops_rag.evaluation import benchmark, release_gate
from agentic_ops_rag.offline import OfflineOperationsRetriever, load_documents
from agentic_ops_rag.pipeline import (
    OperationsRAGPipeline,
    authorized_search,
    route_query,
)

__all__ = [
    "EvaluationCase",
    "MeasurementSource",
    "OfflineOperationsRetriever",
    "OperationDocument",
    "OperationsRAGPipeline",
    "PipelineResult",
    "QueryKind",
    "RetrievalMode",
    "benchmark",
    "authorized_search",
    "load_documents",
    "release_gate",
    "route_query",
    "structural_chunks",
]
