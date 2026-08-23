"""Provider interfaces and built-in adapters."""

from aai_core.providers.openai_compatible import (
    OpenAICompatibleChatModel,
    OpenAICompatibleEmbeddingProvider,
)
from aai_core.providers.resolver import ProviderResolver
from aai_core.providers.search import (
    AzureAISearchRetriever,
    DatabricksAISearchRetriever,
)
from aai_core.providers.types import (
    AzureSemanticRankOptions,
    ChatModel,
    DatabricksRerankOptions,
    EmbeddingProvider,
    ModelCapabilities,
    ModelResponse,
    ProviderConfigurationError,
    ProviderError,
    ProviderRequestError,
    RetrievalMode,
    Retriever,
    SearchResult,
    UnsupportedCapabilityError,
)

__all__ = [
    "AzureAISearchRetriever",
    "AzureSemanticRankOptions",
    "ChatModel",
    "DatabricksAISearchRetriever",
    "DatabricksRerankOptions",
    "EmbeddingProvider",
    "ModelCapabilities",
    "ModelResponse",
    "OpenAICompatibleChatModel",
    "OpenAICompatibleEmbeddingProvider",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderRequestError",
    "ProviderResolver",
    "Retriever",
    "RetrievalMode",
    "SearchResult",
    "UnsupportedCapabilityError",
]
