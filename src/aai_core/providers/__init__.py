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
    ChatModel,
    EmbeddingProvider,
    ModelCapabilities,
    ModelResponse,
    ProviderConfigurationError,
    ProviderError,
    Retriever,
    SearchResult,
    UnsupportedCapabilityError,
)

__all__ = [
    "AzureAISearchRetriever",
    "ChatModel",
    "DatabricksAISearchRetriever",
    "EmbeddingProvider",
    "ModelCapabilities",
    "ModelResponse",
    "OpenAICompatibleChatModel",
    "OpenAICompatibleEmbeddingProvider",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderResolver",
    "Retriever",
    "SearchResult",
    "UnsupportedCapabilityError",
]
