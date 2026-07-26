"""Public API surface and import hygiene guards."""

import subprocess
import sys

import aai_core


def test_public_api_surface_is_snapshotted():
    """Additions and removals to the top-level API must be deliberate — update
    this snapshot in the same change and follow docs/versioning.md."""

    assert sorted(aai_core.__all__) == [
        "AaiCoreError",
        "PlatformContext",
        "PlatformSettings",
        "ResourceContext",
        "__version__",
        "bootstrap",
    ]


def test_importing_aai_core_pulls_no_provider_extras():
    """`import aai_core` must stay light: no provider SDKs, no MLflow."""

    script = (
        "import sys; import aai_core; "
        "heavy = [m for m in ('mlflow', 'openai', 'databricks', 'azure') "
        "if any(loaded == m or loaded.startswith(m + '.') "
        "for loaded in sys.modules)]; "
        "assert not heavy, f'heavy imports at import time: {heavy}'"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_testing_fakes_satisfy_provider_protocols():
    from aai_core.providers import ChatModel, EmbeddingProvider, Retriever
    from aai_core.testing import (
        FakeChatModel,
        FakeEmbeddingProvider,
        FakeRetriever,
        dev_context,
    )

    context = dev_context()
    model = FakeChatModel()
    embedding = FakeEmbeddingProvider()
    retriever = FakeRetriever()
    assert isinstance(model, ChatModel)
    assert isinstance(embedding, EmbeddingProvider)
    assert isinstance(retriever, Retriever)

    context.providers.register_model("general-chat", model)
    context.providers.register_retriever("product-knowledge", retriever)
    response = context.providers.model("general-chat").generate(
        [{"role": "user", "content": "question"}]
    )
    assert response.content == "fake reply"
    results = context.providers.retriever("product-knowledge").search("question")
    assert results[0].as_mlflow_document()["page_content"] == "grounding evidence"
