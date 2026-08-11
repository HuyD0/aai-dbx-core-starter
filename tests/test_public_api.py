"""Public API surface and import hygiene guards."""

import importlib
import inspect
import pkgutil
import subprocess
import sys

import aai_core


def _public_sdk_modules():
    """Yield the root package and importable, non-private SDK modules."""

    yield aai_core
    for module_info in pkgutil.walk_packages(
        aai_core.__path__, prefix=f"{aai_core.__name__}."
    ):
        relative_parts = module_info.name.split(".")[1:]
        if any(part.startswith("_") for part in relative_parts):
            continue
        yield importlib.import_module(module_info.name)


def test_public_api_surface_is_snapshotted():
    """Additions and removals to the top-level API must be deliberate — update
    this snapshot in the same change and follow docs/versioning.md."""

    assert sorted(aai_core.__all__) == [
        "PlatformContext",
        "PlatformSettings",
        "__version__",
        "bootstrap",
    ]


def test_public_exported_classes_and_functions_have_docstrings():
    """Every class/function contract named by ``__all__`` is documented."""

    missing: list[str] = []
    for module in _public_sdk_modules():
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            if (
                inspect.isclass(value) or inspect.isfunction(value)
            ) and not inspect.getdoc(value):
                missing.append(f"{module.__name__}.{name}")

    assert not missing, "public exports missing docstrings: " + ", ".join(missing)


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
